"""Train one exploratory EMF generator, then generate molecules.

Two stages keep the idea visible: learn a graph representation, freeze it,
then learn continuous transport while the terminal decoder remains trainable.
Minibatches are sampled uniformly with replacement, so one saved generator
state is enough to resume the exact sequence of training examples.
"""

import argparse
import csv
import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import yaml

from emf.chem import decode_graph
from emf.contracts import ModelConfig
from emf.data import collate_graphs, load_count_prior, load_split
from emf.evaluation import sample_model
from emf.models import EMFModel
from emf.objectives import flow_decoder_loss, reconstruction_loss

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class RunConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data_dir: str = "2_data_storage/processed/qm9"
    output_dir: str = "4_experimental_result/result/exploration_001"
    seed: int = 42
    batch_size: int = 128
    warmup_steps: int = 2000
    flow_steps: int = 4000
    warmup_learning_rate: float = 1e-3
    learning_rate: float = 3e-4
    T: float = 0.95
    sigma: float = 0.05
    lambda_mid: float = 0.0
    freeze_decoder: bool = False
    flow_time_budget_seconds: float | None = None
    grad_clip: float = 1.0
    bf16: bool = True
    device: str = "auto"
    cpu_threads: int = 4
    log_every: int = 50
    checkpoint_every: int = 500
    sample_every: int = 1000
    num_samples: int = 1000
    nfe: int = 32
    validation_size: int = 256

    def __post_init__(self):
        if not 0 < self.T < 1 or self.sigma < 0 or self.lambda_mid < 0:
            raise ValueError("Require 0<T<1, sigma>=0, lambda_mid>=0")
        if min(self.batch_size, self.log_every, self.checkpoint_every,
               self.cpu_threads, self.validation_size) < 1:
            raise ValueError("Batch/log/checkpoint/thread/validation sizes must be positive")
        if min(self.warmup_steps, self.flow_steps, self.num_samples, self.sample_every) < 0:
            raise ValueError("Step and sample counts must be nonnegative")
        if self.nfe < 2 or min(self.learning_rate, self.warmup_learning_rate, self.grad_clip) <= 0:
            raise ValueError("Require nfe>=2 and positive learning rates/gradient clip")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        if (self.flow_time_budget_seconds is not None
                and (not math.isfinite(self.flow_time_budget_seconds)
                     or self.flow_time_budget_seconds <= 0)):
            raise ValueError("flow_time_budget_seconds must be positive when supplied")


def project_path(path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_config(path) -> RunConfig:
    values = yaml.safe_load(Path(path).read_text()) or {}
    values["model"] = ModelConfig(**values.get("model", {}))
    return RunConfig(**values)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state(batch_generator):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "batches": batch_generator.get_state()}


def restore_rng(state, batch_generator):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
    batch_generator.set_state(state["batches"].cpu())


def atomic_json(path: Path, values):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def save_checkpoint(path, model, optimizer, config, phase, step, batch_generator, data_hash,
                    count_prior, training_smiles, *, immutable=False,
                    flow_training_seconds=0.0, metrics=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format_version": 1, "config": asdict(config), "model": model.state_dict(),
               "optimizer": optimizer.state_dict(), "phase": phase, "step": step,
               "rng": rng_state(batch_generator), "data_hash": data_hash,
               "count_prior": np.asarray(count_prior).tolist(),
               "training_smiles": tuple(training_smiles),
               "flow_training_seconds": flow_training_seconds, "metrics": metrics or {}}
    if immutable and path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=False)
        context_keys = ("phase", "step", "data_hash", "count_prior", "training_smiles")
        same_context = all(saved[key] == payload[key] for key in context_keys)
        same_model = (saved["config"]["model"] == payload["config"]["model"]
                      and saved["model"].keys() == payload["model"].keys()
                      and all(torch.equal(saved["model"][key], value.detach().cpu())
                              for key, value in payload["model"].items()))
        if not same_context or not same_model:
            raise FileExistsError(f"Sampling checkpoint differs: {path}; choose a new output directory")
        return
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def draw_batch(dataset, size, generator, device):
    indices = torch.randint(len(dataset), (size,), generator=generator).tolist()
    return collate_graphs([dataset[i] for i in indices]).to(device)


@torch.no_grad()
def fit_normalizer(model, dataset, device, batch_size):
    model.encoder.eval()

    def encoded_batches():
        for first in range(0, len(dataset), batch_size):
            batch = collate_graphs([dataset[i] for i in range(
                first, min(first + batch_size, len(dataset)))]).to(device)
            yield model.encoder(batch), batch.masks

    model.normalizer.fit(encoded_batches())


@torch.no_grad()
def reconstruction_metrics(model, dataset, device, size):
    was_training = model.training
    model.eval()
    try:
        batch = collate_graphs([dataset[i] for i in range(min(size, len(dataset)))]).to(device)
        logits = model.decode(model.encode(batch), batch.masks)
        atoms = logits.atoms.argmax(-1).cpu().numpy()
        bonds = logits.bonds.argmax(-1).cpu().numpy()
        exact = valid = connected = 0
        for i, count in enumerate(batch.counts.cpu().tolist()):
            outcome = decode_graph(atoms[i], bonds[i], count, max_nodes=model.config.max_nodes)
            valid += outcome.chemically_valid
            connected += outcome.chemically_valid and outcome.connected
            exact += outcome.smiles == batch.smiles[i]
        n = len(atoms)
        return {"reconstruction": exact / n, "reconstruction_validity": valid / n,
                "reconstruction_connected": connected / n}
    finally:
        model.train(was_training)


def optimizer_for(model, phase, config):
    modules = (model.encoder, model.decoder) if phase == "warmup" else (model.flow, model.decoder)
    params = [p for module in modules for p in module.parameters() if p.requires_grad]
    lr = config.warmup_learning_rate if phase == "warmup" else config.learning_rate
    return torch.optim.AdamW(params, lr=lr, weight_decay=0)


def write_metrics(path, row):
    fields = ["phase", "step", "loss", "flow_loss", "end_loss", "mid_loss", "atom_ce",
              "bond_ce", "seconds", "reconstruction", "reconstruction_validity",
              "reconstruction_connected", "training_seconds"]
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != fields:
                previous = list(reader)
                with path.open("w", newline="") as destination:
                    writer = csv.DictWriter(destination, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(previous)
    with path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fields})


def generate(model, count_prior, dataset, config, output, step, checkpoint_path):
    metadata = {"phase": "exploration", "flow_step": step, "lambda_mid": config.lambda_mid,
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()}
    return sample_model(model, count_prior, num_samples=config.num_samples, nfe=config.nfe,
                        T=config.T, batch_size=config.batch_size, seed=config.seed + 10000,
                        training_smiles=dataset.smiles, output_dir=output / f"samples_{step:07d}",
                        metadata=metadata)


def run_training(config: RunConfig, resume=None):
    torch.set_num_threads(config.cpu_threads)
    seed_everything(config.seed)
    device = torch.device("cuda" if config.device == "auto" and torch.cuda.is_available()
                          else "cpu" if config.device == "auto" else config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested without an assigned/usable GPU")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.cuda.reset_peak_memory_stats()
    data_dir, output = project_path(config.data_dir), project_path(config.output_dir)
    dataset, validation = load_split(data_dir, "train"), load_split(data_dir, "validation")
    if not len(dataset) or not len(validation):
        raise ValueError("Training and validation splits must both be nonempty")
    count_prior = load_count_prior(data_dir)
    dataset_manifest = (data_dir / "manifest.json").read_bytes()
    ids_hash = hashlib.sha256("\n".join(dataset.smiles).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = output / "checkpoints"
    checkpoint = None
    if resume:
        # Only load our local training artifacts, including their optimizer/RNG state.
        checkpoint = torch.load(project_path(resume), map_location="cpu", weights_only=False)
        if checkpoint["config"]["model"] != asdict(config.model) or checkpoint["data_hash"] != ids_hash:
            raise ValueError("Resume architecture or training identities differ from the checkpoint")
        if (checkpoint["phase"] in {"flow", "complete"} and checkpoint["step"] > 0
                and checkpoint["config"].get("freeze_decoder", False) != config.freeze_decoder):
            raise ValueError("Changing decoder policy requires a shared step-zero representation")
        if (config.flow_time_budget_seconds is not None and checkpoint["step"] > 0
                and checkpoint["phase"] in {"flow", "complete"}
                and "flow_training_seconds" not in checkpoint):
            raise ValueError("Timed resume needs a recorded counter or a step-zero representation")
    elif (checkpoints / "latest.pt").exists():
        raise FileExistsError("Run already has a checkpoint: choose a new output directory or --resume")
    (output / "config.yaml").write_text(yaml.safe_dump(asdict(config), sort_keys=False))
    (output / "dataset_manifest.json").write_bytes(dataset_manifest)
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in Path(__file__).parent.glob("*.py")}
    atomic_json(output / "manifest.json", {"config": asdict(config), "training_identity_hash": ids_hash,
                "dataset_manifest_sha256": hashlib.sha256(dataset_manifest).hexdigest(),
                "count_prior": count_prior.tolist(),
                "torch": torch.__version__, "device": str(device), "source_hashes": source_hashes,
                "upstream": json.loads((PROJECT_ROOT / "0_external_repo/provenance.json").read_text())})
    model = EMFModel(config.model).to(device)
    batch_generator = torch.Generator().manual_seed(config.seed + 1)
    phase = checkpoint["phase"] if checkpoint else "warmup"
    if phase == "complete":
        phase = "flow"
    start_step = checkpoint["step"] if checkpoint else 0
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
    if phase == "flow":
        model.freeze_encoder()
        if config.freeze_decoder:
            model.freeze_decoder()
    optimizer = optimizer_for(model, phase, config)
    if checkpoint:
        # A representation checkpoint has no moments yet. Starting each arm's
        # optimizer afresh also supports the F-only frozen-decoder policy.
        representation_start = phase == "flow" and start_step == 0
        if representation_start:
            if checkpoint["optimizer"]["state"]:
                raise ValueError("Step-zero representation unexpectedly contains optimizer moments")
        else:
            optimizer.load_state_dict(checkpoint["optimizer"])
        restore_rng(checkpoint["rng"], batch_generator)
    started = time.perf_counter()
    print(json.dumps({"device": str(device), "parameters": sum(p.numel() for p in model.parameters()),
                      "train_molecules": len(dataset), "phase": phase, "resume_step": start_step}), flush=True)

    def train_phase(name, first_step, total_steps):
        model.train()
        metrics = checkpoint.get("metrics", {}) if checkpoint and name == phase else {}
        training_seconds = (checkpoint.get("flow_training_seconds", 0.0)
                            if checkpoint and name == "flow" and phase == "flow" else 0.0)
        last_step = first_step
        budget = config.flow_time_budget_seconds if name == "flow" else None
        if budget is not None and training_seconds >= budget:
            return metrics, last_step, training_seconds
        for step in range(first_step + 1, total_steps + 1):
            update_started = time.perf_counter()
            batch = draw_batch(dataset, config.batch_size, batch_generator, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=config.bf16 and device.type == "cuda"):
                if name == "warmup":
                    loss, values = reconstruction_loss(model, batch)
                else:
                    loss, values = flow_decoder_loss(model, batch, T=config.T, sigma=config.sigma,
                                                    lambda_mid=config.lambda_mid)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite {name} loss at step {step}")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                [p for group in optimizer.param_groups for p in group["params"]], config.grad_clip)
            if not torch.isfinite(norm):
                raise FloatingPointError(f"Nonfinite {name} gradient at step {step}")
            optimizer.step()
            metrics = {key: float(value) for key, value in values.items()}
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            training_seconds += time.perf_counter() - update_started
            last_step = step
            budget_reached = budget is not None and training_seconds >= budget
            final_update = step == total_steps or budget_reached
            if step == 1 or step % config.log_every == 0 or final_update:
                row = {"phase": name, "step": step, "seconds": time.perf_counter() - started,
                       "training_seconds": training_seconds, **metrics}
                if final_update or step % config.checkpoint_every == 0:
                    row.update(reconstruction_metrics(model, validation, device, config.validation_size))
                write_metrics(output / "training.csv", row)
                print(json.dumps(row), flush=True)
            if step % config.checkpoint_every == 0 or final_update:
                save_checkpoint(checkpoints / "latest.pt", model, optimizer, config, name, step,
                                batch_generator, ids_hash, count_prior, dataset.smiles,
                                flow_training_seconds=training_seconds if name == "flow" else 0.0,
                                metrics=metrics)
                atomic_json(output / "status.json", {"phase": name, "step": step, "metrics": metrics,
                            "flow_training_seconds": training_seconds if name == "flow" else 0.0})
            if (name == "flow" and config.num_samples and config.sample_every
                    and step % config.sample_every == 0 and not final_update):
                snapshot = checkpoints / f"flow_{step:07d}.pt"
                save_checkpoint(snapshot, model, optimizer, config, name, step,
                                batch_generator, ids_hash, count_prior, dataset.smiles, immutable=True,
                                flow_training_seconds=training_seconds, metrics=metrics)
                generate(model, count_prior, dataset, config, output, step, snapshot)
                model.train()
            if budget_reached:
                break
        return metrics, last_step, training_seconds

    if phase == "warmup":
        train_phase("warmup", start_step, config.warmup_steps)
        fit_normalizer(model, dataset, device, config.batch_size)
        model.freeze_encoder()
        if config.freeze_decoder:
            model.freeze_decoder()
        optimizer = optimizer_for(model, "flow", config)
        save_checkpoint(checkpoints / "representation.pt", model, optimizer, config, "flow", 0,
                        batch_generator, ids_hash, count_prior, dataset.smiles)
        start_step = 0
    final_metrics, final_step, flow_training_seconds = train_phase("flow", start_step, config.flow_steps)
    save_checkpoint(checkpoints / "latest.pt", model, optimizer, config, "complete", final_step,
                    batch_generator, ids_hash, count_prior, dataset.smiles,
                    flow_training_seconds=flow_training_seconds, metrics=final_metrics)
    summary = {}
    if config.num_samples:
        snapshot = checkpoints / f"flow_{final_step:07d}.pt"
        save_checkpoint(snapshot, model, optimizer, config, "complete", final_step,
                        batch_generator, ids_hash, count_prior, dataset.smiles, immutable=True,
                        flow_training_seconds=flow_training_seconds, metrics=final_metrics)
        _, summary = generate(model, count_prior, dataset, config, output, final_step, snapshot)
    duration = time.perf_counter() - started
    result = {"phase": "complete", "warmup_steps": config.warmup_steps, "flow_steps": final_step,
              "max_flow_steps": config.flow_steps, "flow_training_seconds": flow_training_seconds,
              "flow_time_budget_seconds": config.flow_time_budget_seconds,
              "stop_reason": "time_budget" if (config.flow_time_budget_seconds is not None
                  and flow_training_seconds >= config.flow_time_budget_seconds) else "step_limit",
              "seconds_this_invocation": duration,
              "runtime_gpu_hours_this_invocation": duration / 3600 if device.type == "cuda" else 0,
              "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else 0,
              "metrics": final_metrics, "generation": summary}
    atomic_json(output / "status.json", result)
    print(json.dumps(result), flush=True)
    return model, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "3_model_training/configs/explore.yaml"))
    parser.add_argument("--resume", help="A trusted local EMF checkpoint to resume")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()
    config = load_config(args.config)
    if args.device:
        config.device = args.device
    run_training(config, args.resume)


if __name__ == "__main__":
    main()
