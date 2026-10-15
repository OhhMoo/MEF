"""Generate molecular attempts, preserve failures, and summarize useful yield."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Iterable

import torch

from .chem import decode_graph
from .sampling import model_device, sample_latents


def _synchronize(device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _finite_rows(latents, masks) -> torch.Tensor:
    # Only coordinates belonging to this molecule affect its numerical status.
    nodes = torch.isfinite(latents.nodes).all(dim=-1) | ~masks.nodes
    pairs = torch.isfinite(latents.pairs).all(dim=-1) | ~masks.pairs
    return nodes.all(dim=-1) & pairs.all(dim=-1)


def summarize_attempts(ledger: list[dict], generation_seconds: float) -> dict:
    """All requested attempts are the denominator; yield requires connectivity.

    Uniqueness is distinct connected valid identities / connected valid attempts.
    Novelty is novel distinct identities / distinct connected valid identities.
    These denominator choices are recorded in the key names/counts.
    """
    if not math.isfinite(generation_seconds) or generation_seconds < 0:
        raise ValueError("generation_seconds must be finite and nonnegative")
    total = len(ledger)
    chemical = sum(bool(row["chemically_valid"]) for row in ledger)
    valid = [row for row in ledger if row["chemically_valid"] and row["connected"]]
    distinct = {row["smiles"] for row in valid}
    novel = {row["smiles"] for row in valid if row["novel"]}
    failures = {}
    for row in ledger:
        if row["failure_reason"]:
            reason = row["failure_reason"]
            failures[reason] = failures.get(reason, 0) + 1

    return {
        "attempts": total,
        "chemically_valid": chemical,
        "connected_valid": len(valid),
        "distinct_connected_valid": len(distinct),
        "novel_distinct_connected_valid": len(novel),
        "chemical_validity_rate": chemical / total if total else 0.0,
        "connected_validity_rate": len(valid) / total if total else 0.0,
        "uniqueness_among_connected_valid": len(distinct) / len(valid) if valid else 0.0,
        "novelty_among_distinct_connected_valid": len(novel) / len(distinct) if distinct else 0.0,
        "generation_seconds": generation_seconds,
        "attempts_per_second": total / generation_seconds if generation_seconds > 0 else None,
        "distinct_connected_valid_per_second": len(distinct) / generation_seconds if generation_seconds > 0 else None,
        "failure_counts": failures,
    }


def _csv_value(value):
    return json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value


def write_generation_outputs(output_dir, ledger: list[dict], summary: dict) -> None:
    """Store the raw attempt ledger and summary without dropping unsuccessful rows."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "attempts.jsonl").open("w", encoding="utf-8") as stream:
        for row in ledger:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    with (destination / "attempts.csv").open("w", encoding="utf-8", newline="") as stream:
        if ledger:
            writer = csv.DictWriter(stream, fieldnames=list(ledger[0]))
            writer.writeheader()
            writer.writerows({key: _csv_value(value) for key, value in row.items()}
                             for row in ledger)
    with (destination / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    with (destination / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow({key: _csv_value(value) for key, value in summary.items()})


@torch.no_grad()
def sample_model(model, count_prior, num_samples: int = 1000, nfe: int = 32,
                 T: float = 0.95, batch_size: int = 128, seed: int = 42,
                 training_smiles: Iterable[str] = (), output_dir=None,
                 metadata: dict | None = None) -> tuple[list[dict], dict]:
    """Run count/noise → transport → endpoint → terminal decode → chemistry.

    Timing includes these stages and ledger construction, with CUDA synchronized
    at batch boundaries. Per-attempt seconds amortize its batch's wall time and
    should not be interpreted as independently measured molecule latency.
    """
    if isinstance(num_samples, bool) or not isinstance(num_samples, int) or num_samples < 1:
        raise ValueError("num_samples must be a positive integer")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    prior = torch.as_tensor(count_prior, dtype=torch.float64, device="cpu")
    if (prior.shape != (model.config.max_nodes,) or not torch.isfinite(prior).all()
            or (prior < 0).any() or prior.sum() <= 0):
        raise ValueError("count_prior must have finite nonnegative weights for counts 1..N")
    prior = prior / prior.sum()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    training = set(training_smiles)
    seen = set()
    ledger = []
    generation_seconds = 0.0
    device = model_device(model)
    module_modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        for offset in range(0, num_samples, batch_size):
            size = min(batch_size, num_samples - offset)
            _synchronize(device)
            started = perf_counter()
            counts = torch.multinomial(prior, size, replacement=True, generator=generator) + 1
            sampled = sample_latents(model, counts, nfe=nfe, T=T, generator=generator)
            logits = model.decode(sampled.latents, sampled.masks)
            finite = _finite_rows(sampled.latents, sampled.masks)
            finite_logits = ((torch.isfinite(logits.atoms).all(dim=-1) | ~sampled.masks.nodes).all(dim=-1)
                             & (torch.isfinite(logits.bonds).all(dim=-1) | ~sampled.masks.pairs).all(dim=-1))
            atoms = logits.atoms.argmax(dim=-1).cpu()
            bonds = logits.bonds.argmax(dim=-1).cpu()
            finite, finite_logits = finite.cpu(), finite_logits.cpu()
            rows = []
            for index, count in enumerate(counts.tolist()):
                if not finite[index] or not finite_logits[index]:
                    smiles, chemical, connected = None, False, False
                    reason = "nonfinite_latents" if not finite[index] else "nonfinite_logits"
                else:
                    outcome = decode_graph(atoms[index], bonds[index], count,
                                           max_nodes=model.config.max_nodes)
                    smiles, chemical, connected = outcome.smiles, outcome.chemically_valid, outcome.connected
                    reason = outcome.failure_reason
                valid = chemical and connected
                duplicate = smiles in seen if valid else None
                novel = smiles not in training if valid else None
                if valid:
                    seen.add(smiles)
                rows.append({
                    "attempt_id": offset + index,
                    "seed": seed,
                    "atom_count": count,
                    "nfe": nfe,
                    "T": T,
                    "atoms": atoms[index].tolist(),
                    "bonds": bonds[index].tolist(),
                    "smiles": smiles,
                    "chemically_valid": chemical,
                    "connected": connected,
                    "failure_reason": reason,
                    "duplicate": duplicate,
                    "novel": novel,
                })
            _synchronize(device)
            elapsed = perf_counter() - started
            generation_seconds += elapsed
            for row in rows:
                row["batch_amortized_seconds"] = elapsed / size
            ledger.extend(rows)
    finally:
        for module, training_mode in module_modes:
            module.training = training_mode

    summary = summarize_attempts(ledger, generation_seconds)
    summary.update({"seed": seed, "nfe": nfe, "T": T,
                    "batch_size": batch_size, "device": str(device),
                    "float32_matmul_precision": torch.get_float32_matmul_precision(),
                    "timing_scope": "count/noise,transport,endpoint,decode,host chemistry,ledger; excludes file writing",
                    "metadata": metadata or {}})
    if output_dir is not None:
        write_generation_outputs(output_dir, ledger, summary)
    return ledger, summary


def generate_from_checkpoint(checkpoint, output_dir, *, num_samples=None,
                             nfe=None, seed=None, device=None, batch_size=None,
                             T=None) -> tuple[list[dict], dict]:
    """Generate from one trusted local checkpoint and its saved training context.

    Count prior, normalization, architecture, and novelty identities all come
    from that checkpoint. Mutable processed dataset files are never consulted.
    """
    from .contracts import ModelConfig
    from .models import EMFModel

    path = Path(checkpoint).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError("checkpoint must be a local file")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {"config", "model", "phase", "count_prior", "training_smiles"}
    if not isinstance(payload, dict) or required - payload.keys():
        raise ValueError("checkpoint needs config, model, phase, count_prior, and training_smiles")
    if payload["phase"] not in {"flow", "complete"}:
        raise ValueError("generation requires a flow or complete checkpoint")
    saved = payload["config"]
    model = EMFModel(ModelConfig(**saved["model"]))
    model.load_state_dict(payload["model"])
    if not model.normalizer.fitted.item():
        raise ValueError("generation requires fitted latent normalization")
    identities = payload["training_smiles"]
    if (not isinstance(identities, (list, tuple)) or not identities
            or not all(isinstance(smiles, str) and smiles for smiles in identities)):
        raise ValueError("checkpoint must contain canonical training molecule identities")

    selected_device = saved.get("device", "auto") if device is None else device
    if selected_device == "auto":
        selected_device = "cuda" if torch.cuda.is_available() else "cpu"
    if selected_device not in {"cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if selected_device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if selected_device == "cpu":
        torch.set_num_threads(saved.get("cpu_threads", 4))
    else:
        # Match the training runner's FP32 sampling policy on allocated GPUs.
        torch.set_float32_matmul_precision("high")
    model.to(selected_device)
    model.freeze_encoder()
    model.eval()
    metadata = {
        "checkpoint_path": str(path),
        "checkpoint_sha256": digest,
        "checkpoint_phase": payload["phase"],
        "checkpoint_step": payload.get("step"),
        "data_hash": payload.get("data_hash"),
        "count_prior_source": "checkpoint.count_prior",
        "novelty_source": "checkpoint.training_smiles",
        "training_identity_count": len(set(identities)),
    }
    return sample_model(
        model, payload["count_prior"],
        num_samples=saved["num_samples"] if num_samples is None else num_samples,
        nfe=saved["nfe"] if nfe is None else nfe,
        T=saved["T"] if T is None else T,
        batch_size=saved["batch_size"] if batch_size is None else batch_size,
        seed=saved["seed"] if seed is None else seed,
        training_smiles=identities, output_dir=output_dir, metadata=metadata,
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Generate molecules from a trusted local EMF checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Trusted local .pt checkpoint")
    parser.add_argument("--output-dir", required=True, help="Directory for attempt and summary files")
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--nfe", type=int, help="Total flow calls, including the endpoint prediction")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--T", type=float, help="Integration terminal time, strictly below one")
    args = parser.parse_args(argv)
    _, summary = generate_from_checkpoint(
        args.checkpoint, args.output_dir, num_samples=args.num_samples,
        nfe=args.nfe, seed=args.seed, device=args.device,
        batch_size=args.batch_size, T=args.T,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
