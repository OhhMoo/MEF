"""Inspect local flow/CE gradients on existing checkpoints and validation data.

No training or checkpoint mutation occurs. Losses and masking are imported
from production. CE is the sum of atom/pair categorical marginal losses.
Negative cosine describes raw local objective disagreement; it does not
establish incompatibility of conditional Bayes risks or a causal mechanism.
Flow-parameter gradients hold the saved decoder fixed and exclude its gradients.
They do not describe AdamW updates or production global gradient clipping, which
also includes decoder gradients when the decoder is trainable.
"""

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import statistics
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from emf.contracts import GraphBatch, GraphLogits, GraphMasks, Latents, ModelConfig
from emf.data import collate_graphs, load_split
from emf.models import EMFModel
from emf.objectives import categorical_loss, interpolate_latents, velocity_matching_loss
from emf.train import PROJECT_ROOT

DEFAULT_STUDY = PROJECT_ROOT / "4_experimental_result/result/overnight_20261010"
DEFAULT_TIMES = (0., .1, .2, .3, .4, .5, .6, .7, .8, .9, .94)
PARAMETER_TIMES = (.1, .3, .5, .9)
LOSS_NAMES = ("atom_ce", "bond_ce", "combined_ce")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tensor_digest(tensor):
    tensor = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(str((str(tensor.dtype), tuple(tensor.shape))).encode())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def state_digest(state, prefix):
    digest = hashlib.sha256()
    for name in sorted(key for key in state if key.startswith(prefix)):
        digest.update(name.encode())
        digest.update(tensor_digest(state[name]).encode())
    return digest.hexdigest()


def _json(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def vector_statistics(flow, ce, coefficient=1., zero_tolerance=1e-12):
    """Rows are independent gradient vectors. Undefined quotients return None.

    dot=<g_flow,g_CE>, cosine=dot/(||g_flow|| ||g_CE||), ratio=||g_CE||/||g_flow||.
    d_flow_along_negative_ce=-dot is the infinitesimal flow-loss change along
    the unnormalized negative CE gradient, before AdamW or gradient clipping.
    Float64 reductions inspect gradients computed by the FP32 production loss.
    """
    flow, ce = flow.detach().double(), ce.detach().double()
    if flow.shape != ce.shape or flow.ndim != 2:
        raise ValueError("gradient vectors must have equal [examples,coordinates] shape")
    if not torch.isfinite(flow).all() or not torch.isfinite(ce).all():
        raise FloatingPointError("nonfinite gradient in diagnostic")
    dot = (flow * ce).sum(1)
    nf, nc = flow.norm(dim=1), ce.norm(dim=1)
    rows = []
    for f, c, product in zip(nf.cpu().tolist(), nc.cpu().tolist(), dot.cpu().tolist()):
        resolved_f, resolved_c = f > zero_tolerance, c > zero_tolerance
        cosine = max(-1., min(1., product / (f * c))) if resolved_f and resolved_c else None
        ratio = c / f if resolved_f else None
        rows.append({"flow_norm": f, "ce_norm": c, "weighted_ce_norm": coefficient * c,
                     "flow_dot_ce": product, "cosine": cosine,
                     "norm_ratio": ratio,
                     "weighted_norm_ratio": coefficient * ratio if ratio is not None else None,
                     "d_flow_along_negative_ce": -product,
                     "d_flow_along_weighted_negative_ce": -coefficient * product,
                     "flow_norm_resolved": resolved_f, "ce_norm_resolved": resolved_c,
                     "cosine_defined": cosine is not None, "norm_ratio_defined": ratio is not None,
                     "flow_exactly_zero": f == 0, "ce_exactly_zero": c == 0})
    return rows


def _flatten(latents, masks):
    return torch.cat([torch.where(masks.nodes[..., None], latents.nodes, 0).flatten(1),
                      torch.where(masks.pairs[..., None], latents.pairs, 0).flatten(1)], dim=1)


def _mean(values, mask):
    count = mask.sum(1)
    mean = (values * mask).sum(1) / count.clamp_min(1)
    return [float(value) if n else None for value, n in zip(mean.detach().cpu(), count.cpu())]


def label_statistics(logits, batch):
    """Per-example label accuracy/confidence; no sampling trajectory is projected."""
    atoms, bonds = logits.atoms.float(), logits.bonds.float()
    pa, pb = atoms.softmax(-1), bonds.softmax(-1)
    atom_correct = atoms.argmax(-1) == batch.atoms
    bond_correct = bonds.argmax(-1) == batch.bonds
    exact = ((atom_correct | ~batch.masks.nodes).all(1)
             & (bond_correct | ~batch.masks.pairs).all(1))
    metrics = {
        "atom_accuracy": _mean(atom_correct.float(), batch.masks.nodes),
        "pair_accuracy": _mean(bond_correct.float(), batch.masks.pairs),
        "bonded_pair_accuracy": _mean(bond_correct.float(), batch.masks.pairs & (batch.bonds > 0)),
        "atom_confidence": _mean(pa.max(-1).values, batch.masks.nodes),
        "pair_confidence": _mean(pb.max(-1).values, batch.masks.pairs),
        "atom_true_label_probability": _mean(pa.gather(-1, torch.where(batch.masks.nodes, batch.atoms, 0)[..., None]).squeeze(-1), batch.masks.nodes),
        "pair_true_label_probability": _mean(pb.gather(-1, torch.where(batch.masks.pairs, batch.bonds, 0)[..., None]).squeeze(-1), batch.masks.pairs),
        "exact_label_reconstruction": exact.float().cpu().tolist(),
    }
    return [{name: values[i] for name, values in metrics.items()} for i in range(len(batch.atoms))]


def diagnose_minibatch(model, batch, clean, noise, time, *, coefficient,
                       bf16=False, parameter_gradients=False, zero_tolerance=1e-12):
    """Use production scalar losses; return per-example output and batch θ gradients.

    Graphs do not interact across minibatch rows. Multiplying d(mean loss)/dy_i
    by the minibatch size gives d(per-molecule loss_i)/dy_i. Parameter gradients
    remain gradients of the production minibatch-mean objective.
    """
    size, device = len(batch.atoms), batch.atoms.device
    t = torch.full((size,), float(time), dtype=torch.float32, device=device)
    enabled = bool(bf16) and device.type == "cuda"
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled):
        noisy = interpolate_latents(clean, noise, t, batch.masks)
        if parameter_gradients:
            predicted = model.flow(noisy, t, batch.masks)
        else:
            with torch.no_grad():
                output = model.flow(noisy, t, batch.masks)
            predicted = Latents(output.nodes.detach().requires_grad_(),
                                output.pairs.detach().requires_grad_())
        flow_loss = velocity_matching_loss(predicted, clean, t, batch.masks)
        logits = model.decode(predicted, batch.masks)
        ce_loss, parts = categorical_loss(logits, batch)
    parameters = tuple(model.flow.parameters()) if parameter_gradients else ()
    targets = (predicted.nodes, predicted.pairs, *parameters)
    gradients = {}
    for name, loss in (("flow", flow_loss), ("atom_ce", parts["atom_ce"]),
                        ("bond_ce", parts["bond_ce"])):
        gradients[name] = tuple(torch.zeros_like(target) if value is None else value
                                for target, value in zip(targets, torch.autograd.grad(
                                    loss, targets, retain_graph=name != "bond_ce", allow_unused=True)))
    gradients["combined_ce"] = tuple(a + b for a, b in zip(gradients["atom_ce"], gradients["bond_ce"]))
    flow_latent = Latents(gradients["flow"][0] * size, gradients["flow"][1] * size)
    per_example, theta_rows = [], []
    for name in LOSS_NAMES:
        ce_latent = Latents(gradients[name][0] * size, gradients[name][1] * size)
        stats = vector_statistics(_flatten(flow_latent, batch.masks), _flatten(ce_latent, batch.masks),
                                  coefficient, zero_tolerance)
        node_stats = vector_statistics(torch.where(batch.masks.nodes[..., None], flow_latent.nodes, 0).flatten(1),
                                       torch.where(batch.masks.nodes[..., None], ce_latent.nodes, 0).flatten(1),
                                       coefficient, zero_tolerance)
        pair_stats = vector_statistics(torch.where(batch.masks.pairs[..., None], flow_latent.pairs, 0).flatten(1),
                                       torch.where(batch.masks.pairs[..., None], ce_latent.pairs, 0).flatten(1),
                                       coefficient, zero_tolerance)
        for i, row in enumerate(stats):
            per_example.append({"local_index": i, "ce_component": name, **row,
                                "node_flow_norm": node_stats[i]["flow_norm"],
                                "node_ce_norm": node_stats[i]["ce_norm"],
                                "node_cosine": node_stats[i]["cosine"],
                                "pair_flow_norm": pair_stats[i]["flow_norm"],
                                "pair_ce_norm": pair_stats[i]["ce_norm"],
                                "pair_cosine": pair_stats[i]["cosine"]})
        if parameter_gradients:
            f = torch.cat([value.flatten() for value in gradients["flow"][2:]])[None]
            c = torch.cat([value.flatten() for value in gradients[name][2:]])[None]
            theta_rows.append({"ce_component": name,
                               **vector_statistics(f, c, coefficient, zero_tolerance)[0]})
    losses = {"flow_loss": float(flow_loss.detach()), "combined_ce": float(ce_loss.detach()),
              "atom_ce": float(parts["atom_ce"].detach()), "bond_ce": float(parts["bond_ce"].detach())}
    if not all(math.isfinite(value) for value in losses.values()):
        raise FloatingPointError("nonfinite production loss")
    detached_logits = GraphLogits(logits.atoms.detach(), logits.bonds.detach())
    return per_example, label_statistics(detached_logits, batch), theta_rows, losses


def _slice(batch, first, last, device):
    return GraphBatch(batch.atoms[first:last], batch.bonds[first:last],
                      GraphMasks(batch.masks.nodes[first:last], batch.masks.pairs[first:last]),
                      batch.smiles[first:last], batch.ids[first:last]).to(device)


def _summary(rows, fields):
    result = {"examples": len(rows)}
    for field in fields:
        values = [float(row[field]) for row in rows if row.get(field) is not None]
        result[field + "_defined_count"] = len(values)
        result[field + "_mean"] = statistics.mean(values) if values else None
        result[field + "_median"] = statistics.median(values) if values else None
    cosines = [row["cosine"] for row in rows if row.get("cosine") is not None]
    result["negative_cosine_fraction_among_defined"] = (
        sum(value < 0 for value in cosines) / len(cosines) if cosines else None)
    return result


class CsvRows:
    def __init__(self, stream):
        self.stream, self.writer = stream, None

    def write(self, row):
        if self.writer is None:
            self.writer = csv.DictWriter(self.stream, fieldnames=list(row))
            self.writer.writeheader()
        self.writer.writerow(row)


def run_diagnostic(study_dir, output_dir, *, decoder_mode="frozen_common", step=50000,
                   seeds=(42, 43, 44), coefficients=(0., .1, 1.), examples=256, batch_size=64,
                   noise_draws=2, times=DEFAULT_TIMES, diagnostic_seed=271828,
                   device="cuda", parameter_gradients=False, parameter_times=PARAMETER_TIMES,
                   zero_tolerance=1e-12, cpu_threads=2, max_seconds=540):
    started_clock, started_at = perf_counter(), utc_now()
    if (decoder_mode not in {"frozen_common", "trainable"} or min(examples, batch_size, noise_draws, step) < 1
            or cpu_threads < 1 or not math.isfinite(max_seconds) or max_seconds <= 0
            or not times or any(not math.isfinite(t) or t < 0 or t >= 1 for t in times)):
        raise ValueError("invalid diagnostic sizes, mode, runtime, or time domain")
    if not math.isfinite(zero_tolerance) or zero_tolerance < 0:
        raise ValueError("zero_tolerance must be finite and nonnegative")
    if (not seeds or not coefficients or any(not math.isfinite(value) or value < 0 for value in coefficients)
            or (parameter_gradients and (not parameter_times or any(
                not math.isfinite(point) or not any(math.isclose(point, t, abs_tol=1e-8) for t in times)
                for point in parameter_times)))):
        raise ValueError("requested coefficients or parameter-gradient times are invalid")
    if device not in {"cpu", "cuda"} or (device == "cuda" and not torch.cuda.is_available()):
        raise ValueError("requested diagnostic device is unavailable")
    torch.set_num_threads(cpu_threads)
    if device == "cuda":
        torch.set_float32_matmul_precision("high")
    study_dir = Path(study_dir).resolve(strict=True)
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("choose a new, empty diagnostic output directory")
    study = json.loads((study_dir / "study.json").read_text())
    tasks = [task for task in json.loads((study_dir / "tasks.json").read_text())
             if task["decoder_mode"] == decoder_mode and task["seed"] in seeds
             and task["lambda_mid"] in coefficients]
    if {(task["seed"], task["lambda_mid"]) for task in tasks} != {
            (seed, coefficient) for seed in seeds for coefficient in coefficients}:
        raise ValueError("study does not contain all requested seed/coefficient checkpoints")
    source_hashes = {}
    for relative, expected in study["source_sha256"].items():
        actual = sha256(PROJECT_ROOT / relative)
        if actual != expected:
            raise ValueError(f"pinned study source changed: {relative}")
        source_hashes[relative] = actual
    source_hashes[str(Path(__file__).relative_to(PROJECT_ROOT))] = sha256(__file__)
    data_dir = Path(study["reference_data_dir"])
    for filename, field in (("manifest.json", "dataset_manifest_sha256"),
                             ("validation.npz", "validation_npz_sha256")):
        if sha256(data_dir / filename) != study[field]:
            raise ValueError(f"pinned validation data changed: {filename}")
    dataset = load_split(data_dir, "validation")
    if examples > len(dataset):
        raise ValueError("requested more distinct validation examples than available")
    selection_generator = torch.Generator(device="cpu").manual_seed(diagnostic_seed)
    selected = torch.randperm(len(dataset), generator=selection_generator)[:examples].tolist()
    batch = collate_graphs([dataset[index] for index in selected])
    template = ModelConfig(**study["initial_representation"]["model"])
    if batch.atoms.shape[1] != template.max_nodes:
        raise ValueError("validation atom storage and checkpoint architecture differ")
    noise_generator = torch.Generator(device="cpu").manual_seed(diagnostic_seed + 1)
    noises = [Latents(torch.randn(examples, template.max_nodes, template.node_dim, generator=noise_generator),
                      torch.randn(examples, batch.bonds.shape[1], template.pair_dim, generator=noise_generator))
              .masked(batch.masks) for _ in range(noise_draws)]
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "paired_inputs.npz", validation_indices=np.array(selected),
                        ids=np.array(batch.ids), smiles=np.array(batch.smiles),
                        counts=batch.counts.numpy(),
                        node_noise=torch.stack([noise.nodes for noise in noises]).numpy(),
                        pair_noise=torch.stack([noise.pairs for noise in noises]).numpy())
    manifest = {
        "format_version": 1, "status": "running", "started_at": started_at,
        "study_dir": str(study_dir), "study_sha256": sha256(study_dir / "study.json"),
        "tasks_sha256": sha256(study_dir / "tasks.json"), "source_sha256": source_hashes,
        "decoder_mode": decoder_mode, "checkpoint_step": step, "seeds": list(seeds),
        "lambda_mid": list(coefficients), "examples": examples, "batch_size": batch_size,
        "noise_draws": noise_draws, "times": list(times), "diagnostic_seed": diagnostic_seed,
        "validation_npz_sha256": study["validation_npz_sha256"], "selected_indices": selected,
        "selected_ids": list(batch.ids), "selected_smiles": list(batch.smiles),
        "counts": batch.counts.tolist(), "device": device, "torch": torch.__version__,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "parameter_gradients": parameter_gradients,
        "parameter_gradient_scope": {"noise_draw": 0, "minibatch": 0, "times": list(parameter_times)},
        "zero_tolerance": zero_tolerance, "max_seconds": max_seconds,
        "noise_sha256": [{"nodes": tensor_digest(noise.nodes), "pairs": tensor_digest(noise.pairs)}
                          for noise in noises], "checkpoints": [],
        "scheduler": {key: value for key, value in os.environ.items() if key in {
            "SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURM_JOB_GPUS", "CUDA_VISIBLE_DEVICES"}},
        "definitions": {
            "latent_gradient": "d per-molecule production loss / d normalized clean prediction; batch-mean gradient multiplied by batch size",
            "parameter_gradient": "d production minibatch-mean loss / d flow parameters only; draw0/batch0 at listed times",
            "cosine": "dot(g_flow,g_ce)/(norm(g_flow)*norm(g_ce)); null when either norm<=zero_tolerance",
            "norm_ratio": "norm(g_ce)/norm(g_flow); null when flow norm<=zero_tolerance",
            "weighted_norm_ratio": "lambda_mid*norm_ratio",
            "directional_derivative": "d L_flow along negative unnormalized CE gradient = -dot(g_flow,g_ce)",
            "ce": "atom categorical mean plus unordered-pair categorical mean; no-bond is active, padding excluded",
            "flow": "velocity error averaged over each molecule's active node and unordered-pair latent coordinates",
            "negative_cosine": "finite-model, unclipped local disagreement; not a Bayes-risk incompatibility or mechanism proof",
            "optimizer_scope": "raw flow-parameter gradients through saved fixed q, not AdamW/clipped updates; production global clipping includes q gradients in trainable mode",
            "exact_label_reconstruction": "all active atom and pair labels equal the validation target; not unconditional generation",
            "precision": "production configured CUDA bf16 autocast, FP32 heads/losses; gradient statistic reductions in float64",
        },
    }
    manifest_path = output_dir / "manifest.json"
    _json(manifest_path, manifest)
    module_fingerprints, shared_input_hash = {}, None
    gradient_fields = ("cosine", "norm_ratio", "weighted_norm_ratio", "flow_norm", "ce_norm",
                       "flow_dot_ce", "d_flow_along_negative_ce")
    decoding_fields = ("atom_accuracy", "pair_accuracy", "bonded_pair_accuracy", "atom_confidence",
                       "pair_confidence", "atom_true_label_probability", "pair_true_label_probability",
                       "exact_label_reconstruction")
    try:
        with ExitStack() as stack:
            writers = {name: CsvRows(stack.enter_context((output_dir / name).open("w", newline="")))
                       for name in ("gradients.csv", "gradient_summary.csv", "decoding.csv",
                                    "decoding_summary.csv", "parameter_gradients.csv")}
            for task in sorted(tasks, key=lambda row: (row["seed"], row["lambda_mid"])):
                path = Path(task["output_dir"]) / "checkpoints" / f"flow_{step:07d}.pt"
                digest = sha256(path)
                payload = torch.load(path, map_location="cpu", weights_only=False)
                saved = payload["config"]
                if (payload["step"] != step or payload["phase"] not in {"flow", "complete"}
                        or saved["seed"] != task["seed"] or saved["lambda_mid"] != task["lambda_mid"]
                        or bool(saved["freeze_decoder"]) != (decoder_mode == "frozen_common")
                        or saved["model"] != study["initial_representation"]["model"]):
                    raise ValueError(f"checkpoint task context differs: {path}")
                if max(times) > saved["T"]:
                    raise ValueError("diagnostic times exceed the checkpoint's training terminal time")
                if shared_input_hash is None:
                    shared_input_hash = payload["data_hash"]
                elif shared_input_hash != payload["data_hash"]:
                    raise ValueError("checkpoint training identities differ")
                fingerprints = {name: state_digest(payload["model"], name + ".")
                                for name in ("encoder", "normalizer", "decoder")}
                expected = module_fingerprints.setdefault(task["seed"], fingerprints)
                checked = ("encoder", "normalizer", "decoder") if decoder_mode == "frozen_common" else ("encoder", "normalizer")
                if any(fingerprints[name] != expected[name] for name in checked):
                    raise ValueError("within-seed frozen components differ across coefficients")
                with torch.random.fork_rng(devices=[]):
                    model = EMFModel(ModelConfig(**saved["model"]))
                model.load_state_dict(payload["model"])
                del payload
                model.to(device).eval()
                model.freeze_encoder()
                model.freeze_decoder()  # Analyze d/dF through the saved q; q itself is never updated.
                if not model.normalizer.fitted.item():
                    raise ValueError("checkpoint normalization is not fitted")
                context = {"seed": task["seed"], "lambda_mid": task["lambda_mid"],
                           "decoder_mode": decoder_mode, "checkpoint_step": step,
                           "checkpoint_sha256": digest}
                checkpoint_record = {**context, "path": str(path), "config": saved,
                                     "module_sha256": fingerprints, "configured_bf16": saved["bf16"],
                                     "autocast_enabled": bool(saved["bf16"]) and device == "cuda"}
                manifest["checkpoints"].append(checkpoint_record)
                _json(manifest_path, manifest)
                cached, clean_rows = [], []
                for first in range(0, examples, batch_size):
                    last = min(first + batch_size, examples)
                    minibatch = _slice(batch, first, last, device)
                    with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.bfloat16,
                                                        enabled=bool(saved["bf16"]) and device == "cuda"):
                        encoded = model.encode(minibatch)
                        clean = Latents(encoded.nodes.float(), encoded.pairs.float())
                        decoded = model.decode(clean, minibatch.masks)
                    cached.append((first, last, minibatch, clean))
                    for i, row in enumerate(label_statistics(decoded, minibatch)):
                        full = {**context, "stage": "clean_encoder", "time": None,
                                "noise_draw": None, "example_index": first + i,
                                "validation_index": selected[first + i], **row}
                        writers["decoding.csv"].write(full)
                        clean_rows.append(full)
                writers["decoding_summary.csv"].write({**context, "stage": "clean_encoder", "time": None,
                                                       **_summary(clean_rows, decoding_fields)})
                for time in times:
                    time_gradients, time_decoding = [], []
                    for draw, noise in enumerate(noises):
                        for minibatch_index, (first, last, minibatch, clean) in enumerate(cached):
                            if perf_counter() - started_clock > max_seconds:
                                raise TimeoutError("diagnostic runtime cap reached; partial measurements retained")
                            paired_noise = Latents(noise.nodes[first:last].to(device), noise.pairs[first:last].to(device))
                            theta = (parameter_gradients and draw == 0 and minibatch_index == 0
                                     and any(math.isclose(time, point, abs_tol=1e-8) for point in parameter_times))
                            rows, decoded_rows, theta_rows, losses = diagnose_minibatch(
                                model, minibatch, clean, paired_noise, time,
                                coefficient=task["lambda_mid"], bf16=saved["bf16"],
                                parameter_gradients=theta, zero_tolerance=zero_tolerance)
                            shared = {**context, "time": time, "noise_draw": draw,
                                      "minibatch": minibatch_index, "minibatch_size": last - first,
                                      **{"minibatch_" + key: value for key, value in losses.items()}}
                            for row in rows:
                                index = first + row.pop("local_index")
                                count = int(batch.counts[index])
                                active_pairs = count * (count - 1) // 2
                                full = {**shared, "example_index": index, "validation_index": selected[index],
                                        "molecule_id": batch.ids[index], "atom_count": count,
                                        "active_pairs": active_pairs,
                                        "active_latent_coordinates": count * template.node_dim + active_pairs * template.pair_dim,
                                        **row}
                                writers["gradients.csv"].write(full)
                                time_gradients.append(full)
                            for i, row in enumerate(decoded_rows):
                                full = {**context, "stage": "flow_prediction", "time": time,
                                        "noise_draw": draw, "example_index": first + i,
                                        "validation_index": selected[first + i], **row}
                                writers["decoding.csv"].write(full)
                                time_decoding.append(full)
                            for row in theta_rows:
                                writers["parameter_gradients.csv"].write({**shared, **row})
                    for component in LOSS_NAMES:
                        writers["gradient_summary.csv"].write({**context, "time": time, "ce_component": component,
                            **_summary([row for row in time_gradients if row["ce_component"] == component], gradient_fields)})
                    writers["decoding_summary.csv"].write({**context, "stage": "flow_prediction", "time": time,
                                                           **_summary(time_decoding, decoding_fields)})
                    print(json.dumps({**context, "time": time, "elapsed_seconds": perf_counter() - started_clock}), flush=True)
                del model, cached, clean, encoded, decoded
                gc.collect()
        manifest["status"] = "complete"
    except Exception as error:
        manifest.update(status="failed", error=repr(error))
        raise
    finally:
        manifest["finished_at"] = utc_now()
        manifest["elapsed_seconds"] = perf_counter() - started_clock
        manifest["output_sha256"] = {path.name: sha256(path) for path in output_dir.iterdir()
                                     if path.is_file() and path != manifest_path}
        _json(manifest_path, manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--decoder-mode", choices=("frozen_common", "trainable"), default="frozen_common")
    parser.add_argument("--step", type=int, default=50000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0., .1, 1.])
    parser.add_argument("--examples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--noise-draws", type=int, default=2)
    parser.add_argument("--times", type=float, nargs="+", default=list(DEFAULT_TIMES))
    parser.add_argument("--diagnostic-seed", type=int, default=271828)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--parameter-gradients", action="store_true")
    parser.add_argument("--parameter-times", type=float, nargs="+", default=list(PARAMETER_TIMES))
    parser.add_argument("--zero-tolerance", type=float, default=1e-12)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--max-seconds", type=float, default=540)
    args = parser.parse_args(argv)
    values = vars(args)
    values["coefficients"] = values.pop("lambdas")
    manifest = run_diagnostic(**values)
    print(json.dumps({"status": manifest["status"], "elapsed_seconds": manifest["elapsed_seconds"],
                      "checkpoint_count": len(manifest["checkpoints"]),
                      "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
