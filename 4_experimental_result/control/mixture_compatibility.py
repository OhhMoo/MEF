"""Exact finite-prior compatibility probe for eight fixed, ordered graphs.

For uniform k, X=X_k and Z_t=tX+(1-t)epsilon, posterior weights are
a_k(z,t) proportional to exp(-||z-tX_k||² / [2(1-t)²]). The flow conditional
optimum is mu=sum(a_k X_k). A nonzero d/dmu CE(pi,q(mu)) is a stationarity
residual for this selected finite prior and saved q; it is not a QM9 oracle.
CPU FP32 decoder arithmetic intentionally differs from production CUDA bf16.
"""

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from emf.contracts import GraphLogits, GraphMasks, Latents, ModelConfig
from emf.data import collate_graphs, load_split
from emf.models import EMFModel
from emf.train import PROJECT_ROOT
from supervision_diagnostic import label_statistics, sha256, state_digest, tensor_digest, utc_now

TIMES = (0., .01, .03, .05, .1, .2, .5, .9)
DEFAULT_STUDY = PROJECT_ROOT / "4_experimental_result/result/overnight_20261010"
DEFAULT_OUTPUT = PROJECT_ROOT / "4_experimental_result/result/supervision_diagnostic_20261010/finite_mixture"


def posterior_weights(observed, endpoints, time):
    """All columns are active; all mixture components have the same atom count."""
    if not 0 <= time < 1:
        raise ValueError("finite-mixture posterior requires 0 <= t < 1")
    distance = (observed.double()[:, None] - time * endpoints.double()[None]).square().sum(-1)
    return (-distance / (2 * (1 - time)**2)).softmax(-1)


def soft_categorical_loss(logits, probabilities, masks):
    """Per-molecule soft CE, with production atom/pair means then their sum."""
    parts = {}
    for name, values, targets, mask in (
            ("atom_ce", logits.atoms, probabilities.atoms, masks.nodes),
            ("bond_ce", logits.bonds, probabilities.bonds, masks.pairs)):
        losses = -(targets.detach().float() * values.float().log_softmax(-1)).sum(-1)
        parts[name] = torch.where(mask, losses, 0).sum(1) / mask.sum(1).clamp_min(1)
    return parts["atom_ce"] + parts["bond_ce"], parts


def _entropy(probabilities):
    return -torch.special.xlogy(probabilities, probabilities).sum(-1)


def run(study_dir=DEFAULT_STUDY, output_dir=DEFAULT_OUTPUT, *, components=8, draws=32,
        seeds=(42, 43, 44), diagnostic_seed=271829, cpu_threads=4, times=TIMES):
    clock, started_at = perf_counter(), utc_now()
    torch.set_num_threads(cpu_threads)
    study_dir, output_dir = Path(study_dir).resolve(), Path(output_dir).resolve()
    if components < 1 or draws < 1 or cpu_threads < 1 or not times or any(not 0 <= t < 1 for t in times):
        raise ValueError("invalid mixture sizes, threads, or times")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("choose a fresh finite-mixture output directory")
    study = json.loads((study_dir / "study.json").read_text())
    source_hashes = {}
    for relative, expected in study["source_sha256"].items():
        source_hashes[relative] = sha256(PROJECT_ROOT / relative)
        if source_hashes[relative] != expected:
            raise ValueError(f"pinned study source changed: {relative}")
    for path in (Path(__file__), Path(__file__).with_name("supervision_diagnostic.py")):
        source_hashes[str(path.relative_to(PROJECT_ROOT))] = sha256(path)
    data_dir = Path(study["reference_data_dir"])
    if sha256(data_dir / "validation.npz") != study["validation_npz_sha256"]:
        raise ValueError("validation data differs from the pinned study")
    dataset = load_split(data_dir, "validation")
    generator = torch.Generator().manual_seed(diagnostic_seed)
    selected, seen = [], set()
    for index in torch.randperm(len(dataset), generator=generator).tolist():
        if dataset.counts[index] == 9 and dataset.smiles[index] not in seen:
            seen.add(dataset.smiles[index])
            selected.append(index)
            if len(selected) == components:
                break
    if len(selected) != components:
        raise ValueError("not enough distinct nine-atom validation molecules")
    batch = collate_graphs([dataset[index] for index in selected])
    graph_labels = torch.cat([batch.atoms, batch.bonds], 1)
    if torch.unique(graph_labels, dim=0).shape[0] != components:
        raise ValueError("selected labeled graph components are not distinct")
    config = ModelConfig(**study["initial_representation"]["model"])
    dimension = 9 * config.node_dim + 36 * config.pair_dim
    endpoints = torch.randint(components, (draws,), generator=generator)
    noise = torch.randn(draws, dimension, generator=generator, dtype=torch.float64)
    masks = GraphMasks.from_counts(torch.full((draws,), 9), 9)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "paired_inputs.npz", validation_indices=np.array(selected),
                        ids=np.array(batch.ids), smiles=np.array(batch.smiles),
                        atoms=batch.atoms.numpy(), bonds=batch.bonds.numpy(),
                        endpoints=endpoints.numpy(), noise=noise.numpy())
    metadata = {"started_at": started_at, "scope": __doc__, "status": "running",
                "study_sha256": sha256(study_dir / "study.json"), "source_sha256": source_hashes,
                "validation_npz_sha256": study["validation_npz_sha256"], "selected_indices": selected,
                "selected_ids": list(batch.ids), "selected_smiles": list(batch.smiles),
                "components": components, "draws": draws, "uniform_prior": True,
                "stored_graph_ordering_preserved": True, "times": list(times),
                "diagnostic_seed": diagnostic_seed, "endpoint_draws": endpoints.tolist(),
                "selection_sha256": tensor_digest(torch.tensor(selected)),
                "endpoint_draws_sha256": tensor_digest(endpoints),
                "noise_sha256": tensor_digest(noise), "torch": torch.__version__,
                "precision": "posterior/entropy/mean in float64; q/CE/latent gradients CPU FP32, no autocast",
                "production_precision_difference": "production uses CUDA bf16 trunk autocast with FP32 heads/losses",
                "definitions": {"conditional_graph_entropy": "-sum a_k log a_k; components are distinct labeled graphs",
                    "optimal_marginal_ce": "mean atom marginal entropy plus mean unordered-pair marginal entropy",
                    "ce_excess": "production-normalized soft categorical CE minus optimal_marginal_ce",
                    "gradient_norm": "L2 norm of per-draw CE gradient at exact flow conditional mean, across all active latent coordinates",
                    "flow_conditional_gradient_at_mu": "exactly zero by the finite-mixture squared-error conditional-risk identity"},
                "checkpoints": []}
    rows, clean_rows = [], []
    for seed in seeds:
        path = Path(study["shared_representations"][str(seed)])
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if (payload["step"] != 0 or payload["phase"] != "flow" or payload["config"]["seed"] != seed
                or payload["config"]["model"] != study["initial_representation"]["model"]
                or payload["data_hash"] != study["initial_representation"]["data_hash"]):
            raise ValueError("expected the seed's warm representation checkpoint")
        with torch.random.fork_rng(devices=[]):
            model = EMFModel(config).eval()
        model.load_state_dict(payload["model"])
        model.freeze_encoder()
        model.freeze_decoder()
        if not model.normalizer.fitted.item():
            raise ValueError("warm representation normalization is not fitted")
        record = {"seed": seed, "path": str(path), "checkpoint_sha256": sha256(path),
                  "checkpoint_step": 0, "configured_production_bf16": payload["config"]["bf16"],
                  "module_sha256": {key: state_digest(payload["model"], key + ".")
                                    for key in ("encoder", "normalizer", "decoder")}}
        metadata["checkpoints"].append(record)
        with torch.no_grad():
            clean = model.encode(batch)
            clean_logits = model.decode(clean, batch.masks)
            clean_ce, clean_parts = soft_categorical_loss(clean_logits, GraphLogits(
                F.one_hot(batch.atoms, 4), F.one_hot(batch.bonds, 4)), batch.masks)
        for i, metrics in enumerate(label_statistics(clean_logits, batch)):
            clean_rows.append({"seed": seed, "component": i, "validation_index": selected[i],
                               "clean_ce": float(clean_ce[i]),
                               "clean_atom_ce": float(clean_parts["atom_ce"][i]),
                               "clean_bond_ce": float(clean_parts["bond_ce"][i]), **metrics})
        support = torch.cat([clean.nodes.flatten(1), clean.pairs.flatten(1)], 1).double()
        record["encoded_endpoints_sha256"] = tensor_digest(support)
        atom_labels, pair_labels = F.one_hot(batch.atoms, 4).double(), F.one_hot(batch.bonds, 4).double()
        for time in times:
            z = time * support[endpoints] + (1 - time) * noise
            weights = posterior_weights(z, support, time)
            mu = weights @ support
            atom_pi = torch.einsum("bk,knc->bnc", weights, atom_labels)
            pair_pi = torch.einsum("bk,kpc->bpc", weights, pair_labels)
            predicted = Latents(mu[:, :9 * config.node_dim].reshape(draws, 9, config.node_dim).float().requires_grad_(),
                                mu[:, 9 * config.node_dim:].reshape(draws, 36, config.pair_dim).float().requires_grad_())
            logits = model.decode(predicted, masks)
            loss, parts = soft_categorical_loss(logits, GraphLogits(atom_pi, pair_pi), masks)
            gradients = torch.autograd.grad(loss.sum(), (predicted.nodes, predicted.pairs))
            residual = torch.cat([gradient.flatten(1) for gradient in gradients], 1).double().norm(dim=1)
            entropy = _entropy(weights)
            optimal = _entropy(atom_pi).mean(1) + _entropy(pair_pi).mean(1)
            for draw in range(draws):
                rows.append({"seed": seed, "time": time, "draw": draw, "endpoint_component": int(endpoints[draw]),
                             "posterior_entropy": float(entropy[draw]), "conditional_graph_entropy": float(entropy[draw]),
                             "posterior_effective_support": float(entropy[draw].exp()),
                             "posterior_inverse_simpson_support": float(1 / weights[draw].square().sum()),
                             "posterior_max_weight": float(weights[draw].max()),
                             "posterior_components_above_0p01": int((weights[draw] > .01).sum()),
                             "optimal_marginal_ce": float(optimal[draw]), "soft_ce": float(loss[draw].detach()),
                             "atom_ce": float(parts["atom_ce"][draw].detach()), "bond_ce": float(parts["bond_ce"][draw].detach()),
                             "ce_excess": float(loss[draw].detach().double() - optimal[draw]),
                             "ce_gradient_norm_at_mu": float(residual[draw]),
                             **{f"posterior_weight_{k}": float(weights[draw, k]) for k in range(components)}})
        del model, payload
    for name, values in (("mixture_rows.csv", rows), ("clean_reconstruction.csv", clean_rows)):
        with (output_dir / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    metadata.update(status="complete", finished_at=utc_now(), elapsed_seconds=perf_counter() - clock,
                    output_sha256={path.name: sha256(path) for path in output_dir.iterdir() if path.is_file()})
    (output_dir / "manifest.json").write_text(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--components", type=int, default=8)
    parser.add_argument("--draws", type=int, default=32)
    parser.add_argument("--cpu-threads", type=int, default=4)
    result = run(**vars(parser.parse_args()))
    print(json.dumps({"status": result["status"], "elapsed_seconds": result["elapsed_seconds"]}))
