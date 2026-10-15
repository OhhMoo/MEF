"""Fit a source-only soft-graph critic for artificial RDKit descriptor control.

Labels are MolLogP and TPSA divided by source-molecule standard deviations.
Generated logits are labeled by their decoded, valid connected graph. Invalid
attempts remain in the audit counts and never receive invented labels.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import Tensor

from .chem import decode_graph
from .contracts import GraphLogits, GraphMasks, Latents
from .data import collate_graphs
from .masked_edit_control import (SoftGraphDescriptorCritic,
                                  descriptor_features, fit_descriptor_critic,
                                  hard_graph_probabilities)
from .masked_edit_data import MaskedEditDataset, _descriptor
from .masked_edit_model import decode_masked_edit, hybrid_latents
from .masked_edit_run import (condition_for, graphs_from_logits, load_model,
                              now, pair_table)
from .masked_edit_sampling import make_paired_gaussian, sample_masked_edit_latents
from .redesign_train import atomic_json, digest


@dataclass(frozen=True)
class _Record:
    graph: GraphLogits       # detached CPU category probabilities
    masks: GraphMasks       # detached CPU valid/padding masks
    label: Tensor           # detached CPU scaled RDKit descriptors
    smiles: str
    domain: str             # hard, hybrid, unguided, control
    stream: str             # hard, train, audit
    mask_id: str | None = None


@dataclass(frozen=True)
class _PerturbationPair:
    baseline: GraphLogits   # detached CPU logits
    controlled: GraphLogits
    masks: GraphMasks
    baseline_label: Tensor
    controlled_label: Tensor
    baseline_smiles: str
    controlled_smiles: str


def _select(logits: GraphLogits, masks: GraphMasks, index: int) -> tuple[GraphLogits, GraphMasks]:
    return (GraphLogits(logits.atoms[index:index + 1].detach().cpu(),
                        logits.bonds[index:index + 1].detach().cpu()),
            GraphMasks(masks.nodes[index:index + 1].detach().cpu(),
                       masks.pairs[index:index + 1].detach().cpu()))


def _append_decoded(records: list[_Record], logits: GraphLogits, masks: GraphMasks,
                    scales: np.ndarray, domain: str, stream: str,
                    counts: Counter, mask_ids: str | list[str] | None = None,
                    forbidden_smiles: set[str] | None = None
                    ) -> list[tuple[str | None, Tensor | None]]:
    graphs = graphs_from_logits(logits)
    outcomes = []
    for index, graph in enumerate(graphs):
        counts[f"{domain}_{stream}_attempts"] += 1
        outcome = decode_graph(graph["atoms"], graph["bonds"], graph["count"])
        if not outcome.chemically_valid:
            counts[f"{domain}_{stream}_invalid"] += 1
            counts[f"failure_{outcome.failure_reason}"] += 1
            outcomes.append((None, None))
            continue
        if not outcome.connected:
            counts[f"{domain}_{stream}_disconnected"] += 1
            outcomes.append((None, None))
            continue
        if forbidden_smiles is not None and outcome.smiles in forbidden_smiles:
            counts[f"{domain}_{stream}_excluded_heldout_identity"] += 1
            outcomes.append((None, None))
            continue
        label = torch.tensor(_descriptor(outcome.smiles) / scales, dtype=torch.float32)[None]
        selected, selected_masks = _select(logits, masks, index)
        probabilities = GraphLogits(selected.atoms.softmax(-1), selected.bonds.softmax(-1))
        mask_id = mask_ids[index] if isinstance(mask_ids, list) else mask_ids
        records.append(_Record(probabilities, selected_masks, label, outcome.smiles,
                               domain, stream, mask_id))
        counts[f"{domain}_{stream}_valid_connected"] += 1
        outcomes.append((outcome.smiles, label))
    return outcomes


def _hard_records(dataset: MaskedEditDataset, scales: np.ndarray,
                  records: list[_Record], counts: Counter) -> None:
    source = [dataset.graph(int(index)) for index in dataset.source_graph_indices]
    aligned = [dataset.graph(int(index), aligned=True, task_id=task["task_id"])
               for task in dataset.tasks for index in task["compatible_indices"]["train"]]
    for domain, graphs in (("hard_source", source), ("hard_aligned", aligned)):
        for offset in range(0, len(graphs), 128):
            batch = collate_graphs(graphs[offset:offset + 128])
            onehot = hard_graph_probabilities(batch)
            for index, graph in enumerate(graphs[offset:offset + 128]):
                selected, masks = _select(onehot, batch.masks, index)
                label = torch.tensor(_descriptor(graph["smiles"]) / scales,
                                     dtype=torch.float32)[None]
                records.append(_Record(selected, masks, label, graph["smiles"],
                                       domain, "hard", graph.get("mask_id")))
                counts[f"{domain}_hard_attempts"] += 1
                counts[f"{domain}_hard_valid_connected"] += 1


def _hybrid_records(model, dataset, scales, records, counts, device, seed,
                    chunk_size, forbidden_smiles):
    rows, leads, queries, edits = pair_table(dataset, device)
    rng = torch.Generator(device="cpu").manual_seed(seed + 17)
    for offset in range(0, len(rows), chunk_size):
        stop = min(offset + chunk_size, len(rows))
        index = torch.arange(offset, stop, device=device)
        lead = type(leads)(leads.atoms[index], leads.bonds[index],
                           GraphMasks(leads.masks.nodes[index], leads.masks.pairs[index]))
        query = type(queries)(queries.atoms[index], queries.bonds[index],
                             GraphMasks(queries.masks.nodes[index], queries.masks.pairs[index]))
        edit = GraphMasks(edits.nodes[index], edits.pairs[index])
        with torch.no_grad():
            encoded_lead, encoded_query = model.encode(lead), model.encode(query)
            hybrid = hybrid_latents(encoded_lead, encoded_query, edit)
            state = torch.zeros(len(index), model.redesign_config.state_dim, device=device)
            condition = model.condition(lead, torch.zeros(len(index), 2, device=device),
                                        torch.as_tensor(dataset.context, device=device).expand(len(index), -1),
                                        state, torch.tensor([.25, -.1], device=device).expand(len(index), -1),
                                        edit)
            for sigma in (0., .05, .20):
                if sigma:
                    node_noise = torch.randn(hybrid.nodes.shape, generator=rng).to(device)
                    pair_noise = torch.randn(hybrid.pairs.shape, generator=rng).to(device)
                    target = Latents(hybrid.nodes + sigma * node_noise * edit.nodes[..., None],
                                     hybrid.pairs + sigma * pair_noise * edit.pairs[..., None])
                else:
                    target = hybrid
                logits = decode_masked_edit(model, target, condition)
                _append_decoded(records, logits, query.masks, scales,
                                "hybrid", "train", counts,
                                [row["task_id"] for row in rows[offset:stop]],
                                forbidden_smiles)


def _random_control(condition, radius: float, seed: int, T: float):
    rng = torch.Generator(device="cpu").manual_seed(seed)
    device = condition.lead.nodes.device
    node = torch.randn((condition.lead.nodes.shape[0], 2,
                        *condition.lead.nodes.shape[1:]), generator=rng).to(device)
    pair = torch.randn((condition.lead.pairs.shape[0], 2,
                        *condition.lead.pairs.shape[1:]), generator=rng).to(device)
    node = node * condition.edit_masks.nodes[:, None, :, None]
    pair = pair * condition.edit_masks.pairs[:, None, :, None]
    coordinates = (2 * (condition.edit_masks.nodes.sum(1) * node.shape[-1]
                        + condition.edit_masks.pairs.sum(1) * pair.shape[-1])).clamp_min(1)
    rms = ((node.square().flatten(1).sum(1) + pair.square().flatten(1).sum(1))
           / coordinates).sqrt().clamp_min(1e-8)
    node = node * (radius / rms)[:, None, None, None]
    pair = pair * (radius / rms)[:, None, None, None]

    def controller(state, time, supplied):
        first = (time < T / 2)[:, None, None]
        return Latents(torch.where(first, node[:, 0], node[:, 1]),
                       torch.where(first, pair[:, 0], pair[:, 1]))

    return controller


def _generated_records(model, dataset, scales, records, pairs, counts,
                       device, seed, source_attempts, control_attempts,
                       nfe, chunk_size, forbidden_smiles):
    for mask_number, mask_id in enumerate(dataset.masks):
        for stream, total in (("train", source_attempts - source_attempts // 4),
                              ("audit", source_attempts // 4)):
            for offset in range(0, total, chunk_size):
                size = min(chunk_size, total - offset)
                condition = condition_for(model, dataset, mask_id, size)
                noise_seed = seed + (100000 if stream == "audit" else 10000) + mask_number * 1000 + offset
                noise = make_paired_gaussian(model.config, size, noise_seed, device=device)
                with torch.no_grad():
                    sampled = sample_masked_edit_latents(model, condition, noise, nfe=nfe)
                    logits = decode_masked_edit(model, sampled.latents, condition)
                _append_decoded(records, logits, sampled.masks, scales,
                                "unguided", stream, counts, mask_id,
                                forbidden_smiles)
        for stream, total in (("train", control_attempts - control_attempts // 4),
                              ("audit", control_attempts // 4)):
            for radius_number, radius in enumerate((.1, .5, 1.)):
                size_for_radius = total // 3 + (radius_number < total % 3)
                for offset in range(0, size_for_radius, chunk_size):
                    size = min(chunk_size, size_for_radius - offset)
                    condition = condition_for(model, dataset, mask_id, size)
                    noise_seed = seed + (300000 if stream == "audit" else 20000) + mask_number * 10000 + radius_number * 1000 + offset
                    noise = make_paired_gaussian(model.config, size, noise_seed, device=device)
                    controller = _random_control(condition, radius, noise_seed + 73, .95)
                    with torch.no_grad():
                        base = sample_masked_edit_latents(model, condition, noise, nfe=nfe)
                        guided = sample_masked_edit_latents(model, condition, noise, nfe=nfe,
                                                             velocity_controller=controller)
                        base_logits = decode_masked_edit(model, base.latents, condition)
                        guided_logits = decode_masked_edit(model, guided.latents, condition)
                    base_outcomes = _append_decoded(records, base_logits, base.masks,
                                                    scales, "control_baseline", stream, counts,
                                                    mask_id, forbidden_smiles)
                    guided_outcomes = _append_decoded(records, guided_logits, guided.masks,
                                                      scales, "control", stream, counts,
                                                      mask_id, forbidden_smiles)
                    if stream == "audit":
                        for index, (before, after) in enumerate(zip(base_outcomes, guided_outcomes)):
                            if before[0] is None or after[0] is None:
                                continue
                            base_one, masks = _select(base_logits, base.masks, index)
                            guided_one, _ = _select(guided_logits, guided.masks, index)
                            pairs.append(_PerturbationPair(base_one, guided_one, masks,
                                                           before[1], after[1], before[0], after[0]))


def _identity_split(records: list[_Record], seed: int):
    identities = sorted({record.smiles for record in records})
    ordered = sorted(identities, key=lambda identity: hashlib.sha256(
        f"{seed}:{identity}".encode()).digest())
    audit_names = set(ordered[:max(1, round(.25 * len(ordered)))])
    train = [r for r in records if r.smiles not in audit_names and r.stream != "audit"]
    audit = [r for r in records if r.smiles in audit_names and r.stream != "train"]
    # A hard-graph audit alone does not test terminal soft probabilities.
    # If generated identities collapse onto a train group, keep the separate
    # audit noise stream and disclose any canonical identity overlap.
    grouped = bool(train and audit and any(not r.domain.startswith("hard") for r in audit))
    if not grouped:
        train = [r for r in records if r.stream != "audit"]
        audit = [r for r in records if r.stream == "audit"]
    return train, audit, {"grouped_canonical_identity": grouped,
                          "total_identities": len(identities),
                          "audit_identity_count": len(audit_names),
                          "train_identity_count": len({r.smiles for r in train}),
                          "audit_observed_identity_count": len({r.smiles for r in audit}),
                          "identity_overlap": len({r.smiles for r in train} & {r.smiles for r in audit})}


def _stack(records: list[_Record], device):
    if not records:
        raise ValueError("no valid source descriptor records in requested partition")
    return (GraphLogits(torch.cat([r.graph.atoms for r in records]).to(device),
                        torch.cat([r.graph.bonds for r in records]).to(device)),
            GraphMasks(torch.cat([r.masks.nodes for r in records]).to(device),
                       torch.cat([r.masks.pairs for r in records]).to(device)),
            torch.cat([r.label for r in records]).to(device))


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    _, first, counts = np.unique(sorted_values, return_index=True, return_counts=True)
    ranked = np.empty(len(values), dtype=np.float64)
    for start, count in zip(first, counts):
        ranked[order[start:start + count]] = start + (count - 1) / 2
    return ranked


def _metrics(critic, records: list[_Record], device, chunk_size=128):
    if not records:
        return {"records": 0, "rmse_scaled": None, "rank_correlation": None}
    truth, estimates = [], []
    with torch.no_grad():
        for offset in range(0, len(records), chunk_size):
            subset = records[offset:offset + chunk_size]
            graph, masks, labels = _stack(subset, device)
            truth.append(labels.cpu())
            estimates.append(critic(graph, masks, probabilities=True).cpu())
    labels = torch.cat(truth).numpy()
    predicted = torch.cat(estimates).numpy()
    rmse = np.sqrt(np.mean((predicted - labels) ** 2, axis=0)).tolist()
    rank = []
    for endpoint in range(2):
        a, b = _rank(labels[:, endpoint]), _rank(predicted[:, endpoint])
        rank.append(float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else None)
    return {"records": len(records), "rmse_scaled": rmse, "rank_correlation": rank}


def _direction_agreement(critic, pairs: list[_PerturbationPair], audit_names: set[str], device):
    signs, actuals = [], []
    for pair in pairs:
        if pair.baseline_smiles not in audit_names or pair.controlled_smiles not in audit_names:
            continue
        atom = pair.baseline.atoms.to(device).requires_grad_(True)
        bond = pair.baseline.bonds.to(device).requires_grad_(True)
        masks = pair.masks.to(device)
        predicted = critic(GraphLogits(atom, bond), masks)
        for endpoint in range(2):
            ga, gb = torch.autograd.grad(predicted[:, endpoint].sum(), (atom, bond),
                                         retain_graph=endpoint == 0)
            directional = (ga * (pair.controlled.atoms.to(device) - atom.detach())).sum()
            directional += (gb * (pair.controlled.bonds.to(device) - bond.detach())).sum()
            actual = pair.controlled_label[0, endpoint] - pair.baseline_label[0, endpoint]
            signs.append(float(directional.detach().cpu()))
            actuals.append(float(actual))
    directional = np.asarray(signs).reshape(-1, 2) if signs else np.empty((0, 2))
    actual = np.asarray(actuals).reshape(-1, 2) if actuals else np.empty((0, 2))
    counts, agreements = [], []
    for endpoint in range(2):
        informative = (np.abs(actual[:, endpoint]) > 1e-6) & (np.abs(directional[:, endpoint]) > 1e-8)
        counts.append(int(informative.sum()))
        agreements.append(float(np.mean(np.sign(actual[informative, endpoint]) ==
                                        np.sign(directional[informative, endpoint])))
                          if informative.any() else None)
    return {"paired_valid_audit_perturbations": len(directional),
            "informative_pairs_per_descriptor": counts,
            "gradient_direction_agreement": agreements,
            "method": "critic logit Jacobian dotted with paired control-logit displacement; sign compared with RDKit descriptor finite difference"}


def _lead_and_response_bias(critic, dataset, audit_records, scales, device):
    """Audit explicit lead bias under one known source W for each mask."""
    source = json.loads((dataset.data_dir / "private/oracle.json").read_text())
    by_mask = {}
    for cid in dataset.campaign_ids:
        mask_id = dataset.manifest["campaigns"][cid]["mask_id"]
        by_mask.setdefault(mask_id, cid)
    report = {}
    for mask_id, mask in dataset.masks.items():
        lead = collate_graphs([dataset.graph(mask.lead_index)])
        onehot = hard_graph_probabilities(lead)
        lead_graph = GraphLogits(onehot.atoms.to(device), onehot.bonds.to(device))
        with torch.no_grad():
            predicted_lead = critic(lead_graph, lead.masks.to(device),
                                    probabilities=True)[0].cpu()
        true_lead = torch.tensor(_descriptor(mask.lead_smiles) / scales,
                                 dtype=torch.float32)
        lead_error = predicted_lead - true_lead
        cid = by_mask[mask_id]
        W = torch.tensor(source["coefficients"][cid], dtype=torch.float32)
        predicted_lead_response = torch.einsum(
            "ij,j->i", W, descriptor_features(predicted_lead[None], true_lead[None])[0])
        subset = [r for r in audit_records if r.mask_id == mask_id and
                  not r.domain.startswith("hard")]
        if subset:
            graph, masks, labels = _stack(subset, device)
            with torch.no_grad():
                descriptors = critic(graph, masks, probabilities=True).cpu()
            true_features = descriptor_features(labels.cpu(), true_lead[None].expand(len(subset), -1))
            predicted_features = descriptor_features(descriptors, true_lead[None].expand(len(subset), -1))
            exact = torch.einsum("ij,bj->bi", W, true_features)
            predicted = torch.einsum("ij,bj->bi", W, predicted_features)
            error = predicted - exact
            response = {"audit_generated_records": len(subset),
                        "mean_signed_error": error.mean(0).tolist(),
                        "rmse": error.square().mean(0).sqrt().tolist()}
        else:
            response = {"audit_generated_records": 0,
                        "mean_signed_error": None, "rmse": None}
        report[mask_id] = {"lead_true_scaled": true_lead.tolist(),
                           "lead_predicted_scaled": predicted_lead.tolist(),
                           "hard_lead_scaled_error": lead_error.tolist(),
                           "predicted_lead_response_bias": predicted_lead_response.tolist(),
                           "source_coefficient_campaign": cid,
                           "response_bias_audit": response}
    return report


def train_critic(checkpoint, output, device="cuda", seed=142, steps=3000,
                 source_attempts=512, *, control_attempts=128, nfe=16,
                 chunk_size=32):
    """Build source hard/generated records, fit critic, and save an audit ledger.

    The separate audit noise stream is reserved before sampling. Identity
    grouping is applied after canonical RDKit decoding; if it yields an empty
    partition, the report records a seed-separated fallback and overlap.
    """
    if (steps < 1 or steps > 5000 or source_attempts < 4 or control_attempts < 0
            or nfe < 2 or chunk_size < 1):
        raise ValueError("invalid critic training budget")
    started = perf_counter()
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("allocated CUDA device is required")
    checkpoint, output = Path(checkpoint), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model, payload = load_model(checkpoint, device)
    model.eval().requires_grad_(False)
    dataset = MaskedEditDataset(split="train", kind="vector")
    scales = np.asarray(dataset.manifest["descriptor_scales_from_source_molecules"], dtype=np.float64)
    records: list[_Record] = []
    pairs: list[_PerturbationPair] = []
    counts = Counter()
    heldout_indices = dataset.molecule_splits["validation"] | dataset.molecule_splits["test"]
    forbidden_smiles = {dataset.smiles[index] for index in heldout_indices}
    _hard_records(dataset, scales, records, counts)
    _hybrid_records(model, dataset, scales, records, counts, device, seed,
                    chunk_size, forbidden_smiles)
    _generated_records(model, dataset, scales, records, pairs, counts, device,
                       seed, source_attempts, control_attempts, nfe, chunk_size,
                       forbidden_smiles)
    train_records, audit_records, split = _identity_split(records, seed + 73)
    if not train_records:
        raise RuntimeError("no valid source-only critic training records")
    torch.manual_seed(seed)
    critic = SoftGraphDescriptorCritic(model.config).to(device)
    graph, masks, labels = _stack(train_records, device)
    fitting = fit_descriptor_critic(critic, graph, masks, labels,
                                    probabilities=True, steps=steps, batch_size=128,
                                    learning_rate=1e-3, seed=seed)
    audit_names = {r.smiles for r in audit_records}
    categories = {
        "hard": [r for r in train_records + audit_records if r.domain.startswith("hard")],
        "train_generated": [r for r in train_records if not r.domain.startswith("hard")],
        "audit_generated": [r for r in audit_records if not r.domain.startswith("hard")],
        "audit_hard": [r for r in audit_records if r.domain.startswith("hard")],
    }
    diagnostics = {name: _metrics(critic, subset, device) for name, subset in categories.items()}
    diagnostics["finite_difference_gradient_direction"] = _direction_agreement(
        critic, pairs, audit_names, device)
    diagnostics["lead_and_response_bias"] = _lead_and_response_bias(
        critic, dataset, audit_records, scales, device)
    generated_attempts = sum(value for key, value in counts.items()
                             if key.endswith("_attempts") and
                             key.split("_")[0] in {"unguided", "control"})
    hybrid_attempts = sum(value for key, value in counts.items()
                          if key.startswith("hybrid_") and key.endswith("_attempts"))
    summary = {"schema_version": 1, "checkpoint": str(checkpoint.resolve()),
               "checkpoint_sha256": digest(checkpoint), "source_manifest_sha256":
               hashlib.sha256((dataset.data_dir / "manifest.json").read_bytes()).hexdigest(),
               "source_only": True, "targets": "artificial source-scaled RDKit MolLogP and TPSA",
               "heldout_molecule_identities_excluded": len(forbidden_smiles),
               "descriptor_scales": scales.tolist(), "variant": payload["variant"],
               "seed": seed, "steps": steps, "source_attempts_per_mask": source_attempts,
               "controlled_attempts_per_mask": control_attempts,
               "control_radii": [.1, .5, 1.], "sampling_nfe": nfe,
               "generated_forward_sample_nfe": generated_attempts * nfe,
               "terminal_decoder_graph_calls": generated_attempts + hybrid_attempts,
               "seconds_including_chemistry_and_fit": perf_counter() - started,
               "counts": dict(sorted(counts.items())), "partition": split,
               "fitting": fitting, "diagnostics": diagnostics,
               "critic_architecture": {"hidden_dim": critic.hidden_dim, "depth": critic.depth,
                                       "model_config": asdict(model.config)}}
    checkpoint_payload = {"schema_version": 1, "state_dict":
                          {key: value.detach().cpu() for key, value in critic.state_dict().items()},
                          "model_config": asdict(model.config), "hidden_dim": critic.hidden_dim,
                          "depth": critic.depth, "metadata": summary}
    temporary = output / "critic.pt.tmp"
    torch.save(checkpoint_payload, temporary)
    temporary.replace(output / "critic.pt")
    atomic_json(output / "summary.json", summary)
    return summary


def load_critic(path, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    from .contracts import ModelConfig
    critic = SoftGraphDescriptorCritic(ModelConfig(**payload["model_config"]),
                                       hidden_dim=payload["hidden_dim"],
                                       depth=payload["depth"])
    critic.load_state_dict(payload["state_dict"])
    return critic.to(device).eval().requires_grad_(False), payload["metadata"]
