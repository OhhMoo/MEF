"""Privileged exact-oracle terminal control for the masked-edit diagnostic.

This is a deliberately expensive black-box control, not a deployable evidence
policy. Its basis uses source aligned edit pairs only; every held-out exact
response is charged to a terminal sample after that graph has been committed.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np
import torch

from .contracts import GraphBatch, GraphMasks, Latents
from .masked_edit_evaluation import ExactMaskedEvaluator
from .masked_edit_model import MaskedEditCondition, MaskedEditModel
from .masked_edit_sampling import (decode_masked_edit_result, make_paired_gaussian,
                                   sample_masked_edit_latents)
from .masked_edit_data import EditMask, MaskedEditDataset


def _batch_graphs(graphs: list[dict], device) -> GraphBatch:
    atoms = torch.as_tensor(np.stack([g["atoms"] for g in graphs]), dtype=torch.long, device=device)
    bonds = torch.as_tensor(np.stack([g["bonds"] for g in graphs]), dtype=torch.long, device=device)
    counts = torch.as_tensor([g["count"] for g in graphs], dtype=torch.long, device=device)
    masks = GraphMasks.from_counts(counts, atoms.shape[1])
    return GraphBatch(atoms, bonds, masks,
                      tuple(g.get("smiles", "") for g in graphs),
                      tuple(g.get("id", "") for g in graphs))


def _model_device(model: MaskedEditModel):
    return next(model.parameters()).device


def build_basis(model: MaskedEditModel, dataset: MaskedEditDataset, *, rank: int = 8,
                batch_size: int = 32) -> dict[str, Latents]:
    """Uncentered PCA of source-only aligned E(query)-E(lead) edit vectors.

    Each task gets an independent basis. Components have unit RMS over the
    editable node/pair coordinates, so a coefficient has a comparable velocity
    scale across different masks. No campaign outcomes enter this basis.
    """
    if dataset.split != "train" or rank < 1 or batch_size < 1:
        raise ValueError("basis requires source split and positive rank/batch_size")
    source_indices = set(map(int, dataset.source_graph_indices))
    device = _model_device(model)
    was_training = model.training
    model.eval()
    bases = {}
    try:
        with torch.inference_mode():
            for task in dataset.tasks:
                task_id = task["task_id"]
                indices = list(map(int, task["compatible_indices"]["train"]))
                if not set(indices) <= source_indices:
                    raise ValueError("basis includes a held-out molecule")
                lead = _batch_graphs([dataset.graph(task["lead_index"])], device)
                lead_latent = model.encode(lead)
                node_mask = torch.as_tensor(task["atom_editable"], dtype=torch.bool, device=device)
                pair_mask = torch.as_tensor(task["bond_editable"], dtype=torch.bool, device=device)
                rows = []
                for start in range(0, len(indices), batch_size):
                    graphs = [dataset.graph(index, aligned=True, task_id=task_id)
                              for index in indices[start:start + batch_size]]
                    query = model.encode(_batch_graphs(graphs, device))
                    node_diff = (query.nodes - lead_latent.nodes) * node_mask[None, :, None]
                    pair_diff = (query.pairs - lead_latent.pairs) * pair_mask[None, :, None]
                    rows.append(torch.cat((node_diff.flatten(1), pair_diff.flatten(1)), 1).float().cpu())
                matrix = torch.cat(rows, 0)
                # Gram PCA is cheaper than SVD on the much wider latent axis.
                gram = matrix @ matrix.T
                values, vectors = torch.linalg.eigh(gram)
                order = torch.argsort(values, descending=True)
                values, vectors = values[order], vectors[:, order]
                keep = values > max(float(values[0]), 1.0) * 1e-8
                k = min(rank, int(keep.sum()))
                if k < 1:
                    raise ValueError(f"source edits have zero latent rank for {task_id}")
                components = (vectors[:, :k].T @ matrix) / values[:k, None].sqrt()
                node_shape = lead_latent.nodes.shape[1:]
                pair_shape = lead_latent.pairs.shape[1:]
                node_size = math.prod(node_shape)
                active_dims = int(node_mask.sum()) * node_shape[-1] + int(pair_mask.sum()) * pair_shape[-1]
                components *= math.sqrt(active_dims)
                bases[task_id] = Latents(components[:, :node_size].reshape(k, *node_shape).to(device),
                                          components[:, node_size:].reshape(k, *pair_shape).to(device))
    finally:
        model.train(was_training)
    return bases


def _limit_coefficients(coefficients: np.ndarray, max_velocity_rms: float) -> np.ndarray:
    clipped = np.asarray(coefficients, dtype=np.float32).copy()
    for row in clipped:
        norm = float(np.linalg.norm(row))
        if norm > max_velocity_rms:
            row *= max_velocity_rms / norm
    return clipped


def _control_from_basis(basis: Latents, coefficients: np.ndarray, T: float):
    weights = torch.as_tensor(coefficients, dtype=basis.nodes.dtype, device=basis.nodes.device)

    def velocity_controller(state: Latents, t: torch.Tensor,
                            condition: MaskedEditCondition) -> Latents:
        if state.nodes.shape[0] != 1:
            raise ValueError("oracle control evaluates one starting sample at a time")
        bin_index = int((t.reshape(-1)[0] >= T / 2).item())
        row = weights[bin_index]
        return Latents(torch.einsum("k,knd->nd", row, basis.nodes)[None],
                       torch.einsum("k,kpd->pd", row, basis.pairs)[None])

    return velocity_controller


def _candidate_from_logits(logits, count: int = 9) -> dict:
    return {"atoms": logits.atoms[0].argmax(-1).detach().cpu().numpy().astype(np.uint8),
            "bonds": logits.bonds[0].argmax(-1).detach().cpu().numpy().astype(np.uint8),
            "count": count}


def _selection_key(record: dict) -> tuple[float, float, float]:
    """Support-excluded feasible candidates maximize endpoint 1; otherwise violation."""
    delta = record["exact_delta"]
    excluded = bool(record["lead_copy"] or record["support_match"])
    if delta is None:
        return (2.0, float("inf"), float("inf"))
    if record["support_excluded_success"]:
        return (0.0, -float(delta[0]), -float(delta[1]))
    goal = np.asarray(record["goal"], dtype=np.float64)
    violation = float(np.maximum(goal - np.asarray(delta), 0).sum())
    return (1.0, 10.0 * excluded + violation, -float(delta[0]))


@dataclass(frozen=True)
class ControlSearchResult:
    method: str
    campaign_id: str
    mask_id: str
    attempts: tuple[dict, ...]
    best_attempt: dict
    terminal_calls: int
    forward_nfe: int
    decoder_calls: int
    elapsed_seconds: float
    initial_noise_seed: int | None

    def as_dict(self) -> dict:
        return {"method": self.method, "campaign_id": self.campaign_id,
                "mask_id": self.mask_id, "attempts": list(self.attempts),
                "best_attempt": self.best_attempt, "terminal_calls": self.terminal_calls,
                "forward_nfe": self.forward_nfe, "decoder_calls": self.decoder_calls,
                "elapsed_seconds": self.elapsed_seconds,
                "initial_noise_seed": self.initial_noise_seed}


def _run_terminal(model: MaskedEditModel, condition: MaskedEditCondition,
                  gaussian: Latents, evaluator: ExactMaskedEvaluator,
                  campaign_id: str, mask: EditMask, *, nfe: int, T: float,
                  coefficients: np.ndarray | None, basis: Latents | None,
                  support_smiles, source_smiles) -> dict:
    controller = None if coefficients is None else _control_from_basis(basis, coefficients, T)
    result = sample_masked_edit_latents(model, condition, gaussian, nfe=nfe, T=T,
                                        velocity_controller=controller)
    logits = decode_masked_edit_result(model, result, condition)
    graph = _candidate_from_logits(logits)
    committed = evaluator.ledger.commit(graph, mask)
    evaluated = evaluator.evaluate_committed(committed, campaign_id,
                                             support_smiles=support_smiles,
                                             source_smiles=source_smiles)
    return {**evaluated, "goal": list(evaluator.campaigns[campaign_id]["goal"]),
            "coefficients": None if coefficients is None else coefficients.tolist(),
            "nfe": result.nfe, "decoded_graph": {"atoms": graph["atoms"].tolist(),
                                                  "bonds": graph["bonds"].tolist(),
                                                  "count": graph["count"]}}


def cem_exact_control(model: MaskedEditModel, condition: MaskedEditCondition,
                      gaussian: Latents, basis: Latents, evaluator: ExactMaskedEvaluator,
                      campaign_id: str, mask: EditMask, *, terminal_budget: int = 16,
                      nfe: int = 16, T: float = .95, seed: int = 142,
                      max_velocity_rms: float = .25,
                      support_smiles=(), source_smiles=()) -> ControlSearchResult:
    """Antithetic CEM over two constant velocity bins with exact trial accounting."""
    if not 2 <= terminal_budget <= 16:
        raise ValueError("terminal_budget must be 2..16")
    if condition.lead.nodes.shape[0] != 1 or gaussian.nodes.shape[0] != 1:
        raise ValueError("one starting sample/condition is required")
    if basis.nodes.ndim != 3 or basis.pairs.ndim != 3 or basis.nodes.shape[0] != basis.pairs.shape[0]:
        raise ValueError("basis must have K node and pair directions")
    if not 0 < max_velocity_rms <= 1:
        raise ValueError("max_velocity_rms must be in (0,1]")
    start = time.perf_counter()
    attempts = []
    rng = np.random.default_rng(seed)
    center = np.zeros((2, basis.nodes.shape[0]), dtype=np.float32)
    sigma = np.full_like(center, .08)
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            def trial(coeff):
                clipped = _limit_coefficients(coeff, max_velocity_rms)
                attempt = _run_terminal(model, condition, gaussian, evaluator, campaign_id,
                                        mask, nfe=nfe, T=T, coefficients=clipped, basis=basis,
                                        support_smiles=support_smiles, source_smiles=source_smiles)
                attempts.append(attempt)
                return attempt

            trial(center)  # Unmodified flow is included in the oracle budget.
            generation = []
            while len(attempts) + 1 < terminal_budget:
                draw = rng.standard_normal(center.shape).astype(np.float32) * sigma
                for sign in (+1, -1):
                    attempt = trial(center + sign * draw)
                    generation.append((np.asarray(attempt["coefficients"], dtype=np.float32), attempt))
                if len(generation) >= 4:
                    elite = sorted(generation, key=lambda item: _selection_key(item[1]))[:2]
                    samples = np.stack([item[0] for item in elite])
                    center = _limit_coefficients(samples.mean(0), max_velocity_rms)
                    sigma = np.maximum(.02, .75 * sigma + .25 * samples.std(0))
                    generation.clear()
            if len(attempts) < terminal_budget:
                if terminal_budget == 2:
                    trial(center + rng.standard_normal(center.shape).astype(np.float32) * sigma)
                else:
                    trial(center)
    finally:
        model.train(was_training)
    best = min(attempts, key=_selection_key)
    return ControlSearchResult("exact_oracle_cem", campaign_id, mask.mask_id,
                               tuple(attempts), best, len(attempts),
                               sum(item["nfe"] for item in attempts), len(attempts),
                               time.perf_counter() - start, None)


def unguided_oracle_rank(model: MaskedEditModel, condition: MaskedEditCondition,
                         evaluator: ExactMaskedEvaluator, campaign_id: str,
                         mask: EditMask, *, terminal_budget: int = 16,
                         nfe: int = 16, T: float = .95, seed: int = 142,
                         first_gaussian: Latents | None = None,
                         support_smiles=(), source_smiles=()) -> ControlSearchResult:
    """Independent unguided samples; exact oracle only ranks matched calls."""
    if not 1 <= terminal_budget <= 16:
        raise ValueError("terminal_budget must be 1..16")
    start = time.perf_counter()
    attempts = []
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for index in range(terminal_budget):
                config = model.config.model if hasattr(model.config, "model") else model.config
                gaussian = (first_gaussian if index == 0 and first_gaussian is not None else
                            make_paired_gaussian(config, 1, seed + index,
                                                 device=condition.lead.nodes.device))
                attempts.append(_run_terminal(model, condition, gaussian, evaluator,
                                              campaign_id, mask, nfe=nfe, T=T,
                                              coefficients=None, basis=None,
                                              support_smiles=support_smiles,
                                              source_smiles=source_smiles))
    finally:
        model.train(was_training)
    best = min(attempts, key=_selection_key)
    return ControlSearchResult("unguided_oracle_rank", campaign_id, mask.mask_id,
                               tuple(attempts), best, len(attempts),
                               sum(item["nfe"] for item in attempts), len(attempts),
                               time.perf_counter() - start, seed)
