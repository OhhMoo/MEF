"""Descriptor-aware masked-edit controls with post-commit exact evaluation.

These are privileged structure-descriptor baselines: they compute RDKit
features for a proposed structure before deciding which candidates receive an
exact campaign response call. Finite-library retrieval is reported separately
from direct graph mutation because retrieved structures are already supplied.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np

from .masked_edit_data import (DEFAULT_DIR, MaskedEditDataset, EditMask, _descriptor,
                               _features, fit_support_response)
from .masked_edit_evaluation import CandidateLedger, ExactMaskedEvaluator


def _stable_seed(seed: int, text: str) -> int:
    return seed + int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "little")


def _write_csv(path: Path, records: list[dict], fields: tuple[str, ...]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def _mutation_operations(mask: EditMask) -> list[tuple[str, int, int]]:
    operations = []
    for kind, positions, base in (("atom", np.flatnonzero(mask.atom_editable), mask.lead_atoms),
                                  ("bond", np.flatnonzero(mask.bond_editable), mask.lead_bonds)):
        for position in positions:
            for label in range(4):
                if label != int(base[position]):
                    operations.append((kind, int(position), label))
    return operations


def _direct_mutations(mask: EditMask, budget: int, seed: int):
    """Deterministic one- and two-category proposals; invalid attempts survive."""
    operations = _mutation_operations(mask)
    singles = [(operation,) for operation in operations]
    doubles = [pair for pair in itertools.combinations(operations, 2)
               if (pair[0][0], pair[0][1]) != (pair[1][0], pair[1][1])]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(doubles))
    for sequence in itertools.chain(singles, (doubles[int(i)] for i in order)):
        atoms, bonds = mask.lead_atoms.copy(), mask.lead_bonds.copy()
        for kind, position, label in sequence:
            (atoms if kind == "atom" else bonds)[position] = label
        yield {"atoms": atoms, "bonds": bonds, "count": 9,
               "proposal_edits": len(sequence)}
        budget -= 1
        if budget == 0:
            return


def _predicted_delta(smiles: str, lead_smiles: str, scales, fitted_w: np.ndarray) -> np.ndarray:
    basis = _features(_descriptor(smiles), _descriptor(lead_smiles), np.asarray(scales))
    return basis @ fitted_w.T


def _prediction_key(predicted: np.ndarray, goal: np.ndarray) -> tuple:
    violation = np.maximum(goal - predicted, 0)
    feasible = bool(np.all(violation == 0))
    return ((0, -float(predicted[0]), -float(predicted[1])) if feasible else
            (1, float(violation.sum()), -float(predicted[0])))


def _committed_candidates(method: str, dataset: MaskedEditDataset, cid: str,
                          mask: EditMask, reserve: set[int], budget: int,
                          seed: int, ledger: CandidateLedger):
    if method == "finite_library":
        records = dataset.compatible_library(cid, exclude_indices=reserve)
        records = records[:budget]
    elif method == "direct_mutation":
        records = list(_direct_mutations(mask, budget, seed))
    else:
        raise ValueError(method)
    return [(ledger.commit(graph, mask), graph) for graph in records]


def evaluate_simple_baselines(split: str, output, *, seed: int = 142,
                              budget: int = 256, k: int = 16,
                              data_dir=DEFAULT_DIR, kind: str = "vector") -> dict:
    """Run retrieval and direct mutation under common supports and top-K calls.

    Candidate graphs are committed before any campaign oracle call. The support
    fit and ranking know RDKit structure descriptors; the hidden W and exact
    generated-candidate responses are accessed only after top-K commitment.
    Full-library truth is calculated last, in a separate report.
    """
    if split not in ("validation", "test"):
        raise ValueError("simple baseline evaluation expects a held-out split")
    if budget < 1 or k < 1 or k > budget:
        raise ValueError("require budget>=k>=1")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    dataset = MaskedEditDataset(data_dir, split, kind=kind)
    scales = np.asarray(dataset.manifest["descriptor_scales_from_source_molecules"], dtype=np.float64)
    source_smiles = {dataset.smiles[int(index)] for index in dataset.source_graph_indices}
    all_rows, summaries, calibration, truth_rows = [], [], [], []
    # One ledger per method and support size reports distinct molecular output
    # globally across all campaigns in that comparison arm.
    ledgers = {(method, size): CandidateLedger(scope=f"{method}:support_{size}:all_campaigns")
               for method in ("finite_library", "direct_mutation")
               for size in (4, 8, 16)}
    oracle_path = Path(data_dir) / "private" / "oracle.json"

    for campaign_index, cid in enumerate(dataset.campaign_ids):
        meta = dataset.manifest["campaigns"][cid]
        mask = dataset.masks[meta["mask_id"]]
        reserve = dataset.nested_support(campaign_index,
                                         seed=_stable_seed(seed, mask.mask_id),
                                         sizes=(4, 8, 16))
        max_reserve = set(map(int, reserve["reserve_indices"]))
        max_support_smiles = set(reserve["reserve_smiles"])
        goal = np.asarray(meta["goal"], dtype=np.float64)
        library = dataset.compatible_library(cid, exclude_indices=max_reserve)
        known_library_smiles = {record["smiles"] for record in library}

        for size in (4, 8, 16):
            support = reserve["records"][size]
            fit_started = time.perf_counter()
            fit = fit_support_response([graph["smiles"] for graph in support["graphs"]],
                                       support["values"], mask.lead_smiles, scales)
            fit_seconds = time.perf_counter() - fit_started
            fitted_w = fit["coefficients"]
            for method in ("finite_library", "direct_mutation"):
                method_started = time.perf_counter()
                ledger = ledgers[(method, size)]
                committed = _committed_candidates(method, dataset, cid, mask,
                                                  max_reserve, budget,
                                                  _stable_seed(seed, f"{cid}:{method}"), ledger)
                ranked = []
                rows = []
                ranked_identities = set()
                for candidate, graph in committed:
                    identity = candidate.smiles
                    source_match = bool(identity and identity in source_smiles)
                    support_match = bool(identity and identity in max_support_smiles)
                    usable = bool(candidate.chemically_valid and candidate.connected and
                                  candidate.protected_copy and identity and
                                  identity != mask.lead_smiles and not support_match and
                                  not source_match)
                    predicted = (_predicted_delta(identity, mask.lead_smiles, scales, fitted_w)
                                 if usable else None)
                    row = {"campaign_id": cid, "split": split, "method": method,
                           "support_size": size, "mask_id": mask.mask_id,
                           "attempt_number": candidate.attempt_number,
                           "smiles": identity or "", "chemically_valid": candidate.chemically_valid,
                           "connected": candidate.connected,
                           "failure_reason": candidate.failure_reason or "",
                           "protected_copy": candidate.protected_copy,
                           "atom_edits": candidate.atom_edits,
                           "bond_edits": candidate.bond_edits,
                           "total_edits": candidate.total_edits,
                           "global_unique": candidate.global_unique,
                           "uniqueness_scope": ledger.scope,
                           "source_match": source_match, "support_match": support_match,
                           "known_library_match": bool(identity in known_library_smiles),
                           "predicted_delta_1": "" if predicted is None else float(predicted[0]),
                           "predicted_delta_2": "" if predicted is None else float(predicted[1]),
                           "selected_rank": "", "measured_delta_1": "",
                           "measured_delta_2": "", "measured_goal_success": "",
                           "eligible_novel_success": "",
                           "atoms": json.dumps(list(candidate.atoms)),
                           "bonds": json.dumps(list(candidate.bonds))}
                    rows.append(row)
                    if predicted is not None and identity not in ranked_identities:
                        ranked.append((_prediction_key(predicted, goal), len(rows)-1,
                                       candidate))
                        ranked_identities.add(identity)
                ranked.sort(key=lambda item: (item[0], item[1]))
                selected = ranked[:k]
                # Exact coefficients are loaded only after the complete ranked
                # proposal list has been committed to immutable records.
                evaluator = ExactMaskedEvaluator(oracle_path)
                evaluator.ledger = ledger
                measured_successes = 0
                eligible_successes = 0
                for rank, (_, row_index, candidate) in enumerate(selected, 1):
                    outcome = evaluator.evaluate_committed(candidate, cid,
                        support_smiles=max_support_smiles, source_smiles=source_smiles,
                        known_library_smiles=known_library_smiles)
                    row = rows[row_index]
                    row["selected_rank"] = rank
                    row["measured_delta_1"], row["measured_delta_2"] = outcome["exact_delta"]
                    row["measured_goal_success"] = outcome["goal_success"]
                    row["eligible_novel_success"] = outcome["eligible_novel_success"]
                    measured_successes += bool(outcome["goal_success"])
                    eligible_successes += bool(outcome["eligible_novel_success"])
                all_rows.extend(rows)
                summaries.append({"campaign_id": cid, "split": split,
                                  "method": method, "support_size": size,
                                  "mask_id": mask.mask_id, "candidate_budget": budget,
                                  "attempts": len(rows), "invalid_attempts": sum(
                                      not row["chemically_valid"] or not row["connected"] for row in rows),
                                  "protected_valid_attempts": len(ranked),
                                  "source_matches": sum(row["source_match"] for row in rows),
                                  "unseen_source_valid": sum(
                                      row["chemically_valid"] and row["connected"] and
                                      not row["source_match"] for row in rows),
                                  "distinct_global_attempts": sum(row["global_unique"] for row in rows),
                                  "topk_requested": k, "topk_filled": len(selected),
                                  "topk_unfilled": k-len(selected),
                                  "exact_candidate_calls": len(selected),
                                  "topk_measured_success": measured_successes,
                                  "topk_eligible_novel_success": eligible_successes,
                                  "support_fit_seconds": fit_seconds,
                                  "candidate_and_oracle_seconds": time.perf_counter() - method_started,
                                  "total_seconds_including_host_chemistry":
                                      fit_seconds + time.perf_counter() - method_started})

            # Calibration diagnostics are computed after both ranked arms have
            # committed/evaluated their top K for this support size.
            true_w = evaluator.coefficients[cid]
            query_indices = meta["molecule_indices"]
            features = _features(np.stack([_descriptor(dataset.smiles[index])
                                            for index in query_indices]),
                                 _descriptor(mask.lead_smiles), scales)
            predicted = features @ fitted_w.T
            actual = features @ true_w.T
            calibration.append({"campaign_id": cid, "split": split,
                                "support_size": size, "mask_id": mask.mask_id,
                                "basis_rank": fit["basis_rank"],
                                "singular_values": json.dumps(fit["singular_values"].tolist()),
                                "condition_number": fit["condition_number"],
                                "coefficient_frobenius_error": float(np.linalg.norm(fitted_w - true_w)),
                                "heldout_prediction_rmse": float(np.sqrt(np.mean((predicted-actual)**2))),
                                "fitted_w": json.dumps(fitted_w.tolist()),
                                "true_w_postcommit": json.dumps(true_w.tolist())})

        # Full known-library feasibility is an unsynthesized retrieval ceiling.
        # It is never used to select a generated candidate or fit W.
        truth_evaluator = ExactMaskedEvaluator(oracle_path)
        feasible = sum(bool(np.all(truth_evaluator.exact_delta(cid, graph["smiles"]) >= goal))
                       for graph in library)
        truth_rows.append({"campaign_id": cid, "split": split, "mask_id": mask.mask_id,
                           "compatible_library_outside_max_support": len(library),
                           "feasible_known_library": feasible,
                           "diagnostic_truth_calls": len(library),
                           "classification": "unsynthesized_library_truth"})

    attempts_file = output / "raw_attempts.csv"
    summary_file = output / "topk_summary.csv"
    calibration_file = output / "support_calibration.csv"
    truth_file = output / "library_truth.csv"
    _write_csv(attempts_file, all_rows, tuple(all_rows[0]))
    _write_csv(summary_file, summaries, tuple(summaries[0]))
    _write_csv(calibration_file, calibration, tuple(calibration[0]))
    _write_csv(truth_file, truth_rows, tuple(truth_rows[0]))
    report = {"split": split, "kind": kind, "seed": seed, "candidate_budget": budget,
              "topk": k, "campaigns": len(dataset.campaign_ids),
              "methods": ["finite_library", "direct_mutation"],
              "support_sizes": [4, 8, 16],
              "files": {"raw_attempts": attempts_file.name, "topk_summary": summary_file.name,
                        "support_calibration": calibration_file.name,
                        "library_truth": truth_file.name},
              "privilege": "RDKit structure descriptor basis and exact campaign oracle for selected top K",
              "selection_rule": "predicted feasible then endpoint 1; otherwise minimum predicted shortfall",
              "truth_separation": "full-library feasibility is a diagnostic ceiling and is not generated"}
    (output / "manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
