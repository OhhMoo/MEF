"""Run matched privileged oracle control on a selected masked-edit checkpoint."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .masked_edit_data import DEFAULT_DIR, MaskedEditDataset
from .masked_edit_evaluation import ExactMaskedEvaluator
from .masked_edit_oracle_control import (build_basis, cem_exact_control,
                                         unguided_oracle_rank)
from .masked_edit_sampling import make_paired_gaussian
from .masked_edit_run import condition_for, load_model, now, write_csv
from .redesign_train import atomic_json, digest


def _seed(seed: int, identity: str) -> int:
    return seed + int.from_bytes(hashlib.sha256(identity.encode()).digest()[:4], "little")


def _case_metrics(result, campaign_id: str, method: str, case: int,
                  support_indices, reserve_smiles) -> dict:
    attempts = result.attempts
    best = result.best_attempt
    eligible = [row for row in attempts if row["chemically_valid"] and row["connected"]
                and row["protected_copy"] and not row["lead_copy"]]
    successes = [row for row in attempts if row["support_excluded_success"]]
    strict_successes = [row for row in attempts if row["eligible_novel_success"]]
    return {"row_type": "case", "campaign_id": campaign_id, "method": method,
            "case": case, "support_size": len(support_indices),
            "max_reserve_size": len(reserve_smiles),
            "terminal_calls": result.terminal_calls, "forward_nfe": result.forward_nfe,
            "decoder_calls": result.decoder_calls,
            "elapsed_seconds_including_chemistry": result.elapsed_seconds,
            "valid_protected_attempts": len(eligible),
            "unique_valid_protected_smiles": len({row["smiles"] for row in eligible}),
            "successful_attempts": len(successes),
            "unique_successful_smiles": len({row["smiles"] for row in successes}),
            "strict_novel_successful_attempts": len(strict_successes),
            "strict_novel_unique_successful_smiles": len({row["smiles"] for row in strict_successes}),
            "source_match_successful_attempts": sum(row["source_match"] for row in successes),
            "source_matches": sum(row["source_match"] for row in attempts),
            "support_matches": sum(row["support_match"] for row in attempts),
            "global_unique_attempts": sum(row["global_unique"] for row in attempts),
            "best_attempt_number": best["attempt_number"],
            "best_smiles": best["smiles"],
            "best_support_excluded_success": best["support_excluded_success"],
            "best_eligible_novel_success": best["eligible_novel_success"],
            "best_exact_delta_1": "" if best["exact_delta"] is None else best["exact_delta"][0],
            "best_exact_delta_2": "" if best["exact_delta"] is None else best["exact_delta"][1]}


def _campaign_metrics(case_rows: list[dict], attempts: list[dict], campaign_id: str,
                      method: str) -> dict:
    rows = [row for row in attempts if row["campaign_id"] == campaign_id and row["method"] == method]
    selected = [row for row in case_rows if row["campaign_id"] == campaign_id and
                row["method"] == method]
    successes = [row for row in rows if row["support_excluded_success"]]
    strict_successes = [row for row in rows if row["eligible_novel_success"]]
    return {"row_type": "campaign", "campaign_id": campaign_id, "method": method,
            "case": "", "support_size": selected[0]["support_size"],
            "max_reserve_size": selected[0]["max_reserve_size"],
            "terminal_calls": sum(row["terminal_calls"] for row in selected),
            "forward_nfe": sum(row["forward_nfe"] for row in selected),
            "decoder_calls": sum(row["decoder_calls"] for row in selected),
            "elapsed_seconds_including_chemistry": sum(
                row["elapsed_seconds_including_chemistry"] for row in selected),
            "valid_protected_attempts": sum(row["valid_protected_attempts"] for row in selected),
            "unique_valid_protected_smiles": len({row["smiles"] for row in rows
                if row["chemically_valid"] and row["connected"] and row["protected_copy"]
                and not row["lead_copy"]}),
            "successful_attempts": len(successes),
            "unique_successful_smiles": len({row["smiles"] for row in successes}),
            "strict_novel_successful_attempts": len(strict_successes),
            "strict_novel_unique_successful_smiles": len({row["smiles"] for row in strict_successes}),
            "source_match_successful_attempts": sum(row["source_match"] for row in successes),
            "source_matches": sum(row["source_match"] for row in rows),
            "support_matches": sum(row["support_match"] for row in rows),
            "global_unique_attempts": sum(row["global_unique"] for row in rows),
            "best_cases_successful": sum(row["best_support_excluded_success"] for row in selected),
            "best_cases_strict_novel_successful": sum(row["best_eligible_novel_success"] for row in selected),
            "best_attempt_number": "", "best_smiles": "",
            "best_support_excluded_success": "", "best_eligible_novel_success": "",
            "best_exact_delta_1": "",
            "best_exact_delta_2": ""}


def run_oracle_experiment(task: dict, device: str) -> dict:
    """Evaluate 16 vector campaigns with 4 starts × 16 exact calls per arm.

    The default evaluation costs 2 methods × 16 campaigns × 4 starting draws ×
    16 terminal calls = 2048 terminal graphs, each with 16 forward evaluations.
    `campaign_limit`, `initial_noises`, `terminal_budget`, and `nfe` support an
    operational smoke run without changing the declared full-study defaults.
    """
    checkpoint = Path(task["checkpoint"])
    output = Path(task["output"])
    output.mkdir(parents=True, exist_ok=True)
    split = task.get("split", "validation")
    if split not in ("validation", "test"):
        raise ValueError("oracle experiment requires validation or test")
    initial_noises = int(task.get("initial_noises", 4))
    terminal_budget = int(task.get("terminal_budget", 16))
    nfe = int(task.get("nfe", 16))
    if not 1 <= initial_noises <= 64 or not 2 <= terminal_budget <= 16 or nfe < 2:
        raise ValueError("invalid oracle experiment budget")
    seed = int(task.get("seed", 142))
    max_velocity_rms = float(task.get("max_velocity_rms", .25))
    if not 0 < max_velocity_rms <= 1:
        raise ValueError("max_velocity_rms must be in (0,1]")
    data_dir = Path(task.get("data_dir", DEFAULT_DIR))
    started = time.perf_counter()
    model, checkpoint_payload = load_model(checkpoint, device)
    model.eval()
    source = MaskedEditDataset(data_dir, "train", kind="vector")
    dataset = MaskedEditDataset(data_dir, split, kind="vector")
    basis_started = time.perf_counter()
    basis = build_basis(model, source, rank=int(task.get("basis_rank", 8)))
    basis_seconds = time.perf_counter() - basis_started
    source_smiles = {dataset.smiles[int(i)] for i in dataset.source_graph_indices}
    oracle_path = data_dir / "private" / "oracle.json"
    evaluators = {"exact_oracle_cem": ExactMaskedEvaluator(oracle_path),
                  "unguided_oracle_rank": ExactMaskedEvaluator(oracle_path)}
    attempts, case_metrics = [], []
    campaign_ids = dataset.campaign_ids[:task.get("campaign_limit")]
    for campaign_index, cid in enumerate(campaign_ids):
        meta = dataset.manifest["campaigns"][cid]
        mask = dataset.masks[meta["mask_id"]]
        # Same maximal reserve as the representation evaluation. Every support
        # size nests within these 16, and the entire reserve is excluded from
        # novel-candidate credit even when only four observations are supplied.
        support = dataset.nested_support(campaign_index, seed=77600 + campaign_index,
                                         sizes=(4, 8, 16))
        condition = condition_for(model, dataset, mask.mask_id)
        for case in range(initial_noises):
            noise_seed = _seed(seed, f"{cid}:{case}")
            gaussian = make_paired_gaussian(model.config, 1, noise_seed, device=device)
            controlled = cem_exact_control(
                model, condition, gaussian, basis[mask.mask_id],
                evaluators["exact_oracle_cem"], cid, mask,
                terminal_budget=terminal_budget, nfe=nfe, seed=noise_seed + 1,
                max_velocity_rms=max_velocity_rms,
                support_smiles=support["reserve_smiles"], source_smiles=source_smiles)
            unguided = unguided_oracle_rank(
                model, condition, evaluators["unguided_oracle_rank"], cid, mask,
                terminal_budget=terminal_budget, nfe=nfe, seed=noise_seed + 2,
                first_gaussian=gaussian,
                support_smiles=support["reserve_smiles"], source_smiles=source_smiles)
            for method, result in (("exact_oracle_cem", controlled),
                                   ("unguided_oracle_rank", unguided)):
                case_metrics.append(_case_metrics(result, cid, method, case,
                                                  support["records"][16]["indices"],
                                                  support["reserve_smiles"]))
                for attempt in result.attempts:
                    attempts.append({**attempt, "split": split,
                                     "variant": checkpoint_payload["variant"],
                                     "checkpoint_seed": checkpoint_payload["task"]["seed"],
                                     "method": method, "case": case,
                                     "initial_noise_seed": noise_seed,
                                     "max_reserve_size": 16,
                                     "case_elapsed_seconds_including_chemistry":
                                         result.elapsed_seconds})
    campaign_metrics = [_campaign_metrics(case_metrics, attempts, cid, method)
                        for cid in campaign_ids
                        for method in ("exact_oracle_cem", "unguided_oracle_rank")]
    metrics = case_metrics + campaign_metrics
    write_csv(output / "attempts.csv", attempts)
    write_csv(output / "metrics.csv", metrics)
    paired = []
    for cid in campaign_ids:
        for case in range(initial_noises):
            controlled = next(row for row in case_metrics if row["campaign_id"] == cid and
                              row["case"] == case and row["method"] == "exact_oracle_cem")
            unguided = next(row for row in case_metrics if row["campaign_id"] == cid and
                            row["case"] == case and row["method"] == "unguided_oracle_rank")
            paired.append({"campaign_id": cid, "case": case,
                           "cem_best_success": controlled["best_support_excluded_success"],
                           "unguided_best_success": unguided["best_support_excluded_success"],
                           "success_difference": int(controlled["best_support_excluded_success"]) -
                               int(unguided["best_support_excluded_success"]),
                           "cem_best_strict_novel_success": controlled["best_eligible_novel_success"],
                           "unguided_best_strict_novel_success": unguided["best_eligible_novel_success"],
                           "strict_novel_success_difference":
                               int(controlled["best_eligible_novel_success"]) -
                               int(unguided["best_eligible_novel_success"]),
                           "same_first_gaussian": True,
                           "matched_terminal_calls": controlled["terminal_calls"] == unguided["terminal_calls"],
                           "matched_forward_nfe": controlled["forward_nfe"] == unguided["forward_nfe"]})
    by_method = {}
    for method in ("exact_oracle_cem", "unguided_oracle_rank"):
        rows = [row for row in attempts if row["method"] == method]
        cases = [row for row in case_metrics if row["method"] == method]
        by_method[method] = {"terminal_calls": sum(row["terminal_calls"] for row in cases),
                             "forward_nfe": sum(row["forward_nfe"] for row in cases),
                             "decoder_calls": sum(row["decoder_calls"] for row in cases),
                             "best_cases_successful": sum(row["best_support_excluded_success"] for row in cases),
                             "best_cases_strict_novel_successful": sum(
                                 row["best_eligible_novel_success"] for row in cases),
                             "distinct_valid_protected_smiles": len({row["smiles"] for row in rows
                                 if row["chemically_valid"] and row["connected"] and row["protected_copy"]
                                 and not row["lead_copy"]}),
                             "distinct_novel_success_smiles": len({row["smiles"] for row in rows
                                 if row["eligible_novel_success"]}),
                             "distinct_support_excluded_success_smiles": len({row["smiles"] for row in rows
                                 if row["support_excluded_success"]}),
                             "source_match_success_attempts": sum(row["source_match"] and
                                 row["support_excluded_success"] for row in rows),
                             "global_unique_attempts": sum(row["global_unique"] for row in rows),
                             "source_matches": sum(row["source_match"] for row in rows),
                             "support_matches": sum(row["support_match"] for row in rows),
                             "elapsed_seconds_including_chemistry": sum(
                                 row["elapsed_seconds_including_chemistry"] for row in cases)}
    summary = {"phase": "complete", "study": "privileged_exact_oracle_control",
               "checkpoint": str(checkpoint), "checkpoint_sha256": digest(checkpoint),
               "variant": checkpoint_payload["variant"],
               "checkpoint_seed": checkpoint_payload["task"]["seed"],
               "split": split, "campaigns": len(campaign_ids),
               "initial_noises_per_campaign": initial_noises,
               "terminal_budget_per_case_per_method": terminal_budget,
               "max_velocity_rms": max_velocity_rms,
               "nfe_per_terminal_including_clean_prediction": nfe,
               "basis_rank_requested": int(task.get("basis_rank", 8)),
               "basis_rank_by_mask": {key: int(value.nodes.shape[0]) for key, value in basis.items()},
               "basis_source_molecules": len(source.source_graph_indices),
               "basis_seconds": basis_seconds,
               "support_size": 16, "max_reserve_excluded": True,
               "source_matches_excluded_from_novel_success": True,
               "source_matches_counted_in_main_support_excluded_success": True,
               "all_attempts_retained_including_invalid_and_source_support_matches": True,
               "privilege": "held-out exact RDKit response consulted after each committed terminal graph within a fixed budget",
               "by_method": by_method,
               "paired_cases": paired,
               "paired_success_difference": sum(row["success_difference"] for row in paired),
               "paired_strict_novel_success_difference": sum(
                   row["strict_novel_success_difference"] for row in paired),
               "elapsed_seconds_including_chemistry_and_basis": time.perf_counter() - started,
               "completed_utc": now(), "task": task}
    atomic_json(output / "summary.json", summary)
    return summary
