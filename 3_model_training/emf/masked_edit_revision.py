"""Midcourse evidence revision with charged prefix, continuation, and restart.

N=4 support determines the old two-bin velocity. A later N=16 support fit
updates only the response evidence. The old controlled prefix is replayed on
the same Gaussian and frozen lead condition; continuation starts at its exact
latent state. Restart and all-evidence-from-start share one W16 terminal result
but have different historical costs. Truth is evaluated after graph commitment.
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .contracts import Latents
from .masked_edit_control import optimize_terminal_control
from .masked_edit_data import MaskedEditDataset, _descriptor, fit_support_response
from .masked_edit_evaluation import ExactMaskedEvaluator
from .masked_edit_experiments import (load_critic, metrics_for, score_prediction,
                                      truth_rows)
from .masked_edit_run import (condition_for, graphs_from_logits, load_model,
                              now, write_csv)
from .masked_edit_sampling import (make_paired_gaussian, masked_initial_state,
                                   sample_masked_edit_latents)
from .redesign_train import atomic_json, tensor_fingerprint


def revision_grid(nfe: int, T: float, targets=(.25, .5, .75)) -> tuple[dict, ...]:
    """Choose nearest interior full-grid states; tiny smoke grids may repeat."""
    if isinstance(nfe, bool) or nfe < 3 or not 0 < T < 1:
        raise ValueError("revision needs nfe>=3 and 0<T<1")
    times = torch.linspace(0, T, nfe)
    interior = range(1, nfe - 1)
    return tuple({"target_time": float(target),
                  "grid_index": min(interior, key=lambda index: abs(float(times[index]) - target)),
                  "actual_time": float(times[min(interior, key=lambda index: abs(float(times[index]) - target))])}
                 for target in targets)


@torch.no_grad()
def replay_prefix_state(model, condition, gaussian: Latents,
                        node_control: torch.Tensor, pair_control: torch.Tensor,
                        *, nfe: int, T: float, grid_index: int) -> Latents:
    """Replay exactly grid_index Euler calls, stopping before terminal decode."""
    if not 1 <= grid_index < nfe - 1:
        raise ValueError("grid_index must be an interior solver state")
    masks = condition.masks
    state = masked_initial_state(model.variant, gaussian, condition)
    times = torch.linspace(0, T, nfe, device=state.nodes.device, dtype=torch.float32)
    for step in range(grid_index):
        t = times[step]
        predicted = model.predict_clean(state, t.expand(state.nodes.shape[0]), masks,
                                        condition).masked(masks)
        base = Latents((predicted.nodes - state.nodes) / (1 - t),
                       (predicted.pairs - state.pairs) / (1 - t))
        if model.variant != "A":
            base = base.masked(condition.edit_masks)
        # Match the sampler's tensor comparison exactly at a bin boundary;
        # converting t to Python float can switch a rounded grid point.
        bin_index = int(not bool(t < T / 2))
        controlled = Latents(node_control[:, bin_index], pair_control[:, bin_index])
        controlled = controlled.masked(condition.edit_masks)
        dt = times[step + 1] - t
        state = Latents(state.nodes + dt * (base.nodes + controlled.nodes),
                        state.pairs + dt * (base.pairs + controlled.pairs)).masked(masks)
        if model.variant != "A":
            state = Latents(torch.where(condition.edit_masks.nodes[..., None],
                                        state.nodes, condition.lead.nodes),
                            torch.where(condition.edit_masks.pairs[..., None],
                                        state.pairs, condition.lead.pairs)).masked(masks)
    return state


def _slice_noise(gaussian: Latents, first: int, count: int) -> Latents:
    return Latents(gaussian.nodes[first:first + count],
                   gaussian.pairs[first:first + count])


def _evaluate(method, target, model, critic, result, condition, W,
              lead_raw, scales, dataset, cid, support, evaluator,
              *, first: int, seed: int, physical: bool, source=None):
    logits, predicted = score_prediction(model, critic, result, condition,
                                         W, lead_raw, scales)
    rows = truth_rows(graphs_from_logits(logits), predicted.cpu().numpy(), evaluator,
                      dataset, cid, support,
                      {"method": method, "revision_target": target,
                       "seed": seed, "split": dataset.split})
    for index, row in enumerate(rows):
        row["sample"] = first + index
        row["physical_truth_evaluation"] = physical
        row["logical_reuse_source"] = source
    return rows


def _copy_rows(rows, method: str, target: float, *, physical: bool, source: str):
    return [{**row, "method": method, "revision_target": target,
             "physical_truth_evaluation": physical,
             "logical_reuse_source": source} for row in rows]


def _response_means(rows):
    eligible = [row for row in rows if row["eligible"] and row["exact_delta"] is not None]
    if not eligible:
        return {"eligible_mean_exact_delta": None,
                "eligible_mean_predicted_delta": None}
    return {"eligible_mean_exact_delta": np.mean(
                [row["exact_delta"] for row in eligible], axis=0).tolist(),
            "eligible_mean_predicted_delta": np.mean(
                [row["predicted_delta"] for row in eligible], axis=0).tolist()}


def run_revision(task: dict, device="cuda") -> dict:
    """Execute selected-seed revision and save attempts, metrics, and charges."""
    started = perf_counter()
    seed = int(task.get("seed", 142))
    if seed != 142 and not task.get("allow_other_seed", False):
        raise ValueError("initial revision is restricted to selected seed 142")
    attempts = int(task.get("attempts", 16))
    chunk = int(task.get("chunk", 4))
    nfe = int(task.get("nfe", 16))
    T = float(task.get("T", .95))
    if attempts < 1 or chunk < 1:
        raise ValueError("attempts and chunk must be positive")
    grid = revision_grid(nfe, T)
    output = Path(task["output"])
    output.mkdir(parents=True, exist_ok=True)
    model, model_payload = load_model(task["checkpoint"], device)
    model.eval().requires_grad_(False)
    critic, _ = load_critic(task["critic"], model, device)
    fingerprint = tensor_fingerprint(model.state_dict().items())
    dataset = MaskedEditDataset(split=task.get("split", "validation"), kind="vector")
    source_scales = np.asarray(dataset.manifest["descriptor_scales_from_source_molecules"],
                               dtype=np.float32)
    scales = torch.tensor(source_scales, device=device)
    rows_all, metric_rows, calibration_rows = [], [], []
    physical_forward = physical_backward = physical_truth = 0
    physical_baseline_rollouts = 0
    campaign_limit = min(int(task.get("campaign_limit", 16)), len(dataset.campaign_ids))
    iterations = int(task.get("iterations", 4))
    radii = tuple(task.get("radii", [.02, .05]))
    for ci, cid in enumerate(dataset.campaign_ids[:campaign_limit]):
        campaign_started = perf_counter()
        meta = dataset.manifest["campaigns"][cid]
        mask_id = meta["mask_id"]
        mask = dataset.masks[mask_id]
        support = dataset.nested_support(ci, seed=77600 + ci, sizes=(4, 8, 16))
        old_record, new_record = support["records"][4], support["records"][16]
        old_fit = fit_support_response([g["smiles"] for g in old_record["graphs"]],
                                       old_record["values"], mask.lead_smiles,
                                       source_scales)
        new_fit = fit_support_response([g["smiles"] for g in new_record["graphs"]],
                                       new_record["values"], mask.lead_smiles,
                                       source_scales)
        old_W = torch.tensor(old_fit["coefficients"], device=device,
                             dtype=torch.float32)
        new_W = torch.tensor(new_fit["coefficients"], device=device,
                             dtype=torch.float32)
        calibration_rows.append({"campaign_id": cid, "mask_id": mask_id,
                                 "old_support_size": 4, "new_support_size": 16,
                                 "old_support_indices": old_record["indices"].tolist(),
                                 "new_support_indices": new_record["indices"].tolist(),
                                 "old_coefficients": old_fit["coefficients"].tolist(),
                                 "new_coefficients": new_fit["coefficients"].tolist(),
                                 "old_basis_rank": old_fit["basis_rank"],
                                 "new_basis_rank": new_fit["basis_rank"],
                                 "old_condition_number": old_fit["condition_number"],
                                 "new_condition_number": new_fit["condition_number"],
                                 "coefficient_change_frobenius": float(
                                     np.linalg.norm(new_fit["coefficients"]-
                                                    old_fit["coefficients"]))})
        lead_raw = torch.tensor(_descriptor(mask.lead_smiles),
                                device=device, dtype=torch.float32)[None]
        gaussian = make_paired_gaussian(model.config, attempts, 91200 + ci,
                                        device=device)
        records = defaultdict(list)
        charges = defaultdict(lambda: {"forward": 0, "backward": 0,
                                       "physical_seconds": 0.})
        evaluators = defaultdict(ExactMaskedEvaluator)
        for first in range(0, attempts, chunk):
            count = min(chunk, attempts - first)
            condition = condition_for(model, dataset, mask_id, count)
            noise = _slice_noise(gaussian, first, count)
            W4 = old_W[None].expand(count, -1, -1)
            W16 = new_W[None].expand(count, -1, -1)
            raw = lead_raw.expand(count, -1)
            # One physical unguided rollout is cached for every comparison.
            stage_started = perf_counter()
            with torch.no_grad():
                baseline = sample_masked_edit_latents(model, condition, noise,
                                                       nfe=nfe, T=T)
            baseline_rows = _evaluate("unguided_cached", None, model, critic,
                                      baseline, condition, W16, raw, scales,
                                      dataset, cid, support,
                                      evaluators[("unguided_cached", None)],
                                      first=first, seed=seed, physical=True)
            records[("unguided_cached", None)].extend(baseline_rows)
            baseline_seconds = perf_counter() - stage_started
            charges[("unguided_cached", None)]["forward"] += nfe * count
            charges[("unguided_cached", None)]["physical_seconds"] += baseline_seconds
            charges[("unguided_cached", None)]["charged_seconds"] = charges[("unguided_cached", None)].get("charged_seconds", 0.) + baseline_seconds
            physical_forward += nfe * count
            physical_baseline_rollouts += count
            physical_truth += count
            # The old controller is optimized once and reused for all three
            # revision times; its physical rollout work is counted once.
            stage_started = perf_counter()
            old = optimize_terminal_control(
                model, condition, noise, critic, W4, raw, scales, condition.goal,
                nfe=nfe, T=T, iterations=iterations, radii=radii)
            old_rows = _evaluate("old_evidence_control", None, model, critic,
                                 old.sample, condition, W4, raw, scales,
                                 dataset, cid, support,
                                 evaluators[("old_evidence_control", None)],
                                 first=first, seed=seed, physical=True)
            records[("old_evidence_control", None)].extend(old_rows)
            old_seconds = perf_counter() - stage_started
            old_forward = old.forward_nfe * count
            old_backward = old.backward_nfe * count
            charges[("old_evidence_control", None)]["forward"] += old_forward
            charges[("old_evidence_control", None)]["backward"] += old_backward
            charges[("old_evidence_control", None)]["physical_seconds"] += old_seconds
            charges[("old_evidence_control", None)]["charged_seconds"] = charges[("old_evidence_control", None)].get("charged_seconds", 0.) + old_seconds
            physical_forward += old_forward
            physical_backward += old_backward
            physical_truth += count
            # Restart W16 is optimized once from the same Gaussian. Its exact
            # candidates are reused for each revision-time comparator and the
            # all-evidence-from-start arm, without another truth query.
            stage_started = perf_counter()
            restart = optimize_terminal_control(
                model, condition, noise, critic, W16, raw, scales, condition.goal,
                nfe=nfe, T=T, iterations=iterations, radii=radii)
            restart_rows = _evaluate("restart_after_revision", None, model,
                                     critic, restart.sample, condition, W16,
                                     raw, scales, dataset, cid, support,
                                     evaluators[("restart_after_revision", None)],
                                     first=first, seed=seed, physical=True)
            restart_seconds = perf_counter() - stage_started
            restart_forward = restart.forward_nfe * count
            restart_backward = restart.backward_nfe * count
            physical_forward += restart_forward
            physical_backward += restart_backward
            physical_truth += count
            for item in grid:
                target, index, actual = (item["target_time"], item["grid_index"],
                                         item["actual_time"])
                # A discarded prefix is still charged to a restart after the
                # N=16 evidence arrives; all-evidence-from-start avoids it.
                stage_started = perf_counter()
                prefix = replay_prefix_state(model, condition, noise,
                                             old.node_control, old.pair_control,
                                             nfe=nfe, T=T, grid_index=index)
                prefix_seconds = perf_counter() - stage_started
                prefix_forward = index * count
                physical_forward += prefix_forward
                remaining = nfe - index
                stage_started = perf_counter()
                revised = optimize_terminal_control(
                    model, condition, None, critic, W16, raw, scales,
                    condition.goal, nfe=remaining, T=T,
                    start_t=actual, initial_state=prefix,
                    iterations=iterations, radii=radii)
                revised_rows = _evaluate("continued", target, model, critic,
                                         revised.sample, condition, W16, raw,
                                         scales, dataset, cid, support,
                                         evaluators[("continued", target)],
                                         first=first, seed=seed, physical=True)
                revised_seconds = perf_counter() - stage_started
                revised_forward = revised.forward_nfe * count
                revised_backward = revised.backward_nfe * count
                physical_forward += revised_forward
                physical_backward += revised_backward
                physical_truth += count
                for row, energy in zip(revised_rows, revised.control_energy.cpu().tolist()):
                    row.update(old_control_energy=float(old.control_energy[row["sample"]-first]),
                               revised_control_energy=float(energy),
                               actual_revision_time=actual,
                               prefix_forward_nfe_per_sample=index)
                records[("continued", target)].extend(revised_rows)
                charge = charges[("continued", target)]
                charge["forward"] += old_forward + prefix_forward + revised_forward
                charge["backward"] += old_backward + revised_backward
                charge["physical_seconds"] += revised_seconds + prefix_seconds
                charge["charged_seconds"] = charge.get("charged_seconds", 0.) + old_seconds + prefix_seconds + revised_seconds
                charge["old_forward"] = charge.get("old_forward", 0) + old_forward
                charge["old_backward"] = charge.get("old_backward", 0) + old_backward
                charge["prefix_forward"] = charge.get("prefix_forward", 0) + prefix_forward
                charge["revision_forward"] = charge.get("revision_forward", 0) + revised_forward
                charge["revision_backward"] = charge.get("revision_backward", 0) + revised_backward
                charge["actual_revision_time"] = actual
                charges[("restart_after_revision", target)]["forward"] += (
                    old_forward + prefix_forward + restart_forward)
                charges[("restart_after_revision", target)]["backward"] += (
                    old_backward + restart_backward)
                charges[("restart_after_revision", target)]["old_forward"] = charges[("continued", target)]["old_forward"]
                charges[("restart_after_revision", target)]["prefix_forward"] = charges[("continued", target)]["prefix_forward"]
                charges[("restart_after_revision", target)]["restart_forward"] = charges[("restart_after_revision", target)].get("restart_forward", 0) + restart_forward
                charges[("restart_after_revision", target)]["actual_revision_time"] = actual
                charges[("restart_after_revision", target)]["charged_seconds"] = charges[("restart_after_revision", target)].get("charged_seconds", 0.) + old_seconds + prefix_seconds + restart_seconds
                if target == grid[0]["target_time"]:
                    charges[("restart_after_revision", target)]["physical_seconds"] += restart_seconds
                charges[("all_evidence_from_start", target)]["forward"] += restart_forward
                charges[("all_evidence_from_start", target)]["backward"] += restart_backward
                charges[("all_evidence_from_start", target)]["actual_revision_time"] = actual
                charges[("all_evidence_from_start", target)]["charged_seconds"] = charges[("all_evidence_from_start", target)].get("charged_seconds", 0.) + restart_seconds
                records[("restart_after_revision", target)].extend(
                    _copy_rows(restart_rows, "restart_after_revision", target,
                               physical=target == grid[0]["target_time"],
                               source="restart_W16_once"))
                records[("all_evidence_from_start", target)].extend(
                    _copy_rows(restart_rows, "all_evidence_from_start", target,
                               physical=False, source="restart_W16_once"))
        for (method, target), rows in records.items():
            charge = charges[(method, target)]
            metrics = metrics_for(rows, k=16)
            metric_rows.append({"campaign_id": cid, "mask_id": mask_id,
                                "method": method, "revision_target": target,
                                "seed": seed, "split": dataset.split,
                                "old_support_size": 4, "new_support_size": 16,
                                "charged_forward_sample_nfe": charge["forward"],
                                "charged_backward_sample_nfe": charge["backward"],
                                "old_optimization_forward_nfe": charge.get("old_forward", 0),
                                "old_optimization_backward_nfe": charge.get("old_backward", 0),
                                "prefix_forward_nfe": charge.get("prefix_forward", 0),
                                "revision_forward_nfe": charge.get("revision_forward", 0),
                                "revision_backward_nfe": charge.get("revision_backward", 0),
                                "restart_forward_nfe": charge.get("restart_forward", 0),
                                "actual_revision_time": charge.get("actual_revision_time"),
                                "physical_method_seconds_including_chemistry": charge["physical_seconds"],
                                "charged_seconds_including_chemistry": charge.get("charged_seconds", 0.),
                                "compute_comparison": "exact_charges_compute_pareto",
                                **metrics, **_response_means(rows)})
            rows_all.extend(rows)
        write_csv(output / "metrics.csv", metric_rows)
        write_csv(output / "attempts.csv", rows_all)
        write_csv(output / "calibration.csv", calibration_rows)
        print(json.dumps({"campaign_id": cid, "seconds": perf_counter()-campaign_started,
                          "methods": len(records), "physical_forward_nfe": physical_forward,
                          "physical_backward_nfe": physical_backward}), flush=True)
    if tensor_fingerprint(model.state_dict().items()) != fingerprint:
        raise RuntimeError("frozen generator changed during revision")
    summary = {"phase": "complete", "stage": "revision", "seed": seed,
               "variant": model_payload["variant"], "split": dataset.split,
               "campaigns": campaign_limit, "attempts_per_campaign": attempts,
               "revision_grid": list(grid), "support_sizes": [4, 16],
               "physical_forward_sample_nfe": physical_forward,
               "physical_backward_sample_nfe": physical_backward,
               "physical_terminal_proposals": physical_truth,
               "physical_truth_evaluations": physical_truth,
               "physical_cached_baseline_rollouts": physical_baseline_rollouts,
               "logical_attempt_rows": len(rows_all),
               "seconds_including_chemistry": perf_counter()-started,
               "completed_utc": now(), "task": task,
               "comparison_scope": "Compute Pareto with exact charged NFE. Old control is optimized once per Gaussian, prefixes are replayed and charged, restart W16 is optimized once and logically reused across revision times. All-evidence-from-start has the same terminal candidates as restart without old or discarded-prefix costs. Exact simulator truth is queried only after each unique graph is committed."}
    atomic_json(output / "summary.json", summary)
    return summary
