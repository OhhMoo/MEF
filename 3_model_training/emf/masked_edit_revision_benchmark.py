"""Costed preference and evidence revision on the hard-masked B generator.

The response observations and oracle are immutable. Preference reversal changes
the requested direction of endpoint one; evidence revision adds nested noisy
measurements of the same response. Every generated graph is committed before
exact scoring, and invalid attempts remain in the output ledger.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from time import perf_counter
import json

import numpy as np
import torch

from .contracts import Latents
from .masked_edit_control import optimize_terminal_control
from .masked_edit_data import (MaskedEditDataset, _descriptor, _features,
                               fit_support_response)
from .masked_edit_baselines import _direct_mutations
from .masked_edit_evaluation import ExactMaskedEvaluator
from .masked_edit_experiments import load_critic, score_prediction
from .masked_edit_revision import replay_prefix_state, revision_grid
from .masked_edit_run import condition_for, graphs_from_logits, load_model, now, write_csv
from .masked_edit_sampling import make_paired_gaussian, sample_masked_edit_latents
from .redesign_train import atomic_json, tensor_fingerprint


GOAL = (.25, -.1)
SCENARIOS = ("goal_reversal", "evidence_revision")


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _begin(device):
    _sync(device)
    return perf_counter()


def _elapsed(start, device):
    _sync(device)
    return perf_counter() - start


def _slice(latents: Latents, first: int, count: int) -> Latents:
    return Latents(latents.nodes[first:first + count], latents.pairs[first:first + count])


def _condition(base, coefficients: torch.Tensor, goal=GOAL):
    """Preserve the one cached lead encoding across compared conditions."""
    size = base.goal.shape[0]
    if coefficients.shape != (size, 2, 3):
        raise ValueError("response coefficients must have shape [batch,2,3]")
    state = base.response_state.clone()
    state[:, :6] = coefficients.reshape(size, 6)
    return replace(base, response_state=state,
                   goal=torch.as_tensor(goal, device=state.device, dtype=state.dtype)
                   .expand(size, -1))


def _orient(coefficients: torch.Tensor, sign: tuple[int, int]):
    if len(sign) != 2 or sign[0] not in (-1, 1) or sign[1] != 1:
        raise ValueError("endpoint one sign must be ±1; endpoint two is retained")
    return coefficients * torch.tensor(sign, device=coefficients.device,
                                       dtype=coefficients.dtype)[None, :, None]


def _rank(predicted, goal=GOAL):
    gap = np.maximum(np.asarray(goal, dtype=np.float64) - np.asarray(predicted), 0)
    return float(-(gap @ gap) + 1e-5 * predicted[0])


def _score_rows(result, model, critic, condition, coefficients, raw, scales,
                dataset, cid, support, evaluator, *, scenario, method, target,
                first, sign, seed, split, cost, physical=True, reused_from=None):
    """Decode, commit, then query truth; apply the scenario goal after commitment."""
    logits, predicted = score_prediction(model, critic, result, condition,
                                         coefficients, raw, scales)
    graphs = graphs_from_logits(logits)
    mask = dataset.masks[dataset.manifest["campaigns"][cid]["mask_id"]]
    source = [dataset.smiles[int(index)] for index in dataset.source_graph_indices]
    rows = []
    chemistry_started = _begin(predicted.device)
    for offset, graph in enumerate(graphs):
        row = evaluator.evaluate_graph(graph, mask, cid,
                                       support_smiles=support["reserve_smiles"],
                                       source_smiles=source,
                                       known_library_smiles=dataset.smiles)
        physical_delta = row["exact_delta"]
        oriented = (None if physical_delta is None else
                    (np.asarray(physical_delta) * np.asarray(sign)).tolist())
        eligible = bool(row["chemically_valid"] and row["connected"]
                        and row["protected_copy"] and not row["lead_copy"]
                        and not row["support_match"])
        preservation = bool(eligible and oriented is not None and oriented[1] >= GOAL[1])
        success = bool(preservation and oriented[0] >= GOAL[0]) if oriented is not None else False
        old_success = bool(preservation and physical_delta is not None
                           and physical_delta[0] >= GOAL[0])
        row.update(scenario=scenario, method=method, revision_target=target,
                   actual_revision_time=cost.get("actual_revision_time"),
                   sample=first + offset, seed=seed, split=split,
                   goal_sign=list(sign), requested_goal=list(GOAL),
                   physical_exact_delta=physical_delta, oriented_exact_delta=oriented,
                   predicted_oriented_delta=predicted[offset].detach().cpu().tolist(),
                   predicted_physical_delta=(predicted[offset].detach().cpu().numpy()
                                             * np.asarray(sign)).tolist(),
                   endpoint2_preserved=preservation, old_goal_success=old_success,
                   new_goal_success=success, goal_success=success,
                   support_excluded_success=success,
                   original_locality_goal_success=bool(success and row["original_locality"]),
                   eligible=eligible, eligible_novel_success=bool(success and not row["source_match"]),
                   rank_score=_rank(predicted[offset].detach().cpu().numpy()),
                   physical_truth_evaluation=physical, reused_from=reused_from,
                   **cost)
        rows.append(row)
    chemistry_per_attempt = _elapsed(chemistry_started, predicted.device) / len(rows)
    for row in rows:
        row["host_chemistry_seconds_per_attempt"] = chemistry_per_attempt
    return rows


def _case_metrics(rows, topk: int):
    ranked = sorted((r for r in rows if r["eligible"]),
                    key=lambda r: (-r["rank_score"], r["sample"]))
    unique = {}
    for row in ranked:
        unique.setdefault(row["smiles"], row)
    selected = list(unique.values())[:topk]
    return {"attempts": len(rows),
            "valid_connected_attempts": sum(bool(r["chemically_valid"] and r["connected"]) for r in rows),
            "eligible_attempts": sum(r["eligible"] for r in rows),
            "distinct_eligible": len(unique),
            "distinct_goal_successes": len({r["smiles"] for r in rows if r["goal_success"]}),
            "success_attempts": sum(r["goal_success"] for r in rows),
            "endpoint2_preserved_attempts": sum(r["endpoint2_preserved"] for r in rows),
            "old_goal_success_attempts": sum(r["old_goal_success"] for r in rows),
            "topk": topk, "topk_filled": len(selected),
            "topk_successes": sum(r["goal_success"] for r in selected),
            "topk_success_fraction": sum(r["goal_success"] for r in selected) / topk}


def _cost(*, count, forward=0, backward=0, decode=0, critic=0,
          decode_backward=0, critic_backward=0,
          seconds=0., old_seconds=0., prefix_seconds=0., update_fit_seconds=0.,
          old_forward=0, old_backward=0, prefix_forward=0,
          final_forward=0, final_backward=0, actual_revision_time=None):
    if count < 1:
        raise ValueError("empty cost batch")
    return {"charged_forward_nfe": forward // count,
            "charged_backward_nfe": backward // count,
            "charged_decode_calls": decode // count,
            "charged_critic_calls": critic // count,
            "charged_decode_backward_calls": decode_backward // count,
            "charged_critic_backward_calls": critic_backward // count,
            "charged_seconds_per_attempt": seconds / count,
            "incremental_forward_nfe": final_forward // count,
            "incremental_backward_nfe": final_backward // count,
            "incremental_seconds_per_attempt": (seconds-old_seconds-prefix_seconds) / count,
            "old_seconds_per_attempt": old_seconds / count,
            "prefix_seconds_per_attempt": prefix_seconds / count,
            "update_fit_seconds_per_attempt": update_fit_seconds / count,
            "old_forward_nfe": old_forward // count,
            "old_backward_nfe": old_backward // count,
            "prefix_forward_nfe": prefix_forward // count,
            "final_forward_nfe": final_forward // count,
            "final_backward_nfe": final_backward // count,
            "actual_revision_time": actual_revision_time}


def _control_charge(result, nfe, count):
    if result.forward_nfe % nfe or result.backward_nfe % nfe:
        raise RuntimeError("controller evaluation count is inconsistent")
    return {"forward": result.forward_nfe * count,
            "backward": result.backward_nfe * count,
            "decode": (result.forward_nfe // nfe) * count,
            "critic": (result.forward_nfe // nfe) * count,
            "decode_backward": (result.backward_nfe // nfe) * count,
            "critic_backward": (result.backward_nfe // nfe) * count}


def _combine_cost(count, *parts, actual_revision_time=None):
    fields = ("forward", "backward", "decode", "critic",
              "decode_backward", "critic_backward", "seconds",
              "old_seconds", "prefix_seconds", "update_fit_seconds",
              "old_forward", "old_backward", "prefix_forward",
              "final_forward", "final_backward")
    sums = {field: sum(part.get(field, 0) for part in parts) for field in fields}
    return _cost(count=count, actual_revision_time=actual_revision_time,
                 **sums)


def _append(rows, case, ledger, *, topk, device):
    if rows:
        first = min(row["sample"] for row in rows)
        for row in rows:
            row["batch_first"] = first
            row["batch_count"] = len(rows)
        started = _begin(device)
        _case_metrics(ledger[case] + rows, topk)
        ranking_seconds = _elapsed(started, device) / len(rows)
        for row in rows:
            row["charged_seconds_per_attempt"] += ranking_seconds
            row["incremental_seconds_per_attempt"] += ranking_seconds
            row["ranking_seconds_per_attempt"] = ranking_seconds
    ledger[case].extend(rows)


def _budget_rows(ledger, budgets, topk, *, basis="total"):
    """Predeclared wall budgets; include only completed, indivisible batches."""
    if basis not in {"total", "incremental"}:
        raise ValueError("budget basis must be total or incremental")
    cost_field = ("charged_seconds_per_attempt" if basis == "total"
                  else "incremental_seconds_per_attempt")

    def cost(row):
        return row.get(cost_field, row["charged_seconds_per_attempt"])

    output = []
    for key, rows in ledger.items():
        ordered = sorted(rows, key=lambda row: row["sample"])
        if not ordered:
            continue
        groups = defaultdict(list)
        for row in ordered:
            groups[row["batch_first"]].append(row)
        batches = [groups[first] for first in sorted(groups)]
        first_cost = sum(cost(row) for row in batches[0])
        full_cost = sum(sum(cost(row) for row in batch)
                        for batch in batches)
        for budget in budgets:
            elapsed, incremental_elapsed, prefix = 0., 0., []
            for batch in batches:
                next_elapsed = elapsed + sum(cost(row) for row in batch)
                if next_elapsed > budget + 1e-12:
                    break
                elapsed = next_elapsed
                incremental_elapsed += sum(row.get("incremental_seconds_per_attempt",
                                                   row["charged_seconds_per_attempt"]) for row in batch)
                prefix.extend(batch)
            output.append(dict(campaign_id=key[0], scenario=key[1],
                               method=key[2], revision_target=key[3],
                               budget_basis=basis,
                               wall_budget_seconds=budget,
                               charged_seconds_used=elapsed,
                               total_charged_seconds=sum(row["charged_seconds_per_attempt"] for row in prefix),
                               incremental_charged_seconds=incremental_elapsed,
                               total_charged_forward_nfe=sum(row.get("charged_forward_nfe", 0) for row in prefix),
                               total_charged_backward_nfe=sum(row.get("charged_backward_nfe", 0) for row in prefix),
                               charged_decode_calls=sum(row.get("charged_decode_calls", 0) for row in prefix),
                               charged_critic_calls=sum(row.get("charged_critic_calls", 0) for row in prefix),
                               charged_decode_backward_calls=sum(row.get("charged_decode_backward_calls", 0) for row in prefix),
                               charged_critic_backward_calls=sum(row.get("charged_critic_backward_calls", 0) for row in prefix),
                               host_chemistry_seconds=sum(row.get("host_chemistry_seconds_per_attempt", 0.) for row in prefix),
                               incremental_forward_nfe=sum(row.get("incremental_forward_nfe", 0) for row in prefix),
                               incremental_backward_nfe=sum(row.get("incremental_backward_nfe", 0) for row in prefix),
                               unused_budget_seconds=budget-elapsed,
                               cap_limited=bool(len(prefix)==len(rows) and full_cost < budget),
                               first_batch_seconds=first_cost,
                               **_case_metrics(prefix, topk)))
    return output


def _batch_rows(ledger):
    batches = []
    for (cid, scenario, method, target), rows in ledger.items():
        groups = defaultdict(list)
        for row in rows:
            groups[row["batch_first"]].append(row)
        cumulative_seconds = cumulative_incremental = cumulative_forward = cumulative_backward = 0
        for first in sorted(groups):
            batch = groups[first]
            seconds = sum(row["charged_seconds_per_attempt"] for row in batch)
            forward = sum(row["charged_forward_nfe"] for row in batch)
            backward = sum(row["charged_backward_nfe"] for row in batch)
            incremental = sum(row["incremental_seconds_per_attempt"] for row in batch)
            cumulative_seconds += seconds
            cumulative_incremental += incremental
            cumulative_forward += forward
            cumulative_backward += backward
            batches.append(dict(campaign_id=cid, scenario=scenario, method=method,
                                revision_target=target, batch_first=first,
                                batch_count=len(batch), charged_batch_seconds=seconds,
                                incremental_batch_seconds=incremental,
                                cumulative_charged_seconds=cumulative_seconds,
                                cumulative_incremental_seconds=cumulative_incremental,
                                total_charged_seconds=seconds,
                                incremental_charged_seconds=incremental,
                                charged_forward_nfe=forward,
                                cumulative_charged_forward_nfe=cumulative_forward,
                                charged_backward_nfe=backward,
                                cumulative_charged_backward_nfe=cumulative_backward,
                                incremental_forward_nfe=sum(row["incremental_forward_nfe"] for row in batch),
                                incremental_backward_nfe=sum(row["incremental_backward_nfe"] for row in batch),
                                charged_decode_calls=sum(row["charged_decode_calls"] for row in batch),
                                charged_critic_calls=sum(row["charged_critic_calls"] for row in batch),
                                charged_decode_backward_calls=sum(row["charged_decode_backward_calls"] for row in batch),
                                charged_critic_backward_calls=sum(row["charged_critic_backward_calls"] for row in batch),
                                host_chemistry_seconds=sum(row.get("host_chemistry_seconds_per_attempt", 0.) for row in batch),
                                physical_evaluation_requests=sum(row["physical_truth_evaluation"] for row in batch),
                                physical_exact_response_calls=sum(bool(row["physical_truth_evaluation"]
                                    and row["physical_exact_delta"] is not None) for row in batch),
                                reused_from=batch[0].get("reused_from")))
    return sorted(batches, key=lambda item: (item["campaign_id"], item["scenario"],
                                             item["method"], str(item["revision_target"]),
                                             item["batch_first"]))


def run(task: dict, device="cuda") -> dict:
    """Run the fixed B revision comparison and write raw and per-case ledgers."""
    required = ("checkpoint", "critic", "output", "seed")
    missing = [field for field in required if field not in task]
    if missing:
        raise ValueError(f"missing task fields: {missing}")
    attempts = int(task.get("attempts", 64))
    baseline_attempts = int(task.get("baseline_attempts", 1600))
    conditional_attempts = int(task.get("conditional_attempts", baseline_attempts))
    chunk = int(task.get("chunk", 8))
    nfe = int(task.get("nfe", 16))
    T = float(task.get("T", .95))
    iterations = int(task.get("iterations", 4))
    radii = tuple(task.get("radii", (.02, .05)))
    topk = int(task.get("topk", 16))
    if (min(attempts, baseline_attempts, conditional_attempts, chunk, topk) < 1
            or baseline_attempts < attempts or conditional_attempts < attempts):
        raise ValueError("positive caps/chunk/topk and baseline/conditional caps>=attempts required")
    if task.get("split", "test") == "test" and task.get("wall_budgets_seconds") is None:
        raise ValueError("test budget analysis requires fixed validation-selected wall_budgets_seconds")
    budgets = tuple(float(value) for value in task.get("wall_budgets_seconds", ()))
    if any(not np.isfinite(value) or value <= 0 for value in budgets):
        raise ValueError("wall budgets must be finite and positive")
    grid = revision_grid(nfe, T, targets=tuple(task.get("revision_times", (.5,))))
    seed = int(task["seed"])
    output = Path(task["output"])
    output.mkdir(parents=True, exist_ok=True)
    run_started = _begin(device)
    model, payload = load_model(task["checkpoint"], device)
    if model.variant != "B":
        raise ValueError("revision benchmark requires hard-masked B checkpoint")
    model.eval().requires_grad_(False)
    critic, _ = load_critic(task["critic"], model, device)
    conditional = None
    if task.get("conditional_checkpoint"):
        conditional, conditional_payload = load_model(task["conditional_checkpoint"], device)
        if conditional.variant != "B" or not conditional_payload.get("conditioned", False):
            raise ValueError("conditional checkpoint must be task-conditioned B")
        conditional.eval().requires_grad_(False)
    fingerprints = [tensor_fingerprint(item.state_dict().items()) for item in
                    (model, *((conditional,) if conditional is not None else ()))]
    split = task.get("split", "test")
    dataset = MaskedEditDataset(split=split, kind="vector")
    evaluator = ExactMaskedEvaluator()
    case_evaluators = defaultdict(ExactMaskedEvaluator)
    scales = torch.tensor(evaluator.scales, device=device, dtype=torch.float32)
    ledger = defaultdict(list)
    calibration = []
    old_stage_rows = []
    feasibility_rows = []
    feasibility_seconds = 0.
    physical = defaultdict(float)
    limit = min(int(task.get("campaign_limit", len(dataset.campaign_ids))),
                len(dataset.campaign_ids))
    for ci, cid in enumerate(dataset.campaign_ids[:limit]):
        mask_id = dataset.manifest["campaigns"][cid]["mask_id"]
        mask = dataset.masks[mask_id]
        start = _begin(device)
        support = dataset.nested_support(ci, seed=77600 + ci, sizes=(4, 16))
        fits = {}
        fit_seconds = {}
        for size in (4, 16):
            fit_started = _begin(device)
            record = support["records"][size]
            fit = fit_support_response([graph["smiles"] for graph in record["graphs"]],
                                       record["values"], mask.lead_smiles,
                                       evaluator.scales)
            fit_seconds[size] = _elapsed(fit_started, device)
            fits[size] = torch.tensor(fit["coefficients"], device=device,
                                      dtype=torch.float32)
            calibration.append(dict(campaign_id=cid, support_size=size,
                                    support_indices=record["indices"].tolist(),
                                    coefficients=fit["coefficients"].tolist(),
                                    fit_seconds=fit_seconds[size],
                                    basis_rank=fit["basis_rank"],
                                    condition_number=fit["condition_number"]))
        calibration[-1]["coefficient_change_frobenius"] = float(
            torch.linalg.vector_norm(fits[16] - fits[4]).cpu())
        noise_started = _begin(device)
        noise = make_paired_gaussian(model.config, max(baseline_attempts, conditional_attempts),
                                     seed * 100000 + ci, device=device)
        noise_seconds = _elapsed(noise_started, device)
        setup_seconds = _elapsed(start, device)
        common_setup_seconds = setup_seconds - noise_seconds
        raw = torch.tensor(_descriptor(mask.lead_smiles), device=device,
                           dtype=torch.float32)[None]
        for scenario in SCENARIOS:
            old_size, new_size = ((16, 16) if scenario == "goal_reversal" else (4, 16))
            old_sign = (1, 1)
            new_sign = ((-1, 1) if scenario == "goal_reversal" else (1, 1))
            for first in range(0, attempts, chunk):
                count = min(chunk, attempts - first)
                prep_started = _begin(device)
                lead = condition_for(model, dataset, mask_id, count)
                old_w = _orient(fits[old_size][None].expand(count, -1, -1), old_sign)
                new_w = _orient(fits[new_size][None].expand(count, -1, -1), new_sign)
                old_cond, new_cond = _condition(lead, old_w), _condition(lead, new_w)
                draw = _slice(noise, first, count)
                raw_batch = raw.expand(count, -1)
                prep = _elapsed(prep_started, device) + (
                    common_setup_seconds + noise_seconds
                    if first==0 else 0.)
                update_fit_seconds = (fit_seconds[16] if
                                      scenario == "evidence_revision" and first == 0 else 0.)

                started = _begin(device)
                old = optimize_terminal_control(model, old_cond, draw, critic,
                    old_w, raw_batch, scales, old_cond.goal, nfe=nfe, T=T,
                    iterations=iterations, radii=radii)
                old_charge = _control_charge(old, nfe, count)
                old_charge.update(seconds=_elapsed(started, device) + prep,
                                  old_forward=old_charge["forward"],
                                  old_backward=old_charge["backward"])
                old_eval_started = _begin(device)
                old_rows = _score_rows(old.sample, model, critic, old_cond,
                    old_w, raw_batch, scales, dataset, cid, support,
                    case_evaluators[(cid, scenario, "old_control", None)],
                    scenario=scenario, method="old_control", target=None,
                    first=first, sign=old_sign, seed=seed, split=split,
                    cost={})
                old_charge["seconds"] += _elapsed(old_eval_started, device)
                old_charge["decode"] += count
                old_charge["critic"] += count
                old_charge["old_seconds"] = old_charge["seconds"] - update_fit_seconds
                old_charge["update_fit_seconds"] = update_fit_seconds
                old_cost = _combine_cost(count, old_charge)
                for row in old_rows:
                    row.update(old_cost)
                old_stage_rows.extend(old_rows)
                physical["forward_nfe"] += old_charge["forward"]
                physical["backward_nfe"] += old_charge["backward"]
                physical["truth_evaluations"] += count

                started = _begin(device)
                fresh = optimize_terminal_control(model, new_cond, draw, critic,
                    new_w, raw_batch, scales, new_cond.goal, nfe=nfe, T=T,
                    iterations=iterations, radii=radii)
                fresh_charge = _control_charge(fresh, nfe, count)
                fresh_charge.update(seconds=_elapsed(started, device),
                                    final_forward=fresh_charge["forward"],
                                    final_backward=fresh_charge["backward"])
                physical["forward_nfe"] += fresh_charge["forward"]
                physical["backward_nfe"] += fresh_charge["backward"]
                eval_start = _begin(device)
                fresh_rows = _score_rows(fresh.sample, model, critic, new_cond,
                    new_w, raw_batch, scales, dataset, cid, support,
                    case_evaluators[(cid, scenario, "all_evidence", None)], scenario=scenario,
                    method="all_evidence", target=None, first=first,
                    sign=new_sign, seed=seed, split=split, cost={})
                eval_seconds = _elapsed(eval_start, device)
                physical["truth_evaluations"] += count
                final_eval = dict(decode=count, critic=count,
                                  seconds=eval_seconds)
                cost_all = _combine_cost(count, fresh_charge, final_eval,
                    dict(seconds=prep))
                for row in fresh_rows:
                    previous = old_rows[row["sample"]-first]
                    row.update(cost_all, old_smiles=previous["smiles"],
                               changed_from_old=previous["smiles"] != row["smiles"])
                _append(fresh_rows, (cid, scenario, "all_evidence", None), ledger,
                        topk=topk, device=device)

                for grid_item in grid:
                    target, index, actual = (grid_item["target_time"],
                                             grid_item["grid_index"], grid_item["actual_time"])
                    started = _begin(device)
                    prefix = replay_prefix_state(model, old_cond, draw,
                        old.node_control, old.pair_control, nfe=nfe, T=T,
                        grid_index=index)
                    prefix_charge = dict(forward=index * count,
                                         prefix_forward=index * count,
                                         seconds=_elapsed(started, device))
                    prefix_charge["prefix_seconds"] = prefix_charge["seconds"]
                    physical["forward_nfe"] += index * count
                    started = _begin(device)
                    revised = optimize_terminal_control(model, new_cond, None,
                        critic, new_w, raw_batch, scales, new_cond.goal,
                        nfe=nfe-index, T=T, start_t=actual,
                        initial_state=prefix, iterations=iterations,
                        radii=radii)
                    revised_charge = _control_charge(revised, nfe-index, count)
                    revised_charge.update(seconds=_elapsed(started, device),
                                          final_forward=revised_charge["forward"],
                                          final_backward=revised_charge["backward"])
                    physical["forward_nfe"] += revised_charge["forward"]
                    physical["backward_nfe"] += revised_charge["backward"]
                    eval_start = _begin(device)
                    revised_rows = _score_rows(revised.sample, model, critic,
                        new_cond, new_w, raw_batch, scales, dataset, cid,
                        support, case_evaluators[(cid, scenario, "continue_control", target)], scenario=scenario,
                        method="continue_control", target=target, first=first,
                        sign=new_sign, seed=seed, split=split, cost={})
                    eval_charge = dict(decode=count, critic=count,
                                       seconds=_elapsed(eval_start, device))
                    physical["truth_evaluations"] += count
                    cost_continue = _combine_cost(count, old_charge, prefix_charge,
                                                  revised_charge, eval_charge,
                                                  actual_revision_time=actual)
                    for row in revised_rows:
                        previous = old_rows[row["sample"]-first]
                        row.update(cost_continue,
                                   old_control_energy=float(old.control_energy[row["sample"]-first]),
                                   revised_control_energy=float(revised.control_energy[row["sample"]-first]),
                                   old_smiles=previous["smiles"],
                                   changed_from_old=previous["smiles"] != row["smiles"])
                    _append(revised_rows, (cid, scenario, "continue_control", target), ledger,
                            topk=topk, device=device)
                    cost_restart = _combine_cost(count, old_charge, prefix_charge,
                                                 fresh_charge, final_eval,
                                                 actual_revision_time=actual)
                    restart_rows = [{**row, **cost_restart,
                                     "method": "restart_control",
                                     "revision_target": target,
                                     "physical_truth_evaluation": False,
                                     "reused_from": "all_evidence"}
                                    for row in fresh_rows]
                    _append(restart_rows, (cid, scenario, "restart_control", target), ledger,
                            topk=topk, device=device)

            if conditional is not None:
                for first in range(0, conditional_attempts, chunk):
                    count = min(chunk, conditional_attempts - first)
                    old_w = _orient(fits[old_size][None].expand(count, -1, -1), old_sign)
                    new_w = _orient(fits[new_size][None].expand(count, -1, -1), new_sign)
                    draw = _slice(noise, first, count)
                    raw_batch = raw.expand(count, -1)
                    # The conditional model sees the same lead, mask and Gaussian.
                    c_prep_started = _begin(device)
                    c_lead = condition_for(conditional, dataset, mask_id, count)
                    c_old = _condition(c_lead, old_w)
                    c_new = _condition(c_lead, new_w)
                    c_prep = _elapsed(c_prep_started, device) + (
                        common_setup_seconds + noise_seconds
                        if first==0 else 0.)
                    c_update_fit = (fit_seconds[16] if
                                    scenario == "evidence_revision" and first == 0 else 0.)
                    started = _begin(device)
                    with torch.no_grad():
                        c_fresh = sample_masked_edit_latents(conditional, c_new,
                                                             draw, nfe=nfe, T=T)
                    c_fresh_charge = dict(forward=nfe*count,
                                          final_forward=nfe*count,
                                          seconds=_elapsed(started, device))
                    physical["forward_nfe"] += nfe*count
                    eval_start = _begin(device)
                    c_rows = _score_rows(c_fresh, conditional, critic, c_new,
                        new_w, raw_batch, scales, dataset, cid, support,
                        case_evaluators[(cid, scenario, "conditional_all_evidence", None)], scenario=scenario,
                        method="conditional_all_evidence", target=None,
                        first=first, sign=new_sign, seed=seed, split=split,
                        cost={})
                    c_eval = dict(decode=count, critic=count,
                                  seconds=_elapsed(eval_start, device))
                    physical["truth_evaluations"] += count
                    c_all_cost = _combine_cost(count, c_fresh_charge, c_eval,
                                               dict(seconds=c_prep))
                    for row in c_rows:
                        row.update(c_all_cost)
                    _append(c_rows, (cid, scenario, "conditional_all_evidence", None), ledger,
                            topk=topk, device=device)
                    for grid_item in grid:
                        target, index, actual = (grid_item["target_time"],
                                                 grid_item["grid_index"],
                                                 grid_item["actual_time"])
                        zeros_nodes = torch.zeros(count, 2, *c_lead.lead.nodes.shape[1:], device=device)
                        zeros_pairs = torch.zeros(count, 2, *c_lead.lead.pairs.shape[1:], device=device)
                        started = _begin(device)
                        with torch.no_grad():
                            c_prefix = replay_prefix_state(conditional, c_old,
                                draw, zeros_nodes, zeros_pairs, nfe=nfe, T=T,
                                grid_index=index)
                        c_prefix_charge = dict(forward=index*count,
                                               prefix_forward=index*count,
                                               seconds=_elapsed(started, device) + c_prep)
                        c_prefix_charge["prefix_seconds"] = c_prefix_charge["seconds"] - c_update_fit
                        c_prefix_charge["update_fit_seconds"] = c_update_fit
                        physical["forward_nfe"] += index*count
                        started = _begin(device)
                        with torch.no_grad():
                            c_cont = sample_masked_edit_latents(conditional,
                                c_new, nfe=nfe-index, T=T, start_t=actual,
                                initial_state=c_prefix)
                        c_cont_charge = dict(forward=(nfe-index)*count,
                                             final_forward=(nfe-index)*count,
                                             seconds=_elapsed(started, device))
                        physical["forward_nfe"] += (nfe-index)*count
                        eval_start = _begin(device)
                        c_cont_rows = _score_rows(c_cont, conditional, critic,
                            c_new, new_w, raw_batch, scales, dataset, cid,
                            support, case_evaluators[(cid, scenario, "conditional_continue", target)], scenario=scenario,
                            method="conditional_continue", target=target,
                            first=first, sign=new_sign, seed=seed, split=split,
                            cost={})
                        c_cont_eval = dict(decode=count, critic=count,
                                           seconds=_elapsed(eval_start, device))
                        physical["truth_evaluations"] += count
                        c_cont_cost = _combine_cost(count, c_prefix_charge,
                            c_cont_charge, c_cont_eval,
                            actual_revision_time=actual)
                        for row in c_cont_rows:
                            row.update(c_cont_cost)
                        _append(c_cont_rows,
                                (cid, scenario, "conditional_continue", target), ledger,
                                topk=topk, device=device)
                        c_restart_cost = _combine_cost(count, c_prefix_charge,
                            c_fresh_charge, c_eval,
                            actual_revision_time=actual)
                        c_restart_rows = [{**row, **c_restart_cost,
                                           "method": "conditional_restart",
                                           "revision_target": target,
                                           "physical_truth_evaluation": False,
                                           "reused_from": "conditional_all_evidence"}
                                          for row in c_rows]
                        _append(c_restart_rows,
                                (cid, scenario, "conditional_restart", target), ledger,
                                topk=topk, device=device)

            # Fresh, paired noise is used for each unguided batch. Evidence
            # changes only surrogate ranking; the generated structures are fixed.
            for first in range(0, baseline_attempts, chunk):
                count = min(chunk, baseline_attempts - first)
                baseline_prep_started = _begin(device)
                base = condition_for(model, dataset, mask_id, count)
                new_w = _orient(fits[new_size][None].expand(count, -1, -1), new_sign)
                condition = _condition(base, new_w)
                draw = _slice(noise, first, count)
                baseline_prep = _elapsed(baseline_prep_started, device) + (
                    common_setup_seconds + noise_seconds
                    if first==0 else 0.)
                started = _begin(device)
                with torch.no_grad():
                    sample = sample_masked_edit_latents(model, condition,
                                                        draw, nfe=nfe, T=T)
                sample_seconds = _elapsed(started, device)
                eval_start = _begin(device)
                baseline_rows = _score_rows(sample, model, critic, condition,
                    new_w, raw.expand(count, -1), scales, dataset, cid,
                    support, case_evaluators[(cid, scenario, "unguided_rerank", None)], scenario=scenario,
                    method="unguided_rerank", target=None, first=first,
                    sign=new_sign, seed=seed, split=split, cost={})
                baseline_cost = _combine_cost(count,
                    dict(forward=nfe*count, final_forward=nfe*count,
                         seconds=sample_seconds + baseline_prep),
                    dict(decode=count, critic=count,
                         seconds=_elapsed(eval_start, device)))
                for row in baseline_rows:
                    row.update(baseline_cost)
                _append(baseline_rows, (cid, scenario, "unguided_rerank", None), ledger,
                        topk=topk, device=device)
                physical["forward_nfe"] += nfe*count
                physical["truth_evaluations"] += count

            # Structural mutation uses the same lead mask, N16 reserve, final
            # goal and top-k rule. Its ranking accesses exact RDKit descriptors.
            mutation_started = _begin(device)
            mutation_cap = int(task.get("mutation_attempts", 256))
            if mutation_cap < 1:
                raise ValueError("mutation_attempts must be positive")
            mutations = list(_direct_mutations(mask, mutation_cap,
                                               seed*100000+ci))
            enumeration_seconds = _elapsed(mutation_started, device)
            mutation_setup = enumeration_seconds + common_setup_seconds
            mut_evaluator = case_evaluators[(cid, scenario, "direct_mutation", None)]
            source_set = {dataset.smiles[int(index)] for index in dataset.source_graph_indices}
            lead_descriptor = _descriptor(mask.lead_smiles)
            physical_w = fits[new_size].detach().cpu().numpy()
            for first in range(0, len(mutations), chunk):
                batch = mutations[first:first + chunk]
                started = _begin(device)
                mutation_rows = []
                for offset, graph in enumerate(batch):
                    row = mut_evaluator.evaluate_graph(graph, mask, cid,
                        support_smiles=support["reserve_smiles"],
                        source_smiles=source_set,
                        known_library_smiles=dataset.smiles)
                    physical_delta = row["exact_delta"]
                    oriented = (None if physical_delta is None else
                                (np.asarray(physical_delta) * np.asarray(new_sign)).tolist())
                    eligible = bool(row["chemically_valid"] and row["connected"]
                                    and row["protected_copy"] and not row["lead_copy"]
                                    and not row["support_match"])
                    preservation = bool(eligible and oriented is not None
                                        and oriented[1] >= GOAL[1])
                    success = bool(preservation and oriented[0] >= GOAL[0]) if oriented is not None else False
                    old_success = bool(preservation and physical_delta is not None
                                       and physical_delta[0] >= GOAL[0])
                    if row["smiles"] is None:
                        estimated = None
                        physical_estimated = None
                        rank = -1e30
                    else:
                        features = _features(_descriptor(row["smiles"]),
                                             lead_descriptor, evaluator.scales)
                        estimated = ((physical_w @ features) * np.asarray(new_sign)).tolist()
                        physical_estimated = (np.asarray(estimated) * np.asarray(new_sign)).tolist()
                        rank = _rank(estimated)
                    row.update(scenario=scenario, method="direct_mutation",
                               revision_target=None, sample=first+offset,
                               seed=seed, split=split, goal_sign=list(new_sign),
                               requested_goal=list(GOAL),
                               physical_exact_delta=physical_delta,
                               oriented_exact_delta=oriented,
                               predicted_oriented_delta=estimated,
                               predicted_physical_delta=physical_estimated,
                               endpoint2_preserved=preservation,
                               old_goal_success=old_success,
                               new_goal_success=success, goal_success=success,
                               support_excluded_success=success,
                               original_locality_goal_success=bool(success and row["original_locality"]),
                               eligible=eligible,
                               eligible_novel_success=bool(success and not row["source_match"]),
                               rank_score=rank, physical_truth_evaluation=True,
                               reused_from=None,
                               predictor_access="exact_RDKit_descriptors")
                    mutation_rows.append(row)
                seconds = _elapsed(started, device) + (mutation_setup if first==0 else 0.)
                mutation_cost = _combine_cost(len(batch), dict(seconds=seconds))
                for row in mutation_rows:
                    row.update(mutation_cost)
                    row["host_chemistry_seconds_per_attempt"] = (
                        seconds - (mutation_setup if first==0 else 0.)
                        + (enumeration_seconds if first==0 else 0.)) / len(batch)
                _append(mutation_rows,
                        (cid, scenario, "direct_mutation", None), ledger,
                        topk=topk, device=device)
                physical["truth_evaluations"] += len(batch)
        # This finite-library audit is deliberately post-hoc. It cannot select
        # a campaign, fit a response, tune a budget, or alter any generated arm.
        feasibility_started = _begin(device)
        reference = ExactMaskedEvaluator()
        library = dataset.compatible_library(cid,
            exclude_indices=support["reserve_indices"])
        eligible_identities = set()
        retained_identities = set()
        positive_identities = set()
        negative_identities = set()
        for graph in library:
            item = reference.evaluate_graph(graph, mask, cid,
                support_smiles=support["reserve_smiles"])
            delta = item["exact_delta"]
            eligible = (item["chemically_valid"] and item["connected"]
                        and item["protected_copy"] and not item["lead_copy"]
                        and not item["support_match"])
            if not eligible or delta is None:
                continue
            identity = item["smiles"]
            eligible_identities.add(identity)
            if delta[1] >= GOAL[1]:
                retained_identities.add(identity)
                if delta[0] >= GOAL[0]:
                    positive_identities.add(identity)
                if -delta[0] >= GOAL[0]:
                    negative_identities.add(identity)
        measured = _elapsed(feasibility_started, device)
        feasibility_seconds += measured
        feasibility_rows.append(dict(campaign_id=cid, mask_id=mask_id,
            split=split, seed=seed,
            scope="compatible_split_library_excluding_common_N16_reserve",
            reference_attempts=len(library), eligible_distinct=len(eligible_identities),
            endpoint2_retained_distinct=len(retained_identities),
            positive_goal_feasible_distinct=len(positive_identities),
            negative_goal_feasible_distinct=len(negative_identities),
            both_goals_feasible_distinct=len(positive_identities & negative_identities),
            common_support_reserve_count=len(support["reserve_smiles"]),
            posthoc_seconds=measured))
        print(json.dumps({"campaign_id": cid, "completed_scenarios": list(SCENARIOS),
                          "elapsed_seconds": _elapsed(start, device)}), flush=True)

    for item, fingerprint in zip((model, *((conditional,) if conditional is not None else ())),
                                 fingerprints):
        if tensor_fingerprint(item.state_dict().items()) != fingerprint:
            raise RuntimeError("frozen checkpoint changed during benchmark")
    all_rows = [row for rows in ledger.values() for row in rows]
    cases = []
    for (cid, scenario, method, target), rows in ledger.items():
        costs = ("charged_forward_nfe", "charged_backward_nfe",
                 "charged_decode_calls", "charged_critic_calls",
                 "charged_decode_backward_calls", "charged_critic_backward_calls",
                 "charged_seconds_per_attempt")
        cases.append(dict(campaign_id=cid, scenario=scenario, method=method,
                          revision_target=target, seed=seed, split=split,
                          old_support_size=16 if scenario=="goal_reversal" else 4,
                          new_support_size=16, old_goal_sign=[1,1],
                          new_goal_sign=[-1,1] if scenario=="goal_reversal" else [1,1],
                          goal_sign=[-1,1] if scenario=="goal_reversal" else [1,1],
                          **{field: sum(row[field] for row in rows) for field in costs},
                          total_charged_seconds=sum(row["charged_seconds_per_attempt"] for row in rows),
                          incremental_charged_seconds=sum(row["incremental_seconds_per_attempt"] for row in rows),
                          incremental_forward_nfe=sum(row["incremental_forward_nfe"] for row in rows),
                          incremental_backward_nfe=sum(row["incremental_backward_nfe"] for row in rows),
                          host_chemistry_seconds=sum(row.get("host_chemistry_seconds_per_attempt", 0.) for row in rows),
                          **_case_metrics(rows, topk)))
    write_csv(output / "attempts.csv", all_rows)
    write_csv(output / "old_stage_attempts.csv", old_stage_rows)
    write_csv(output / "cases.csv", cases)
    write_csv(output / "cost_batches.csv", _batch_rows(ledger))
    write_csv(output / "calibration.csv", calibration)
    write_csv(output / "feasibility.csv", feasibility_rows)
    if budgets:
        write_csv(output / "budget_frontier.csv",
                  _budget_rows(ledger, budgets, topk, basis="total")
                  + _budget_rows(ledger, budgets, topk, basis="incremental"))
    summary = dict(phase="complete", stage="revision_efficiency",
                   seed=seed, split=split, variant=payload["variant"],
                   campaigns=limit, attempts=attempts,
                   baseline_attempts=baseline_attempts,
                   conditional_attempts=conditional_attempts if conditional is not None else 0,
                   revision_grid=list(grid),
                   wall_budgets_seconds=list(budgets),
                   physical_forward_nfe=int(physical["forward_nfe"]),
                   physical_backward_nfe=int(physical["backward_nfe"]),
                   physical_evaluation_requests=int(physical["truth_evaluations"]),
                   physical_exact_response_calls=sum(bool(row["physical_truth_evaluation"]
                       and row["physical_exact_delta"] is not None) for row in all_rows+old_stage_rows),
                   logical_attempt_rows=len(all_rows),
                   posthoc_feasibility_seconds=feasibility_seconds,
                   seconds_including_chemistry=_elapsed(run_started, device),
                   completed_utc=now(), task=task,
                   interpretation="Goal reversal changes requested sign, not observed responses. Evidence revision adds nested N4->N16 noisy observations of the same truth. All methods exclude the same N16 reserve, retain source hits, and count invalid attempts. Restart candidates are physically evaluated once and logically reused at each reveal time; charged costs include discarded history. Common support preparation and the full paired Gaussian draw are charged to each neural strategy; mutation has no Gaussian draw. Offline exact-response auditing is included in strategy wall time; the post-hoc finite-library feasibility audit is timed separately and cannot affect selection. physical_evaluation_requests counts every committed generated graph and physical_exact_response_calls counts eligible generated graphs whose response was queried.")
    atomic_json(output / "summary.json", summary)
    return summary


run_revision_efficiency = run
