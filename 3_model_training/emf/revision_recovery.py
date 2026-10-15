"""Paired late-goal recovery of the frozen, hard-masked B conditional flow.

This module deliberately treats exact response calls as an evaluation resource.
Only the explicitly named oracle diagnostics may use their outcomes to select
or alter a proposal. All other arms rank with the fitted response surrogate.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from uuid import uuid4
import hashlib
import json
import math

import numpy as np
import torch

from .contracts import Latents
from .masked_edit_data import MaskedEditDataset, _descriptor, _features, fit_support_response
from .masked_edit_baselines import _direct_mutations
from .masked_edit_evaluation import ExactMaskedEvaluator
from .masked_edit_experiments import load_critic
from .masked_edit_revision import revision_grid
from .masked_edit_revision_benchmark import (
    GOAL, _begin, _elapsed, _condition, _orient, _rank, _score_rows,
    _case_metrics, _slice,
)
from .masked_edit_run import condition_for, load_model, now, write_csv
from .masked_edit_sampling import make_paired_gaussian, masked_initial_state, sample_masked_edit_latents
from .redesign_train import atomic_json, tensor_fingerprint


@dataclass(frozen=True)
class Batch:
    campaign_id: str
    method: str
    first: int
    count: int
    candidate_count: int
    total_seconds: float
    post_seconds: float
    prefix_seconds: float
    forward_nfe: int
    prefix_nfe: int
    exact_calls: int
    selected: tuple[dict, ...]
    raw_valid_connected: int = 0
    raw_protected_copy: int = 0


def _fingerprint(state: Latents) -> str:
    digest = hashlib.sha256()
    for item in (state.nodes, state.pairs):
        array = item.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
        digest.update(array.tobytes())
    return digest.hexdigest()


def _draw_id(seed: int, campaign_index: int, first: int) -> str:
    return f"{seed}:{campaign_index}:{first}"


def perturb_masked_state(state: Latents, condition, *, radius: float,
                         seed: int) -> Latents:
    """Apply a reproducible editable-coordinate RMS perturbation at one state.

    Radius is RMS over all editable node and pair latent channels, per example.
    Protected positions are restored from the lead and padding is zero.
    """
    if not math.isfinite(radius) or radius < 0:
        raise ValueError("radius must be finite and nonnegative")
    condition.validate(condition.masks)
    if state.nodes.shape != condition.lead.nodes.shape or state.pairs.shape != condition.lead.pairs.shape:
        raise ValueError("state and condition shapes differ")
    if radius == 0:
        return Latents(torch.where(condition.edit_masks.nodes[..., None], state.nodes,
                                   condition.lead.nodes),
                       torch.where(condition.edit_masks.pairs[..., None], state.pairs,
                                   condition.lead.pairs)).masked(condition.masks)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    node = torch.randn(state.nodes.shape, generator=generator,
                       dtype=torch.float32).to(state.nodes.device, state.nodes.dtype)
    pair = torch.randn(state.pairs.shape, generator=generator,
                       dtype=torch.float32).to(state.pairs.device, state.pairs.dtype)
    nm = condition.edit_masks.nodes[..., None]
    pm = condition.edit_masks.pairs[..., None]
    node = node * nm
    pair = pair * pm
    n = nm.sum(dim=(1, 2)) * state.nodes.shape[-1]
    p = pm.sum(dim=(1, 2)) * state.pairs.shape[-1]
    norm = torch.sqrt(((node.float() ** 2).sum(dim=(1, 2)) +
                       (pair.float() ** 2).sum(dim=(1, 2))) /
                      (n + p).clamp_min(1)).clamp_min(1e-12)
    node = node * (radius / norm)[:, None, None]
    pair = pair * (radius / norm)[:, None, None]
    return Latents(torch.where(nm, state.nodes + node, condition.lead.nodes),
                   torch.where(pm, state.pairs + pair, condition.lead.pairs)).masked(condition.masks)


def _prefix_trace(model, condition, gaussian: Latents, *, nfe: int, T: float,
                  indices: tuple[int, ...], device) -> tuple[dict[int, Latents], dict[int, float]]:
    """Run the old condition once, caching exact interior Euler states."""
    wanted = set(indices)
    if min(wanted) < 1 or max(wanted) >= nfe - 1:
        raise ValueError("prefix cache requires interior states")
    state = masked_initial_state(model.variant, gaussian, condition)
    times = torch.linspace(0, T, nfe, device=state.nodes.device, dtype=torch.float32)
    states, elapsed = {}, {}
    began = _begin(device)
    with torch.no_grad():
        for step in range(max(wanted)):
            time = times[step]
            predicted = model.predict_clean(state, time.expand(state.nodes.shape[0]),
                                            condition.masks, condition).masked(condition.masks)
            dt = times[step + 1] - time
            base = Latents((predicted.nodes-state.nodes)/(1-time),
                           (predicted.pairs-state.pairs)/(1-time)).masked(condition.edit_masks)
            state = Latents(state.nodes + dt*base.nodes,
                            state.pairs + dt*base.pairs).masked(condition.masks)
            state = Latents(torch.where(condition.edit_masks.nodes[..., None],
                                        state.nodes, condition.lead.nodes),
                            torch.where(condition.edit_masks.pairs[..., None],
                                        state.pairs, condition.lead.pairs)).masked(condition.masks)
            if step + 1 in wanted:
                states[step + 1] = Latents(state.nodes.detach().clone(),
                                           state.pairs.detach().clone())
                elapsed[step + 1] = _elapsed(began, device)
    return states, elapsed


def _oracle_rank(row: dict) -> tuple:
    delta = row.get("oriented_exact_delta")
    if not row.get("eligible") or delta is None:
        return (0, -1e30)
    deficit = np.maximum(np.asarray(GOAL) - np.asarray(delta), 0.)
    return (int(row["goal_success"]), -float(deficit @ deficit) + 1e-5 * float(delta[0]))


def _mark_oracle_rank(rows: list[dict]) -> None:
    for row in rows:
        row["surrogate_rank_score"] = row["rank_score"]
        success, value = _oracle_rank(row)
        row["rank_score"] = value + 1e6 * success


def _select_roots(rows: list[dict], *, oracle: bool) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[row["root_sample"]].append(row)
    selected = []
    for root in sorted(groups):
        key = (_oracle_rank if oracle else
               lambda row: (int(row.get("eligible", False)), row.get("rank_score", -1e30)))
        winner = max(groups[root], key=key)
        selected.append(winner)
    return selected


def _witness_audit(dataset, evaluator, ci: int, cid: str, support) -> tuple[dict, list[dict]]:
    """Commit every finite compatible reference before model inference."""
    mask_id = dataset.manifest["campaigns"][cid]["mask_id"]
    mask = dataset.masks[mask_id]
    library = dataset.compatible_library(cid, exclude_indices=support["reserve_indices"])
    local = ExactMaskedEvaluator()
    witnesses = []
    for graph in library:
        row = local.evaluate_graph(graph, mask, cid,
                                   support_smiles=support["reserve_smiles"])
        delta = row["exact_delta"]
        if (row["chemically_valid"] and row["connected"] and row["protected_copy"]
                and not row["lead_copy"] and not row["support_match"] and delta is not None):
            witnesses.append({"campaign_id": cid, "mask_id": mask_id,
                              "smiles": row["smiles"], "exact_delta": delta,
                              "negative_goal_success": bool(-delta[0] >= GOAL[0]
                                                            and delta[1] >= GOAL[1]),
                              "positive_goal_success": bool(delta[0] >= GOAL[0]
                                                            and delta[1] >= GOAL[1])})
    feasible = [row for row in witnesses if row["negative_goal_success"]]
    audit = {"campaign_id": cid, "mask_id": mask_id, "split": dataset.split,
             "reference_attempts": len(library), "eligible_distinct": len({r["smiles"] for r in witnesses}),
             "negative_goal_feasible_distinct": len({r["smiles"] for r in feasible}),
             "positive_goal_feasible_distinct": len({r["smiles"] for r in witnesses if r["positive_goal_success"]}),
             "feasible_stratum": bool(feasible),
             "witness_smiles": [r["smiles"] for r in feasible],
             "witness_exact_deltas": [r["exact_delta"] for r in feasible],
             "scope": "compatible_split_library_excluding_common_N16_reserve"}
    return audit, witnesses


def budget_frontier(batches: list[Batch], budgets: tuple[float, ...], *, topk: int) -> list[dict]:
    """Use only completed batches; a trial pool becomes available together."""
    by_case = defaultdict(list)
    for batch in batches:
        by_case[batch.campaign_id, batch.method].append(batch)
    output = []
    for (cid, method), ordered in by_case.items():
        ordered.sort(key=lambda b: b.first)
        for basis in ("total", "post_update"):
            for budget in budgets:
                spent, selected, used = 0., [], []
                for batch in ordered:
                    charge = batch.total_seconds if basis == "total" else batch.post_seconds
                    if spent + charge > budget + 1e-12:
                        break
                    spent += charge
                    used.append(batch)
                    selected.extend(batch.selected)
                metrics = _case_metrics(selected, topk)
                output.append({"campaign_id": cid, "method": method,
                               "budget_basis": basis, "wall_budget_seconds": budget,
                               "charged_seconds_used": spent,
                               "total_charged_seconds": sum(b.total_seconds for b in used),
                               "post_update_charged_seconds": sum(b.post_seconds for b in used),
                               "candidate_attempts": sum(b.candidate_count for b in used),
                               "raw_candidate_attempts": sum(b.candidate_count for b in used),
                               "raw_valid_connected_attempts": sum(b.raw_valid_connected for b in used),
                               "raw_protected_copy_attempts": sum(b.raw_protected_copy for b in used),
                               "raw_candidate_valid_connected_attempts": sum(b.raw_valid_connected for b in used),
                               "raw_candidate_protected_attempts": sum(b.raw_protected_copy for b in used),
                               "exact_calls": sum(b.exact_calls for b in used),
                               "charged_forward_nfe": sum(b.forward_nfe for b in used),
                               "prefix_forward_nfe": sum(b.prefix_nfe for b in used),
                               "cap_limited": len(used) == len(ordered),
                               **metrics})
    return output


def _write_case_checkpoint(output: Path, cid: str, payload: dict) -> None:
    safe = cid.replace("/", "_")
    atomic_json(output / "campaign_checkpoints" / f"{safe}.json", payload)


def _atomic_csv(path: Path, rows: list[dict]) -> None:
    """Publish a complete ledger only after its temporary file is closed."""
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.touch()
        write_csv(temporary, rows)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def run(task: dict, device="cuda") -> dict:
    """Run E1 rollback and E2 explicit suffix interventions on frozen B.

    Required task fields: conditional_checkpoint, critic, output, seed. The
    optional prior checkpoint supplies a strong independent sampling baseline.
    A validation-selected budget grid must be passed for test analysis.
    """
    missing = [key for key in ("conditional_checkpoint", "critic", "output", "seed") if key not in task]
    if missing:
        raise ValueError(f"missing task fields: {missing}")
    split = task.get("split", "validation")
    budgets = tuple(map(float, task.get("wall_budgets_seconds", ())))
    if split == "test" and not budgets:
        raise ValueError("test requires validation-selected wall_budgets_seconds")
    if any(not math.isfinite(b) or b <= 0 for b in budgets):
        raise ValueError("wall budgets must be finite and positive")
    attempts = int(task.get("attempts", 128))
    diagnostic_roots = int(task.get("diagnostic_roots", attempts))
    baseline_attempts = int(task.get("baseline_attempts", 1600))
    chunk = int(task.get("chunk", 8))
    nfe = int(task.get("nfe", 16))
    T = float(task.get("T", .95))
    topk = int(task.get("topk", 16))
    trials = int(task.get("intervention_trials", 8))
    search_trials = int(task.get("search_trials", trials))
    mutation_attempts = int(task.get("mutation_attempts", 256))
    radius = float(task.get("intervention_radius", .5))
    default_methods = {"conditional_continue", "conditional_restart", "rollback_25",
                       "rollback_50", "randomized_suffix", "oracle_select_same_suffix_pool",
                       "oracle_adaptive_suffix", "oracle_independent_restart",
                       "conditional_sampling_baseline", "prior_sampling_baseline",
                       "direct_mutation", "oracle_direct_mutation"}
    if task.get("adapter_checkpoint"):
        default_methods.add("adapter_continue")
    methods = set(task.get("methods", default_methods))
    if not methods or methods - (default_methods | {"adapter_continue"}):
        raise ValueError("unknown or empty recovery methods")
    if "oracle_select_same_suffix_pool" in methods and "randomized_suffix" not in methods:
        raise ValueError("oracle selection requires the identical randomized suffix pool")
    if "oracle_direct_mutation" in methods and "direct_mutation" not in methods:
        raise ValueError("oracle mutation selection requires direct_mutation")
    if "adapter_continue" in methods and not task.get("adapter_checkpoint"):
        raise ValueError("adapter_continue requires adapter_checkpoint")
    if min(attempts, diagnostic_roots, baseline_attempts, mutation_attempts,
           chunk, topk, trials) < 1 or search_trials < 0:
        raise ValueError("positive attempt, batch, selection and intervention counts required")
    if baseline_attempts < attempts or diagnostic_roots > attempts or not math.isfinite(radius) or radius <= 0:
        raise ValueError("baseline cap must cover main attempts; intervention radius must be positive")
    if diagnostic_roots < attempts and diagnostic_roots % chunk:
        raise ValueError("diagnostic_roots must end on a completed batch boundary")
    grid = revision_grid(nfe, T, targets=(.25, .5, .75))
    indices = {item["target_time"]: item["grid_index"] for item in grid}
    reveal = indices[.75]
    if len(set(indices.values())) < 3:
        raise ValueError("nfe grid must distinguish rollback and reveal states")
    output = Path(task["output"])
    output.mkdir(parents=True, exist_ok=True)
    seed = int(task["seed"])
    started = _begin(device)
    dataset = MaskedEditDataset(split=split, kind="vector")
    evaluator = ExactMaskedEvaluator()
    declared = task.get("campaign_ids")
    if declared is None:
        limit = min(int(task.get("campaign_limit", len(dataset.campaign_ids))),
                    len(dataset.campaign_ids))
        campaign_ids = tuple(dataset.campaign_ids[:limit])
    else:
        campaign_ids = tuple(declared)
        if (len(campaign_ids) != len(set(campaign_ids))
                or any(cid not in dataset.campaign_ids for cid in campaign_ids)):
            raise ValueError("campaign_ids must be distinct campaigns in the requested split")
        limit = len(campaign_ids)
    if limit < 1:
        raise ValueError("at least one campaign is required")
    campaign_indices = {cid: dataset.campaign_ids.index(cid) for cid in campaign_ids}
    # No model is loaded or queried until the reference audit is complete.
    supports, audits, witnesses = {}, [], []
    for cid in campaign_ids:
        ci = campaign_indices[cid]
        support = dataset.nested_support(ci, seed=77600 + ci, sizes=(16,))
        supports[cid] = support
        audit, records = _witness_audit(dataset, evaluator, ci, cid, support)
        audits.append(audit)
        witnesses.extend(records)
    _atomic_csv(output / "reference_witnesses.csv", witnesses)
    _atomic_csv(output / "feasibility.csv", audits)
    atomic_json(output / "reference_manifest.json", {"created_utc": now(), "split": split,
        "campaign_ids": list(campaign_ids), "seed": seed,
        "audit": audits, "model_evaluations_before_audit": 0})

    model, payload = load_model(task["conditional_checkpoint"], device)
    if model.variant != "B" or not payload.get("conditioned", False):
        raise ValueError("conditional checkpoint must be a task-conditioned hard-masked B")
    model.eval().requires_grad_(False)
    prior = None
    if task.get("checkpoint"):
        prior, prior_payload = load_model(task["checkpoint"], device)
        if prior.variant != "B":
            raise ValueError("sampling baseline checkpoint must be hard-masked B")
        prior.eval().requires_grad_(False)
    critic, _ = load_critic(task["critic"], model, device)
    adapter = None
    if task.get("adapter_checkpoint"):
        from .revision_adapter import load_adapter
        adapter, _ = load_adapter(task["adapter_checkpoint"], device,
                                  base_checkpoint=task["conditional_checkpoint"])
        adapter.eval().requires_grad_(False)
    frozen = [(item, tensor_fingerprint(item.state_dict().items())) for item in
              (model, critic, *((prior,) if prior is not None else ()),
               *((adapter,) if adapter is not None else ()))]
    scales = torch.tensor(evaluator.scales, device=device, dtype=torch.float32)
    all_rows, selected_rows, traces, batches = [], [], [], []
    exact_physical = 0
    exact_function_calls = 0
    task_sha256 = hashlib.sha256(json.dumps(task, sort_keys=True, default=str).encode()).hexdigest()
    for cid in campaign_ids:
        ci = campaign_indices[cid]
        checkpoint_path = output / "campaign_checkpoints" / f"{cid.replace('/', '_')}.json"
        if checkpoint_path.exists():
            saved = json.loads(checkpoint_path.read_text())
            if saved.get("task_sha256") == task_sha256 and "rows" in saved:
                all_rows.extend(saved["rows"])
                selected_rows.extend(saved["selected_rows"])
                traces.extend(saved["traces"])
                batches.extend(Batch(**{**item, "selected": tuple(item["selected"])})
                               for item in saved["batches"])
                exact_physical += saved["physical_exact_evaluation_requests"]
                exact_function_calls += saved["physical_exact_response_function_calls"]
                print(json.dumps({"campaign_id": cid, "resumed_from_checkpoint": True}), flush=True)
                continue
        campaign_start = _begin(device)
        first_row, first_selected, first_trace, first_batch = (
            len(all_rows), len(selected_rows), len(traces), len(batches))
        first_physical, first_function = exact_physical, exact_function_calls
        support = supports[cid]
        mask_id = dataset.manifest["campaigns"][cid]["mask_id"]
        mask = dataset.masks[mask_id]
        fit_started = _begin(device)
        record = support["records"][16]
        fit = fit_support_response([g["smiles"] for g in record["graphs"]],
                                   record["values"], mask.lead_smiles, evaluator.scales)
        fit_seconds = _elapsed(fit_started, device)
        coeff = torch.as_tensor(fit["coefficients"], device=device, dtype=torch.float32)
        raw = torch.as_tensor(_descriptor(mask.lead_smiles), device=device,
                              dtype=torch.float32)[None]
        noise_started = _begin(device)
        noise = make_paired_gaussian(model.config, max(attempts, baseline_attempts),
                                     seed * 100000 + ci, device=device)
        noise_seconds = _elapsed(noise_started, device)
        case_evaluators = defaultdict(ExactMaskedEvaluator)

        def record_batch(method, first, count, candidate_rows, winners, *,
                         wall, prefix_wall, nforward, prefix_nfe, physical_calls):
            nonlocal exact_physical, exact_function_calls
            exact_physical += physical_calls
            exact_function_calls += sum(row.get("physical_exact_delta") is not None
                                        for row in candidate_rows)
            exact_count = len(candidate_rows)
            if exact_count != physical_calls:
                raise RuntimeError("physical exact-call count mismatch")
            batch_id = f"{cid}:{method}:{first}"
            for row in candidate_rows:
                row.update(batch_id=batch_id, root_sample=row.get("root_sample", row["sample"]),
                           root_draw_id=row.get("root_draw_id", _draw_id(seed, ci, row["sample"])),
                           charged_prefix_seconds=prefix_wall / exact_count,
                           charged_post_update_seconds=(wall-prefix_wall) / exact_count,
                           charged_total_seconds=wall / exact_count,
                           charged_forward_nfe=nforward / exact_count,
                           physical_exact_call=True)
            for row in winners:
                row["selected_within_root"] = True
            all_rows.extend(candidate_rows)
            selected_rows.extend(winners)
            batches.append(Batch(cid, method, first, count, exact_count, wall,
                                 wall-prefix_wall, prefix_wall, nforward, prefix_nfe,
                                 physical_calls, tuple(winners),
                                 sum(bool(r["chemically_valid"] and r["connected"])
                                     for r in candidate_rows),
                                 sum(bool(r["protected_copy"]) for r in candidate_rows)))

        for first in range(0, attempts, chunk):
            count = min(chunk, attempts-first)
            draw = _slice(noise, first, count)
            prep_started = _begin(device)
            base = condition_for(model, dataset, mask_id, count)
            old_w = _orient(coeff[None].expand(count, -1, -1), (1, 1))
            new_w = _orient(coeff[None].expand(count, -1, -1), (-1, 1))
            old_cond, new_cond = _condition(base, old_w), _condition(base, new_w)
            raw_batch = raw.expand(count, -1)
            prep_seconds = _elapsed(prep_started, device)
            trace, timings = _prefix_trace(model, old_cond, draw, nfe=nfe, T=T,
                                            indices=tuple(indices.values()), device=device)
            historical = timings[reveal] + prep_seconds + (
                fit_seconds + noise_seconds if first == 0 else 0.)
            reveal_state = trace[reveal]
            root_ids = [_draw_id(seed, ci, first+i) for i in range(count)]
            for offset in range(count):
                traces.append({"campaign_id": cid, "root_sample": first+offset,
                               "root_draw_id": root_ids[offset], "reveal_grid_index": reveal,
                               "reveal_time": grid[2]["actual_time"],
                               "reveal_state_sha256": _fingerprint(_slice(reveal_state, offset, 1)),
                               "rollback_25_state_sha256": _fingerprint(_slice(trace[indices[.25]], offset, 1)),
                               "rollback_50_state_sha256": _fingerprint(_slice(trace[indices[.5]], offset, 1))})

            def rollout(method, state, index, *, extra_wall=0., intervention=None):
                begun = _begin(device)
                with torch.no_grad():
                    result = sample_masked_edit_latents(model, new_cond,
                        nfe=nfe-index, T=T, start_t=float(torch.linspace(0,T,nfe)[index]),
                        initial_state=state)
                flow_wall = _elapsed(begun, device)
                begun = _begin(device)
                rows = _score_rows(result, model, critic, new_cond, new_w,
                    raw_batch, scales, dataset, cid, support,
                    case_evaluators[method], scenario="goal_reversal", method=method,
                    target=.75, first=first, sign=(-1,1), seed=seed, split=split,
                    cost={})
                eval_wall = _elapsed(begun, device)
                for i, row in enumerate(rows):
                    row.update(root_sample=first+i, root_draw_id=root_ids[i],
                               reveal_state_sha256=traces[-count+i]["reveal_state_sha256"],
                               intervention=intervention, oracle_access=False)
                wall = historical + extra_wall + flow_wall + eval_wall
                record_batch(method, first, count, rows, rows, wall=wall,
                             prefix_wall=historical, nforward=(reveal+nfe-index)*count,
                             prefix_nfe=reveal*count, physical_calls=count)

            if "conditional_continue" in methods:
                rollout("conditional_continue", reveal_state, reveal)
            if "rollback_25" in methods:
                rollout("rollback_25", trace[indices[.25]], indices[.25])
            if "rollback_50" in methods:
                rollout("rollback_50", trace[indices[.5]], indices[.5])
            if "adapter_continue" in methods:
                begun = _begin(device)
                with torch.no_grad():
                    mapped = adapter.apply_state(reveal_state, grid[2]["actual_time"],
                                                 old_cond, new_cond)
                map_wall = _elapsed(begun, device)
                rollout("adapter_continue", mapped, reveal, extra_wall=map_wall,
                        intervention="source_trained_bounded_state_adapter")

            if "conditional_restart" in methods:
                begun = _begin(device)
                with torch.no_grad():
                    fresh = sample_masked_edit_latents(model, new_cond, draw, nfe=nfe, T=T)
                flow_wall = _elapsed(begun, device)
                begun = _begin(device)
                fresh_rows = _score_rows(fresh, model, critic, new_cond, new_w,
                    raw_batch, scales, dataset, cid, support,
                    case_evaluators["conditional_restart"], scenario="goal_reversal",
                    method="conditional_restart", target=.75, first=first, sign=(-1,1),
                    seed=seed, split=split, cost={})
                eval_wall = _elapsed(begun, device)
                for i, row in enumerate(fresh_rows):
                    row.update(root_sample=first+i, root_draw_id=root_ids[i], oracle_access=False,
                               reveal_state_sha256=traces[-count+i]["reveal_state_sha256"])
                record_batch("conditional_restart", first, count, fresh_rows, fresh_rows,
                             wall=historical+flow_wall+eval_wall, prefix_wall=historical,
                             nforward=(reveal+nfe)*count, prefix_nfe=reveal*count,
                             physical_calls=count)

            # Explicitly randomize editable coordinates at the *same* reveal
            # state. No rerun of the deterministic ODE is counted as a trial.
            pool_rows, pool_wall = [], 0.
            for trial in range(trials if ("randomized_suffix" in methods and
                                      first < diagnostic_roots) else 0):
                trial_seed = seed*10_000_000 + ci*100_000 + first*100 + trial
                begun = _begin(device)
                moved = perturb_masked_state(reveal_state, new_cond,
                                             radius=radius, seed=trial_seed)
                with torch.no_grad():
                    sample = sample_masked_edit_latents(model, new_cond,
                        nfe=nfe-reveal, T=T,
                        start_t=grid[2]["actual_time"], initial_state=moved)
                flow_time = _elapsed(begun, device)
                begun = _begin(device)
                rows = _score_rows(sample, model, critic, new_cond, new_w,
                    raw_batch, scales, dataset, cid, support,
                    case_evaluators["randomized_suffix"], scenario="goal_reversal",
                    method="randomized_suffix", target=.75, first=first,
                    sign=(-1,1), seed=seed, split=split, cost={})
                pool_wall += flow_time + _elapsed(begun, device)
                for i, row in enumerate(rows):
                    row.update(root_sample=first+i, root_draw_id=root_ids[i],
                               reveal_state_sha256=traces[-count+i]["reveal_state_sha256"],
                               trial=trial, intervention_radius_rms=radius,
                               intervention_seed=trial_seed, oracle_access=False)
                pool_rows.extend(rows)
            if pool_rows:
                chosen = _select_roots(pool_rows, oracle=False)
                record_batch("randomized_suffix", first, count, pool_rows, chosen,
                             wall=historical+pool_wall, prefix_wall=historical,
                             nforward=(reveal+trials*(nfe-reveal))*count,
                             prefix_nfe=reveal*count, physical_calls=len(pool_rows))
                if "oracle_select_same_suffix_pool" in methods:
                    oracle_rows = [{**row, "method": "oracle_select_same_suffix_pool",
                                    "oracle_access": True, "physical_exact_call": False,
                                    "reused_from": "randomized_suffix"} for row in pool_rows]
                    _mark_oracle_rank(oracle_rows)
                    oracle_chosen = _select_roots(oracle_rows, oracle=True)
                    # Logical strategy charge includes all privileged exact calls.
                    for row in oracle_rows:
                        row["batch_id"] = f"{cid}:oracle_select_same_suffix_pool:{first}"
                    all_rows.extend(oracle_rows)
                    selected_rows.extend(oracle_chosen)
                    batches.append(Batch(cid, "oracle_select_same_suffix_pool", first, count,
                                         len(oracle_rows), historical+pool_wall,
                                         pool_wall, historical,
                                         (reveal+trials*(nfe-reveal))*count,
                                         reveal*count, len(oracle_rows), tuple(oracle_chosen),
                                         sum(bool(r["chemically_valid"] and r["connected"])
                                             for r in oracle_rows),
                                         sum(bool(r["protected_copy"]) for r in oracle_rows)))

            if search_trials and first < diagnostic_roots:
                # Privileged adaptive local search and an equally sized pool of
                # independent complete restarts. Both spend K exact calls/root.
                for method in ("oracle_adaptive_suffix", "oracle_independent_restart"):
                    if method not in methods:
                        continue
                    candidates, search_wall = [], 0.
                    current_radius = np.full(count, radius, dtype=np.float64)
                    for trial in range(search_trials):
                        trial_seed = seed*20_000_000 + ci*200_000 + first*100 + trial
                        begun = _begin(device)
                        if method == "oracle_adaptive_suffix":
                            parts = []
                            for i in range(count):
                                one = _slice(reveal_state, i, 1)
                                one_cond = new_cond.index_select(torch.tensor([i], device=device))
                                parts.append(perturb_masked_state(one, one_cond,
                                    radius=float(current_radius[i]), seed=trial_seed+i))
                            state = Latents(torch.cat([x.nodes for x in parts]),
                                            torch.cat([x.pairs for x in parts]))
                            with torch.no_grad():
                                sample = sample_masked_edit_latents(model, new_cond,
                                    nfe=nfe-reveal, T=T,
                                    start_t=grid[2]["actual_time"], initial_state=state)
                        else:
                            independent = (draw if trial == 0 else
                                make_paired_gaussian(model.config, count,
                                                     trial_seed, device=device))
                            with torch.no_grad():
                                sample = sample_masked_edit_latents(model, new_cond,
                                    independent, nfe=nfe, T=T)
                        flow_time = _elapsed(begun, device)
                        begun = _begin(device)
                        rows = _score_rows(sample, model, critic, new_cond, new_w,
                            raw_batch, scales, dataset, cid, support,
                            case_evaluators[method], scenario="goal_reversal",
                            method=method, target=.75, first=first, sign=(-1,1),
                            seed=seed, split=split, cost={})
                        search_wall += flow_time + _elapsed(begun, device)
                        for i, row in enumerate(rows):
                            row.update(root_sample=first+i, root_draw_id=root_ids[i],
                                       reveal_state_sha256=traces[-count+i]["reveal_state_sha256"],
                                       trial=trial, oracle_access=True,
                                       intervention_radius_rms=(float(current_radius[i])
                                           if method == "oracle_adaptive_suffix" else None),
                                       intervention_seed=trial_seed+i)
                            if method == "oracle_adaptive_suffix":
                                    current_radius[i] = min(radius, max(.05, current_radius[i]*.5)) if row["goal_success"] else min(radius, current_radius[i]*1.5)
                        candidates.extend(rows)
                    _mark_oracle_rank(candidates)
                    winners = _select_roots(candidates, oracle=True)
                    suffix_nfe = nfe-reveal if method == "oracle_adaptive_suffix" else nfe
                    record_batch(method, first, count, candidates, winners,
                        wall=historical+search_wall, prefix_wall=historical,
                        nforward=(reveal+search_trials*suffix_nfe)*count,
                        prefix_nfe=reveal*count, physical_calls=len(candidates))

        # The prior and conditional baseline each draw independent full samples
        # in completed batches, up to the same predeclared 1,600 proposal cap.
        for baseline_name, baseline_model in (("conditional_sampling_baseline", model),
                                               ("prior_sampling_baseline", prior)):
            if baseline_model is None or baseline_name not in methods:
                continue
            for first in range(0, baseline_attempts, chunk):
                count = min(chunk, baseline_attempts-first)
                baseline_prep_started = _begin(device)
                b = condition_for(baseline_model, dataset, mask_id, count)
                W = _orient(coeff[None].expand(count,-1,-1), (-1,1))
                condition = _condition(b, W)
                draw = _slice(noise, first, count)
                baseline_prep = _elapsed(baseline_prep_started, device)
                begun = _begin(device)
                with torch.no_grad():
                    sampled = sample_masked_edit_latents(baseline_model, condition,
                                                         draw, nfe=nfe, T=T)
                flow_wall = _elapsed(begun, device)
                begun = _begin(device)
                rows = _score_rows(sampled, baseline_model, critic, condition, W,
                    raw.expand(count,-1), scales, dataset, cid, support,
                    case_evaluators[baseline_name], scenario="goal_reversal",
                    method=baseline_name, target=None, first=first,
                    sign=(-1,1), seed=seed, split=split, cost={})
                eval_wall = _elapsed(begun, device)
                for i, row in enumerate(rows):
                    row.update(root_sample=first+i, oracle_access=False,
                               root_draw_id=_draw_id(seed,ci,first+i))
                record_batch(baseline_name, first, count, rows, rows,
                             wall=flow_wall+eval_wall+baseline_prep+
                                  (fit_seconds+noise_seconds if first==0 else 0.),
                             prefix_wall=0., nforward=nfe*count,
                             prefix_nfe=0, physical_calls=count)

        if "direct_mutation" in methods:
            # Same protected graph and N16 support reserve. Exact chemistry
            # descriptors are public inputs to the fitted response surrogate;
            # hidden response coefficients enter only post-commit evaluation.
            mutation_started = _begin(device)
            mutations = list(_direct_mutations(mask, mutation_attempts,
                                               seed*100000+ci))
            enumeration_seconds = _elapsed(mutation_started, device)
            source_set = {dataset.smiles[int(j)] for j in dataset.source_graph_indices}
            lead_descriptor = _descriptor(mask.lead_smiles)
            physical_w = fit["coefficients"]
            # One mutation group spends the same K exact evaluation requests
            # as one local suffix search root.
            mutation_group_size = trials
            for first in range(0, len(mutations), mutation_group_size):
                chosen_graphs = mutations[first:first+mutation_group_size]
                begun = _begin(device)
                rows = []
                for offset, graph in enumerate(chosen_graphs):
                    row = case_evaluators["direct_mutation"].evaluate_graph(
                        graph, mask, cid, support_smiles=support["reserve_smiles"],
                        source_smiles=source_set, known_library_smiles=dataset.smiles)
                    delta = row["exact_delta"]
                    oriented = None if delta is None else [-delta[0], delta[1]]
                    eligible = bool(row["chemically_valid"] and row["connected"]
                                    and row["protected_copy"] and not row["lead_copy"]
                                    and not row["support_match"])
                    if row["smiles"] is None:
                        predicted = None
                        score = -1e30
                    else:
                        feature = _features(_descriptor(row["smiles"]),
                                            lead_descriptor, evaluator.scales)
                        predicted = ((physical_w @ feature) * np.asarray((-1,1))).tolist()
                        score = _rank(predicted)
                    success = bool(eligible and oriented is not None
                                   and oriented[0] >= GOAL[0] and oriented[1] >= GOAL[1])
                    row.update(scenario="goal_reversal", method="direct_mutation",
                               revision_target=None, sample=first+offset,
                               root_sample=first+offset, root_draw_id=f"mutation:{seed}:{ci}:{first+offset}",
                               seed=seed, split=split, goal_sign=[-1,1],
                               requested_goal=list(GOAL), physical_exact_delta=delta,
                               oriented_exact_delta=oriented,
                               predicted_oriented_delta=predicted,
                               eligible=eligible, goal_success=success,
                               new_goal_success=success,
                               old_goal_success=bool(eligible and delta is not None
                                   and delta[0] >= GOAL[0] and delta[1] >= GOAL[1]),
                               endpoint2_preserved=bool(eligible and delta is not None
                                   and delta[1] >= GOAL[1]),
                               rank_score=score, oracle_access=False,
                               predictor_access="exact_RDKit_descriptors")
                    rows.append(row)
                wall = _elapsed(begun, device) + (enumeration_seconds+fit_seconds if first==0 else 0.)
                record_batch("direct_mutation", first, len(rows), rows, rows,
                             wall=wall, prefix_wall=0., nforward=0,
                             prefix_nfe=0, physical_calls=len(rows))
                if "oracle_direct_mutation" in methods:
                    oracle_rows = [{**row, "method": "oracle_direct_mutation",
                                    "oracle_access": True, "physical_exact_call": False,
                                    "root_sample": first // mutation_group_size,
                                    "reused_from": "direct_mutation"} for row in rows]
                    _mark_oracle_rank(oracle_rows)
                    oracle_chosen = _select_roots(oracle_rows, oracle=True)
                    for row in oracle_rows:
                        row["batch_id"] = f"{cid}:oracle_direct_mutation:{first}"
                    all_rows.extend(oracle_rows)
                    selected_rows.extend(oracle_chosen)
                    batches.append(Batch(cid, "oracle_direct_mutation", first,
                                         len(oracle_chosen), len(oracle_rows), wall, wall,
                                         0., 0, 0, len(oracle_rows), tuple(oracle_chosen),
                                         sum(bool(r["chemically_valid"] and r["connected"])
                                             for r in oracle_rows),
                                         sum(bool(r["protected_copy"]) for r in oracle_rows)))
        _write_case_checkpoint(output, cid, {"campaign_id": cid,
            "task_sha256": task_sha256,
            "completed_utc": now(), "seed": seed,
            "raw_attempts": sum(r["campaign_id"] == cid for r in all_rows),
            "rows": all_rows[first_row:],
            "selected_rows": selected_rows[first_selected:],
            "traces": traces[first_trace:],
            "batches": [vars(b) for b in batches[first_batch:]],
            "physical_exact_evaluation_requests": exact_physical-first_physical,
            "physical_exact_response_function_calls": exact_function_calls-first_function,
            "elapsed_seconds": _elapsed(campaign_start, device)})
        print(json.dumps({"campaign_id": cid, "elapsed_seconds": _elapsed(campaign_start, device)}), flush=True)

    for item, before in frozen:
        if tensor_fingerprint(item.state_dict().items()) != before:
            raise RuntimeError("a frozen model changed during recovery benchmark")
    audit_map = {row["campaign_id"]: row for row in audits}
    by_root = defaultdict(dict)
    for row in selected_rows:
        by_root[(row["campaign_id"], row["root_sample"])][row["method"]] = row
    randomized_successes = defaultdict(bool)
    adaptive_successes = defaultdict(bool)
    for row in all_rows:
        key = (row["campaign_id"], row["root_sample"])
        if row["method"] == "randomized_suffix" and row["goal_success"]:
            randomized_successes[key] = True
        if row["method"] == "oracle_adaptive_suffix" and row["goal_success"]:
            adaptive_successes[key] = True
    root_rows = []
    for trace in traces:
        key = (trace["campaign_id"], trace["root_sample"])
        cases_for_root = by_root[key]
        direct = cases_for_root.get("conditional_continue")
        random_recovery = randomized_successes[key]
        adaptive_recovery = adaptive_successes[key]
        root_rows.append({"campaign_id": key[0], "root_id": trace["root_draw_id"],
                          "seed": seed, "root_sample": key[1],
                          "feasible_stratum": audit_map[key[0]]["feasible_stratum"],
                          "reveal_state_sha256": trace["reveal_state_sha256"],
                          "direct_success": None if direct is None else bool(direct["goal_success"]),
                          "diagnostic_tested": key[1] < diagnostic_roots,
                          "randomized_suffix_any_success": random_recovery,
                          "oracle_adaptive_any_success": adaptive_recovery,
                          "intervention_recovered": bool(direct is not None and
                              not direct["goal_success"] and (random_recovery or adaptive_recovery)),
                          "bounded_radius_rms": radius,
                          "bounded_radius": radius <= .5})
    _atomic_csv(output / "roots.csv", root_rows)
    _atomic_csv(output / "attempts.csv", all_rows)
    _atomic_csv(output / "selected.csv", selected_rows)
    _atomic_csv(output / "root_traces.csv", traces)
    batch_rows = [{key: getattr(b,key) for key in Batch.__dataclass_fields__ if key != "selected"}
                  for b in batches]
    _atomic_csv(output / "cost_batches.csv", batch_rows)
    cases = []
    grouped = defaultdict(list)
    for row in selected_rows:
        grouped[row["campaign_id"], row["method"]].append(row)
    for (cid, method), rows in grouped.items():
        charged = [b for b in batches if b.campaign_id == cid and b.method == method]
        cases.append({"campaign_id": cid, "method": method,
                      "timing_role": ("fresh_start_reference" if method in
                          {"conditional_sampling_baseline", "prior_sampling_baseline",
                           "direct_mutation", "oracle_direct_mutation"}
                          else "late_revision_strategy"),
                      "feasible_stratum": audit_map[cid]["feasible_stratum"],
                      "root_attempts": len(rows),
                      "candidate_attempts": sum(b.candidate_count for b in charged),
                      "raw_candidate_attempts": sum(b.candidate_count for b in charged),
                      "raw_valid_connected_attempts": sum(b.raw_valid_connected for b in charged),
                      "raw_protected_copy_attempts": sum(b.raw_protected_copy for b in charged),
                      "raw_candidate_valid_connected_attempts": sum(b.raw_valid_connected for b in charged),
                      "raw_candidate_protected_attempts": sum(b.raw_protected_copy for b in charged),
                      "selected_root_validity_fraction": (sum(bool(r["chemically_valid"] and r["connected"])
                          for r in rows) / len(rows)),
                      "exact_calls": sum(b.exact_calls for b in charged),
                      "total_charged_seconds": sum(b.total_seconds for b in charged),
                      "post_update_charged_seconds": sum(b.post_seconds for b in charged),
                      "charged_forward_nfe": sum(b.forward_nfe for b in charged),
                      **_case_metrics(rows, topk)})
    _atomic_csv(output / "cases.csv", cases)
    if budgets:
        _atomic_csv(output / "budget_frontier.csv", budget_frontier(batches,budgets,topk=topk))
    summary = {"stage": task.get("stage", "revision_recovery"), "phase": "complete",
               "seed": seed, "split": split, "campaigns": limit,
               "feasible_campaigns": sum(row["feasible_stratum"] for row in audits),
               "main_attempts_per_campaign": attempts,
               "diagnostic_roots_per_campaign": diagnostic_roots,
               "baseline_attempts_per_campaign": baseline_attempts,
               "intervention_trials": trials, "search_trials": search_trials,
               "intervention_radius_rms": radius,
               "mutation_attempts_per_campaign": mutation_attempts if "direct_mutation" in methods else 0,
               "physical_exact_evaluation_requests": exact_physical,
               "physical_exact_response_function_calls": exact_function_calls,
               "logical_attempt_rows": len(all_rows),
               "charged_exact_calls": sum(b.exact_calls for b in batches),
               "baseline_failed_roots": sum(r["direct_success"] is False for r in root_rows),
               "bounded_recovered_failed_roots": sum(r["intervention_recovered"] and r["bounded_radius"] for r in root_rows),
               "campaigns_with_bounded_recovery": len({r["campaign_id"] for r in root_rows
                   if r["intervention_recovered"] and r["bounded_radius"]}),
               "source_gate_8_roots_2_campaigns": bool(
                   sum(r["intervention_recovered"] and r["bounded_radius"] for r in root_rows) >= 8
                   and len({r["campaign_id"] for r in root_rows
                            if r["intervention_recovered"] and r["bounded_radius"]}) >= 2),
               "reveal_grid_index": reveal, "revision_grid": list(grid),
               "wall_budgets_seconds": list(budgets),
               "seconds_including_chemistry": _elapsed(started,device),
               "completed_utc": now(), "task": task,
               "interpretation": "Every campaign was reference-audited before model evaluation. Oracle arms are privileged diagnostics and pay for all decoded exact-call requests. Local search chooses one candidate per paired root before the final top-16 ranking; the mutation comparator ranks its full pool. Each local trial explicitly perturbs editable latent coordinates at the same saved root state; deterministic ODE replay alone is not an independent trial. Invalid graphs remain in the attempt ledger. Total rollback cost includes the original prefix to the reveal time."}
    atomic_json(output / "summary.json", summary)
    return summary


run_recovery = run
