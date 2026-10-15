"""Joint-profile scoring, committed proposal ledgers, and external evaluation.

Predicted probabilities rank proposals. Only the private exact oracle or hidden
recorded measurements supply truth, after the proposal/ranking files are sealed.
"""

import csv
import hashlib
import json
import math
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from scipy.integrate import quad
from scipy.special import ndtr

from .chem import decode_graph, smiles_to_graph
from .contracts import GraphBatch, GraphMasks, Latents
from .redesign_contracts import FlowCondition
from .redesign_sampling import SamplingPlan, frozen_deployment, sample_redesign_latents
from .sampling import model_device, sample_latents


def _array(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def joint_profile_probability(mean, covariance, baseline, goal):
    """P(delta1 >= kappa, baseline2+delta2 >= tau) under a joint Gaussian.

    Accept either one distribution or leading batch dimensions. Deterministic
    one-dimensional quadrature integrates the conditional normal distribution;
    full covariance is used and no scoring randomness changes the rankings.
    """
    mean, covariance, baseline, goal = map(_array, (mean, covariance, baseline, goal))
    if mean.shape[-1:] != (2,) or covariance.shape[-2:] != (2, 2):
        raise ValueError("mean/covariance must end in [2] and [2,2]")
    if baseline.shape[-1:] != (2,) or goal.shape[-1:] != (2,):
        raise ValueError("baseline and goal must end in [2]")
    shape = np.broadcast_shapes(mean.shape[:-1], covariance.shape[:-2],
                                baseline.shape[:-1], goal.shape[:-1])
    mean = np.broadcast_to(mean, shape + (2,)).reshape(-1, 2)
    covariance = np.broadcast_to(covariance, shape + (2, 2)).reshape(-1, 2, 2)
    baseline = np.broadcast_to(baseline, shape + (2,)).reshape(-1, 2)
    goal = np.broadcast_to(goal, shape + (2,)).reshape(-1, 2)
    if not all(np.isfinite(item).all() for item in (mean, covariance, baseline, goal)):
        raise ValueError("joint-profile parameters must be finite")
    if not np.allclose(covariance, covariance.swapaxes(-1, -2), rtol=1e-6, atol=1e-8):
        raise ValueError("response covariance must be symmetric")
    try:
        np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as error:
        raise ValueError("response covariance must be positive definite") from error
    scales = np.sqrt(np.diagonal(covariance, axis1=-2, axis2=-1))
    thresholds = np.stack((goal[:, 0], goal[:, 1] - baseline[:, 1]), axis=-1)
    standardized = (thresholds - mean) / scales
    probabilities = np.empty(len(mean), dtype=np.float64)
    for index, ((a, b), matrix, scale) in enumerate(zip(standardized, covariance, scales)):
        correlation = float(matrix[0, 1] / (scale[0] * scale[1]))
        correlation = float(np.clip(correlation, -1.0 + 1e-12, 1.0 - 1e-12))
        # Integrate along the more restrictive marginal to improve tail accuracy.
        if a < b:
            a, b = b, a
        if a >= 12.0:
            probability = 0.0  # omitted probability is below 2e-33
        elif abs(correlation) < 1e-12:
            probability = float(ndtr(-a) * ndtr(-b))
        else:
            conditional_scale = math.sqrt(1.0 - correlation * correlation)

            def integrand(x):
                return (math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
                        * float(ndtr((correlation * x - b) / conditional_scale)))

            lower = max(float(a), -12.0)
            transition = b / correlation
            width = conditional_scale / abs(correlation)
            points = sorted({transition + multiple * width for multiple in (-8, -2, 0, 2, 8)
                             if lower < transition + multiple * width < 12.0})
            probability, _ = quad(integrand, lower, 12.0, points=points,
                                  epsabs=1e-9, epsrel=1e-8, limit=150)
        probabilities[index] = float(np.clip(probability, 0.0, 1.0))
    result = probabilities.reshape(shape)
    return float(result) if not shape else result


def top_k_candidates(ledger: list[dict], k: int = 16) -> list[dict]:
    """Choose distinct eligible identities; ties use canonical SMILES then ID."""
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    seen = set()
    candidates = []
    for row in ledger:
        if not row.get("eligible") or row.get("smiles") in seen:
            continue
        score = row.get("profile_score")
        if score is None or not math.isfinite(float(score)):
            continue
        seen.add(row["smiles"])
        candidates.append(row)
    candidates.sort(key=lambda row: (-float(row["profile_score"]), row["smiles"],
                                      row["attempt_id"]))
    selected = candidates[:k]
    ranks = {row["attempt_id"]: rank for rank, row in enumerate(selected, 1)}
    for row in ledger:
        row["selected_rank"] = ranks.get(row["attempt_id"])
    return [{"rank": rank, "attempt_id": row["attempt_id"], "smiles": row["smiles"],
             "profile_score": row["profile_score"]}
            for rank, row in enumerate(selected, 1)]


def _json_value(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False) if isinstance(value, (list, dict)) else value


def _write_json(path: Path, value) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: _json_value(value) for key, value in row.items()} for row in rows)


def _synchronize(device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _active_finite(latents, masks):
    return (((torch.isfinite(latents.nodes).all(-1) | ~masks.nodes).all(-1))
            & ((torch.isfinite(latents.pairs).all(-1) | ~masks.pairs).all(-1)))


def _candidate_batch(smiles, max_nodes, device) -> GraphBatch:
    records = [smiles_to_graph(identity, max_nodes=max_nodes) for identity in smiles]
    atoms = torch.as_tensor(np.stack([row["atoms"] for row in records]), dtype=torch.long,
                            device=device)
    bonds = torch.as_tensor(np.stack([row["bonds"] for row in records]), dtype=torch.long,
                            device=device)
    counts = torch.tensor([row["count"] for row in records], device=device)
    return GraphBatch(atoms, bonds, GraphMasks.from_counts(counts, max_nodes), tuple(smiles))


def _assert_shared_representation(scorer, generator) -> None:
    """Paired architecture runs use identical frozen E and normalization."""
    for name in ("encoder", "normalizer"):
        left, right = getattr(scorer, name, None), getattr(generator, name, None)
        if left is None and right is None:
            continue  # permits parameterized operational test oracles
        if left is None or right is None:
            raise ValueError(f"proposal and scoring models must share frozen {name}")
        left_state, right_state = left.state_dict(), right.state_dict()
        if left_state.keys() != right_state.keys():
            raise ValueError(f"proposal and scoring models have different {name} states")
        if any(left_state[key].shape != right_state[key].shape
               or left_state[key].dtype != right_state[key].dtype
               or not torch.equal(left_state[key].detach().cpu(), right_state[key].detach().cpu())
               for key in left_state):
            raise ValueError(f"proposal and scoring models must share frozen {name} values")


@torch.no_grad()
def generate_redesign_proposals(model, condition: FlowCondition, plan: SamplingPlan,
                                lead_smiles: str, reserved_support_smiles,
                                output_dir, *, neighbourhood_fn,
                                batch_size: int = 128, nfe: int = 32,
                                T: float = 0.95, k: int = 16,
                                proposal_model=None, proposal_condition=None,
                                flow_state_override=None,
                                base_only: bool = False,
                                eligible_library_smiles=None,
                                metadata: dict | None = None) -> tuple[list[dict], list[dict], dict]:
    """Generate and commit a campaign proposal/ranking ledger without an oracle.

    A standard pretrained or separately trained redesign ``proposal_model`` can
    supply proposals with the same count/noise plan and native response scorer.
    A redesign generator receives its own explicit ``proposal_condition`` and
    inferred evidence state. ``flow_state_override``
    changes only evidence entering our flow, keeping the native evidence state
    in the common predictor. Support-matching refers to the entire reserved
    largest nested support pool, shared at every N.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if condition.baseline.shape != (1, 2) or condition.goal.shape != (1, 2):
        raise ValueError("generation requires a single lead condition")
    if base_only and proposal_model is not None:
        raise ValueError("base_only and an external standard proposal model are separate controls")
    if proposal_condition is not None and proposal_model is None:
        raise ValueError("proposal_condition requires an explicit proposal_model")
    if len(plan.counts) < 1 or plan.noise.nodes.shape[0] != len(plan.counts) or plan.noise.pairs.shape[0] != len(plan.counts):
        raise ValueError("sampling plan must contain one Gaussian draw per attempt")
    device = model_device(model)
    condition = condition.to(device)
    generator_model = model if proposal_model is None else proposal_model
    generator_is_redesign = hasattr(generator_model, "predict_clean")
    if proposal_model is not None and generator_is_redesign and proposal_condition is None:
        raise ValueError("a separate redesign proposal model requires its own explicit proposal_condition")
    if proposal_condition is not None and not generator_is_redesign:
        raise ValueError("legacy unconditional proposals do not take a proposal_condition")
    if not generator_is_redesign and flow_state_override is not None:
        raise ValueError("a legacy unconditional generator has no response state to override")
    flow_condition = condition if proposal_condition is None else proposal_condition.to(device)
    if flow_condition.baseline.shape != (1, 2):
        raise ValueError("proposal_condition requires a single lead")
    for name in ("lead_features", "baseline", "context", "goal"):
        if not torch.equal(getattr(flow_condition, name), getattr(condition, name)):
            raise ValueError(f"proposal and scoring conditions must have the same {name}")
    if flow_state_override is not None:
        state = torch.as_tensor(flow_state_override, device=device,
                                dtype=flow_condition.response_state.dtype)
        if state.shape != flow_condition.response_state.shape:
            raise ValueError("flow_state_override must match the native evidence state")
        flow_condition = replace(flow_condition, response_state=state)
    reserved = set(reserved_support_smiles)
    reserved.add(lead_smiles)
    library = None if eligible_library_smiles is None else set(eligible_library_smiles)
    if generator_model.config != model.config:
        raise ValueError("standard proposals must use the same graph and latent dimensions")
    if model_device(generator_model) != device:
        raise ValueError("proposal model and response scorer must be on the same device")
    if generator_model is not model:
        _assert_shared_representation(model, generator_model)
    seen = set()
    chemistry_cache = {}
    score_cache = {}
    ledger = []
    generation_seconds = 0.0
    score_seconds = 0.0
    with ExitStack() as stack:
        stack.enter_context(frozen_deployment(model))
        if generator_model is not model:
            stack.enter_context(frozen_deployment(generator_model))
        for offset in range(0, len(plan.counts), batch_size):
            size = min(batch_size, len(plan.counts) - offset)
            _synchronize(device)
            started = perf_counter()
            counts = plan.counts[offset:offset + size]
            noise = Latents(plan.noise.nodes[offset:offset + size],
                            plan.noise.pairs[offset:offset + size])
            if generator_is_redesign:
                sampled = sample_redesign_latents(generator_model, counts, flow_condition.expand(size),
                                                  nfe=nfe, T=T, initial_noise=noise,
                                                  base_only=base_only)
            else:
                sampled = sample_latents(generator_model, counts, nfe=nfe, T=T,
                                         initial_noise=noise)
            logits = generator_model.decode(sampled.latents, sampled.masks)
            finite_latents = _active_finite(sampled.latents, sampled.masks).cpu()
            finite_logits = (((torch.isfinite(logits.atoms).all(-1) | ~sampled.masks.nodes).all(-1))
                             & ((torch.isfinite(logits.bonds).all(-1) | ~sampled.masks.pairs).all(-1))).cpu()
            atoms, bonds = logits.atoms.argmax(-1).cpu(), logits.bonds.argmax(-1).cpu()
            rows = []
            for index, count in enumerate(counts.tolist()):
                smiles, chemically_valid, connected, reason = None, False, False, None
                if not finite_latents[index]:
                    reason = "nonfinite_latents"
                elif not finite_logits[index]:
                    reason = "nonfinite_logits"
                else:
                    outcome = decode_graph(atoms[index], bonds[index], count,
                                           max_nodes=model.config.max_nodes)
                    smiles, chemically_valid, connected, reason = (outcome.smiles,
                        outcome.chemically_valid, outcome.connected, outcome.failure_reason)
                valid = chemically_valid and connected
                neighbour, supported = {}, False
                if valid:
                    if smiles not in chemistry_cache:
                        try:
                            smiles_to_graph(smiles, max_nodes=model.config.max_nodes)
                            chemistry_cache[smiles] = (neighbourhood_fn(smiles, lead_smiles), True, None)
                        except ValueError as error:
                            chemistry_cache[smiles] = ({}, False, f"unsupported_chemistry:{error}")
                    neighbour, supported, unsupported_reason = chemistry_cache[smiles]
                    if unsupported_reason:
                        reason = unsupported_reason
                duplicate = smiles in seen if valid else None
                support_match = smiles in reserved if valid else False
                lead_match = smiles == lead_smiles if valid else False
                local = bool(neighbour.get("eligible", False)) if valid else False
                library_match = None if library is None else bool(valid and smiles in library)
                if valid:
                    seen.add(smiles)
                rows.append({
                    "attempt_id": offset + index,
                    "count_seed": plan.count_seed, "noise_seed": plan.noise_seed,
                    "lead_atom_count": plan.lead_count, "atom_count": count,
                    "nfe": nfe, "T": T,
                    "atoms": atoms[index].tolist(), "bonds": bonds[index].tolist(),
                    "smiles": smiles, "chemically_valid": chemically_valid,
                    "connected": connected, "supported_chemistry": supported,
                    "failure_reason": reason,
                    "duplicate": duplicate, "reserved_support_match": support_match,
                    "lead_match": lead_match, "within_neighbourhood": local,
                    "lead_similarity": neighbour.get("similarity"),
                    "lead_atom_count_difference": neighbour.get("count_difference"),
                    "in_measured_library": library_match,
                    "eligible": bool(valid and supported and not duplicate and not support_match and not lead_match and local
                                     and (library is None or library_match)),
                    "response_mean": None, "response_covariance": None,
                    "profile_score": None, "scoring_failure": None,
                    "selected_rank": None,
                })
            pending = list(dict.fromkeys(row["smiles"] for row in rows
                                        if row["chemically_valid"] and row["connected"]
                                        and row["supported_chemistry"]
                                        and row["smiles"] not in score_cache))
            _synchronize(device)
            scoring_started = perf_counter()
            if pending:
                try:
                    candidates = _candidate_batch(pending, model.config.max_nodes, device)
                    features = model.graph_features(candidates)
                    scorer = condition.expand(len(pending))
                    response = model.predict_response_features(scorer.lead_features, features,
                                                                scorer.baseline, scorer.context,
                                                                scorer.response_state)
                    means = _array(response.mean)
                    covariance = _array(response.covariance_matrix)
                    for index, smiles in enumerate(pending):
                        try:
                            probability = joint_profile_probability(means[index], covariance[index],
                                                                    condition.baseline[0], condition.goal[0])
                            score_cache[smiles] = (means[index].tolist(), covariance[index].tolist(), probability, None)
                        except ValueError as error:
                            score_cache[smiles] = (None, None, None, str(error))
                except (ValueError, RuntimeError) as error:
                    reason = f"response_scoring_failed:{type(error).__name__}:{error}"
                    for smiles in pending:
                        score_cache[smiles] = (None, None, None, reason)
            for row in rows:
                if row["smiles"] in score_cache:
                    mean, covariance, score, failure = score_cache[row["smiles"]]
                    row.update(response_mean=mean, response_covariance=covariance,
                               profile_score=score, scoring_failure=failure)
                    if failure:
                        row["eligible"] = False
            _synchronize(device)
            score_seconds += perf_counter() - scoring_started
            elapsed = perf_counter() - started
            generation_seconds += elapsed
            for row in rows:
                row["batch_amortized_seconds"] = elapsed / size
            ledger.extend(rows)
    ranking = top_k_candidates(ledger, k=k)
    summary = {
        "attempts": len(ledger), "connected_valid_attempts": sum(row["chemically_valid"] and row["connected"] for row in ledger),
        "supported_connected_valid_attempts": sum(row["supported_chemistry"] for row in ledger),
        "distinct_connected_valid": len(seen), "eligible_distinct": sum(row["eligible"] for row in ledger),
        "selected_distinct": len(ranking), "allocated_selection_slots": k,
        "reserved_support_identities": len(reserved),
        "nfe": nfe, "T": T, "batch_size": batch_size,
        "count_seed": plan.count_seed, "noise_seed": plan.noise_seed,
        "generation_seconds": generation_seconds, "response_scoring_seconds": score_seconds,
        "sampling_plan_seconds": plan.creation_seconds,
        "end_to_end_proposal_seconds": plan.creation_seconds + generation_seconds,
        "timing_scope": "transport,terminal clean prediction,one decode,host chemistry,neighbourhood checks,response scoring,ledger; excludes file writing",
        "end_to_end_scope": "CPU conditional-count/Gaussian-noise draws plus generation_seconds; cached lead/support-state inference and file writing are separately accounted",
        "proposal_kind": ("calibrated" if proposal_model is None else
                           "separate_redesign" if generator_is_redesign else "standard_pretrained"),
        "proposal_variant": getattr(getattr(generator_model, "redesign_config", None), "variant", None),
        "flow_state_override": flow_state_override is not None,
        "base_only": base_only,
        "native_response_state_used_for_scoring": True,
        "measured_library_eligibility_required": library is not None,
        "metadata": metadata or {},
    }
    failures = {}
    for row in ledger:
        if row["failure_reason"]:
            failures[row["failure_reason"]] = failures.get(row["failure_reason"], 0) + 1
    summary["failure_counts"] = failures
    commit = commit_proposals(output_dir, ledger, ranking, metadata=summary)
    summary["attempts_sha256"] = commit["attempts_sha256"]
    summary["ranking_sha256"] = commit["ranking_sha256"]
    _write_json(Path(output_dir) / "generation_summary.json", summary)
    _write_csv(Path(output_dir) / "generation_summary.csv", [summary])
    return ledger, ranking, summary


def commit_proposals(output_dir, ledger: list[dict], ranking: list[dict],
                     metadata: dict | None = None) -> dict:
    """Seal predictions and selection before any private oracle call.

    Existing proposal files are never overwritten. Subsequent outcomes are saved
    separately, so the files whose hashes were committed remain unchanged.
    """
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    names = ("attempts.jsonl", "attempts.csv", "ranking.json", "proposal_commit.json")
    if any((destination / name).exists() for name in names):
        raise FileExistsError("proposal directory already contains a committed or partial run")
    _write_jsonl(destination / "attempts.jsonl", ledger)
    _write_csv(destination / "attempts.csv", ledger)
    _write_json(destination / "ranking.json", ranking)
    commit = {
        "schema": "emf.redesign.proposals.v1",
        "attempts": len(ledger), "selected": len(ranking),
        "attempts_sha256": hashlib.sha256((destination / "attempts.jsonl").read_bytes()).hexdigest(),
        "ranking_sha256": hashlib.sha256((destination / "ranking.json").read_bytes()).hexdigest(),
        "oracle_access_before_commit": False,
        "metadata": metadata or {},
    }
    _write_json(destination / "proposal_commit.json", commit)
    return commit


def load_committed_proposals(output_dir) -> tuple[list[dict], list[dict], dict]:
    """Verify the selection ledger before obtaining hidden evaluation outcomes."""
    destination = Path(output_dir)
    commit = json.loads((destination / "proposal_commit.json").read_text())
    for name, key in (("attempts.jsonl", "attempts_sha256"), ("ranking.json", "ranking_sha256")):
        digest = hashlib.sha256((destination / name).read_bytes()).hexdigest()
        if digest != commit[key]:
            raise ValueError(f"committed proposal file changed: {name}")
    ledger = [json.loads(line) for line in (destination / "attempts.jsonl").read_text().splitlines()]
    ranking = json.loads((destination / "ranking.json").read_text())
    return ledger, ranking, commit


def observed_profile_success(outcomes, baseline, goal) -> bool:
    """An incomplete or failed external test is an unsuccessful budget slot."""
    if outcomes is None:
        return False
    outcomes, baseline, goal = map(_array, (outcomes, baseline, goal))
    if outcomes.shape != (2,) or not np.isfinite(outcomes).all():
        return False
    delta = outcomes - baseline
    return bool(delta[0] >= goal[0] and outcomes[1] >= goal[1])


def evaluate_committed_proposals(output_dir, outcome_reader, *, baseline, goal,
                                 support_size: int, k: int = 16,
                                 evaluation_kind: str = "exact_simulator") -> dict:
    """Reveal external truth once per distinct valid unmeasured proposal.

    ``outcome_reader(smiles)`` is private and never passed to generation/scoring.
    ``recorded_library`` uses only hidden recorded measurements; missing records
    return None and are failures. Full-pool oracle calls are extra evaluation
    costs, recorded separately from N support compounds and k selected slots.
    """
    if evaluation_kind not in {"exact_simulator", "recorded_library", "independent_measurement"}:
        raise ValueError("unknown external evaluation kind")
    if isinstance(support_size, bool) or not isinstance(support_size, int) or support_size < 1:
        raise ValueError("support_size must count distinct measured compounds including the lead")
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    destination = Path(output_dir)
    if (destination / "evaluation_outcomes.jsonl").exists():
        raise FileExistsError("evaluation has already revealed hidden outcomes")
    ledger, ranking, commit = load_committed_proposals(destination)
    if len(ranking) > k:
        raise ValueError("committed selection exceeds allocated k")
    if evaluation_kind == "recorded_library":
        by_id = {row["attempt_id"]: row for row in ledger}
        if any(by_id[row["attempt_id"]].get("in_measured_library") is not True for row in ranking):
            raise ValueError("retrospective selection must be restricted to the measured library before ranking")
    baseline, goal = _array(baseline), _array(goal)
    if baseline.shape != (2,) or goal.shape != (2,) or not np.isfinite(baseline).all() or not np.isfinite(goal).all():
        raise ValueError("baseline and goal must be finite [2] vectors")
    by_smiles = {}
    selected_smiles = {row["smiles"] for row in ranking}
    for row in ledger:
        if (row.get("chemically_valid") and row.get("connected") and row.get("smiles")
                and row.get("supported_chemistry", True)
                and not row.get("reserved_support_match") and not row.get("lead_match")
                and (evaluation_kind != "recorded_library" or row.get("in_measured_library") is True)
                and (evaluation_kind != "independent_measurement" or row["smiles"] in selected_smiles)):
            by_smiles.setdefault(row["smiles"], row)
    started = perf_counter()
    evaluated = []
    success_by_smiles = {}
    calls = 0
    for smiles, row in by_smiles.items():
        calls += 1
        reason = None
        try:
            outcomes = outcome_reader(smiles)
            array = None if outcomes is None else _array(outcomes)
            if array is None or array.shape != (2,) or not np.isfinite(array).all():
                array = None
                reason = "missing_or_incomplete_external_outcome"
        except (ValueError, RuntimeError, TypeError, KeyError) as error:
            array = None
            reason = f"external_test_failed:{type(error).__name__}"
        success = observed_profile_success(array, baseline, goal)
        success_by_smiles[smiles] = success
        evaluated.append({"smiles": smiles, "attempt_id": row["attempt_id"],
                          "outcomes": None if array is None else array.tolist(),
                          "delta": None if array is None else (array - baseline).tolist(),
                          "profile_success": success,
                          "within_neighbourhood": bool(row.get("within_neighbourhood")),
                          "profile_score": row.get("profile_score"),
                          "evaluation_failure": reason})
    selected_successes = sum(success_by_smiles.get(row["smiles"], False) for row in ranking)
    successes = sum(row["profile_success"] for row in evaluated)
    local_successes = sum(row["profile_success"] and row["within_neighbourhood"] for row in evaluated)
    attempts = len(ledger)
    complete = sum(row["outcomes"] is not None for row in evaluated)
    selected_complete = sum(next((row["outcomes"] is not None for row in evaluated
                                  if row["smiles"] == selected["smiles"]), False)
                            for selected in ranking)
    summary = {
        "evaluation_kind": evaluation_kind,
        "attempts": attempts, "support_size": support_size, "allocated_selection_slots": k,
        "selected_distinct": len(ranking), "unfilled_selection_slots": k - len(ranking),
        "selected_joint_profile_successes": selected_successes,
        "success": selected_successes / k,
        "efficiency": selected_successes / (support_size + k),
        "allocated_compound_budget": support_size + k,
        "actual_selected_compounds_tested": len(ranking),
        "actual_selected_complete_outcomes": selected_complete,
        "actual_selected_endpoint_measurements": 2 * selected_complete,
        "external_outcome_reader_calls": calls,
        "full_pool_outcome_reader_calls": calls if evaluation_kind != "independent_measurement" else 0,
        "full_pool_complete_outcomes": complete,
        "full_pool_endpoint_measurements": 2 * complete,
        "full_pool_joint_profile_successes": successes,
        "pre_ranking_unmeasured_joint_profile_yield_per_attempt": (successes / attempts if attempts else 0.0)
            if evaluation_kind != "independent_measurement" else None,
        "pre_ranking_neighbourhood_joint_profile_yield_per_attempt": (local_successes / attempts if attempts else 0.0)
            if evaluation_kind != "independent_measurement" else None,
        "external_evaluation_seconds": perf_counter() - started,
        "oracle_access_after_proposal_commit": True,
        "attempts_sha256": commit["attempts_sha256"],
        "ranking_sha256": commit["ranking_sha256"],
        "metadata": commit.get("metadata", {}),
    }
    valid_scores = [(float(row["profile_score"]), float(row["profile_success"]))
                    for row in evaluated if row["profile_score"] is not None and row["outcomes"] is not None]
    summary["joint_event_brier_score"] = (float(np.mean([(score - truth) ** 2 for score, truth in valid_scores]))
                                           if valid_scores else None)
    log_likelihoods = []
    for outcome in evaluated:
        row = by_smiles[outcome["smiles"]]
        if outcome["delta"] is None or row.get("response_mean") is None or row.get("response_covariance") is None:
            continue
        covariance = _array(row["response_covariance"])
        residual = _array(outcome["delta"]) - _array(row["response_mean"])
        sign, logdet = np.linalg.slogdet(covariance)
        if sign <= 0:
            continue
        log_likelihoods.append(-math.log(2 * math.pi) - 0.5 * logdet
                               - 0.5 * float(residual @ np.linalg.solve(covariance, residual)))
    summary["generated_pool_response_mean_log_likelihood"] = (float(np.mean(log_likelihoods))
                                                               if log_likelihoods else None)
    bins = []
    for index in range(10):
        low, high = index / 10, (index + 1) / 10
        values = [(score, truth) for score, truth in valid_scores
                  if low <= score and (score < high or index == 9 and score <= high)]
        bins.append({"lower": low, "upper": high, "count": len(values),
                     "mean_predicted_probability": float(np.mean([item[0] for item in values])) if values else None,
                     "observed_success_fraction": float(np.mean([item[1] for item in values])) if values else None})
    _write_jsonl(destination / "evaluation_outcomes.jsonl", evaluated)
    _write_csv(destination / "evaluation_outcomes.csv", evaluated)
    _write_json(destination / "evaluation_summary.json", summary)
    _write_csv(destination / "evaluation_summary.csv", [summary])
    _write_json(destination / "joint_event_calibration.json", bins)
    return summary
