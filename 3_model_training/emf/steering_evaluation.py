"""Blind paired-evidence evaluation for the response-reversal experiment.

Each sign pair shares a lead, a nested support pool, count draws and Gaussian
draws.  All proposals and rankings are committed before loading the private
oracle.  Opposite-evidence ledgers replay the exact paired native condition;
their logical attempt count is retained and their physical sampling cost is zero.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr, t as student_t

from .campaigns import neighbourhood
from .data import collate_graphs
from .redesign_contracts import FlowCondition, SupportBatch
from .redesign_evaluation import (commit_proposals, evaluate_committed_proposals,
                                  generate_redesign_proposals, load_committed_proposals,
                                  _write_csv, _write_json)
from .redesign_sampling import frozen_deployment, make_sampling_plan
from .redesign_train import cache_graph_features, digest, project_path, tensor_fingerprint


def _value(config, name, default):
    return getattr(config, name, default)


def _support(records, features, device):
    indices = torch.as_tensor(records["indices"], dtype=torch.long)
    size = len(indices)
    return SupportBatch(
        features.index_select(0, indices)[None].to(device),
        torch.as_tensor(records["values"], dtype=torch.float32, device=device)[None],
        torch.as_tensor(records["context"], dtype=torch.float32, device=device)[None],
        torch.ones((1, size), dtype=torch.bool, device=device),
    )


def _load_model(path, device):
    # The steering runner owns checkpoint serialization; import here so its
    # evaluation entry point can import this module without a cycle.
    try:
        from .steering_train import load_source_checkpoint
    except ImportError:
        from .redesign_train import load_source_checkpoint
    return load_source_checkpoint(path, device)


def _model_and_config(loaded):
    if not isinstance(loaded, tuple) or len(loaded) < 2:
        raise TypeError("source checkpoint loader must return (model, config, payload)")
    return loaded[0], loaded[1]


def _family_lead(dataset, family, family_number):
    leads = dataset.family_leads
    if isinstance(leads, dict):
        value = leads[family]
    else:
        value = leads[family_number]
    if isinstance(value, dict):
        value = value.get("lead_index", value.get("index"))
    return int(value)


def _campaign_groups(dataset):
    """Return deterministic (family, campaign, opposite, lead) cases."""
    names = tuple(dataset.campaign_ids)
    if len(names) != len(set(names)):
        raise ValueError("duplicate campaign identities")
    index = {name: position for position, name in enumerate(names)}
    partners = dataset.campaign_pairs
    families = dataset.campaign_family
    family_order = list(dict.fromkeys(families[name] for name in names))
    cases = []
    pairs = []
    seen = set()
    for family_number, family in enumerate(family_order):
        lead = _family_lead(dataset, family, family_number)
        for name in names:
            if families[name] != family or name in seen:
                continue
            other = partners[name]
            if other not in index or partners.get(other) != name or families[other] != family:
                raise ValueError("campaign pair must be symmetric within one family")
            seen.update((name, other))
            pair_number = len(pairs)
            pair = {"family": family, "family_number": family_number,
                    "pair_number": pair_number, "lead_index": lead,
                    "campaigns": [index[name], index[other]]}
            pairs.append(pair)
            for campaign, opposite in ((index[name], index[other]),
                                       (index[other], index[name])):
                cases.append({**pair, "campaign": campaign, "opposite": opposite})
    if len(seen) != len(names):
        raise ValueError("not every campaign belongs to a sign pair")
    return cases, pairs


def _case_seed(eval_seed, family_number, pair_number):
    # Only chemistry family and magnitude-pair identity enter these seeds.
    # Campaign sign, support N, model variant and source seed do not.
    return int(eval_seed + 100_000 * family_number + 17 * pair_number)


def _validate_shared_case(first, second):
    for key in ("reserve_indices", "reserve_ids", "reserve_smiles"):
        if list(first[key]) != list(second[key]):
            raise ValueError(f"paired campaigns differ in {key}")
    if first["lead"]["smiles"] != second["lead"]["smiles"]:
        raise ValueError("paired campaigns have different leads")
    if not np.array_equal(first["lead_values"], second["lead_values"]):
        raise ValueError("paired campaigns have different baselines")
    for size in first["records"]:
        if list(first["records"][size]["indices"]) != list(second["records"][size]["indices"]):
            raise ValueError("paired campaign support order differs")


def _replay_committed(source, destination, metadata):
    ledger, ranking, original = load_committed_proposals(source)
    destination = Path(destination)
    replay_summary = dict(original.get("metadata", {}))
    replay_summary.update(metadata=metadata, proposal_kind="paired_native_replay",
                          physical_attempts=0, logical_attempts=len(ledger),
                          physical_generation_seconds=0.0,
                          replayed_source_generation_seconds=replay_summary.get("generation_seconds"),
                          generation_seconds=0.0, response_scoring_seconds=0.0,
                          sampling_plan_seconds=0.0, end_to_end_proposal_seconds=0.0,
                          replay_source=str(Path(source).resolve()),
                          replay_source_attempts_sha256=original["attempts_sha256"])
    commit = commit_proposals(destination, ledger, ranking, metadata=replay_summary)
    replay_summary.update(attempts_sha256=commit["attempts_sha256"],
                          ranking_sha256=commit["ranking_sha256"])
    _write_json(destination / "generation_summary.json", replay_summary)
    _write_csv(destination / "generation_summary.csv", [replay_summary])


def _paired_metrics(ledger, outcomes, coefficient=None):
    """Count all successful slots; retain distinct discovery as a separate view."""
    truth = {row["smiles"]: row for row in outcomes}
    local_hits = profile_hits = 0
    distinct_local_hits = distinct_profile_hits = 0
    valid = unique = local = 0
    measured_du = []
    for row in ledger:
        usable = bool(row.get("chemically_valid") and row.get("connected")
                      and row.get("supported_chemistry"))
        valid += usable
        distinct = usable and not row.get("duplicate")
        unique += distinct
        local += bool(distinct and row.get("within_neighbourhood"))
        eligible = usable and not row.get("reserved_support_match") and not row.get("lead_match")
        response = truth.get(row.get("smiles")) if eligible else None
        success = bool(response and response.get("profile_success"))
        profile_hits += success
        local_hits += bool(success and row.get("within_neighbourhood"))
        distinct_profile_hits += bool(success and distinct)
        distinct_local_hits += bool(success and distinct and row.get("within_neighbourhood"))
        if response and response.get("delta") is not None:
            if coefficient is not None and coefficient != 0:
                measured_du.append(response["delta"][0] / coefficient)
    total = len(ledger)
    return {"attempts": total, "connected_valid_attempts": valid,
            "distinct_valid_attempts": unique, "local_distinct_attempts": local,
            "profile_success_attempts": distinct_profile_hits,
            "local_profile_success_attempts": distinct_local_hits,
            "raw_slot_profile_success_attempts": profile_hits,
            "raw_slot_local_profile_success_attempts": local_hits,
            "profile_yield_per_attempt": distinct_profile_hits / total if total else 0.0,
            "local_profile_yield_per_attempt": distinct_local_hits / total if total else 0.0,
            "raw_slot_profile_yield_per_attempt": profile_hits / total if total else 0.0,
            "raw_slot_local_profile_yield_per_attempt": local_hits / total if total else 0.0,
            "mean_generated_u_change": float(np.mean(measured_du)) if measured_du else None}


@torch.no_grad()
def _withheld_response_metrics(scorer, dataset, features, nested, campaign,
                               evidence_campaign, family, size, device):
    """Reveal reserved-pool-excluded library labels only after blind commits."""
    withheld = np.setdiff1d(dataset.family_indices[family], nested["reserve_indices"])
    if not len(withheld):
        return {"withheld_library_candidates": 0,
                "response_first_endpoint_sign_accuracy": None,
                "response_first_endpoint_spearman": None,
                "response_mean_log_likelihood": None}
    evidence_records = dict(nested["records"][size])
    evidence_records["values"] = dataset.observations[evidence_campaign,
                                                        evidence_records["indices"]]
    support = _support(evidence_records, features, device)
    state = scorer.support_state(support)
    lead = collate_graphs([nested["lead"]]).to(device)
    lead_features = scorer.graph_features(lead).expand(len(withheld), -1)
    candidate_features = features.index_select(0, torch.as_tensor(withheld, dtype=torch.long)).to(device)
    baseline = torch.as_tensor(nested["lead_values"], dtype=torch.float32,
                               device=device)[None].expand(len(withheld), -1)
    context = torch.as_tensor(dataset.context, dtype=torch.float32,
                              device=device)[None].expand(len(withheld), -1)
    response = scorer.predict_response_features(
        lead_features, candidate_features, baseline, context,
        state.expand(len(withheld), -1))
    means = response.mean.detach().cpu().numpy().astype(np.float64)
    covariance = response.covariance_matrix.detach().cpu().numpy().astype(np.float64)
    truth = (dataset.observations[campaign, withheld].astype(np.float64)
             - np.asarray(nested["lead_values"], dtype=np.float64))
    nonzero = np.abs(truth[:, 0]) > 1e-10
    sign_accuracy = (float(np.mean(np.sign(means[nonzero, 0]) == np.sign(truth[nonzero, 0])))
                     if nonzero.any() else None)
    correlation = spearmanr(means[:, 0], truth[:, 0]).statistic if len(withheld) > 1 else np.nan
    likelihoods = []
    for predicted, matrix, observed in zip(means, covariance, truth):
        sign, logdet = np.linalg.slogdet(matrix)
        if sign <= 0 or not np.isfinite(matrix).all():
            continue
        residual = observed - predicted
        likelihoods.append(float(-np.log(2 * np.pi) - .5 * logdet
                                 - .5 * residual @ np.linalg.solve(matrix, residual)))
    return {"withheld_library_candidates": len(withheld),
            "withheld_library_outcome_reads": len(withheld),
            "response_first_endpoint_sign_accuracy": sign_accuracy,
            "response_first_endpoint_spearman": float(correlation) if np.isfinite(correlation) else None,
            "response_mean_log_likelihood": float(np.mean(likelihoods)) if likelihoods else None}


def _paired_identity_metrics(first, second):
    a = [row.get("smiles") for row in first]
    b = [row.get("smiles") for row in second]
    if len(a) != len(b):
        raise ValueError("paired ledgers have different attempted slot counts")
    changed = sum(x != y for x, y in zip(a, b))
    return {"paired_decoded_changed_attempts": changed,
            "paired_decoded_changed_fraction": changed / len(a) if a else 0.0,
            "paired_both_valid_same_identity": sum(x == y and x is not None for x, y in zip(a, b))}


def run_evaluation(config, checkpoint, split="validation", output_dir=None, *,
                   scorer_checkpoint=None, oracle=None, flow_state_override=None,
                   model_loader=None, proposal_setup=None, protocol_extra=None,
                   support_state_override=None):
    """Evaluate all declared family/sign cases with fixed plans and blind commits.

    ``flow_state_override(model, inferred_state, campaign_id, dataset)`` is reserved for a
    separately labelled privileged diagnostic. It changes only the generator's
    evidence state; the common r8 scorer always sees measured support.
    """
    if split not in {"validation", "test"}:
        raise ValueError("split must be validation or test")
    if output_dir is None:
        output_dir = Path(_value(config, "output_dir", "")) / f"evaluation_{split}"
    output = project_path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(_value(config, "device", "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("evaluation CUDA requires an allocated usable GPU")
    torch.set_num_threads(int(_value(config, "cpu_threads", 1)))
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    loader = model_loader or _load_model
    generator, source_config = _model_and_config(loader(checkpoint, device))
    scorer_path = scorer_checkpoint or _value(config, "common_scorer_checkpoint", None) or checkpoint
    if Path(project_path(checkpoint)).resolve() == Path(project_path(scorer_path)).resolve():
        scorer = generator
        scorer_config = source_config
    else:
        scorer, scorer_config = _model_and_config(loader(scorer_path, device))
    scorer_redesign = _value(scorer_config, "redesign", scorer_config)
    if (_value(scorer_redesign, "variant", None) != "factorized"
            or _value(scorer_redesign, "rank", None) != 8):
        raise ValueError("common response scorer must be a factorized r8 checkpoint")
    from .steering_data import SteeringDataset, SteeringOracle
    dataset = SteeringDataset(project_path(config.data_dir), split)
    sizes = tuple(sorted(set(int(n) for n in _value(config, "support_sizes", (4, 16, 64)))))
    if not sizes or sizes[0] < 1:
        raise ValueError("support sizes must be positive")
    maximum_support = max(_value(dataset, "support_sizes", sizes))
    if max(sizes) > maximum_support:
        raise ValueError("requested support exceeds the declared maximum reserve")
    nested_sizes = tuple(sorted(set((*sizes, maximum_support))))
    attempts = int(_value(config, "num_attempts", 256))
    nfe = int(_value(config, "nfe", 32))
    k = int(_value(config, "k", 16))
    if min(attempts, k) < 1 or nfe < 2:
        raise ValueError("invalid attempt, rank or NFE budget")
    cases, pairs = _campaign_groups(dataset)
    features, cache_seconds = cache_graph_features(
        scorer, dataset.graph_dataset, batch_size=int(_value(config, "cache_batch_size", 512)),
        bf16=bool(_value(config, "bf16", True)))
    protocol = {
        "schema": "emf.steering.evaluation.v1", "split": split,
        "campaigns": list(dataset.campaign_ids), "cases": cases,
        "support_sizes": list(sizes), "num_attempts": attempts, "nfe": nfe,
        "reserved_support_size": maximum_support,
        "T": float(_value(config, "T", .95)), "k": k,
        "eval_seed": int(_value(config, "eval_seed", 271828)),
        "generator_checkpoint_sha256": digest(project_path(checkpoint)),
        "common_r8_scorer_checkpoint_sha256": digest(project_path(scorer_path)),
        "data_manifest_sha256": digest(project_path(config.data_dir) / "manifest.json"),
        "count_prior": np.asarray(dataset.count_prior).tolist(),
        "feature_cache_seconds": cache_seconds,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "privileged_diagnostic": flow_state_override is not None,
    }
    if protocol_extra is not None:
        protocol["extension"] = protocol_extra
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        previous = json.loads(protocol_path.read_text())
        stable = lambda record: {key: value for key, value in record.items()
                                 if key not in {"feature_cache_seconds", "created_utc"}}
        if stable(previous) != stable(protocol):
            raise FileExistsError("evaluation protocol differs; use a fresh output directory")
        protocol = previous
    else:
        _write_json(protocol_path, protocol)
    pending = []
    plans = {}
    nested_cache = {}
    before = tensor_fingerprint(scorer.state_dict().items())
    generator_before = tensor_fingerprint(generator.state_dict().items())
    with frozen_deployment(scorer):
        generator_scope = frozen_deployment(generator) if generator is not scorer else None
        if generator_scope is not None:
            generator_scope.__enter__()
        try:
            for pair in pairs:
                first, second = pair["campaigns"]
                seed = _case_seed(protocol["eval_seed"], pair["family_number"], pair["pair_number"])
                support_seed = (dataset.paired_support_seed(dataset.campaign_ids[first])
                                if hasattr(dataset, "paired_support_seed") else seed)
                left = dataset.nested_support(first, pair["lead_index"], seed=support_seed, sizes=nested_sizes)
                right = dataset.nested_support(second, pair["lead_index"], seed=support_seed, sizes=nested_sizes)
                _validate_shared_case(left, right)
                nested_cache[first], nested_cache[second] = left, right
                lead = collate_graphs([left["lead"]]).to(device)
                with torch.no_grad():
                    lead_features = scorer.graph_features(lead)
                plan = make_sampling_plan(scorer.config, dataset.count_prior, int(lead.counts[0]),
                                          attempts, count_seed=seed + 1, noise_seed=seed + 2)
                plans[first] = plans[second] = plan
                for size in sizes:
                    for campaign in (first, second):
                        nested = nested_cache[campaign]
                        campaign_id = dataset.campaign_ids[campaign]
                        support = _support(nested["records"][size], features, device)
                        baseline = torch.as_tensor(nested["lead_values"], dtype=torch.float32, device=device)[None]
                        context = torch.as_tensor(dataset.context, dtype=torch.float32, device=device)[None]
                        goal = torch.stack((torch.full_like(baseline[:, 0], .25),
                                            baseline[:, 1] - .10), dim=-1)
                        with torch.no_grad():
                            state = scorer.support_state(support)
                            generator_state = generator.support_state(support)
                        condition = FlowCondition(lead_features, baseline, context, state, goal)
                        generator_condition = FlowCondition(lead_features, baseline, context,
                                                            generator_state, goal)
                        if support_state_override is not None:
                            from dataclasses import replace
                            generator_condition = replace(generator_condition, response_state=
                                support_state_override(generator, generator_state, support,
                                                       campaign_id, dataset))
                        kwargs: dict[str, Any] = {}
                        if generator is not scorer:
                            kwargs.update(proposal_model=generator,
                                          proposal_condition=generator_condition)
                        if flow_state_override is not None:
                            overridden = flow_state_override(generator, generator_state,
                                                             campaign_id, dataset)
                            if generator is scorer:
                                kwargs["flow_state_override"] = overridden
                            else:
                                from dataclasses import replace
                                kwargs["proposal_condition"] = replace(generator_condition,
                                                                        response_state=overridden)
                        proposal_plan = plan
                        if proposal_setup is not None:
                            if generator is scorer:
                                raise ValueError("proposal_setup requires a separate generator and scorer")
                            prepared_condition, proposal_plan = proposal_setup(
                                generator, kwargs["proposal_condition"], plan, lead,
                                campaign_id, dataset, scorer, condition)
                            kwargs["proposal_condition"] = prepared_condition
                        destination = output / f"pair_{pair['pair_number']:03d}" / f"campaign_{campaign:03d}" / f"support_{size:03d}" / "correct"
                        metadata = {"family": pair["family"], "pair_number": pair["pair_number"],
                                    "campaign": campaign_id, "evidence_campaign": campaign_id,
                                    "opposite_campaign": dataset.campaign_ids[second if campaign == first else first],
                                    "lead_id": left["lead"]["id"], "support_size": size,
                                    "support_ids": list(nested["reserve_ids"][:size]),
                                    "source_seed": _value(source_config, "seed", None),
                                    "source_variant": _value(_value(source_config, "redesign", source_config), "variant", None),
                                    "physical_attempts": attempts, "logical_attempts": attempts,
                                    "condition": "correct", "count_seed": plan.count_seed,
                                    "noise_seed": plan.noise_seed,
                                    "support_seed": support_seed}
                        if not (destination / "proposal_commit.json").exists():
                            generate_redesign_proposals(
                                scorer, condition, proposal_plan, nested["lead"]["smiles"],
                                nested["reserve_smiles"], destination,
                                neighbourhood_fn=neighbourhood,
                                batch_size=int(_value(config, "sample_batch_size", 128)),
                                nfe=nfe, T=protocol["T"], k=k, metadata=metadata, **kwargs)
                        pending.append((destination, campaign, size, metadata, baseline, goal))
                    # Exact replay: the opposite labels here are the paired
                    # campaign's correct labels on identical molecular records.
                    for campaign, opposite in ((first, second), (second, first)):
                        native = output / f"pair_{pair['pair_number']:03d}" / f"campaign_{opposite:03d}" / f"support_{size:03d}" / "correct"
                        destination = output / f"pair_{pair['pair_number']:03d}" / f"campaign_{campaign:03d}" / f"support_{size:03d}" / "opposite"
                        nested = nested_cache[campaign]
                        baseline = torch.as_tensor(nested["lead_values"], dtype=torch.float32, device=device)[None]
                        goal = torch.stack((torch.full_like(baseline[:, 0], .25),
                                            baseline[:, 1] - .10), dim=-1)
                        metadata = {"family": pair["family"], "pair_number": pair["pair_number"],
                                    "campaign": dataset.campaign_ids[campaign],
                                    "evidence_campaign": dataset.campaign_ids[opposite],
                                    "lead_id": nested["lead"]["id"], "support_size": size,
                                    "support_ids": list(nested["reserve_ids"][:size]),
                                    "source_seed": _value(source_config, "seed", None),
                                    "condition": "opposite", "physical_attempts": 0,
                                    "logical_attempts": attempts, "count_seed": plans[campaign].count_seed,
                                    "noise_seed": plans[campaign].noise_seed,
                                    "support_seed": support_seed}
                        if not (destination / "proposal_commit.json").exists():
                            _replay_committed(native, destination, metadata)
                        pending.append((destination, campaign, size, metadata, baseline, goal))
        finally:
            if generator_scope is not None:
                generator_scope.__exit__(None, None, None)
    if tensor_fingerprint(scorer.state_dict().items()) != before:
        raise RuntimeError("evaluation changed frozen scorer weights")
    if tensor_fingerprint(generator.state_dict().items()) != generator_before:
        raise RuntimeError("evaluation changed frozen generator weights")
    # This is the first point at which hidden response parameters are loaded.
    if oracle is None:
        oracle_path = project_path(config.data_dir) / "private" / "oracle.json"
        expected = dataset.manifest.get("private_oracle_sha256")
        if expected is not None and digest(oracle_path) != expected:
            raise ValueError("private oracle differs from data manifest")
        oracle = SteeringOracle.load(oracle_path)
    feasibility = {}
    feasibility_name = dataset.manifest.get("feasibility_file")
    if feasibility_name:
        feasibility_path = project_path(config.data_dir) / feasibility_name
        expected = dataset.manifest.get("feasibility_sha256")
        if expected is not None and digest(feasibility_path) != expected:
            raise ValueError("feasibility report differs from data manifest")
        feasibility = {case["campaign_id"]: case for case in
                       json.loads(feasibility_path.read_text())["cases"]
                       if case["split"] == split}
    truth_cache = {}
    previous_rows = {}
    if (output / "metrics.csv").exists():
        import csv
        with (output / "metrics.csv").open(newline="") as stream:
            previous_rows = {(row["campaign"], int(row["support_size"]), row["condition"]): row
                             for row in csv.DictReader(stream)}
    rows = []
    for destination, campaign, size, metadata, baseline, goal in pending:
        campaign_id = dataset.campaign_ids[campaign]
        earlier = previous_rows.get((campaign_id, size, metadata["condition"]))
        physical_calls = 0

        def read_truth(smiles):
            nonlocal physical_calls
            key = (campaign_id, smiles)
            if key not in truth_cache:
                truth_cache[key] = oracle.score_smiles(campaign_id, smiles)
                physical_calls += 1
            return truth_cache[key]

        already_evaluated = (destination / "evaluation_summary.json").exists()
        if not already_evaluated:
            evaluated = evaluate_committed_proposals(
                destination, read_truth,
                baseline=baseline.detach().cpu().numpy()[0],
                goal=goal.detach().cpu().numpy()[0], support_size=size, k=k)
        else:
            evaluated = json.loads((destination / "evaluation_summary.json").read_text())
        ledger, _, _ = load_committed_proposals(destination)
        outcomes = [json.loads(line) for line in (destination / "evaluation_outcomes.jsonl").read_text().splitlines()]
        coefficient = getattr(oracle, "coefficients", {}).get(campaign_id)
        metrics = _paired_metrics(ledger, outcomes, coefficient)
        family = dataset.campaign_family[campaign_id]
        evidence_campaign = dataset.campaign_ids.index(metadata["evidence_campaign"])
        with frozen_deployment(scorer):
            response_metrics = _withheld_response_metrics(
                scorer, dataset, features, nested_cache[campaign], campaign,
                evidence_campaign, family, size, device)
        declared = feasibility.get(campaign_id, {})
        if declared and list(declared["reserve_indices"]) != list(nested_cache[campaign]["reserve_indices"]):
            raise ValueError("evaluation support reserve differs from committed feasibility case")
        old_calls = int(earlier["physical_oracle_calls"]) if earlier else 0
        row = {**metadata, **metrics,
               **response_metrics,
               "feasible_library": declared.get("feasible_library"),
               "feasible_outside_max_support": declared.get("feasible_outside_max_support"),
               "feasible_unmeasured_stratum": (declared["feasible_outside_max_support"] > 0
                                               if declared else None),
               "top_k_success_per_slot": evaluated["success"],
               "allocated_selection_slots": k,
               "selected_joint_profile_successes": evaluated["selected_joint_profile_successes"],
               "logical_oracle_reader_calls": evaluated["external_outcome_reader_calls"],
               "physical_oracle_calls": old_calls + physical_calls,
               "physical_oracle_calls_this_invocation": physical_calls,
               "proposal_attempts_sha256": evaluated["attempts_sha256"],
               "proposal_ranking_sha256": evaluated["ranking_sha256"]}
        rows.append(row)
        _write_csv(output / "metrics.csv", rows)
    by_case = {(row["campaign"], row["support_size"], row["condition"]): row for row in rows}
    for pair in pairs:
        for campaign in pair["campaigns"]:
            campaign_id = dataset.campaign_ids[campaign]
            for size in sizes:
                correct = by_case[(campaign_id, size, "correct")]
                opposite = by_case[(campaign_id, size, "opposite")]
                for condition, row in (("correct", correct), ("opposite", opposite)):
                    destination = output / f"pair_{pair['pair_number']:03d}" / f"campaign_{campaign:03d}" / f"support_{size:03d}" / condition
                    ledger, _, _ = load_committed_proposals(destination)
                    other = output / f"pair_{pair['pair_number']:03d}" / f"campaign_{campaign:03d}" / f"support_{size:03d}" / ("opposite" if condition == "correct" else "correct")
                    paired_ledger, _, _ = load_committed_proposals(other)
                    row.update(_paired_identity_metrics(ledger, paired_ledger))
                benefit = correct["local_profile_yield_per_attempt"] - opposite["local_profile_yield_per_attempt"]
                correct["steering_benefit"] = opposite["steering_benefit"] = benefit
    _write_csv(output / "metrics.csv", rows)
    pair_effects = []
    for pair in pairs:
        for size in sizes:
            benefits = [by_case[(dataset.campaign_ids[campaign], size, "correct")]["steering_benefit"]
                        for campaign in pair["campaigns"]]
            pair_effects.append({"family": pair["family"],
                                 "pair_number": pair["pair_number"],
                                 "support_size": size,
                                 "mean_sign_target_benefit": float(np.mean(benefits)),
                                 "target_campaign_benefits": benefits})
    _write_csv(output / "pair_effects.csv", pair_effects)

    def _uncertainty(values):
        values = np.asarray(values, dtype=np.float64)
        mean = float(np.mean(values)) if len(values) else None
        if len(values) < 2:
            return {"mean_steering_benefit": mean, "pair_standard_error": None,
                    "pair_t95_lower": None, "pair_t95_upper": None}
        error = float(values.std(ddof=1) / np.sqrt(len(values)))
        margin = float(student_t.ppf(.975, df=len(values) - 1) * error)
        return {"mean_steering_benefit": mean, "pair_standard_error": error,
                "pair_t95_lower": mean - margin, "pair_t95_upper": mean + margin}

    family_curves = []
    for family in dict.fromkeys(pair["family"] for pair in pairs):
        for size in sizes:
            group = [row["mean_sign_target_benefit"] for row in pair_effects
                     if row["family"] == family and row["support_size"] == size]
            family_curves.append({"family": family, "support_size": size,
                                  "pairs": len(group), **_uncertainty(group)})
    _write_csv(output / "family_curves.csv", family_curves)
    aggregate = []
    for size in sizes:
        subset = [row for row in rows if row["support_size"] == size]
        effects = [row["mean_sign_target_benefit"] for row in pair_effects
                   if row["support_size"] == size]
        aggregate.append({"support_size": size, "campaigns": len(subset) // 2,
                          "independent_pairs": len(effects), **_uncertainty(effects),
                          "physical_sampling_attempts": sum(row["physical_attempts"] for row in subset),
                          "logical_sampling_attempts": sum(row["logical_attempts"] for row in subset),
                          "physical_oracle_calls": sum(row["physical_oracle_calls"] for row in subset)})
    report = {"phase": "evaluation_complete", "split": split,
              "campaigns": len(dataset.campaign_ids), "pairs": len(pairs),
              "support_sizes": list(sizes), "batches": len(rows),
              "aggregate": aggregate, "source_weights_unchanged": True,
              "protocol": str(protocol_path)}
    _write_json(output / "summary.json", report)
    return report
