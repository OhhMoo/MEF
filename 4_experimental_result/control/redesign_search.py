"""Few-shot response predictor guiding local molecular search.

This control uses the same frozen evidence-conditioned response scorer as the
flow. It enumerates graph mutations and records every attempted edit before
chemistry filtering. Beam updates use predicted profile probabilities only.
The private response oracle is deliberately absent from this module.
"""

from collections import Counter
import csv
from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from emf.campaigns import neighbourhood
from emf.chem import decode_graph, smiles_to_graph
from emf.data import collate_graphs
from emf.redesign_contracts import FlowCondition
from emf.redesign_evaluation import (commit_proposals, joint_profile_probability,
                                     top_k_candidates)
from emf.redesign_sampling import SamplingPlan, frozen_deployment
from emf.sampling import model_device


@dataclass(frozen=True)
class Mutation:
    atoms: np.ndarray
    bonds: np.ndarray
    count: int
    operation: str
    positions: tuple[int, ...]
    new_label: int | None


def enumerate_mutations(graph: dict, max_nodes: int = 9) -> list[Mutation]:
    """Enumerate categorical edits without valence/connectivity preselection."""
    count = int(graph["count"])
    atoms = np.asarray(graph["atoms"], dtype=np.uint8)
    bonds = np.asarray(graph["bonds"], dtype=np.uint8)
    left, right = np.triu_indices(max_nodes, k=1)
    adjacency = np.zeros((max_nodes, max_nodes), dtype=np.uint8)
    adjacency[left, right] = adjacency[right, left] = bonds
    mutations = []
    for atom in range(count):
        for label in range(4):
            if label != int(atoms[atom]):
                changed = atoms.copy()
                changed[atom] = label
                mutations.append(Mutation(changed, bonds.copy(), count,
                                          "atom_substitution", (atom,), label))
    for index, (i, j) in enumerate(zip(left, right)):
        if j >= count:
            continue
        for label in range(4):
            if label != int(bonds[index]):
                changed = bonds.copy()
                changed[index] = label
                mutations.append(Mutation(atoms.copy(), changed, count,
                                          "bond_change", (int(i), int(j)), label))
    if count < max_nodes:
        for attachment in range(count):
            for atom_label in range(4):
                for bond_label in (1, 2, 3):
                    changed_atoms, changed_adjacency = atoms.copy(), adjacency.copy()
                    changed_atoms[count] = atom_label
                    changed_adjacency[attachment, count] = changed_adjacency[count, attachment] = bond_label
                    mutations.append(Mutation(changed_atoms, changed_adjacency[left, right], count + 1,
                                              "terminal_atom_addition", (attachment, count), atom_label))
    if count > 1:
        degrees = (adjacency[:count, :count] > 0).sum(axis=1)
        for removed in np.flatnonzero(degrees == 1):
            retained = [index for index in range(count) if index != removed]
            changed_atoms = np.zeros_like(atoms)
            changed_atoms[:count - 1] = atoms[retained]
            changed_adjacency = np.zeros_like(adjacency)
            changed_adjacency[:count - 1, :count - 1] = adjacency[np.ix_(retained, retained)]
            mutations.append(Mutation(changed_atoms, changed_adjacency[left, right], count - 1,
                                      "terminal_atom_deletion", (int(removed),), None))
    return mutations


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def generate_predictor_search_proposals(model, condition: FlowCondition, plan: SamplingPlan,
                                        lead_smiles: str, reserved_support_smiles, output_dir,
                                        *, num_attempts: int = 1000, k: int = 16,
                                        batch_size: int = 128, metadata: dict | None = None):
    """Commit exactly the allocated mutation attempts and their predicted ranking.

    Fixed batches amortize frozen graph encoding and response prediction. The
    top 16 scored local parents form a beam, with the original lead retained as
    an exploration anchor. Parent catalogues are shuffled by ``noise_seed``.
    Exhaustion restarts a catalogue and records the resulting duplicate edits.
    The plan's Gaussian/count draws are not used by this non-flow control.
    """
    for name, value in (("num_attempts", num_attempts), ("k", k), ("batch_size", batch_size)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if len(plan.counts) != num_attempts:
        raise ValueError("search and supplied sampling plan must allocate the same attempted budget")
    if condition.baseline.shape != (1, 2) or condition.goal.shape != (1, 2):
        raise ValueError("predictor search requires one lead condition")
    setup_started = perf_counter()
    lead = smiles_to_graph(lead_smiles, model.config.max_nodes)
    lead_smiles = lead["smiles"]
    if int(lead["count"]) != plan.lead_count:
        raise ValueError("sampling plan and lead heavy-atom count differ")
    destination = Path(output_dir)
    if any((destination / name).exists() for name in
           ("attempts.jsonl", "ranking.json", "proposal_commit.json", "evaluation_outcomes.jsonl")):
        raise FileExistsError("search output directory already contains proposals or external outcomes")
    device = model_device(model)
    condition = condition.to(device)
    reserved = {smiles_to_graph(smiles, model.config.max_nodes)["smiles"]
                for smiles in reserved_support_smiles}
    reserved.add(lead_smiles)
    rng = np.random.default_rng(plan.noise_seed)
    # A parent enters this pool only from an already-recorded, predicted edit.
    parents = {lead_smiles: {"graph": lead, "score": None}}
    catalogues, cycles = {}, Counter()
    seen, score_cache = set(), {}
    ledger = []
    feature_calls = response_calls = scored_candidates = 0
    score_seconds = 0.0
    _synchronize(device)
    started = perf_counter()

    def catalogue(parent):
        if parent not in catalogues:
            candidates = enumerate_mutations(parents[parent]["graph"], model.config.max_nodes)
            order = rng.permutation(len(candidates)).tolist()
            catalogues[parent] = [candidates[index] for index in order]
        return catalogues[parent]

    with frozen_deployment(model):
        for offset in range(0, num_attempts, batch_size):
            size = min(batch_size, num_attempts - offset)
            scored_parents = sorted((identity for identity, values in parents.items()
                                     if values["score"] is not None),
                                    key=lambda identity: (-parents[identity]["score"], identity))[:16]
            beam = list(dict.fromkeys(scored_parents + [lead_smiles]))
            rows = []
            batch_started = perf_counter()
            for local_index in range(size):
                available = [parent for parent in beam if catalogue(parent)]
                if not available:
                    # Exhausted chemical possibilities are still budgeted attempts.
                    catalogues.pop(lead_smiles, None)
                    cycles[lead_smiles] += 1
                    available = [lead_smiles]
                    if not catalogue(lead_smiles):
                        raise RuntimeError("supported lead unexpectedly has no categorical mutations")
                probabilities = 1 / (np.arange(len(available), dtype=float) + 1)
                probabilities /= probabilities.sum()
                parent = available[int(rng.choice(len(available), p=probabilities))]
                mutation = catalogue(parent).pop()
                outcome = decode_graph(mutation.atoms, mutation.bonds, mutation.count,
                                       model.config.max_nodes)
                smiles, valid, connected = outcome.smiles, outcome.chemically_valid, outcome.connected
                failure_reason = outcome.failure_reason
                scope_supported = valid and connected
                if scope_supported:
                    try:
                        smiles_to_graph(smiles, model.config.max_nodes)
                    except ValueError as error:
                        # RDKit cleanup can introduce charges after sanitization.
                        # Such an edit is a failed attempt in the declared neutral scope.
                        scope_supported = valid = False
                        failure_reason = f"unsupported_chemistry:{error}"
                usable = valid and connected
                neighbour = neighbourhood(smiles, lead_smiles) if usable else {}
                duplicate = smiles in seen if usable else None
                support_match = bool(usable and smiles in reserved)
                local = bool(usable and neighbour.get("eligible", False))
                if usable:
                    seen.add(smiles)
                rows.append({
                    "attempt_id": offset + local_index, "proposal_kind": "predictor_guided_local_search",
                    "count_seed": plan.count_seed, "noise_seed": plan.noise_seed,
                    "lead_atom_count": plan.lead_count, "atom_count": mutation.count,
                    "nfe": 0, "T": None, "parent_smiles": parent,
                    "mutation_operation": mutation.operation,
                    "mutation_positions": list(mutation.positions), "mutation_new_label": mutation.new_label,
                    "parent_catalogue_cycle": cycles[parent],
                    "atoms": mutation.atoms.tolist(), "bonds": mutation.bonds.tolist(),
                    "smiles": smiles, "chemically_valid": valid, "connected": connected,
                    "sanitized_chemically_valid": outcome.chemically_valid,
                    "supported_chemistry": scope_supported,
                    "failure_reason": failure_reason, "duplicate": duplicate,
                    "reserved_support_match": support_match, "lead_match": smiles == lead_smiles if usable else False,
                    "within_neighbourhood": local, "lead_similarity": neighbour.get("similarity"),
                    "lead_atom_count_difference": neighbour.get("count_difference"),
                    "in_measured_library": None,
                    "eligible": bool(usable and not duplicate and not support_match and local and smiles != lead_smiles),
                    "response_mean": None, "response_covariance": None,
                    "profile_score": None, "scoring_failure": None, "selected_rank": None,
                })
            pending = list(dict.fromkeys(row["smiles"] for row in rows
                                        if row["chemically_valid"] and row["connected"]
                                        and row["smiles"] not in score_cache))
            _synchronize(device)
            scoring_started = perf_counter()
            if pending:
                try:
                    graphs = collate_graphs([{**smiles_to_graph(smiles, model.config.max_nodes),
                                              "id": f"search:{smiles}"} for smiles in pending]).to(device)
                    feature_calls += 1
                    features = model.graph_features(graphs)
                    native = condition.expand(len(pending))
                    response_calls += 1
                    distribution = model.predict_response_features(native.lead_features, features,
                                                                    native.baseline, native.context,
                                                                    native.response_state)
                    means = distribution.mean.detach().cpu().numpy()
                    covariances = distribution.covariance_matrix.detach().cpu().numpy()
                    scored_candidates += len(pending)
                    for index, smiles in enumerate(pending):
                        try:
                            score = float(joint_profile_probability(means[index], covariances[index],
                                                                     condition.baseline[0], condition.goal[0]))
                            score_cache[smiles] = (means[index].tolist(), covariances[index].tolist(), score, None)
                        except ValueError as error:
                            score_cache[smiles] = (None, None, None, str(error))
                except (ValueError, RuntimeError) as error:
                    for smiles in pending:
                        score_cache[smiles] = (None, None, None,
                            f"response_scoring_failed:{type(error).__name__}:{error}")
            for row in rows:
                if row["smiles"] in score_cache:
                    mean, covariance, score, failure = score_cache[row["smiles"]]
                    row.update(response_mean=mean, response_covariance=covariance,
                               profile_score=score, scoring_failure=failure)
                    if failure:
                        row["eligible"] = False
                    elif row["within_neighbourhood"]:
                        parents.setdefault(row["smiles"], {
                            "graph": smiles_to_graph(row["smiles"], model.config.max_nodes), "score": score})
            _synchronize(device)
            score_seconds += perf_counter() - scoring_started
            elapsed = perf_counter() - batch_started
            for row in rows:
                row["batch_amortized_seconds"] = elapsed / size
            ledger.extend(rows)
    _synchronize(device)
    ranking = top_k_candidates(ledger, k)
    elapsed = perf_counter() - started
    failures = Counter(row["failure_reason"] for row in ledger if row["failure_reason"])
    summary = {
        "proposal_kind": "predictor_guided_local_search", "attempts": len(ledger),
        "connected_valid_attempts": sum(row["chemically_valid"] and row["connected"] for row in ledger),
        "distinct_connected_valid": len(seen), "eligible_distinct": sum(row["eligible"] for row in ledger),
        "selected_distinct": len(ranking), "allocated_selection_slots": k,
        "reserved_support_identities": len(reserved), "batch_size": batch_size, "beam_width": 16,
        "count_seed": plan.count_seed, "noise_seed": plan.noise_seed, "nfe": 0,
        "actual_flow_evaluations": 0, "actual_decode_evaluations": 0,
        "actual_graph_feature_calls": feature_calls, "actual_response_scoring_calls": response_calls,
        "actual_model_calls": feature_calls + response_calls, "distinct_candidates_scored": scored_candidates,
        "generation_seconds": elapsed, "response_scoring_seconds": score_seconds,
        "proposal_setup_seconds": started - setup_started,
        "end_to_end_proposal_seconds": perf_counter() - setup_started,
        "timing_scope": "mutation enumeration/selection,host chemistry,neighbourhood checks,graph encoding,response scoring,ranking,ledger; excludes file writing",
        "native_response_state_used_for_scoring": True, "private_oracle_calls_before_commit": 0,
        "count_prior_used_for_proposals": False, "gaussian_noise_used_for_proposals": False,
        "mutation_seed": plan.noise_seed,
        "mutation_operations": dict(Counter(row["mutation_operation"] for row in ledger)),
        "failure_counts": dict(failures),
        "invalid_attempts": sum(not row["chemically_valid"] for row in ledger),
        "disconnected_attempts": sum(row["chemically_valid"] and not row["connected"] for row in ledger),
        "duplicate_attempts": sum(bool(row["duplicate"]) for row in ledger),
        "reserved_support_attempts": sum(row["reserved_support_match"] for row in ledger),
        "out_of_neighbourhood_attempts": sum(row["chemically_valid"] and row["connected"]
                                              and not row["within_neighbourhood"] for row in ledger),
        "scoring_failure_attempts": sum(row["scoring_failure"] is not None for row in ledger),
        "metadata": metadata or {},
    }
    commit = commit_proposals(destination, ledger, ranking, metadata=summary)
    summary.update(attempts_sha256=commit["attempts_sha256"], ranking_sha256=commit["ranking_sha256"])
    (destination / "generation_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    with (destination / "generation_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow({name: json.dumps(value) if isinstance(value, (dict, list)) else value
                         for name, value in summary.items()})
    return ledger, ranking, summary
