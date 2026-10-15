"""Independent saved-state, accounting, and chemistry checks for recovery."""

import json
import csv
import ast
import importlib.util
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import Crippen, rdMolDescriptors
import torch
import pytest

from emf.contracts import GraphBatch, GraphMasks, Latents
from emf.masked_edit_model import MaskedEditCondition
from emf.masked_edit_sampling import masked_initial_state, sample_masked_edit_latents
import emf.revision_recovery as recovery


def audit_completed_artifacts(output):
    """Read-only independent exact-response and accounting audit for a GPU run."""
    from emf.masked_edit_data import DEFAULT_DIR

    output = Path(output)
    summary = json.loads((output / "summary.json").read_text())
    private = json.loads((DEFAULT_DIR / "private/oracle.json").read_text())
    scales = np.asarray(private["scales"])
    with (output / "attempts.csv").open() as stream:
        attempts = list(csv.DictReader(stream))
    with (output / "cases.csv").open() as stream:
        cases = list(csv.DictReader(stream))

    def feature(smiles):
        mol = Chem.MolFromSmiles(smiles)
        a, b = np.asarray([Crippen.MolLogP(mol), rdMolDescriptors.CalcTPSA(mol)]) / scales
        return np.asarray([a, b, a*b])

    exact_checked = 0
    for row in attempts:
        if not row.get("physical_exact_delta"):
            continue
        cid = row["campaign_id"]
        physical = np.asarray(private["coefficients"][cid]) @ (
            feature(row["smiles"])-feature(private["campaigns"][cid]["lead_smiles"]))
        np.testing.assert_allclose(ast.literal_eval(row["physical_exact_delta"]), physical,
                                   atol=1e-10, rtol=1e-10)
        expected = row["eligible"] == "True" and -physical[0] >= .25 and physical[1] >= -.1
        assert (row["goal_success"] == "True") == expected
        exact_checked += 1
    for case in cases:
        raw = [row for row in attempts if row["campaign_id"] == case["campaign_id"]
               and row["method"] == case["method"]]
        valid = sum(row["chemically_valid"] == "True" and row["connected"] == "True" for row in raw)
        assert int(case["raw_candidate_attempts"]) == len(raw)
        assert int(case["raw_valid_connected_attempts"]) == valid
    physical_requests = sum(row["physical_exact_call"] == "True" for row in attempts)
    assert physical_requests == summary["physical_exact_evaluation_requests"]
    assert len(attempts) == summary["logical_attempt_rows"]
    with (output / "cost_batches.csv").open() as stream:
        for batch in csv.DictReader(stream):
            np.testing.assert_allclose(float(batch["total_seconds"]),
                                       float(batch["post_seconds"])+float(batch["prefix_seconds"]),
                                       rtol=1e-10, atol=1e-10)
    return dict(logical_rows=len(attempts), independently_checked_exact_rows=exact_checked,
                physical_evaluation_requests=physical_requests, cases=len(cases), passed=True)


def _condition():
    valid = GraphMasks.from_counts(torch.tensor([3, 4]), 4)
    edits = GraphMasks(torch.tensor([[1, 0, 1, 0], [0, 1, 0, 0]], dtype=torch.bool),
                       torch.tensor([[1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0]], dtype=torch.bool))
    lead = Latents(torch.randn(2, 4, 3), torch.randn(2, 6, 2)).masked(valid)
    graph = GraphBatch(torch.zeros(2, 4, dtype=torch.long),
                       torch.zeros(2, 6, dtype=torch.long), valid)
    return MaskedEditCondition(torch.zeros(2, 6), torch.zeros(2, 2),
                               torch.zeros(2, 2), torch.zeros(2, 3),
                               torch.zeros(2, 2), lead, graph, edits)


class _Flow:
    variant = "B"

    def __init__(self):
        self.calls = 0

    def predict_clean(self, state, time, masks, condition):
        self.calls += 1
        return Latents(.7 * state.nodes + .3 * condition.lead.nodes + time[:, None, None],
                       .7 * state.pairs + .3 * condition.lead.pairs - time[:, None, None])


def test_random_intervention_is_bounded_repeatable_and_keeps_saved_state():
    condition = _condition()
    state = masked_initial_state("B", Latents(torch.randn_like(condition.lead.nodes),
                                               torch.randn_like(condition.lead.pairs)), condition)
    before = Latents(state.nodes.clone(), state.pairs.clone())
    torch.manual_seed(919)
    global_rng = torch.random.get_rng_state().clone()
    moved = recovery.perturb_masked_state(state, condition, radius=.5, seed=17)
    again = recovery.perturb_masked_state(state, condition, radius=.5, seed=17)
    assert torch.equal(global_rng, torch.random.get_rng_state())
    for value, repeated, saved, original, mask in zip(
            (moved.nodes, moved.pairs), (again.nodes, again.pairs),
            (state.nodes, state.pairs), (before.nodes, before.pairs),
            (condition.edit_masks.nodes, condition.edit_masks.pairs)):
        torch.testing.assert_close(value, repeated, rtol=0, atol=0)
        torch.testing.assert_close(saved, original, rtol=0, atol=0)
        torch.testing.assert_close(value[~mask], original[~mask], rtol=0, atol=0)
    delta_n, delta_p = moved.nodes-state.nodes, moved.pairs-state.pairs
    dims = condition.edit_masks.nodes.sum(1)*3 + condition.edit_masks.pairs.sum(1)*2
    rms = torch.sqrt((delta_n.square().sum((1, 2))+delta_p.square().sum((1, 2)))/dims)
    torch.testing.assert_close(rms, torch.full((2,), .5))
    zero = recovery.perturb_masked_state(state, condition, radius=0, seed=99)
    torch.testing.assert_close(zero.nodes, state.nodes, rtol=0, atol=0)
    torch.testing.assert_close(zero.pairs, state.pairs, rtol=0, atol=0)


def test_cached_prefix_and_zero_intervention_reproduce_full_suffix_with_terminal_nfe():
    condition = _condition()
    noise = Latents(torch.randn_like(condition.lead.nodes), torch.randn_like(condition.lead.pairs))
    model = _Flow()
    full = sample_masked_edit_latents(model, condition, noise, nfe=16, T=.95,
                                      return_trace=True)
    assert model.calls == 16
    grid = recovery.revision_grid(16, .95)
    indices = tuple(row["grid_index"] for row in grid)
    model.calls = 0
    cache, timings = recovery._prefix_trace(model, condition, noise, nfe=16, T=.95,
                                            indices=indices, device="cpu")
    assert model.calls == max(indices)
    assert list(timings.values()) == sorted(timings.values())
    for index in indices:
        torch.testing.assert_close(cache[index].nodes, full.states[index].nodes)
        torch.testing.assert_close(cache[index].pairs, full.states[index].pairs)
        model.calls = 0
        state = recovery.perturb_masked_state(cache[index], condition, radius=0, seed=5)
        suffix = sample_masked_edit_latents(model, condition, nfe=16-index, T=.95,
                                            start_t=float(full.times[index]), initial_state=state)
        assert model.calls == 16-index
        torch.testing.assert_close(suffix.latents.nodes, full.latents.nodes)
        torch.testing.assert_close(suffix.latents.pairs, full.latents.pairs)


def _row(index):
    return dict(sample=index, eligible=True, rank_score=1., smiles=f"candidate-{index}",
                chemically_valid=True, connected=True, goal_success=True,
                endpoint2_preserved=True, old_goal_success=False)


def test_budget_frontier_has_indivisible_pools_and_keeps_sunk_prefix():
    batches = [recovery.Batch("case", "rollback_25", first, 2, 8, 5., 3., 2.,
                              30, 12, 8, tuple(_row(i) for i in range(first, first+2)))
               for first in (0, 2)]
    results = recovery.budget_frontier(batches, (2.9, 3., 5., 6., 10.), topk=16)
    total = [r for r in results if r["budget_basis"] == "total"]
    post = [r for r in results if r["budget_basis"] == "post_update"]
    assert [r["candidate_attempts"] for r in total] == [0, 0, 8, 8, 16]
    assert [r["candidate_attempts"] for r in post] == [0, 8, 8, 16, 16]
    assert post[1]["total_charged_seconds"] == 5.
    assert total[2]["topk_success_fraction"] == 2/16
    assert total[-1]["exact_calls"] == 16


def test_exact_response_recomputed_from_committed_identity_without_response_table():
    from emf.masked_edit_data import MaskedEditDataset, DEFAULT_DIR
    from emf.masked_edit_evaluation import ExactMaskedEvaluator

    dataset = MaskedEditDataset(split="validation", kind="vector")
    cid = dataset.campaign_ids[0]
    mask = dataset.masks[dataset.manifest["campaigns"][cid]["mask_id"]]
    private = json.loads((DEFAULT_DIR / "private" / "oracle.json").read_text())
    scale = np.asarray(private["scales"])
    coefficients = np.asarray(private["coefficients"][cid])

    def features(smiles):
        mol = Chem.MolFromSmiles(smiles)
        a, b = np.array([Crippen.MolLogP(mol), rdMolDescriptors.CalcTPSA(mol)]) / scale
        return np.array([a, b, a*b])

    evaluator = ExactMaskedEvaluator()
    tested = 0
    for graph in dataset.compatible_library(cid)[:12]:
        row = evaluator.evaluate_graph(graph, mask, cid)
        if row["exact_delta"] is None:
            continue
        independent = coefficients @ (features(row["smiles"])-features(mask.lead_smiles))
        np.testing.assert_allclose(row["exact_delta"], independent, rtol=1e-12, atol=1e-12)
        tested += 1
    assert tested > 0


def test_cpu_runner_charges_late_prefix_and_preserves_root_pairing(tmp_path, monkeypatch):
    from emf.masked_edit_control import SoftGraphDescriptorCritic
    from emf.masked_edit_run import new_model

    torch.set_num_threads(1)
    model = new_model("B", "cpu", conditioned=True).eval().requires_grad_(False)
    critic = SoftGraphDescriptorCritic(model.config, hidden_dim=16, depth=1).eval().requires_grad_(False)
    monkeypatch.setattr(recovery, "load_model", lambda *args:
                        (model, {"variant": "B", "conditioned": True}))
    monkeypatch.setattr(recovery, "load_critic", lambda *args: (critic, {}))
    methods = ["conditional_continue", "conditional_restart", "rollback_25", "rollback_50",
               "randomized_suffix", "oracle_select_same_suffix_pool"]
    summary = recovery.run(dict(conditional_checkpoint="fake", critic="fake", seed=142,
                                split="validation", output=str(tmp_path), campaign_limit=1,
                                attempts=2, baseline_attempts=2, chunk=2, nfe=7,
                                intervention_trials=2, search_trials=0,
                                methods=methods, wall_budgets_seconds=[100.]), "cpu")
    with (tmp_path / "cost_batches.csv").open() as stream:
        batches = {row["method"]: row for row in csv.DictReader(stream)}
    reveal = summary["reveal_grid_index"]
    index = {x["target_time"]: x["grid_index"] for x in summary["revision_grid"]}
    for method, start in [("conditional_continue", reveal), ("rollback_25", index[.25]),
                          ("rollback_50", index[.5]), ("conditional_restart", 0)]:
        row = batches[method]
        assert int(row["prefix_nfe"]) == 2*reveal
        assert int(row["forward_nfe"]) == 2*(reveal+7-start)
        assert float(row["total_seconds"]) > float(row["post_seconds"]) > 0
    with (tmp_path / "attempts.csv").open() as stream:
        attempts = list(csv.DictReader(stream))
    assert sum(r["physical_exact_call"] == "True" for r in attempts) == summary["physical_exact_evaluation_requests"]
    paired = [r for r in attempts if r["method"] == "conditional_continue"]
    assert len({r["root_draw_id"] for r in paired}) == 2
    for root in ("0", "1"):
        hashes = {r["reveal_state_sha256"] for r in attempts if r["root_sample"] == root}
        assert len(hashes) == 1
    assert int(batches["randomized_suffix"]["exact_calls"]) == 4
    assert int(batches["oracle_select_same_suffix_pool"]["exact_calls"]) == 4
    with (tmp_path / "cases.csv").open() as stream:
        cases = {row["method"]: row for row in csv.DictReader(stream)}
    for method in methods:
        raw = [row for row in attempts if row["method"] == method]
        valid = sum(row["chemically_valid"] == "True" and row["connected"] == "True" for row in raw)
        assert int(cases[method]["raw_candidate_attempts"]) == len(raw)
        assert int(cases[method]["raw_valid_connected_attempts"]) == valid


def test_source_gate_counts_failed_bounded_unique_roots_across_campaigns(tmp_path):
    script = Path(__file__).resolve().parents[1] / "3_model_training/scripts/revision_recovery_workflow.py"
    spec = importlib.util.spec_from_file_location("recovery_workflow_review", script)
    workflow = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(workflow)
    rows = [dict(campaign_id=f"case-{i//4}", seed=142, root_sample=i,
                 bounded_radius_rms=.5, direct_success=False,
                 randomized_suffix_any_success=True, oracle_adaptive_any_success=False)
            for i in range(8)]
    rows += [dict(rows[0]),  # Duplicate records cannot pass the threshold twice.
             dict(rows[0], root_sample=99, direct_success=True),
             dict(rows[0], root_sample=100, bounded_radius_rms=.6),
             dict(rows[0], root_sample=101, randomized_suffix_any_success=False)]

    def write(data):
        with (tmp_path / "roots.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(data)

    write(rows)
    tasks = [dict(output=str(tmp_path), split="train")]
    criteria = dict(max_delta_rms=.5, min_recovered_roots=8, min_campaigns=2)
    result = workflow.source_gate(tasks, criteria)
    assert result["passed"] and result["recovered_roots"] == 8
    assert len(result["campaigns"]) == 2
    write([dict(row, campaign_id="one-case") for row in rows])
    assert not workflow.source_gate(tasks, criteria)["passed"]
    with pytest.raises(ValueError, match="source"):
        workflow.source_gate([dict(tasks[0], split="test")], criteria)
