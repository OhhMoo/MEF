"""Exact prefix replay and small source-backed revision operation."""

import torch

from emf.contracts import GraphBatch, GraphMasks, Latents
from emf.masked_edit_model import MaskedEditCondition
from emf.masked_edit_revision import replay_prefix_state, revision_grid, run_revision
from emf.masked_edit_sampling import sample_masked_edit_latents


class _DriftingFlow:
    variant = "C"

    def predict_clean(self, state, time, masks, condition):
        return Latents(.7 * state.nodes + .3 * condition.lead.nodes + time[:, None, None],
                       .7 * state.pairs + .3 * condition.lead.pairs - time[:, None, None])


def test_prefix_replay_matches_full_controlled_trace_at_each_grid_state():
    valid = GraphMasks.from_counts(torch.tensor([4]), 4)
    edit = GraphMasks(torch.tensor([[1, 1, 0, 0]], dtype=torch.bool),
                      torch.tensor([[1, 0, 0, 0, 0, 0]], dtype=torch.bool))
    graph = GraphBatch(torch.zeros(1, 4, dtype=torch.long),
                       torch.zeros(1, 6, dtype=torch.long), valid)
    lead = Latents(torch.randn(1, 4, 3), torch.randn(1, 6, 2))
    condition = MaskedEditCondition(torch.zeros(1, 6), torch.zeros(1, 2),
                                    torch.zeros(1, 2), torch.zeros(1, 3),
                                    torch.zeros(1, 2), lead, graph, edit)
    gaussian = Latents(torch.randn_like(lead.nodes), torch.randn_like(lead.pairs))
    node_u = torch.randn(1, 2, 4, 3)
    pair_u = torch.randn(1, 2, 6, 2)

    def controller(state, time, supplied):
        first = (time < .95 / 2)[:, None, None]
        return Latents(torch.where(first, node_u[:, 0], node_u[:, 1]),
                       torch.where(first, pair_u[:, 0], pair_u[:, 1]))

    model = _DriftingFlow()
    full = sample_masked_edit_latents(model, condition, gaussian, nfe=7,
                                      velocity_controller=controller, return_trace=True)
    for index in range(1, 6):
        prefix = replay_prefix_state(model, condition, gaussian, node_u, pair_u,
                                     nfe=7, T=.95, grid_index=index)
        torch.testing.assert_close(prefix.nodes, full.states[index].nodes)
        torch.testing.assert_close(prefix.pairs, full.states[index].pairs)
    grid = revision_grid(3, .95)
    assert [item["grid_index"] for item in grid] == [1, 1, 1]


def test_small_revision_operation_counts_reused_restart(tmp_path, monkeypatch):
    import csv
    import emf.masked_edit_revision as module
    from emf.masked_edit_control import SoftGraphDescriptorCritic
    from emf.masked_edit_run import new_model

    model = new_model("C", "cpu").eval().requires_grad_(False)
    critic = SoftGraphDescriptorCritic(model.config, hidden_dim=16, depth=1).eval().requires_grad_(False)
    monkeypatch.setattr(module, "load_model", lambda path, device: (model, {"variant": "C"}))
    monkeypatch.setattr(module, "load_critic", lambda path, loaded, device: (critic, {}))
    summary = run_revision({"stage": "revision", "seed": 142,
                            "checkpoint": "operational-model", "critic": "operational-critic",
                            "output": str(tmp_path), "campaign_limit": 1,
                            "attempts": 2, "chunk": 2, "nfe": 3,
                            "iterations": 0}, "cpu")
    assert summary["physical_cached_baseline_rollouts"] == 2
    assert summary["physical_truth_evaluations"] == 12
    assert summary["physical_forward_sample_nfe"] == 36
    assert summary["logical_attempt_rows"] == 22
    assert (tmp_path / "attempts.csv").exists()
    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "summary.json").exists()
    with (tmp_path / "attempts.csv").open(newline="") as stream:
        attempts = list(csv.DictReader(stream))
    assert sum(row["physical_truth_evaluation"] == "True" for row in attempts) == 12
