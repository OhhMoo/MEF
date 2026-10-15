"""Operational checks for the fixed-count, lead-aligned local-edit model."""

import json

import torch

from emf.contracts import GraphBatch, GraphMasks, Latents, ModelConfig
from emf.local_edit_factorized import NormalizedFactorizedFlow
from emf.local_edit_model import LocalEditModel
from emf.local_edit_objectives import local_edit_loss
from emf.local_edit_sampling import anchored_initial_noise
from emf.local_edit_trace import trace_pair
from emf.models import EMFModel
from emf.redesign import RedesignModel
from emf.redesign_contracts import CampaignBatch, RedesignConfig, SupportBatch
from emf.redesign_sampling import frozen_deployment, sample_redesign_latents


def _fixture():
    torch.manual_seed(11)
    config = RedesignConfig(ModelConfig(max_nodes=4, node_dim=8, pair_dim=4,
                                        hidden_dim=16, depth=1, num_heads=4),
                            state_dim=3, context_dim=2)
    representation = EMFModel(config.model)
    masks = GraphMasks.from_counts(torch.tensor([4, 4]), 4)
    lead = GraphBatch(torch.zeros(2, 4, dtype=torch.long),
                      torch.zeros(2, 6, dtype=torch.long), masks)
    candidate_atoms = lead.atoms.clone()
    candidate_atoms[0, 1] = 1
    candidate_atoms[1, 2] = 2
    candidate = GraphBatch(candidate_atoms, lead.bonds, masks)
    support = SupportBatch(torch.randn(1, 2, config.feature_dim),
                           torch.zeros(1, 2, 2), torch.zeros(1, 2),
                           torch.ones(1, 2, dtype=torch.bool))
    goals = torch.zeros(2, 2)
    goals[1, 0] = 1.
    batch = CampaignBatch(candidate, lead, support, torch.zeros(2, dtype=torch.long),
                          torch.zeros(2, 2), torch.zeros(2, 2), goals,
                          torch.zeros(2, 2), torch.tensor([1., 0.]))
    return config, representation, batch


def test_anchor_residual_and_standard_terminal_nfe():
    config, representation, batch = _fixture()
    model = LocalEditModel(config, representation)
    condition = model.condition(batch.lead, batch.baseline, batch.context,
                                torch.zeros(2, 3), batch.goal)
    gaussian = Latents(torch.randn(2, 4, 8), torch.randn(2, 6, 4))
    start = anchored_initial_noise(gaussian, batch.query.masks, condition)
    assert torch.allclose(start.nodes - condition.lead.nodes, gaussian.nodes)
    prediction = model.predict_clean(start, torch.full((2,), .3),
                                     batch.query.masks, condition)
    # The zero-start residual heads make the initial destination exactly X0.
    assert torch.allclose(prediction.nodes, condition.lead.nodes)
    assert torch.allclose(prediction.pairs, condition.lead.pairs)
    with frozen_deployment(model):
        sampled = sample_redesign_latents(model, batch.query.counts, condition,
                                          nfe=4, initial_noise=start)
    assert sampled.nfe == 4
    assert torch.allclose(sampled.latents.nodes, condition.lead.nodes)


def test_task_zero_prior_and_goal_filtered_terminal_gradient():
    config, representation, batch = _fixture()
    model = LocalEditModel(config, representation, task_conditioned=False)
    assert all(not parameter.requires_grad for parameter in model.encoder.parameters())
    state = torch.zeros(2, 3)
    condition = model.condition(batch.lead, batch.baseline, batch.context, state, batch.goal)
    other = model.condition(batch.lead, batch.baseline, batch.context,
                            torch.ones_like(state), batch.goal)
    with torch.no_grad():
        model.flow.node_head.linear.weight.normal_(std=.01)
    z = anchored_initial_noise(Latents(torch.randn(2, 4, 8), torch.randn(2, 6, 4)),
                               batch.query.masks, condition)
    t = torch.full((2,), .4)
    assert torch.allclose(model.predict_clean(z, t, batch.query.masks, condition).nodes,
                          model.predict_clean(z, t, batch.query.masks, other).nodes)
    loss, metrics = local_edit_loss(model, batch, flow_state_override=state,
                                    terminal_ce_weight=.1,
                                    generator=torch.Generator().manual_seed(12))
    assert torch.isfinite(loss)
    assert metrics["accepted_fraction"].item() == .5
    loss.backward()
    assert model.flow.node_head.linear.weight.grad is not None
    assert all(parameter.grad is None for parameter in model.encoder.parameters())


def test_normalized_factorized_selector_preserves_zero_head():
    config, representation, _ = _fixture()
    flow = NormalizedFactorizedFlow(config, representation.flow)
    assert torch.count_nonzero(flow.selector.selector[-1].weight) == 0
    inputs = torch.cat((torch.full((1, 16), 1000.),
                        torch.full((1, 16), 200.), torch.zeros(1, flow.selector.selector[0].in_features - 32)), -1)
    captured = {}
    def capture(_module, args):
        captured["inputs"] = args[0]
    hook = flow.selector.selector[0].register_forward_pre_hook(capture)
    flow.selector(inputs)
    hook.remove()
    assert torch.allclose(captured["inputs"][0, :32], torch.ones(32), atol=1e-5)


def test_paired_trace_is_jsonable_and_keeps_model_frozen():
    config, representation, batch = _fixture()
    model = LocalEditModel(config, representation)
    positive = model.condition(batch.lead, batch.baseline, batch.context,
                               torch.zeros(2, 3), batch.goal)
    negative = model.condition(batch.lead, batch.baseline, batch.context,
                               torch.ones(2, 3), batch.goal)
    start = anchored_initial_noise(Latents(torch.randn(2, 4, 8), torch.randn(2, 6, 4)),
                                   batch.query.masks, positive)
    model.freeze_deployment()
    trace = trace_pair(model, batch.query.counts, positive, negative, start,
                       nfe=5, known_state_jvp=True, include_canonical=True)
    assert set(trace["probes_same_positive_state"]) == {"0.1", "0.5", "0.9"}
    assert trace["nfe_per_path"] == 5
    assert len(trace["paired_endpoint"]["atom_category_flips"]) == 2
    assert len(trace["paired_endpoint"]["positive_outcomes"]) == 2
    json.dumps(trace)
    assert all(not parameter.requires_grad and parameter.grad is None
               for parameter in model.parameters())


def test_factorized_trace_records_basis_and_selector_activation():
    config, representation, batch = _fixture()
    model = RedesignModel(config, representation)
    model.flow = NormalizedFactorizedFlow(config, representation.flow)
    positive = model.condition(batch.lead, batch.baseline, batch.context,
                               torch.zeros(2, 3), batch.goal)
    negative = model.condition(batch.lead, batch.baseline, batch.context,
                               torch.ones(2, 3), batch.goal)
    noise = Latents(torch.randn(2, 4, 8), torch.randn(2, 6, 4))
    model.freeze_deployment()
    trace = trace_pair(model, batch.query.counts, positive, negative, noise, nfe=5)
    middle = trace["probes_same_positive_state"]["0.5"]
    assert "composed_correction_rms_difference" in middle
    assert len(middle["selector_activation"]) == 2
    json.dumps(trace)


def test_trace_jvp_accepts_expanded_single_lead_condition():
    config, representation, batch = _fixture()
    model = LocalEditModel(config, representation)
    first = torch.tensor([0])
    lead = GraphBatch(batch.lead.atoms.index_select(0, first),
                      batch.lead.bonds.index_select(0, first),
                      GraphMasks(batch.lead.masks.nodes.index_select(0, first),
                                 batch.lead.masks.pairs.index_select(0, first)))
    positive = model.condition(lead, batch.baseline[:1], batch.context[:1],
                               torch.zeros(1, 3), batch.goal[:1]).expand(4)
    negative = model.condition(lead, batch.baseline[:1], batch.context[:1],
                               torch.ones(1, 3), batch.goal[:1]).expand(4)
    masks = GraphMasks.from_counts(torch.full((4,), 4), 4)
    start = anchored_initial_noise(Latents(torch.randn(4, 4, 8), torch.randn(4, 6, 4)),
                                   masks, positive)
    model.freeze_deployment()
    result = trace_pair(model, torch.full((4,), 4), positive, negative, start,
                        nfe=5, known_state_jvp=True)
    assert len(result["probes_same_positive_state"]["0.5"]["known_state_jvp_rms"]) == 4
