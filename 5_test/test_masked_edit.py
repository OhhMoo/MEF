"""Masked-edit algebra, protected categories, and controller gradients."""

from dataclasses import replace

import pytest
import torch

from emf.contracts import GraphBatch, GraphLogits, GraphMasks, Latents, ModelConfig
from emf.masked_edit_model import (MaskedEditCondition, MaskedEditModel,
                                   clamp_graph_logits, hybrid_latents)
from emf.masked_edit_objectives import masked_edit_loss
from emf.masked_edit_sampling import sample_masked_edit_latents
from emf.models import EMFModel
from emf.redesign_contracts import CampaignBatch, RedesignConfig, SupportBatch


def _fixture():
    torch.manual_seed(9)
    config = RedesignConfig(ModelConfig(max_nodes=4, node_dim=8, pair_dim=4,
                                        hidden_dim=16, depth=1, num_heads=4),
                            state_dim=3, context_dim=2)
    valid = GraphMasks.from_counts(torch.tensor([4, 4]), 4)
    editable = GraphMasks(torch.tensor([[1, 1, 0, 0], [1, 1, 0, 0]], dtype=torch.bool),
                          torch.tensor([[1, 0, 0, 0, 0, 0],
                                        [1, 0, 0, 0, 0, 0]], dtype=torch.bool))
    lead = GraphBatch(torch.zeros(2, 4, dtype=torch.long),
                      torch.zeros(2, 6, dtype=torch.long), valid)
    atoms, bonds = lead.atoms.clone(), lead.bonds.clone()
    atoms[:, 0] = 1
    bonds[:, 0] = 1
    query = GraphBatch(atoms, bonds, valid)
    support = SupportBatch(torch.randn(1, 2, config.feature_dim),
                           torch.zeros(1, 2, 2), torch.zeros(1, 2),
                           torch.ones(1, 2, dtype=torch.bool))
    batch = CampaignBatch(query, lead, support, torch.zeros(2, dtype=torch.long),
                          torch.zeros(2, 2), torch.zeros(2, 2), torch.zeros(2, 2),
                          torch.zeros(2, 2), torch.ones(2))
    return config, EMFModel(config.model), batch, editable


def test_hybrid_preserves_exact_lead_latents_and_graph_clamp():
    config, representation, batch, edit = _fixture()
    model = MaskedEditModel(config, representation, variant="C")
    with torch.no_grad():
        lead, query = model.encode(batch.lead), model.encode(batch.query)
    hybrid = hybrid_latents(lead, query, edit)
    torch.testing.assert_close(hybrid.nodes[:, 2:], lead.nodes[:, 2:])
    torch.testing.assert_close(hybrid.pairs[:, 1:], lead.pairs[:, 1:])
    torch.testing.assert_close(hybrid.nodes[:, :2], query.nodes[:, :2])
    assert not torch.allclose(query.nodes[:, 2:], lead.nodes[:, 2:])
    condition = model.condition(batch.lead, batch.baseline, batch.context,
                                torch.zeros(2, 3), batch.goal, edit)
    logits = GraphLogits(torch.full((2, 4, 4), -2.), torch.full((2, 6, 4), -2.))
    logits.atoms[..., 3] = 10.
    logits.bonds[..., 3] = 10.
    clamped = clamp_graph_logits(logits, condition)
    assert torch.equal(clamped.atoms.argmax(-1)[:, 2:], batch.lead.atoms[:, 2:])
    assert torch.equal(clamped.bonds.argmax(-1)[:, 1:], batch.lead.bonds[:, 1:])
    assert torch.equal(clamped.atoms.argmax(-1)[:, :2], torch.full((2, 2), 3))
    assert torch.equal(clamped.bonds.argmax(-1)[:, :1], torch.full((2, 1), 3))


@pytest.mark.parametrize("variant", ["A", "B", "C"])
def test_flow_decoder_gradients_and_frozen_encoder(variant):
    config, representation, batch, edit = _fixture()
    model = MaskedEditModel(config, representation, variant=variant)
    loss, metrics = masked_edit_loss(model, batch, edit, accept_all=True,
                                     generator=torch.Generator().manual_seed(7))
    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["flow_loss"])
    loss.backward()
    assert model.flow.node_head.linear.weight.grad is not None
    assert model.decoder.node_head[-1].weight.grad is not None
    assert all(parameter.grad is None for parameter in model.encoder.parameters())


def test_protected_category_edit_rejected():
    config, representation, batch, edit = _fixture()
    changed = batch.query.atoms.clone()
    changed[0, 3] = 2
    invalid = replace(batch, query=GraphBatch(changed, batch.query.bonds, batch.query.masks))
    with pytest.raises(ValueError, match="protected"):
        masked_edit_loss(MaskedEditModel(config, representation, variant="A"),
                         invalid, edit)


class _ControlledDestination:
    def __init__(self, variant):
        self.variant = variant
        self.calls = 0

    def predict_clean(self, state, time, masks, condition):
        self.calls += 1
        response = condition.response_state[:, :1, None]
        return Latents(condition.lead.nodes + 1 + response,
                       condition.lead.pairs + 1 + response)


@pytest.mark.parametrize("variant", ["A", "B", "C"])
def test_nfe_mask_boundary_and_full_controller_gradient(variant):
    _, _, batch, edit = _fixture()
    lead = Latents(torch.full((2, 4, 8), 2.), torch.full((2, 6, 4), 3.))
    condition = MaskedEditCondition(torch.zeros(2, 13), torch.zeros(2, 2),
                                    torch.zeros(2, 2), torch.zeros(2, 3),
                                    torch.zeros(2, 2), lead, batch.lead, edit)
    gaussian = Latents(torch.ones_like(lead.nodes), torch.ones_like(lead.pairs))
    gain = torch.tensor(.2, requires_grad=True)

    def controller(state, time, base):
        signal = gain * state.nodes[:, 0, :].mean(-1, keepdim=True)
        return replace(base, response_state=torch.cat((signal, base.response_state[:, 1:]), -1))

    model = _ControlledDestination(variant)
    result = sample_masked_edit_latents(model, condition, gaussian, nfe=5,
                                         controller=controller)
    assert result.nfe == model.calls == 5
    result.latents.nodes[:, 0].sum().backward()
    assert gain.grad is not None and gain.grad.abs() > 0
    if variant == "A":
        assert not torch.equal(result.latents.nodes[:, 2:], lead.nodes[:, 2:])
    else:
        torch.testing.assert_close(result.latents.nodes[:, 2:], lead.nodes[:, 2:])
        torch.testing.assert_close(result.latents.pairs[:, 1:], lead.pairs[:, 1:])


def test_full_rollout_keeps_earlier_controller_gradients():
    _, _, batch, edit = _fixture()
    lead = Latents(torch.ones(2, 4, 8), torch.ones(2, 6, 4))
    condition = MaskedEditCondition(torch.zeros(2, 13), torch.zeros(2, 2),
                                    torch.zeros(2, 2), torch.zeros(2, 3),
                                    torch.zeros(2, 2), lead, batch.lead, edit)
    gaussian = Latents(torch.ones_like(lead.nodes), torch.ones_like(lead.pairs))
    gain = torch.tensor(.2, requires_grad=True)

    def controller(state, time, base):
        signal = gain * state.nodes[:, 0, :].mean(-1, keepdim=True)
        return replace(base, response_state=torch.cat((signal, base.response_state[:, 1:]), -1))

    full = sample_masked_edit_latents(_ControlledDestination("C"), condition,
                                      gaussian, nfe=5, controller=controller)
    full.latents.nodes[:, 0].sum().backward()
    full_gradient = gain.grad.item()
    gain.grad = None
    truncated = sample_masked_edit_latents(_ControlledDestination("C"), condition,
                                           gaussian, nfe=5, controller=controller,
                                           truncate_steps=1)
    truncated.latents.nodes[:, 0].sum().backward()
    assert abs(full_gradient - gain.grad.item()) > 1e-4


@pytest.mark.parametrize("variant", ["A", "B", "C"])
def test_control_projection_terminal_contribution_and_prefix_continuation(variant):
    _, _, batch, edit = _fixture()
    lead = Latents(torch.ones(2, 4, 8), torch.ones(2, 6, 4))
    condition = MaskedEditCondition(torch.zeros(2, 13), torch.zeros(2, 2),
                                    torch.zeros(2, 2), torch.zeros(2, 3),
                                    torch.zeros(2, 2), lead, batch.lead, edit)
    gaussian = Latents(torch.ones_like(lead.nodes), torch.ones_like(lead.pairs))

    def control(state, time, supplied):
        return Latents(torch.full_like(state.nodes, 2.), torch.full_like(state.pairs, 2.))

    base = sample_masked_edit_latents(_ControlledDestination(variant), condition,
                                      gaussian, nfe=5)
    guided = sample_masked_edit_latents(_ControlledDestination(variant), condition,
                                        gaussian, nfe=5,
                                        velocity_controller=control, return_trace=True)
    assert len(guided.states) == guided.nfe == len(guided.times)
    # The terminal destination is clean + (1-T)u(T), and u is editable only.
    torch.testing.assert_close(guided.latents.nodes[:, 0] - base.latents.nodes[:, 0],
                               torch.full_like(lead.nodes[:, 0], .1))
    torch.testing.assert_close(guided.latents.pairs[:, 0] - base.latents.pairs[:, 0],
                               torch.full_like(lead.pairs[:, 0], .1))
    torch.testing.assert_close(guided.latents.nodes[:, 2:] - base.latents.nodes[:, 2:],
                               torch.zeros_like(lead.nodes[:, 2:]))
    torch.testing.assert_close(guided.latents.pairs[:, 1:] - base.latents.pairs[:, 1:],
                               torch.zeros_like(lead.pairs[:, 1:]))
    continued = sample_masked_edit_latents(
        _ControlledDestination(variant), condition, nfe=3, start_t=.475,
        initial_state=guided.states[2], velocity_controller=control)
    torch.testing.assert_close(continued.latents.nodes, guided.latents.nodes)
    torch.testing.assert_close(continued.latents.pairs, guided.latents.pairs)


def test_delta_goal_weights_match_new_campaign_semantics():
    config, representation, batch, edit = _fixture()
    goals = torch.tensor([[0., 0.], [0., 1.]])
    weights = torch.tensor([1., 0.])
    changed = replace(batch, goal=goals, flow_weights=weights,
                      baseline=torch.full((2, 2), 99.))
    loss, metrics = masked_edit_loss(MaskedEditModel(config, representation, variant="C"),
                                     changed, edit,
                                     generator=torch.Generator().manual_seed(4))
    assert torch.isfinite(loss)
    assert metrics["accepted_fraction"] == .5


def test_task_zero_prior_skips_unused_evidence_and_response():
    config, representation, batch, edit = _fixture()
    model = MaskedEditModel(config, representation, variant="C", task_conditioned=False)
    loss, metrics = masked_edit_loss(model, batch, edit, accept_all=True,
                                     response_weight=0.,
                                     generator=torch.Generator().manual_seed(5))
    assert metrics["response_loss"] == 0
    loss.backward()
    assert all(parameter.grad is None for parameter in model.evidence.parameters())
    assert all(parameter.grad is None for parameter in model.response.parameters())
