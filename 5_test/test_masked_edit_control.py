"""Soft-graph descriptor basis and differentiable terminal control checks."""

import numpy as np
import torch
from torch import nn

from emf.contracts import GraphBatch, GraphLogits, GraphMasks, Latents, ModelConfig
from emf.masked_edit_control import (SoftGraphDescriptorCritic, descriptor_features,
                                     fit_descriptor_critic, flip_first_response_row,
                                     hard_graph_probabilities, optimize_terminal_control)
from emf.masked_edit_data import _features
from emf.masked_edit_model import MaskedEditCondition


def _condition():
    masks = GraphMasks.from_counts(torch.tensor([4]), 4)
    edit = GraphMasks(torch.tensor([[1, 1, 0, 0]], dtype=torch.bool),
                      torch.tensor([[1, 0, 0, 0, 0, 0]], dtype=torch.bool))
    graph = GraphBatch(torch.zeros(1, 4, dtype=torch.long),
                       torch.zeros(1, 6, dtype=torch.long), masks)
    lead = Latents(torch.zeros(1, 4, 2), torch.zeros(1, 6, 1))
    return MaskedEditCondition(torch.zeros(1, 4), torch.zeros(1, 2),
                               torch.zeros(1, 2), torch.zeros(1, 3),
                               torch.zeros(1, 2), lead, graph, edit)


def test_descriptor_features_match_data_exactly_and_flip_one_row():
    raw = np.array([[2., 30.], [1., 20.]])
    lead = np.array([1.5, 15.])
    scales = np.array([2., 10.])
    expected = _features(raw, lead, scales)
    actual = descriptor_features(torch.tensor(raw / scales),
                                 torch.tensor(lead / scales).expand(2, -1))
    np.testing.assert_allclose(actual.numpy(), expected)
    W = torch.arange(12.).reshape(2, 2, 3)
    opposite = flip_first_response_row(W)
    torch.testing.assert_close(opposite[:, 0], -W[:, 0])
    torch.testing.assert_close(opposite[:, 1], W[:, 1])


def test_soft_graph_critic_fit_and_input_gradient():
    config = ModelConfig(max_nodes=4, node_dim=2, pair_dim=1,
                         hidden_dim=16, depth=1, num_heads=4)
    critic = SoftGraphDescriptorCritic(config, hidden_dim=16, depth=1)
    mask = GraphMasks.from_counts(torch.full((4,), 4), 4)
    graph = GraphBatch(torch.tensor([[0, 0, 0, 0], [1, 0, 0, 0],
                                     [0, 1, 0, 0], [1, 1, 0, 0]]),
                       torch.zeros(4, 6, dtype=torch.long), mask)
    probs = hard_graph_probabilities(graph)
    labels = graph.atoms.float().sum(-1, keepdim=True).expand(-1, 2) / 2
    summary = fit_descriptor_critic(critic, probs, mask, labels,
                                    steps=25, batch_size=4, learning_rate=.01)
    assert len(summary["source_rmse_scaled"]) == 2
    logits = GraphLogits(torch.randn(4, 4, 4, requires_grad=True),
                         torch.randn(4, 6, 4, requires_grad=True))
    predicted = critic(logits, mask)
    gradients = torch.autograd.grad(predicted.sum(), (logits.atoms, logits.bonds))
    assert gradients[0].abs().sum() > 0
    assert gradients[1].abs().sum() > 0


class _IdentityFlow(nn.Module):
    variant = "C"

    def __init__(self):
        super().__init__()
        self.unused = nn.Parameter(torch.zeros(()))

    def predict_clean(self, state, time, masks, condition):
        return state

    def decode(self, state, masks):
        atoms = torch.zeros(*state.nodes.shape[:2], 4, device=state.nodes.device)
        bonds = torch.zeros(*state.pairs.shape[:2], 4, device=state.pairs.device)
        atoms[..., 1] = state.nodes[..., 0]
        bonds[..., 1] = state.pairs[..., 0]
        return GraphLogits(atoms, bonds)


class _SimpleCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.unused = nn.Parameter(torch.zeros(()))

    def forward(self, logits, masks):
        return torch.stack((logits.atoms[:, 0].softmax(-1)[:, 1],
                            logits.bonds[:, 0].softmax(-1)[:, 1]), -1)


def test_control_repair_rerollout_counts_and_protected_projection():
    condition = _condition()
    model, critic = _IdentityFlow(), _SimpleCritic()
    W = torch.tensor([[[1., 0., 0.], [0., 1., 0.]]])
    gaussian = Latents(torch.zeros_like(condition.lead.nodes),
                       torch.zeros_like(condition.lead.pairs))
    result = optimize_terminal_control(model, condition, gaussian, critic, W,
                                        torch.zeros(1, 2), torch.ones(2),
                                        torch.tensor([[.26, .26]]),
                                        nfe=4, iterations=2, radii=(.05, .1))
    assert result.forward_nfe >= 3 * 4
    assert result.backward_nfe >= 2 * 4
    assert result.predicted_after[0].sum() > result.predicted_before[0].sum()
    assert result.control_energy[0] > 0
    squared = (result.node_control.square().flatten(2).sum(-1)
               + result.pair_control.square().flatten(2).sum(-1))
    editable_coordinates = (condition.edit_masks.nodes.sum() * condition.lead.nodes.shape[-1]
                            + condition.edit_masks.pairs.sum() * condition.lead.pairs.shape[-1])
    # nfe=4, T=.95: left-Euler weights put two intervals in bin 0, one
    # interval plus the terminal [T,1] correction in bin 1.
    expected = (squared[:, 0] * (2 * .95 / 3)
                + squared[:, 1] * (.95 / 3 + .05)) / editable_coordinates
    torch.testing.assert_close(result.control_energy, expected)
    torch.testing.assert_close(result.control_rms, result.control_energy.sqrt())
    assert torch.equal(result.node_control[:, :, 2:],
                       torch.zeros_like(result.node_control[:, :, 2:]))
    assert torch.equal(result.pair_control[:, :, 1:],
                       torch.zeros_like(result.pair_control[:, :, 1:]))
    torch.testing.assert_close(result.sample.latents.nodes[:, 2:],
                               condition.lead.nodes[:, 2:])
    torch.testing.assert_close(result.sample.latents.pairs[:, 1:],
                               condition.lead.pairs[:, 1:])
    assert model.unused.requires_grad and critic.unused.requires_grad
