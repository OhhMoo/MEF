"""Feasible-goal weighting preserves the original batch and active coordinates."""

import pytest
import torch

from emf.contracts import GraphMasks, Latents
from emf.redesign_objectives import goal_satisfied, weighted_velocity_matching_loss


def test_joint_profile_uses_absolute_preservation_threshold():
    baseline = torch.tensor([[0., 1.], [0., 2.], [0., 1.]])
    changes = torch.tensor([[.3, -.2], [.3, -.2], [.1, .1]])
    goals = torch.tensor([[.2, .9], [.2, .9], [.2, .9]])
    assert goal_satisfied(changes, baseline, goals).tolist() == [False, True, False]


def test_weighted_velocity_uses_fixed_batch_denominator_and_masked_coordinate_count():
    masks = GraphMasks.from_counts(torch.tensor([2, 1]), 2)
    clean = Latents(torch.zeros(2, 2, 2), torch.zeros(2, 1, 1))
    predicted = Latents(torch.ones(2, 2, 2, requires_grad=True), torch.ones(2, 1, 1, requires_grad=True))
    time = torch.tensor([.5, .25])
    loss = weighted_velocity_matching_loss(predicted, clean, time, masks, torch.tensor([True, False]))
    assert loss == 2  # One velocity error of four, divided by the original two queries.
    nodes, pairs = torch.autograd.grad(loss, (predicted.nodes, predicted.pairs))
    torch.testing.assert_close(nodes[0], torch.full((2, 2), .8))
    torch.testing.assert_close(pairs[0], torch.full((1, 1), .8))
    assert not nodes[1].any() and not pairs[1].any()
    rejected = weighted_velocity_matching_loss(predicted, clean, time, masks, torch.zeros(2))
    assert rejected.requires_grad and rejected == 0
    grads = torch.autograd.grad(rejected, (predicted.nodes, predicted.pairs))
    assert all(not grad.any() for grad in grads)
    with pytest.raises(ValueError, match='zero/one'):
        weighted_velocity_matching_loss(predicted, clean, time, masks, torch.tensor([.3, 0]))
