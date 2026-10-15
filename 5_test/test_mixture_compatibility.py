"""Exact finite-prior posterior and conditional categorical-risk identities."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from emf.contracts import GraphBatch, GraphLogits, GraphMasks
from emf.objectives import categorical_loss

CONTROL = Path(__file__).resolve().parents[1] / "4_experimental_result/control"
sys.path.insert(0, str(CONTROL))
SPEC = importlib.util.spec_from_file_location("mixture_compatibility", CONTROL / "mixture_compatibility.py")
mixture = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mixture)


def test_at_zero_time_posterior_is_uniform_for_every_observation():
    support = torch.tensor([[1., 2., 3.], [-7., 4., 1.], [12., -3., 0.]])
    observed = torch.tensor([[.1, 0., -1.], [-2., 4., 3.]])
    weights = mixture.posterior_weights(observed, support, 0)
    torch.testing.assert_close(weights, torch.full((2, 3), 1 / 3, dtype=torch.float64))
    torch.testing.assert_close(weights @ support.double(), support.double().mean(0).expand(2, -1))
    torch.testing.assert_close(mixture._entropy(weights), torch.full((2,), torch.log(torch.tensor(3.)).item(), dtype=torch.float64))


def test_one_point_mixture_has_exact_mean_and_zero_conditional_entropy():
    support = torch.tensor([[1., -2.]])
    for time in (0., .01, .5, .9):
        weights = mixture.posterior_weights(torch.tensor([[7., -8.], [-1., 0.]]), support, time)
        torch.testing.assert_close(weights, torch.ones(2, 1, dtype=torch.float64))
        torch.testing.assert_close(weights @ support.double(), support.double().expand(2, -1))
        assert not mixture._entropy(weights).any()
    with pytest.raises(ValueError, match="0 <= t < 1"):
        mixture.posterior_weights(support, support, 1.)


def test_soft_ce_and_gradient_equal_weighted_production_hard_ce_with_padding():
    torch.manual_seed(15)
    masks = GraphMasks.from_counts(torch.tensor([3]), 4)
    atom_labels = torch.tensor([[0, 1, 2, 3], [1, 0, 2, 1], [3, 1, 0, 2]])
    pair_labels = torch.tensor([[1, 2, 3, 0, 1, 3], [0, 1, 2, 1, 3, 2], [3, 0, 1, 2, 0, 1]])
    weights = torch.tensor([.2, .3, .5])
    probabilities = GraphLogits(
        (weights[:, None, None] * F.one_hot(atom_labels, 4)).sum(0, keepdim=True),
        (weights[:, None, None] * F.one_hot(pair_labels, 4)).sum(0, keepdim=True))
    logits = GraphLogits(torch.randn(1, 4, 4, requires_grad=True), torch.randn(1, 6, 4, requires_grad=True))
    soft, _ = mixture.soft_categorical_loss(logits, probabilities, masks)
    expected = sum(weight * categorical_loss(logits, GraphBatch(atom_labels[k:k + 1],
                                      pair_labels[k:k + 1], masks))[0] for k, weight in enumerate(weights))
    torch.testing.assert_close(soft.mean(), expected)
    actual_gradient = torch.autograd.grad(soft.mean(), (logits.atoms, logits.bonds), retain_graph=True)
    expected_gradient = torch.autograd.grad(expected, (logits.atoms, logits.bonds))
    for actual, expected in zip(actual_gradient, expected_gradient):
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
    assert not actual_gradient[0][~masks.nodes].any()
    assert not actual_gradient[1][~masks.pairs].any()


def test_compatible_categorical_decoder_is_stationary_at_conditional_marginals():
    masks = GraphMasks.from_counts(torch.tensor([2]), 2)
    atoms = torch.tensor([[[.2, .3, .1, .4], [.1, .2, .5, .2]]])
    pairs = torch.tensor([[[.4, .1, .2, .3]]])
    logits = GraphLogits(atoms.log().requires_grad_(), pairs.log().requires_grad_())
    loss, _ = mixture.soft_categorical_loss(logits, GraphLogits(atoms, pairs), masks)
    optimum = mixture._entropy(atoms).mean(1) + mixture._entropy(pairs).mean(1)
    torch.testing.assert_close(loss, optimum)
    gradients = torch.autograd.grad(loss.sum(), (logits.atoms, logits.bonds))
    assert max(gradient.abs().max().item() for gradient in gradients) < 1e-7
