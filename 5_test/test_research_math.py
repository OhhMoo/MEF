"""Independent analytic checks of the proposed finite-mixture explanation."""

import importlib.util
import sys
from pathlib import Path

import torch

CONTROL = Path(__file__).resolve().parents[1] / "4_experimental_result/control"
sys.path.insert(0, str(CONTROL))
SPEC = importlib.util.spec_from_file_location("reviewer_mixture", CONTROL / "mixture_compatibility.py")
mixture = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mixture)


def test_binary_positive_time_posterior_equals_independent_logistic_identity():
    support = torch.tensor([[-1.], [1.]], dtype=torch.float64)
    z = torch.tensor([[-2.], [-.2], [0.], [.3], [2.]], dtype=torch.float64)
    for time in (0., .01, .3, .7, .9):
        weights = mixture.posterior_weights(z, support, time)
        probability = torch.sigmoid(2 * time * z[:, 0] / (1 - time)**2)
        torch.testing.assert_close(weights[:, 1], probability, rtol=1e-12, atol=1e-14)
        torch.testing.assert_close((weights @ support)[:, 0], 2 * probability - 1,
                                   rtol=1e-12, atol=1e-14)


def test_nonlinear_and_barycentric_decoders_differ_at_the_same_exact_mean():
    # At this posterior, the flow optimum is mu=.6. q(y)=sigmoid(3y)
    # disagrees with the categorical marginal; q(y)=(y+1)/2 preserves it.
    probability = torch.tensor(.8, dtype=torch.float64)
    mu = (2 * probability - 1).detach().requires_grad_()
    nonlinear = torch.sigmoid(3 * mu)
    nonlinear_ce = -probability * nonlinear.log() - (1 - probability) * (1 - nonlinear).log()
    gradient = torch.autograd.grad(nonlinear_ce, mu)[0]
    torch.testing.assert_close(gradient, 3 * (nonlinear.detach() - probability))
    assert gradient.abs() > .1
    affine_mu = mu.detach().requires_grad_()
    barycentric = (affine_mu + 1) / 2
    affine_ce = -probability * barycentric.log() - (1 - probability) * (1 - barycentric).log()
    affine_gradient = torch.autograd.grad(affine_ce, affine_mu)[0]
    assert affine_gradient.abs() < 1e-12
    # Nonlinearity alone does not imply conflict: the symmetric t=0 posterior
    # has mu=0, and the same nonlinear decoder is stationary there.
    zero = torch.zeros((), dtype=torch.float64, requires_grad=True)
    half = torch.sigmoid(3 * zero)
    symmetric_ce = -.5 * half.log() - .5 * (1 - half).log()
    assert torch.autograd.grad(symmetric_ce, zero)[0].item() == 0
