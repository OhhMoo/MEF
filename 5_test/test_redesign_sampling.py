"""Analytic deployment checks for explicit conditioning and paired randomness."""

import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from emf.contracts import Latents, ModelConfig
from emf.redesign_contracts import FlowCondition
from emf.redesign_sampling import (frozen_deployment, make_sampling_plan,
                                    sample_redesign_latents)


class CleanDestination(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.5))
        self.config = ModelConfig(max_nodes=3, node_dim=2, pair_dim=1,
                                  hidden_dim=8, depth=1, num_heads=2)
        self.calls = []

    def predict_clean(self, z, t, masks, condition):
        assert not self.training and not self.weight.requires_grad
        self.calls.append((z, t.clone(), masks, condition))
        return Latents(torch.full_like(z.nodes, 0.5),
                       torch.full_like(z.pairs, -0.25)).masked(masks)


def condition(batch=1):
    return FlowCondition(torch.zeros(batch, 4), torch.zeros(batch, 2),
                         torch.zeros(batch, 4), torch.ones(batch, 8),
                         torch.tensor([[0.5, -0.1]]).expand(batch, -1))


def test_31_steps_terminal_prediction_and_condition_are_exactly_counted():
    model = CleanDestination()
    before = model.weight.detach().clone()
    noise = Latents(torch.full((2, 3, 2), 2.0), torch.full((2, 3, 1), 3.0))
    query = condition(2)
    with frozen_deployment(model):
        result = sample_redesign_latents(model, torch.tensor([1, 3]), query,
                                         nfe=32, T=0.95, initial_noise=noise)
    assert result.nfe == len(model.calls) == 32
    for state, time, masks, supplied in model.calls:
        assert supplied is query
        t = time[:, None, None]
        torch.testing.assert_close(state.nodes, ((1 - t) * noise.nodes + t * 0.5) * masks.nodes[..., None])
        torch.testing.assert_close(state.pairs, ((1 - t) * noise.pairs - t * 0.25) * masks.pairs[..., None])
    torch.testing.assert_close(result.latents.nodes, torch.full_like(noise.nodes, 0.5) * result.masks.nodes[..., None])
    torch.testing.assert_close(result.latents.pairs, torch.full_like(noise.pairs, -0.25) * result.masks.pairs[..., None])
    assert model.calls[-1][1].tolist() == pytest.approx([0.95, 0.95])
    assert model.training and model.weight.requires_grad
    torch.testing.assert_close(before, model.weight)


def test_count_prior_is_lead_conditioned_and_owned_rngs_are_repeatable():
    model = CleanDestination()
    prior = torch.eye(3)
    global_rng = torch.get_rng_state().clone()
    first = make_sampling_plan(model.config, prior, 1, 17, count_seed=9, noise_seed=11)
    second = make_sampling_plan(model.config, prior, 1, 17, count_seed=9, noise_seed=11)
    third = make_sampling_plan(model.config, prior, 3, 17, count_seed=9, noise_seed=11)
    assert first.counts.tolist() == [1] * 17
    assert third.counts.tolist() == [3] * 17
    torch.testing.assert_close(first.noise.nodes, second.noise.nodes)
    torch.testing.assert_close(first.noise.pairs, second.noise.pairs)
    torch.testing.assert_close(first.noise.nodes, third.noise.nodes)
    assert torch.equal(global_rng, torch.get_rng_state())
    with pytest.raises(ValueError, match="do not support"):
        make_sampling_plan(model.config, torch.zeros(3, 3), 1, 1)


@pytest.mark.parametrize("arguments", [{"nfe": 1}, {"nfe": True}, {"nfe": 2.5},
                                       {"T": 1}, {"T": math.nan}])
def test_bad_solver_arguments_are_rejected(arguments):
    model = CleanDestination()
    with frozen_deployment(model), pytest.raises(ValueError):
        sample_redesign_latents(model, torch.tensor([2]), condition(), **arguments)


def test_unfrozen_deployment_is_rejected():
    with pytest.raises(ValueError, match="frozen"):
        sample_redesign_latents(CleanDestination(), torch.tensor([2]), condition())


def test_base_only_excludes_selector_correction_without_mutating_weights():
    model = CleanDestination()
    calls = []

    def decompose(z, t, masks, supplied):
        calls.append(supplied)
        return SimpleNamespace(base=Latents(torch.zeros_like(z.nodes), torch.zeros_like(z.pairs)))

    model.flow = SimpleNamespace(decompose=decompose)
    with frozen_deployment(model):
        result = sample_redesign_latents(model, torch.tensor([2]), condition(), nfe=4, base_only=True)
    assert len(calls) == 4 and not model.calls
    assert not result.latents.nodes.any() and not result.latents.pairs.any()
