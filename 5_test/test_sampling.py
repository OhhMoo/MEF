"""Check the continuous solver against an analytic destination oracle."""

import pytest
import torch
from torch import nn

from emf.contracts import Latents, ModelConfig
from emf.sampling import sample_latents


class DestinationOracle(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = ModelConfig(max_nodes=3, node_dim=2, pair_dim=1,
                                  hidden_dim=8, depth=1, num_heads=2)
        self.calls = []

    def flow(self, z, t, masks):
        self.calls.append((z, t.clone(), masks))
        return Latents(torch.full_like(z.nodes, 0.5),
                       torch.full_like(z.pairs, -0.25)).masked(masks)


def test_euler_trajectory_and_endpoint_follow_exact_straight_path():
    model = DestinationOracle()
    noise = Latents(torch.full((2, 3, 2), 2.0), torch.full((2, 3, 1), 3.0))
    result = sample_latents(model, torch.tensor([1, 3]), nfe=6, T=0.9,
                            initial_noise=noise)
    assert len(model.calls) == result.nfe == 6
    for state, time, masks in model.calls:
        t = time[:, None, None]
        expected_nodes = ((1 - t) * noise.nodes + t * 0.5) * masks.nodes[..., None]
        expected_pairs = ((1 - t) * noise.pairs - t * 0.25) * masks.pairs[..., None]
        torch.testing.assert_close(state.nodes, expected_nodes)
        torch.testing.assert_close(state.pairs, expected_pairs)
        assert torch.count_nonzero(state.nodes[~masks.nodes]) == 0
        assert torch.count_nonzero(state.pairs[~masks.pairs]) == 0
    torch.testing.assert_close(result.latents.nodes,
                               torch.full_like(noise.nodes, 0.5) * result.masks.nodes[..., None])
    torch.testing.assert_close(result.latents.pairs,
                               torch.full_like(noise.pairs, -0.25) * result.masks.pairs[..., None])
    assert model.calls[-1][1].tolist() == pytest.approx([0.9, 0.9])
    assert result.counts.tolist() == [1, 3]


def test_owned_generator_is_reproducible_without_changing_global_rng():
    global_rng = torch.get_rng_state().clone()
    first, second = DestinationOracle(), DestinationOracle()
    sample_latents(first, torch.tensor([2]), nfe=3,
                   generator=torch.Generator().manual_seed(11))
    sample_latents(second, torch.tensor([2]), nfe=3,
                   generator=torch.Generator().manual_seed(11))
    torch.testing.assert_close(first.calls[0][0].nodes, second.calls[0][0].nodes)
    torch.testing.assert_close(first.calls[0][0].pairs, second.calls[0][0].pairs)
    assert torch.equal(global_rng, torch.get_rng_state())


@pytest.mark.parametrize("kwargs", [{"nfe": 1}, {"nfe": 2.5}, {"nfe": True},
                                    {"T": 1.0}, {"T": 0.0}, {"T": float("nan")}])
def test_invalid_solver_budget_and_endpoint_are_rejected(kwargs):
    with pytest.raises(ValueError):
        sample_latents(DestinationOracle(), torch.tensor([2]), **kwargs)


@pytest.mark.parametrize("counts", [torch.tensor([0]), torch.tensor([4]),
                                    torch.tensor([1.5]), torch.tensor([[2]])])
def test_invalid_graph_counts_are_rejected(counts):
    with pytest.raises(ValueError):
        sample_latents(DestinationOracle(), counts)
