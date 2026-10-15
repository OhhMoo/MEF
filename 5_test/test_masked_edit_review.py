"""Independent masked flow checks: all states, adjoints, and boundary chemistry."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import numpy as np

from emf.chem import decode_graph, smiles_to_graph
from emf.contracts import GraphLogits, GraphMasks, Latents
from emf.data import collate_graphs
from emf.masked_edit_model import MaskedEditCondition, decode_masked_edit
from emf.masked_edit_sampling import sample_masked_edit_latents


def fixture():
    graph = collate_graphs([dict(smiles_to_graph('CCC', 3), id='lead')])
    edit = GraphMasks(torch.tensor([[True, False, False]]),
                      torch.tensor([[True, False, False]]))
    lead = Latents(torch.tensor([[[.4], [.7], [1.1]]], dtype=torch.float64),
                   torch.tensor([[[.3], [.8], [1.2]]], dtype=torch.float64))
    condition = MaskedEditCondition(torch.zeros(1, 3), torch.zeros(1, 2),
        torch.zeros(1, 2), torch.zeros(1, 2), torch.zeros(1, 2), lead, graph, edit)
    gaussian = Latents(torch.full_like(lead.nodes, .2), torch.full_like(lead.pairs, .1))
    return condition, gaussian


class AffineFlow:
    variant = 'C'

    def __init__(self):
        self.observed = []

    def predict_clean(self, z, t, masks, condition):
        self.observed.append(z)
        signal = condition.response_state[:, :1, None]
        # Base deliberately tries to move every protected coordinate.
        return Latents(.3*z.nodes + signal + .9, .2*z.pairs + signal + .8)


def rollout(gain, truncate_steps=None):
    condition, gaussian = fixture()
    model = AffineFlow()
    def feedback(z, t, supplied):
        state = torch.cat((gain*z.nodes[:, 0], supplied.response_state[:, 1:]), -1)
        return replace(supplied, response_state=state)
    result = sample_masked_edit_latents(model, condition, gaussian, nfe=7,
        controller=feedback, truncate_steps=truncate_steps)
    return result.latents.nodes[:, 0].sum(), model, result, condition


def test_terminal_rollout_gradient_matches_finite_difference_and_full_history():
    gain = torch.tensor(.16, dtype=torch.float64, requires_grad=True)
    value, _, _, _ = rollout(gain)
    derivative = torch.autograd.grad(value, gain)[0]
    epsilon = 1e-5
    plus, *_ = rollout(gain.detach() + epsilon)
    minus, *_ = rollout(gain.detach() - epsilon)
    torch.testing.assert_close(derivative, (plus-minus)/(2*epsilon), rtol=1e-5, atol=1e-7)
    truncated, *_ = rollout(gain, truncate_steps=1)
    short_derivative = torch.autograd.grad(truncated, gain)[0]
    assert abs(float(derivative-short_derivative)) > .01
    assert derivative.isfinite() and derivative.abs() > .01


def test_protected_coordinates_survive_hostile_base_at_every_flow_call():
    _, model, result, condition = rollout(torch.tensor(.16, dtype=torch.float64))
    assert len(model.observed) == result.nfe == 7
    for state in [*model.observed, result.latents]:
        for name in ('nodes', 'pairs'):
            keep = ~getattr(condition.edit_masks, name)
            assert torch.equal(getattr(state, name)[keep], getattr(condition.lead, name)[keep])


def test_terminal_graph_copy_keeps_invalid_boundary_attempt_visible():
    condition, _ = fixture()
    # Protected middle carbon has two fixed adjacent substituents in this
    # representation. Triple bond to editable F creates invalid boundary valence.
    atoms = condition.lead_graph.atoms.clone(); atoms[:, 0] = 3
    bonds = condition.lead_graph.bonds.clone(); bonds[:, 0] = 3
    logits = GraphLogits(torch.full((1, 3, 4), -10.), torch.full((1, 3, 4), -10.))
    logits.atoms.scatter_(-1, atoms[..., None], 10.)
    logits.bonds.scatter_(-1, bonds[..., None], 10.)
    # Deliberate wrong predictions on protected positions must be copied out.
    logits.atoms[:, 1:, 2] = 100.
    logits.bonds[:, 1:, 2] = 100.
    model = SimpleNamespace(variant='C', decode=lambda clean, masks: logits)
    decoded = decode_masked_edit(model, condition.lead, condition)
    got_atoms, got_bonds = decoded.atoms.argmax(-1), decoded.bonds.argmax(-1)
    assert torch.equal(got_atoms[:, 1:], condition.lead_graph.atoms[:, 1:])
    assert torch.equal(got_bonds[:, 1:], condition.lead_graph.bonds[:, 1:])
    assert got_atoms[0, 0] == 3 and got_bonds[0, 0] == 3
    outcome = decode_graph(got_atoms[0].numpy(), got_bonds[0].numpy(), 3, 3)
    assert not outcome.chemically_valid


@pytest.mark.parametrize('field', ['lead', 'lead_graph', 'edit_masks'])
def test_feedback_cannot_silently_change_protected_contract(field):
    condition, gaussian = fixture()
    replacement = {
        'lead': Latents(condition.lead.nodes.clone(), condition.lead.pairs.clone()),
        'lead_graph': replace(condition.lead_graph),
        'edit_masks': GraphMasks(condition.edit_masks.nodes.clone(), condition.edit_masks.pairs.clone()),
    }[field]
    with pytest.raises(ValueError, match='preserve lead'):
        sample_masked_edit_latents(AffineFlow(), condition, gaussian, nfe=3,
            controller=lambda z, t, c: replace(c, **{field: replacement}))


@pytest.mark.parametrize('nfe', [3, 9, 17])
def test_velocity_units_terminal_extrapolation_and_resume_match_analytic_solution(nfe):
    condition, gaussian = fixture()
    class ZeroBase:
        variant = 'C'
        def predict_clean(self, z, t, masks, condition):
            return z
    gain = torch.tensor(.3, dtype=torch.float64, requires_grad=True)
    def control(z, t, c):
        return Latents(torch.ones_like(z.nodes)*gain, torch.ones_like(z.pairs)*gain)
    result = sample_masked_edit_latents(ZeroBase(), condition, gaussian, nfe=nfe,
        velocity_controller=control, return_trace=True)
    expected_nodes = condition.lead.nodes + (gaussian.nodes + gain)*condition.edit_masks.nodes[..., None]
    expected_pairs = condition.lead.pairs + (gaussian.pairs + gain)*condition.edit_masks.pairs[..., None]
    torch.testing.assert_close(result.latents.nodes, expected_nodes, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(result.latents.pairs, expected_pairs, rtol=1e-6, atol=1e-7)
    middle = nfe//2
    resumed = sample_masked_edit_latents(ZeroBase(), condition, nfe=nfe-middle,
        start_t=float(result.times[middle]), initial_state=result.states[middle],
        velocity_controller=control)
    torch.testing.assert_close(resumed.latents.nodes, result.latents.nodes, rtol=1e-6, atol=1e-7)
    derivative = torch.autograd.grad(result.latents.nodes[:, 0].sum(), gain)[0]
    torch.testing.assert_close(derivative, torch.ones_like(gain), rtol=1e-6, atol=1e-7)


def test_prepared_data_hides_other_split_labels_and_reserves_nested_support():
    from emf.masked_edit_data import MaskedEditDataset
    dataset = MaskedEditDataset(split='train')
    sets = list(dataset.molecule_splits.values())
    assert not any(a & b for i, a in enumerate(sets) for b in sets[i+1:])
    assert set.union(*sets) == set(range(len(dataset.smiles)))
    rng = np.random.default_rng(917)
    for i, campaign_id in enumerate(dataset.campaign_ids):
        meta = dataset.manifest['campaigns'][campaign_id]
        mask = dataset.masks[meta['mask_id']]
        allowed = set(meta['support_pool_indices']) | {mask.lead_index}
        outside = sorted(set(range(len(dataset.smiles))) - allowed)
        assert np.isnan(dataset.observations[campaign_id][outside]).all()
        for n in dataset.support_sizes:
            episode = dataset.sample_episode(rng, campaign_index=i, support_size=n)
            assert len(set(episode.support_indices)) == n
            assert not set(episode.query_indices) & set(episode.support_indices)
            assert np.isfinite(episode.support_values).all()
            assert np.isfinite(episode.query_values).all()
            for query in episode.query_graphs:
                assert np.array_equal(query['atoms'][~mask.nodes], mask.lead_atoms[~mask.nodes])
                assert np.array_equal(query['bonds'][~mask.pairs], mask.lead_bonds[~mask.pairs])
        support = dataset.nested_support(i, seed=13)
        longest = support['reserve_indices']
        assert len(set(longest)) == len(longest)
        for n, record in support['records'].items():
            assert np.array_equal(record['indices'], longest[:n])
        assert not set(support['query_indices']) & set(longest)


def test_full_terminal_optimizer_repairs_linear_response_and_counts_rerollouts():
    from emf.masked_edit_control import optimize_terminal_control
    condition, gaussian = fixture()
    condition = replace(condition, lead=Latents(condition.lead.nodes.float(),
                                                 condition.lead.pairs.float()))
    gaussian = Latents(gaussian.nodes.float(), gaussian.pairs.float())
    class Model(torch.nn.Module):
        variant = 'C'
        def __init__(self):
            super().__init__(); self.anchor = torch.nn.Parameter(torch.ones(()))
        def predict_clean(self, state, t, masks, condition):
            return Latents(state.nodes*self.anchor, state.pairs*self.anchor)
        def decode(self, state, masks):
            return GraphLogits(torch.cat((state.nodes, torch.zeros(1, 3, 3)), -1),
                               torch.cat((state.pairs, torch.zeros(1, 3, 3)), -1))
    class Critic(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.anchor = torch.nn.Parameter(torch.ones(()))
        def forward(self, graph, masks):
            return torch.stack((graph.atoms[:, 0, 0], graph.bonds[:, 0, 0]), -1)*self.anchor
    model, critic = Model(), Critic()
    result = optimize_terminal_control(model, condition, gaussian, critic,
        torch.tensor([[[1., 0., 0.], [0., 1., 0.]]]), torch.zeros(1, 2), torch.ones(2),
        torch.ones(1, 2), nfe=3, iterations=1, radii=(2.,))
    torch.testing.assert_close(result.predicted_before, torch.tensor([[.6, .4]]))
    torch.testing.assert_close(result.predicted_after, torch.ones(1, 2), rtol=1e-5, atol=1e-6)
    assert result.forward_nfe == 6  # Base differentiable rollout plus one trial.
    assert result.backward_nfe == 6  # One full rollout VJP for each endpoint.
    assert result.accepted_steps.tolist() == [1]
    assert torch.equal(model.anchor, torch.ones(())) and model.anchor.requires_grad
    assert torch.equal(critic.anchor, torch.ones(())) and critic.anchor.requires_grad
    assert model.training and critic.training
    assert not result.sample.latents.nodes.requires_grad
