"""Support-only controls preserve measurement budgets and source parameters."""

import importlib.util
from pathlib import Path

import pytest
import torch

from emf.chem import smiles_to_graph
from emf.contracts import ModelConfig
from emf.data import collate_graphs
from emf.redesign import RedesignModel
from emf.redesign_contracts import RedesignConfig, SupportBatch

PATH = Path(__file__).resolve().parents[1] / '4_experimental_result/control/redesign_adaptation.py'
SPEC = importlib.util.spec_from_file_location('redesign_adaptation', PATH)
adaptation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adaptation)
torch.set_num_threads(2)


def fixture(smiles=('CCCO', 'CCCCO', 'CCN', 'CCCN', 'CO', 'CN')):
    torch.manual_seed(119)
    graph = ModelConfig(max_nodes=9, node_dim=4, pair_dim=2, hidden_dim=8, depth=1, num_heads=2)
    model = RedesignModel(RedesignConfig(model=graph, rank=2, state_dim=4, context_dim=2))
    records = []
    for i, identity in enumerate(smiles):
        record = smiles_to_graph(identity)
        record['id'] = f'known-{i}'
        records.append(record)
    graphs = collate_graphs(records)
    values = torch.stack([torch.linspace(0, .4, len(smiles)), torch.zeros(len(smiles))], -1)
    support = SupportBatch(model.graph_features(graphs)[None], values[None],
                           torch.zeros(1, 2), torch.ones(1, len(smiles), dtype=torch.bool))
    model.freeze_deployment()
    return model, support, graphs


def snapshot(model):
    return ({name: value.clone() for name, value in model.state_dict().items()},
            [parameter.requires_grad for parameter in model.parameters()],
            [module.training for module in model.modules()])


def unchanged(model, saved):
    tensors, flags, modes = saved
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, tensors[name], rtol=0, atol=0)
    assert [parameter.requires_grad for parameter in model.parameters()] == flags
    assert [module.training for module in model.modules()] == modes
    assert all(parameter.grad is None for parameter in model.parameters())


def test_task_state_changes_only_state_and_reduces_known_pair_likelihood_loss():
    model, support, graphs = fixture()
    saved, rng = snapshot(model), torch.get_rng_state().clone()
    initial = model.support_state(support)
    # Validation may wrap the source in no_grad; the adapter must enable b gradients.
    with torch.no_grad():
        state, metadata = adaptation.adapt_task_state(model, support, graphs, steps=16, learning_rate=.1)
    unchanged(model, saved)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    assert state.shape == (1, 4) and not state.requires_grad
    assert not torch.equal(initial, state)
    assert metadata['final_loss'] < metadata['initial_loss']
    assert metadata['trained_source_parameters'] == 0
    assert metadata['trainable_parameters'] == 4 and metadata['steps'] == 16
    assert metadata['unique_observations'] == 6 and metadata['known_endpoint_measurements'] == 12
    assert metadata['additional_oracle_calls'] == 0 and metadata['source_weights_unchanged']
    assert metadata['goal_free'] and metadata['covariance_floor'] == 1e-4


def test_selector_copy_updates_only_selector_and_freezes_before_return():
    model, support, graphs = fixture()
    saved, rng = snapshot(model), torch.get_rng_state().clone()
    with torch.no_grad():
        adapted, state, metadata = adaptation.adapt_selector(model, support, graphs, steps=8, learning_rate=.01, seed=78)
    unchanged(model, saved)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    assert adapted is not model
    changed = []
    for name, value in adapted.state_dict().items():
        if not torch.equal(value, saved[0][name]):
            changed.append(name)
    assert changed and all(name.startswith('flow.selector.') for name in changed)
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in adapted.parameters())
    assert all(not module.training for module in adapted.modules())
    torch.testing.assert_close(state, model.support_state(support))
    assert metadata['steps'] == 8 and metadata['source_weights_unchanged']
    assert all(name.startswith('flow.selector.') for name in metadata['trainable_names'])
    assert metadata['unique_observations'] == 6 and metadata['additional_oracle_calls'] == 0


def test_closest_fallback_relaxes_only_similarity_and_preserves_observation_count():
    model, support, graphs = fixture(('C', 'N', 'O', 'F'))
    state, metadata = adaptation.adapt_task_state(model, support, graphs, steps=0)
    assert metadata['similarity_fallback'] and metadata['strict_candidate_pairs'] == 0
    assert metadata['pairs_selected'] == 2 and metadata['pair_similarity_max'] == 0
    assert metadata['unique_observations'] == 4 and metadata['known_endpoint_measurements'] == 8
    assert all(pair['count_difference'] <= 2 for pair in metadata['pairs'])
    torch.testing.assert_close(state, model.support_state(support))


def test_no_count_compatible_pair_returns_initial_state_without_adaptation():
    model, support, graphs = fixture(('C', 'CCCCCCCC'))
    state, metadata = adaptation.adapt_task_state(model, support, graphs, steps=8)
    assert metadata['pairs_selected'] == 0 and metadata['steps'] == 0
    assert not metadata['similarity_fallback'] and metadata['final_loss'] is None
    torch.testing.assert_close(state, model.support_state(support))


def test_goal_random_draws_and_pair_construction_do_not_use_support_outcomes():
    model, support, graphs = fixture()
    altered_values = support.values.clone()
    altered_values[..., 0] = altered_values[..., 0] * 30 - 4
    altered = SupportBatch(support.features, altered_values, support.context, support.mask)
    _, _, first = adaptation.adapt_selector(model, support, graphs, steps=3, seed=142)
    _, _, second = adaptation.adapt_selector(model, altered, graphs, steps=3, seed=142)
    assert first['pairs'] == second['pairs']
    assert first['goal_uniform_sha256'] == second['goal_uniform_sha256']


def test_pair_cap_and_support_permutation_have_stable_identity_selection():
    model, support, graphs = fixture()
    _, first = adaptation.adapt_task_state(model, support, graphs, steps=0, max_pairs=2)
    order = torch.tensor([3, 1, 5, 0, 4, 2])
    moved = SupportBatch(support.features[:, order], support.values[:, order], support.context, support.mask[:, order])
    _, second = adaptation.adapt_task_state(model, moved, adaptation._take(graphs, order), steps=0, max_pairs=2)
    assert first['pairs_selected'] <= 2 and first['pairs'] == second['pairs']
