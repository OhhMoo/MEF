"""Operational mathematics of the factorized and ordinary conditional model."""

from dataclasses import replace

import pytest
import torch

from emf.contracts import GraphBatch, GraphMasks, Latents, ModelConfig, pair_indices
from emf.models import EMFModel
from emf.redesign import RedesignModel, pool_latents
from emf.redesign_contracts import CampaignBatch, RedesignConfig, SupportBatch
from emf.redesign_objectives import goal_satisfied, redesign_loss, weighted_velocity_matching_loss

torch.set_num_threads(2)


@pytest.fixture
def foundation():
    torch.manual_seed(17)
    config = ModelConfig(max_nodes=4, node_dim=8, pair_dim=4, hidden_dim=16, depth=2, num_heads=2)
    representation = EMFModel(config).eval()
    masks = GraphMasks.from_counts(torch.tensor([4, 2, 1]), 4)
    query = GraphBatch(torch.tensor([[0, 1, 0, 2], [1, 0, 0, 0], [2, 0, 0, 0]]),
                       torch.tensor([[1, 0, 0, 2, 1, 0], [1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]), masks)
    representation.normalizer.fit([(representation.encoder(query).detach(), masks)])
    torch.nn.init.normal_(representation.flow.node_head.linear.weight, std=.1)
    torch.nn.init.normal_(representation.flow.pair_head.linear.weight, std=.1)
    index = torch.tensor([1, 0, 2])
    lead = GraphBatch(query.atoms[index], query.bonds[index],
                      GraphMasks(masks.nodes[index], masks.pairs[index]))
    return representation, query, lead


def build_episode(foundation, variant='factorized'):
    representation, query, lead = foundation
    torch.manual_seed(29)
    config = RedesignConfig(model=representation.config, rank=2, state_dim=6, context_dim=3, variant=variant)
    model = RedesignModel(config, representation)
    features = model.graph_features(query)
    support = SupportBatch(features[torch.tensor([[0, 1, 2], [1, 2, 0]])],
                           torch.tensor([[[0., 1.], [1., 2.], [-1., .5]], [[1., 2.], [0., 1.], [0., 0.]]]),
                           torch.tensor([[.2, .1, -.3], [.1, .4, .2]]),
                           torch.tensor([[True, True, True], [True, True, False]]))
    episode_index = torch.tensor([0, 0, 1])
    baseline = torch.tensor([[0., 1.], [0., 1.], [0., 1.]])
    context = support.context[episode_index]
    goal = torch.tensor([[.2, .9], [.2, .9], [.2, .9]])
    delta = torch.tensor([[.5, 0.], [-.5, .1], [.7, -.4]])
    episode = CampaignBatch(query, lead, support, episode_index, baseline, context, goal, delta,
                            goal_satisfied(delta, baseline, goal))
    return model, episode


def condition(model, episode):
    states = model.support_state(episode.support)[episode.episode_index]
    return model.condition(episode.lead, episode.baseline, episode.context, states, episode.goal)


def permute_pairs(values, permutation):
    i, j = pair_indices(len(permutation))
    full = values.new_zeros((len(values), len(permutation), len(permutation), *values.shape[2:]))
    full[:, i, j], full[:, j, i] = values, values
    full = full[:, permutation][:, :, permutation]
    return full[:, i, j]


@pytest.mark.parametrize('variant', ['factorized', 'conditional'])
def test_initial_prediction_reuses_pretrained_flow_and_candidate_equivariance(foundation, variant):
    model, episode = build_episode(foundation, variant)
    cond = condition(model, episode)
    latent = Latents(torch.randn(3, 4, 8), torch.randn(3, 6, 4)).masked(episode.query.masks)
    time = torch.tensor([.1, .4, .9])
    predicted = model.predict_clean(latent, time, episode.query.masks, cond)
    expected = foundation[0].flow(latent, time, episode.query.masks)
    torch.testing.assert_close(predicted.nodes, expected.nodes, rtol=0, atol=0)
    torch.testing.assert_close(predicted.pairs, expected.pairs, rtol=0, atol=0)
    # Activate task paths so this check includes conditioning, not only initial F.
    head = model.flow.selector[-1] if variant == 'factorized' else model.flow.task_projection[-1]
    torch.nn.init.normal_(head.weight, std=.03)
    permutation = torch.tensor([2, 0, 3, 1])
    masks = GraphMasks(episode.query.masks.nodes[:, permutation], permute_pairs(episode.query.masks.pairs, permutation))
    moved = Latents(latent.nodes[:, permutation], permute_pairs(latent.pairs, permutation))
    actual = model.predict_clean(moved, time, masks, cond)
    original = model.predict_clean(latent, time, episode.query.masks, cond)
    torch.testing.assert_close(actual.nodes, original.nodes[:, permutation], atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(actual.pairs, permute_pairs(original.pairs, permutation), atol=2e-5, rtol=2e-5)
    original_pool = pool_latents(latent, episode.query.masks, model.config.max_nodes)
    torch.testing.assert_close(pool_latents(moved, masks, model.config.max_nodes), original_pool)


def test_factorized_chemical_directions_exclude_measurements_and_goals(foundation):
    model, episode = build_episode(foundation)
    torch.nn.init.normal_(model.flow.selector[-1].weight, std=.04)
    cond = condition(model, episode)
    changed = replace(cond, baseline=cond.baseline + 2, context=cond.context - 1,
                       response_state=cond.response_state * 2 + 1, goal=cond.goal + 4)
    latent = model.encode(episode.query).detach()
    time = torch.full((3,), .4)
    original = model.flow.decompose(latent, time, episode.query.masks, cond)
    updated = model.flow.decompose(latent, time, episode.query.masks, changed)
    torch.testing.assert_close(updated.base.nodes, original.base.nodes, rtol=0, atol=0)
    torch.testing.assert_close(updated.node_directions, original.node_directions, rtol=0, atol=0)
    torch.testing.assert_close(updated.pair_directions, original.pair_directions, rtol=0, atol=0)
    assert not torch.allclose(original.strengths, updated.strengths)


def test_nonzero_basis_learns_selector_then_routes_flow_signal_to_evidence(foundation):
    model, episode = build_episode(foundation)
    clean = model.encode(episode.query).detach()
    latent = Latents(torch.randn_like(clean.nodes), torch.randn_like(clean.pairs)).masked(episode.query.masks)
    time = torch.full((3,), .3)
    parts = model.flow.decompose(latent, time, episode.query.masks, condition(model, episode))
    assert not parts.strengths.any() and parts.node_directions.abs().sum() > 0
    loss = weighted_velocity_matching_loss(parts.compose(), clean, time, episode.query.masks, torch.ones(3))
    loss.backward()
    assert model.flow.selector[-1].weight.grad.abs().sum() > 0
    optimizer = torch.optim.SGD(model.flow.selector.parameters(), lr=.05)
    optimizer.step()
    model.zero_grad(set_to_none=True)
    prediction = model.predict_clean(latent, time, episode.query.masks, condition(model, episode))
    second = weighted_velocity_matching_loss(prediction, clean, time, episode.query.masks, torch.ones(3))
    second.backward()
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in model.evidence.parameters())
    assert all(parameter.grad is None for parameter in model.encoder.parameters())


@pytest.mark.parametrize('variant', ['factorized', 'conditional'])
def test_full_objective_trains_source_functions_and_all_rejected_batch_still_learns(foundation, variant):
    model, episode = build_episode(foundation, variant)
    encoder = {key: value.clone() for key, value in model.encoder.state_dict().items()}
    normalizer = {key: value.clone() for key, value in model.normalizer.state_dict().items()}
    loss, metrics = redesign_loss(model, episode, generator=torch.Generator().manual_seed(71))
    loss.backward()
    assert torch.isfinite(loss) and metrics['accepted_fraction'] == pytest.approx(1 / 3)
    for module in (model.flow, model.evidence, model.response, model.decoder):
        assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in module.parameters())
        assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in module.parameters())
    assert all(not value.requires_grad for value in metrics.values())
    model.zero_grad(set_to_none=True)
    rejected_goal = episode.goal.clone()
    rejected_goal[:, 0] = 999
    rejected = replace(episode, goal=rejected_goal, flow_weights=torch.zeros(3, dtype=torch.bool))
    failed_loss, failed_metrics = redesign_loss(model, rejected, generator=torch.Generator().manual_seed(71))
    failed_loss.backward()
    assert failed_metrics['flow_loss'] == 0 and failed_metrics['accepted_queries'] == 0
    assert all(parameter.grad is None or not parameter.grad.any() for parameter in model.flow.parameters())
    for module in (model.evidence, model.response, model.decoder):
        assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in module.parameters())
    for saved, module in ((encoder, model.encoder), (normalizer, model.normalizer)):
        for key, value in module.state_dict().items():
            torch.testing.assert_close(value, saved[key], rtol=0, atol=0)


def test_goal_has_no_path_to_response_and_deployment_changes_no_weights(foundation):
    model, episode = build_episode(foundation)
    goals = episode.goal.clone().requires_grad_()
    episode = replace(episode, goal=goals)
    cond = condition(model, episode)
    response = model.predict_response(episode.lead, episode.query, cond.baseline, cond.context, cond.response_state)
    dependency = torch.autograd.grad(response.mean.sum() + response.covariance_matrix.sum(), goals, allow_unused=True)[0]
    assert dependency is None
    model.freeze_deployment()
    model.train()
    assert not model.training and all(not parameter.requires_grad for parameter in model.parameters())
    saved = {key: value.clone() for key, value in model.state_dict().items()}
    clean = model.encode(episode.query)
    prediction = model.predict_clean(clean, torch.full((3,), .95), episode.query.masks, condition(model, episode))
    model.decode(prediction, episode.query.masks)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, saved[key], rtol=0, atol=0)


def test_paired_noise_is_independent_of_variant_initialization(foundation):
    factorized, episode = build_episode(foundation, 'factorized')
    conditional, ordinary_episode = build_episode(foundation, 'conditional')
    global_rng = torch.get_rng_state().clone()
    _, factor_metrics = redesign_loss(factorized, episode, generator=torch.Generator().manual_seed(89))
    _, ordinary_metrics = redesign_loss(conditional, ordinary_episode, generator=torch.Generator().manual_seed(89))
    for key in ('flow_loss', 'response_loss', 'decode_loss'):
        torch.testing.assert_close(factor_metrics[key], ordinary_metrics[key], rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), global_rng, rtol=0, atol=0)


def test_default_variants_have_comparable_parameter_budgets():
    representation = EMFModel(ModelConfig())
    a = RedesignModel(RedesignConfig(variant='factorized'), representation)
    b = RedesignModel(RedesignConfig(variant='conditional'), representation)
    totals = [sum(parameter.numel() for parameter in model.parameters()) for model in (a, b)]
    flows = [sum(parameter.numel() for parameter in model.flow.parameters()) for model in (a, b)]
    assert abs(totals[0] - totals[1]) / totals[0] < .01
    assert abs(flows[0] - flows[1]) / flows[0] < .01
    assert totals[0] < 6_000_000


def test_frozen_encoder_cache_and_source_amp_use_one_fp32_map(foundation):
    model, episode = build_episode(foundation)
    cached = model.graph_features(episode.query)
    clean = model.encode(episode.query)
    with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
        source_target = model.encode(episode.query)
        deployment_features = model.graph_features(episode.query)
    assert source_target.nodes.dtype == source_target.pairs.dtype == torch.float32
    torch.testing.assert_close(source_target.nodes, clean.nodes, rtol=0, atol=0)
    torch.testing.assert_close(source_target.pairs, clean.pairs, rtol=0, atol=0)
    torch.testing.assert_close(deployment_features, cached, rtol=0, atol=0)


def test_single_atom_factorization_and_objective_have_no_pair_failure():
    graph = ModelConfig(max_nodes=1, node_dim=4, pair_dim=2, hidden_dim=8, depth=1, num_heads=2)
    model = RedesignModel(RedesignConfig(model=graph, rank=2, state_dim=4, context_dim=2))
    masks = GraphMasks.from_counts(torch.tensor([1, 1]), 1)
    query = GraphBatch(torch.tensor([[0], [1]]), torch.empty(2, 0, dtype=torch.long), masks)
    features = model.graph_features(query)
    support = SupportBatch(features[None], torch.zeros(1, 2, 2), torch.zeros(1, 2), torch.ones(1, 2, dtype=torch.bool))
    episode = CampaignBatch(query, query, support, torch.zeros(2, dtype=torch.long),
                            torch.zeros(2, 2), torch.zeros(2, 2), torch.zeros(2, 2), torch.zeros(2, 2),
                            torch.ones(2, dtype=torch.bool))
    with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
        loss, metrics = redesign_loss(model, episode, generator=torch.Generator().manual_seed(3))
    assert loss.dtype == torch.float32 and torch.isfinite(loss) and metrics['bond_ce'] == 0
    loss.backward()
