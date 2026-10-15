"""Mathematical and operational checks for the small ELF-derived graph model."""

import pytest
import torch

from emf import objectives
from emf.contracts import (GraphBatch, GraphMasks, Latents, ModelConfig, pair_indices)
from emf.models import EMFModel, LatentNormalizer
from emf.objectives import (categorical_loss, flow_decoder_loss, interpolate_latents,
                            reconstruction_loss, velocity_matching_loss)


@pytest.fixture
def batch():
    masks = GraphMasks.from_counts(torch.tensor([4, 2, 1]), max_nodes=4)
    return GraphBatch(torch.tensor([[0, 0, 1, 2], [0, 3, 0, 0], [1, 0, 0, 0]]),
                      torch.tensor([[1, 0, 0, 2, 0, 1], [1, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0]]), masks)


@pytest.fixture
def model():
    torch.manual_seed(7)
    return EMFModel(ModelConfig(max_nodes=4, node_dim=8, pair_dim=4,
                                hidden_dim=16, depth=2, num_heads=2))


def _permute_pairs(values, permutation):
    n = len(permutation)
    i, j = pair_indices(n)
    full = values.new_zeros((values.shape[0], n, n, *values.shape[2:]))
    full[:, i, j], full[:, j, i] = values, values
    full = full[:, permutation][:, :, permutation]
    return full[:, i, j]


def _permute_latents(latents, permutation):
    return Latents(latents.nodes[:, permutation], _permute_pairs(latents.pairs, permutation))


def _assert_latents_close(actual, expected):
    torch.testing.assert_close(actual.nodes, expected.nodes, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(actual.pairs, expected.pairs, atol=2e-5, rtol=2e-5)


def test_atom_permutation_moves_nodes_pairs_and_predictions_together(model, batch):
    # Initialize F heads nonzero so equivariance is not tested only on zero output.
    torch.nn.init.normal_(model.flow.node_head.linear.weight, std=.1)
    torch.nn.init.normal_(model.flow.pair_head.linear.weight, std=.1)
    permutation = torch.tensor([2, 0, 3, 1])
    masks = GraphMasks(batch.masks.nodes[:, permutation],
                        _permute_pairs(batch.masks.pairs, permutation))
    permuted = GraphBatch(batch.atoms[:, permutation],
                          _permute_pairs(batch.bonds, permutation), masks)
    clean = model.encode(batch)
    moved_clean = model.encode(permuted)
    _assert_latents_close(moved_clean, _permute_latents(clean, permutation))

    noisy = Latents(torch.randn_like(clean.nodes), torch.randn_like(clean.pairs))
    time = torch.tensor([.1, .4, .95])
    predicted = model.flow(noisy, time, batch.masks)
    moved_prediction = model.flow(_permute_latents(noisy, permutation), time, masks)
    _assert_latents_close(moved_prediction, _permute_latents(predicted, permutation))
    logits = model.decode(noisy, batch.masks)
    moved_logits = model.decode(_permute_latents(noisy, permutation), masks)
    torch.testing.assert_close(moved_logits.atoms, logits.atoms[:, permutation], atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(moved_logits.bonds, _permute_pairs(logits.bonds, permutation),
                               atol=2e-5, rtol=2e-5)


def test_inactive_latents_cannot_change_active_predictions(model, batch):
    torch.nn.init.normal_(model.flow.node_head.linear.weight, std=.1)
    torch.nn.init.normal_(model.flow.pair_head.linear.weight, std=.1)
    clean = model.encode(batch).detach()
    dirty = Latents(clean.nodes + 999 * (~batch.masks.nodes[..., None]),
                    clean.pairs - 999 * (~batch.masks.pairs[..., None]))
    time = torch.tensor([.2, .5, .9])
    _assert_latents_close(model.flow(dirty, time, batch.masks),
                          model.flow(clean, time, batch.masks))
    expected, actual = model.decode(clean, batch.masks), model.decode(dirty, batch.masks)
    torch.testing.assert_close(actual.atoms, expected.atoms)
    torch.testing.assert_close(actual.bonds, expected.bonds)


def test_velocity_loss_matches_interpolation_derivative_per_molecule(model, batch):
    clean = model.encode(batch).detach()
    noise = Latents(torch.randn_like(clean.nodes), torch.randn_like(clean.pairs)).masked(batch.masks)
    prediction = Latents(torch.randn_like(clean.nodes), torch.randn_like(clean.pairs))
    time = torch.tensor([0., .6, .95])
    noisy = interpolate_latents(clean, noise, time, batch.masks)
    measured = velocity_matching_loss(prediction, clean, time, batch.masks)
    expected = []
    for k in range(3):
        node_v = (prediction.nodes[k] - noisy.nodes[k]) / (1 - time[k])
        pair_v = (prediction.pairs[k] - noisy.pairs[k]) / (1 - time[k])
        node_target = clean.nodes[k] - noise.nodes[k]
        pair_target = clean.pairs[k] - noise.pairs[k]
        active_error = torch.cat([(node_v - node_target)[batch.masks.nodes[k]].flatten(),
                                  (pair_v - pair_target)[batch.masks.pairs[k]].flatten()])
        expected.append(active_error.square().mean())
    torch.testing.assert_close(measured, torch.stack(expected).mean())
    assert not noisy.nodes[~batch.masks.nodes].any()
    assert not noisy.pairs[~batch.masks.pairs].any()
    with pytest.raises(ValueError, match="0 <= t < 1"):
        velocity_matching_loss(prediction, clean, torch.ones(3), batch.masks)


def test_normalizer_fits_active_training_features_and_inverts(batch):
    raw = Latents(torch.randn(3, 4, 8) * 3 + 2, torch.randn(3, 6, 4) * 2 - 1)
    normalizer = LatentNormalizer(8, 4)
    _assert_latents_close(normalizer(raw, batch.masks), raw.masked(batch.masks))
    normalizer.fit([(raw, batch.masks)])
    normalized = normalizer(raw, batch.masks)
    for values, mask in ((normalized.nodes, batch.masks.nodes),
                          (normalized.pairs, batch.masks.pairs)):
        active = values[mask]
        torch.testing.assert_close(active.mean(0), torch.zeros(active.shape[-1]), atol=1e-6, rtol=0)
        torch.testing.assert_close(active.std(0, correction=0), torch.ones(active.shape[-1]),
                                   atol=1e-6, rtol=0)
    _assert_latents_close(normalizer.inverse(normalized, batch.masks), raw.masked(batch.masks))
    assert normalizer.fitted


def test_normalization_preserves_warmed_decoder_function(model, batch):
    raw = model.encoder(batch).detach()
    before = model.decoder(raw, batch.masks)
    model.normalizer.fit([(raw, batch.masks)])
    after = model.decode(model.normalizer(raw, batch.masks), batch.masks)
    torch.testing.assert_close(before.atoms, after.atoms, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(before.bonds, after.bonds, atol=2e-5, rtol=2e-5)


def test_warmup_then_flow_has_finite_gradients_and_frozen_encoder(model, batch):
    warmup, _ = reconstruction_loss(model, batch)
    warmup.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters())
    model.zero_grad(set_to_none=True)
    model.freeze_encoder()
    model.train()
    assert not model.encoder.training
    saved = {name: value.clone() for name, value in model.encoder.state_dict().items()}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    loss, metrics = flow_decoder_loss(model, batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(not value.requires_grad for value in metrics.values())
    assert all(p.grad is None for p in model.encoder.parameters())
    for module in (model.flow, model.decoder):
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
        assert any(gradient.abs().sum() > 0 for gradient in gradients)
    optimizer.step()
    for name, value in model.encoder.state_dict().items():
        torch.testing.assert_close(value, saved[name], atol=0, rtol=0)


def test_terminal_categorical_loss_does_not_train_flow_at_zero_lambda(model, batch, monkeypatch):
    model.freeze_encoder()
    # Suppress velocity gradients: any remaining F gradient would come from CE.
    def zero_velocity(predicted, *_):
        return predicted.nodes.sum() * 0 + predicted.pairs.sum() * 0
    monkeypatch.setattr(objectives, "velocity_matching_loss", zero_velocity)
    loss, _ = flow_decoder_loss(model, batch, lambda_mid=0)
    loss.backward()
    assert all(p.grad is None or not p.grad.any() for p in model.flow.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.decoder.parameters())
    model.zero_grad(set_to_none=True)
    supervised, _ = flow_decoder_loss(model, batch, lambda_mid=1)
    supervised.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.flow.parameters())


def test_single_atom_has_finite_losses_with_no_pair_coordinates():
    config = ModelConfig(max_nodes=1, node_dim=8, pair_dim=4, hidden_dim=16, depth=1, num_heads=2)
    model = EMFModel(config)
    masks = GraphMasks.from_counts(torch.tensor([1, 1]), 1)
    batch = GraphBatch(torch.tensor([[0], [1]]), torch.zeros(2, 0, dtype=torch.long), masks)
    reconstruction, metrics = reconstruction_loss(model, batch)
    assert torch.isfinite(reconstruction) and metrics["bond_ce"] == 0
    model.normalizer.fit([(model.encoder(batch).detach(), masks)])
    torch.testing.assert_close(model.normalizer.pair_std, torch.ones(4))
    model.freeze_encoder()
    loss, metrics = flow_decoder_loss(model, batch)
    loss.backward()
    assert torch.isfinite(loss) and metrics["bond_ce"] == 0


def test_default_generator_is_small():
    model = EMFModel(ModelConfig())
    assert sum(parameter.numel() for parameter in model.parameters()) < 10_000_000


def test_bfloat16_trunk_keeps_predictions_logits_and_losses_float32(model, batch):
    model.freeze_encoder()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        clean = model.encode(batch)
        prediction = model.flow(clean, torch.full((3,), .95), batch.masks)
        logits = model.decode(prediction, batch.masks)
        loss, _ = flow_decoder_loss(model, batch)
    assert clean.nodes.dtype == clean.pairs.dtype == torch.float32
    assert prediction.nodes.dtype == prediction.pairs.dtype == torch.float32
    assert logits.atoms.dtype == logits.bonds.dtype == loss.dtype == torch.float32
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
