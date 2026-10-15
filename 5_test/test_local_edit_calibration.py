"""Source-only bridge learns an identifiable state and preserves its generator."""

import json
from pathlib import Path

import torch

from emf.contracts import ModelConfig
from emf.local_edit_calibration import (CalibrationBridge, _fit_head, bridge_state,
                                        load_bridge, train_bridge)
from emf.models import EMFModel
from emf.redesign import RedesignModel
from emf.redesign_contracts import RedesignConfig
from emf.redesign_train import tensor_fingerprint
from emf.steering_data import SteeringDataset


ROOT = Path(__file__).resolve().parents[1]


def test_bridge_fits_known_linear_state():
    rng = torch.Generator().manual_seed(42)
    states = torch.randn(256, 64, generator=rng)
    targets = (.5 * states[:, :1] - .25 * states[:, 1:2] + .1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(5)
        head = CalibrationBridge()
    _fit_head(head, states, targets, steps=400, seed=5, learning_rate=3e-3)
    with torch.no_grad():
        prediction = head(states)
    assert (prediction - targets).square().mean() < .01
    inferred = bridge_state(head, states[:3])
    assert torch.allclose(inferred[:, 0], prediction[:3, 0])
    assert torch.count_nonzero(inferred[:, 1:]) == 0


def test_source_training_preserves_generator_and_rejects_heldout(tmp_path):
    data_dir = ROOT / "2_data_storage/processed/evidence_steering_toy"
    dataset = SteeringDataset(data_dir, "train")
    coefficients = json.loads((data_dir / "private/oracle.json").read_text())["coefficients"]
    config = RedesignConfig(ModelConfig(max_nodes=9, node_dim=8, pair_dim=4,
                                        hidden_dim=16, depth=1, num_heads=4),
                            state_dim=64, context_dim=4)
    model = RedesignModel(config, EMFModel(config.model))
    before = tensor_fingerprint(model.state_dict().items())
    checkpoint = tmp_path / "bridge.pt"
    head, summary = train_bridge(model, dataset, coefficients, "cpu", steps=3,
                                 checkpoint_path=checkpoint, seed=7,
                                 episodes_per_size=2, validation_per_size=1)
    assert summary["training_episodes"] == 6
    assert summary["generator_sha256_before"] == summary["generator_sha256_after"] == before
    assert tensor_fingerprint(model.state_dict().items()) == before
    loaded = load_bridge(checkpoint, "cpu")
    probe = torch.randn(3, 64)
    assert torch.allclose(head(probe), loaded(probe))
    heldout = SteeringDataset(data_dir, "test")
    try:
        train_bridge(model, heldout, coefficients, "cpu", steps=1,
                     checkpoint_path=tmp_path / "illegal.pt", episodes_per_size=1,
                     validation_per_size=1)
    except ValueError as error:
        assert "source train split" in str(error)
    else:
        raise AssertionError("held-out bridge training was accepted")
