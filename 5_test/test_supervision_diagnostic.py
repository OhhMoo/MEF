"""Check loss geometry, padding, aggregation, and checkpoint immutability."""

import csv
import importlib.util
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

from emf.contracts import GraphBatch, GraphLogits, GraphMasks, Latents, ModelConfig
from emf.models import EMFModel
from emf.objectives import categorical_loss, interpolate_latents, velocity_matching_loss

PATH = Path(__file__).resolve().parents[1] / "4_experimental_result/control/supervision_diagnostic.py"
SPEC = importlib.util.spec_from_file_location("supervision_diagnostic", PATH)
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)
torch.set_num_threads(2)


@pytest.fixture
def tiny_model_batch():
    torch.manual_seed(91)
    config = ModelConfig(max_nodes=3, node_dim=4, pair_dim=3, hidden_dim=8, depth=1, num_heads=2)
    model = EMFModel(config)
    torch.nn.init.normal_(model.flow.node_head.linear.weight, std=.15)
    torch.nn.init.normal_(model.flow.pair_head.linear.weight, std=.15)
    masks = GraphMasks.from_counts(torch.tensor([3, 2, 1]), 3)
    batch = GraphBatch(torch.tensor([[0, 1, 2], [1, 0, 0], [2, 0, 0]]),
                       torch.tensor([[1, 0, 2], [1, 0, 0], [0, 0, 0]]), masks,
                       ("CNO", "NC", "O"), ("a", "b", "c"))
    model.normalizer.fit([(model.encoder(batch).detach(), masks)])
    model.freeze_encoder()
    model.freeze_decoder()
    model.eval()
    with torch.no_grad():
        clean = model.encode(batch)
    noise = Latents(torch.randn_like(clean.nodes), torch.randn_like(clean.pairs)).masked(masks)
    return model, batch, clean, noise


def test_gradient_cosine_ratios_and_undefined_values():
    flow = torch.tensor([[3., 4.], [3., 4.], [0., 0.], [3., 4.]])
    ce = torch.tensor([[6., 8.], [-6., -8.], [2., 1.], [0., 0.]])
    aligned, conflict, no_flow, no_ce = diagnostic.vector_statistics(flow, ce, coefficient=.1)
    assert aligned["cosine"] == 1
    assert aligned["norm_ratio"] == 2
    assert aligned["weighted_norm_ratio"] == pytest.approx(.2)
    assert aligned["d_flow_along_negative_ce"] == -50
    assert conflict["cosine"] == -1
    assert conflict["d_flow_along_weighted_negative_ce"] == 5
    assert no_flow["cosine"] is None and no_flow["norm_ratio"] is None
    assert no_flow["flow_exactly_zero"] and not no_flow["cosine_defined"]
    assert no_ce["cosine"] is None and no_ce["norm_ratio"] == 0
    assert no_ce["ce_exactly_zero"] and no_ce["norm_ratio_defined"]
    unresolved = diagnostic.vector_statistics(torch.tensor([[1e-14]]), torch.ones(1, 1))[0]
    assert unresolved["cosine"] is None and not unresolved["flow_exactly_zero"]


def test_velocity_gradient_uses_per_molecule_coordinates_and_time_weight():
    masks = GraphMasks.from_counts(torch.tensor([3, 1]), 3)
    clean = Latents(torch.zeros(2, 3, 2), torch.zeros(2, 3, 1))
    predicted = Latents(torch.ones(2, 3, 2, requires_grad=True),
                        torch.ones(2, 3, 1, requires_grad=True))
    time = torch.tensor([.25, .75])
    loss = velocity_matching_loss(predicted, clean, time, masks)
    nodes, pairs = torch.autograd.grad(loss, (predicted.nodes, predicted.pairs))
    # Six node + three pair coordinates in molecule 0; two node coordinates in 1.
    expected0 = 2 / (2 * 9 * .75**2)
    expected1 = 2 / (2 * 2 * .25**2)
    assert nodes[0, 0, 0].item() == pytest.approx(expected0)
    assert pairs[0, 0, 0].item() == pytest.approx(expected0)
    assert nodes[1, 0, 0].item() == pytest.approx(expected1)
    assert not nodes[~masks.nodes].any()
    assert not pairs[~masks.pairs].any()


def test_negative_ce_direction_matches_finite_difference_of_production_flow_loss():
    masks = GraphMasks.from_counts(torch.tensor([2, 1]), 2)
    batch = GraphBatch(torch.zeros(2, 2, dtype=torch.long), torch.zeros(2, 1, dtype=torch.long), masks)
    clean = Latents(torch.zeros(2, 2, 2), torch.zeros(2, 1, 2))
    predicted = Latents(torch.tensor([[[1., -1.], [1., -1.]], [[1., -1.], [999., -999.]]], requires_grad=True),
                        torch.tensor([[[1., -1.]], [[999., -999.]]], requires_grad=True))
    time = torch.full((2,), .4)
    flow = velocity_matching_loss(predicted, clean, time, masks)
    ce, _ = categorical_loss(GraphLogits(predicted.nodes, predicted.pairs), batch)
    g_flow = torch.autograd.grad(flow, (predicted.nodes, predicted.pairs), retain_graph=True)
    g_ce = torch.autograd.grad(ce, (predicted.nodes, predicted.pairs))
    expected = -sum((a * b).sum() for a, b in zip(g_flow, g_ce)).item()
    assert expected > 0  # This deliberate local example has opposing objectives.
    step = .001
    plus = Latents(predicted.nodes.detach() - step * g_ce[0], predicted.pairs.detach() - step * g_ce[1])
    minus = Latents(predicted.nodes.detach() + step * g_ce[0], predicted.pairs.detach() + step * g_ce[1])
    measured = ((velocity_matching_loss(plus, clean, time, masks)
                 - velocity_matching_loss(minus, clean, time, masks)) / (2 * step)).item()
    assert measured == pytest.approx(expected, rel=.002, abs=.0001)
    assert not g_ce[0][~masks.nodes].any()
    assert not g_ce[1][~masks.pairs].any()


def test_per_example_gradient_magnitudes_do_not_depend_on_minibatch_size(tiny_model_batch):
    model, batch, clean, noise = tiny_model_batch
    before = {key: value.clone() for key, value in model.state_dict().items()}
    rng_before = torch.get_rng_state().clone()
    rows, _, _, _ = diagnostic.diagnose_minibatch(model, batch, clean, noise, .3, coefficient=.1)
    keyed = {(row["local_index"], row["ce_component"]): row for row in rows}
    for i in range(3):
        single = diagnostic._slice(batch, i, i + 1, "cpu")
        one_clean = Latents(clean.nodes[i:i + 1], clean.pairs[i:i + 1])
        one_noise = Latents(noise.nodes[i:i + 1], noise.pairs[i:i + 1])
        separate, _, _, _ = diagnostic.diagnose_minibatch(model, single, one_clean, one_noise, .3, coefficient=.1)
        for row in separate:
            joint = keyed[i, row["ce_component"]]
            for field in ("flow_norm", "ce_norm", "flow_dot_ce", "cosine"):
                if row[field] is None:
                    assert joint[field] is None
                else:
                    assert row[field] == pytest.approx(joint[field], rel=5e-4, abs=5e-5)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), rng_before, rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in model.parameters())
    singleton = keyed[2, "bond_ce"]
    assert singleton["ce_exactly_zero"] and singleton["cosine"] is None
    assert singleton["pair_flow_norm"] == 0 and singleton["pair_cosine"] is None


def test_parameter_statistics_match_direct_production_autograd(tiny_model_batch):
    model, batch, clean, noise = tiny_model_batch
    _, _, theta, losses = diagnostic.diagnose_minibatch(
        model, batch, clean, noise, .5, coefficient=1, parameter_gradients=True)
    time = torch.full((3,), .5)
    predicted = model.flow(interpolate_latents(clean, noise, time, batch.masks), time, batch.masks)
    flow = velocity_matching_loss(predicted, clean, time, batch.masks)
    ce, _ = categorical_loss(model.decode(predicted, batch.masks), batch)
    parameters = tuple(model.flow.parameters())
    f = torch.autograd.grad(flow, parameters, retain_graph=True, allow_unused=True)
    c = torch.autograd.grad(ce, parameters, allow_unused=True)
    flat = lambda grads: torch.cat([(torch.zeros_like(p) if g is None else g).flatten()
                                    for p, g in zip(parameters, grads)])[None]
    expected = diagnostic.vector_statistics(flat(f), flat(c))[0]
    actual = next(row for row in theta if row["ce_component"] == "combined_ce")
    for field in ("flow_norm", "ce_norm", "flow_dot_ce", "cosine"):
        assert actual[field] == pytest.approx(expected[field], rel=1e-5, abs=1e-7)
    assert losses["flow_loss"] == pytest.approx(flow.item())
    assert losses["combined_ce"] == pytest.approx(ce.item())
    assert all(parameter.grad is None for parameter in model.parameters())


def test_single_atom_zero_pair_storage_keeps_null_pair_metrics():
    model = EMFModel(ModelConfig(max_nodes=1, node_dim=4, pair_dim=2,
                                hidden_dim=8, depth=1, num_heads=2)).eval()
    model.freeze_encoder()
    model.freeze_decoder()
    masks = GraphMasks.from_counts(torch.ones(2, dtype=torch.long), 1)
    batch = GraphBatch(torch.tensor([[0], [1]]), torch.empty(2, 0, dtype=torch.long), masks)
    with torch.no_grad():
        clean = model.encode(batch)
    noise = Latents(torch.randn_like(clean.nodes), torch.empty_like(clean.pairs))
    rows, decoding, theta, losses = diagnostic.diagnose_minibatch(
        model, batch, clean, noise, .2, coefficient=0, parameter_gradients=True)
    assert losses["bond_ce"] == 0
    assert all(row["ce_norm"] == 0 and row["cosine"] is None
               for row in rows if row["ce_component"] == "bond_ce")
    assert all(row["pair_accuracy"] is None and row["pair_confidence"] is None for row in decoding)
    assert next(row for row in theta if row["ce_component"] == "bond_ce")["ce_norm"] == 0


def test_checkpoint_diagnostic_pairs_noise_and_preserves_artifacts(tmp_path, tiny_model_batch):
    model, batch, _, _ = tiny_model_batch
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "manifest.json").write_text('{}\n')
    np.savez(data_dir / "validation.npz", atoms=batch.atoms.numpy(), bonds=batch.bonds.numpy(),
             counts=batch.counts.numpy(), ids=np.array(batch.ids), smiles=np.array(batch.smiles))
    study_dir = tmp_path / "study"
    study_dir.mkdir()
    tasks, checkpoint_paths = [], []
    for coefficient in (0., .1):
        run = study_dir / str(coefficient)
        (run / "checkpoints").mkdir(parents=True)
        config = {"model": asdict(model.config), "seed": 42, "lambda_mid": coefficient,
                  "freeze_decoder": True, "T": .95, "bf16": False}
        path = run / "checkpoints/flow_0000002.pt"
        torch.save({"config": config, "model": model.state_dict(), "phase": "flow", "step": 2,
                    "data_hash": "same-training-split"}, path)
        checkpoint_paths.append(path)
        tasks.append({"seed": 42, "lambda_mid": coefficient, "decoder_mode": "frozen_common",
                      "output_dir": str(run)})
    study = {"initial_representation": {"model": asdict(model.config)}, "source_sha256": {},
             "reference_data_dir": str(data_dir),
             "dataset_manifest_sha256": diagnostic.sha256(data_dir / "manifest.json"),
             "validation_npz_sha256": diagnostic.sha256(data_dir / "validation.npz")}
    (study_dir / "study.json").write_text(json.dumps(study))
    (study_dir / "tasks.json").write_text(json.dumps(tasks))
    before = [diagnostic.sha256(path) for path in checkpoint_paths]
    rng_before = torch.get_rng_state().clone()
    out = tmp_path / "out"
    manifest = diagnostic.run_diagnostic(study_dir, out, seeds=(42,), coefficients=(0., .1),
                                         step=2, examples=3, batch_size=2, noise_draws=1, times=(.1,),
                                         parameter_gradients=True, parameter_times=(.1,), device="cpu")
    assert manifest["status"] == "complete"
    assert manifest["checkpoint_step"] == 2 and len(manifest["checkpoints"]) == 2
    assert [diagnostic.sha256(path) for path in checkpoint_paths] == before
    torch.testing.assert_close(torch.get_rng_state(), rng_before, rtol=0, atol=0)
    rows = list(csv.DictReader((out / "gradients.csv").open()))
    zero = [row for row in rows if float(row["lambda_mid"]) == 0]
    positive = [row for row in rows if float(row["lambda_mid"]) == .1]
    assert len(zero) == len(positive) == 9
    for a, b in zip(zero, positive):
        assert a["validation_index"] == b["validation_index"]
        assert a["flow_norm"] == b["flow_norm"] and a["ce_norm"] == b["ce_norm"]
    assert any(row["cosine"] == "" for row in zero if row["ce_component"] == "bond_ce")
    with np.load(out / "paired_inputs.npz", allow_pickle=False) as paired:
        assert paired["node_noise"].shape == (1, 3, 3, 4)
        assert len(set(paired["validation_indices"].tolist())) == 3
    assert len(list(csv.DictReader((out / "parameter_gradients.csv").open()))) == 6
