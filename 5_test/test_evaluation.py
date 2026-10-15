"""Exercise terminal decoding, attempt accounting, and output storage."""

import csv
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import emf.evaluation as evaluation
from emf.contracts import GraphLogits, Latents, ModelConfig


class EthaneOracle(nn.Module):
    def __init__(self, nonfinite=False):
        super().__init__()
        self.config = ModelConfig(max_nodes=2, node_dim=2, pair_dim=1,
                                  hidden_dim=8, depth=1, num_heads=2)
        self.encoder = nn.Dropout(0.2)
        self.flow_calls = 0
        self.decode_calls = 0
        self.nonfinite = nonfinite

    def flow(self, z, t, masks):
        self.flow_calls += 1
        value = float("nan") if self.nonfinite else 0.0
        return Latents(torch.full_like(z.nodes, value), torch.zeros_like(z.pairs))

    def decode(self, latents, masks):
        self.decode_calls += 1
        assert not self.training
        atoms = torch.zeros((*masks.nodes.shape, 4))
        bonds = torch.zeros((*masks.pairs.shape, 4))
        atoms[..., 0] = 1.0  # carbon
        bonds[..., 1] = 1.0  # single bond
        return GraphLogits(atoms, bonds)


def test_complete_chemistry_path_one_decode_and_all_attempts_saved(tmp_path):
    model = EthaneOracle().train()
    model.encoder.eval()  # Frozen submodule mode must also survive evaluation.
    ledger, summary = evaluation.sample_model(model, [0, 1], num_samples=5,
                                               nfe=4, batch_size=2,
                                               training_smiles=("CC",), output_dir=tmp_path)
    assert len(ledger) == summary["attempts"] == 5
    assert model.flow_calls == 3 * 4
    assert model.decode_calls == 3
    assert model.training and not model.encoder.training
    assert all(row["smiles"] == "CC" for row in ledger)
    assert [row["duplicate"] for row in ledger] == [False, True, True, True, True]
    assert all(row["nfe"] == 4 and row["atom_count"] == 2 for row in ledger)
    assert summary["connected_valid"] == 5
    assert summary["distinct_connected_valid"] == 1
    assert summary["novel_distinct_connected_valid"] == 0
    assert summary["generation_seconds"] > 0
    assert sum(row["batch_amortized_seconds"] for row in ledger) == pytest.approx(summary["generation_seconds"])
    loaded = [json.loads(line) for line in (tmp_path / "attempts.jsonl").read_text().splitlines()]
    assert loaded == ledger
    with (tmp_path / "attempts.csv").open(newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    assert len(csv_rows) == 5
    assert json.loads(csv_rows[0]["bonds"]) == ledger[0]["bonds"]
    assert json.loads((tmp_path / "summary.json").read_text()) == summary


def test_mixed_failures_and_duplicates_do_not_change_attempt_denominator(monkeypatch):
    outcomes = iter([
        ("CC", True, True, None), ("CC", True, True, None),
        ("N", True, True, None), ("C.C", True, False, "disconnected"),
        (None, False, False, "sanitize_failed"),
    ])

    def decode(*args, **kwargs):
        smiles, chemically_valid, connected, failure_reason = next(outcomes)
        return SimpleNamespace(smiles=smiles, chemically_valid=chemically_valid,
                               connected=connected, failure_reason=failure_reason)

    monkeypatch.setattr(evaluation, "decode_graph", decode)
    ledger, summary = evaluation.sample_model(EthaneOracle(), [0, 1], num_samples=5,
                                               nfe=2, batch_size=5, training_smiles=("CC",))
    assert summary["attempts"] == 5
    assert summary["chemical_validity_rate"] == 4 / 5
    assert summary["connected_validity_rate"] == 3 / 5
    assert summary["uniqueness_among_connected_valid"] == 2 / 3
    assert summary["novelty_among_distinct_connected_valid"] == 1 / 2
    assert summary["failure_counts"] == {"disconnected": 1, "sanitize_failed": 1}
    assert ledger[3]["smiles"] == "C.C"
    assert ledger[3]["novel"] is None
    assert ledger[4]["duplicate"] is None


def test_nonfinite_predictions_remain_explicit_failures():
    ledger, summary = evaluation.sample_model(EthaneOracle(nonfinite=True), [0, 1],
                                               num_samples=3, nfe=2, batch_size=3)
    assert len(ledger) == 3
    assert summary["failure_counts"] == {"nonfinite_latents": 3}
    assert summary["connected_valid"] == 0
    assert summary["distinct_connected_valid_per_second"] == 0.0


@pytest.mark.parametrize("prior", [[0, 0], [-1, 1], [1], [float("nan"), 1]])
def test_invalid_count_priors_are_rejected(prior):
    with pytest.raises(ValueError):
        evaluation.sample_model(EthaneOracle(), prior, num_samples=1)


def checkpoint_fixture(path, *, phase="complete", fitted=True):
    from emf.models import EMFModel

    config = ModelConfig(max_nodes=2, node_dim=2, pair_dim=1,
                         hidden_dim=8, depth=1, num_heads=2)
    model = EMFModel(config)
    model.normalizer.fitted.fill_(fitted)
    # A single carbon makes saved training-set novelty deterministic.
    with torch.no_grad():
        model.decoder.node_head[-1].weight.zero_()
        model.decoder.node_head[-1].bias.zero_()
        model.decoder.node_head[-1].bias[0] = 1.0
    torch.save({
        "model": model.state_dict(), "phase": phase, "step": 7,
        "count_prior": [1.0, 0.0], "training_smiles": ("C",),
        "data_hash": "test-dataset-identity",
        "config": {"model": asdict(config), "num_samples": 4, "nfe": 4,
                   "batch_size": 2, "T": 0.8, "seed": 9, "device": "cpu",
                   "cpu_threads": 1, "data_dir": "/does/not/exist"},
    }, path)


@pytest.mark.parametrize("overrides,expected", [
    ([], {"attempts": 4, "nfe": 4, "seed": 9, "T": 0.8, "batch_size": 2}),
    (["--num-samples", "3", "--nfe", "3", "--seed", "101", "--T", "0.85",
      "--batch-size", "4"],
     {"attempts": 3, "nfe": 3, "seed": 101, "T": 0.85, "batch_size": 4}),
])
def test_checkpoint_cli_uses_saved_generation_context_without_dataset(tmp_path, overrides, expected):
    checkpoint = tmp_path / "trusted.pt"
    output = tmp_path / "generated"
    checkpoint_fixture(checkpoint)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "3_model_training")
    command = [sys.executable, "-m", "emf.evaluation", "--checkpoint", str(checkpoint),
               "--output-dir", str(output), "--device", "cpu", *overrides]
    result = subprocess.run(command, env=environment, capture_output=True, text=True, check=True)
    summary = json.loads(result.stdout)
    assert summary == json.loads((output / "summary.json").read_text())
    for key, value in expected.items():
        assert summary[key] == value
    assert summary["connected_valid"] == expected["attempts"]
    assert summary["novel_distinct_connected_valid"] == 0
    assert summary["metadata"]["checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert summary["metadata"]["checkpoint_path"] == str(checkpoint.resolve())
    assert summary["metadata"]["count_prior_source"] == "checkpoint.count_prior"
    assert summary["metadata"]["novelty_source"] == "checkpoint.training_smiles"
    rows = [json.loads(line) for line in (output / "attempts.jsonl").read_text().splitlines()]
    assert len(rows) == expected["attempts"]
    assert all(row["smiles"] == "C" and row["atom_count"] == 1 for row in rows)


@pytest.mark.parametrize("phase,fitted", [("warmup", True), ("flow", False)])
def test_incomplete_checkpoints_are_rejected(tmp_path, phase, fitted):
    checkpoint = tmp_path / "incomplete.pt"
    checkpoint_fixture(checkpoint, phase=phase, fitted=fitted)
    with pytest.raises(ValueError):
        evaluation.generate_from_checkpoint(checkpoint, tmp_path / "generated")
