"""Exercise one exploratory generator through training, resume, and chemistry.

Small real molecular fixtures keep these implementation checks inexpensive.
They impose no generation-quality or reconstruction-accuracy prerequisite.
"""

import csv
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from emf.contracts import ModelConfig
from emf.data import prepare_records
from emf.train import RunConfig, run_training


MOLECULES = (
    "C", "N", "O", "F", "CC", "CN", "CO", "CF", "NN", "NO",
    "OO", "C=C", "C#N", "CCC", "CCO", "CCN", "CCF", "COC", "CNC", "FCF",
    "CC=O", "CC(=O)N", "C1CC1", "C1CCC1",
)


@pytest.fixture(autouse=True)
def small_cpu_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def _config(tmp_path, *, device="cpu", bf16=False, output_name="run", num_samples=7):
    data = tmp_path / "fixture_data"
    prepare_records(enumerate(MOLECULES), data, train_limit=None, max_nodes=4, seed=19)
    return RunConfig(
        model=ModelConfig(max_nodes=4, node_dim=8, pair_dim=4,
                          hidden_dim=16, depth=1, num_heads=2),
        data_dir=str(data), output_dir=str(tmp_path / output_name),
        seed=19, batch_size=6, warmup_steps=8, flow_steps=5,
        cpu_threads=2, log_every=1, checkpoint_every=2,
        sample_every=0, num_samples=num_samples, nfe=4,
        validation_size=2, device=device, bf16=bf16,
    )


def _checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _exercise_pipeline(tmp_path, *, device, bf16):
    config = replace(_config(tmp_path, device=device, bf16=bf16), sample_every=2)
    model, result = run_training(config)
    output = tmp_path / "run"
    representation = _checkpoint(output / "checkpoints/representation.pt")
    final = _checkpoint(output / "checkpoints/latest.pt")

    assert final["phase"] == "complete" and final["step"] == config.flow_steps
    assert final["rng"]["batches"].dtype == torch.uint8
    assert final["data_hash"] == representation["data_hash"]
    source_manifest = Path(config.data_dir) / "manifest.json"
    assert (output / "dataset_manifest.json").read_bytes() == source_manifest.read_bytes()
    run_manifest = json.loads((output / "manifest.json").read_text())
    dataset_manifest = json.loads(source_manifest.read_text())
    assert run_manifest["dataset_manifest_sha256"] == hashlib.sha256(source_manifest.read_bytes()).hexdigest()
    assert final["training_smiles"] == tuple(dataset_manifest["splits"]["train"]["smiles"])
    assert final["count_prior"] == run_manifest["count_prior"]
    assert final["count_prior"] == pytest.approx(dataset_manifest["count_prior"])
    assert result["phase"] == "complete"
    assert bool(model.normalizer.fitted)
    assert not model.encoder.training
    assert all(not parameter.requires_grad for parameter in model.encoder.parameters())
    assert all(parameter.requires_grad for parameter in model.decoder.parameters())
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    assert all(math.isfinite(value) for value in result["metrics"].values())

    # The actual transition checkpoint captures E and normalization once. Flow
    # training must preserve both while updating F and its terminal decoder.
    for name, value in final["model"].items():
        if name.startswith(("encoder.", "normalizer.")):
            torch.testing.assert_close(value, representation["model"][name], rtol=0, atol=0)
    for prefix in ("flow.", "decoder."):
        assert any(not torch.equal(value, representation["model"][name])
                   for name, value in final["model"].items() if name.startswith(prefix))

    with (output / "training.csv").open(newline="") as stream:
        updates = list(csv.DictReader(stream))
    assert [row["phase"] for row in updates] == (["warmup"] * config.warmup_steps
                                                + ["flow"] * config.flow_steps)
    assert all(torch.isfinite(torch.tensor(float(row["loss"]))) for row in updates)
    assert all(float(row["mid_loss"]) == 0 for row in updates if row["phase"] == "flow")

    samples = output / f"samples_{config.flow_steps:07d}"
    ledger = [json.loads(line) for line in (samples / "attempts.jsonl").read_text().splitlines()]
    summary = json.loads((samples / "summary.json").read_text())
    assert len(ledger) == summary["attempts"] == config.num_samples
    assert [row["attempt_id"] for row in ledger] == list(range(config.num_samples))
    assert all(row["nfe"] == config.nfe and 1 <= row["atom_count"] <= 4 for row in ledger)
    assert all((row["smiles"] is not None) or (row["failure_reason"] is not None) for row in ledger)
    assert sum(row["batch_amortized_seconds"] for row in ledger) == pytest.approx(summary["generation_seconds"])
    assert "host chemistry" in summary["timing_scope"]
    assert torch.device(summary["device"]).type == device
    # Earlier samples must remain tied to their actual model after latest.pt
    # advances. Check the saved bytes, model state, prior, and generation ledger.
    sampling_checkpoints = []
    for step in (2, 4, config.flow_steps):
        sample_summary = json.loads((output / f"samples_{step:07d}/summary.json").read_text())
        snapshot = Path(sample_summary["metadata"]["checkpoint"])
        assert snapshot.name != "latest.pt"
        assert sample_summary["metadata"]["checkpoint_sha256"] == hashlib.sha256(snapshot.read_bytes()).hexdigest()
        saved = _checkpoint(snapshot)
        assert saved["step"] == step
        assert saved["count_prior"] == final["count_prior"]
        assert saved["training_smiles"] == final["training_smiles"]
        sampling_checkpoints.append(saved)
    assert any(not torch.equal(value, final["model"][name])
               for name, value in sampling_checkpoints[0]["model"].items()
               if name.startswith("flow."))
    for name, value in sampling_checkpoints[-1]["model"].items():
        torch.testing.assert_close(value, final["model"][name], rtol=0, atol=0)
    if device == "cuda":
        assert result["peak_gpu_memory_bytes"] > 0
        assert result["runtime_gpu_hours_this_invocation"] > 0
    return model, result


@pytest.mark.operation
def test_exploratory_training_freezes_representation_and_keeps_every_attempt(tmp_path):
    _exercise_pipeline(tmp_path, device="cpu", bf16=False)


@pytest.mark.operation
def test_interrupted_flow_resume_reproduces_uninterrupted_updates(tmp_path):
    config = _config(tmp_path, output_name="uninterrupted", num_samples=0)
    expected_model, _ = run_training(config)
    interrupted = replace(config, output_dir=str(tmp_path / "interrupted"), flow_steps=2)
    run_training(interrupted)
    resumed_config = replace(interrupted, flow_steps=config.flow_steps)
    resumed_model, _ = run_training(
        resumed_config, resume=tmp_path / "interrupted/checkpoints/latest.pt")
    for name, expected in expected_model.state_dict().items():
        torch.testing.assert_close(resumed_model.state_dict()[name], expected, rtol=0, atol=0)
    expected_checkpoint = _checkpoint(tmp_path / "uninterrupted/checkpoints/latest.pt")
    resumed_checkpoint = _checkpoint(tmp_path / "interrupted/checkpoints/latest.pt")
    assert torch.equal(expected_checkpoint["rng"]["torch"], resumed_checkpoint["rng"]["torch"])
    assert torch.equal(expected_checkpoint["rng"]["batches"], resumed_checkpoint["rng"]["batches"])
    for key, expected in expected_checkpoint["optimizer"]["state"].items():
        for name, value in expected.items():
            torch.testing.assert_close(resumed_checkpoint["optimizer"]["state"][key][name],
                                       value, rtol=0, atol=0)


@pytest.mark.operation
def test_completed_resume_reuses_immutable_checkpoint_and_replays_scientific_ledger(tmp_path):
    config = _config(tmp_path)
    original_model, _ = run_training(config)
    output = tmp_path / "run"
    directory = output / f"samples_{config.flow_steps:07d}"
    summary_before = json.loads((directory / "summary.json").read_text())
    snapshot = Path(summary_before["metadata"]["checkpoint"])
    checkpoint_hash = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    ledger_before = [json.loads(line) for line in (directory / "attempts.jsonl").read_text().splitlines()]

    # Retry generation after the training phase has already completed, as when
    # a job is interrupted after saving its final immutable checkpoint.
    resumed_model, _ = run_training(config, resume=output / "checkpoints/latest.pt")
    assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == checkpoint_hash
    summary_after = json.loads((directory / "summary.json").read_text())
    assert summary_after["metadata"]["checkpoint_sha256"] == checkpoint_hash
    ledger_after = [json.loads(line) for line in (directory / "attempts.jsonl").read_text().splitlines()]
    scientific = lambda rows: [{key: value for key, value in row.items()
                               if key != "batch_amortized_seconds"} for row in rows]
    assert scientific(ledger_after) == scientific(ledger_before)
    for name, value in original_model.state_dict().items():
        torch.testing.assert_close(resumed_model.state_dict()[name], value, rtol=0, atol=0)


@pytest.mark.operation
@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an allocated CUDA GPU")
def test_allocated_cuda_bf16_training_and_full_sampling(tmp_path):
    _exercise_pipeline(tmp_path, device="cuda", bf16=True)
