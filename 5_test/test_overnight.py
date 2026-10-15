"""Check paired task design, bounded orchestration, and honest cost reporting."""

import importlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from emf.train import RunConfig

CONTROL = Path(__file__).resolve().parents[1] / "4_experimental_result/control"
sys.path.insert(0, str(CONTROL))
overnight = importlib.import_module("overnight")
analysis = importlib.import_module("study_analysis")


@pytest.fixture
def prepared(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "manifest.json").write_text('{"fixture": true}\n')
    (data / "train.npz").write_bytes(b"synthetic-training-file")
    (data / "validation.npz").write_bytes(b"synthetic-validation-file")
    config = RunConfig(data_dir=str(data), device="cpu")
    path = tmp_path / "base.yaml"
    path.write_text(yaml.safe_dump(asdict(config)))
    representation = tmp_path / "representation.pt"
    torch.save({"phase": "flow", "step": 0, "config": asdict(config),
                "data_hash": "training-identities"}, representation)
    study, tasks = overnight.prepare(
        study_dir=tmp_path / "study", config_dir=tmp_path / "configs",
        baseline_config=path, baseline_representation=representation,
        flow_budget_seconds=0.2, max_flow_steps=37, generation_attempts=2)
    for seed in study["seeds"]:
        shared = Path(study["shared_representations"][str(seed)])
        shared.parent.mkdir(parents=True)
        saved = {**asdict(config), "seed": seed}
        torch.save({"phase": "flow", "step": 0, "config": saved,
                    "data_hash": "training-identities"}, shared)
        provenance = {"state": "complete", "checkpoint": overnight._representation_context(shared, seed)}
        (shared.parents[1] / "representation_provenance.json").write_text(json.dumps(provenance))
    return study, tasks


def test_grid_pairs_every_arm_with_one_representation_per_seed(prepared):
    study, tasks = prepared
    assert len(tasks) == 18
    assert [task["index"] for task in tasks] == list(range(18))
    assert {(task["seed"], task["lambda_mid"], task["decoder_mode"]) for task in tasks} == {
        (seed, coefficient, mode) for seed in (42, 43, 44)
        for coefficient in (0.0, 0.1, 1.0) for mode in ("trainable", "frozen_common")}
    for seed in (42, 43, 44):
        assert len({task["representation"] for task in tasks if task["seed"] == seed}) == 1
        assert Path(study["shared_representations"][str(seed)]).is_relative_to(Path(study["study_dir"]))
        warmup = yaml.safe_load((Path(study["config_dir"]) / f"representation_{seed}.yaml").read_text())
        assert warmup["seed"] == seed and warmup["flow_steps"] == warmup["num_samples"] == 0
    for task in tasks:
        config = yaml.safe_load(Path(task["config"]).read_text())
        assert config["freeze_decoder"] == (task["decoder_mode"] == "frozen_common")
        assert config["flow_time_budget_seconds"] == 0.2
        assert config["flow_steps"] == 37
        assert config["sample_every"] == config["checkpoint_every"] == 10000
    assert study["generation_seed"] == 314159
    assert study["nfe_grid"] == [8, 16, 32, 64, 128]
    for relative, digest in study["source_sha256"].items():
        assert overnight.sha256(Path(study["source_snapshot"]) / relative) == digest


def test_task_uses_actual_budget_stop_checkpoint_and_fatal_errors(prepared, monkeypatch):
    study, tasks = prepared
    calls = []

    def fake_training(config, resume):
        calls.append((config, resume))
        output = Path(config.output_dir) / "checkpoints"
        output.mkdir()
        (output / "flow_0000007.pt").write_bytes(b"final-immutable-checkpoint")
        return object(), {"flow_steps": 7, "flow_training_seconds": 0.201, "stop_reason": "time_budget"}

    grids = []
    monkeypatch.setattr(overnight, "run_training", fake_training)
    monkeypatch.setattr(overnight, "sample_grid", lambda *args, **kwargs: grids.append((args, kwargs)))
    result = overnight.run_task(0, study_dir=study["study_dir"])
    assert result["state"] == "complete"
    assert calls[0][1] == tasks[0]["representation"]
    assert calls[0][0].flow_steps == 37
    assert grids[0][0][0].name == "flow_0000007.pt"
    assert grids[0][1]["seed"] == 314159 and grids[0][1]["num_samples"] == 2
    with pytest.raises(FileExistsError, match="already attempted"):
        overnight.run_task(0, study_dir=study["study_dir"])

    monkeypatch.setattr(overnight, "run_training", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("fixture failure")))
    with pytest.raises(RuntimeError, match="fixture failure"):
        overnight.run_task(1, study_dir=study["study_dir"])
    failed = json.loads((Path(tasks[1]["output_dir"]) / "task_provenance.json").read_text())
    assert failed["state"] == "failed" and "fixture failure" in failed["traceback"]


def test_underbudget_step_ceiling_is_a_failed_arm(prepared, monkeypatch):
    study, tasks = prepared
    monkeypatch.setattr(overnight, "run_training", lambda *_args, **_kwargs: (
        object(), {"flow_steps": 37, "flow_training_seconds": 0.1, "stop_reason": "step_limit"}))
    with pytest.raises(RuntimeError, match="update ceiling"):
        overnight.run_task(0, study_dir=study["study_dir"])
    failed = json.loads((Path(tasks[0]["output_dir"]) / "task_provenance.json").read_text())
    assert failed["state"] == "failed"
    assert failed["training_result"]["flow_training_seconds"] == 0.1


def test_configuration_mutation_is_rejected_before_training(prepared, monkeypatch):
    study, tasks = prepared
    Path(tasks[0]["config"]).write_text(Path(tasks[0]["config"]).read_text() + "# mutation\n")
    monkeypatch.setattr(overnight, "run_training", lambda *_a, **_k: pytest.fail("Should not train mutated config"))
    with pytest.raises(ValueError, match="configuration changed"):
        overnight.run_task(0, study_dir=study["study_dir"])


def test_sampling_grid_reuses_noise_seed_and_checkpoint_hash(tmp_path, monkeypatch):
    checkpoint = tmp_path / "fixture.pt"
    checkpoint.write_bytes(b"trusted-checkpoint")
    calls = []

    def fake_generate(path, destination, **kwargs):
        calls.append(kwargs)
        return [], {"attempts": kwargs["num_samples"], "nfe": kwargs["nfe"],
                    "metadata": {"checkpoint_sha256": overnight.sha256(path)}}

    monkeypatch.setattr(overnight, "generate_from_checkpoint", fake_generate)
    monkeypatch.setitem(sys.modules, "study_metrics", SimpleNamespace(
        analyze_samples=lambda *_args: {"heldout_atom_label_js_nats": None}))
    output = tmp_path / "grid"
    summaries = overnight.sample_grid(checkpoint, output, reference_data_dir=tmp_path,
                                     num_samples=3, nfe_grid=(8, 16, 32), device="cpu")
    assert len(summaries) == len(calls) == 3
    assert {row["seed"] for row in calls} == {314159}
    assert {row["batch_size"] for row in calls} == {128}
    assert [row["nfe"] for row in calls] == [8, 16, 32]
    assert json.loads((output / "grid_manifest.json").read_text())["state"] == "complete"
    with pytest.raises(FileExistsError, match="already attempted"):
        overnight.sample_grid(checkpoint, output, reference_data_dir=tmp_path, device="cpu")


def test_paired_analysis_keeps_seed_deltas_and_null_distances():
    records = [
        {"seed": seed, "nfe": 32, "lambda_mid": coefficient, "decoder_mode": mode,
         "attempts": 10000, "generation_seed": 314159,
         "connected_validity_rate": 0.2 + coefficient * 0.1 + (0.05 if mode == "trainable" else 0),
         "heldout_atom_label_js_nats": None}
        for seed in (42, 43, 44) for coefficient in (0.0, 0.1, 1.0)
        for mode in ("trainable", "frozen_common")]
    deltas, summaries = analysis.paired_effects(records)
    assert len(deltas) == 21
    assert len(summaries) == 7
    assert {row["seed_pairs"] for row in summaries} == {3}
    assert {row["metric"] for row in summaries} == {"connected_validity_rate"}
    contrast = next(row for row in summaries if row["comparison"] == "trainable_vs_frozen_common_lambda_0")
    assert contrast["mean_delta"] == pytest.approx(0.05)
    assert contrast["sample_sd"] == pytest.approx(0)
    records[-1]["generation_seed"] = 7
    with pytest.raises(ValueError, match="same attempts and generation seed"):
        analysis.paired_effects(records)


def test_paired_failure_counts_include_zero_without_filling_missing_quality():
    common = {"seed": 42, "nfe": 32, "decoder_mode": "trainable",
              "attempts": 10000, "generation_seed": 314159}
    records = [
        {**common, "lambda_mid": 0.0, "failure_invalid_valence_count": 3,
         "heldout_atom_label_js_nats": 0.4},
        {**common, "lambda_mid": 0.1, "failure_disconnected_count": 2},
    ]
    deltas, summaries = analysis.paired_effects(records)
    values = {row["metric"]: row["delta"] for row in deltas}
    assert values == {"failure_invalid_valence_count": -3, "failure_disconnected_count": 2}
    assert len(summaries) == 2


def test_scheduler_cost_excludes_steps_and_counts_actual_gpu_allocation(monkeypatch):
    output = ("100|training|COMPLETED|3600|cpu=8,gres/gpu=1,gres/gpu:l40s=1|0:0\n"
              "100.batch|batch|COMPLETED|3600|cpu=8,gres/gpu=1|0:0\n"
              "101|training|FAILED|1800|cpu=8,gres/gpu:l40s=2|1:0\n")
    monkeypatch.setattr(analysis.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
        returncode=0, stdout=output, stderr=""))
    costs, accounting = analysis.scheduler_costs({"100", "101"})
    assert len(costs) == 2
    assert accounting["allocated_gpu_hours"] == pytest.approx(2)
    assert costs[0]["allocated_gpu_count"] == 1
    assert costs[1]["allocated_gpu_count"] == 2


def test_common_update_comparison_reads_actual_phase_update_cost(prepared):
    study, tasks = prepared
    for task in tasks[:6]:
        output = Path(task["output_dir"])
        sample = output / "samples_0010000"
        sample.mkdir()
        (sample / "summary.json").write_text(json.dumps({
            "attempts": 1000, "seed": 10042, "nfe": 32,
            "connected_validity_rate": 0.2,
            "metadata": {"flow_step": 10000}}))
        (output / "training.csv").write_text("phase,step,training_seconds\nflow,10000,600.5\n")
    records = analysis._same_update_records(tasks)
    assert len(records) == 6
    assert {row["seed"] for row in records} == {42}
    assert {row["generation_seed"] for row in records} == {10042}
    assert {float(row["flow_training_seconds"]) for row in records} == {600.5}
    assert {row["flow_steps"] for row in records} == {10000}


def test_aggregate_joins_quality_with_training_seed_and_records_partial_completion(prepared, monkeypatch):
    study, tasks = prepared
    task = tasks[0]
    output = Path(task["output_dir"])
    (output / "task_provenance.json").write_text(json.dumps({
        "state": "complete", "training_result": {"flow_steps": 7, "flow_training_seconds": 0.201},
        "checkpoint_sha256": "saved-model", "scheduler": {"SLURM_JOB_ID": "100"}}))
    for nfe in study["nfe_grid"]:
        sample = output / "final_grid" / f"nfe_{nfe:03d}"
        sample.mkdir(parents=True)
        (sample / "summary.json").write_text(json.dumps({
            "attempts": 2, "seed": 314159, "nfe": nfe,
            "connected_validity_rate": 0.5, "metadata": {"checkpoint_sha256": "saved-model"}}))
        (sample / "quality_metrics.json").write_text(json.dumps({"seed": 314159, "heldout_atom_label_js_nats": None}))
    monkeypatch.setattr(analysis, "scheduler_costs", lambda _ids: ([], {"available": True, "allocated_gpu_hours": 1}))
    report = analysis.aggregate(study_dir=study["study_dir"])
    assert report["complete_tasks"] == 1 and report["planned_tasks"] == 18
    assert report["final_nfe_records"] == 5
    import csv
    with (Path(study["study_dir"]) / "analysis/quality_and_yield_vs_nfe.csv").open(newline="") as stream:
        records = list(csv.DictReader(stream))
    assert {row["seed"] for row in records} == {"42"}
    assert {row["generation_seed"] for row in records} == {"314159"}
