"""Actual checkpoint smoke for matched exact-oracle control accounting."""

import csv
from pathlib import Path

import pytest

from emf.masked_edit_oracle_experiment import run_oracle_experiment
from emf.masked_edit_run import setup


CHECKPOINT = Path("4_experimental_result/result/masked_control_001/operation/C/checkpoint.pt")


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="operation checkpoint unavailable")
def test_actual_checkpoint_tiny_oracle_experiment(tmp_path):
    setup(142, "cpu")
    task = {"checkpoint": str(CHECKPOINT), "output": str(tmp_path / "oracle"),
            "split": "validation", "seed": 142, "campaign_limit": 1,
            "initial_noises": 1, "terminal_budget": 2, "nfe": 3, "basis_rank": 2,
            "max_velocity_rms": .05}
    summary = run_oracle_experiment(task, "cpu")
    assert summary["campaigns"] == 1
    assert summary["basis_source_molecules"] == 182
    assert summary["max_velocity_rms"] == .05
    assert len(summary["paired_cases"]) == 1
    assert summary["paired_cases"][0]["same_first_gaussian"]
    assert summary["paired_cases"][0]["matched_forward_nfe"]
    with (tmp_path / "oracle" / "attempts.csv").open(newline="") as stream:
        attempts = list(csv.DictReader(stream))
    assert len(attempts) == 4
    assert all(int(row["nfe"]) == 3 for row in attempts)
    assert all(row["support_match"] in ("True", "False") for row in attempts)
    assert all(row["support_excluded_success"] in ("True", "False") for row in attempts)
    with (tmp_path / "oracle" / "metrics.csv").open(newline="") as stream:
        metrics = list(csv.DictReader(stream))
    assert len(metrics) == 4  # case and campaign rows for both methods
