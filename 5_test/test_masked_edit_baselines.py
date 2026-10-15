"""Smoke the committed-candidate descriptor controls and separate truth report."""

import csv

from emf.masked_edit_baselines import evaluate_simple_baselines
from emf.masked_edit_data import prepare_masked_edit_data


def _read(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def test_simple_baselines_keep_invalid_attempts_and_truth_separate(tmp_path):
    data = tmp_path / "data"
    prepare_masked_edit_data(data)
    output = tmp_path / "baseline"
    report = evaluate_simple_baselines("validation", output, budget=8, k=4,
                                       data_dir=data, kind="vector")
    assert report["campaigns"] == 16
    attempts = _read(output / "raw_attempts.csv")
    summaries = _read(output / "topk_summary.csv")
    calibration = _read(output / "support_calibration.csv")
    truth = _read(output / "library_truth.csv")
    assert len(summaries) == 16 * 3 * 2
    assert len(calibration) == 16 * 3
    assert len(truth) == 16
    assert any(row["method"] == "direct_mutation" and row["chemically_valid"] == "False"
               for row in attempts)
    assert all(int(row["topk_filled"]) <= 4 and
               int(row["exact_candidate_calls"]) == int(row["topk_filled"])
               for row in summaries)
    assert all(row["classification"] == "unsynthesized_library_truth" for row in truth)
    assert all(row["selected_rank"] == "" or row["measured_delta_1"] != ""
               for row in attempts)
