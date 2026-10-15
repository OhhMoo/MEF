"""A source completion copies exact optimizer/RNG state into a new output."""

from dataclasses import replace
import json
from pathlib import Path

import torch
import yaml

from emf.local_edit_run import _prepare_resume_checkpoint
from emf.redesign_train import digest, load_redesign_config, tensor_fingerprint


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = (ROOT / "4_experimental_result/result/local_edit_001/operation_repeat"
           / "anchored_known_seed_142")


def _setup(tmp_path):
    old = tmp_path / "partial"
    new = tmp_path / "completion"
    (old / "checkpoints").mkdir(parents=True)
    payload = torch.load(EXAMPLE / "checkpoints/source_final.pt",
                         map_location="cpu", weights_only=False)
    original_config = dict(payload["config"])
    original_config["output_dir"] = str(old)
    payload["config"] = original_config
    payload["step"] = 40  # make this small completed operation a partial source
    payload["training_seconds"] = 60.
    checkpoint = old / "checkpoints/source_final.pt"
    torch.save(payload, checkpoint)
    manifest = json.loads((EXAMPLE / "manifest.json").read_text())
    manifest["config"] = original_config
    (old / "manifest.json").write_text(json.dumps(manifest))
    (old / "config.yaml").write_text(yaml.safe_dump(original_config))
    old_task = json.loads((EXAMPLE / "local_edit_task.json").read_text())
    (old / "local_edit_task.json").write_text(json.dumps(old_task))
    new_config = dict(original_config)
    new_config["output_dir"] = str(new)
    new_config["train_seconds"] = 180.
    config_path = tmp_path / "completion.yaml"
    config_path.write_text(yaml.safe_dump(new_config))
    task = {**old_task, "config": str(config_path), "resume_checkpoint": str(checkpoint),
            "resume_checkpoint_sha256": digest(checkpoint)}
    return task, load_redesign_config(config_path), payload


def _assert_exact_tree(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_exact_tree(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for item, reference in zip(actual, expected):
            _assert_exact_tree(item, reference)
    else:
        assert actual == expected


def test_resume_copy_preserves_training_state_and_rejects_hyperparameter_change(tmp_path):
    task, config, original = _setup(tmp_path)
    path, amendment = _prepare_resume_checkpoint(task, config)
    copied = torch.load(path, map_location="cpu", weights_only=False)
    assert amendment["changed_fields"] == ["output_dir", "train_seconds"]
    assert copied["step"] == original["step"] == 40
    assert copied["training_seconds"] == original["training_seconds"] == 60.
    assert tensor_fingerprint(copied["model"].items()) == tensor_fingerprint(original["model"].items())
    _assert_exact_tree(copied["rng"], original["rng"])
    _assert_exact_tree(copied["optimizer"], original["optimizer"])
    assert (Path(config.output_dir) / "resume_provenance.json").exists()
    assert digest(task["resume_checkpoint"]) == task["resume_checkpoint_sha256"]
    changed = replace(config, learning_rate=config.learning_rate * 2,
                      output_dir=str(tmp_path / "illegal"))
    try:
        _prepare_resume_checkpoint(task, changed)
    except ValueError as error:
        assert "only output_dir and train_seconds" in str(error)
    else:
        raise AssertionError("changed learning rate was accepted")
