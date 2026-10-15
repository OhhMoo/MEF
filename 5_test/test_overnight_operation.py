"""Exercise the overnight study's frozen decoder and cumulative time budget."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from emf import objectives
from emf.data import collate_graphs, load_split
from emf.models import EMFModel
from emf.objectives import flow_decoder_loss
from emf.train import run_training
from test_operation import _checkpoint, _config


@pytest.fixture(autouse=True)
def small_cpu_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def _representation(tmp_path, *, device="cpu", bf16=False):
    config = replace(_config(tmp_path, output_name="representation", num_samples=0,
                             device=device, bf16=bf16), warmup_steps=3, flow_steps=0)
    run_training(config)
    path = Path(config.output_dir) / "checkpoints/representation.pt"
    saved = _checkpoint(path)
    assert saved["step"] == 0 and not saved["optimizer"]["state"]
    return config, path


@pytest.mark.operation
def test_frozen_decoder_preserves_categorical_gradient_to_flow(tmp_path, monkeypatch):
    config, checkpoint = _representation(tmp_path)
    model = EMFModel(config.model)
    model.load_state_dict(_checkpoint(checkpoint)["model"])
    model.freeze_encoder()
    model.freeze_decoder()
    model.train()
    dataset = load_split(config.data_dir, "train")
    batch = collate_graphs([dataset[index] for index in range(6)])

    # Remove velocity gradients so a nonzero F gradient must pass through the
    # fixed decoder's intermediate categorical branch.
    def zero_velocity(predicted, *_):
        return predicted.nodes.sum() * 0 + predicted.pairs.sum() * 0
    monkeypatch.setattr(objectives, "velocity_matching_loss", zero_velocity)

    zero, _ = flow_decoder_loss(model, batch, lambda_mid=0)
    zero.backward()
    assert all(parameter.grad is None or not parameter.grad.any()
               for parameter in model.flow.parameters())
    model.zero_grad(set_to_none=True)
    positive, metrics = flow_decoder_loss(model, batch, lambda_mid=.1)
    positive.backward()
    gradients = [parameter.grad for parameter in model.flow.parameters()
                 if parameter.grad is not None]
    assert metrics["mid_loss"] > 0
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(gradient.abs().sum() > 0 for gradient in gradients)
    assert not model.encoder.training and not model.decoder.training
    assert all(parameter.grad is None for module in (model.encoder, model.decoder)
               for parameter in module.parameters())


def _frozen_budget_operation(tmp_path, *, device="cpu", bf16=False):
    base, start = _representation(tmp_path, device=device, bf16=bf16)
    config = replace(base, output_dir=str(tmp_path / "frozen_arm"), flow_steps=100,
                     freeze_decoder=True, lambda_mid=.1,
                     flow_time_budget_seconds=1e-6, num_samples=7)
    model, status = run_training(config, resume=start)
    initial = _checkpoint(start)
    latest_path = Path(config.output_dir) / "checkpoints/latest.pt"
    final = _checkpoint(latest_path)
    assert status["stop_reason"] == "time_budget"
    assert status["flow_steps"] == final["step"] == 1
    assert status["max_flow_steps"] == 100
    assert status["flow_training_seconds"] == final["flow_training_seconds"]
    assert status["flow_training_seconds"] >= config.flow_time_budget_seconds
    assert status["generation"]["attempts"] == 7
    assert status["metrics"]["mid_loss"] > 0
    assert model.flow.training and not model.decoder.training
    assert all(not parameter.requires_grad for parameter in model.decoder.parameters())
    assert all(parameter.grad is None for parameter in model.decoder.parameters())
    for name, value in final["model"].items():
        if name.startswith(("encoder.", "normalizer.", "decoder.")):
            torch.testing.assert_close(value, initial["model"][name], rtol=0, atol=0)
    assert any(not torch.equal(value, initial["model"][name])
               for name, value in final["model"].items() if name.startswith("flow."))
    assert len(final["optimizer"]["param_groups"][0]["params"]) == len(list(model.flow.parameters()))
    if device == "cuda":
        assert status["peak_gpu_memory_bytes"] > 0
        assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    return config, model, status, final


@pytest.mark.operation
def test_frozen_branch_stops_on_actual_training_time_and_preserves_representation(tmp_path):
    _frozen_budget_operation(tmp_path)


@pytest.mark.operation
def test_shared_start_restores_paired_rng_with_fresh_policy_optimizers(tmp_path):
    base, start = _representation(tmp_path)
    source = _checkpoint(start)
    checkpoints = []
    for frozen in (False, True):
        config = replace(base, output_dir=str(tmp_path / f"paired_{frozen}"),
                         flow_steps=2, lambda_mid=.1, freeze_decoder=frozen,
                         learning_rate=1e-3)
        _, status = run_training(config, resume=start)
        saved = _checkpoint(Path(config.output_dir) / "checkpoints/latest.pt")
        assert status["flow_steps"] == 2
        assert saved["optimizer"]["param_groups"][0]["lr"] == config.learning_rate
        for name, value in saved["model"].items():
            if name.startswith(("encoder.", "normalizer.")) or (frozen and name.startswith("decoder.")):
                torch.testing.assert_close(value, source["model"][name], rtol=0, atol=0)
        if not frozen:
            assert any(not torch.equal(value, source["model"][name])
                       for name, value in saved["model"].items() if name.startswith("decoder."))
        checkpoints.append(saved)
    assert torch.equal(checkpoints[0]["rng"]["torch"], checkpoints[1]["rng"]["torch"])
    assert torch.equal(checkpoints[0]["rng"]["batches"], checkpoints[1]["rng"]["batches"])


@pytest.mark.operation
def test_budget_resume_carries_time_and_permits_only_newly_budgeted_updates(tmp_path):
    config, before_model, before, checkpoint = _frozen_budget_operation(tmp_path)
    path = Path(config.output_dir) / "checkpoints/latest.pt"
    resumed_model, resumed = run_training(config, resume=path)
    assert resumed["flow_steps"] == before["flow_steps"]
    assert resumed["flow_training_seconds"] == before["flow_training_seconds"]
    assert resumed["metrics"] == before["metrics"]
    for name, value in before_model.state_dict().items():
        torch.testing.assert_close(resumed_model.state_dict()[name], value, rtol=0, atol=0)
    after = _checkpoint(path)
    assert torch.equal(after["rng"]["torch"], checkpoint["rng"]["torch"])
    assert torch.equal(after["rng"]["batches"], checkpoint["rng"]["batches"])

    # Grant just one more update's worth of positive time beyond the saved
    # counter. This catches treating every resume as a fresh full budget.
    extended = replace(config, flow_time_budget_seconds=before["flow_training_seconds"] + 1e-6,
                       num_samples=0)
    _, result = run_training(extended, resume=path)
    assert result["flow_steps"] == before["flow_steps"] + 1
    assert result["flow_training_seconds"] >= extended.flow_time_budget_seconds
    assert result["stop_reason"] == "time_budget"


@pytest.mark.operation
def test_nonzero_resume_cannot_switch_decoder_policy(tmp_path):
    config, _, _, _ = _frozen_budget_operation(tmp_path)
    changed = replace(config, freeze_decoder=False, output_dir=str(tmp_path / "policy_change"))
    with pytest.raises(ValueError, match="decoder policy"):
        run_training(changed, resume=Path(config.output_dir) / "checkpoints/latest.pt")


@pytest.mark.operation
@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an allocated CUDA GPU")
def test_allocated_cuda_frozen_decoder_mid_supervision_and_time_budget(tmp_path):
    _frozen_budget_operation(tmp_path, device="cuda", bf16=True)
