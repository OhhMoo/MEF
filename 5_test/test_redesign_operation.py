"""Independent small operation of measured episodes, training and deployment.

The molecular fixture is prepared by the data tests. This review exercises its
public observation cache through the actual model, rather than assigning oracle
features or hidden campaign coefficients to the network.
"""

from contextlib import nullcontext
from dataclasses import asdict, replace
import hashlib
import json

import numpy as np
import pytest
import torch

from emf.campaigns import CampaignDataset, collate_episodes, neighbourhood
from emf.data import collate_graphs
from emf.models import EMFModel
from emf.objectives import reconstruction_loss
from emf.contracts import ModelConfig
from emf.redesign import RedesignModel
from emf.redesign_contracts import RedesignConfig, SupportBatch
from emf.redesign_objectives import redesign_loss
from emf.redesign_sampling import make_sampling_plan
from emf.redesign_evaluation import (
    evaluate_committed_proposals, generate_redesign_proposals,
    load_committed_proposals,
)
from emf.toy_oracle import ToyOracle

# Reuse the real-graph preparation fixture; the reviewer owns this test only.
from test_redesign_campaigns import campaign_dir  # noqa: F401


torch.set_num_threads(2)


def _snapshot(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _unchanged(module, before):
    for key, value in module.state_dict().items():
        torch.testing.assert_close(value.detach().cpu(), before[key], rtol=0, atol=0)


def _has_changed(module, before):
    return any(not torch.equal(value.detach().cpu(), before[key])
               for key, value in module.state_dict().items())


def _features(model, dataset, device):
    blocks = []
    for offset in range(0, len(dataset), 64):
        graphs = collate_graphs([dataset.graph(index)
                                for index in range(offset, min(offset + 64, len(dataset)))])
        blocks.append(model.graph_features(graphs.to(device)).cpu())
    return torch.cat(blocks)


def _amp(device):
    return (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if torch.device(device).type == "cuda" else nullcontext())


def _scientific_rows(rows):
    return [{key: value for key, value in row.items()
             if key != "batch_amortized_seconds"} for row in rows]


def _exercise_real_pipeline(tmp_path, campaign_dir, device):
    torch.manual_seed(271)
    source = CampaignDataset(campaign_dir, "train")
    graph = ModelConfig(max_nodes=9, node_dim=8, pair_dim=4,
                        hidden_dim=16, depth=2, num_heads=2)
    foundation = EMFModel(graph).to(device)
    warm_graphs = collate_graphs([source.graph(index) for index in range(24)]).to(device)
    warm_optimizer = torch.optim.AdamW(foundation.parameters(), lr=2e-3)
    for _ in range(8):
        warm_optimizer.zero_grad(set_to_none=True)
        with _amp(device):
            warm_loss, _ = reconstruction_loss(foundation, warm_graphs)
        assert warm_loss.dtype == torch.float32 and torch.isfinite(warm_loss)
        warm_loss.backward()
        warm_optimizer.step()
    with torch.no_grad():
        foundation.normalizer.fit([(foundation.encoder(warm_graphs).detach(), warm_graphs.masks)])
    foundation.freeze_encoder()
    foundation.freeze_decoder()
    original_decoder = _snapshot(foundation.decoder)

    config = RedesignConfig(model=graph, rank=2, state_dim=6, context_dim=4)
    model = RedesignModel(config, foundation).to(device).train()
    assert all(parameter.requires_grad for parameter in model.decoder.parameters())
    assert all(not parameter.requires_grad for parameter in model.encoder.parameters())
    encoder, normalizer = _snapshot(model.encoder), _snapshot(model.normalizer)
    initial = {name: _snapshot(getattr(model, name))
               for name in ("flow", "evidence", "response", "decoder")}
    frozen_features = model.graph_features(warm_graphs)
    with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16):
        mixed_features = model.graph_features(warm_graphs)
    torch.testing.assert_close(mixed_features, frozen_features, rtol=0, atol=0)
    features = _features(model, source, device)
    observation_rng = np.random.default_rng(991)
    interpolation_rng = torch.Generator(device="cpu").manual_seed(992)
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters()
                                  if parameter.requires_grad), lr=1e-3)
    accepted = 0
    selector_received_gradient = False
    metrics_log = []
    for _ in range(8):
        records = source.sample_batch(observation_rng, episodes=2, queries_per_lead=2)
        assert all(not set(record.support_ids) & set(record.query_ids) for record in records)
        batch = collate_episodes(records, pooled_features=features, device=device)
        optimizer.zero_grad(set_to_none=True)
        with _amp(device):
            loss, metrics = redesign_loss(model, batch, generator=interpolation_rng)
        assert loss.dtype == torch.float32 and torch.isfinite(loss)
        loss.backward()
        assert all(parameter.grad is None or torch.isfinite(parameter.grad).all()
                   for parameter in model.parameters())
        gradient = model.flow.selector[-1].weight.grad
        selector_received_gradient |= gradient is not None and bool(gradient.abs().sum() > 0)
        accepted += int(metrics["accepted_queries"])
        optimizer.step()
        metrics_log.append({name: float(value) for name, value in metrics.items()})
    assert accepted > 0 and selector_received_gradient
    for name in initial:
        assert _has_changed(getattr(model, name), initial[name]), name
    _unchanged(model.encoder, encoder)
    _unchanged(model.normalizer, normalizer)
    _unchanged(foundation.decoder, original_decoder)
    assert all(parameter.grad is None for parameter in model.encoder.parameters())

    # Reject the same queries through goals only: the proper all-query response
    # likelihood must stay identical, while the flow objective becomes zero.
    rejected_goal = batch.goal.clone()
    rejected_goal[:, 0] = 1e6
    rejected = replace(batch, goal=rejected_goal,
                       flow_weights=torch.zeros_like(batch.flow_weights))
    comparison_seed = 187
    with _amp(device):
        _, ordinary_metrics = redesign_loss(
            model, batch, generator=torch.Generator().manual_seed(comparison_seed))
        rejected_loss, rejected_metrics = redesign_loss(
            model, rejected, generator=torch.Generator().manual_seed(comparison_seed))
    torch.testing.assert_close(rejected_metrics["response_loss"], ordinary_metrics["response_loss"],
                               rtol=0, atol=0)
    assert rejected_metrics["flow_loss"] == 0
    model.zero_grad(set_to_none=True)
    rejected_loss.backward()
    assert all(parameter.grad is None or not parameter.grad.any() for parameter in model.flow.parameters())
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for parameter in model.response.parameters())

    checkpoint = tmp_path / "review_source.pt"
    torch.save({"config": asdict(config), "model": model.state_dict()}, checkpoint)
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    restored = RedesignModel(config).to(device)
    restored.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["model"])
    restored.freeze_deployment()
    restored.train()  # The source-training method cannot thaw target deployment.
    assert all(not parameter.requires_grad for parameter in restored.parameters())
    assert all(not module.training for module in restored.modules())
    frozen = _snapshot(restored)

    target = CampaignDataset(campaign_dir, "test")
    lead_index = int(target.eligible_leads[0])
    nested = target.nested_support(0, lead_index, seed=511, sizes=(4, 8))
    lead_graph = collate_graphs([nested["lead"]]).to(device)
    baseline = torch.from_numpy(nested["lead_values"])[None].to(device)
    goal = torch.stack([baseline[:, 0] * 0 + .25, baseline[:, 1] - .1], -1)
    native = nested["records"][8]
    support_graphs = collate_graphs(native["graphs"]).to(device)
    support = SupportBatch(restored.graph_features(support_graphs)[None],
                           torch.from_numpy(native["values"])[None].to(device),
                           torch.from_numpy(native["context"])[None].to(device),
                           torch.ones(1, 8, dtype=torch.bool, device=device))
    state = restored.support_state(support)
    condition = restored.condition(lead_graph, baseline, support.context, state, goal)
    distribution = restored.predict_response(lead_graph, lead_graph, baseline,
                                              support.context, state)
    assert distribution.mean.dtype == distribution.covariance_matrix.dtype == torch.float32
    eigenvalues = torch.linalg.eigvalsh(distribution.covariance_matrix)
    assert torch.all(eigenvalues >= config.covariance_floor * .999)

    plan = make_sampling_plan(graph, target.count_prior, nested["lead"]["count"],
                              num_attempts=11, count_seed=512, noise_seed=513)
    torch_state = torch.get_rng_state().clone()
    output = tmp_path / "committed"
    rows, ranking, proposal_summary = generate_redesign_proposals(
        restored, condition, plan, nested["lead"]["smiles"], nested["reserve_smiles"], output,
        neighbourhood_fn=lambda candidate, lead: neighbourhood(candidate, lead, min_similarity=.1),
        nfe=4, batch_size=4, k=3, metadata={"source_checkpoint_sha256": checkpoint_hash})
    assert len(rows) == proposal_summary["attempts"] == 11
    assert [row["attempt_id"] for row in rows] == list(range(11))
    assert all(row["nfe"] == 4 for row in rows)
    assert all(not row["eligible"] for row in rows if row["reserved_support_match"])
    assert all(not row["eligible"] for row in rows if row["duplicate"])
    assert len({row["smiles"] for row in ranking}) == len(ranking) <= 3
    assert proposal_summary["generation_seconds"] >= proposal_summary["response_scoring_seconds"] >= 0
    assert not (output / "evaluation_outcomes.jsonl").exists()
    saved_rows, saved_ranking, commit = load_committed_proposals(output)
    assert saved_rows == rows and saved_ranking == ranking
    assert commit["oracle_access_before_commit"] is False
    sealed = {name: (output / name).read_bytes()
              for name in ("attempts.jsonl", "ranking.json", "proposal_commit.json")}
    evaluator_calls = []
    oracle = ToyOracle.load(campaign_dir / "private" / "oracle.json")

    def reveal(smiles):
        # Every external result is obtained after a verified proposal commit.
        assert all((output / name).read_bytes() == content for name, content in sealed.items())
        evaluator_calls.append(smiles)
        return oracle.score_smiles(target.campaign_ids[0], smiles)

    evaluation = evaluate_committed_proposals(output, reveal, baseline=baseline[0],
                                              goal=goal[0], support_size=8, k=3)
    assert len(evaluator_calls) == len(set(evaluator_calls)) == evaluation["external_outcome_reader_calls"]
    assert not set(evaluator_calls) & set(nested["reserve_smiles"])
    assert evaluation["allocated_compound_budget"] == 11
    assert evaluation["unfilled_selection_slots"] == 3 - len(ranking)
    assert evaluation["success"] == evaluation["selected_joint_profile_successes"] / 3
    assert evaluation["efficiency"] == evaluation["selected_joint_profile_successes"] / 11
    assert all((output / name).read_bytes() == content for name, content in sealed.items())
    _unchanged(restored, frozen)
    torch.testing.assert_close(torch.get_rng_state(), torch_state, rtol=0, atol=0)

    replay_rows, replay_ranking, _ = generate_redesign_proposals(
        restored, condition, plan, nested["lead"]["smiles"], nested["reserve_smiles"],
        tmp_path / "replayed", neighbourhood_fn=lambda candidate, lead: neighbourhood(candidate, lead, min_similarity=.1),
        nfe=4, batch_size=4, k=3)
    assert _scientific_rows(replay_rows) == _scientific_rows(rows)
    assert replay_ranking == ranking
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == checkpoint_hash
    (tmp_path / "operation_review.json").write_text(json.dumps({
        "device": str(torch.device(device)), "source_updates": 8,
        "accepted_queries": accepted, "checkpoint_sha256": checkpoint_hash,
        "training": metrics_log, "proposal_summary": proposal_summary,
        "evaluation": evaluation,
    }, indent=2) + "\n")


@pytest.mark.operation
def test_redesign_real_cpu_operation(tmp_path, campaign_dir):
    _exercise_real_pipeline(tmp_path, campaign_dir, "cpu")


@pytest.mark.gpu
@pytest.mark.operation
def test_redesign_real_gpu_operation(tmp_path, campaign_dir):
    if not torch.cuda.is_available():
        pytest.skip("requires an allocated CUDA GPU")
    torch.set_float32_matmul_precision("high")
    _exercise_real_pipeline(tmp_path, campaign_dir, "cuda:0")


def _small_runner_config(tmp_path, campaign_dir):
    from emf.redesign_train import RedesignRunConfig

    torch.manual_seed(372)
    graph = ModelConfig(max_nodes=9, node_dim=8, pair_dim=4,
                        hidden_dim=16, depth=2, num_heads=2)
    foundation = EMFModel(graph)
    data = CampaignDataset(campaign_dir, "train")
    examples = collate_graphs([data.graph(index) for index in range(24)])
    with torch.no_grad():
        foundation.normalizer.fit([(foundation.encoder(examples), examples.masks)])
    initializer = tmp_path / "foundation.pt"
    torch.save({"config": {"model": asdict(graph)}, "model": foundation.state_dict()}, initializer)
    return RedesignRunConfig(
        redesign=RedesignConfig(model=graph, rank=2, state_dim=6, context_dim=4),
        data_dir=str(campaign_dir), initializer=str(initializer),
        output_dir=str(tmp_path / "runner"), seed=373, episodes=2, queries_per_lead=2,
        train_steps=4, train_seconds=None, cpu_threads=2, device="cpu", bf16=False,
        cache_batch_size=64, log_every=1, checkpoint_every=2,
        eval_campaigns=1, leads_per_campaign=1, support_sizes=[4, 8],
        num_attempts=7, nfe=4, k=3, sample_batch_size=4,
    )


@pytest.mark.operation
def test_completed_source_resume_preserves_metrics_and_checkpoint(tmp_path, campaign_dir):
    from emf.redesign_train import load_source_checkpoint, run_training

    config = _small_runner_config(tmp_path, campaign_dir)
    first = run_training(config)
    root = tmp_path / "runner"
    checkpoint = root / "checkpoints" / "source_final.pt"
    saved_bytes = checkpoint.read_bytes()
    model, restored_config, payload = load_source_checkpoint(checkpoint, torch.device("cpu"))
    assert restored_config == config and payload["step"] == first["step"] == 4
    assert all(parameter.requires_grad for parameter in model.decoder.parameters())
    source_values = _snapshot(model)
    second = run_training(config, resume=checkpoint)
    assert second["step"] == first["step"]
    assert second["metrics"] == first["metrics"]
    assert checkpoint.read_bytes() == saved_bytes
    model_after, _, _ = load_source_checkpoint(checkpoint, torch.device("cpu"))
    _unchanged(model_after, source_values)


@pytest.mark.operation
def test_interrupted_source_resume_replays_training_exactly(tmp_path, campaign_dir, monkeypatch):
    import emf.redesign_train as runner

    config = _small_runner_config(tmp_path, campaign_dir)
    reference = replace(config, output_dir=str(tmp_path / "reference"))
    interrupted = replace(config, output_dir=str(tmp_path / "interrupted"))
    runner.run_training(reference)
    real_save = runner._save_checkpoint

    def save_then_interrupt(*args, **kwargs):
        real_save(*args, **kwargs)
        if str(args[0]).endswith("latest.pt") and args[4] == 2:
            raise RuntimeError("review interruption after source checkpoint")

    monkeypatch.setattr(runner, "_save_checkpoint", save_then_interrupt)
    with pytest.raises(RuntimeError, match="review interruption"):
        runner.run_training(interrupted)
    monkeypatch.setattr(runner, "_save_checkpoint", real_save)
    runner.run_training(interrupted, resume=tmp_path / "interrupted" / "checkpoints" / "latest.pt")
    expected = torch.load(tmp_path / "reference" / "checkpoints" / "source_final.pt",
                          map_location="cpu", weights_only=False)
    actual = torch.load(tmp_path / "interrupted" / "checkpoints" / "source_final.pt",
                        map_location="cpu", weights_only=False)
    assert actual["step"] == expected["step"] == 4
    assert actual["metrics"] == expected["metrics"]
    assert actual["representation_sha256"] == expected["representation_sha256"]
    for key in expected["model"]:
        torch.testing.assert_close(actual["model"][key], expected["model"][key], rtol=0, atol=0)
    torch.testing.assert_close(actual["rng"]["noise"], expected["rng"]["noise"], rtol=0, atol=0)
    assert actual["rng"]["episodes"] == expected["rng"]["episodes"]
    for key, state in expected["optimizer"]["state"].items():
        for name, value in state.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(actual["optimizer"]["state"][key][name], value, rtol=0, atol=0)
            else:
                assert actual["optimizer"]["state"][key][name] == value


@pytest.mark.operation
def test_partial_evaluation_resume_keeps_completed_batches_and_protocol(tmp_path, campaign_dir, monkeypatch):
    import csv
    import emf.redesign_train as runner

    config = _small_runner_config(tmp_path, campaign_dir)
    runner.run_training(config)
    checkpoint = tmp_path / "runner" / "checkpoints" / "source_final.pt"
    real_evaluation = runner.evaluate_committed_proposals
    calls = 0

    def interrupted(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("review interruption after first completed evaluation")
        return real_evaluation(*args, **kwargs)

    monkeypatch.setattr(runner, "evaluate_committed_proposals", interrupted)
    with pytest.raises(RuntimeError, match="review interruption"):
        runner.run_evaluation(config, checkpoint)
    eval_root = tmp_path / "runner" / "evaluation_validation"
    first_summary = next(eval_root.glob("case_*/support_*/*/evaluation_summary.json"))
    first_bytes = first_summary.read_bytes()
    monkeypatch.setattr(runner, "evaluate_committed_proposals", real_evaluation)
    report = runner.run_evaluation(config, checkpoint)
    assert report["batches"] == 2
    assert {row["support_size"] for row in report["curves"]} == {4, 8}
    with (eval_root / "metrics.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2 and {int(row["support_size"]) for row in rows} == {4, 8}
    assert first_summary.read_bytes() == first_bytes
    changed = replace(config, num_attempts=config.num_attempts + 1)
    with pytest.raises((FileExistsError, ValueError), match="protocol|different|differ|configuration"):
        runner.run_evaluation(changed, checkpoint)


@pytest.mark.operation
@pytest.mark.parametrize("variant,mode", [
    ("factorized", "native"),
    ("conditional", "native"),
    ("factorized", "selector_adaptation"),
])
def test_evaluation_reuses_one_lead_encoding_across_shared_models(
        tmp_path, campaign_dir, monkeypatch, variant, mode):
    """A second encoding perturbs the lead, making the historical failure visible.

    Run real checkpoint loading, native/adapted evidence inference, the strict
    sampler and blind evaluation. Different source seeds keep the generator's
    evidence distinct from the common scorer's evidence.
    """
    import emf.redesign_train as runner

    config = _small_runner_config(tmp_path, campaign_dir)
    scorer_config = replace(config, output_dir=str(tmp_path / "scorer"))
    generator_config = replace(
        config, output_dir=str(tmp_path / "generator"), seed=997,
        redesign=replace(config.redesign, variant=variant))
    runner.run_training(scorer_config)
    runner.run_training(generator_config)
    scorer_checkpoint = tmp_path / "scorer" / "checkpoints" / "source_final.pt"
    generator_checkpoint = tmp_path / "generator" / "checkpoints" / "source_final.pt"
    eval_config = replace(
        generator_config, output_dir=str(tmp_path / "evaluation"),
        common_scorer_checkpoint=str(scorer_checkpoint),
        evaluation_modes=[mode], adaptation_steps=2)

    dataset = CampaignDataset(campaign_dir, "validation")
    lead_index = int(np.random.default_rng(eval_config.eval_seed).choice(
        dataset.eligible_leads, size=1, replace=False)[0])
    lead_id = dataset.ids[lead_index]
    real_features = RedesignModel.graph_features
    real_state = RedesignModel.support_state
    real_generate = runner.generate_redesign_proposals
    lead_calls, generated_conditions = [], []
    inferred_states = {}

    def features_with_visible_repeat(model, graphs):
        result = real_features(model, graphs)
        if graphs.ids == (lead_id,):
            lead_calls.append(model)
            if len(lead_calls) > 1:
                result = result + .001 * (len(lead_calls) - 1)
        return result

    def record_state(model, support):
        state = real_state(model, support)
        inferred_states[id(model)] = state.detach().clone()
        return state

    def generate_with_checked_conditions(scorer, condition, *args, **kwargs):
        proposal_model = kwargs["proposal_model"]
        proposal_condition = kwargs["proposal_condition"]
        # Let the production guards reject a duplicated or incompatible lead
        # before adding assertions about evidence routing and storage reuse.
        result = real_generate(scorer, condition, *args, **kwargs)
        assert proposal_model is not scorer
        for name in ("lead_features", "baseline", "context", "goal"):
            assert getattr(proposal_condition, name) is getattr(condition, name)
        torch.testing.assert_close(condition.response_state, inferred_states[id(scorer)],
                                   rtol=0, atol=0)
        torch.testing.assert_close(proposal_condition.response_state,
                                   inferred_states[id(proposal_model)], rtol=0, atol=0)
        if mode == "native":
            assert not torch.equal(proposal_condition.response_state, condition.response_state)
        generated_conditions.append(condition)
        return result

    monkeypatch.setattr(RedesignModel, "graph_features", features_with_visible_repeat)
    monkeypatch.setattr(RedesignModel, "support_state", record_state)
    monkeypatch.setattr(runner, "generate_redesign_proposals", generate_with_checked_conditions)
    checkpoint_bytes = {path: path.read_bytes()
                        for path in (scorer_checkpoint, generator_checkpoint)}
    report = runner.run_evaluation(eval_config, generator_checkpoint)
    assert len(lead_calls) == 1
    assert len(generated_conditions) == report["batches"] == 2
    assert generated_conditions[0].lead_features is generated_conditions[1].lead_features
    assert {row["support_size"] for row in report["curves"]} == {4, 8}
    assert all(path.read_bytes() == content for path, content in checkpoint_bytes.items())


@pytest.mark.operation
@pytest.mark.parametrize("mismatch", ["lead_features", "encoder", "normalizer"])
def test_shared_lead_encoding_preserves_strict_sampling_guards(tmp_path, mismatch):
    from emf.chem import smiles_to_graph
    from emf.redesign_contracts import FlowCondition

    graph = ModelConfig(max_nodes=2, node_dim=2, pair_dim=1,
                        hidden_dim=8, depth=1, num_heads=2)
    foundation = EMFModel(graph)
    foundation.normalizer.fitted.fill_(True)
    architecture = RedesignConfig(model=graph, rank=2, state_dim=6)
    scorer = RedesignModel(architecture, foundation)
    generator = RedesignModel(architecture, foundation)
    lead_record = smiles_to_graph("C", max_nodes=2)
    lead_record["id"] = "review-lead"
    lead = collate_graphs([lead_record])
    condition = FlowCondition(scorer.graph_features(lead), torch.zeros(1, 2),
                              torch.zeros(1, 4), torch.zeros(1, 6),
                              torch.tensor([[.25, -.1]]))
    proposal = replace(condition, response_state=torch.ones(1, 6))
    if mismatch == "lead_features":
        proposal = replace(proposal, lead_features=condition.lead_features + .001)
    elif mismatch == "encoder":
        with torch.no_grad():
            next(generator.encoder.parameters()).add_(.001)
    else:
        with torch.no_grad():
            next(generator.normalizer.buffers()).add_(.001)
    plan = make_sampling_plan(graph, [[1, 0], [0, 1]], 1, num_attempts=2,
                              count_seed=51, noise_seed=52)
    with pytest.raises(ValueError, match=mismatch):
        generate_redesign_proposals(
            scorer, condition, plan, "C", ["C"], tmp_path,
            proposal_model=generator, proposal_condition=proposal,
            neighbourhood_fn=lambda *args: pytest.fail("a mismatch reached chemistry"),
            nfe=4, k=1)
    assert not (tmp_path / "proposal_commit.json").exists()
