"""CPU checks for paired evidence, blind commitments and attempt denominators."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from emf.chem import smiles_to_graph
from emf import steering_evaluation as evaluation
from emf.redesign_evaluation import commit_proposals


class TinyDataset:
    def __init__(self, data_dir, split):
        self.data_dir = Path(data_dir)
        self.split = split
        self.manifest = {}
        self.graph_dataset = object()
        self.campaign_ids = ("plus", "minus")
        self.campaign_pairs = {"plus": "minus", "minus": "plus"}
        self.campaign_family = {"plus": 0, "minus": 0}
        self.family_leads = (0,)
        self.family_indices = (np.arange(4),)
        self.smiles = ("CC", "CF", "CO", "CN")
        self.ids = self.smiles
        self.context = np.zeros(4, dtype=np.float32)
        self.observations = np.zeros((2, 4, 2), dtype=np.float32)
        self.observations[0, 1, 0] = 1.0
        self.observations[1, 1, 0] = -1.0
        self.observations[0, 2:, 0] = [.5, -.5]
        self.observations[1, 2:, 0] = [-.5, .5]
        self.count_prior = np.zeros((4, 4), dtype=np.float64)
        self.count_prior[1, 1] = 1.0

    def graph(self, index):
        return {**smiles_to_graph(self.smiles[index], max_nodes=4), "id": self.ids[index]}

    def nested_support(self, campaign_index, lead_index, *, seed, sizes):
        assert lead_index == 0
        pool = (0, 1)
        records = {size: {"indices": np.asarray(pool[:size]),
                          "values": self.observations[campaign_index, :size],
                          "context": self.context,
                          "graphs": [self.graph(index) for index in pool[:size]]}
                   for size in sizes}
        return {"records": records, "reserve_indices": pool,
                "reserve_ids": tuple(self.ids[index] for index in pool),
                "reserve_smiles": tuple(self.smiles[index] for index in pool),
                "lead": self.graph(0), "lead_values": self.observations[campaign_index, 0]}


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.config = SimpleNamespace(max_nodes=4, node_dim=2, pair_dim=2)

    def graph_features(self, graph):
        return torch.ones((len(graph.counts), 3), device=self.weight.device)

    def support_state(self, support):
        return support.values[:, :, :1].mean(1).repeat(1, 64)

    def predict_response_features(self, lead, candidate, baseline, context, state):
        first = state[:, 0] * (5 - 2 * candidate[:, 0])
        mean = torch.stack((first, torch.zeros_like(first)), dim=-1)
        covariance = .1 * torch.eye(2).expand(len(first), 2, 2)
        return SimpleNamespace(mean=mean, covariance_matrix=covariance)


def test_paired_evaluation_commits_before_oracle_and_replays(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "manifest.json").write_text("{}\n")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    dataset = TinyDataset(data_dir, "test")
    import emf.steering_data as steering_data
    monkeypatch.setattr(steering_data, "SteeringDataset", lambda *_: dataset)
    model = TinyModel()
    source = SimpleNamespace(redesign=SimpleNamespace(variant="factorized", rank=8), seed=142)
    monkeypatch.setattr(evaluation, "_load_model", lambda *_: (model, source, {}))
    monkeypatch.setattr(evaluation, "cache_graph_features",
                        lambda *_args, **_kwargs: (
                            torch.arange(4, dtype=torch.float32)[:, None].repeat(1, 3), 0.0))
    generated = []

    def fake_generate(scorer, condition, plan, lead_smiles, reserved, destination, **kwargs):
        assert not scorer.training
        assert len(plan.counts) == 4
        sign = float(condition.response_state[0, 0])
        smiles = "CO" if sign > 0 else "CN"
        rows = []
        for attempt, identity in enumerate((smiles, smiles, None, "CF")):
            valid = identity is not None
            rows.append({"attempt_id": attempt, "smiles": identity,
                         "chemically_valid": valid, "connected": valid,
                         "supported_chemistry": valid, "duplicate": attempt == 1,
                         "reserved_support_match": identity == "CF", "lead_match": False,
                         "within_neighbourhood": valid, "profile_score": .9,
                         "eligible": attempt == 0})
        ranking = [{"rank": 1, "attempt_id": 0, "smiles": smiles, "profile_score": .9}]
        commit_proposals(destination, rows, ranking, metadata={"metadata": kwargs["metadata"]})
        generated.append((str(destination), plan.count_seed, plan.noise_seed,
                          tuple(kwargs["metadata"]["support_ids"])))
        return rows, ranking, {}

    monkeypatch.setattr(evaluation, "generate_redesign_proposals", fake_generate)

    class Oracle:
        calls = 0
        coefficients = {"plus": 1.0, "minus": -1.0}

        def score_smiles(self, campaign, smiles):
            # All four proposal ledgers (two native, two opposite) must exist.
            assert len(list(output.rglob("proposal_commit.json"))) == 4
            self.calls += 1
            favorable = (campaign == "plus" and smiles == "CO") or (
                campaign == "minus" and smiles == "CN")
            return np.asarray([.5 if favorable else -.5, 0.0])

    output = tmp_path / "evaluation"
    config = SimpleNamespace(data_dir=str(data_dir), device="cpu", cpu_threads=1,
                             support_sizes=[2], num_attempts=4, nfe=2, k=1,
                             eval_seed=100, T=.95, sample_batch_size=4,
                             cache_batch_size=4, bf16=False)
    oracle = Oracle()
    report = evaluation.run_evaluation(config, checkpoint, "test", output, oracle=oracle)
    assert report["batches"] == 4
    assert len(generated) == 2  # opposite condition is an exact paired replay
    assert generated[0][1:] == generated[1][1:]
    assert oracle.calls == 4  # two identities for each target campaign
    assert report["aggregate"][0]["physical_sampling_attempts"] == 8
    assert report["aggregate"][0]["logical_sampling_attempts"] == 16
    import csv
    rows = list(csv.DictReader((output / "metrics.csv").open()))
    assert all(float(row["steering_benefit"]) == .25 for row in rows)
    assert all(float(row["paired_decoded_changed_fraction"]) == .5 for row in rows)
    assert all(float(row["local_profile_yield_per_attempt"]) ==
               (.25 if row["condition"] == "correct" else 0) for row in rows)
    assert all(float(row["raw_slot_local_profile_yield_per_attempt"]) ==
               (.5 if row["condition"] == "correct" else 0) for row in rows)
    assert all(float(row["response_first_endpoint_sign_accuracy"]) ==
               (1.0 if row["condition"] == "correct" else 0.0) for row in rows)
    assert all(float(row["response_first_endpoint_spearman"]) ==
               pytest.approx(1.0 if row["condition"] == "correct" else -1.0) for row in rows)
    assert all(json.loads(path.read_text())["oracle_access_before_commit"] is False
               for path in output.rglob("proposal_commit.json"))
    repeated = evaluation.run_evaluation(config, checkpoint, "test", output, oracle=oracle)
    assert oracle.calls == 4
    assert repeated["aggregate"][0]["physical_oracle_calls"] == 4
    assert len(generated) == 2


def test_rejects_unpaired_campaigns():
    dataset = SimpleNamespace(campaign_ids=("a", "b"),
                              campaign_pairs={"a": "b", "b": "b"},
                              campaign_family={"a": 0, "b": 0},
                              family_leads=(0,))
    with pytest.raises(ValueError, match="symmetric"):
        evaluation._campaign_groups(dataset)
