"""Operational checks for source PCA and budgeted exact terminal control."""

import numpy as np
import pytest
import torch

from emf.contracts import GraphMasks, Latents, ModelConfig
from emf.masked_edit_data import MaskedEditDataset, prepare_masked_edit_data
from emf.masked_edit_evaluation import ExactMaskedEvaluator
from emf.masked_edit_model import MaskedEditModel
from emf.masked_edit_oracle_control import (_batch_graphs, _control_from_basis,
                                            _selection_key, build_basis,
                                            cem_exact_control, unguided_oracle_rank)
from emf.masked_edit_sampling import make_paired_gaussian
from emf.models import EMFModel
from emf.redesign_contracts import RedesignConfig


def test_source_basis_and_terminal_budget(tmp_path):
    directory = tmp_path / "masked"
    prepare_masked_edit_data(directory)
    source = MaskedEditDataset(directory, "train", kind="vector")
    heldout = MaskedEditDataset(directory, "validation", kind="vector")
    config = RedesignConfig(ModelConfig(max_nodes=9, node_dim=8, pair_dim=4,
                                        hidden_dim=16, depth=1, num_heads=4),
                            rank=2, state_dim=8, variant="conditional")
    model = MaskedEditModel(config, EMFModel(config.model), variant="C")
    basis = build_basis(model, source, rank=3)
    assert set(basis) == {task["task_id"] for task in source.tasks}
    assert all(1 <= value.nodes.shape[0] <= 3 and value.pairs.shape[0] == value.nodes.shape[0]
               for value in basis.values())
    with pytest.raises(ValueError, match="source split"):
        build_basis(model, heldout)

    cid = heldout.campaign_ids[0]
    meta = heldout.manifest["campaigns"][cid]
    mask = heldout.masks[meta["mask_id"]]
    lead = _batch_graphs([heldout.graph(mask.lead_index)], "cpu")
    edit = GraphMasks(torch.as_tensor(mask.atom_editable[None]),
                      torch.as_tensor(mask.bond_editable[None]))
    condition = model.condition(lead, torch.zeros(1, 2), torch.zeros(1, 4),
                                torch.zeros(1, 8), torch.tensor([[.25, -.10]]), edit)
    gaussian = make_paired_gaussian(config.model, 1, 123)
    controlled = cem_exact_control(model, condition, gaussian, basis[mask.mask_id],
                                   ExactMaskedEvaluator(directory / "private" / "oracle.json"),
                                   cid, mask, terminal_budget=3, nfe=2, seed=11)
    assert controlled.terminal_calls == 3
    assert controlled.forward_nfe == 6
    assert controlled.decoder_calls == 3
    assert controlled.attempts[0]["coefficients"] == [[0.0] * basis[mask.mask_id].nodes.shape[0]] * 2
    assert len(controlled.attempts) == 3
    assert all("protected_copy" in attempt and "exact_delta" in attempt
               for attempt in controlled.attempts)
    assert controlled.best_attempt in controlled.attempts
    matched = unguided_oracle_rank(model, condition,
                                   ExactMaskedEvaluator(directory / "private" / "oracle.json"),
                                   cid, mask, terminal_budget=3, nfe=2, seed=11)
    assert matched.terminal_calls == controlled.terminal_calls
    assert matched.forward_nfe == controlled.forward_nfe
    assert len({attempt["attempt_number"] for attempt in matched.attempts}) == 3


def test_selection_uses_declared_goal_for_valid_misses_and_counts_source_success(tmp_path):
    directory = tmp_path / "masked"
    prepare_masked_edit_data(directory)
    source = MaskedEditDataset(directory, "train", kind="vector")
    cid = source.campaign_ids[0]
    meta = source.manifest["campaigns"][cid]
    mask = source.masks[meta["mask_id"]]
    evaluator = ExactMaskedEvaluator(directory / "private" / "oracle.json")
    outcomes = []
    for index in meta["molecule_indices"]:
        graph = source.graph(index, aligned=True, task_id=mask.mask_id)
        outcomes.append(evaluator.evaluate_graph(graph, mask, cid,
                                                 source_smiles={graph["smiles"]}))
    miss = next(row for row in outcomes if row["exact_delta"] is not None and
                not row["goal_success"])
    source_success = next(row for row in outcomes if row["goal_success"])
    assert _selection_key(miss)[0] == 1.0
    assert source_success["support_excluded_success"]
    assert not source_success["eligible_novel_success"]
    assert _selection_key(source_success)[0] == 0.0


def test_velocity_bin_boundary_uses_tensor_time():
    basis = Latents(torch.ones(1, 9, 1), torch.ones(1, 36, 1))
    controller = _control_from_basis(basis, np.asarray([[1.0], [2.0]]), .95)
    state = Latents(torch.zeros(1, 9, 1), torch.zeros(1, 36, 1))
    assert torch.all(controller(state, torch.tensor([.4749]), None).nodes == 1)
    assert torch.all(controller(state, torch.tensor([.4750]), None).nodes == 2)
