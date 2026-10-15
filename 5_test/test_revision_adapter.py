"""Checks for source-only bounded state revision, including a tiny fit/save run."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from emf.contracts import GraphBatch, GraphMasks, Latents, ModelConfig, pair_indices
from emf.masked_edit_model import MaskedEditCondition
from emf.revision_adapter import (AdapterConfig, BoundedStateAdapter, TeacherPair,
                                  _build_pairs, editable_rms, load_adapter,
                                  state_error, train)
from emf.redesign_train import digest, tensor_fingerprint


def fixture(batch=2):
    rng = torch.Generator().manual_seed(892)
    valid = GraphMasks.from_counts(torch.full((batch,), 3), 4)
    edit = GraphMasks(torch.tensor([[False, True, True, False]]).expand(batch, -1),
                      torch.tensor([[True, False, False, True, False, False]]).expand(batch, -1))
    lead = Latents(torch.randn(batch, 4, 3, generator=rng),
                   torch.randn(batch, 6, 2, generator=rng)).masked(valid)
    graph = GraphBatch(torch.zeros(batch, 4, dtype=torch.long),
                       torch.zeros(batch, 6, dtype=torch.long), valid)
    old = MaskedEditCondition(torch.zeros(batch, 6), torch.zeros(batch, 2),
                              torch.ones(batch, 4), torch.zeros(batch, 64),
                              torch.tensor([[.25, -.1]]).expand(batch, -1),
                              lead, graph, edit)
    evidence = old.response_state.clone()
    evidence[:, :6] = torch.tensor([1., .3, -.2, .1, -.5, .2])
    new = replace(old, response_state=evidence)
    state = Latents(torch.where(edit.nodes[..., None],
                               torch.randn(batch, 4, 3, generator=rng), lead.nodes),
                    torch.where(edit.pairs[..., None],
                                torch.randn(batch, 6, 2, generator=rng), lead.pairs))
    return state, old, new


def adapter():
    return BoundedStateAdapter(AdapterConfig(3, 2))


def move_heads(model):
    with torch.no_grad():
        for net in (model.node_map, model.pair_map):
            net[-1].weight.normal_(0., .2)
            net[-1].bias.fill_(10.)


def test_initial_identity_and_first_step_have_finite_gradient():
    state, old, new = fixture()
    model = adapter()
    out = model.apply_state(state, .75, old, new)
    torch.testing.assert_close(out.nodes, state.nodes, rtol=0, atol=0)
    torch.testing.assert_close(out.pairs, state.pairs, rtol=0, atol=0)
    target = Latents(state.nodes + .2 * new.edit_masks.nodes[..., None],
                     state.pairs - .1 * new.edit_masks.pairs[..., None])
    state_error(out, target, new.edit_masks).mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.node_map[-1].bias.grad.abs().sum() > 0
    assert model.pair_map[-1].bias.grad.abs().sum() > 0


def test_hard_projection_and_per_example_joint_rms_bound():
    state, old, new = fixture()
    model = adapter()
    move_heads(model)
    # Corrupt protected/padded input deliberately: output must restore lead/zero.
    corrupt = Latents(state.nodes + (~new.edit_masks.nodes)[..., None] * 123.,
                      state.pairs - (~new.edit_masks.pairs)[..., None] * 456.)
    out = model.apply_state(corrupt, torch.tensor([.7, .75]), old, new)
    for name in ("nodes", "pairs"):
        value, lead = getattr(out, name), getattr(new.lead, name)
        edit, valid = getattr(new.edit_masks, name), getattr(new.masks, name)
        assert torch.equal(value[valid & ~edit], lead[valid & ~edit])
        assert torch.count_nonzero(value[~valid]) == 0
    displacement = Latents(out.nodes - state.nodes, out.pairs - state.pairs)
    torch.testing.assert_close(editable_rms(displacement, new.edit_masks),
                               torch.full((2,), .5), rtol=1e-5, atol=1e-6)
    half = model.apply_state(state, .75, old, new, gain=.5)
    assert torch.all(editable_rms(Latents(half.nodes - state.nodes,
                                        half.pairs - state.pairs), new.edit_masks) <= .250001)


def test_unchanged_context_is_exact_identity_even_after_training():
    state, old, _ = fixture()
    model = adapter()
    move_heads(model)
    out = model.apply_state(state, .75, old, old)
    assert torch.equal(out.nodes, state.nodes)
    assert torch.equal(out.pairs, state.pairs)


def test_empty_edit_region_and_invalid_inputs():
    state, old, new = fixture()
    model = adapter()
    no_edit = GraphMasks(torch.zeros_like(old.edit_masks.nodes),
                         torch.zeros_like(old.edit_masks.pairs))
    empty_old, empty_new = replace(old, edit_masks=no_edit), replace(new, edit_masks=no_edit)
    out = model.apply_state(state, .75, empty_old, empty_new)
    assert torch.equal(out.nodes, new.lead.nodes)
    assert torch.equal(out.pairs, new.lead.pairs)
    assert torch.equal(editable_rms(out, no_edit), torch.zeros(2))
    with pytest.raises(ValueError, match="gain"):
        model.apply_state(state, .75, old, new, gain=2.)
    with pytest.raises(ValueError, match="time"):
        model.apply_state(state, 1., old, new)
    with pytest.raises(ValueError, match="preserve aligned lead"):
        model.apply_state(state, .75, old,
                           replace(new, lead=Latents(new.lead.nodes + 1., new.lead.pairs)))


def test_atom_and_induced_pair_permutation_equivariance():
    state, old, new = fixture()
    model = adapter()
    move_heads(model)
    permutation = torch.tensor([2, 0, 1, 3])
    indices = pair_indices(4).T.tolist()
    lookup = {tuple(pair): index for index, pair in enumerate(indices)}
    pair_permutation = torch.tensor([lookup[tuple(sorted((int(permutation[i]), int(permutation[j]))))]
                                     for i, j in indices])

    def permute_latents(value):
        return Latents(value.nodes[:, permutation], value.pairs[:, pair_permutation])

    def permute_masks(value):
        return GraphMasks(value.nodes[:, permutation], value.pairs[:, pair_permutation])

    def permute_condition(value):
        graph = value.lead_graph
        return replace(value, lead=permute_latents(value.lead), edit_masks=permute_masks(value.edit_masks),
                       lead_graph=GraphBatch(graph.atoms[:, permutation], graph.bonds[:, pair_permutation],
                                             permute_masks(graph.masks)))

    expected = permute_latents(model.apply_state(state, .75, old, new))
    actual = model.apply_state(permute_latents(state), .75,
                               permute_condition(old), permute_condition(new))
    torch.testing.assert_close(actual.nodes, expected.nodes)
    torch.testing.assert_close(actual.pairs, expected.pairs)


def test_serialization_and_base_hash_guard(tmp_path):
    state, old, new = fixture()
    model = adapter()
    move_heads(model)
    base = tmp_path / "base.pt"
    base.write_bytes(b"unchanged conditional B")
    path = tmp_path / "adapter.pt"
    torch.save({"schema_version": 1, "kind": "bounded_revision_adapter",
                "adapter_config": vars(model.config), "adapter_state": model.state_dict(),
                "base_sha256": digest(base), "base_fingerprint_before": "same",
                "base_fingerprint_after": "same"}, path)
    restored, _ = load_adapter(path, base_checkpoint=base)
    expected = model.apply_state(state, .75, old, new)
    actual = restored.apply_state(state, .75, old, new)
    assert torch.equal(actual.nodes, expected.nodes)
    assert torch.equal(actual.pairs, expected.pairs)
    assert all(not p.requires_grad for p in restored.parameters())
    base.write_bytes(b"other model")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_adapter(path, base_checkpoint=base)


def test_fixed_fit_freezes_base_preserves_rng_and_saves_final_even_if_validation_worse(tmp_path):
    state, old, new = fixture(batch=3)
    target = Latents(state.nodes + .2 * new.edit_masks.nodes[..., None],
                     state.pairs + .2 * new.edit_masks.pairs[..., None])
    provenance = {"campaign_id": "source_c0", "target_editable_rms": [.2] * 3}
    source_pair = TeacherPair(state, target, old, new, .75, provenance)
    # Opposite validation target ensures final checkpoint is retained regardless
    # of validation improvement; validation never chooses a checkpoint here.
    val_target = Latents(state.nodes - .2 * new.edit_masks.nodes[..., None],
                         state.pairs - .2 * new.edit_masks.pairs[..., None])
    val_pair = replace(source_pair, new_state=val_target,
                       provenance={"campaign_id": "validation_c0", "target_editable_rms": [.2] * 3})
    base_model = torch.nn.Linear(2, 2)
    base_model.variant = "B"
    base_model.config = ModelConfig(max_nodes=4, node_dim=3, pair_dim=2,
                                     hidden_dim=4, depth=1, num_heads=1)
    base_model.redesign_config = SimpleNamespace(context_dim=4, state_dim=64)
    before = tensor_fingerprint(base_model.state_dict().items())
    base_checkpoint = tmp_path / "base.pt"
    torch.save(base_model.state_dict(), base_checkpoint)
    for name in ("manifest.json", "train.npz", "validation.npz"):
        (tmp_path / name).write_bytes(b"mock public data")

    def dataset(data_dir, split, kind):
        return SimpleNamespace(split=split, molecule_splits={"train": {1, 2}})

    def pairs(model, data, task, device):
        assert all(not p.requires_grad for p in model.parameters())
        return [source_pair] if data.split == "train" else [val_pair]

    random_before = torch.get_rng_state().clone()
    with patch("emf.revision_adapter.load_model", return_value=(base_model, {"conditioned": True})), \
            patch("emf.revision_adapter.MaskedEditDataset", side_effect=dataset), \
            patch("emf.revision_adapter._build_pairs", side_effect=pairs):
        result = train({"checkpoint": str(base_checkpoint), "output": str(tmp_path / "fit"),
                        "data_dir": str(tmp_path), "seed": 81, "steps": 12,
                        "batch_size": 4, "validation_every": 6}, "cpu")
    assert torch.equal(torch.get_rng_state(), random_before)
    assert tensor_fingerprint(base_model.state_dict().items()) == before
    assert all(p.grad is None and not p.requires_grad for p in base_model.parameters())
    assert result["steps"] == 12
    assert result["base_unchanged"]
    assert result["final_metrics"]["validation_state_mse"] > result["identity_validation_state_mse"]
    restored, payload = load_adapter(result["checkpoint"], base_checkpoint=base_checkpoint)
    assert payload["final_metrics"]["step"] == 12
    assert payload["selected_gain"] == 1.
    assert payload["source_provenance"]["training_split"] == "train"
    assert not payload["source_provenance"]["test_access"]
    learned = restored.apply_state(state, .75, old, new)
    assert state_error(learned, target, new.edit_masks).mean() < state_error(state, target, new.edit_masks).mean()


def test_source_only_and_fixed_architecture_guards(tmp_path):
    task = {"checkpoint": "unused", "output": str(tmp_path), "seed": 1}
    with pytest.raises(ValueError, match="source train"):
        train({**task, "split": "test"})
    with pytest.raises(ValueError, match="no sweep"):
        train({**task, "gains": [.5, 1.]})
    with pytest.raises(ValueError, match="source or validation"):
        _build_pairs(None, SimpleNamespace(split="test"), task, "cpu")
