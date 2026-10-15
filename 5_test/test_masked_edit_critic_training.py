"""Source critic collection keeps valid labels and honest audit partitions."""

from collections import Counter

import numpy as np
import torch

from emf.contracts import GraphLogits, GraphMasks
from emf.masked_edit_critic_training import (_Record, _append_decoded,
                                             _identity_split, _rank)


def _logits(atoms, bonds):
    return GraphLogits(torch.nn.functional.one_hot(torch.tensor(atoms), 4).float() * 20,
                       torch.nn.functional.one_hot(torch.tensor(bonds), 4).float() * 20)


def test_only_valid_connected_decodes_receive_rdkit_labels():
    # Four C atoms, a connected chain versus four disconnected atoms.
    connected = _logits([[0, 0, 0, 0]], [[1, 0, 0, 1, 0, 1]])
    disconnected = _logits([[0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0]])
    logits = GraphLogits(torch.cat((connected.atoms, disconnected.atoms)),
                         torch.cat((connected.bonds, disconnected.bonds)))
    masks = GraphMasks.from_counts(torch.tensor([4, 4]), 4)
    records, counts = [], Counter()
    outcomes = _append_decoded(records, logits, masks, np.ones(2),
                               "unguided", "audit", counts, "mask_a")
    assert len(records) == 1
    assert outcomes[0][0] == "CCCC" and outcomes[1][0] is None
    assert records[0].mask_id == "mask_a"
    assert counts["unguided_audit_attempts"] == 2
    assert counts["unguided_audit_valid_connected"] == 1
    assert counts["unguided_audit_disconnected"] == 1
    withheld, withheld_counts = [], Counter()
    _append_decoded(withheld, connected,
                    GraphMasks.from_counts(torch.tensor([4]), 4), np.ones(2),
                    "unguided", "train", withheld_counts, "mask_a", {"CCCC"})
    assert not withheld
    assert withheld_counts["unguided_train_excluded_heldout_identity"] == 1


def test_identity_group_fallback_preserves_separate_audit_seed():
    graph = _logits([[0, 0, 0, 0]], [[1, 0, 0, 1, 0, 1]])
    masks = GraphMasks.from_counts(torch.tensor([4]), 4)
    label = torch.ones(1, 2)
    records = [_Record(graph, masks, label, "CCCC", "hard_source", "hard"),
               _Record(graph, masks, label, "CCCC", "unguided", "train"),
               _Record(graph, masks, label, "CCCC", "unguided", "audit")]
    train, audit, split = _identity_split(records, 17)
    assert not split["grouped_canonical_identity"]
    assert len(train) == 2 and len(audit) == 1
    assert split["identity_overlap"] == 1


def test_rank_assigns_equal_values_equal_average_rank():
    np.testing.assert_allclose(_rank(np.array([2., 1., 2., 4.])),
                               np.array([1.5, 0., 1.5, 3.]))
