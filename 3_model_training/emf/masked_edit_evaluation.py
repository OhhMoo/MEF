"""Exact masked-edit evaluation from a committed graph and RDKit descriptors.

Generated candidates are scored from their chemical identity. The molecular
response table used for synthetic observations is never queried here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from rdkit import DataStructs

from .campaigns import molecular_fingerprint
from .chem import decode_graph
from .masked_edit_data import DEFAULT_DIR, MAX_NODES, EditMask, _descriptor, _features


@dataclass(frozen=True)
class CommittedCandidate:
    """Immutable chemistry audit captured before any exact response call."""

    mask_id: str
    atoms: tuple[int, ...]
    bonds: tuple[int, ...]
    count: int
    smiles: str | None
    chemically_valid: bool
    connected: bool
    failure_reason: str | None
    protected_copy: bool
    atom_edits: int
    bond_edits: int
    global_unique: bool
    attempt_number: int

    @property
    def total_edits(self) -> int:
        return self.atom_edits + self.bond_edits


@dataclass
class CandidateLedger:
    """Register every attempt and count global molecular identities once."""

    seen_smiles: set[str] = field(default_factory=set)
    attempts: list[CommittedCandidate] = field(default_factory=list)
    scope: str = "one_method"

    def commit(self, graph: dict, mask: EditMask) -> CommittedCandidate:
        atoms = np.asarray(graph.get("atoms", ()))
        bonds = np.asarray(graph.get("bonds", ()))
        count = graph.get("count", -1)
        valid_shape = atoms.shape == (MAX_NODES,) and bonds.shape == (len(mask.lead_bonds),)
        if valid_shape and atoms.dtype.kind in "iu" and bonds.dtype.kind in "iu":
            atom_edits = int(np.count_nonzero(atoms != mask.lead_atoms))
            bond_edits = int(np.count_nonzero(bonds != mask.lead_bonds))
            protected = (isinstance(count, (int, np.integer)) and int(count) == MAX_NODES and
                         not np.any((atoms != mask.lead_atoms) & ~mask.atom_editable) and
                         not np.any((bonds != mask.lead_bonds) & ~mask.bond_editable))
        else:
            atom_edits, bond_edits, protected = -1, -1, False
        outcome = decode_graph(atoms, bonds, count, MAX_NODES)
        unique = bool(outcome.smiles and outcome.smiles not in self.seen_smiles)
        if outcome.smiles is not None:
            self.seen_smiles.add(outcome.smiles)
        record = CommittedCandidate(
            mask.mask_id, tuple(map(int, atoms.reshape(-1))) if atoms.dtype.kind in "iu" else (),
            tuple(map(int, bonds.reshape(-1))) if bonds.dtype.kind in "iu" else (),
            int(count) if isinstance(count, (int, np.integer)) else -1,
            outcome.smiles, outcome.chemically_valid, outcome.connected,
            outcome.failure_reason, bool(protected), atom_edits, bond_edits,
            unique, len(self.attempts))
        self.attempts.append(record)
        return record


class ExactMaskedEvaluator:
    """Oracle-held-out scoring with explicit validity, scope and novelty flags."""

    def __init__(self, oracle_path=DEFAULT_DIR / "private" / "oracle.json"):
        payload = json.loads(Path(oracle_path).read_text())
        if payload.get("schema_version") != 1:
            raise ValueError("incompatible masked-edit oracle")
        self.coefficients = {key: np.asarray(value, dtype=np.float64)
                             for key, value in payload["coefficients"].items()}
        self.scales = np.asarray(payload["scales"], dtype=np.float64)
        self.campaigns = payload["campaigns"]
        self.ledger = CandidateLedger()

    def exact_delta(self, campaign_id: str, smiles: str) -> np.ndarray:
        """Independent truth for any valid supported nine-atom molecule."""
        meta = self.campaigns[campaign_id]
        lead = _descriptor(meta["lead_smiles"]) if "lead_smiles" in meta else None
        if lead is None:
            # The family lead is fixed by the declared mask and the public
            # chemistry axis, not by a response-table entry.
            raise ValueError("oracle lacks lead identity")
        return _features(_descriptor(smiles), lead, self.scales) @ self.coefficients[campaign_id].T

    def evaluate_committed(self, candidate: CommittedCandidate, campaign_id: str,
                           *, support_smiles: Iterable[str] = (),
                           source_smiles: Iterable[str] = (),
                           known_library_smiles: Iterable[str] = ()) -> dict:
        """Call after commit; no outcome is used for validity or uniqueness."""
        meta = self.campaigns[campaign_id]
        if candidate.mask_id != meta["mask_id"]:
            raise ValueError("candidate/campaign mask mismatch")
        support_set = set(support_smiles)
        source_set = set(source_smiles)
        library_set = set(known_library_smiles)
        identity = candidate.smiles
        eligible = (candidate.chemically_valid and candidate.connected and
                    candidate.protected_copy and identity is not None and
                    identity != meta["lead_smiles"])
        delta = self.exact_delta(campaign_id, identity) if eligible else None
        similarity = (float(DataStructs.TanimotoSimilarity(
            molecular_fingerprint(identity), molecular_fingerprint(meta["lead_smiles"])))
            if candidate.chemically_valid and candidate.connected and identity else None)
        success = bool(delta is not None and np.all(delta >= np.asarray(meta["goal"])))
        return {"campaign_id": campaign_id, "mask_id": candidate.mask_id,
                "goal": list(meta["goal"]),
                "attempt_number": candidate.attempt_number, "smiles": identity,
                "chemically_valid": candidate.chemically_valid,
                "connected": candidate.connected,
                "failure_reason": candidate.failure_reason,
                "protected_copy": candidate.protected_copy,
                "atom_edits": candidate.atom_edits, "bond_edits": candidate.bond_edits,
                "total_edits": candidate.total_edits,
                "lead_copy": identity == meta["lead_smiles"],
                "support_match": identity in support_set,
                "source_match": identity in source_set,
                "known_library_match": identity in library_set,
                "global_unique": candidate.global_unique,
                "uniqueness_scope": self.ledger.scope,
                "morgan_similarity": similarity,
                "original_locality": bool(similarity is not None and similarity >= 0.4),
                "exact_delta": None if delta is None else delta.tolist(),
                "goal_success": success,
                "support_excluded_success": bool(success and identity not in support_set
                                                 and identity != meta["lead_smiles"]),
                "original_locality_goal_success": bool(success and similarity is not None
                                                       and similarity >= 0.4),
                "eligible_novel_success": bool(success and identity not in support_set
                                               and identity not in source_set)}

    def evaluate_graph(self, graph: dict, mask: EditMask, campaign_id: str,
                       **novelty_sets) -> dict:
        candidate = self.ledger.commit(graph, mask)
        return self.evaluate_committed(candidate, campaign_id, **novelty_sets)


def exact_evaluate(graph: dict, mask: EditMask, campaign_id: str,
                   evaluator: ExactMaskedEvaluator, **novelty_sets) -> dict:
    """Convenience entry point; commits the graph before querying the oracle."""
    return evaluator.evaluate_graph(graph, mask, campaign_id, **novelty_sets)
