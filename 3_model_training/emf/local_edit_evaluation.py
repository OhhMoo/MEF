"""Slot-preserving reconstruction and locality audit for local-edit proposals.

The caller commits raw attempt records before supplying an exact-response
callback. This module does not read hidden campaign coefficients or query labels.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from time import perf_counter
from typing import Callable, Iterable

import numpy as np
import torch
from rdkit import DataStructs

from .campaigns import molecular_fingerprint
from .chem import decode_graph, smiles_to_graph
from .data import collate_graphs
from .local_edit_data import aligned_graph


def audit_attempts(lead_smiles: str, attempts: Iterable[dict], *,
                   reserved_support_smiles: Iterable[str] = (),
                   response_lookup: Callable[[str], np.ndarray] | None = None,
                   goal=(0.25, -0.10), min_similarity: float = 0.4,
                   max_edits: int | None = None,
                   max_count_difference: int = 2,
                   max_nodes: int = 9) -> tuple[list[dict], dict]:
    """Audit each decoded slot and, if supplied, its external exact response.

    ``attempts`` needs ``atoms``, ``bonds``, and ``count`` or ``atom_count``.
    A previous ``smiles`` field is checked against decoding. Locality requires
    the original Morgan and atom-count-difference rule; an optional edit limit
    is a separately requested secondary rule. Exact edits for same-count graphs
    are reported. Distinct success credits the first occurrence of each identity;
    every slot, including invalid and duplicate slots, stays in the denominator.
    ``response_lookup`` is evaluation-only and is called after reconstruction.
    """
    if (not 0 <= min_similarity <= 1 or max_count_difference < 0
            or (max_edits is not None and max_edits < 0)):
        raise ValueError("invalid locality rule")
    goal = np.asarray(goal, dtype=np.float64)
    if goal.shape != (2,) or not np.isfinite(goal).all():
        raise ValueError("goal must contain two finite endpoints")
    started = perf_counter()
    lead_graph = smiles_to_graph(lead_smiles, max_nodes)
    lead_smiles = lead_graph["smiles"]
    lead_count = lead_graph["count"]
    lead_fingerprint = molecular_fingerprint(lead_smiles)
    reserved = {smiles_to_graph(identity, max_nodes)["smiles"]
                for identity in reserved_support_smiles}
    reserved.add(lead_smiles)
    seen = set()
    chemistry_cache = {}
    rows = []
    for slot, attempt in enumerate(attempts):
        count = attempt.get("count", attempt.get("atom_count"))
        outcome = decode_graph(attempt["atoms"], attempt["bonds"], count, max_nodes)
        smiles = outcome.smiles
        prior_smiles = attempt.get("smiles")
        reconstruction_match = prior_smiles is None or prior_smiles == smiles
        if not reconstruction_match:
            raise ValueError(f"attempt {slot} recorded a different decoded identity")
        valid = bool(outcome.chemically_valid and outcome.connected)
        duplicate = valid and smiles in seen
        if valid:
            seen.add(smiles)
        similarity = None
        alignment = None
        if valid:
            if smiles not in chemistry_cache:
                try:
                    chemistry_cache[smiles] = (
                        float(DataStructs.TanimotoSimilarity(
                            lead_fingerprint, molecular_fingerprint(smiles))),
                        (aligned_graph(lead_smiles, smiles, max_nodes)
                         if count == lead_count == max_nodes else None))
                except ValueError:
                    chemistry_cache[smiles] = (None, None)
            similarity, alignment = chemistry_cache[smiles]
        within_neighbourhood = bool(valid and similarity is not None
                                    and similarity >= min_similarity
                                    and abs(count - lead_count) <= max_count_difference)
        within_edit_limit = (None if max_edits is None else
                             bool(alignment is not None and alignment["total_edits"] <= max_edits))
        local = within_neighbourhood and (max_edits is None or within_edit_limit)
        eligible = bool(local and not duplicate and smiles not in reserved)
        response = None
        if eligible and response_lookup is not None:
            response = np.asarray(response_lookup(smiles), dtype=np.float64)
            if response.shape != (2,) or not np.isfinite(response).all():
                raise ValueError("external response must contain two finite endpoints")
        success = bool(response is not None and np.all(response >= goal))
        rows.append({"attempt_id": attempt.get("attempt_id", slot),
                     "smiles": smiles, "chemically_valid": outcome.chemically_valid,
                     "connected": outcome.connected, "failure_reason": outcome.failure_reason,
                     "reconstruction_match": reconstruction_match,
                     "duplicate": bool(duplicate), "reserved_support_match": smiles in reserved,
                     "lead_similarity": similarity,
                     "lead_atom_count_difference": abs(count - lead_count)
                                                   if isinstance(count, (int, np.integer)) else None,
                     "canonical_to_lead": (alignment["canonical_to_lead"].tolist()
                                           if alignment is not None else None),
                     "lead_to_canonical": (alignment["lead_to_canonical"].tolist()
                                           if alignment is not None else None),
                     "atom_edits": alignment["atom_edits"] if alignment is not None else None,
                     "bond_edits": alignment["bond_edits"] if alignment is not None else None,
                     "total_edits": alignment["total_edits"] if alignment is not None else None,
                     "within_neighbourhood": within_neighbourhood,
                     "within_edit_limit": within_edit_limit,
                     "within_local_edit_rule": local, "eligible": eligible,
                     "response": response.tolist() if response is not None else None,
                     "profile_success": success})
    total = len(rows)
    summary = {"attempts": total,
               "connected_valid_attempts": sum(row["connected"] and row["chemically_valid"]
                                               for row in rows),
               "local_distinct_attempts": sum(row["within_local_edit_rule"]
                                              and not row["duplicate"] for row in rows),
               "eligible_distinct_attempts": sum(row["eligible"] for row in rows),
               "distinct_local_profile_success_attempts": sum(row["profile_success"] for row in rows),
               "distinct_local_profile_yield_per_attempt":
                   (sum(row["profile_success"] for row in rows) / total if total else 0.0),
               "host_chemistry_audit_seconds": perf_counter() - started}
    return rows, summary


@torch.no_grad()
def reconstruction_audit(model, dataset, device, *, batch_size: int = 64,
                         output_path=None) -> dict:
    """Decode E(graph) for every lead-aligned family member and audit edits.

    This is a decoder check, independent of source or held-out response labels.
    ``dataset`` is an ``AlignedSteeringDataset``. The JSON contains one row per
    family member and aggregate identity, label, and local reconstruction rates.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not hasattr(dataset, "alignment_cache"):
        raise TypeError("reconstruction audit requires aligned family graphs")
    device = torch.device(device)
    was_training = model.training
    model.eval()
    started = perf_counter()
    records = []
    for family, members in dataset.alignment_cache.items():
        lead_smiles = dataset.smiles[dataset.family_leads[family]]
        graphs = [dataset.aligned_graph(family, index) for index in sorted(members)]
        for offset in range(0, len(graphs), batch_size):
            chunk = graphs[offset:offset + batch_size]
            batch = collate_graphs(chunk).to(device)
            logits = model.decode(model.encode(batch), batch.masks)
            decoded_atoms = logits.atoms.argmax(-1).cpu().numpy()
            decoded_bonds = logits.bonds.argmax(-1).cpu().numpy()
            for graph, atoms, bonds in zip(chunk, decoded_atoms, decoded_bonds):
                outcome = decode_graph(atoms, bonds, graph["count"], dataset.max_nodes)
                valid = bool(outcome.chemically_valid and outcome.connected)
                edit = aligned_graph(lead_smiles, outcome.smiles, dataset.max_nodes) if valid else None
                records.append({"family": family, "smiles": graph["smiles"],
                                "identity_match": valid and outcome.smiles == graph["smiles"],
                                "decoded_smiles": outcome.smiles, "chemically_valid": valid,
                                "failure_reason": outcome.failure_reason,
                                "atom_label_matches": int(np.count_nonzero(atoms == graph["atoms"])),
                                "bond_label_matches": int(np.count_nonzero(bonds == graph["bonds"])),
                                "target_total_edits": graph["total_edits"],
                                "decoded_atom_edits": edit["atom_edits"] if edit else None,
                                "decoded_bond_edits": edit["bond_edits"] if edit else None,
                                "decoded_total_edits": edit["total_edits"] if edit else None})
    model.train(was_training)
    total = len(records)
    summary = {"molecules": total,
               "identity_matches": sum(row["identity_match"] for row in records),
               "identity_accuracy": (sum(row["identity_match"] for row in records) / total if total else 0.0),
               "connected_valid_reconstructions": sum(row["chemically_valid"] for row in records),
               "atom_label_accuracy": sum(row["atom_label_matches"] for row in records)
                                      / (total * dataset.max_nodes) if total else 0.0,
               "bond_label_accuracy": sum(row["bond_label_matches"] for row in records)
                                      / (total * dataset.max_nodes * (dataset.max_nodes - 1) / 2)
                                      if total else 0.0,
               "target_edit_distribution": dict(sorted(Counter(
                   row["target_total_edits"] for row in records).items())),
               "decoded_edit_distribution": dict(sorted(Counter(
                   row["decoded_total_edits"] for row in records
                   if row["decoded_total_edits"] is not None).items())),
               "host_chemistry_and_model_seconds": perf_counter() - started}
    report = {"summary": summary, "records": records}
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
