"""One restricted molecular vocabulary, from training SMILES to final graphs.

Atom labels are C/N/O/F; packed pairs are no/single/double/triple bonds.
Hydrogens are implicit, stereochemistry is removed, and padding is ignored.
"""

from dataclasses import dataclass

import numpy as np
from rdkit import Chem, rdBase

from .contracts import ATOM_TYPES, BOND_TYPES

_ATOM_INDEX = {symbol: index for index, symbol in enumerate(ATOM_TYPES)}
_BONDS = (None, Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE)
_BOND_INDEX = {bond: index for index, bond in enumerate(_BONDS) if bond is not None}


@dataclass(frozen=True)
class ChemistryOutcome:
    """A terminal attempt, including identities of sanitized disconnected graphs."""

    smiles: str | None
    chemically_valid: bool
    connected: bool
    failure_reason: str | None

    @property
    def canonical_smiles(self) -> str | None:
        return self.smiles


def smiles_to_graph(smiles: str, max_nodes: int = 9) -> dict:
    """Standardize one molecule and return padded labels in the shared ordering.

    Unsupported chemistry raises ValueError. Identity is canonical, with no
    stereo or isotope labels; unsupported isotope input is rejected explicitly.
    Kekulization supplies integer bond orders, including for aromatic rings.
    """
    if max_nodes < 1:
        raise ValueError("max_nodes must be positive")
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("invalid_smiles")
    mol = Chem.RemoveHs(mol)
    Chem.RemoveStereochemistry(mol)
    count = mol.GetNumAtoms()
    if not 1 <= count <= max_nodes:
        raise ValueError("atom_count")
    if len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError("disconnected")
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in _ATOM_INDEX:
            raise ValueError("unsupported_element")
        if atom.GetFormalCharge() != 0:
            raise ValueError("formal_charge")
        if atom.GetNumRadicalElectrons() != 0:
            raise ValueError("radical")
        if atom.GetIsotope() != 0:
            raise ValueError("isotope")

    identity = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    # Reparse the canonical identity to establish a reproducible atom order.
    mol = Chem.MolFromSmiles(identity)
    Chem.Kekulize(mol, clearAromaticFlags=True)
    atoms = np.zeros(max_nodes, dtype=np.uint8)
    adjacency = np.zeros((max_nodes, max_nodes), dtype=np.uint8)
    for atom in mol.GetAtoms():
        atoms[atom.GetIdx()] = _ATOM_INDEX[atom.GetSymbol()]
    for bond in mol.GetBonds():
        label = _BOND_INDEX.get(bond.GetBondType())
        if label is None:
            raise ValueError("unsupported_bond")
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adjacency[i, j] = adjacency[j, i] = label
    i, j = np.triu_indices(max_nodes, k=1)
    return {"atoms": atoms, "bonds": adjacency[i, j], "count": count,
            "smiles": identity}


def decode_graph(atoms, bonds, count: int, max_nodes: int | None = None) -> ChemistryOutcome:
    """Choose no new categories: validate one already-decoded categorical graph.

    Values beyond count are padding and are ignored. Every active pair is read
    once in upper-triangle order. Invalid outputs are returned, never repaired.
    """
    def invalid(reason: str) -> ChemistryOutcome:
        return ChemistryOutcome(None, False, False, reason)

    try:
        atoms, bonds = np.asarray(atoms), np.asarray(bonds)
        max_nodes = len(atoms) if max_nodes is None else max_nodes
        if (atoms.shape != (max_nodes,) or
                bonds.shape != (max_nodes * (max_nodes - 1) // 2,)):
            return invalid("shape")
        if (not isinstance(count, (int, np.integer)) or
                not 1 <= count <= max_nodes):
            return invalid("atom_count")
        if atoms.dtype.kind not in "iu" or bonds.dtype.kind not in "iu":
            return invalid("noninteger_label")
        i, j = np.triu_indices(max_nodes, k=1)
        active_pairs = j < count
        if np.any(atoms[:count] >= len(ATOM_TYPES)) or np.any(atoms[:count] < 0):
            return invalid("atom_label")
        if (np.any(bonds[active_pairs] >= len(BOND_TYPES)) or
                np.any(bonds[active_pairs] < 0)):
            return invalid("bond_label")

        builder = Chem.RWMol()
        for label in atoms[:count]:
            builder.AddAtom(Chem.Atom(ATOM_TYPES[int(label)]))
        for left, right, label in zip(i[active_pairs], j[active_pairs], bonds[active_pairs]):
            if label:
                builder.AddBond(int(left), int(right), _BONDS[int(label)])
        mol = builder.GetMol()
        with rdBase.BlockLogs():
            Chem.SanitizeMol(mol)
        if any(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()):
            return invalid("radical")
        connected = len(Chem.GetMolFrags(mol)) == 1
        smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
        return ChemistryOutcome(smiles, True, connected,
                                None if connected else "disconnected")
    except Chem.AtomValenceException:
        return invalid("invalid_valence")
    except (ValueError, TypeError, RuntimeError, OverflowError):
        return invalid("sanitize_error")
