"""Private, exact graph-response oracle for the first redesign mechanism test.

Responses are synthetic dimensionless endpoints, both oriented larger-is-better.
Their features and nonlinear interactions are independent of the learned graph
embedding. Oracle coefficients never belong in campaign/model conditioning.
"""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import Crippen, Lipinski, rdMolDescriptors

from .chem import smiles_to_graph


FEATURE_NAMES = (
    "heavy_atoms_over_9", "carbon_over_9", "nitrogen_over_9",
    "oxygen_over_9", "fluorine_over_9", "hetero_fraction",
    "rings_over_4", "aromatic_rings_over_3", "hbd_over_4",
    "hba_over_6", "tpsa_over_100", "logp_over_5",
    "rotatable_bonds_over_5", "fraction_csp3",
    "multiple_bonds_over_8", "molar_refractivity_over_50",
)


def structural_features(smiles: str, max_nodes: int = 9) -> np.ndarray:
    """Validate supported chemistry and compute one deterministic feature row."""
    graph = smiles_to_graph(smiles, max_nodes)
    mol = Chem.MolFromSmiles(graph["smiles"])
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    n = len(atoms)
    multiple_bonds = sum(bond.GetBondTypeAsDouble() > 1 for bond in mol.GetBonds())
    values = (
        n / 9, atoms.count("C") / 9, atoms.count("N") / 9,
        atoms.count("O") / 9, atoms.count("F") / 9,
        sum(symbol != "C" for symbol in atoms) / n,
        rdMolDescriptors.CalcNumRings(mol) / 4,
        rdMolDescriptors.CalcNumAromaticRings(mol) / 3,
        Lipinski.NumHDonors(mol) / 4, Lipinski.NumHAcceptors(mol) / 6,
        rdMolDescriptors.CalcTPSA(mol) / 100, Crippen.MolLogP(mol) / 5,
        Lipinski.NumRotatableBonds(mol) / 5,
        rdMolDescriptors.CalcFractionCSP3(mol), multiple_bonds / 8,
        Crippen.MolMR(mol) / 50,
    )
    features = np.asarray(values, dtype=np.float64)
    if not np.isfinite(features).all():
        raise ValueError("nonfinite_structural_features")
    return features


@dataclass(frozen=True)
class OracleCampaign:
    linear: np.ndarray
    quadratic: np.ndarray
    frequencies: np.ndarray
    phases: np.ndarray
    sine_weights: np.ndarray
    offset: np.ndarray

    @classmethod
    def draw(cls, rng: np.random.Generator, dimension: int = len(FEATURE_NAMES)):
        # Each endpoint has its own dense interactions and nonlinear projections.
        return cls(
            rng.normal(size=(dimension, 2)) / np.sqrt(dimension),
            rng.normal(size=(dimension, dimension, 2)) / dimension,
            rng.normal(scale=2.0, size=(dimension, 4, 2)),
            rng.uniform(-np.pi, np.pi, size=(4, 2)),
            rng.normal(scale=0.2, size=(4, 2)),
            rng.normal(scale=0.2, size=2),
        )

    def evaluate(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float64)
        linear = x @ self.linear
        interactions = np.einsum("...i,ije,...j->...e", x, self.quadratic, x)
        angles = np.einsum("...i,ike->...ke", x, self.frequencies) + self.phases
        nonlinear = (np.sin(angles) * self.sine_weights).sum(axis=-2)
        return linear + 0.35 * np.tanh(interactions) + nonlinear + self.offset

    def as_dict(self) -> dict:
        return {name: getattr(self, name).tolist() for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, values):
        return cls(**{name: np.asarray(values[name], dtype=np.float64)
                      for name in cls.__dataclass_fields__})


class ToyOracle:
    """Evaluator-only response functions and a source-fitted endpoint transform."""

    def __init__(self, campaigns: dict[str, OracleCampaign], *,
                 mean=None, std=None, max_nodes: int = 9):
        self.campaigns = campaigns
        self.mean = np.zeros(2) if mean is None else np.asarray(mean, dtype=np.float64)
        self.std = np.ones(2) if std is None else np.asarray(std, dtype=np.float64)
        self.max_nodes = max_nodes
        if self.mean.shape != (2,) or self.std.shape != (2,) or np.any(self.std <= 0):
            raise ValueError("endpoint normalizer needs two finite positive scales")
        if not np.isfinite(self.mean).all() or not np.isfinite(self.std).all():
            raise ValueError("endpoint normalizer must be finite")

    @classmethod
    def create(cls, *, seed: int = 42, source: int = 32,
               validation: int = 8, test: int = 8, max_nodes: int = 9):
        rng = np.random.default_rng(seed)
        campaigns = {f"{split}_{index:03d}": OracleCampaign.draw(rng)
                     for split, size in (("train", source), ("validation", validation),
                                         ("test", test)) for index in range(size)}
        return cls(campaigns, max_nodes=max_nodes)

    def fit_source_normalization(self, source_features: np.ndarray) -> None:
        """Use source campaigns and source molecular identities only."""
        values = np.stack([campaign.evaluate(source_features)
                           for name, campaign in self.campaigns.items()
                           if name.startswith("train_")])
        self.mean = values.mean(axis=(0, 1))
        self.std = values.std(axis=(0, 1)).clip(1e-4)

    def score_features(self, campaign_id: str, features: np.ndarray) -> np.ndarray:
        raw = self.campaigns[campaign_id].evaluate(features)
        return (raw - self.mean) / self.std

    def score_smiles(self, campaign_id: str, smiles: str) -> np.ndarray:
        """Exact outcome for any supported valid graph, including novel graphs."""
        return self.score_features(campaign_id, structural_features(smiles, self.max_nodes))

    evaluate_smiles = score_smiles

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        values = {"schema_version": 1, "feature_names": list(FEATURE_NAMES),
                  "mean": self.mean.tolist(), "std": self.std.tolist(),
                  "max_nodes": self.max_nodes,
                  "campaigns": {name: campaign.as_dict()
                                for name, campaign in self.campaigns.items()}}
        path.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path):
        values = json.loads(Path(path).read_text())
        if values["feature_names"] != list(FEATURE_NAMES):
            raise ValueError("incompatible oracle feature specification")
        return cls({name: OracleCampaign.from_dict(campaign)
                    for name, campaign in values["campaigns"].items()},
                   mean=values["mean"], std=values["std"],
                   max_nodes=values["max_nodes"])
