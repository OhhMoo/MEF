"""Bounded, release-checked availability audit for paired measured endpoints.

This retrieves at most 1,000 IC50 records for each of EGFR and HER2. It reports
observed document/assay overlap; it does not treat the bounded sample as the
full ChEMBL database or establish biological campaign comparability.
"""

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import urllib.parse
import urllib.request

import numpy as np
from rdkit import Chem

from .chem import smiles_to_graph


API = "https://www.ebi.ac.uk/chembl/api/data"
TARGETS = {"on_target": "CHEMBL203", "off_target": "CHEMBL1824"}


def _fetch_json(url: str, path: Path) -> tuple[dict, dict]:
    started = time.monotonic()
    if path.exists():
        payload = path.read_bytes()
        retrieval = "cached"
    else:
        request = urllib.request.Request(url, headers={"User-Agent": "EMF-bounded-data-audit/1.0"})
        with urllib.request.urlopen(request, timeout=45) as response:
            payload = response.read(8 * 1024 * 1024 + 1)
        if len(payload) > 8 * 1024 * 1024:
            raise ValueError("bounded API response exceeded 8 MiB")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        retrieval = datetime.now(timezone.utc).isoformat()
    return json.loads(payload), {"url": url, "file": str(path),
                                 "sha256": hashlib.sha256(payload).hexdigest(),
                                 "retrieved_utc": retrieval, "bytes": len(payload),
                                 "seconds": time.monotonic() - started}


def _complete_value(record):
    if record.get("standard_units") != "nM" or record.get("standard_relation") != "=":
        return None
    if record.get("standard_flag") != 1 or record.get("data_validity_comment"):
        return None
    try:
        value = float(record["standard_value"])
    except (TypeError, ValueError, KeyError):
        return None
    if not np.isfinite(value) or value <= 0:
        return None
    return 9 - float(np.log10(value))  # pIC50: -log10(IC50 in mol/L).


def _chemistry(smiles: str | None) -> dict:
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return {"parsed": False, "supported": False, "reason": "invalid_smiles"}
    result = {"parsed": True, "heavy_atoms": mol.GetNumHeavyAtoms(),
              "elements": sorted({atom.GetSymbol() for atom in mol.GetAtoms()}),
              "charged": any(atom.GetFormalCharge() for atom in mol.GetAtoms()),
              "stereochemistry": any(atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
                                      for atom in mol.GetAtoms()) or
                                any(bond.GetStereo() != Chem.BondStereo.STEREONONE
                                    for bond in mol.GetBonds())}
    try:
        smiles_to_graph(smiles)
        result.update(supported=True, reason=None)
    except ValueError as error:
        result.update(supported=False, reason=str(error))
    return result


def audit_chembl(raw_dir, output_dir, *, expected_release: str = "ChEMBL_37",
                 records_per_target: int = 1000) -> dict:
    if not 1 <= records_per_target <= 1000:
        raise ValueError("bounded audit accepts at most 1000 records per target")
    raw_dir, output_dir = Path(raw_dir), Path(output_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    status, status_provenance = _fetch_json(f"{API}/status.json", raw_dir / "status.json")
    if status.get("chembl_db_version") != expected_release:
        raise ValueError(f"Expected {expected_release}, observed {status.get('chembl_db_version')}")
    provenance = [status_provenance]
    records_by_role = {}
    for role, target in TARGETS.items():
        query = urllib.parse.urlencode({"target_chembl_id": target, "standard_type": "IC50",
                                       "limit": records_per_target, "offset": 0,
                                       "order_by": "activity_id"})
        response, source = _fetch_json(f"{API}/activity.json?{query}", raw_dir / f"{role}_activities.json")
        provenance.append(source)
        records_by_role[role] = response["activities"]
        print(json.dumps({"event": "chembl_records", "role": role,
                          "downloaded": len(response["activities"]),
                          "database_matches": response["page_meta"]["total_count"]}), flush=True)
    measured, molecule_smiles, exclusions = defaultdict(list), {}, Counter()
    for role, records in records_by_role.items():
        for record in records:
            molecule = record["molecule_chembl_id"]
            if record.get("canonical_smiles"):
                molecule_smiles[molecule] = record["canonical_smiles"]
            value = _complete_value(record)
            if value is None:
                exclusions[(role, "non_exact_non_nM_invalid_or_missing")] += 1
                continue
            key = (record["document_chembl_id"], role, record["assay_chembl_id"], molecule)
            measured[key].append((value, record["activity_id"]))
    # Replicates are summarized within an assay only, in oriented log units.
    rows = {}
    for key, values in measured.items():
        rows[key] = {"pIC50": float(np.median([value for value, _ in values])),
                     "replicate_count": len(values), "activity_ids": [identity for _, identity in values]}
    groups = defaultdict(lambda: {"on_target": defaultdict(dict), "off_target": defaultdict(dict)})
    for (document, role, assay, molecule), values in rows.items():
        groups[document][role][assay][molecule] = values
    overlaps, paired_rows = [], []
    for document, assays_by_role in groups.items():
        for on_assay, on_compounds in assays_by_role["on_target"].items():
            for off_assay, off_compounds in assays_by_role["off_target"].items():
                common = sorted(set(on_compounds) & set(off_compounds))
                if not common:
                    continue
                chemistry = [_chemistry(molecule_smiles.get(molecule)) for molecule in common]
                counts = [item["heavy_atoms"] for item in chemistry if item["parsed"]]
                overlaps.append({"document_chembl_id": document, "on_assay": on_assay,
                                 "off_assay": off_assay, "paired_compounds": len(common),
                                 "on_only_compounds": len(set(on_compounds) - set(off_compounds)),
                                 "off_only_compounds": len(set(off_compounds) - set(on_compounds)),
                                 "supported_current_representation": sum(item["supported"] for item in chemistry),
                                 "heavy_atoms_min": min(counts) if counts else None,
                                 "heavy_atoms_max": max(counts) if counts else None,
                                 "charged_compounds": sum(item.get("charged", False) for item in chemistry),
                                 "stereochemical_compounds": sum(item.get("stereochemistry", False) for item in chemistry),
                                 "unsupported_reasons": dict(Counter(item.get("reason") for item in chemistry
                                                                      if not item["supported"]))})
                for molecule in common:
                    paired_rows.append({"document": document, "on_assay": on_assay,
                                        "off_assay": off_assay, "molecule": molecule,
                                        "smiles": molecule_smiles.get(molecule),
                                        "endpoint_1_off_target_avoidance": -off_compounds[molecule]["pIC50"],
                                        "endpoint_2_on_target_potency": on_compounds[molecule]["pIC50"],
                                        "on_replicates": on_compounds[molecule]["replicate_count"],
                                        "off_replicates": off_compounds[molecule]["replicate_count"]})
    overlaps.sort(key=lambda row: (-row["paired_compounds"], row["document_chembl_id"], row["on_assay"], row["off_assay"]))
    for document in sorted({row["document_chembl_id"] for row in overlaps[:5]}):
        _, source = _fetch_json(f"{API}/document/{document}.json", raw_dir / f"document_{document}.json")
        provenance.append(source)
    assays = sorted({row[key] for row in overlaps[:5] for key in ("on_assay", "off_assay")})
    for assay in assays:
        _, source = _fetch_json(f"{API}/assay/{assay}.json", raw_dir / f"assay_{assay}.json")
        provenance.append(source)
    with (output_dir / "paired_observations.csv").open("w", newline="") as stream:
        fields = ["document", "on_assay", "off_assay", "molecule", "smiles",
                  "endpoint_1_off_target_avoidance", "endpoint_2_on_target_potency",
                  "on_replicates", "off_replicates"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(paired_rows)
    summary = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
               "observed_release": status["chembl_db_version"],
               "release_date": status["chembl_release_date"], "targets": TARGETS,
               "scope": f"First {records_per_target} IC50 activity IDs per target; bounded availability sample",
               "record_counts": {role: len(records) for role, records in records_by_role.items()},
               "complete_exact_record_counts": {role: sum(len(values) for key, values in measured.items()
                                                            if key[1] == role) for role in TARGETS},
               "excluded_record_counts": {f"{role}:{reason}": count for (role, reason), count in exclusions.items()},
               "replicate_policy": "median pIC50 within the same document, target, assay and molecule",
               "units": "IC50 nM -> mol/L; pIC50=9-log10(IC50_nM)",
               "orientation": "endpoint1=-off-target pIC50; endpoint2=on-target pIC50; both larger-is-better",
               "partial_or_censored": "retained in original activity JSON; not assigned complete endpoints",
               "overlap_assay_pairs": len(overlaps), "overlaps": overlaps, "provenance": provenance,
               "boundary": "Document/assay overlap is a campaign proxy; methods and series require further review"}
    (output_dir / "audit.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "chembl_audit_ready", "overlap_assay_pairs": len(overlaps),
                      "largest_pair": overlaps[0] if overlaps else None,
                      "output_dir": str(output_dir)}), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="2_data_storage/raw/chembl37_redesign_audit")
    parser.add_argument("--output-dir", default="2_data_storage/processed/chembl37_redesign_audit")
    parser.add_argument("--release", default="ChEMBL_37")
    parser.add_argument("--records-per-target", type=int, default=1000)
    args = parser.parse_args()
    audit_chembl(args.raw_dir, args.output_dir, expected_release=args.release,
                 records_per_target=args.records_per_target)


if __name__ == "__main__":
    main()
