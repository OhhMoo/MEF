# Measured campaign data — bounded ChEMBL audit

Audit completed 11 October 2026 at 16:06 PDT. The public API reported **ChEMBL 37**, released 1 May 2026. The audit sampled the first 1,000 IC50 activity IDs for EGFR (`CHEMBL203`) and the first 1,000 for HER2 (`CHEMBL1824`). This is a bounded availability sample.

**Finding:** Paired measured endpoints are available, but this sample needs larger campaigns and a larger chemical representation before it can support the planned measured transfer study. The exact-oracle QM9 pilot can proceed independently.

## Records and overlap

| Quantity | Observed value |
|---|---:|
| Downloaded EGFR / HER2 activities | 1,000 / 1,000 |
| Exact, valid, positive nM measurements retained | 725 / 795 |
| Document–assay pairs with overlapping compounds | 31 |
| Documents represented by those pairs | 8 |
| Paired observation rows / distinct compounds | 228 / 123 |
| Largest three paired libraries | 41, 34, 16 compounds |
| Assay pairs with at least 16 / 32 / 64 / 128 compounds | 3 / 2 / 0 / 0 |

Counts above use the retained records in this sample. A compound can appear in several assay pairs. The 31 pairs therefore do not represent 31 independent experimental campaigns.

Retain only exact (`=`), positive standardized nM outcomes with `standard_flag=1` and no recorded validity concern. Keep censored, partial, and rejected records in the original JSON. Summarize replicates by the median log outcome within the same document, target, assay, and compound. Do not merge different assay conditions.

Use $\mathrm{pIC}_{50}=9-\log_{10}(\mathrm{IC}_{50}\text{ in nM})$. For a prospective selectivity task, endpoint 1 can be $-\mathrm{pIC}_{50}^{\mathrm{HER2}}$ and endpoint 2 $\mathrm{pIC}_{50}^{\mathrm{EGFR}}$: both then increase in the requested direction. These target roles define an example task and require a biological campaign rationale.

## Most populated assay pairs

| Document | EGFR assay | HER2 assay | Paired compounds | Heavy atoms |
|---|---|---|---:|---|
| CHEMBL1136802 | CHEMBL676067 | CHEMBL769140 | 41 | 15–28 |
| CHEMBL1145020 | CHEMBL680028 | CHEMBL769139 | 34 | 25–36 |
| CHEMBL1146682 | CHEMBL674106 | CHEMBL682312 | 16 | 26–43 |

The largest pair has one charged compound and 36 compounds with recorded stereochemistry. All 41 exceed the current nine-heavy-atom limit. Its source document is a 3D-QSAR study of benzylidene malonitrile derivatives; review the origin of its measurements before treating it as an independent campaign. [ChEMBL document](https://www.ebi.ac.uk/chembl/api/data/document/CHEMBL1136802.json), [publication](https://doi.org/10.1021/jm030065n).

The second pair concerns irreversible kinase inhibitors, and the third concerns dual EGFR/erbB2 inhibitors. Their assay descriptions and source publications require review before combining them or assigning a selectivity objective. [Second document](https://www.ebi.ac.uk/chembl/api/data/document/CHEMBL1145020.json), [third document](https://www.ebi.ac.uk/chembl/api/data/document/CHEMBL1146682.json).

## Consequences for dataset selection

For $k=16$, a library needs at least $N+16$ distinct compounds to fill the support and selection budgets. The largest sampled library can fill $N=16$, but cannot fill $N=32$ with 16 separate query compounds. The planned $N=64$ and $128$ curves require larger libraries.

The next data audit should locate larger coherent paired-assay collections, trace reused measurements across documents, and define chemical series before making campaign splits. Representation expansion must address atom count, elements, charge, and stereochemistry on the selected chemistry. Fit endpoint and latent normalization on source data after the split.

## Reproducible artifacts

- [Audit summary and URL/SHA-256 provenance](../../2_data_storage/processed/chembl37_redesign_audit/audit.json).
- [Paired measured observations](../../2_data_storage/processed/chembl37_redesign_audit/paired_observations.csv).
- [Original API records and assay/document metadata](../../2_data_storage/raw/chembl37_redesign_audit/).
- [Audit command output](../../4_experimental_result/result/calibrated_redesign_001/chembl_audit.log).
- [Audit implementation](../../3_model_training/emf/chembl_audit.py).

Primary API sources: [release status](https://www.ebi.ac.uk/chembl/api/data/status.json), [EGFR target](https://www.ebi.ac.uk/chembl/api/data/target/CHEMBL203.json), and [HER2 target](https://www.ebi.ac.uk/chembl/api/data/target/CHEMBL1824.json). The saved release response and source hashes identify the evidence used here.
