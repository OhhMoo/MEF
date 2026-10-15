# Redesign transfer — related work and data study

Sources checked 11 October 2026. This note supports the [research guide](../research_plan.md) and [execution plan](../detail_operation/02_execution_plan.md). It records knowledge used to choose comparisons and data.

## 1. Closest precedents

| Source | Established scope | Implication for EMF |
|---|---|---|
| CoMole | Reports controllable molecular generation and transfer to unseen properties by optimizing task embeddings while freezing the generator | Frozen-generator adaptation is a precedent. Compare explicit evidence-state inference and reusable-direction structure against ordinary task conditioning. |
| SMER / SMER-Opt | Learns directional responses to local edits and composes reusable action primitives in guided molecular optimization | Chemical change–response learning is a precedent. EMF's question concerns its transfer and use inside a continuous generator. |
| Conditional Neural Processes for Molecules | Studies few-shot molecular prediction on DOCKSTRING and a probabilistic Bayesian-optimization experiment | An evidence-conditioned predictor is a strong alternative. Test whether coupling evidence to molecular generation improves decisions. |

Primary sources: [CoMole, arXiv:2605.15354v2](https://arxiv.org/abs/2605.15354v2), [SMER, arXiv:2605.10035](https://arxiv.org/abs/2605.10035), and [Conditional Neural Processes for Molecules, arXiv:2210.09211v3](https://arxiv.org/abs/2210.09211v3).

The EMF architectural hypothesis combines a shared chemical realization mechanism with experimentally inferred response information. Its contribution depends on measured transfer behavior under comparable budgets.

## 2. Candidate source for measured campaigns

ChEMBL brings together chemical structures, bioactivity, and related information. Its official download documentation lists database release 37, dated May 2026, with downloadable database formats and a release DOI. Pin a release when building the audit. [ChEMBL](https://www.ebi.ac.uk/chembl/), [official download documentation](https://chembl.gitbook.io/chembl-interface-documentation/downloads).

Use a coherent document/series and a documented pair of assays as an initial campaign proxy. Verify overlapping compounds, units, targets, experimental conditions, duplicate/censored records, and chemical coverage before choosing a task.

A database entry or target label alone does not establish a comparable experimental campaign. The audit must identify which measurements belong together and which observations are withheld.

## 3. Evidence supplied by each data source

The current local QM9 preparation supplies restricted molecular graphs for the generator. Its stored tensors contain no campaign-specific paired assay outcomes. Synthetic response functions supply exact computational outcomes for the first mechanism test.

A held-out measured library can test candidate prioritization and constrained search using withheld recorded outcomes. An independently evaluated new molecule supplies evidence about generative experimental utility. Keep these evidence types explicit in figures and result notes.

## 4. Questions for the next literature/data review

Read the full methods and public implementations of the closest alternatives before fixing the controlled protocol. Check adaptation inputs, observed-label counts, campaign/chemistry splits, generation versus ranking effects, and evaluator costs.

Choose the first measured task from the audit's coverage and assay coherence. Record the selection and source versions here; put downloaded/processed records in `2_data_storage/`.
