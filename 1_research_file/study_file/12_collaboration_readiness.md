# Collaboration readiness — when to start and what to bring

This note turns the current experiment into a staged collaboration plan. It supports the [research guide](../research_plan.md), the [transfer and data study](07_redesign_transfer_study.md), the [measured-campaign audit](08_measured_campaign_data_study.md), and the [1014 experiment record](../result_log/experiment_results_1014.md).

## Readiness decision

Start exploratory conversations with relevant professors now. A reproducible proof of concept shows that sparse measurements steer a continuous molecular generator on held-out simulated campaigns; incorporating newly revealed evidence lets a partially generated design continue with similar success at lower charged compute than restarting. Use an initial conversation to test the question, comparisons, and next mechanism experiment. No statistical-win threshold is needed for asking for scientific feedback.

## Readiness ladder

| Stage | Start point | What the conversation or claim can cover |
|---|---|---|
| Exploratory scientific feedback | Now, with the current reproducible result | Ask whether the problem framing, strongest alternative, and proposed mechanism test are scientifically useful. Share the current evidence and its measured costs plainly. |
| Substantive methods collaboration | After a three-seed paired-compute benchmark and one focused mechanism test | Ask a generative-learning or control researcher whether evidence changes useful proposal dynamics beyond matched sample-and-rank, ordinary conditioning, or response prediction. A mixed result can still guide the next experiment. |
| Chemistry and data collaboration | Can start now, in parallel with model work | The measured-data bottleneck is already established. Ask a medicinal chemist or assay/data group to identify one coherent lead series with two jointly measured endpoints, define chemical eligibility and a meaningful design decision, and advise on series splits. The bounded ChEMBL 37 audit found only three paired assay libraries with at least 16 compounds and none with 64; the largest sampled pair had 41 compounds, all above the current nine-heavy-atom limit. |
| Retrospective measured-series claims | After a held-out measured assay/series evaluation | Claims can cover decision quality on recorded experimental outcomes for the defined series and split, with source provenance, assay comparability, and matched measurement/candidate budgets. |
| Prospective experimental claims | After independently measuring selected new compounds in a defined series | Base utility claims on synthesis or acquisition and assay outcomes for newly selected compounds under a recorded budget. |

These are evidence stages, not gates for starting contact. Data and chemistry partners can make the next stage possible before the methods benchmark is complete.

## Short internal brief for an initial conversation

**Question.** Can sparse measurements from a new molecular-design campaign recalibrate reusable chemical changes in a pretrained continuous generator, so it improves one endpoint while preserving another without relearning molecular generation?

**Current evidence and scope.** On held-out synthetic response campaigns, 16 observations give 57.3% selected joint-profile success for terminal control from 64 proposals per case; the small unguided pool gives 51.6%, and a 1,600-proposal sample-and-rank pool gives 70.8%. The larger pool uses 25 times as many proposals. Control has 81.2% valid attempts versus 90.8% for the small pool, and costs 19.6 times as much recorded time. With direct response coefficients, control and ordinary conditioning have similar test success (440/768 and 435/768). Continuing after evidence arrives near 25%, 50%, and 75% of generation gives 195, 193, and 191 distinct campaign-specific successes, versus 194 for restart, with 12.9%, 24.8%, and 37.3% fewer charged traversals. This is a computational result on two small chemical families with known descriptors and synthetic responses. The [1014 record](../result_log/experiment_results_1014.md) links seed counts and artifacts.

**Next three figures.**

1. **B validity and steering.** Pair validity and protected-category preservation with evidence-driven steering across support sizes. This tests whether the more reliable masked representation retains useful response to campaign evidence.
2. **Matched quality–cost frontier.** Plot joint-profile successes against completed-batch inference cost for control, ordinary conditioning, fresh sampling and ranking, and masked mutation. Show cap-limited cases, unused selection slots, validity, and training cost separately.
3. **Conflict and revision versus restart.** Compare continuation and restart as evidence is refined or the goal reverses. Show campaign-paired success and charged/incremental cost at common wall budgets, with campaign-cluster uncertainty across seeds.

Unfamiliar chemical series and biological assay transfer remain later evidence stages; they are not implied by these immediate figures.

**The ask.** For a machine-learning/control researcher: help design and interpret the paired-compute mechanism test, especially a strong response-predictor-plus-search control and a useful measure of evidence-driven local changes. For a medicinal-chemistry or assay researcher: help identify a coherent lead series with two comparable measured endpoints, enough compounds for support and query sets, and a decision that would matter in practice. A data partner can start immediately; the current audit shows that finding the right measured series is a project bottleneck.

## What to have ready before a substantive methods meeting

- A two-page brief tailored to the recipient, with the figure questions and one specific ask.
- A runnable checkpoint, pinned configuration, source/data versions, exact command, and a command or notebook that regenerates the headline table.
- Seed-level outcomes, invalid attempts, compute accounting, and a clear label for synthetic versus measured evidence; use the artifact links in the [1014 result record](../result_log/experiment_results_1014.md).
- A one-page protocol covering campaign splits, equal observations and candidate/evaluator budgets, training cost, validity rules, and uncertainty summaries.
- For data discussions, a series audit of endpoints, units, assay conditions, replicates/censoring, structure normalization, overlap, series boundaries, and support/query counts.

Use the current brief for exploratory feedback before every item is polished. The checklist is for a focused methods or data collaboration.

## Stop and pivot criteria

Predefine the paired-compute evaluation and summarize campaign-level uncertainty. Do not make a single p-value or arbitrary minimum gain the continuation gate.

- If the advantage disappears against matched response-prediction/search or ordinary conditioning, narrow the contribution to evidence-responsive generation and test why explicit control adds no decision value.
- If gains depend on invalid chemistry, revise constraints. If valid local yield matches sampling while control remains much more expensive, pursue a cheaper mechanism or use sample-and-rank as the practical method.
- If seeds disagree or uncertainty spans meaningful benefit and harm, add campaigns or seeds only when they resolve the ambiguity; keep the result mixed until then.
- If no coherent measured series supports a held-out test, make data access and assay definition the active work. If measured outcomes show no decision benefit over strong prediction-and-search, stop the experimental-recalibration claim.
- Reserve prospective utility claims for independently measured new compounds.

## Candidate public data resources

These are starting points for an audit, not preselected campaigns. Each source combines records from multiple assays or contributors, so provenance and assay comparability still need review.

- [ChEMBL downloads](https://chembl.gitbook.io/chembl-interface-documentation/downloads): the official download page reports release 37 (May 2026) and provides database exports. The local bounded audit is already pinned to this release.
- [PubChem BioAssay downloads](https://pubchem.ncbi.nlm.nih.gov/docs/downloads): NCBI documents BioAssay record downloads and programmatic access; source provenance and contributor licensing are surfaced with records.
- [BindingDB downloads](https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp): the public download page is described by the BindingDB resource paper as providing date-stamped database and subset files. Check access and release details when auditing; the current site did not respond to the page fetch during this note's review.

The immediate ask is early feedback on the paired quality–cost and revision design, alongside help finding a coherent measured series with two relevant endpoints.
