# Experiment-calibrated molecular redesign

## A research question for discussion

Can a saved state in a chemically constrained molecular flow make redesign cheaper when campaign evidence or goals change, while retaining valid edits and the required molecular profile?

The project studies molecular redesign under a limited synthesis-and-assay budget. A team has a lead molecule, a small number of measured analogues, and two goals: improve one property and retain another. The model should reuse its prior chemical experience as measurements or design goals change.

## What is established

The October 14 study provides a reproducible computational proof of concept. On 16 held-out simulated campaigns with 16 measured compounds per campaign, terminal control selected 440 successful compounds among 768 requested slots (57.3%). An unguided pool of 64 proposals per case selected 396/768 (51.6%). A sample-and-rank pool of 1,600 proposals per case selected 544/768 (70.8%). The larger pool used 25 times as many proposals and about 28% more recorded time than terminal control. It remains the strongest result for selected success.

Control has a clear operating cost: 706.7 seconds recorded generation/evaluation time across seeds and 81.2% valid connected attempts, compared with 36.1 seconds and 90.8% validity for the small unguided pool. At N=16 with inferred response coefficients, conditional generation selects 428/768 successes (55.7%) versus control's 440/768 (57.3%). With coefficients supplied directly, conditional generation selects 435/768 versus control's 440/768. Conditional generation has far lower deployment cost. These comparisons establish evidence responsiveness in the simulator while leaving efficient constrained redesign as the research question.

The model uses two small chemical families, known molecular descriptors, and synthetic noisy campaign responses. The original encoder/flow initializer predates the molecular split. The result therefore supports a mechanism study in this setting; it does not yet measure transfer to biological assays or unfamiliar chemical series. The [complete result record](../result_log/experiment_results_1014.md) links seed counts, timing, validity, and output artifacts.

One encouraging revision result is that incorporating evidence revealed partway through generation retained similar final success to restarting. At 25%, 50%, and 75% generation progress, continuation produced 195, 193, and 191 distinct campaign-specific successes versus 194 for restart, with 12.9%, 24.8%, and 37.3% fewer charged forward/backward traversals. This is a descriptive success–compute comparison, not a matched-compute superiority result; the restart reference was reused across timing comparisons.

## The next test, now being prepared

The next experiment asks whether reusing a saved continuous state improves the quality–cost tradeoff when new evidence changes the decision, and whether that behavior survives a more chemically reliable masked generator. The selected masked model B had 94.4% prior validity, exact preservation of protected graph categories, and a local yield similar to the other representations in the earlier pilot. B preparation is starting; the revision benchmark follows, and both outcomes are pending.

The paired benchmark will compare controlled and ordinary-conditional continuation and restart, unguided sample-and-rank, and discrete masked mutation. It will pair campaigns, seeds, leads, masks, support sets, and sampling noise. It will report successful distinct compounds among 16 selections across completed-batch inference budgets, with chemistry validation, ranking, and terminal decoding included in cost. Training and inference costs will be reported separately. A larger sampling pool is the main practical comparator.

The central mechanism question is: **when new evidence arrives, does the saved continuous state help produce more valid, useful local edits per unit of computation than restarting generation, ordinary conditional continuation, or spending the budget on fresh sampling?** The [revision plan](06_revision_efficiency_experiment_plan.md) defines evidence refinement and goal reversal as separate update settings, the paired comparisons, accounting, and pivot decisions.

## Two complementary collaboration paths

**Methods and dynamics.** A collaborator in generative learning, control, or optimization could help sharpen the paired-compute test and interpret how evidence changes local proposal dynamics. Useful advice would include whether to add a response-predictor-plus-search control and how to distinguish changes in proposals from improvements due to ranking. The result can guide whether to retain continuous-state revision, use ordinary conditioning, or simplify toward sampling and ranking.

**Measured campaign data.** A medicinal chemistry, assay, or data partner could help identify one coherent lead series with two comparable measured endpoints, define which compounds are eligible, and specify a meaningful joint-profile decision. This work can begin before the methods benchmark finishes. A bounded ChEMBL 37 audit found three paired assay libraries with at least 16 compounds and none with 64; the largest sampled pair had 41 compounds, all above the current nine-heavy-atom limit. Finding an appropriate series and expanding representation coverage are active research needs. See the [audit](../study_file/08_measured_campaign_data_study.md).

We welcome early scientific advice on the question and next test, and collaboration on either path before a biological benchmark is complete. The immediate goal is to make the test decisive and identify a measured series that can support a later held-out study. Retrospective assay-transfer claims require held-out measured series; prospective utility claims require independent measurements on newly selected compounds.

## Useful materials for a first discussion

- [Current 1014 results and artifacts](../result_log/experiment_results_1014.md).
- [Revision efficiency experiment plan](06_revision_efficiency_experiment_plan.md).
- [Collaboration readiness and stage criteria](../study_file/12_collaboration_readiness.md).
- [ChEMBL 37 measured-campaign audit](../study_file/08_measured_campaign_data_study.md).

The immediate ask is early advice on the paired quality–cost test and help identifying a coherent two-endpoint molecular series for a later measured study.
