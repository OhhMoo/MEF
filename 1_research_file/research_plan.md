# EMF — Experiment-Calibrated Molecular Flows

This is the high-level research guide. The scientific direction is efficient, experimentally recalibratable molecular redesign. The [math plan](detail_operation/01_math_plan.md) and [first execution plan](detail_operation/02_execution_plan.md) record the initial factorized implementation. The current [revision-efficiency plan](detail_operation/06_revision_efficiency_experiment_plan.md) tests whether retained continuous states make changed design decisions cheaper while preserving molecular requirements.

## 1. The question

> Can a continuous molecular generator learn reusable chemical changes whose experimental effects can be recalibrated from a few measurements in a new drug-discovery campaign, allowing it to improve one property while preserving another without relearning how to generate molecules?

The learning question is: **can we transfer the ability to redesign molecules?** The current mechanism question is: **can reusable, chemically constrained trajectories be redirected efficiently as evidence or design requirements change?**

A new campaign should benefit from previous chemical experience. Its first measurements should teach the model how to use that experience in the new context.

## 2. The practical decision

A team has a promising lead. One endpoint needs improvement; another is already acceptable. The team has a small measurement set and must choose which analogues to test next.

Start with reduced off-target activity while maintaining on-target potency, or a developability endpoint paired with potency when the data are stronger. Success means meeting both requirements in the same compound.

The eventual industrial outcome is **more compounds meeting the complete profile per synthesis-and-assay budget**. Generation validity, speed, and diversity support that outcome. They remain useful diagnostics throughout development.

Report $\operatorname{success}(N,k)$, the fraction of $k$ selected compounds that meet both requirements, and $\operatorname{efficiency}(N,k)$, successes per allocated $N+k$ compound budget. Plot them over support sizes $N\in\{16,32,64,128\}$ on held-out campaigns, using external outcomes and counting the measured lead within $N$. The [math plan](detail_operation/01_math_plan.md) defines both quantities.

## 3. The architectural idea

Separate three roles:

| Knowledge or input | Role |
|---|---|
| Reusable chemical capability | Realize context-sensitive changes in continuous molecular representations |
| Campaign evidence | Infer what those changes are expected to accomplish here |
| Design goal | Select changes that improve one endpoint while retaining the other |

Keep the continuous graph representation, clean-prediction flow matching, and terminal decoding developed from ELF. Preserve the full lead structure and enforce permitted edits with hard graph masks. Use measured evidence to calibrate the effects of available edits. Compare direct conditioning and explicit response-based control, including their ability to continue from a saved intermediate state.

During source training, learn the chemical generator and its use across campaigns. On a new campaign, freeze the pretrained model and infer its response state from the available measurements. As observations accumulate, the state changes how the flow proposes chemistry and how it scores candidates.

**The hypothesis is that experiment-calibrated local edit dynamics make redesign more transferable and more efficient in its use of measurements and computation.** The completed controlled study demonstrates local evidence-dependent steering. The next requirement is useful revision at a common inference budget, with chemical validity, protected structure and retained endpoint requirements measured on decoded molecules.

The initial low-rank correction showed little evidence-dependent generation. The subsequent [masked-edit study](result_log/experiment_results_1014.md) established useful steering but found that larger sample-and-rank pools outperform expensive control. Continuing after more evidence retained similar success with less charged computation. These findings motivate efficient revision as the current focus. The contribution we seek is a demonstrated improvement in transferable local redesign and joint-profile decisions. The [related-work study](study_file/07_redesign_transfer_study.md) tracks the closest precedents and their implications.

## 4. The scientific contribution to earn

The ICML target is a learning principle for structured inverse design: **share the machinery that realizes changes, and infer their campaign-specific consequences from sparse observations**.

The evidence should establish three points:

1. New observations change useful generative behavior on unfamiliar campaigns.
2. The change improves joint-profile success with fewer target measurements.
3. The architectural organization improves useful decisions or revision efficiency beyond a strong response predictor, ordinary conditioning, or additional search.

Hold out genuine assay/campaign contexts and chemical series. Measure transfer separately from interpolation within familiar series. Compare under the same observations, chemical eligibility, candidate/evaluator budgets, and transparent compute costs.

The strongest alternative is a good few-shot predictor coupled to a standard generator or Bayesian optimization. It defines the practical comparison the architecture must meet.

## 5. The path from a small model to a useful system

| Stage | Research purpose | Evidence |
|---|---|---|
| Exact-response molecular pilot | Establish evidence-driven recalibration inside the flow | Held-out nonlinear campaigns, generated molecules scored by an independent known oracle, support-size curves |
| Measured-library transfer | Establish experimental relevance on unfamiliar chemistry | Jointly measured endpoints, campaign/series splits, retrospective decisions using withheld observations |
| Collaborative prospective test | Establish utility for new compounds | A lead series, selected analogues, independent synthesis/assay results, and a counted experimental budget |

The existing small 2D generator supplies the computational foundation. Extend it directly into the first exploratory model, then add comparisons to explain the behavior observed.

The exact-response pilot supports a computational mechanism claim. A measured library supports retrospective decision claims. Newly generated chemistry needs independent evaluation to establish its experimental effects.

## 6. The three central figures

| Figure | Message |
|---|---|
| Joint-profile success versus target measurements | How efficiently does the model learn useful redesign decisions? |
| Transfer across held-out campaigns and chemical series | What chemical and experimental knowledge is reused? |
| Revision success versus actual computation and update time | When does retaining a continuous state help, compared with restart, conditional generation and expanded search? |

Measure proposal quality before ranking as well as success after selection. Report structural proximity to the lead, distinct successes, failed attempts, and uncertainty across independent campaigns.

The property predictions of a fixed molecule depend on evidence and assay context. Changing the requested goal changes selection and generation; changing the measurements changes expected effects.

## 7. Collaboration strategy

The small reproducible result now supports an exploratory scientific discussion. Seek feedback on the constrained-revision question and help defining a coherent measured campaign while the next benchmark runs.

Complete one bounded study of rollback, intermediate-state recoverability, and a small learned revision map, following the [final experiment plan](detail_operation/08_revision_recovery_experiment_plan.md). Stop after these three experiments and their recorded decision gates. A clear explanation of the failure can support outreach even when the learned revision map does not improve performance.

Prepare a one-page proposal, two main experimental figures, and a short reproducibility appendix containing source/data provenance, configurations, costs, and a runnable checkpoint. Lead with the discovery, its best-supported explanation, and the proposed next mechanism.

Seek complementary expertise in structured generative learning, medicinal chemistry, and assay design. The first useful collaboration supplies a coherent lead series with two measured endpoints and helps define an experimental decision worth improving.

For a methods collaboration, bring paired quality–cost curves, a clear unresolved mechanism, and a runnable checkpoint. For a chemistry/data collaboration, ask for a coherent lead series with jointly measured endpoints and a meaningful preservation requirement. The [readiness note](study_file/12_collaboration_readiness.md) and [collaboration brief](detail_operation/07_collaboration_brief.md) define those distinct discussions. A prospective experiment follows retrospective measured-series validation.

The eventual proposal is concrete: **test whether experimentally recalibrated chemical capability reduces the measurements and computation needed to improve a lead's joint profile**. Choose collaborators around that question and the evidence needed to answer it.

## 8. Decisions that can change the direction

| Established finding | Research decision |
|---|---|
| Evidence improves pre-ranking redesign and the advantage survives matched comparisons | Extend to measured campaigns and broader source chemistry |
| Benefits are explained by response prediction/ranking alone | Rework the route by which evidence shapes the flow |
| Ordinary conditioning matches the structured correction | Revise the factorization around a demonstrated transfer failure |
| New campaigns require changes outside the learned chemical repertoire | Study how chemical capability expands while retaining earlier knowledge |
| Measured joint-profile gains persist across independent series | Seek a prospective experimental collaboration |

Keep the first model focused on 2D lead redesign and two endpoints. Let observed failures determine extensions.

## 9. How this guide is maintained

Update this guide when an explicit research decision or an established finding changes the question, mechanism, evidence required, or collaboration goal. Record routine work and preliminary results in daily logs.

Use the research folders consistently:

- `detail_operation/`: undated numbered mathematics and execution plans.
- `study_file/`: undated numbered knowledge notes, source studies, and literature/data studies.
- `result_log/`: implementation and collected-result records named `implementation_log_MMDD.md` and `experiment_results_MMDD.md`, with actual run dates in the body.

The project's value will be measured by how much previous chemical experience helps a new campaign learn to improve its molecules.
