# Experiment results — 1013

<!-- REVIEWED_LOCAL_EDIT_RESULTS_BEGIN -->
## Reviewed overnight continuation: local steering now responds to a few measurements

**Reviewed: 2026-10-13 11:54:45 PDT. Runs finished October 13 at 03:42 PDT.** The workflow used **4.222 GPU-hours**, including the repaired operation check, against an eight GPU-hour cap. All eight initial models reached 20,000 updates. Three of four longer-training comparisons completed their planned additional 10,000 updates; the remaining comparison stopped after 9,356. Both frozen calibration runs completed. No GPU jobs remain active for this study.

**The main result:** the lead-relative model can use four simulated measurements to produce more successful local molecular changes while all generator weights remain fixed. It produced **119 successes with correct evidence and 66 with reversed evidence**, each from 8,192 attempts. The advantage appears in both seeds and both chemical families.

### What counts as success

A candidate must be chemically valid and connected, satisfy the supported chemistry rules, differ from the lead and reserved measurement molecules, remain within the declared lead neighborhood, improve endpoint 1 by at least 0.25, and lose no more than 0.10 on endpoint 2. Locality uses the original Morgan similarity threshold of 0.4 and atom-count rule. Invalid attempts, repeated proposals, and unsuccessful candidates remain in the denominator.

The numbers below sum **distinct successful molecules within each campaign-and-seed case**. Each condition uses 16 test campaigns × two seeds × 256 attempts = 8,192 slots. A molecule that succeeds in several campaigns can count once in each. These are not counts of globally different molecules or biological experiments.

### 1. Giving the model the full lead structure produced more useful local candidates

These first comparisons supply the true simulated campaign coefficient. They test generation with the correct response already known.

| Model | Training updates | Correct coefficient: local successes | Reversed coefficient: local successes |
|---|---:|---:|---:|
| Ordinary conditional flow with pooled lead features | 20,000 | 8 | 4 |
| Factorized flow with repaired input normalization | 20,000 | 3 | 3 |
| Full aligned lead with a lead-relative prediction | 20,000 | **96** | **65** |
| Same lead-relative model, original objective, extra training | 30,000 total | **119** | **64** |
| Lead-relative chemistry prior, task inputs fixed | 20,000 | 92 | 92 |

At the same 20,000-update budget, the lead-relative model found 12 times as many correct-context local successes as the pooled conditional model: 96 versus eight. This tests the full lead-relative package—aligned lead tokens, lead-centered initialization, and residual prediction. The experiment does not separate those three changes.

The chemistry prior produces many local candidates but shows no evidence response: 92 versus 92. The trained conditional route changes which local candidates succeed under the campaign response. More training increased correct-context success in every seed-by-family cell. The two 30,000-update seeds separately scored **58/31** and **61/33**, correct/reversed.

Input normalization restored changes in the factorized generator's outputs: opposite coefficients changed 2,084 of 8,192 paired molecular identities. Its local successes stayed at three versus three. Restoring numerical sensitivity alone did not restore useful local steering.

The tested explicit response controller also produced 92/92 local successes at each declared radius, nearly matching the uncontrolled prior. It did not establish useful evidence-dependent local improvement in this run.

### 2. Four observations preserved the useful control with the generator frozen

Validation selected the longer-trained lead-relative model using the original loss. We froze the complete generator and trained a small calibration head on source campaigns. At deployment, this head converts the supplied measurements into the scalar response condition already used by the generator. Its weights and the generator's weights stay fixed for each held-out campaign.

| Available response information | Correct evidence: local successes | Reversed evidence: local successes | Correct-evidence success per attempted slot |
|---|---:|---:|---:|
| True coefficient supplied directly | 119 | 64 | 1.453% |
| Four observations, including the lead | **119** | **66** | **1.453%** |
| 16 observations | 119 | 64 | 1.453% |
| 64 observations | 116 | 67 | 1.416% |

Four observations recover approximately the same aggregate yield as supplying the true coefficient. More observations did not increase the yield on this simple task. At four observations, the two seeds scored **54/31** and **65/35**. All four seed-by-family comparisons favor correct evidence. Calibration and evaluation records verify that the full generator state dictionary stayed unchanged.

The four-observation model supplied 273 eligible distinct candidates across 512 available selection slots (16 per case). Of those, 119 met the joint profile. Candidate supply has improved, but the model still leaves almost half the selection budget unused. The 119/8,192 figure measures distinct success per generation attempt; 119/512 measures success per allocated selection slot. Neither is an observed wet-lab success rate.

The calibration summary's coefficient-accuracy figures use reserved support episodes from source campaigns. The evidence for held-out adaptation is the generated-molecule comparison above; those coefficient-accuracy figures are not target-test assay accuracy.

### 3. What the automated continuation selected

None of the initial models passed every predeclared validation gate. The controller then ran matched extra-training branches with categorical-loss weights zero and 0.1. **The selected branch was weight zero: more training with the original objective.** It passed the required locality and directional-response gates in both seeds.

The added categorical loss remains an incomplete comparison. Seed 142 completed all 10,000 additional updates and scored 60/31, compared with 58/31 without that loss. Seed 143 stopped at 9,356 additional updates when its training timer expired; its 55/34 result is retained as partial. The overnight results do not establish that the added loss helps or hurts consistently.

One recorded validation decision labelled the small-radius controller as lacking two completed evaluations. Both artifacts are now present. Recomputing the declared gates still rejects it, with 29 correct-context validation successes and a failed directional check in family 0. The selected continuation is unchanged by this availability discrepancy.

### 4. What this establishes, and the remaining research question

The study supports **few-shot, experiment-responsive local generation on familiar chemistry**. The useful route is an ordinary conditional flow that retains the full lead structure, followed by a frozen-generator calibration head. The current factorized and explicit-controller routes did not add useful local response control.

The dataset still contains 294 nine-heavy-atom molecules in two families. Responses are exact descriptor-based simulated measurements, with held-out campaign coefficients; molecular structures can recur across splits. Across the 119 successful case-level designs, the known-coefficient and four-observation models each produced **44 globally distinct successful identities; 43 belong to the source library and one lies outside it**. This is evidence for response adaptation, with broad chemical generalization and experimental assay transfer still open. Evidence arriving partway through a trajectory was not tested this night.

The immediate scientific question is now sharper: does the improvement survive on held-out chemical series and against equally trained models with the same full lead representation? That comparison is needed to separate transferable edit learning from familiar-analogue generation and ordinary conditioning.

### Records

- [Study manifest and fixed budget](../../4_experimental_result/result/local_edit_001/study_manifest.json)
- [Actual allocation and completed jobs](../../4_experimental_result/result/local_edit_001/workflow_state.json)
- [Validation-only continuation decision](../../4_experimental_result/result/local_edit_001/continuation_decision.json)
- [Completed source checkpoints used for comparison](../../4_experimental_result/result/local_edit_001/source_completion_decision.json)
- [Independently verified complete-run counts](../../4_experimental_result/result/local_edit_001/analysis/verified_heldout_test_summary.json)
- [Frozen four/16/64-observation test metrics, seed 142](../../4_experimental_result/result/local_edit_001/continuation/recovery_calibration_seed_142/evaluation_test/metrics.csv)
- [Frozen four/16/64-observation test metrics, seed 143](../../4_experimental_result/result/local_edit_001/continuation/recovery_calibration_seed_143/evaluation_test/metrics.csv)
- [Independent completed-result review](../../5_test/results/local_edit_completed_results_review.json)

<!-- REVIEWED_LOCAL_EDIT_RESULTS_END -->

<!-- REVIEWED_STEERING_RESULTS_BEGIN -->
## Reviewed results: evidence reaches prediction, but barely steers the factorized generator

**Recorded: 2026-10-13 01:07:10 PDT.** All six main models and the additional true-context diagnostic completed training and evaluation. The factorized models showed no improvement in local redesign when given correct measurements instead of opposite-campaign measurements. The ordinary conditional model responded to the measurements, but produced few useful local analogues.

There are two observed problems to explain: **weak evidence-to-generation response in the factorized models**, and **poor generation of local analogues across all three architectures**. The results do not yet establish which training or architectural choice caused either problem.

### 1. What we tested

We asked: with the starting molecule, goal, model weights, and sampling noise fixed, does changing only the measured outcomes change the generated chemistry in the correct direction?

The controlled dataset contains **294 small molecules in two local chemical families**, with 161 and 133 molecules. Every molecule has nine heavy atoms. We used 64 source campaigns, eight validation campaigns, and 16 test campaigns. Molecular structures can recur across these splits; the held-out information is the response function. This tests adaptation to new simulated responses on familiar chemistry.

The two endpoints come from exact descriptor-based functions:

$$
y_1(G;s)=s[u(G)-u(G_0)],
\qquad
y_2(G)=-[q(G)-q(G_0)].
$$

Here, $u$ is scaled MolLogP, $q$ is scaled TPSA, and $s$ is a hidden campaign coefficient. Opposite signs of $s$ make opposite chemical changes beneficial. The goal is always $\Delta y_1\geq0.25$ while $\Delta y_2\geq-0.10$. These scores are simulated outcomes, not biological measurements.

We trained an **8-direction factorized flow**, a **32-direction factorized flow**, and an **ordinary conditional flow**, each with seeds 142 and 143. All six used the same initial backbone and completed 20,000 updates. The conditional model approximately matches the parameter count of the 32-direction model. The encoder stayed frozen during source training; all weights stayed frozen during evaluation.

Each test used 4, 16, or 64 measured compounds, including the lead; 256 generation attempts per campaign; and 32 model evaluations per attempt, including the terminal clean prediction. Correct and opposite evidence use the same molecular support identities, lead, goal, and noise. Only their measured outcomes change. Proposals and rankings were committed before the exact evaluator disclosed outcomes.

### 2. Main result: correct evidence does not improve factorized local redesign

A success must be valid, supported, distinct within its campaign/seed case, outside the reserved measurement pool, sufficiently similar to the lead, and satisfy both endpoints. Locality means Morgan similarity of at least 0.4. All attempts, including invalid and repeated proposals, remain in the denominator.

**Every cell below is a count per 8,192 attempted slots**, pooled over two seeds and 16 test campaigns. Each entry is **correct evidence / opposite evidence**.

| Model | 4 measurements | 16 measurements | 64 measurements |
|---|---:|---:|---:|
| 8 directions | 3 / 3 | 3 / 3 | 3 / 3 |
| 32 directions | 7 / 7 | 7 / 7 | 7 / 7 |
| Ordinary conditioning | 7 / 6 | 9 / 6 | 9 / 6 |

At 64 measurements, correct-evidence yields are **0.037%, 0.085%, and 0.110%**, respectively. Ordinary conditioning gains three successes per 8,192 slots, or 0.037 percentage points. Increasing evidence from four to 64 compounds does not improve either factorized model's success count.

The two seeds agree on the absence of factorized steering benefit. At 64 measurements, the separate seed counts are 1/1 and 2/2 for eight directions; 2/2 and 5/5 for 32 directions; and 3/2 and 6/4 for ordinary conditioning. Counts are not globally unique compounds. These are descriptive results from two families and two seeds; paired campaign signs and sampled slots are not independent experiments.

### 3. Where useful candidates disappear

At 64 measurements with correct evidence:

| Model | Attempted slots | Valid, connected, supported slots | Distinct eligible local candidates | Local joint-profile successes |
|---|---:|---:|---:|---:|
| 8 directions | 8,192 | 2,139 | 10 | 3 |
| 32 directions | 8,192 | 2,026 | 18 | 7 |
| Ordinary conditioning | 8,192 | 2,180 | 23 | 9 |

The valid column counts slots; the last two columns remove duplicates and reserved support matches. Only 10–23 eligible local candidates survive across 32 campaign/seed cases. Every test campaign nevertheless has at least one known feasible library molecule outside the reserved support pool.

**The top-16 selection budget was mostly empty.** No case supplied more than three eligible candidates, so ranking selected every eligible candidate. Across 512 allocated selection slots, successes were 3/3, 7/7, and 9/6 for correct/opposite evidence. Selection therefore cannot repair the present shortage of local proposals.

Removing the locality requirement exposes a different result:

| Model, 64 measurements | Distinct joint-profile successes, correct / opposite evidence |
|---|---:|
| 8 directions | 756 / 757 |
| 32 directions | 712 / 712 |
| Ordinary conditioning | 961 / 639 |

These counts still use 8,192 attempted slots and exclude reserved measured compounds. Ordinary conditioning improves the property-only yield from **7.80% to 11.73%**. Thus, evidence can steer property-directed generation in this task; most of that benefit does not survive the requirement to remain near the lead.

### 4. Measurements help response prediction, but the factorized generator barely changes

The shared response scorer predicts whether an unseen library candidate improves or worsens endpoint 1:

| Measured compounds | Correct-evidence sign accuracy | Opposite-evidence sign accuracy |
|---|---:|---:|
| 4 | 94.62% | 5.38% |
| 16 | 94.48% | 5.52% |
| 64 | 93.95% | 6.05% |

Its rank correlation is approximately +0.848 with correct evidence and −0.848 with opposite evidence. These are campaign/seed means from the **shared 8-direction response scorer**. They do not measure the native response predictors of the other two architectures.

At fixed noise and 64 measurements, switching evidence changes the recorded canonical SMILES in:

- **4 of 8,192 slots** for eight directions.
- **0 of 8,192 slots** for 32 directions.
- **4,406 of 8,192 slots (53.78%)** for ordinary conditioning.

This comparison uses recorded molecular identities, including recorded disconnected structures; it does not assert equality of raw atom/bond tensors when both attempts lack a SMILES. Together, prediction accuracy and proposal comparisons show that useful response information is learned, while the factorized generation path barely uses the evidence intervention.

### 5. Separate diagnostic: tracing the older checkpoints

This trace used the **earlier factorized checkpoints and their original nonlinear response tasks**, not the six newly trained models. It covered 24 campaign/seed cases, 16 trajectories per case, and times $t=0.1,0.5,0.9$.

At a fixed noisy state, the architecture gives:

$$
b_{\mathcal D}=H_\psi(\mathcal D),
\qquad
\widehat X=F_{\mathrm{chem}}+B_{\mathrm{chem}}a_\omega(b_{\mathcal D},\ldots),
$$

$$
\Delta\widehat X
=B_{\mathrm{chem}}\!\left(a_{\mathcal D'}-a_{\mathcal D}\right),
\qquad
\frac{\partial\widehat X}{\partial b}=B_{\mathrm{chem}}J_a,
\qquad
\frac{\partial v}{\partial b}=\frac{B_{\mathrm{chem}}J_a}{1-t}.
$$

The measured median changes were:

| Intervention | Evidence-state change, $\|\Delta b\|_2$ | Composed correction change, $\|B\Delta a\|_2$ |
|---|---:|---:|
| Increase observations from 16 to 128 | 1.337 | $1.15\times10^{-9}$ |
| Permute outcomes among the 128 observed molecules | 2.480 | $2.78\times10^{-9}$ |

Each row summarizes 1,152 fixed-state probes. No probed atom/bond category changed. Across full paired rollouts, neither increasing observations nor permuting their outcomes changed any of 384 recorded identities relative to the 16-observation reference.

Across 48 exact composed-Jacobian probes, the median $\|B J_a\|_F$ was **$4.96\times10^{-9}$**. Numerical ranks ranged from two to eight. A nonzero rank therefore coexists with extremely small local sensitivity in these checkpoints.

The norms of $b$ and $X$ use different coordinate systems; their ratio is not a dimensionless attenuation measure. Recomputing the nominally evidence-independent base and basis also produced numerical differences around $4\times10^{-6}$, larger than the composed correction. Small decoder-logit differences cannot all be attributed to evidence. The direct composed sensitivity and unchanged decoded outcomes are the cleaner observations.

### 6. Training facts relevant to the mathematical discussion

The implemented objective combines goal-filtered velocity matching, response likelihood, and reconstruction:

$$
\mathcal L
=
\mathbb E\!\left[
\mathbf 1_{\{\Delta y\in A_g\}}
\frac{\|\widehat X-X\|_{\mathrm{active}}^2}{d_{\mathrm{active}}(1-t)^2}
\right]
+\mathcal L_{\mathrm{response}}
+\mathcal L_{\mathrm{decode}}.
$$

Rejected flow queries remain in the original batch denominator. Every query contributes to the full-Gaussian response likelihood. Decoder reconstruction uses known clean embeddings with small added noise; it does not supervise generated terminal embeddings directly. There is no property-loss gradient through a completed sampling trajectory. The lead enters through pooled atom/bond features and atom count; the sampler begins with noise for the complete molecule.

These facts identify questions for discussion, without choosing a cause:

| Observation | Mathematical question left open |
|---|---|
| Response predictions adapt; factorized proposals barely move | Can the shared evidence representation support prediction while the flow objective learns to ignore it? |
| Older checkpoints have tiny $B J_a$ despite nonzero numerical rank | Is useful sensitivity lost in the basis, selector, their composition, or optimization? |
| Increasing directions from 8 to 32 gives no steering benefit | Does the model fail to use its capacity, rather than lack enough directions? |
| Conditional flow improves properties but supplies few local analogues | Does pooled lead conditioning and whole-molecule denoising provide enough structure for local redesign? |

We have not measured the composed Jacobians of the six new models. The older trace supports a mechanism to investigate; it does not establish the same internal cause in the new runs.

### 7. Completed diagnostic: even the true campaign coefficient does not restore steering

The additional eight-direction model completed 20,000 updates and evaluation with seed 142. It supplies the **true hidden campaign coefficient** directly to the selector, replacing the inferred evidence state in that branch. The response scorer still uses measured evidence.

At each support size—4, 16, and 64—the result was:

| Observable | Correct coefficient | Opposite coefficient |
|---|---:|---:|
| Attempted slots | 4,096 | 4,096 |
| Distinct local joint-profile successes | 5 | 5 |
| Distinct profile-only successes | 454 | 454 |

Switching the supplied coefficient changed **0 of 4,096 recorded canonical identities** at every support size. The coefficient reaches the intended selector input in both training and evaluation; the protocol and paired committed ledgers were checked independently.

**Bypassing evidence inference alone did not restore steering in this setup.** This narrows the immediate question to how the flow learns to use campaign context, realizes continuous corrections, and decodes them. It does not isolate a cause or prove that the evidence encoder needs no improvement. This is one simulation-only diagnostic seed, reported separately from the six main models.

The workflow completed at **2026-10-13 01:05:29 PDT**. All training and evaluation jobs completed without operational failure. Total allocated GPU time, including the operation check, old-model trace, six main runs, and additional diagnostic, was **2.773 GPU-hours**.

The six main test evaluations contain **147,456 logical attempted slots and 73,728 physical generation attempts**. Opposite-evidence comparisons reuse the exact paired campaign proposals, so logical comparisons must not be counted as separate generated samples. The additional diagnostic contributes 24,576 logical test slots and 12,288 physical generation attempts. These test counts exclude validation and operation checks.

### 8. Records for review

- [Executed experiment plan](../detail_operation/03_evidence_steering_experiment_plan.md).
- [Reviewed aggregates and per-campaign/seed counts](../../4_experimental_result/result/evidence_steering_001/analysis/reviewed_results_1013.json).
- [All evaluation metrics](../../4_experimental_result/result/evidence_steering_001/analysis/combined_metrics.csv) and [automated report](../../4_experimental_result/result/evidence_steering_001/analysis/report.md).
- [Pinned study manifest](../../4_experimental_result/result/evidence_steering_001/study_manifest.json) and [common checkpoint selection](../../4_experimental_result/result/evidence_steering_001/checkpoint_selection.json).
- [True-context diagnostic protocol](../../4_experimental_result/result/evidence_steering_001/privileged/seed_142/evaluation_test/protocol.json) and [test metrics](../../4_experimental_result/result/evidence_steering_001/privileged/seed_142/evaluation_test/metrics.csv).
- [Old-checkpoint trace manifest](../../4_experimental_result/result/evidence_steering_001/trace/manifest.json); each case stores stage probes, Jacobians, proposals, and outcomes.
- [GPU operation audit](../../5_test/results/steering_gpu_operation_review.json) and [design review](../../5_test/results/steering_design_review.json).

The counts were recomputed from completed evaluation CSVs. An independent reviewer checked the committed attempt ledgers and paired molecular identities. The live block below records final workflow status. The earlier 00:45 entry is retained as a historical snapshot; its pending diagnostic has now completed.
<!-- REVIEWED_STEERING_RESULTS_END -->

<!-- EVIDENCE_STEERING_OVERNIGHT_BEGIN -->
## Evidence-steering overnight status

Updated: 2026-10-13 23:59:32 PDT
Study phase: **complete**. Tracked jobs: evaluation `375144`, operation `375128`, privileged_evaluation `375152`, privileged_train `375150`, source `375130`, trace `375129`.
Full derived report: [analysis/report.md](../../4_experimental_result/result/evidence_steering_001/analysis/report.md)
Completed test source runs: 6 of 6 declared runs.

## What is being measured

This study changes the measured outcomes while holding the model, lead, goal, and paired sampling draws fixed. `correct` uses the campaign's own measurements; `opposite` replays the paired campaign's proposals under the other campaign's measurements. A positive steering benefit means correct evidence produced a higher local profile yield per attempted slot than opposite evidence.

Yields use every attempted slot as the denominator, including invalid and duplicate proposals. Distinct yields count each successful molecule once; raw-slot yields count a repeated successful molecule in every slot where it appeared. The support curves show local profile yield, with separate labels for distinct and raw-slot counting.
Completed test run paths: `source/conditional_seed_142`, `source/conditional_seed_143`, `source/r32_seed_142`, `source/r32_seed_143`, `source/r8_seed_142`, `source/r8_seed_143`.

The intervention changes measured outcomes while keeping the model, lead, goal, and paired sampling draws fixed. `correct` uses the campaign's own measurements; `opposite` uses the paired campaign's measurement table.

The response scores are simulated values from the study's declared toy response functions, not biological assay measurements. The withheld-response diagnostic uses the shared rank-8 scorer across generator comparisons; it does not measure each model's native response quality. The values below are descriptive means across recorded campaigns and seeds. They make no uncertainty claim across campaign signs.

### Correct versus opposite evidence

Distinct local-profile successes per attempted slot; invalid attempts count in the denominator.

| Split | Model | Measured compounds | Correct evidence yield | Opposite evidence yield |
|---|---|---:|---:|---:|
| test | Known campaign coefficient (diagnostic) | 16 | 0.12% | 0.12% |
| test | Known campaign coefficient (diagnostic) | 4 | 0.12% | 0.12% |
| test | Known campaign coefficient (diagnostic) | 64 | 0.12% | 0.12% |
| test | Ordinary conditioning, seed 142 | 16 | 0.07% | 0.05% |
| test | Ordinary conditioning, seed 142 | 4 | 0.07% | 0.07% |
| test | Ordinary conditioning, seed 142 | 64 | 0.07% | 0.05% |
| test | Ordinary conditioning, seed 143 | 16 | 0.15% | 0.10% |
| test | Ordinary conditioning, seed 143 | 4 | 0.10% | 0.07% |
| test | Ordinary conditioning, seed 143 | 64 | 0.15% | 0.10% |
| test | 32 directions, seed 142 | 16 | 0.05% | 0.05% |
| test | 32 directions, seed 142 | 4 | 0.05% | 0.05% |
| test | 32 directions, seed 142 | 64 | 0.05% | 0.05% |
| test | 32 directions, seed 143 | 16 | 0.12% | 0.12% |
| test | 32 directions, seed 143 | 4 | 0.12% | 0.12% |
| test | 32 directions, seed 143 | 64 | 0.12% | 0.12% |
| test | 8 directions, seed 142 | 16 | 0.02% | 0.02% |
| test | 8 directions, seed 142 | 4 | 0.02% | 0.02% |
| test | 8 directions, seed 142 | 64 | 0.02% | 0.02% |
| test | 8 directions, seed 143 | 16 | 0.05% | 0.05% |
| test | 8 directions, seed 143 | 4 | 0.05% | 0.05% |
| test | 8 directions, seed 143 | 64 | 0.05% | 0.05% |
| validation | Operation check (80 updates) | 4 | 0.00% | 0.00% |
| validation | Known campaign coefficient (diagnostic) | 16 | 0.00% | 0.00% |
| validation | Known campaign coefficient (diagnostic) | 4 | 0.00% | 0.00% |
| validation | Known campaign coefficient (diagnostic) | 64 | 0.00% | 0.00% |
| validation | Ordinary conditioning, seed 142 | 16 | 0.39% | 0.39% |
| validation | Ordinary conditioning, seed 142 | 4 | 0.39% | 0.39% |
| validation | Ordinary conditioning, seed 142 | 64 | 0.39% | 0.39% |
| validation | Ordinary conditioning, seed 143 | 16 | 0.00% | 0.00% |
| validation | Ordinary conditioning, seed 143 | 4 | 0.00% | 0.00% |
| validation | Ordinary conditioning, seed 143 | 64 | 0.00% | 0.00% |
| validation | 32 directions, seed 142 | 16 | 0.00% | 0.00% |
| validation | 32 directions, seed 142 | 4 | 0.00% | 0.00% |
| validation | 32 directions, seed 142 | 64 | 0.00% | 0.00% |
| validation | 32 directions, seed 143 | 16 | 0.00% | 0.00% |
| validation | 32 directions, seed 143 | 4 | 0.00% | 0.00% |
| validation | 32 directions, seed 143 | 64 | 0.00% | 0.00% |
| validation | 8 directions, seed 142 | 16 | 0.20% | 0.20% |
| validation | 8 directions, seed 142 | 4 | 0.20% | 0.20% |
| validation | 8 directions, seed 142 | 64 | 0.20% | 0.20% |
| validation | 8 directions, seed 143 | 16 | 0.00% | 0.00% |
| validation | 8 directions, seed 143 | 4 | 0.00% | 0.00% |
| validation | 8 directions, seed 143 | 64 | 0.00% | 0.00% |

<!-- EVIDENCE_STEERING_OVERNIGHT_END -->


## First completed main comparisons — 2026-10-13T00:45:03.207654-07:00

All six source runs completed 20,000 updates, and all six paired-evidence evaluations completed without job failures. The simulator-only diagnostic that supplies the true campaign coefficient is now training because both rank-8 validation seeds had zero mean steering benefit.

With 64 observed compounds, each architecture has 8,192 attempted slots across two seeds and 16 test campaign cases. The rank-8 model found 3 distinct local joint-profile successes with correct evidence and 3 with opposite evidence. Rank 32 found 7 versus 7. Ordinary conditioning found 9 versus 6. Counts are distinct within each campaign/seed case, not globally unique molecules.

The first results show no measured-evidence advantage for either factorized version. Ordinary conditioning changes the result slightly, with overall yield still low. The ongoing privileged-coefficient diagnostic tests whether directly supplying the response context can produce useful steering. It has no result yet.

The old-checkpoint trace also completed: changing evidence altered the evidence state but produced a median composed correction change around 1e-9; neither tested intervention changed any of 384 paired decoded molecular identities. This diagnoses the earlier checkpoints, separately from the six new models. Completed allocations used 2.421 GPU-hours before the ongoing diagnostic, within the eight-hour cap.

<!-- LOCAL_EDIT_OVERNIGHT_BEGIN -->
## Local-edit overnight update

Updated: 2026-10-13 03:43:01 PDT
Workflow phase: **complete**. Tracked jobs: calibration `375177`, continuation `375172`, control `375170`, control_final `375191`, final `375179`, operation `375156`, operation_attempt_1 `375155`, source `375157`, source_completion `375167`.
Full report: [analysis/report.md](../../4_experimental_result/result/local_edit_001/analysis/report.md)
The report distinguishes the privileged true-coefficient diagnostic from measured evidence: bridge runs use frozen H plus a source-trained scalar head, while calibrated controls use a support least-squares scalar plus a source-fitted surrogate. The chemistry is familiar and the coefficient is held out; scores are simulated and attempted slots are the denominator.
Budget accounting: the previous attempt was allocated/used 2.773 GPU-hours; this study is capped at 8 GPU-hours, for a combined ceiling of 10.773 GPU-hours.
Historical operations: original operation: failed before source experiments (job 375155) — Repeated CUDA lead encodings differed numerically; paired audit now reuses one encoding. Training80 updates completed with frozen encoder.. Setup or operation failures are execution history, not scientific outcome measurements; the recovery run is assessed separately.

Continuation decision

Validation-only selection has no winner (no candidate passed all locality/steering gates). Gate review: anchored_known: 142/0 eligible local yield 0.045; 142/0 valid-slot fraction 0.639; 142/1 eligible local yield 0.049; normalized_known: 142/0 eligible local yield 0.000; 142/0 valid-slot fraction 0.199; 142/1 eligible local yield 0.000; pooled_known: 142/0 eligible local yield 0.006; 142/0 valid-slot fraction 0.121; 142/1 eligible local yield 0.004.
Recovery selection chose terminal supervision weight `0.0` after its validation gates passed; the resulting calibration continuation is recorded separately from the primary selection.

| Run / condition | Split | Seed | Family | N | Attempts | Distinct profile hits | Distinct local-profile hits | Local yield / attempt |
|---|---|---:|---|---:|---:|---:|---:|---:|
| `continuation/decoded_ce_0.0_seed_142` · correct | test | 142 | 0 | 64 | 2048.0 | 129.0 | 24.0 | 0.0117 |
| `continuation/decoded_ce_0.0_seed_142` · opposite | test | 142 | 0 | 64 | 2048.0 | 60.0 | 14.0 | 0.0068 |
| `continuation/decoded_ce_0.0_seed_142` · correct | test | 142 | 1 | 64 | 2048.0 | 170.0 | 34.0 | 0.0166 |
| `continuation/decoded_ce_0.0_seed_142` · opposite | test | 142 | 1 | 64 | 2048.0 | 38.0 | 17.0 | 0.0083 |
| `continuation/decoded_ce_0.0_seed_142` · correct | validation | 142 | 0 | 64 | 512.0 | 46.0 | 9.0 | 0.0176 |
| `continuation/decoded_ce_0.0_seed_142` · opposite | validation | 142 | 0 | 64 | 512.0 | 17.0 | 7.0 | 0.0137 |
| `continuation/decoded_ce_0.0_seed_142` · correct | validation | 142 | 1 | 64 | 512.0 | 62.0 | 18.0 | 0.0352 |
| `continuation/decoded_ce_0.0_seed_142` · opposite | validation | 142 | 1 | 64 | 512.0 | 14.0 | 7.0 | 0.0137 |
| `continuation/decoded_ce_0.0_seed_143` · correct | test | 143 | 0 | 64 | 2048.0 | 178.0 | 21.0 | 0.0103 |
| `continuation/decoded_ce_0.0_seed_143` · opposite | test | 143 | 0 | 64 | 2048.0 | 57.0 | 12.0 | 0.0059 |
| `continuation/decoded_ce_0.0_seed_143` · correct | test | 143 | 1 | 64 | 2048.0 | 243.0 | 40.0 | 0.0195 |
| `continuation/decoded_ce_0.0_seed_143` · opposite | test | 143 | 1 | 64 | 2048.0 | 79.0 | 21.0 | 0.0103 |
| `continuation/decoded_ce_0.0_seed_143` · correct | validation | 143 | 0 | 64 | 512.0 | 52.0 | 8.0 | 0.0156 |
| `continuation/decoded_ce_0.0_seed_143` · opposite | validation | 143 | 0 | 64 | 512.0 | 21.0 | 7.0 | 0.0137 |
| `continuation/decoded_ce_0.0_seed_143` · correct | validation | 143 | 1 | 64 | 512.0 | 78.0 | 13.0 | 0.0254 |
| `continuation/decoded_ce_0.0_seed_143` · opposite | validation | 143 | 1 | 64 | 512.0 | 33.0 | 8.0 | 0.0156 |
| `continuation/decoded_ce_0.1_seed_142` · correct | test | 142 | 0 | 64 | 2048.0 | 185.0 | 22.0 | 0.0107 |
| `continuation/decoded_ce_0.1_seed_142` · opposite | test | 142 | 0 | 64 | 2048.0 | 38.0 | 8.0 | 0.0039 |
| `continuation/decoded_ce_0.1_seed_142` · correct | test | 142 | 1 | 64 | 2048.0 | 198.0 | 38.0 | 0.0186 |
| `continuation/decoded_ce_0.1_seed_142` · opposite | test | 142 | 1 | 64 | 2048.0 | 84.0 | 23.0 | 0.0112 |
| `continuation/decoded_ce_0.1_seed_142` · correct | validation | 142 | 0 | 64 | 512.0 | 67.0 | 4.0 | 0.0078 |
| `continuation/decoded_ce_0.1_seed_142` · opposite | validation | 142 | 0 | 64 | 512.0 | 17.0 | 6.0 | 0.0117 |
| `continuation/decoded_ce_0.1_seed_142` · correct | validation | 142 | 1 | 64 | 512.0 | 75.0 | 18.0 | 0.0352 |
| `continuation/decoded_ce_0.1_seed_142` · opposite | validation | 142 | 1 | 64 | 512.0 | 28.0 | 7.0 | 0.0137 |
| `continuation/decoded_ce_0.1_seed_143` · correct | test | 143 | 0 | 64 | 2048.0 | 214.0 | 21.0 | 0.0103 |
| `continuation/decoded_ce_0.1_seed_143` · opposite | test | 143 | 0 | 64 | 2048.0 | 115.0 | 12.0 | 0.0059 |
| `continuation/decoded_ce_0.1_seed_143` · correct | test | 143 | 1 | 64 | 2048.0 | 374.0 | 34.0 | 0.0166 |
| `continuation/decoded_ce_0.1_seed_143` · opposite | test | 143 | 1 | 64 | 2048.0 | 138.0 | 22.0 | 0.0107 |
| `continuation/decoded_ce_0.1_seed_143` · correct | validation | 143 | 0 | 64 | 512.0 | 79.0 | 9.0 | 0.0176 |
| `continuation/decoded_ce_0.1_seed_143` · opposite | validation | 143 | 0 | 64 | 512.0 | 41.0 | 4.0 | 0.0078 |
| `continuation/decoded_ce_0.1_seed_143` · correct | validation | 143 | 1 | 64 | 512.0 | 120.0 | 20.0 | 0.0391 |
| `continuation/decoded_ce_0.1_seed_143` · opposite | validation | 143 | 1 | 64 | 512.0 | 35.0 | 6.0 | 0.0117 |
| `continuation/recovery_calibration_seed_142` · correct | test | 142 | 0 | 16 | 2048.0 | 132.0 | 24.0 | 0.0117 |
| `continuation/recovery_calibration_seed_142` · opposite | test | 142 | 0 | 16 | 2048.0 | 57.0 | 14.0 | 0.0068 |
| `continuation/recovery_calibration_seed_142` · correct | test | 142 | 0 | 4 | 2048.0 | 130.0 | 23.0 | 0.0112 |
| `continuation/recovery_calibration_seed_142` · opposite | test | 142 | 0 | 4 | 2048.0 | 54.0 | 14.0 | 0.0068 |
| `continuation/recovery_calibration_seed_142` · correct | test | 142 | 0 | 64 | 2048.0 | 130.0 | 24.0 | 0.0117 |
| `continuation/recovery_calibration_seed_142` · opposite | test | 142 | 0 | 64 | 2048.0 | 59.0 | 14.0 | 0.0068 |
| `continuation/recovery_calibration_seed_142` · correct | test | 142 | 1 | 16 | 2048.0 | 169.0 | 33.0 | 0.0161 |
| `continuation/recovery_calibration_seed_142` · opposite | test | 142 | 1 | 16 | 2048.0 | 36.0 | 17.0 | 0.0083 |
| `continuation/recovery_calibration_seed_142` · correct | test | 142 | 1 | 4 | 2048.0 | 164.0 | 31.0 | 0.0151 |
| `continuation/recovery_calibration_seed_142` · opposite | test | 142 | 1 | 4 | 2048.0 | 48.0 | 17.0 | 0.0083 |
| `continuation/recovery_calibration_seed_142` · correct | test | 142 | 1 | 64 | 2048.0 | 166.0 | 33.0 | 0.0161 |
| `continuation/recovery_calibration_seed_142` · opposite | test | 142 | 1 | 64 | 2048.0 | 42.0 | 19.0 | 0.0093 |
| `continuation/recovery_calibration_seed_142` · correct | validation | 142 | 0 | 16 | 512.0 | 46.0 | 9.0 | 0.0176 |
| `continuation/recovery_calibration_seed_142` · opposite | validation | 142 | 0 | 16 | 512.0 | 17.0 | 7.0 | 0.0137 |
| `continuation/recovery_calibration_seed_142` · correct | validation | 142 | 0 | 4 | 512.0 | 46.0 | 9.0 | 0.0176 |
| `continuation/recovery_calibration_seed_142` · opposite | validation | 142 | 0 | 4 | 512.0 | 16.0 | 6.0 | 0.0117 |
| `continuation/recovery_calibration_seed_142` · correct | validation | 142 | 0 | 64 | 512.0 | 47.0 | 9.0 | 0.0176 |
| `continuation/recovery_calibration_seed_142` · opposite | validation | 142 | 0 | 64 | 512.0 | 17.0 | 7.0 | 0.0137 |
| `continuation/recovery_calibration_seed_142` · correct | validation | 142 | 1 | 16 | 512.0 | 61.0 | 18.0 | 0.0352 |
| `continuation/recovery_calibration_seed_142` · opposite | validation | 142 | 1 | 16 | 512.0 | 17.0 | 8.0 | 0.0156 |
| `continuation/recovery_calibration_seed_142` · correct | validation | 142 | 1 | 4 | 512.0 | 63.0 | 17.0 | 0.0332 |
| `continuation/recovery_calibration_seed_142` · opposite | validation | 142 | 1 | 4 | 512.0 | 19.0 | 9.0 | 0.0176 |
| `continuation/recovery_calibration_seed_142` · correct | validation | 142 | 1 | 64 | 512.0 | 61.0 | 18.0 | 0.0352 |
| `continuation/recovery_calibration_seed_142` · opposite | validation | 142 | 1 | 64 | 512.0 | 14.0 | 7.0 | 0.0137 |
| `continuation/recovery_calibration_seed_143` · correct | test | 143 | 0 | 16 | 2048.0 | 185.0 | 20.0 | 0.0098 |
| `continuation/recovery_calibration_seed_143` · opposite | test | 143 | 0 | 16 | 2048.0 | 56.0 | 12.0 | 0.0059 |
| `continuation/recovery_calibration_seed_143` · correct | test | 143 | 0 | 4 | 2048.0 | 181.0 | 21.0 | 0.0103 |
| `continuation/recovery_calibration_seed_143` · opposite | test | 143 | 0 | 4 | 2048.0 | 56.0 | 12.0 | 0.0059 |
| `continuation/recovery_calibration_seed_143` · correct | test | 143 | 0 | 64 | 2048.0 | 178.0 | 19.0 | 0.0093 |
| `continuation/recovery_calibration_seed_143` · opposite | test | 143 | 0 | 64 | 2048.0 | 57.0 | 12.0 | 0.0059 |
| `continuation/recovery_calibration_seed_143` · correct | test | 143 | 1 | 16 | 2048.0 | 244.0 | 42.0 | 0.0205 |
| `continuation/recovery_calibration_seed_143` · opposite | test | 143 | 1 | 16 | 2048.0 | 77.0 | 21.0 | 0.0103 |
| `continuation/recovery_calibration_seed_143` · correct | test | 143 | 1 | 4 | 2048.0 | 249.0 | 44.0 | 0.0215 |
| `continuation/recovery_calibration_seed_143` · opposite | test | 143 | 1 | 4 | 2048.0 | 101.0 | 23.0 | 0.0112 |
| `continuation/recovery_calibration_seed_143` · correct | test | 143 | 1 | 64 | 2048.0 | 247.0 | 40.0 | 0.0195 |
| `continuation/recovery_calibration_seed_143` · opposite | test | 143 | 1 | 64 | 2048.0 | 86.0 | 22.0 | 0.0107 |
| `continuation/recovery_calibration_seed_143` · correct | validation | 143 | 0 | 16 | 512.0 | 53.0 | 8.0 | 0.0156 |
| `continuation/recovery_calibration_seed_143` · opposite | validation | 143 | 0 | 16 | 512.0 | 19.0 | 6.0 | 0.0117 |
| `continuation/recovery_calibration_seed_143` · correct | validation | 143 | 0 | 4 | 512.0 | 56.0 | 8.0 | 0.0156 |
| `continuation/recovery_calibration_seed_143` · opposite | validation | 143 | 0 | 4 | 512.0 | 19.0 | 7.0 | 0.0137 |
| `continuation/recovery_calibration_seed_143` · correct | validation | 143 | 0 | 64 | 512.0 | 51.0 | 8.0 | 0.0156 |
| `continuation/recovery_calibration_seed_143` · opposite | validation | 143 | 0 | 64 | 512.0 | 23.0 | 7.0 | 0.0137 |
| `continuation/recovery_calibration_seed_143` · correct | validation | 143 | 1 | 16 | 512.0 | 76.0 | 13.0 | 0.0254 |
| `continuation/recovery_calibration_seed_143` · opposite | validation | 143 | 1 | 16 | 512.0 | 39.0 | 8.0 | 0.0156 |
| `continuation/recovery_calibration_seed_143` · correct | validation | 143 | 1 | 4 | 512.0 | 71.0 | 13.0 | 0.0254 |
| `continuation/recovery_calibration_seed_143` · opposite | validation | 143 | 1 | 4 | 512.0 | 41.0 | 10.0 | 0.0195 |
| `continuation/recovery_calibration_seed_143` · correct | validation | 143 | 1 | 64 | 512.0 | 79.0 | 13.0 | 0.0254 |
| `continuation/recovery_calibration_seed_143` · opposite | validation | 143 | 1 | 64 | 512.0 | 34.0 | 8.0 | 0.0156 |
| `operation_repeat/anchored_known_seed_142` · correct | validation | 142 | 0 | 4 | 64.0 | 0.0 | 0.0 | 0.0000 |
| `operation_repeat/anchored_known_seed_142` · opposite | validation | 142 | 0 | 4 | 64.0 | 0.0 | 0.0 | 0.0000 |
| `operation_repeat/anchored_known_seed_142` · correct | validation | 142 | 1 | 4 | 64.0 | 4.0 | 3.0 | 0.0469 |
| `operation_repeat/anchored_known_seed_142` · opposite | validation | 142 | 1 | 4 | 64.0 | 3.0 | 3.0 | 0.0469 |
| `source/anchored_known_seed_142` · correct | test | 142 | 0 | 64 | 2048.0 | 109.0 | 14.0 | 0.0068 |
| `source/anchored_known_seed_142` · opposite | test | 142 | 0 | 64 | 2048.0 | 56.0 | 12.0 | 0.0059 |
| `source/anchored_known_seed_142` · correct | test | 142 | 1 | 64 | 2048.0 | 200.0 | 33.0 | 0.0161 |
| `source/anchored_known_seed_142` · opposite | test | 142 | 1 | 64 | 2048.0 | 135.0 | 24.0 | 0.0117 |
| `source/anchored_known_seed_142` · correct | validation | 142 | 0 | 64 | 512.0 | 37.0 | 7.0 | 0.0137 |
| `source/anchored_known_seed_142` · opposite | validation | 142 | 0 | 64 | 512.0 | 18.0 | 6.0 | 0.0117 |
| `source/anchored_known_seed_142` · correct | validation | 142 | 1 | 64 | 512.0 | 61.0 | 9.0 | 0.0176 |
| `source/anchored_known_seed_142` · opposite | validation | 142 | 1 | 64 | 512.0 | 51.0 | 10.0 | 0.0195 |
| `source/anchored_known_seed_143` · correct | test | 143 | 0 | 64 | 2048.0 | 171.0 | 19.0 | 0.0093 |
| `source/anchored_known_seed_143` · opposite | test | 143 | 0 | 64 | 2048.0 | 57.0 | 9.0 | 0.0044 |
| `source/anchored_known_seed_143` · correct | test | 143 | 1 | 64 | 2048.0 | 251.0 | 30.0 | 0.0146 |
| `source/anchored_known_seed_143` · opposite | test | 143 | 1 | 64 | 2048.0 | 105.0 | 20.0 | 0.0098 |
| `source/anchored_known_seed_143` · correct | validation | 143 | 0 | 64 | 512.0 | 54.0 | 8.0 | 0.0156 |
| `source/anchored_known_seed_143` · opposite | validation | 143 | 0 | 64 | 512.0 | 22.0 | 5.0 | 0.0098 |
| `source/anchored_known_seed_143` · correct | validation | 143 | 1 | 64 | 512.0 | 74.0 | 14.0 | 0.0273 |
| `source/anchored_known_seed_143` · opposite | validation | 143 | 1 | 64 | 512.0 | 36.0 | 10.0 | 0.0195 |
| `source/anchored_prior_seed_142` · correct | validation | 142 | 0 | 64 | 512.0 | 16.0 | 4.0 | 0.0078 |
| `source/anchored_prior_seed_142` · opposite | validation | 142 | 0 | 64 | 512.0 | 16.0 | 4.0 | 0.0078 |
| `source/anchored_prior_seed_142` · correct | validation | 142 | 1 | 64 | 512.0 | 37.0 | 8.0 | 0.0156 |
| `source/anchored_prior_seed_142` · opposite | validation | 142 | 1 | 64 | 512.0 | 37.0 | 8.0 | 0.0156 |
| `source/anchored_prior_seed_143` · correct | test | 143 | 0 | 64 | 2048.0 | 61.0 | 18.0 | 0.0088 |
| `source/anchored_prior_seed_143` · opposite | test | 143 | 0 | 64 | 2048.0 | 61.0 | 18.0 | 0.0088 |
| `source/anchored_prior_seed_143` · correct | test | 143 | 1 | 64 | 2048.0 | 107.0 | 33.0 | 0.0161 |
| `source/anchored_prior_seed_143` · opposite | test | 143 | 1 | 64 | 2048.0 | 107.0 | 33.0 | 0.0161 |
| `source/anchored_prior_seed_143` · correct | validation | 143 | 0 | 64 | 512.0 | 9.0 | 4.0 | 0.0078 |
| `source/anchored_prior_seed_143` · opposite | validation | 143 | 0 | 64 | 512.0 | 9.0 | 4.0 | 0.0078 |
| `source/anchored_prior_seed_143` · correct | validation | 143 | 1 | 64 | 512.0 | 38.0 | 12.0 | 0.0234 |
| `source/anchored_prior_seed_143` · opposite | validation | 143 | 1 | 64 | 512.0 | 38.0 | 12.0 | 0.0234 |
| `source/anchored_prior_seed_143/control_0.05` · correct | test | 143 | 0 | 64 | 2048.0 | 61.0 | 18.0 | 0.0088 |
| `source/anchored_prior_seed_143/control_0.05` · opposite | test | 143 | 0 | 64 | 2048.0 | 61.0 | 18.0 | 0.0088 |
| `source/anchored_prior_seed_143/control_0.05` · correct | test | 143 | 1 | 64 | 2048.0 | 107.0 | 33.0 | 0.0161 |
| `source/anchored_prior_seed_143/control_0.05` · opposite | test | 143 | 1 | 64 | 2048.0 | 106.0 | 33.0 | 0.0161 |
| `source/anchored_prior_seed_143/control_0.05` · correct | validation | 143 | 0 | 64 | 512.0 | 9.0 | 4.0 | 0.0078 |
| `source/anchored_prior_seed_143/control_0.05` · opposite | validation | 143 | 0 | 64 | 512.0 | 9.0 | 4.0 | 0.0078 |
| `source/anchored_prior_seed_143/control_0.05` · correct | validation | 143 | 1 | 64 | 512.0 | 39.0 | 12.0 | 0.0234 |
| `source/anchored_prior_seed_143/control_0.05` · opposite | validation | 143 | 1 | 64 | 512.0 | 37.0 | 12.0 | 0.0234 |
| `source/anchored_prior_seed_143/control_0.15` · correct | test | 143 | 0 | 64 | 2048.0 | 61.0 | 18.0 | 0.0088 |
| `source/anchored_prior_seed_143/control_0.15` · opposite | test | 143 | 0 | 64 | 2048.0 | 61.0 | 18.0 | 0.0088 |
| `source/anchored_prior_seed_143/control_0.15` · correct | test | 143 | 1 | 64 | 2048.0 | 107.0 | 33.0 | 0.0161 |
| `source/anchored_prior_seed_143/control_0.15` · opposite | test | 143 | 1 | 64 | 2048.0 | 106.0 | 33.0 | 0.0161 |
| `source/anchored_prior_seed_143/control_0.15` · correct | validation | 143 | 0 | 64 | 512.0 | 9.0 | 4.0 | 0.0078 |
| `source/anchored_prior_seed_143/control_0.15` · opposite | validation | 143 | 0 | 64 | 512.0 | 9.0 | 4.0 | 0.0078 |
| `source/anchored_prior_seed_143/control_0.15` · correct | validation | 143 | 1 | 64 | 512.0 | 39.0 | 12.0 | 0.0234 |
| `source/anchored_prior_seed_143/control_0.15` · opposite | validation | 143 | 1 | 64 | 512.0 | 37.0 | 12.0 | 0.0234 |
| `source/normalized_known_seed_142` · correct | validation | 142 | 0 | 64 | 512.0 | 83.0 | 0.0 | 0.0000 |
| `source/normalized_known_seed_142` · opposite | validation | 142 | 0 | 64 | 512.0 | 65.0 | 1.0 | 0.0020 |
| `source/normalized_known_seed_142` · correct | validation | 142 | 1 | 64 | 512.0 | 78.0 | 1.0 | 0.0020 |
| `source/normalized_known_seed_142` · opposite | validation | 142 | 1 | 64 | 512.0 | 66.0 | 1.0 | 0.0020 |
| `source/normalized_known_seed_143` · correct | test | 143 | 0 | 64 | 2048.0 | 103.0 | 0.0 | 0.0000 |
| `source/normalized_known_seed_143` · opposite | test | 143 | 0 | 64 | 2048.0 | 110.0 | 0.0 | 0.0000 |
| `source/normalized_known_seed_143` · correct | test | 143 | 1 | 64 | 2048.0 | 169.0 | 1.0 | 0.0005 |
| `source/normalized_known_seed_143` · opposite | test | 143 | 1 | 64 | 2048.0 | 137.0 | 1.0 | 0.0005 |
| `source/normalized_known_seed_143` · correct | validation | 143 | 0 | 64 | 512.0 | 29.0 | 0.0 | 0.0000 |
| `source/normalized_known_seed_143` · opposite | validation | 143 | 0 | 64 | 512.0 | 33.0 | 0.0 | 0.0000 |
| `source/normalized_known_seed_143` · correct | validation | 143 | 1 | 64 | 512.0 | 38.0 | 0.0 | 0.0000 |
| `source/normalized_known_seed_143` · opposite | validation | 143 | 1 | 64 | 512.0 | 34.0 | 0.0 | 0.0000 |
| `source/pooled_known_seed_142` · correct | test | 142 | 0 | 64 | 2048.0 | 117.0 | 2.0 | 0.0010 |
| `source/pooled_known_seed_142` · opposite | test | 142 | 0 | 64 | 2048.0 | 71.0 | 0.0 | 0.0000 |
| `source/pooled_known_seed_142` · correct | test | 142 | 1 | 64 | 2048.0 | 135.0 | 1.0 | 0.0005 |
| `source/pooled_known_seed_142` · opposite | test | 142 | 1 | 64 | 2048.0 | 89.0 | 0.0 | 0.0000 |
| `source/pooled_known_seed_142` · correct | validation | 142 | 0 | 64 | 512.0 | 26.0 | 1.0 | 0.0020 |
| `source/pooled_known_seed_142` · opposite | validation | 142 | 0 | 64 | 512.0 | 18.0 | 0.0 | 0.0000 |
| `source/pooled_known_seed_142` · correct | validation | 142 | 1 | 64 | 512.0 | 27.0 | 0.0 | 0.0000 |
| `source/pooled_known_seed_142` · opposite | validation | 142 | 1 | 64 | 512.0 | 23.0 | 0.0 | 0.0000 |
| `source/pooled_known_seed_143` · correct | validation | 143 | 0 | 64 | 512.0 | 51.0 | 0.0 | 0.0000 |
| `source/pooled_known_seed_143` · opposite | validation | 143 | 0 | 64 | 512.0 | 35.0 | 0.0 | 0.0000 |
| `source/pooled_known_seed_143` · correct | validation | 143 | 1 | 64 | 512.0 | 82.0 | 0.0 | 0.0000 |
| `source/pooled_known_seed_143` · opposite | validation | 143 | 1 | 64 | 512.0 | 43.0 | 0.0 | 0.0000 |
| `source_completion/anchored_prior_seed_142` · correct | test | 142 | 0 | 64 | 2048.0 | 34.0 | 13.0 | 0.0063 |
| `source_completion/anchored_prior_seed_142` · opposite | test | 142 | 0 | 64 | 2048.0 | 34.0 | 13.0 | 0.0063 |
| `source_completion/anchored_prior_seed_142` · correct | test | 142 | 1 | 64 | 2048.0 | 79.0 | 28.0 | 0.0137 |
| `source_completion/anchored_prior_seed_142` · opposite | test | 142 | 1 | 64 | 2048.0 | 79.0 | 28.0 | 0.0137 |
| `source_completion/anchored_prior_seed_142` · correct | validation | 142 | 0 | 64 | 512.0 | 8.0 | 4.0 | 0.0078 |
| `source_completion/anchored_prior_seed_142` · opposite | validation | 142 | 0 | 64 | 512.0 | 8.0 | 4.0 | 0.0078 |
| `source_completion/anchored_prior_seed_142` · correct | validation | 142 | 1 | 64 | 512.0 | 32.0 | 9.0 | 0.0176 |
| `source_completion/anchored_prior_seed_142` · opposite | validation | 142 | 1 | 64 | 512.0 | 32.0 | 9.0 | 0.0176 |
| `source_completion/anchored_prior_seed_142/control_0.05` · correct | test | 142 | 0 | 64 | 2048.0 | 34.0 | 13.0 | 0.0063 |
| `source_completion/anchored_prior_seed_142/control_0.05` · opposite | test | 142 | 0 | 64 | 2048.0 | 34.0 | 13.0 | 0.0063 |
| `source_completion/anchored_prior_seed_142/control_0.05` · correct | test | 142 | 1 | 64 | 2048.0 | 79.0 | 28.0 | 0.0137 |
| `source_completion/anchored_prior_seed_142/control_0.05` · opposite | test | 142 | 1 | 64 | 2048.0 | 79.0 | 28.0 | 0.0137 |
| `source_completion/anchored_prior_seed_142/control_0.05` · correct | validation | 142 | 0 | 64 | 512.0 | 8.0 | 4.0 | 0.0078 |
| `source_completion/anchored_prior_seed_142/control_0.05` · opposite | validation | 142 | 0 | 64 | 512.0 | 8.0 | 4.0 | 0.0078 |
| `source_completion/anchored_prior_seed_142/control_0.05` · correct | validation | 142 | 1 | 64 | 512.0 | 32.0 | 9.0 | 0.0176 |
| `source_completion/anchored_prior_seed_142/control_0.05` · opposite | validation | 142 | 1 | 64 | 512.0 | 32.0 | 9.0 | 0.0176 |
| `source_completion/anchored_prior_seed_142/control_0.15` · correct | test | 142 | 0 | 64 | 2048.0 | 34.0 | 13.0 | 0.0063 |
| `source_completion/anchored_prior_seed_142/control_0.15` · opposite | test | 142 | 0 | 64 | 2048.0 | 34.0 | 13.0 | 0.0063 |
| `source_completion/anchored_prior_seed_142/control_0.15` · correct | test | 142 | 1 | 64 | 2048.0 | 79.0 | 28.0 | 0.0137 |
| `source_completion/anchored_prior_seed_142/control_0.15` · opposite | test | 142 | 1 | 64 | 2048.0 | 79.0 | 28.0 | 0.0137 |
| `source_completion/anchored_prior_seed_142/control_0.15` · correct | validation | 142 | 0 | 64 | 512.0 | 8.0 | 4.0 | 0.0078 |
| `source_completion/anchored_prior_seed_142/control_0.15` · opposite | validation | 142 | 0 | 64 | 512.0 | 8.0 | 4.0 | 0.0078 |
| `source_completion/anchored_prior_seed_142/control_0.15` · correct | validation | 142 | 1 | 64 | 512.0 | 32.0 | 9.0 | 0.0176 |
| `source_completion/anchored_prior_seed_142/control_0.15` · opposite | validation | 142 | 1 | 64 | 512.0 | 32.0 | 9.0 | 0.0176 |
| `source_completion/normalized_known_seed_142` · correct | test | 142 | 0 | 64 | 2048.0 | 101.0 | 1.0 | 0.0005 |
| `source_completion/normalized_known_seed_142` · opposite | test | 142 | 0 | 64 | 2048.0 | 120.0 | 1.0 | 0.0005 |
| `source_completion/normalized_known_seed_142` · correct | test | 142 | 1 | 64 | 2048.0 | 136.0 | 1.0 | 0.0005 |
| `source_completion/normalized_known_seed_142` · opposite | test | 142 | 1 | 64 | 2048.0 | 103.0 | 1.0 | 0.0005 |
| `source_completion/normalized_known_seed_142` · correct | validation | 142 | 0 | 64 | 512.0 | 22.0 | 0.0 | 0.0000 |
| `source_completion/normalized_known_seed_142` · opposite | validation | 142 | 0 | 64 | 512.0 | 16.0 | 0.0 | 0.0000 |
| `source_completion/normalized_known_seed_142` · correct | validation | 142 | 1 | 64 | 512.0 | 30.0 | 0.0 | 0.0000 |
| `source_completion/normalized_known_seed_142` · opposite | validation | 142 | 1 | 64 | 512.0 | 28.0 | 0.0 | 0.0000 |
| `source_completion/pooled_known_seed_143` · correct | test | 143 | 0 | 64 | 2048.0 | 191.0 | 1.0 | 0.0005 |
| `source_completion/pooled_known_seed_143` · opposite | test | 143 | 0 | 64 | 2048.0 | 173.0 | 0.0 | 0.0000 |
| `source_completion/pooled_known_seed_143` · correct | test | 143 | 1 | 64 | 2048.0 | 258.0 | 4.0 | 0.0020 |
| `source_completion/pooled_known_seed_143` · opposite | test | 143 | 1 | 64 | 2048.0 | 146.0 | 4.0 | 0.0020 |
| `source_completion/pooled_known_seed_143` · correct | validation | 143 | 0 | 64 | 512.0 | 56.0 | 0.0 | 0.0000 |
| `source_completion/pooled_known_seed_143` · opposite | validation | 143 | 0 | 64 | 512.0 | 44.0 | 0.0 | 0.0000 |
| `source_completion/pooled_known_seed_143` · correct | validation | 143 | 1 | 64 | 512.0 | 55.0 | 0.0 | 0.0000 |
| `source_completion/pooled_known_seed_143` · opposite | validation | 143 | 1 | 64 | 512.0 | 42.0 | 0.0 | 0.0000 |

<!-- LOCAL_EDIT_OVERNIGHT_END -->

<!-- masked-control-001 -->

## Masked edits and terminal control — overnight study

Updated 2026-10-14T09:52:02.883982+00:00. Coordinator state: **completed_with_recorded_failures**.

We compare lead-relative generation (A), hard-masked graph generation (B), and their combination (C), then test known-response control and calibration from 4/8/16 observations. Outcomes are exact simulated two-property responses. The inherited initializer predates the new molecular split.

[Live report](../../4_experimental_result/result/masked_control_001/live_report.md) · [Experiment plan](../detail_operation/05_masked_edit_control_experiment_plan.md) · [Run records](../../4_experimental_result/result/masked_control_001/)

Validation selected representation **A**. Selection uses distinct valid protected nonlead molecules per all attempts, averaged over three seeds.

| Representation | Mean distinct local yield |
|---|---:|
| A | 0.2165 |
| B | 0.2155 |
| C | 0.2100 |

| Seed | Method | Measurements (0 = known coefficients) | Correct evidence: distinct joint successes | Opposite evidence | Correct top-16 successes / slots |
|---|---|---:|---:|---:|---:|
| 142 | conditional_flow | 0 | 210 | 139 | 150 / 256 |
| 142 | terminal_control | 0 | 199 | 118 | 148 / 256 |
| 142 | unguided_compute_pool | 0 | 571 | 571 | 181 / 256 |
| 142 | unguided_rerank | 0 | 174 | 174 | 127 / 256 |
| 142 | conditional_flow | 4 | 184 | 152 | 124 / 256 |
| 142 | terminal_control | 4 | 191 | 113 | 134 / 256 |
| 142 | unguided_compute_pool | 4 | 571 | 571 | 162 / 256 |
| 142 | unguided_rerank | 4 | 174 | 174 | 124 / 256 |
| 142 | conditional_flow | 8 | 205 | 148 | 144 / 256 |
| 142 | terminal_control | 8 | 192 | 111 | 143 / 256 |
| 142 | unguided_compute_pool | 8 | 571 | 571 | 181 / 256 |
| 142 | unguided_rerank | 8 | 174 | 174 | 127 / 256 |
| 142 | conditional_flow | 16 | 208 | 144 | 146 / 256 |
| 142 | terminal_control | 16 | 195 | 111 | 146 / 256 |
| 142 | unguided_compute_pool | 16 | 571 | 571 | 180 / 256 |
| 142 | unguided_rerank | 16 | 174 | 174 | 127 / 256 |
| 143 | conditional_flow | 0 | 196 | 144 | 151 / 256 |
| 143 | terminal_control | 0 | 198 | 135 | 143 / 256 |
| 143 | unguided_compute_pool | 0 | 621 | 621 | 187 / 256 |
| 143 | unguided_rerank | 0 | 180 | 180 | 132 / 256 |
| 143 | conditional_flow | 4 | 165 | 147 | 121 / 256 |
| 143 | terminal_control | 4 | 187 | 153 | 125 / 256 |
| 143 | unguided_compute_pool | 4 | 621 | 621 | 172 / 256 |
| 143 | unguided_rerank | 4 | 180 | 180 | 125 / 256 |
| 143 | conditional_flow | 8 | 181 | 142 | 139 / 256 |
| 143 | terminal_control | 8 | 194 | 136 | 142 / 256 |
| 143 | unguided_compute_pool | 8 | 621 | 621 | 190 / 256 |
| 143 | unguided_rerank | 8 | 180 | 180 | 132 / 256 |
| 143 | conditional_flow | 16 | 189 | 140 | 145 / 256 |
| 143 | terminal_control | 16 | 197 | 132 | 144 / 256 |
| 143 | unguided_compute_pool | 16 | 621 | 621 | 190 / 256 |
| 143 | unguided_rerank | 16 | 180 | 180 | 132 / 256 |
| 144 | conditional_flow | 0 | 188 | 120 | 134 / 256 |
| 144 | terminal_control | 0 | 190 | 134 | 149 / 256 |
| 144 | unguided_compute_pool | 0 | 522 | 522 | 173 / 256 |
| 144 | unguided_rerank | 0 | 174 | 174 | 137 / 256 |
| 144 | conditional_flow | 4 | 173 | 132 | 117 / 256 |
| 144 | terminal_control | 4 | 185 | 142 | 138 / 256 |
| 144 | unguided_compute_pool | 4 | 522 | 522 | 155 / 256 |
| 144 | unguided_rerank | 4 | 174 | 174 | 133 / 256 |
| 144 | conditional_flow | 8 | 191 | 127 | 141 / 256 |
| 144 | terminal_control | 8 | 187 | 141 | 147 / 256 |
| 144 | unguided_compute_pool | 8 | 522 | 522 | 174 / 256 |
| 144 | unguided_rerank | 8 | 174 | 174 | 138 / 256 |
| 144 | conditional_flow | 16 | 190 | 127 | 137 / 256 |
| 144 | terminal_control | 16 | 190 | 141 | 150 / 256 |
| 144 | unguided_compute_pool | 16 | 522 | 522 | 174 / 256 |
| 144 | unguided_rerank | 16 | 174 | 174 | 137 / 256 |

Distinct successes in this table are summed across campaign cases. See each summary for globally unique successful identities. Unused top-16 slots remain in the denominator. These figures must be read with the forward/backward computation and invalid-attempt counts in the CSV files.

Revision decision: Run nested-evidence revision: validation control improves joint successes over opposite evidence and unguided generation in all three seeds.

Recorded failures: CalledProcessError(1, ['squeue', '-h', '-j', '375676', '-o', '%i|%T'])

The exact-oracle search is a privileged diagnostic with a declared terminal-query budget. The learned controller uses a source-trained terminal surrogate. Finite-library and direct-mutation baselines use the declared descriptor basis and support-only calibration. No result here establishes drug-discovery performance.

<!-- /masked-control-001 -->
