# Experiment results — 1014

<!-- REVISION_EFFICIENCY_RESULTS_1014_BEGIN -->
## Updated experiment: masking, cost, and revision

Snapshot: **2026-10-14 12:10:57 PDT**. Study: `revision_efficiency_001`.

**The main finding is that reusing a trajectory handles additional measurements much better than reversing the design goal.** Hard masking preserves the protected graph exactly. The guided generator still loses chemical validity, and the large sample-and-rank pool still achieves higher selection success.

The static B comparison is complete on held-out test campaigns. The revision numbers below are complete **validation** results. Three final revision test jobs are still running; their matched-time results are pending. This section is the latest snapshot. Earlier sections below describe the previous experiment at their stated times.

### Completed test: hard-masked generation B

Each method uses 16 measured compounds, 16 campaigns and three model seeds. Each campaign/seed case requests 16 candidates, for 768 requested slots per method. A success must satisfy both simulated property thresholds, preserve protected graph categories, and pass the candidate eligibility rules. Unfilled slots count as failures. Repeated identities across campaigns or seeds are separate decisions, not new molecular discoveries.

| Method | Successful requested slots | Valid connected attempts | Recorded generation/evaluation time, summed across cases |
|---|---:|---:|---:|
| Sample and rank, 64 proposals | 369/768 (48.0%) | 95.4% | 37.6 s |
| Ordinary conditional flow, 64 proposals | 398/768 (51.8%) | 95.6% | 37.5 s |
| Terminal control, 64 proposals | 475/768 (61.8%) | 72.2% | 738.1 s |
| Sample and rank, 1,600 proposals | 560/768 (72.9%) | 95.4% | 940.6 s |

All four methods preserve protected categories on **100% of attempts**. The mask therefore does its structural job. It does not enforce valid chemistry in the editable region: control has **72.2% validity**, compared with **95.4%** for the large unguided pool.

The B-based controller reaches **61.8% selection success**, compared with **57.3%** in the earlier A-based experiment. Its validity falls from **81.2% to 72.2%**. This comparison includes the separately fitted B response surrogate; it does not isolate masking as the cause of either change.

Control takes about **19.6 times** the time of small-pool sampling. The large pool takes **27.4% more time than control** and achieves **72.9% success**, versus **61.8%**. These are unequal-cost comparisons. The final revision test measures the full success-versus-time curve.

Evidence still changes useful proposals: terminal control produces **660 distinct successful identities summed within campaign/seed cases under correct evidence**, versus **303 under reversed first-endpoint evidence**, at the same proposal count. This is separate from changing the requested goal during a trajectory.

### Completed validation: continuation versus restart

We tested two updates at approximately 25%, 50% and 75% of generation. The first adds measurements of the same response function. The second keeps the measurements fixed but reverses the requested change in endpoint 1. Endpoint 2 and the protected structure remain constraints.

The table shows **ordinary conditional generation at the midpoint**, using 1,600 proposals per case for both continuation and restart. This inexpensive baseline can reuse a trajectory without the iterative terminal controller.

| Update | Continue: successful requested slots | Restart: successful requested slots | Reduction in total charged time from continuing |
|---|---:|---:|---:|
| Measurements increase from 4 to 16; goal stays fixed | 608/768 (79.2%) | 618/768 (80.5%) | 27.6% |
| Requested direction of endpoint 1 reverses; measurements stay fixed | 121/768 (15.8%) | 229/768 (29.8%) | 27.6% |

For additional measurements, continuing takes **1,001.5 s** in total versus **1,383.1 s** for restarting. Time charged after the update is **548.7 s versus 930.4 s**, a **41.0% reduction**. Success falls by only 10 slots out of 768. This is descriptive validation evidence, not a demonstrated statistical equivalence.

For a reversed goal, continuing takes **999.5 s** versus **1,380.8 s**, but loses **108 successful slots**. Revealing the reversal earlier helps: continuation succeeds in **151/768 slots at 25%**, versus **121/768 at 50% and 75%**. Restart succeeds in **229/768**. Conditional validity remains about 95%, so this loss concerns achieving the revised objective even when the decoded graph is valid.

The expensive controller shows the same direction of change. At the midpoint it gives **393/768 versus 409/768** successes after additional measurements, and **100/768 versus 121/768** after goal reversal, for continuation versus restart. Its validation proposal cap is 32 per case, compared with 1,600 for the conditional baseline. Cross-method percentages at these caps are not matched-budget comparisons.

The reversed task also has fewer known feasible examples in the small reference library. The post-hoc validation audit finds at least one eligible positive-goal example in 13 of 16 campaigns and a negative-goal example in 8 of 16. This audit covers only the compatible validation library and does not prove that other campaigns have no solution. Continue and restart are compared on the same campaigns and final goals; no cases are removed based on success.

### Interpretation and next decision

The current evidence supports useful reuse for a modest evidence update. A conflicting objective exposes a larger loss. This suggests a question about how far an intermediate state has committed to an earlier objective; it does not yet identify the dynamical cause.

The next decision should use the final paired success-versus-time curves. If the reversal gap persists, test whether a small rollback or a flow trained for redirection can recover it. Ordinary conditional continuation must remain a principal comparator: it already provides most of the simple-update reuse benefit.

These experiments use small chemical families, a known descriptor basis, and synthetic noisy response functions. They test held-out campaign responses within that setting. Transfer to unfamiliar chemical series and real assays remains untested.

### Running work and measurements

Three test jobs (`375871_0`, `375871_1`, `375871_2`) are active. Completed campaigns reported by their logs at this snapshot: 11/16, 11/16, 11/16. The common time budgets span **0.0625 to 32 seconds per case**, selected from validation timing alone. Final results will include both total strategy cost and cost after the update. Timing includes decoding and host chemistry; invalid attempts remain counted.

- [Static B raw measurements](../../4_experimental_result/result/revision_efficiency_001/static_b/).
- [Revision validation measurements](../../4_experimental_result/result/revision_efficiency_001/validation/).
- [Machine-readable snapshot and source hashes](../../4_experimental_result/result/revision_efficiency_001/analysis/running_results_1014.json).
- [Live report](../../4_experimental_result/result/revision_efficiency_001/live_report.md).
- [Locked time-budget selection](../../4_experimental_result/result/revision_efficiency_001/budget_selection.json).

<!-- REVISION_EFFICIENCY_RESULTS_1014_END -->

<!-- MASKED_CONTROL_FINAL_1014_BEGIN -->
## Final overnight result: held-out campaigns and revision

Updated 2026-10-14 09:14:43 PDT. **All planned stages completed at 02:51 PDT on October 14**, using **9.63 allocated GPU hours**. No experiment jobs are running. The recorded coordinator error is the earlier scheduler-polling issue; recovery completed the remaining stages.

**Result:** sparse evidence can steer this continuous molecular generator on held-out simulated campaigns. The strongest selection result comes from generating a larger pool and ranking it. Continuing a partially generated trajectory after new evidence preserves approximately the same success as restarting while reducing charged computation.

### Held-out design success

Each entry is the number of successful candidates selected across 16 campaigns and three model seeds, divided by 768 requested selection slots. Every campaign requests 16 candidates. Invalid candidates and unfilled slots do not count as successes. A candidate must meet both property thresholds, preserve protected graph categories and lie outside the common support reserve.

| Measured compounds | Sample and rank, 64 proposals/case | Terminal control, 64 proposals/case | Conditional flow, 64 proposals/case | Larger sample-and-rank pool, 1,600 proposals/case |
|---:|---:|---:|---:|---:|
| 4 | 382/768 (49.7%) | 397/768 (51.7%) | 362/768 (47.1%) | 489/768 (63.7%) |
| 8 | 397/768 (51.7%) | 432/768 (56.2%) | 424/768 (55.2%) | 545/768 (71.0%) |
| 16 | 396/768 (51.6%) | 440/768 (57.3%) | 428/768 (55.7%) | 544/768 (70.8%) |

With 16 measurements, terminal control selects **440/768 successful candidates (57.3%)**, compared with **396/768 (51.6%)** from the small unguided pool. The larger pool reaches **544/768 (70.8%)**. All methods use the same campaign evidence for scoring or conditioning. The larger pool uses 25 times as many proposals; its advantage must be read alongside its computation.

The 16-measurement controller produces **582 distinct successful identities summed within campaign/seed cases**, versus **384 under reversed first-endpoint evidence**, and **528** from the small unguided pool. Successes can repeat across campaigns and seeds; these totals are not counts of newly discovered molecules.

When coefficients are supplied directly, control gives 440/768 selection successes and ordinary conditioning gives 435/768. The earlier validation ordering changes slightly on test: these two methods have similar test selection success. Conditional generation remains far cheaper at deployment. It received 10,000 additional source updates; that cost should be included when comparing adaptation strategies.

### Validity and computation

| Method, 16 measurements | Valid connected attempts | Recorded generation/evaluation time across seeds |
|---|---:|---:|
| Small pool | 90.8% | 36.1 s |
| Terminal control | 81.2% | 706.7 s |
| Conditional flow | 92.4% | 36.1 s |
| Larger pool | 90.8% | 903.1 s |

Control takes about **19.6 times** the recorded time of small-pool sampling and lowers validity to **81.2%**, from **90.8%**. The larger pool takes about **28% more recorded time than control** and achieves higher selection success. This is an explicit budget comparison with different costs, rather than an exactly matched-compute experiment.

On test, A/B/C again have similar distinct local yields: 21.65%, 21.55% and 21.00%. Hard masking preserves protected categories exactly; masked graph generation B has the highest prior validity, 94.4%. The selected A model has no hard mask on its base flow, so a few controlled/conditional attempts violate protected categories and are excluded from successes.

### Revision after evidence increases from 4 to 16 measurements

The revision study uses 16 proposals per campaign, giving 768 attempted slots across three seeds. Each row below is a separate evidence-revelation time; the same restart reference is reused. Costs include the old evidence optimization and the prefix already executed.

| Evidence arrives near generation progress | Continue: distinct campaign-specific successes | Restart: successes | Fewer charged forward/backward traversals when continuing |
|---:|---:|---:|---:|
| 25% | 195 | 194 | 12.9% |
| 50% | 193 | 194 | 24.8% |
| 75% | 191 | 194 | 37.3% |

Continuing gives **195, 193 and 191 successes**, while restart gives **194** at each comparison time. The useful observation is similar final success with **13–37% fewer charged traversals**, with larger savings for later updates. These are descriptive counts; the small success differences do not establish superiority. All evidence from the beginning also gives 194 successes and is cheaper, as it avoids the earlier evidence round.

The restart/all-evidence candidates are cached and reused across timing comparisons. Reports distinguish actual executed work from the charged cost each strategy would incur. The comparisons trace a success–compute tradeoff; they do not use identical computation budgets.

### Simple alternatives and exact-response diagnostic

The direct masked-mutation baseline, using exact structural descriptors and support-only response calibration, selects **140/256 successes (54.7%)** at 16 measurements in one deterministic baseline run. It proposes up to 256 candidates per campaign. The flow-control mean is 57.3% from 64 proposals per campaign across three seeds. The different proposal budgets and descriptor access should remain explicit. The simple baseline is competitive and requires no learned chemical generator.

With 16 exact response calls per case, the bounded velocity-search diagnostic succeeds in **80/192 cases**, compared with **171/192** for independent sampling plus exact ranking. The current small PCA-basis search does not earn its evaluation budget.

### What to take forward

**Evidence responsiveness is established for this controlled setting. Efficient constrained redesign remains the research question.** The next architectural work should make useful edits cheaper and chemically valid, with a larger sample-and-rank pool as the main comparator. Revision is the clearest positive result for retaining a continuous state: updated evidence can be used without discarding an already computed prefix.

The study uses two small chemical families, known descriptor features and synthetic noisy responses. The original encoder/flow initializer predates this molecular split. Results establish transfer to held-out campaign response functions within this setting; they do not yet establish biological assay transfer or generation on unfamiliar chemical series.

### Final evidence

- [Held-out control and candidate-pool results](../../4_experimental_result/result/masked_control_001/control_test/).
- [Revision measurements and charged costs](../../4_experimental_result/result/masked_control_001/revision/).
- [Simple baselines](../../4_experimental_result/result/masked_control_001/baselines/simple_baselines_test/).
- [Exact-response search](../../4_experimental_result/result/masked_control_001/oracle/).
- [Final coordinator status](../../4_experimental_result/result/masked_control_001/status.json).

<!-- MASKED_CONTROL_FINAL_1014_END -->


<!-- MASKED_CONTROL_REVIEW_1014_BEGIN -->
## Masked edits and experimental steering: current result

**Snapshot: 2026-10-14 01:13:31 PDT.** Study launched October 13 evening. Completed results below are from validation campaigns. Held-out test and revision stages are continuing after coordinator recovery.

**Main finding:** the model can now change decoded molecular proposals in response to experimental evidence. Terminal control improves joint-property success over unguided sampling in all three seeds. Ordinary conditional generation gives the strongest known-response result, while control costs much more and produces more invalid molecules.

### What we tested

We want to improve one molecular property while preserving another and keeping changes inside a permitted part of the lead. We compared lead-relative generation (A), hard-masked graph generation (B), and masked lead-relative generation (C). Each model was trained for 20,000 updates with three seeds. Three conditional models then received 10,000 additional source-training updates.

The library has 294 nine-heavy-atom molecules from two families: 182 source, 56 validation and 56 test. Each campaign has a different simulated response function built from molecular lipophilicity, polar surface area and their interaction. Measurements include noise. A successful candidate improves endpoint 1 by at least 0.25 and lets endpoint 2 fall by at most 0.1, in the simulator's units. Sixteen validation campaigns and three training seeds are used for the main comparisons.

The evidence enters through a response map estimated from molecule–measurement pairs. In this study the map uses a known three-feature descriptor basis. The chemical generator stays frozen during support calibration and control.

### 1. Local generation works; masking gives a preservation guarantee

Each seed generated 256 candidates per mask, across four masks. Invalid and repeated attempts remain in the denominator. “Distinct local yield” counts valid, protected, nonlead identities within each mask.

| Representation | Valid connected attempts | Protected categories unchanged | Distinct local yield | Exact target reconstruction |
|---|---:|---:|---:|---:|
| A — lead-relative | 90.5% | 99.8% | 21.65% | 100% |
| B — masked graph | 94.4% | 100.0% | 21.55% | 100% |
| C — masked lead-relative | 91.7% | 100.0% | 21.00% | 100% |

A was selected by the predeclared validation-yield rule. Its margin over B is about 0.10 percentage points, smaller than the variation across seeds. The result supports similar diversity for these three implementations. B gives higher chemical validity and B/C enforce exact protected-category copying. Reconstruction is measured on known target embeddings; generated validity is measured separately.

### 2. A known response can steer generation

We supplied the true campaign response to isolate control from evidence inference. Each method made 64 proposals per campaign and seed: 3,072 attempts per evidence orientation. “Opposite” reverses the first endpoint response while retaining the preservation endpoint. Initial noise, lead and requested goal are paired.

| Method | Correct response: distinct joint successes | Opposite response: distinct joint successes | Correct response: successful top-16 slots |
|---|---:|---:|---:|
| unguided rerank | 510 | 510 | 392 / 768 (51.0%) |
| terminal control | 563 | 378 | 431 / 768 (56.1%) |
| conditional flow | 613 | 430 | 480 / 768 (62.5%) |

Counts of distinct successes are summed within each campaign and seed. The same molecular identity can therefore contribute in several campaigns or seeds. They are not counts of newly invented molecules. All successes exclude the common maximal support reserve. The top-16 denominator includes unfilled slots.

Terminal control adds 53 campaign-specific distinct successes over the unguided pool: 563 versus 510, a 10.4% increase. Its selected success fraction rises from 51.0% to 56.1%. Ordinary conditioning reaches 62.5% and also uses fewer generation evaluations. It received extra source-training updates, so training and deployment costs must both enter the final comparison.

Changing the response changes decoded identities in 1,241 of 3,072 paired controlled proposals (40.4%). Ordinary conditioning changes 2,038 (66.3%). The unguided pool changes zero identities; its evidence affects ranking alone. This establishes a functioning evidence-to-generation route in the current prototype.

### 3. Sixteen noisy measurements recover most of the known-response effect

| Seed | Control: correct evidence | Control: opposite evidence | Unguided pool: correct evidence | Control top-16 successes |
|---|---:|---:|---:|---:|
| 142 | 197 | 107 | 171 | 143 / 256 |
| 143 | 191 | 128 | 179 | 149 / 256 |
| 144 | 173 | 141 | 160 | 139 / 256 |

Across seeds, 16-measurement control gives 561 distinct campaign-specific successes, compared with 563 when the response coefficients are known. It fills 431 successful selection slots out of 768 (56.1%); unguided ranking gives 391 (50.9%). Correct evidence beats both opposite evidence and unguided generation in every seed, so the predeclared revision gate passes.

Opposite evidence changes 1,209 of 3,072 paired controlled identities (39.4%). The main generator results at 4 and 8 measurements are pending in the held-out evaluation. The measured response basis makes this a controlled few-shot calibration result.

### 4. Control has a clear validity and compute cost

With correct known response, unguided generation produces 90.8% valid connected attempts. Terminal control produces 78.3%; ordinary conditioning produces 92.5%. The permitted edit mask controls structural scope, but it does not enforce boundary valence or general chemical validity.

Recorded generation-and-evaluation time is about 713 seconds for correct-response control versus 36.5 seconds for the unguided comparison: roughly 19.6 times longer. Control uses 510,720 forward evaluations plus 336,384 backward traversals, versus 49,152 forward evaluations for unguided sampling. These are sums across the three seeds and include host chemistry work.

The source-trained terminal predictors rank decoded descriptors well, but their held-out generated-structure errors remain nonzero. The gradient-direction audit has zero informative perturbation pairs, so it does not yet validate which local gradients lead to useful discrete edits. The next comparison must ask whether the extra control work beats a larger unguided candidate pool under comparable computation.

### 5. Budgeted exact-response search loses to fresh sampling

The separate oracle diagnostic searches two time-bin velocity controls in an eight-dimensional source-edit basis. Each case gets 16 exact response calls. Its comparator spends the same 16 calls on independent unguided candidates and exact ranking.

| Seed | Successful search cases: velocity search | Successful search cases: fresh sampling and ranking |
|---|---:|---:|
| 142 | 29 / 64 | 58 / 64 |
| 143 | 28 / 64 | 57 / 64 |
| 144 | 29 / 64 | 59 / 64 |

The totals are 86/192 successful cases for velocity search and 174/192 for sampling. Each method uses 1,024 terminal calls per seed. This particular small, bounded search is inefficient even when exact responses are available. Its failure does not locate the best possible controlled trajectory; it shows that this search procedure does not earn its query budget.

### Status and continuation

All nine source models, three conditional models, five predictor fits, nine control settings, three oracle diagnostics and three 16-measurement validation runs completed. Scheduler accounting reports 5.28 allocated GPU hours through these stages.

The coordinator stopped when Slurm removed a completed job from its queue and returned “Invalid job id.” A second reporting defect recursively included the previous live JSON report, causing excessive report growth and delaying coordination. Both defects are corrected; eight recovery checks pass. Original results, the failed coordinator record and the previous derived report remain archived.

The coordinator has resumed the previously authorized plan. The held-out control/oracle/baseline array is job **375713**, currently queued for GPU resources. Remaining work includes test campaigns at 4/8/16 observations, the larger unguided pool, direct constrained mutations and finite-library selection, test representation evaluation, and evidence revision versus restarting. Current results support validation steering; claims about held-out performance and revision await those outputs.

### What this changes in our discussion

The next mathematical question should connect **useful endpoint steering, chemically valid edits, and computation**. The experiments now establish evidence responsiveness, but a freely moving latent intervention can damage validity and a simple conditional flow can outperform it. Hard masking solves structural preservation. The central unresolved problem is choosing interventions that retain chemical plausibility and deliver more useful designs for their cost.

These experiments use familiar small-molecule families and synthetic responses. The inherited initializer predates the source/validation/test split. The results support a mechanistic study of local adaptation; biological assays and transfer across new chemical families remain later stages.

### Evidence files

- [Experiment plan](../detail_operation/05_masked_edit_control_experiment_plan.md).
- [Representation selection](../../4_experimental_result/result/masked_control_001/representation_selection.json).
- [Known-response control selection and seed records](../../4_experimental_result/result/masked_control_001/control_selection.json).
- [Sixteen-measurement validation records](../../4_experimental_result/result/masked_control_001/fewshot_validation/).
- [Matched-query oracle comparisons](../../4_experimental_result/result/masked_control_001/oracle/).
- [Coordinator recovery record](../../4_experimental_result/result/masked_control_001/poll_recovery_1014.json).
- [Live report](../../4_experimental_result/result/masked_control_001/live_report.md).
- [Recovery regression tests](../../5_test/results/masked_reporting_recovery_1014.xml).

<!-- MASKED_CONTROL_REVIEW_1014_END -->



<!-- EVIDENCE_STEERING_OVERNIGHT_BEGIN -->
## Evidence-steering overnight status

Updated: 2026-10-14 23:59:42 PDT
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

<!-- REVISION_EFFICIENCY_001_BEGIN -->
## Revision efficiency — next-stage study

Updated 2026-10-14T19:39:43.680580+00:00. Stage: **complete_with_recorded_failures**.

This study uses the saved hard-masked B generator. It compares terminal control, ordinary conditional generation, expanded sampling and masked mutation. Revision tests distinguish more measurements from a reversed improvement request, with the second endpoint retained.

[Plan](../detail_operation/06_revision_efficiency_experiment_plan.md) · [Live measurements](../../4_experimental_result/result/revision_efficiency_001/live_report.md) · [Task and scheduler records](../../4_experimental_result/result/revision_efficiency_001/)

Allocated GPU hours recorded at the last stage boundary: 12.58. A running stage is not a completed result.

Recorded operational failures: critic_refinement: incomplete tasks ['critic_B_seed142_refined', 'critic_B_seed143_refined', 'critic_B_seed144_refined']

<!-- REVISION_EFFICIENCY_001_END -->
