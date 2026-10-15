# Experiment results — 1012

Actual dates: source training and the initial comparisons ran on 11 October 2026, America/Los_Angeles. Nine repaired evaluations completed on 12 October. This summary uses exact simulated responses. Updated: 2026-10-12T19:22:56.759443+00:00.

## Main result

The current shared-direction flow has not demonstrated a useful local-redesign advantage. All flow methods found zero selected molecules that met the complete design request. Predictor-guided local search found some. More observations did not improve the observed success.

All six source models completed training. All 24 planned method/seed comparisons are now complete. The study contains 768 evaluation batches and 768,000 attempted candidates: eight methods, three source seeds, eight test campaigns, one lead per campaign, and four evidence sizes.

## Success among 16 testing slots

| Method | 16 observations | 32 observations | 64 observations | 128 observations |
|---|---:|---:|---:|---:|
| Proposed shared-direction flow | 0.00% | 0.00% | 0.00% | 0.00% |
| Ordinary conditional flow | 0.00% | 0.00% | 0.00% | 0.00% |
| Flow with adapted selector | 0.00% | 0.00% | 0.00% | 0.00% |
| Flow with swapped evidence | 0.00% | 0.00% | 0.00% | 0.00% |
| Flow with directions disabled | 0.00% | 0.00% | 0.00% | 0.00% |
| Standard-generator proposals | 0.00% | 0.00% | 0.00% | 0.00% |
| Flow with fitted task state | 0.00% | 0.00% | 0.00% | 0.00% |
| Predictor-guided local search | 9.11% | 9.11% | 8.59% | 8.33% |

Each percentage uses all 16 allocated testing slots. A success must be a distinct, valid, unmeasured molecule that meets both property targets and stays within the declared structural neighbourhood of the starting molecule. An empty slot counts as a failure. Each table cell averages 24 campaign/seed cases, with 384 testing slots.

With 16 observations, search found 35 successes in 384 slots, about 1.46 successful molecules per 16 slots. With 128 observations, it found 32. Its 95% interval was 2.08%–17.19% at 16 observations and 1.30%–16.67% at 128. These intervals resample campaigns and source seeds. The observed change does not support a benefit from additional evidence in this test.

## What fails

The flow can produce molecules that meet the property targets. The proposed model produced about 58.5 such distinct molecules per 1,000 attempts, before requiring proximity to the lead. The ordinary conditional model produced about 65.8–67.0. None of their local candidates met the complete profile.

The practical gap is local redesign. At 16 observations, the direction-disabled, swapped-evidence and fitted-state controls each produced only 11 eligible local candidates across 24,000 attempts. Standard proposals produced none. Search produced 637 eligible candidates and filled 362 of 384 testing slots. Generating property-profile hits away from the lead does not satisfy this redesign task.

These results do not support an advantage from the shared-direction factorization at the tested model size and training settings. They point to local chemical proposal generation as the next implementation issue to examine. The high-level question—whether experimental evidence can recalibrate reusable chemical changes—remains open.

## Effect of additional evidence

A paired replay compared 16 and 128 observations using the same trained model, lead, goal, counts and Gaussian draws. Across 24,000 attempts, only 11 decoded atom/bond graphs changed (0.046%). Among 11,163 pairs that both decoded to valid connected molecules, only one canonical molecular identity changed. The effect is small, rather than exactly zero.

The proposed model filled only 44 of 1,536 allocated selection slots over its 96 batches and found zero full-profile successes. Ordinary conditioning filled 62 slots; selector adaptation filled 44; neither found a success. Search filled 1,448 slots and found 135 successes over its four evidence sizes.

These counts explain the flat success curves: additional observations scarcely changed the proposed chemistry, and the flow provided too few eligible local candidates. [Paired evidence analysis](../../4_experimental_result/result/calibrated_redesign_001/analysis/evidence_sensitivity.json).

## Completion and verification

Nine initial evaluations stopped on a runner check comparing independently recomputed lead embeddings. The fix computes one lead encoding per case and reuses it across generators, the scorer and support sizes. Strict mismatch checks remain intact. The repair changes no trained weights, goals, evidence, count/noise seeds, candidate budget or outcome functions.

Six regression checks and a new allocated-GPU operation test passed. The independent reviewer verified all recovery snapshot hashes and original scientific protocols, reconciled six probe batches plus the first 96 full recovery batches, and independently recomputed 847 exact outcomes/probabilities. Original failed logs and partial proposal commitments remain preserved; recovered results use separate output directories.

The source models have approximately 5.36 million parameters. Four source runs reached 20,000 updates; the two seed-43 runs reached 17,858 and 17,884 under the 30-minute training cap. Frozen encoders and normalization stayed unchanged. Total allocated GPU time, including failures, pilot, operation and recovery, was 3.966 hours, within the original 12-hour allowance.

Full-pool exact-oracle evaluations are extra diagnostic costs, separate from the N+16 compound budget. The completed controlled study made 325,737 full-pool oracle calls.

[Case measurements](../../4_experimental_result/result/calibrated_redesign_001/analysis/case_metrics.csv), [support curves](../../4_experimental_result/result/calibrated_redesign_001/analysis/support_curves.csv), [paired comparisons](../../4_experimental_result/result/calibrated_redesign_001/analysis/paired_curves.csv), [success figure](../../4_experimental_result/result/calibrated_redesign_001/analysis/support_curves_test.png), [recovery audit](../../5_test/results/redesign_recovery_operation_review.json).


## What the first experiment tested — diagnosis at 2026-10-12T21:57:38.136951+00:00

This was a controlled computational lead-redesign task. Molecular graphs came from QM9, restricted to neutral C/N/O/F molecules with at most nine heavy atoms. Their two outcomes were computer-defined scores: hidden nonlinear functions of 16 graph-derived descriptors. They were not measured potency or developability endpoints. Thirty-two response functions supplied source training; eight supplied validation and eight supplied test cases.

For each held-out test function, one measured lead and 16/32/64/128 campaign observations were supplied. The request was to improve score one by at least 0.25 source-standardized units and lose no more than 0.1 units of score two. Eligibility also required fingerprint similarity at least 0.4, at most two heavy atoms of count difference, valid supported chemistry, and a distinct candidate outside the full 128-compound reserved support pool. Each method attempted 1,000 candidates and could fill 16 testing slots.

### Observed bottleneck

The proposed model generated property-only hits, but supplied only 44 eligible candidate slots over 96 evaluation batches, where 1,536 testing slots were allocated. All 44 tested local candidates missed the joint property profile. The final 0% includes the many unfilled slots. This identifies local candidate supply as a failure before comparing experimental recalibration. Increasing observations from 16 to 128 changed only 11 decoded graphs across 24,000 paired attempts and one supported valid canonical identity among 11,163 pairs valid in both conditions.

### Prototype and training choices

The sampler starts from Gaussian noise. The lead enters the denoiser as averaged node/pair embeddings plus its size—a 97-dimensional summary. It does not supply the complete lead graph to the denoiser or explicitly preserve a scaffold. Local analogue generation must be learned from those summaries and source pairs. The search control starts from the actual lead and proposes atom/bond edits. Both use the same response scorer and declared proposal/testing budgets, but search has explicit local proposal structure.

The factorization is implemented as a correction to complete-molecule denoising. A usable repertoire of lead-relative chemical edits has not been demonstrated. Lead summarization and weak task-dependent steering are plausible contributors; this study does not isolate their exact causal contributions.

Source queries use up to the four nearest compatible analogues. Training goals draw improvement uniformly from [-1,1] and allowable second-score loss uniformly from [0,1]. Evaluation requests positive improvement 0.25 with loss limited to 0.1. Half of training requests permit a decrease in score one; 90% allow greater loss in score two than evaluation.

A read-only expectation over source campaigns/leads/pairs finds that14.66% of source pairs satisfy the fixed evaluation request. Under the actual goal prior,42.29% of query draws receive flow supervision, while0.204% receive feasible supervision for requests at least as strict on both endpoints as evaluation. This is about1308 instances in a 20,000-update, 32-query run. This measures training coverage; it does not prove a gradient-level cause or exclude useful learning from easier requests.

### Feasibility and interpretation

A read-only check of the 13,017 held-out library compounds found only1–11 unmeasured eligible local candidates per test lead after reserving support. Only one of eight leads had a known library candidate meeting the joint target. The library does not enumerate novel supported graphs. Search actually found selected successes in 5 of eight campaigns at 16 observations, so the target was attainable in at least those cases.

The experiment is a genuine negative result for the current complete pipeline. It combines local analogue generation, response inference for unfamiliar functions, unfamiliar molecule identities, and joint-target control. With local generation and evidence-driven proposal changes already weak, it does not cleanly isolate the main transfer hypothesis. The first corrective experiment should establish usable local changes and target-relevant supervision before testing campaign recalibration.

[Read-only feasibility/training-coverage audit](../../4_experimental_result/result/calibrated_redesign_001/analysis/feasibility_and_training_coverage.json). No models, original requests, rankings, or measured results were changed by this diagnosis.


## New evidence steering study — preparation at 2026-10-12T23:49:25.631516-07:00

The next experiment asks whether the same frozen generator changes its chemical preference when only the observed effects of chemistry reverse. It uses familiar local molecules with new simulated response functions. This isolates evidence-driven steering before testing unfamiliar chemistry.

Two chemical families contain 161 and 133 molecules. All 16 declared held-out campaign cases have a known feasible unmeasured analogue outside their fixed 64-compound support pool; coverage ranges from 1 to 20 such analogues. The fixed request is used during training and evaluation. Average source feasibility is about 10.1% of local analogues.

No training or evaluation result is available yet for this study. The recorder will place completed model comparisons, exact outcome counts, and figures in [the live study report](../../4_experimental_result/result/evidence_steering_001/analysis/report.md). This new task does not change the earlier study's zero-success result.


### Evidence steering study launched — 2026-10-12T23:54:31.590953-07:00

The allocated GPU operation completed successfully in job 375128. It trained for 80 updates and exercised 256 logical evaluation slots using 128 physical samples. An independent audit recomputed all 76 disclosed outcome rows, verified paired replay and retained failures, and confirmed all 113 encoder/normalizer tensors remained unchanged. This is an operation check, not a scientific result.

Training array 375130 contains six source runs; its first three tasks are running and the second three are queued. Trace job 375129 is running alongside training. The tmux session is **emf_steering_20261012**. The controller selects a common training update count before evaluation and runs the privileged-context diagnostic only if both rank-8 validation seeds show no positive mean steering benefit. Initial committed wall limits total 7.167 GPU-hours; the overall cap is eight.

[Live results](../../4_experimental_result/result/evidence_steering_001/analysis/report.md), [workflow state](../../4_experimental_result/result/evidence_steering_001/workflow_state.json), [GPU audit](../../5_test/results/steering_gpu_operation_review.json). Withheld-response diagnostics refer to the shared rank-8 scorer used across generators.

<!-- EVIDENCE_STEERING_OVERNIGHT_BEGIN -->
## Evidence-steering overnight status

Updated: 2026-10-12 23:59:21 PDT
Study phase: **source training and existing-checkpoint trace**. Tracked jobs: operation `375128`, source `375130`, trace `375129`.
Full derived report: [analysis/report.md](../../4_experimental_result/result/evidence_steering_001/analysis/report.md)
Completed test source runs: 0 of 6 declared runs.

## What is being measured

This study changes the measured outcomes while holding the model, lead, goal, and paired sampling draws fixed. `correct` uses the campaign's own measurements; `opposite` replays the paired campaign's proposals under the other campaign's measurements. A positive steering benefit means correct evidence produced a higher local profile yield per attempted slot than opposite evidence.

Yields use every attempted slot as the denominator, including invalid and duplicate proposals. Distinct yields count each successful molecule once; raw-slot yields count a repeated successful molecule in every slot where it appeared. The support curves show local profile yield, with separate labels for distinct and raw-slot counting.
No completed test-model evaluation artifacts are recorded yet; missing evaluations remain pending.

The intervention changes measured outcomes while keeping the model, lead, goal, and paired sampling draws fixed. `correct` uses the campaign's own measurements; `opposite` uses the paired campaign's measurement table.

The response scores are simulated values from the study's declared toy response functions, not biological assay measurements. The withheld-response diagnostic uses the shared rank-8 scorer across generator comparisons; it does not measure each model's native response quality. The values below are descriptive means across recorded campaigns and seeds. They make no uncertainty claim across campaign signs.

### Correct versus opposite evidence

Distinct local-profile successes per attempted slot; invalid attempts count in the denominator.

| Split | Model | Measured compounds | Correct evidence yield | Opposite evidence yield |
|---|---|---:|---:|---:|
| validation | Operation check (80 updates) | 4 | 0.00% | 0.00% |

<!-- EVIDENCE_STEERING_OVERNIGHT_END -->
