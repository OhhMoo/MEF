# Implementation log — 1012

Actual review/recovery date: 12 October 2026, America/Los_Angeles. Original training and comparisons ran on 11 October.

## Evaluation repair

At 2026-10-12T19:08:32.110507+00:00, all six source models had completed training. Fifteen of 24 evaluation jobs completed; nine stopped when separately computed lead embeddings failed an exact-equality check. These interrupted jobs are missing results, rather than zero scientific successes.

The runner now computes a lead embedding once per case. Every generator and the common scorer receives that same tensor; each keeps its own intended evidence state. The copied-selector control uses the adapted evidence state while the common response predictor retains native evidence. The strict checks for actual encoder, normalization, baseline, goal, context, and lead-feature mismatches remain in place.

Six independent regression checks passed in 4.47 seconds. The reviewer also verified that sampling/evaluation code remains byte-identical to the original snapshot. Checkpoints and scientific settings are unchanged.

The recovery uses a fresh pinned source snapshot and fresh output directories. Original failed logs and partial proposal commitments remain preserved. One allocated-GPU probe will check native factorized, native conditional, and copied-selector routes before the nine evaluations start. At most four evaluation allocations run together.

Prior actual GPU use: 3.631 hours. Recovery allocation caps: 2.417 GPU-hours. Even full use stays below the original 12-hour allowance.

[Recovery manifest](../../4_experimental_result/result/calibrated_redesign_001/recovery_manifest_20261012.json), [independent review](../../5_test/results/redesign_shared_lead_review.json), [regressions](../../5_test/results/redesign_shared_lead_cpu.xml).


## Recovery complete — 2026-10-12T19:22:56.759443+00:00

Probe372620 and all nine tasks in evaluation array372621 completed successfully. The full study now has all24 planned comparisons and768controlled evaluation batches. No GPU job remains queued or running. Actual allocated GPU use is 3.966 hours.

The independent recovery audit passed: all36 pinned files verified, only the evaluation function changed, original trained-model/scientific-protocol hashes preserved, and96,384 audited attempts/36,517 oracle calls reconciled. It independently recomputed847 outcomes/probabilities and768 terminal graphs. [Full audit](../../5_test/results/redesign_recovery_operation_review.json).

[Completed plain-language results](experiment_results_1012.md) supersede the pre-recovery status above. The live recorder continues to maintain the cumulative1011 tables and plots without modifying manual1012 conclusions.


## Evidence steering overnight implementation — 2026-10-12T23:49:25.631516-07:00

The user authorized the reviewed steering plan for overnight execution. GPT-6-sol built the local-family dataset and paired-evidence evaluator; GPT-6-luna built the recorder; GPT-6-astra reviewed the mathematics, input isolation, and composed-Jacobian trace. The coordinator integrated the training runner, source snapshots, common-update checkpoint selection, and Slurm budget controls.

The experiment uses two nine-heavy-atom families (161 and 133 molecules), source-scaled MolLogP/TPSA simulated outcomes, and opposite-sign hidden response coefficients. Each family has 32 source, four validation, and eight held-out test campaigns. Goals are fixed at improvement 0.25 and maximum second-endpoint loss 0.10. The initial N/O-count task-construction pilot was discarded and preserved; the final continuous descriptor definitions use fresh held-out coefficient values.

Main models: factorized rank 8, factorized rank 32, ordinary conditioning matched to the rank-32 parameter budget; seeds 142 and 143. Support sizes 4/16/64, 256 attempts per held-out condition, 32 flow evaluations including the terminal clean prediction. Opposite-evidence conditions replay exact paired proposals; logical comparison counts and physical compute are separate.

The runner preserves the original source objective by default. New explicit dataset/objective injections enable this study and a separately labelled privileged coefficient diagnostic. The diagnostic changes only flow conditioning; response prediction still uses inferred evidence. All main deployment weights remain frozen.

Integrated CPU regression run: 36 passed; final independent review/report run: 9 passed, including additional common-checkpoint and privileged-state checks. The allocated-GPU operation is the next execution stage.

[Plan](../detail_operation/03_evidence_steering_experiment_plan.md), [independent review](../../5_test/results/steering_design_review.json), [study artifacts](../../4_experimental_result/result/evidence_steering_001/).


### Evidence steering study launched — 2026-10-12T23:54:31.590953-07:00

The allocated GPU operation completed successfully in job 375128. It trained for 80 updates and exercised 256 logical evaluation slots using 128 physical samples. An independent audit recomputed all 76 disclosed outcome rows, verified paired replay and retained failures, and confirmed all 113 encoder/normalizer tensors remained unchanged. This is an operation check, not a scientific result.

Training array 375130 contains six source runs; its first three tasks are running and the second three are queued. Trace job 375129 is running alongside training. The tmux session is **emf_steering_20261012**. The controller selects a common training update count before evaluation and runs the privileged-context diagnostic only if both rank-8 validation seeds show no positive mean steering benefit. Initial committed wall limits total 7.167 GPU-hours; the overall cap is eight.

[Live results](../../4_experimental_result/result/evidence_steering_001/analysis/report.md), [workflow state](../../4_experimental_result/result/evidence_steering_001/workflow_state.json), [GPU audit](../../5_test/results/steering_gpu_operation_review.json). Withheld-response diagnostics refer to the shared rank-8 scorer used across generators.
