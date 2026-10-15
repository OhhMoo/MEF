# Implementation log — 1013

<!-- SELECTOR_SATURATION_REVIEW_BEGIN -->
## Selector activation and gradient diagnosis

Recorded: 2026-10-13T08:46:06.012583+00:00. Read-only CPU analysis; float32.

The completed factorized models have a nearly inactive context selector. Large pooled chemical features feed the selector without normalization. Its second hidden layer is predominantly below −10, where SiLU passes little signal or gradient. The learned correction is consequently close to a context-independent final bias. This gives a specific optimization mechanism to test through matched retraining.

The audit covers both chemical families, times 0.1/0.5/0.9, and two kinds of state: source interpolation and the actual 32-evaluation sampling trajectory. Each checkpoint has 12 fixed-state probes. Opposite outcomes or opposite true coefficients are evaluated at the same states. A separate counterfactual normalizes each pooled node/pair vector to RMS one while keeping all learned weights fixed.

| Checkpoint | Units below −10, median | Hidden-selector flow-gradient norm | Composed Jacobian norm | Jacobian after input normalization |
|---|---:|---:|---:|---:|
| Evidence r8, seed 142 | 99.2% | 0.000495 | 9.85e-05 | 0.111 |
| Evidence r8, seed 143 | 99.2% | 0.00023 | 9.24e-05 | 0.0954 |
| Known coefficient, seed 142 | 100.0% | 2.68e-06 | 1.71e-06 | 0.099 |

The known-coefficient model is the strongest example: its median final selector weight gradient is 6.48×10⁻⁶, while the final bias gradient is 0.145. Its median node/pair pooled RMS values are 26.2/10.1. Normalizing these inputs brings the hidden-selector flow gradient from 2.68×10⁻⁶ to 0.197 in this fixed-weight counterfactual. Gradients use one feasible source query and the existing active-coordinate velocity loss; only the flow objective contributes.

The zero-initialized final selector head is connected to learning. At the shared initializer, the random nonzero basis produces a median final-weight gradient of 3.16. Upstream selector and basis gradients are initially zero because the final head is zero; they can begin learning after that head updates. The minimal repair therefore changes pooled-input normalization alone and preserves initialization, data draws, and loss weights.

This establishes saturation in the inspected trained checkpoints. The fixed-weight normalization intervention measures restored continuous sensitivity; generation quality and the training cause require the planned matched original-versus-normalized experiment. Full aligned lead tokens and residual generation address the separate locality problem.

Artifacts: [probe JSON](../../5_test/results/selector_saturation_review.json), [reproducible audit](../../5_test/review_selector_saturation.py).
<!-- SELECTOR_SATURATION_REVIEW_END -->

<!-- LOCAL_EDIT_IMPLEMENTATION_BEGIN -->
## Local molecular edits: overnight continuation

Recorded: 2026-10-13T02:27:44.250952-07:00. Date label: 1013. The actual source jobs run on October 13, 2026.

**Question:** can a generator make useful local molecular changes when it already knows the campaign response? If it can, can a small calibration head recover that control from four, 16, or 64 measured examples while keeping the generator fixed?

The model now receives the lead's full aligned atom and bond representations. It predicts a change relative to that lead. This tests the locality problem separately from the old factorized selector's weak response. A second arm changes only the selector's pooled-input scale. A third arm applies a source-trained response controller to a lead-relative chemistry prior. All primary outcomes remain decoded molecular validity, locality, and the exact joint simulated property goal.

The scientific plan is [04_local_edit_continuation_plan.md](../detail_operation/04_local_edit_continuation_plan.md). The pinned study record is [study_manifest.json](../../4_experimental_result/result/local_edit_001/study_manifest.json). Eight source runs cover four model configurations and two seeds. Each run targets 20,000 updates. Validation alone selects either frozen calibration or a matched 10,000-update repair with categorical-loss weights zero and 0.1. Final held-out tests follow that recorded decision. Evidence-timing tests wait until local steering and calibration work.

Parallel roles: Astra reviewed the scientific design, selector saturation, and integrated execution. Two 6-sol agents implemented the local model/calibration and the aligned data/explicit controller. Luna implemented the live results report. The coordinator integrated the runner, owns GPU submission, and checks operational results. Each worker had separate code ownership.

Checks before launch: 22 integrated CPU checks passed; a separate independent review passed seven checks; ten compatibility checks passed. Reports are in [5_test/results](../../5_test/results). The full CPU runner completed training, paired trace, and committed evaluation. The allocated-GPU operation additionally exercises normalized-selector gradients, source-only surrogate fitting, frozen calibration, retained invalid proposals, and oracle access after proposal commitment.

The first GPU job, 375155, completed 80 training updates with the encoder unchanged. It then stopped at a strict paired-input assertion: repeated CUDA reductions produced slightly different lead encodings. The correction reuses one encoded lead across both contexts in the trace and across paired generation calls. The original code, manifest, accounting, and failure are preserved under the first snapshot and [revisions/attempt_1](../../4_experimental_result/result/local_edit_001/revisions/attempt_1). Revision 2 was pinned before repeat job 375156. This was an execution defect; it supplies no evidence for or against local steering.

The tmux session is `emf_local_edit_20261013`, with controller, results, and progress windows. Code and inputs are hashed. The controller permits at most four concurrent L40S GPUs. The new study's maximum requested allocation is eight GPU-hours, including both operation attempts. Two controller-validation caps were reduced from 20 to 15 minutes to pay for the repeat. The prior study's actual 2.773 GPU-hours remains separately recorded; the combined ceiling is 10.773 GPU-hours.

The live scientific record is [experiment_results_1013.md](experiment_results_1013.md). Detailed outputs, decisions, checkpoints, and logs are stored in [local_edit_001](../../4_experimental_result/result/local_edit_001).
<!-- LOCAL_EDIT_IMPLEMENTATION_END -->

### Launch verification and completion safeguard

Recorded: 2026-10-13T02:38:15.417381-07:00.

Repeat GPU operation 375156 passed on an NVIDIA L40S. All three supplemental deployment routes completed: known-response control, measured-support least-squares control, and a frozen-generator calibration bridge. The normalized selector had finite nonzero gradients while encoder and normalizer weights stayed unchanged. The reviewer checked proposal hashes, exact oracle timing, and the fixed generator fingerprint. The report is [local_edit_gpu_operation.json](../../5_test/results/local_edit_gpu_operation.json); independent post-launch review is [local_edit_postlaunch_review.json](../../5_test/results/local_edit_postlaunch_review.json).

Source array **375157** contains eight jobs. Four started in parallel, with finite training losses; the remaining four wait for those slots. Source jobs execute snapshot version 2. There are no scientific outcome conclusions from these early training losses.

Measured throughput projected the normalized and chemistry-prior models slightly beyond their internal 1,560-second training deadline. Version 3 adds a completion stage for at most four such runs, three GPU minutes each. It restores the complete optimizer and random-number state into a new output directory, then finishes the original 20,000-update schedule. A focused nine-test CPU suite passed, including exact model/optimizer/RNG preservation and rejection of learning-rate changes. The original partial checkpoints and evaluations remain immutable. Only full 20,000-update runs enter model selection.

The source jobs continue without a restart. The controller was explicitly handed over to version 3, which adopts the already-submitted array. It uses version 3 for completion and later stages. The [manifest](../../4_experimental_result/result/local_edit_001/study_manifest.json) retains both versions' hashes and declares the stage-to-snapshot mapping. Final evaluation caps were reduced from five to four minutes to fund completion; the worst-case total remains eight GPU-hours, including the original failed operation.

The validation feasibility review found 69–97 distinct primary-local library candidates outside each reserved support pool. Each of the eight campaigns has at least one known feasible candidate; counts are 1, 7, 3, 11, 14, 2, 14, and 6. These bound success within the finite known library, not the set of all novel molecules. The 1% gate concerns distinct local candidates, not 1% joint-profile success in every campaign. See [validation_feasibility.json](../../4_experimental_result/result/local_edit_001/analysis/validation_feasibility.json).

## Masked edits and terminal control — overnight launch

Actual launch: 2026-10-13 evening PDT (2026-10-14 UTC). Study: `masked_control_001`. This section adds a new experiment; earlier 1013 records remain intact.

The new implementation compares three approximately 5.21-million-parameter models: A predicts lead-relative edits with full graph freedom; B flows over masked absolute graph coordinates; C combines lead-relative prediction with the hard mask. All retain the complete aligned lead. B/C copy protected atom and bond categories at decoding, while independent RDKit checks count boundary-invalid outputs.

The data contain 182 source, 56 validation and 56 test molecules from the existing two-family library. Campaign responses now have two independently varying rows over three descriptor features, with noisy observations and nested support sizes 4/8/16. The inherited encoder/flow initializer predates this split. Masks are fixed from the lead, and campaign feasibility is conditioned only on source chemistry.

The controller differentiates through the full terminal rollout, solves a local two-constraint repair, and accepts a correction only after rerunning its predicted consequences. A source-trained graph critic replaces the old pooled surrogate. Separate exact-response search, direct mutation, finite-library selection, ordinary conditional generation, and a larger unguided candidate pool expose simpler explanations for any improvement. The exact-response arm has a declared oracle budget and is reported separately.

The team used Astra for design and independent review, 6-sol for model/data/controller implementation, and Luna for reporting. Review found and corrected duplicate support leads, an inconsistent delta threshold, oracle goal metadata, global-success counting, and a floating-point time-bin boundary mismatch. Protected-coordinate tests, full-rollout finite differences, conditional training, critic fitting, oracle selection, and continuation tests pass. Independent allocated-L40S checks passed for A/B/C, frozen weights, terminal gradients, and control-call accounting.

GPU smoke jobs: 375561, 375563, 375564. Independent GPU review: 375565. Full source array: **375566**, nine runs, three seeds per variant, 20,000 updates each, at most eight simultaneous L40S GPUs. Source code/input hashes are in the study manifest; follow-up integration is pinned separately. The coordinator runs in tmux `emf_masked_control_20261013` with `coordinator` and `progress` windows.

Follow-ups select the representation and control settings on validation only. A poor source-generated critic audit triggers one larger source-only fit. Every failed job and failed scientific arm remains recorded. Sparse calibration is run even if steering fails. Revision runs only after decoded validation gains over opposite evidence and unguided generation in all three seeds. All intermediate optimization rollouts, backward traversals, invalid outputs, support exclusions, and host chemistry time are recorded.

[Plan](../detail_operation/05_masked_edit_control_experiment_plan.md) · [Live report](../../4_experimental_result/result/masked_control_001/live_report.md) · [GPU review](../../5_test/results/masked_edit_gpu_operation.json) · [Focused test report](../../5_test/results/masked_edit_final_suite.xml).
