# Implementation log — 1014

## Masked-control coordinator recovery

Actual recovery: 2026-10-14, approximately 01:08–01:15 PDT. The previously authorized overnight study had completed source training, known-response validation, oracle comparisons and 16-measurement validation. All 36 tracked scheduler allocations reported COMPLETED. The coordinator itself had stopped at 23:40 PDT after a completed job disappeared from `squeue`.

The coordinator now checks completed result artifacts before queue polling. If a queue ID expires while outputs are incomplete, it checks terminal states in `sacct`, retaining genuine failures for retry. Scientific settings, completed models and experiment outcomes are unchanged. The recovery uses the separate `source_snapshot_v3`; original coordinator state is saved in `poll_recovery_1014.json`.

The reporting script also included its own previous JSON output as an input. That caused recursive growth to 15 GB and delayed coordination. Discovery now reads only primary result summary filenames and excludes code snapshots and derived reports. The previous derived file is retained under the study's `analysis` folder. Eight focused recovery tests pass, including repeated report refresh and completed/failed scheduler cases.

The previously planned held-out control, oracle and simple-baseline stage was submitted as Slurm array 375713. Test representation and revision stages remain in the coordinator. No source training was repeated for recovery.

The plain-language scientific snapshot is [experiment_results_1014.md](experiment_results_1014.md). Its measurements were recomputed from primary summary and attempt/metric CSV files. [Recovery tests](../../5_test/results/masked_reporting_recovery_1014.xml) and [recovery record](../../4_experimental_result/result/masked_control_001/poll_recovery_1014.json) provide operational evidence.

<!-- REVISION_EFFICIENCY_IMPLEMENTATION_BEGIN -->
## Efficient revision: implementation and launch

Actual launch: 2026-10-14 09:33 PDT. The user approved a shift toward efficient, chemically constrained revision and requested a mixed-model parallel build.

- Reused the three completed hard-masked B priors. Slurm array `375825` prepares three source-only descriptor critics and three 10,000-update conditional adapters. Apply the same source-audit refit policy as the A study.
- GPT-6-astra reviewed design, dynamics and accounting; GPT-6-sol built the new revision evaluator; GPT-6-luna prepared the collaboration brief and result reporter. Root owns shared protocol, snapshots, scheduling and integration.
- Added a benchmark for an unchanged assay with a reversed goal and for nested 4-to-16 noisy evidence refinement. Ordinary conditional continuation and restart, expanded sampling and direct one/two-edit mutation are included. All methods use the same support reserve and candidate eligibility.
- Record total and post-update cost, valid/invalid attempts, decoded edits, retained endpoint 2, and whole-batch quality–time curves. Goal orientation remains separate from physical experimental response.
- Added a persistent coordinator in tmux `emf_revision_20261014`, with immutable task arrays, recorded bounded retries, a 48 GPU-hour operating ceiling, and six concurrent GPUs at most. The existing study remains intact.
- CPU integration and scheduler regressions passed before launch. The full revision stage waits for a hashed code snapshot, successful allocated-GPU smoke, and test artifacts. Validation timing alone selects the common test budget grid.

[Execution plan](../detail_operation/06_revision_efficiency_experiment_plan.md) · [Review](../study_file/11_revision_efficiency_review.md) · [Collaboration brief](../detail_operation/07_collaboration_brief.md) · [Study status](../../4_experimental_result/result/revision_efficiency_001/status.json)

Outcomes of this new benchmark remain pending. The completed October 14 study's numbers are retained in the earlier result section.
<!-- REVISION_EFFICIENCY_IMPLEMENTATION_END -->

### Operational check and full launch

The integrated suite passed **27 tests**. Allocated L40S smoke job `375834` completed the two update settings and eight method arms, producing 864 logical attempt rows with **zero protected-category violations** and nonnegative incremental timing. These are operational checks, not scientific performance estimates.

Full B comparison array `375838` and revision-validation array `375841` run concurrently, three seeds each. The coordinator adopts the validation submission from its immutable task record, locks the test wall budgets using validation timings, and then starts the held-out revision tasks. Results and failures remain under `revision_efficiency_001`; the study has a 48 GPU-hour operating ceiling.

The coordinator initially retried the three completed source-critic refits after its artifact checks reported them incomplete. Both original and retry artifacts and Slurm success records are retained. Completion now allows 90 seconds for shared-filesystem artifact visibility, and the selected preparation paths are frozen across coordinator restarts. The recorded operational error remains in the status history. GPU accounting includes those retries and the independent smoke allocation.

[Operational check](../../5_test/results/revision_efficiency_gpu_operation_1014.json) · [Test report](../../5_test/results/revision_efficiency_final_1014.xml) · [Input hashes](../../4_experimental_result/result/revision_efficiency_001/input_provenance.json) · [Reviewed code and launch record](../../4_experimental_result/result/revision_efficiency_001/reviewed_ready.json)

<!-- REVISION_RECOVERY_001_BEGIN -->
## Final focused revision experiments

Started 2026-10-14 18:11:39 PDT. Study: `revision_recovery_001`.

Built the user's final three experiments on the existing frozen hard-masked B models: rollback after a late goal reversal; bounded suffix interventions and exact-response diagnostics; one small state-adjustment network gated by source recovery and exact validation. The scope stops after these three experiments and their decision gates.

GPT-6-sol built the recovery evaluator, GPT-6.1-sol built the adapter, GPT-6-astra independently reviewed math/code and real GPU operation, and GPT-6-luna built reporting and figures. Root owns frozen task contracts, provenance, Slurm/tmux supervision, and bounded retries.

Verification: **22 tests passed**. Allocated-L40S checks passed for the pretrained recovery evaluator, two-step adapter training, and integrated adapter rollout (jobs 376053 and 376055). The reviewer independently recomputed molecular outcomes, checked masks and saved states, and verified every candidate's cost accounting. Operational smoke outputs are excluded from scientific results and gates.

Source diagnostics and validation are running on six GPUs, jobs **376056** and **376057**, each with three model seeds. Session: `emf_recovery_20261014`. The coordinator automatically starts the main test, trains the fixed 500-update adapter only if source recovery passes, and starts its test only if exact validation passes. Failure of either scientific gate is recorded as a result; it does not trigger a new architecture search. Operational failures retain their artifacts and receive at most one retry.

All model, data, and source inputs are hashed. The first two arrays use source snapshot v1. Snapshot v2 corrects only the report's definition of goal reversal and explanation of proposal caps; scientific code and task manifests are identical. Existing jobs were preserved while the coordinator adopted the report clarification.

The report separates raw candidate validity from validity after choosing one candidate per saved state. It reports all campaigns, independently verified feasibility, paired campaign uncertainty, and total/post-update timing. Fresh sampling references retain their distinct history costs. The original known-descriptor synthetic benchmark remains the scope of the results.

- [Experiment plan](../detail_operation/08_revision_recovery_experiment_plan.md).
- [Plain-language results](experiment_results_1014_3.md).
- [Live detailed report](../../4_experimental_result/result/revision_recovery_001/live_report.md).
- [Frozen tasks and status](../../4_experimental_result/result/revision_recovery_001/).
- [Independent review](../study_file/13_revision_recovery_review.md).
- [Final CPU checks](../../5_test/results/revision_recovery_final.xml).
- [GPU review](../../5_test/results/revision_recovery_gpu_review.json).

<!-- REVISION_RECOVERY_001_END -->
