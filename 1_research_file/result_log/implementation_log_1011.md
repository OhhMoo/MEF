# Implementation log — 1011

Actual work date: 11 October 2026, America/Los_Angeles. This record follows the
corrected math and execution plans approved for parallel implementation.

## Current implementation

Three agents built campaign data, the reusable-direction denoiser and joint
response model, and frozen sampling/evaluation. The coordinator integrated the
components and submitted the durable tmux/Slurm workflow. An independent reviewer
checks the code, math, and integrated operation. Current status appears in the
automatic record below.

The build extends the existing ELF-derived graph generator. Source training
uses independent goals with feasibility-weighted flow loss, all-query response
likelihood and terminal decoder training. Deployment freezes all learned weights.

| Role | Owned deliverable |
|---|---|
| Data agent | Nonlinear exact-response campaigns, source-pair conditional count rows, support/query records, bounded ChEMBL audit |
| Model agent | Shared F/B flow, evidence/selector modules, joint Gaussian response, ordinary conditional-flow alternative |
| Sampling agent | Frozen generation, committed rankings, external oracle scores, budgeted success metrics and evidence interventions |
| Coordinator | Shared contracts, source training runner, parallel run matrix, tmux monitoring and daily records |
| Independent reviewer | Code/math review and integrated CPU/GPU operation after a builder hands off |

## Runtime

Environment: `/hopper/home/miyao/miniconda3/envs/emf`. Installed versions checked:
Python 3.12.13, Torch 2.11.0+cu130, SciPy 1.17.1, RDKit 2026.09.1.
SciPy is now recorded as a direct dependency for joint response scoring.

The scheduler exposes 15 L40S nodes with four GPUs per node in the main partition.
Use one L40S, eight CPUs and 32 GiB per training allocation, with at most four
experiment allocations active together. The initial study stays within the
execution plan's 12 GPU-hour allowance.

## Handoffs and launches

The coordinator updates this section with actual job IDs, reviewed interfaces,
operation results and source hashes as each stage launches.

Knowledge studies belong in `../study_file/`; machine-readable run records
belong under `../../4_experimental_result/result/calibrated_redesign_001/`.

### 2026-10-11T23:08:41.157949+00:00

CPU preparation submitted as Slurm **371249**. Persistent tmux session: **emf_redesign_20261011**, windows agents/data/results/audit. The agents window displays actual handoffs and Slurm states; /agents in the Codex client lists the collaboration workers.


### Integrated handoff — 2026-10-11T23:34:57.554654+00:00

The three builders delivered campaign data, the shared-direction model and joint response predictor, and frozen sampling/evaluation. The independent reviewer exercised real cached molecular graphs, training, checkpoint reload, and external-oracle evaluation. The integrated CPU suite passed 62 checks in 10.45 seconds. The frozen encoder now uses the same float32 computation during support caching, training, and sampling. Restart checks preserve completed model files and all prior evaluation rows.

The study now has a pinned source snapshot and 32 GPU tasks: one operation check, one native validation pilot, six paired source training runs, and 24 held-out per-method evaluations. Each array runs at most four tasks at once. Their wall-time caps total 11.17 GPU-hours. The launch controller requires each earlier stage to finish successfully before starting its dependent stage.

The public ChEMBL 37 audit examined 2,000 raw endpoint records. The largest coherent two-endpoint assay library contains 41 compounds, with 15–28 heavy atoms; it cannot support this nine-atom model or the requested 64/128-compound evidence settings. The present study therefore uses exact simulated responses. The measured-data audit is saved separately in study_file/08_measured_campaign_data_study.md.


### Allocation revision — 2026-10-11T23:37:49.088293+00:00

Initial eight-CPU tasks were cancelled while still pending. Free L40S devices existed only on CPU-saturated nodes. One CPU/one Torch thread per GPU task allows four allocations on those nodes; all paired methods receive the same resource limit. The cancelled jobs consumed no GPU time. Original submission manifests/configurations remain archived. Pinned model and control source bytes are unchanged.

<!-- EMF_REDESIGN_REPORT_BEGIN -->
## Automatic implementation and runtime record

Updated 2026-10-15T20:28:13.509487+00:00. Phase: all 24 planned comparisons complete; nine interrupted evaluations recovered.

Source runs with a final summary: 8 of 8 discovered runs. Committed generation batches: 842. Externally evaluated batches: 794.

| Validation report | Passed | Failed/error | Skipped |
|---|---:|---:|---:|
| [redesign_integrated_cpu.xml](../../5_test/results/redesign_integrated_cpu.xml) | 62 | 0 | 0 |
| [redesign_data_worker.xml](../../5_test/results/redesign_data_worker.xml) | 9 | 0 | 0 |
| [redesign_model_cpu.xml](../../5_test/results/redesign_model_cpu.xml) | 15 | 0 | 0 |
| [redesign_sampling_evaluation_cpu.xml](../../5_test/results/redesign_sampling_evaluation_cpu.xml) | 22 | 0 | 0 |
| [redesign_reporter_cpu.xml](../../5_test/results/redesign_reporter_cpu.xml) | 7 | 0 | 0 |
| [redesign_search_worker.xml](../../5_test/results/redesign_search_worker.xml) | 5 | 0 | 0 |
| [redesign_adaptation_cpu.xml](../../5_test/results/redesign_adaptation_cpu.xml) | 6 | 0 | 0 |
| [redesign_operation_gpu_371269.xml](../../5_test/results/redesign_operation_gpu_371269.xml) | 1 | 0 | 0 |

Focused reports overlap with the integrated report; their counts are displayed separately.

| Run | Status | Updates | Total parameters | Source training seconds | Peak Torch GiB |
|---|---|---:|---:|---:|---:|
| operation | source_complete | 50 | 5,355,397 | 4.6 | 0.20 |
| pilot | source_complete | 3000 | 5,355,397 | 260.0 | 0.20 |
| source/conditional_seed_42 | source_complete | 20000 | 5,355,144 | 1708.2 | 0.20 |
| source/conditional_seed_43 | source_complete | 17884 | 5,355,144 | 1800.0 | 0.20 |
| source/conditional_seed_44 | source_complete | 20000 | 5,355,144 | 1689.5 | 0.20 |
| source/factorized_seed_42 | source_complete | 20000 | 5,355,397 | 1772.0 | 0.20 |
| source/factorized_seed_43 | source_complete | 17858 | 5,355,397 | 1800.0 | 0.20 |
| source/factorized_seed_44 | source_complete | 20000 | 5,355,397 | 1706.5 | 0.20 |

Tracked allocated GPU time: 3.966 GPU-hours. This uses elapsed allocation time multiplied by the allocated GPU count; CPU-only preparation contributes zero.

| Slurm job | State | Exit | GPUs | Elapsed seconds | GPU-hours |
|---|---|---|---:|---:|---:|
| 371249 | COMPLETED | 0:0 | 0 | 195 | 0.000 |
| 371265 | CANCELLED by 30071 | 0:0 | 0 | 0 | 0.000 |
| 371266 | CANCELLED by 30071 | 0:0 | 0 | 0 | 0.000 |
| 371267_[0-5%4] | CANCELLED by 30071 | 0:0 | 0 | 0 | 0.000 |
| 371268_[0-23%4] | CANCELLED by 30071 | 0:0 | 0 | 0 | 0.000 |
| 371269 | COMPLETED | 0:0 | 1 | 27 | 0.007 |
| 371270 | COMPLETED | 0:0 | 1 | 319 | 0.089 |
| 371271_0 | COMPLETED | 0:0 | 1 | 1787 | 0.496 |
| 371271_1 | COMPLETED | 0:0 | 1 | 1722 | 0.478 |
| 371271_2 | COMPLETED | 0:0 | 1 | 1816 | 0.504 |
| 371271_3 | COMPLETED | 0:0 | 1 | 1816 | 0.504 |
| 371271_4 | COMPLETED | 0:0 | 1 | 1721 | 0.478 |
| 371271_5 | COMPLETED | 0:0 | 1 | 1703 | 0.473 |
| 371272_0 | FAILED | 1:0 | 1 | 22 | 0.006 |
| 371272_1 | COMPLETED | 0:0 | 1 | 127 | 0.035 |
| 371272_2 | COMPLETED | 0:0 | 1 | 134 | 0.037 |
| 371272_3 | COMPLETED | 0:0 | 1 | 123 | 0.034 |
| 371272_4 | COMPLETED | 0:0 | 1 | 138 | 0.038 |
| 371272_5 | FAILED | 1:0 | 1 | 34 | 0.009 |
| 371272_6 | COMPLETED | 0:0 | 1 | 61 | 0.017 |
| 371272_7 | FAILED | 1:0 | 1 | 37 | 0.010 |
| 371272_8 | FAILED | 1:0 | 1 | 26 | 0.007 |
| 371272_9 | COMPLETED | 0:0 | 1 | 144 | 0.040 |
| 371272_10 | COMPLETED | 0:0 | 1 | 156 | 0.043 |
| 371272_11 | COMPLETED | 0:0 | 1 | 124 | 0.034 |
| 371272_12 | COMPLETED | 0:0 | 1 | 159 | 0.044 |
| 371272_13 | FAILED | 1:0 | 1 | 34 | 0.009 |
| 371272_14 | COMPLETED | 0:0 | 1 | 66 | 0.018 |
| 371272_15 | FAILED | 1:0 | 1 | 42 | 0.012 |
| 371272_16 | FAILED | 1:0 | 1 | 23 | 0.006 |
| 371272_17 | COMPLETED | 0:0 | 1 | 144 | 0.040 |
| 371272_18 | COMPLETED | 0:0 | 1 | 141 | 0.039 |
| 371272_19 | COMPLETED | 0:0 | 1 | 126 | 0.035 |
| 371272_20 | COMPLETED | 0:0 | 1 | 174 | 0.048 |
| 371272_21 | FAILED | 1:0 | 1 | 33 | 0.009 |
| 371272_22 | COMPLETED | 0:0 | 1 | 68 | 0.019 |
| 371272_23 | FAILED | 1:0 | 1 | 23 | 0.006 |
| 372620 | COMPLETED | 0:0 | 1 | 26 | 0.007 |
| 372621_0 | COMPLETED | 0:0 | 1 | 120 | 0.033 |
| 372621_1 | COMPLETED | 0:0 | 1 | 157 | 0.044 |
| 372621_2 | COMPLETED | 0:0 | 1 | 101 | 0.028 |
| 372621_3 | COMPLETED | 0:0 | 1 | 119 | 0.033 |
| 372621_4 | COMPLETED | 0:0 | 1 | 175 | 0.049 |
| 372621_5 | COMPLETED | 0:0 | 1 | 111 | 0.031 |
| 372621_6 | COMPLETED | 0:0 | 1 | 117 | 0.033 |
| 372621_7 | COMPLETED | 0:0 | 1 | 173 | 0.048 |
| 372621_8 | COMPLETED | 0:0 | 1 | 109 | 0.030 |

Accounting is refreshed at intervals of at least 60 seconds and queries only job IDs registered in workflow_state.json. Failed and pending jobs stay in the report.

Proposal timing includes transport, the terminal clean prediction, final decoding, host chemistry, neighbourhood checks, and response scoring. Count/noise preparation and support-state inference have separate timing records. Invalid, unsupported, duplicate, and excluded proposals remain in the ledgers.

[Source and runtime table](../../4_experimental_result/result/calibrated_redesign_001/analysis/source_and_compute.csv), [training history](../../4_experimental_result/result/calibrated_redesign_001/analysis/training_history.csv), [Slurm accounting](../../4_experimental_result/result/calibrated_redesign_001/analysis/slurm_accounting.json).
<!-- EMF_REDESIGN_REPORT_END -->


### GPU operation and durable launch — 2026-10-11T23:40:29.647211+00:00

Slurm operation job 371269 used an allocated L40S on gpu06. The independent GPU pipeline test passed (one GPU test; four CPU tests deselected). The real model then completed 50 source updates in 4.57 training seconds, with its frozen encoder and normalization unchanged. Peak tensor allocation was 0.197 GiB; this excludes CUDA runtime memory. The job completed all four small external evaluations.

Current dependent jobs are native validation pilot 371270, paired source array 371271 (six tasks; at most four active), and per-method evaluation array 371272 (24 tasks; at most four active). Each later stage starts only after its predecessor succeeds. The pilot is running. The arrays are submitted and waiting on those dependencies. Source and control code are pinned in source_snapshot; study_manifest.json records exact configurations and hashes.


### Parallel source stage active — 2026-10-11T23:47:07.132830+00:00

The validation pilot completed successfully. Source tasks 371271_0–3 are running on gpu06/gpu07 with finite losses; tasks 4–5 wait for the four-task array limit. The 24 evaluation tasks wait for all six source checkpoints. The recorder is running in tmux emf_redesign_20261011:recorder and updates marked journal sections every 20 seconds. Scheduler accounting refreshes at most once per minute, using raw and array job IDs. Original manual records remain intact.


### Independent GPU audit complete — 2026-10-11T23:50:02.143827+00:00

The reviewer independently reconciled all 256 operation attempts and 56 external-oracle calls across the four support sizes. Probability and exact-outcome recomputation matched saved records. The source encoder/normalization stayed identical; the source decoder changed while remaining trainable. Deployment remained frozen and proposal hashes verified. The short operation changed response scores slightly as evidence increased, but decoded proposals were identical across evidence sizes. [Full GPU audit](../../5_test/results/redesign_gpu_operation_review.json).
