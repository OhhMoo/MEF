# Implementation log

We built a working generator for small molecules. It learns to represent a
molecule as continuous numbers, learns to move random noise toward those numbers,
and converts the result into atoms and bonds once at the end.

The first run produced 368 different valid, connected molecules from 1,000
attempts. The later overnight comparison completed all 18 training runs and
supports continuing this idea. Read the
[experiment results](experiment_results_1009.md) for the findings in plain
language. The sections below preserve the implementation and testing history.

10 October 2026. All original research files are under `1_research_file`; upstream
ELF main and its official PyTorch branch are pinned under `0_external_repo`.
Three source studies were completed and reviewed before implementation. The
adaptation note explains which ELF primitives and training ideas we retained.

The data/chemistry, model/objective, and sampling/evaluation builders delivered
their components in parallel. An independent reviewer checked code, math,
and actual operation. During the real training run, model learning, molecular
failures, and independent review were inspected concurrently.

## Runnable generator

- 5,052,856 parameters, with explicit encoder E, clean-prediction flow F, and
  terminal decoder q; symmetric unordered pairs preserve graph information.
- Official QM9 data prepared into 20,000 training, 13,016 validation, and 13,017
  test molecules. Split identities are disjoint; all retained graph encodings
  were checked against their canonical molecular identities.
- Reconstruction warm-up trains E/q. Flow training freezes E and normalization,
  and trains F/q with velocity matching plus noisy terminal decoding (λ=0).
- Sampling uses 31 Euler calls plus one terminal clean prediction, then one
  categorical decode. Every attempted molecule is recorded, including failures.
- Checkpoints include optimizer/RNG states, count prior, and training identities.
  Generated batches reference immutable checkpoints and their SHA256 hashes.
  The standalone generation CLI needs no original processed-data directory.

## Environment and operation

Isolated environment: `/hopper/home/miyao/miniconda3/envs/emf`. PyTorch
2.11.0+cu130, RDKit 2026.9.1, pytest 9.1.1. Allocated GPU: one L40S, driver
595.71.05, CUDA runtime 13.0. Existing environments and upstream code are intact.

The final complete CPU suite passed **53 tests**, with one CUDA test deselected.
Checks include math/equivariance, chemistry, immutable sample provenance, exact
interrupted-flow resume, completed-run replay, and four generation CLI checks.
Report: `5_test/results/pytest_final_cpu.xml`.

Slurm job 370019 passed CUDA forward/backward preflight. Job 370020 exercised
the full suite and GPU training/sampling but failed a reviewer assertion that
expected `cuda` instead of the reported `cuda:0`. The corrected expanded CUDA
operation test passed in job 370022, including periodic/final sampling snapshots
and frozen-encoder invariance. Independent findings are in
`5_test/results/implementation_review.md`.

## First exploration

Job **370023 completed successfully** on gpu14 with one L40S, eight CPUs and
32 GiB RAM. Its configuration is
`3_model_training/configs/explore.yaml`: 2,000 representation updates, 4,000
flow/decoder updates, and 1,000 attempted molecules at flow steps 1,000, 2,000,
3,000 and 4,000. All artifacts are under
`4_experimental_result/result/exploration_001`.

The invocation took **285 seconds**; Slurm recorded **289 allocated GPU seconds**
(0.0803 GPU-hours). Peak Torch-allocated GPU memory was 464,978,432 bytes.
Reconstruction was exact on the logged 256 validation molecules. That diagnostic
was measured throughout training; generation had no reconstruction quality gate.

| Flow updates | Attempts | Chemically valid | Distinct connected valid | Absent from selected training |
|---:|---:|---:|---:|---:|
| 1,000 | 1,000 | 410 | 201 | 194 |
| 2,000 | 1,000 | 536 | 246 | 237 |
| 3,000 | 1,000 | 496 | 287 | 280 |
| 4,000 | 1,000 | 548 | 368 | 352 |

The final batch took 1.473 seconds including transport, terminal decoding, host
chemistry, and ledger construction; file writing is excluded. Its failures were
452 invalid-valence graphs and 180 disconnected graphs. All 368 connected valid
molecules were distinct. Novelty here means absent from the selected 20,000
training molecules, rather than novel chemistry in general.

The reviewer independently recomputed outcomes/accounting for all 4,000 attempts
and verified that E and normalization were unchanged through the final checkpoint.
Standalone GPU job **370068** regenerated the final 1,000 attempts from the saved
checkpoint: all scientific ledger fields matched exactly. Its outputs/comparison
are under `5_test/results/final_checkpoint_reproduction` and
`5_test/results/checkpoint_reproduction_comparison.json`.

After training, operation fixes made completed-run replay reuse a matching
immutable snapshot and aligned the fresh CUDA CLI with the runner's `high` FP32
matmul precision. The run manifest records the source hashes used by the training
job; the final replay and CPU suite exercise the current source.

All five GPU jobs, including the earlier failed assertion check, consumed
**0.0858 allocated GPU-hours** total. Scheduler accounting is saved in
`4_experimental_result/result/compute_usage.csv` and `.json`.

Read `model_observations.md` and `molecule_observations.md` beside the run for the
learning/chemistry diagnoses. The first generator and its observed valence and
connectivity failures motivated the overnight sampling and supervision study below.

## Overnight study

**Completion update:** all 18 branches and 90 final sampling batches finished
successfully by **08:51 PDT on 10 October**. No study GPU jobs remain active.
Allocated time was **18.8206 L40S GPU-hours**. Read
[experiment_results_1009.md](experiment_results_1009.md) for the completed results,
the sampling frontier and the distinctions between matched time and matched
updates. At 32 evaluations, connected validity for λ=0/0.1/1 averaged
84.03%/80.60%/77.40% with a trainable decoder and
87.42%/60.27%/55.72% with a fixed decoder. This supports terminal-only
supervision for valid yield in the tested setting, with distribution/diversity
tradeoffs and an update-count qualification.

At 03:37 PDT on 10 October, the coordinator submitted three new representation
preparations, an existing-checkpoint sampling grid, and 18 paired training runs.
Jobs **370075–370078 completed successfully**; training array **370079** started
its first four branches at 03:39 PDT. The remaining branches start automatically,
with four L40S allocations active at once.

Seeds 42/43/44 each share one freshly pretrained representation across six
branches: λ=0/0.1/1 with trainable or frozen-common decoder. Each branch receives
3,600 seconds of measured, synchronized training-update time, followed by 10,000
generation attempts at each of 8/16/32/64/128 flow evaluations. The original
checkpoint has already completed the same 50,000-attempt sampling grid.

Parallel workers built the study orchestration and molecular distribution metrics,
while the independent reviewer checked the mathematics and exercised the runner.
The prelaunch CPU suite passed 70 tests; final workflow checks passed 10 tests.
GPU job **370073** passed both end-to-end operation tests. The prepared-study
audit passed all configuration, data, source-snapshot and initialization checks.
Evidence is under `5_test/results/overnight_*`.

The protocol and interpretation rules are in [07_overnight_experiment_plan.md](../detail_operation/07_overnight_experiment_plan.md).
All active-study artifacts live in
`4_experimental_result/result/overnight_20261010`; the tmux session
`emf_overnight_20261010` refreshes the dashboard, paired effects, generation
frontiers and scheduler accounting once per minute. Source/configuration hashes
remained pinned throughout this batch. The completed run took about five hours
14 minutes from submission. The monitor now displays all completed results;
its tmux session remains available for inspection.
