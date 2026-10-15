# Experiment results — 1015


<!-- EVIDENCE_STEERING_OVERNIGHT_BEGIN -->
## Evidence-steering overnight status

Updated: 2026-10-15 13:27:48 PDT
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
