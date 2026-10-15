# Experiment results — 1011

Actual study date: 11 October 2026, America/Los_Angeles.

## Question

Can a frozen molecular generator use a few observations from a new campaign
to propose chemical changes that improve one endpoint while retaining another?

## Study status

The model and exact-response data are built. The allocated-GPU operation completed.
The automatic record below shows the current training and comparison status.

## Planned comparisons

The first complete model is followed by parallel paired-seed runs of the
structured direction flow and ordinary conditioning. The same source checkpoint
supports cheap evidence-swap, selector-zero and standard-proposal comparisons.
Task-state fitting and selector adaptation use the same target observations.

Every research comparison uses the declared lead neighbourhood, nested support
sizes 16/32/64/128, 1,000 attempted candidates and 16 selection slots. Flow methods
use 32 total flow evaluations per attempt. Predictor-guided search uses zero flow
evaluations and records its actual predictor calls and host chemistry time. The lead measurement counts within support. Rankings are saved before
oracle evaluation; failed or unfilled slots remain in the denominators.

## How to read the results

Pre-ranking yield measures how often the generator proposes an independently
confirmed joint-profile success. Top-16 success measures how many compounds
selected for testing meet both requirements. Budget efficiency divides successes
by the allocated support-plus-testing compound budget. Full-pool toy-oracle calls
are reported separately.

The exact-response experiment supplies computational truth. The ChEMBL audit
reports available measured records and endpoint overlap. Measured-library and
prospective claims require their corresponding independent outcomes.

Detailed CSV/JSON, checkpoints and sample ledgers:
`../../4_experimental_result/result/calibrated_redesign_001/`.


### Pre-launch evidence — 2026-10-11T23:34:57.554654+00:00

Data preparation completed in Slurm job 371249. It retained 20,000 source molecules, 13,016 validation molecules, and 13,017 test molecules, with 32 source, eight validation, and eight test response functions. Source and held-out molecule identities are separate. No measured biological result has been collected in this new study.

The controlled study will compare the proposed flow with an ordinary conditional flow, swapped campaign evidence, a disabled direction branch, standard generator proposals, support-fitted task state, selector adaptation, and predictor-guided molecular search. Each comparison uses three source seeds, eight held-out campaigns, one lead per campaign, and evidence sizes of 16, 32, 64, and 128 compounds. Each run receives 1,000 attempted candidates and 16 selection slots. The paired proposed model supplies the same response scorer for every method. Unfilled slots count as failures.

The primary results will state how many of the 16 selected candidates meet both endpoint requirements, how often useful candidates appear before ranking, and how those outcomes change as observations increase. Private exact outcomes are read only after proposal and ranking files are committed. All extra full-pool oracle calls, invalid attempts, duplicates, compute time, and actual model calls are recorded.

<!-- EMF_REDESIGN_REPORT_BEGIN -->
## Automatic experiment record

Updated 2026-10-15T20:28:13.509487+00:00. Phase: all 24 planned comparisons complete; nine interrupted evaluations recovered.

Completed exact-oracle evaluation batches: 768. Held-out test batches: 768. Each success meets both requested endpoint requirements.

| Model and comparison | Split | N=16 | N=32 | N=64 | N=128 |
|---|---|---:|---:|---:|---:|
| conditional: Native evidence | test | 0.0% | 0.0% | 0.0% | 0.0% |
| factorized: Selector disabled | test | 0.0% | 0.0% | 0.0% | 0.0% |
| factorized: Native evidence | test | 0.0% | 0.0% | 0.0% | 0.0% |
| factorized: Predictor search | test | 9.1% | 9.1% | 8.6% | 8.3% |
| factorized: Adapted selector | test | 0.0% | 0.0% | 0.0% | 0.0% |
| factorized: Standard proposals | test | 0.0% | 0.0% | 0.0% | 0.0% |
| factorized: Swapped flow evidence | test | 0.0% | 0.0% | 0.0% | 0.0% |
| factorized: Fitted task state | test | 0.0% | 0.0% | 0.0% | 0.0% |

Each percentage is the fraction of 16 allocated testing slots that met the complete profile. A missing selection or failed external test contributes zero. The lead is included in N.

The CSV curves average leads within each campaign and seed, then campaigns and source seeds. The 95% bootstrap intervals resample campaigns and source seeds; individual proposals are not treated as independent campaigns.

The native architecture comparison uses the same factorized response scorer within each paired seed, campaign, lead, and support size.

| Split | Support N | Conditional − factorized success | 95% interval | Matched cases |
|---|---:|---:|---:|---:|
| test | 16 | +0.0 points | 0.0 to 0.0 points | 24 |
| test | 32 | +0.0 points | 0.0 to 0.0 points | 24 |
| test | 64 | +0.0 points | 0.0 to 0.0 points | 24 |
| test | 128 | +0.0 points | 0.0 to 0.0 points | 24 |

Pre-ranking yield counts distinct unmeasured successes per attempted proposal; the local version also requires the declared lead neighbourhood. Efficiency is successes divided by N+16. The simulator's full-pool oracle calls are extra evaluation costs.

Full-pool oracle calls in controlled batches: 325,737.

| Preliminary run | Split | Support N | Attempts/testing slots | Joint yield before ranking | Local joint yield | Selected success |
|---|---|---:|---:|---:|---:|---:|
| operation: conditional/native | validation | 16 | 64/4 | 6.2% | 0.0% | 0.0% |
| operation: conditional/native | validation | 128 | 64/4 | 6.2% | 0.0% | 0.0% |
| operation: factorized/native | validation | 16 | 64/4 | 10.9% | 0.0% | 0.0% |
| operation: factorized/native | validation | 32 | 64/4 | 10.9% | 0.0% | 0.0% |
| operation: factorized/native | validation | 64 | 64/4 | 10.9% | 0.0% | 0.0% |
| operation: factorized/native | validation | 128 | 64/4 | 10.9% | 0.0% | 0.0% |
| operation: factorized/selector_adaptation | validation | 16 | 64/4 | 10.9% | 0.0% | 0.0% |
| operation: factorized/selector_adaptation | validation | 128 | 64/4 | 10.9% | 0.0% | 0.0% |
| pilot: factorized/native | validation | 16 | 1000/16 | 3.3% | 0.0% | 0.0% |
| pilot: factorized/native | validation | 32 | 1000/16 | 3.3% | 0.0% | 0.0% |
| pilot: factorized/native | validation | 64 | 1000/16 | 3.3% | 0.0% | 0.0% |
| pilot: factorized/native | validation | 128 | 1000/16 | 3.3% | 0.0% | 0.0% |

These rows exercise the integrated pipeline or guide development. Their declared attempt and testing budgets are retained in the case CSV.

The bounded ChEMBL_37 availability audit examined 1,000 on-target and 1,000 off-target records. It found 31 overlapping assay pairs, 228 joined records over 123 distinct molecules, and at most 41 compounds in one assay pair.

Current-representation coverage: 0 joined rows. Observed heavy-atom counts span 15 to 45. This is a data-availability audit; measured transfer evaluation will follow dataset and representation preparation.

[Case measurements](../../4_experimental_result/result/calibrated_redesign_001/analysis/case_metrics.csv), [support-size curves](../../4_experimental_result/result/calibrated_redesign_001/analysis/support_curves.csv), [paired comparisons](../../4_experimental_result/result/calibrated_redesign_001/analysis/paired_curves.csv), [current machine-readable report](../../4_experimental_result/result/calibrated_redesign_001/analysis/report.json).

All design-success values above use the exact response oracle. Predicted probabilities are used for ranking.
<!-- EMF_REDESIGN_REPORT_END -->


### Small operation result — 2026-10-11T23:40:29.647211+00:00

This was a pipeline check with one validation campaign, 64 attempts per evidence setting, eight flow evaluations per attempt, and four selection slots. It is separate from the 1,000-attempt research comparisons.

For each of the four evidence sizes, the run produced 14 distinct supported connected molecules from 64 attempts. Seven met both simulated property targets. None met the declared structural-neighbourhood rule for this lead, so all four selection slots were unfilled and selected success was zero. The private evaluator made 14 full-pool calls after rankings were committed. These preliminary figures establish operation and expose the local-redesign requirement; they do not establish an advantage from campaign evidence.

The 3,000-update validation pilot is running. The larger source/control arrays have been submitted with explicit dependencies, so later results will be added automatically as jobs finish.


### First exploration complete — 2026-10-11T23:47:07.132830+00:00

The native model completed 3,000 source updates in 260.03 training seconds. It then made 1,000 attempts for each of four validation campaigns at each evidence size (16, 32, 64, 128). All model weights stayed fixed during evaluation. These are 16 completed exploratory batches, separate from the held-out control study.

Across the four campaigns, about 33 of every 1,000 attempts produced a distinct, unmeasured molecule meeting both simulated property targets. The mean was 33.25 at N=32 and 33 at the other evidence sizes. No proposal meeting both targets also met the required lead-neighbourhood rule. Selected success was therefore zero at every evidence size. The number of observations has not yet shown a useful redesign benefit in this pilot.

This result identifies local chemical redesign as the current limitation. The paired source models and shared-predictor controls are running to determine whether longer training, conditioning, or proposal search changes that outcome. Four of the six source runs are now active; all 24 per-method evaluations are submitted behind the source-stage dependency.
