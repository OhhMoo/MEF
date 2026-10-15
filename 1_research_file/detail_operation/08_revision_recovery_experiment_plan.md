# Final focused experiments: recovering a molecular design after a goal change

## Research question and stopping point

**Can we recover useful molecular edits from an existing continuous generation state when the design goal reverses, and what computation does that recovery require?**

Run three bounded experiments: rollback, recoverability diagnostics, and one small learned state adjustment. Then assemble the discovery, explanation, and attempted solution for professor outreach. A well-supported negative result is sufficient to start a scientific collaboration discussion. This round does not expand into another architecture sweep.

The preceding revision study is complete. At approximately 75% progress on its held-out campaigns, ordinary conditional continuation selected 213/768 successful candidates after goal reversal, compared with 292/768 for restart. When additional measurements refined the same goal, continuation gave 577/768 versus 586/768. These full-cap comparisons used different computational costs. The new experiments study the reversal gap directly.

## Shared setup

Reuse the frozen hard-masked B prior and conditional checkpoints for seeds 142, 143, and 144. Use their existing response surrogates. Keep the encoder, generator, decoder, and response surrogate fixed during evaluation. Only the small revision network is trained in Experiment 3.

Use the existing small molecular benchmark and independent campaign response functions. Keep 16 support measurements fixed. Generate under an increase request for endpoint 1 until the solver state nearest 75% progress; then request a decrease. Endpoint 2 and protected graph categories retain their requirements. Physical response labels never change when the goal changes.

Use 16 total flow evaluations for a complete trajectory, including its terminal clean prediction. Cache the actual states nearest 25%, 50%, and 75%. Pair lead, mask, campaign, model seed, support, and original Gaussian noise across compared strategies. Store state hashes and root identifiers.

The existing test campaigns have already informed the choice to investigate reversal. Describe the new results as additional interventions on held-out campaign functions previously examined in an earlier study. Chemical-series and assay transfer remain later collaboration tasks.

## Experiment 1: rollback

At the 75% goal change, compare:

| Strategy | Starting point after the update |
|---|---|
| Continue | Current 75% state |
| Roll back halfway | Saved 50% state |
| Roll back earlier | Saved 25% state |
| Restart | Original Gaussian noise at time zero |

All four strategies pay for the original computation through 75%. A rollback saves recomputing an early prefix, but does not erase the cost of the later work already discarded. Record total strategy cost and work performed after the update separately.

Use 128 root trajectories per validation campaign and 256 per test campaign, in batches of eight. Include up to 1,600 independent ordinary conditional and prior samples as strong sampling references. Fresh sampling and direct mutation have no historical prefix; label them as fresh-start references and use post-update costs for comparisons with an already available saved state. Do not interpret their fresh-only time as a total-history comparison.

## Experiment 2: recoverability and selection

A deterministic ODE has one completion from a fixed state and condition. To obtain different completions, apply explicit perturbations to editable latent coordinates before the remaining suffix. Protected coordinates are restored exactly.

For the first 32 paired roots per campaign, use eight trials within an editable-coordinate RMS radius of 0.5. Compare:

| Diagnostic | Information used to propose or select |
|---|---|
| Ordinary continuation | Revised calibrated condition; one deterministic suffix |
| Randomized suffix and learned scoring | Bounded random interventions; evidence-derived response scoring |
| Exact scoring of the same suffix pool | The identical proposals, with exact simulated outcomes used for selection |
| Bounded oracle-informed search | Exact outcomes inform the bounded search; every tested candidate is counted |
| Independent restart with exact scoring | The same number of attempted exact evaluations from independent full trajectories |
| Direct masked mutation | A simple discrete editing reference, including a privileged exact-scoring diagnostic |

The oracle arms are diagnostic: their hidden response information is unavailable to deployed methods. Report attempted evaluations, actual exact-response calls, forward evaluations, and wall time. Invalid trials count against the attempted-candidate budget. Root diagnostics select one candidate per root, so their success is a restricted revision policy rather than an unrestricted top-16 pool optimum.

The principal diagnostic is the fraction of direct-failed roots for which a tested bounded intervention produces an independently verified valid success. Distinguish availability of such a candidate from the learned scorer's ability to select it. Failure of eight trials establishes failure within this tested intervention class and budget, not mathematical unreachability.

## Feasible examples and independent evaluation

Before querying a model, audit compatible reference molecules using the exact simulator. Save their identities, masks, measured-property changes, support exclusion, and goal eligibility. Keep every declared campaign in the main result. Separately report campaigns with verified eligible examples for both the original and reversed goals.

A missing witness means the finite reference library supplies no witness. It does not establish that a solution is impossible. Never choose the feasible stratum using the generator's success. The reviewer independently recomputes responses from committed molecular identities and fixed simulator coefficients.

## Experiment 3: one minimal revision network

Use a small masked residual state map with hidden width 64, gain 1, and an editable RMS adjustment bound of 0.5. Train for 500 updates. Keep the base model frozen and verify its tensors and checkpoint hash before and after training.

Source training examples pair old-goal and revised-goal trajectories from the same Gaussian noise. Both use campaign response coefficients fitted from source observations. The network learns to map the saved old-goal state toward the corresponding revised-goal state. This is a single state-distillation hypothesis; decoded molecular performance determines whether it works.

Two decisions are fixed before the scientific runs:

1. **Training gate:** run the network training only if source diagnostics recover at least eight direct-failed roots across at least two source campaigns within RMS 0.5. Use 16 source campaigns, four per mask, and three model seeds. Held-out outcomes cannot open this gate.
2. **Deployment gate:** evaluate the fixed final checkpoint once on validation. Continue to its held-out evaluation only if it improves paired selected success over direct continuation and loses no more than five percentage points of raw candidate validity. Otherwise, retain its training and validation failure and stop the mechanism study.

Use all source campaigns for teacher training, with eight paired noises per campaign. Validation state error is a diagnostic, not a tuning criterion. Keep the architecture, radius, gain, and final checkpoint fixed. Report teacher targets that fall outside the permitted adjustment radius.

Compare an accepted revision network with direct continuation, both rollback states, and restart using the same root states and proposal counts. Training cost is reported separately from inference. Its held-out run does not retrain the frozen generator.

## Metrics and uncertainty

Select up to 16 distinct candidates per campaign using the information allowed to that arm. Count successful requested slots; empty slots are failures. Also report raw candidate validity, exact protected-category preservation, endpoint-2 retention, distinct candidates, and root-level recovery.

Keep raw candidate validity separate from validity after choosing one winner per root. Preserve every attempted candidate, including invalid graphs, duplicates, and failed interventions.

Compare complete batches at common wall budgets of 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, and 32 seconds per campaign. This grid was set using the preceding study's validation timing. Record proposal caps and unused budgets. Include decoding, scoring, host chemistry, and evidence calibration in charged time; show forward evaluations separately.

Use paired campaign-level effects and 95% bootstrap intervals. Resample campaign IDs while carrying all three seeds and methods together. Report all campaigns and the independently verified feasible stratum. The 16 campaign cases share two small chemical families, which defines the scope of this uncertainty estimate.

## Parallel execution and records

The root coordinator owns task manifests, immutable code snapshots, Slurm submissions, and recovery. GPT-6-sol builds rollback and diagnostics. GPT-6.1-sol builds the small revision network. GPT-6-astra reviews the mathematics, code, and integrated operation. GPT-6-luna prepares the report and figures after implementation slots become free.

Run source diagnostics and validation in parallel. Source recovery can start adapter training while validation continues. Run the main test alongside adapter validation, then the gated adapter test. Use at most six concurrent single-L40S jobs, three model seeds per stage. Keep one bounded retry with a distinct output directory for operational failures. Stop after these experiments and the recorded gate outcomes.

Artifacts belong in `4_experimental_result/result/revision_recovery_001`. Tests and review evidence belong in `5_test`. The coordinator runs in a dedicated tmux session. The daily plain-language report is [experiment_results_1014_3.md](../result_log/experiment_results_1014_3.md), with actual run timestamps. Store two main figures with the raw results: success versus computation, and recovery from direct-failed states.

The outreach package should contain a one-page question and mechanism proposal, those two figures, and a specific collaboration need. It can be prepared whether the small revision network succeeds or fails. Contact follows review of the completed evidence and a targeted choice of collaborators.
