# Efficient revision of constrained molecular flows

## Research decision

**Can a saved continuous molecular state make a changed design decision cheaper while retaining valid chemistry and the required properties?**

The October 14 study established evidence-dependent generation in the simulator. Its main obstacle is cost: terminal control selected 57.3% successes, versus 51.6% from 64 unguided proposals and 70.8% from 1,600 proposals. The large pool cost about 28% more than control. Continuing after additional evidence retained similar success with less charged computation. The next study measures whether that reuse earns an advantage at a common inference budget.

Use existing hard-masked B checkpoints, source chemistry, and three seeds (142–144). The change from A to B is a prospective decision informed by the completed pilot: B had comparable local yield, 94.4% prior validity, and exact protected-category copying. Hard masking guarantees protected categories; chemical validity still needs terminal evaluation.

## What we build now

| Workstream | Concrete output | Decision it resolves |
|---|---|---|
| B preparation | Source-only B descriptor critics and ordinary conditional adapters, using the existing 20,000-update priors | Does the more reliable prior retain useful steering? |
| Direct comparison | Same controller, same 64 proposals, same evidence sizes and larger-pool reference as the completed A study | How do validity, preservation and useful yield change? |
| Revision benchmark | Paired controlled and ordinary-conditional continuation/restart, unguided ranking, and discrete masked mutation | Does a saved state improve the quality–cost tradeoff? |
| Analysis | Per-campaign completed-batch cost curves, paired uncertainty, feasibility diagnostics and a compact collaboration brief | Which mechanism and collaboration would advance the research? |

The first run reuses the existing controller: 16 flow evaluations per full rollout, four optimization iterations, and trust radii 0.5 and 2.0. Its additional forward and backward work is counted. B conditional adapters receive the same 10,000 source updates as A; critics receive 3,000 fitting steps on source examples and 512 generated source attempts. Apply the preceding A study's source-only audit rule: a generated-descriptor RMSE above 0.2 triggers one refit with 5,000 steps and 1,024 source attempts; retain the fit with the lower finite sum of source audit errors. Report these preparation costs separately from inference. No new molecular encoder or broad architecture sweep is needed for this stage.

## Two distinct update settings

**Evidence refinement.** Keep the true campaign function and goal fixed. Start with four noisy measured compounds, including the lead; reveal a nested set of 16. Refit the response model using the available measurements. Record the change in the estimated response map. Call this refinement, because additional observations need not contradict the early estimate.

**Goal reversal.** Keep all 16 observations and the physical response function fixed. Initially request an increase in endpoint 1; later request a decrease. Preserve endpoint 2 and the protected graph throughout. Store goal orientation separately from measurements:

$$
s_{\mathrm{old}}=(1,1),\qquad s_{\mathrm{new}}=(-1,1),
\qquad s\odot\Delta y\geq(0.25,-0.10).
$$

Only the decision computation uses the oriented response. Exact evaluation retains the original physical response and applies the requested orientation after graph commitment. No assay label is changed to imitate a preference change.

Keep every predeclared campaign, including those with no observed feasible molecule under the reversed goal. Report compatible-library feasibility for each orientation after proposals are committed. The previous campaign generator guaranteed source solutions only for the original goal. This first stress test therefore measures revision on its existing task distribution; it is not a balanced new benchmark of bidirectionally feasible goals.

Updates occur at the nearest interior solver states to 25%, 50% and 75% progress. Seed, lead, mask, support and Gaussian noise remain paired. Generation progress is a computational coordinate; laboratory measurements are represented as information revealed between computational decisions.

## Comparators and costs

For each update setting, compare controlled continuation, controlled restart, ordinary conditional continuation, ordinary conditional restart, and unguided sample-and-rank. Include a direct masked-mutation comparator with the same support reserve, eligibility and final selection rule. All-new-information-from-start is a reference for the cost of having complete information earlier, not a causal alternative available before revelation.

Conditional continuation is essential: it changes the response context at the saved state and needs no optimization through the remaining trajectory. Each method pays for its own old context and prefix. The expensive controller's old computation is not charged to cheap conditional generation.

Record two cost views:

1. **Total strategy cost:** evidence calibration and old-context preparation, the executed prefix, and new-context continuation or restart, including terminal decoding, chemistry validation and ranking.
2. **Post-update cost:** only the work required after the new information is supplied. The already executed work is recorded alongside this view.

Count synchronized wall time, forward flow evaluations, backward traversals, decoder/critic calls, and attempted candidates separately. A forward/backward sum is a bookkeeping metric, not a measure of equal hardware work. Record warm-up/model loading separately and keep batch sizes and GPU type fixed. Reused candidates or cached prefixes carry both physical and strategy-charged costs.

Save chronological batches. A candidate is available only when its complete batch, decoding and scoring have finished. At a given wall budget use complete batches, with no fractional candidates or interpolated success. Unused selection slots remain failures. First profile validation to lock a fixed budget grid and generation caps; use the locked grid on test. Expanding the number of independent proposals is the way each method spends additional budget. Any curve that ends before a budget is marked as truncated rather than extrapolated.

The direct A/B comparison preserves the old pipeline for attribution. The new cost benchmark gives the stronger computational comparison. Differences between the two scoring/accounting protocols must remain explicit.

## Evaluation contract

Primary quantity: successful distinct compounds among 16 requested selections, after ranking with available evidence, at each inference budget. Also report valid connected attempts, exact protected-category preservation, retained endpoint 2, distinct eligible yield, edit count, source novelty, and unused slots.

Use the same maximal 16-compound support reserve for every compared method. Source compounds are allowed in the primary candidate pool, matching the prior flow evaluation; source-excluded successes are a separate statistic. Score the independently decoded graph with the exact simulator. Its hidden response coefficients never enter a deployed decision arm.

Report per-seed results and paired campaign effects. For confidence intervals, resample campaign IDs as clusters, carrying their seeds, methods and update times together. Sixteen campaigns and three seeds provide a small paired study; report interval widths and campaign-level variation directly. The familiar two-family chemistry and known three-feature calibration remain the scope of this run. The original representation initializer predates the molecular split.

## Next mechanism: selected by the failure we observe

| Observation | Next small intervention |
|---|---|
| Conditional continuation matches expensive control | Use ordinary conditioning as the working revision mechanism; investigate why and when its saved state remains useful |
| Late revision cannot change decoded decisions | Test a single rollback rule that resumes from an earlier saved state |
| Terminal backpropagation dominates a useful control regime | Fit one source-trained terminal-response approximation, then test one-step permitted guidance against the full-rollout teacher |
| Guidance changes properties but damages validity | Study bounded interventions and decoded validity near source-supported trajectories |
| Expanded sampling or mutation dominates all tested regimes | Retain the result and narrow the claim; improve the task or representation before adding modules |

A later controllability objective must connect an editable perturbation to an independently checked terminal change. Matching a soft property gradient alone cannot establish chemical usefulness. Develop one such mechanism after this benchmark identifies the useful regime.

## Execution and review

Root owns the manifest, code snapshots, stage runner, scheduler and reports. A GPT-6-sol worker builds the revision evaluator; GPT-6-astra reviews scientific design, math and operational behavior; GPT-6-luna prepares the collaboration brief. Source preparation runs in parallel while the evaluator is built. Review meaningful tests and an allocated-GPU smoke before submitting the full revision benchmark. Use distinct outputs under `4_experimental_result/result/revision_efficiency_001`, a dedicated tmux coordinator, and a bounded study with at most six concurrent GPUs. Preserve every failure and retry record.

Daily status and observations go to `1_research_file/result_log`. The general research guide records the approved direction; this file owns experiment details. [Collaboration readiness](../study_file/12_collaboration_readiness.md) distinguishes a scientific feedback conversation, a methods collaboration, and measured/prospective chemical validation.
