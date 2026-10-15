# Efficient revision under chemical and response constraints

This review fixes the interpretation and acceptance checks for the next diagnostic stage. The coordinator owns the executable plan and scheduler. The implementation reuses the three trained B priors, the selected terminal controller, and source-trained descriptor critics and conditional models. The generator and encoder remain frozen during campaign inference.

## 1. The question supported by the completed study

The completed A study establishes that evidence changes generated identities. At sixteen measurements, selected top-16 success is 57.3% for terminal control, 51.6% for a small unguided pool, 55.7% for conditional generation, and 70.8% for a larger unguided pool. The larger pool costs approximately 1.28 times the controller's measured time. Control costs approximately twenty times the small pool and lowers validity. Continuing after a 4-to-16 evidence update gives similar success with fewer charged traversals, at unequal budgets.

The next question is whether retaining an intermediate molecular state produces useful constrained revisions at a favorable inference cost. B provides exact protected-category copying and previously showed the highest prior validity. Boundary valence, connectivity, retained response constraints, and final joint success still require separate evaluation.

Measurements and scope are recorded in [the final result section](../result_log/experiment_results_1014.md). The simulator uses a known three-feature response basis, two small chemical families, and an inherited initializer that predates the split. Campaign generalization within this setting is the current scope.

## 2. Separate a changed objective from corrected evidence

Let physical response be `d(G) = W [phi(G) - phi(lead)]`. Store the physical coefficient estimate, the observations used to fit it, and the requested response direction separately. Write a goal as

`S d(G) >= g`, where `S = diag(s1, +1)` and `g = (0.25, -0.1)`.

For the current controller and conditional model, the effective coefficient matrix is `S W_hat`. The independent evaluator keeps physical response unchanged and applies `S` only when testing the requested goal. Store both physical and goal-oriented predictions and outcomes.

| Regime | Early information | New information | Held fixed |
|---|---|---|---|
| Evidence refinement | Four noisy observations | Nested sixteen observations | True response and requested direction |
| Goal reversal | Positive endpoint-1 objective | Negative endpoint-1 objective | Observation table, physical response, endpoint-2 preservation, lead and mask |
| Evidence correction | Declared corrupted or noisy measurements | Corrected observation versions and additional measurements | True response, requested direction, endpoint-2 preservation, lead and mask |

Use positive-to-negative reversal for the first conflicting-goal diagnostic. Original campaign construction guarantees feasibility only for the original goal. Retain every predefined campaign and report both directions' compatible-library feasible counts after evaluation. These finite-library counts describe available reference chemistry; zero counts do not prove that no feasible molecule exists.

A sign error deliberately injected into early measurements is a controlled corruption experiment. Corrected versions replace their obsolete records; unaffected observations remain. Naturally noisy four-to-sixteen refinement receives its own label. Record coefficient change and predicted feasible-set disagreement to describe conflict strength. The new evidence package must be identical across compared methods.

Existing helper functions require care: `truth_rows` hardcodes the original rank thresholds, and `ExactMaskedEvaluator` computes original-goal success. A new runner must recompute all direction-dependent success and ranking fields consistently, including source-excluded success. Negating fitted coefficients alone changes the policy without changing the scored objective.

## 3. Comparators and shared information

The compact comparison includes B terminal-control continuation and restart, conditional continuation and restart, all-new-information-from-start references, unguided generation with a growing ranked pool, and direct constrained mutation. The all-information reference has privileged timing because its final information is available from the start; it is an informative reference for the corresponding policy.

Conditional continuation switches the response context at the exact saved solver state. It receives its own inexpensive conditional prefix. Charging it the expensive controller's prefix would obscure the cheaper way to retain a continuous state. Controller continuation and controller restart share the same expensive historical prefix. Every policy keeps the second endpoint constraint and protected graph categories.

Pair campaign, model seed, initial Gaussian draws, support records, lead, mask, and requested final goal. Use the same maximal support exclusion reserve and primary eligibility rule in every method. The old direct-mutation runner excludes source molecules and uses different support seeds; its output cannot serve as a matched primary comparator unchanged. Report source-excluded success separately from the common primary outcome. Descriptor-aware mutation has exact structural descriptor access; neural ranking uses an approximate descriptor critic. Record this distinction and charge descriptor computation.

Keep top-16 slots as the selection denominator, including unfilled slots. Also report distinct successes, valid connected attempts, protected-category preservation, retained endpoint-2 satisfaction, old and new goal satisfaction, source-excluded success, and identity changes under paired updates. Invalid and duplicate attempts remain in the attempt ledger.

## 4. Success against actual computation

Record a cumulative curve at completed batch boundaries. Each point contains the available committed proposals, distinct ranked selections, success, and elapsed wall time. Include calibration, lead encoding, rollout, controller optimization, decoding, chemistry, deduplication, and ranking. Synchronize CUDA at timing boundaries. State separately whether offline exact-response audit time is included; apply the same rule across arms.

Use validation to declare the budget grid and per-method attempt caps. At a budget, select the last completed batch whose cost fits; report unused budget and attempts. Avoid fractional batches or interpolated successes. A method whose attempt cap prevents it spending the budget is cap-limited and must be labeled accordingly. Extra computation can buy extra complete proposals for every method, including restart and conditional baselines.

Maintain three ledgers:

1. Physical work actually executed by the experiment, including any cached outputs.
2. Strategy cost from the beginning: old optimization, executed prefix, update calibration, new optimization and rollout, and selection.
3. Incremental cost after the update: only work performed after information arrives.

Report forward model evaluations including terminal clean prediction, backward traversals, decoder/critic work, and host chemistry time separately. Forward plus backward counts are an accounting summary, not a hardware-independent compute measure. Model/critic training cost is separate from deployment cost. Logical reuse of a restart result at several revision times must not multiply physical work.

For paired uncertainty, resample campaign IDs and retain all seeds, methods, update regimes and budget points within each sampled cluster. Compute the method difference within paired campaign/seed cases before aggregation. Report seed-specific differences alongside the campaign-cluster interval. The current three seeds provide limited information about variation across new training runs. Campaign-cluster intervals describe the sampled campaign distribution within the existing chemical families.

## 5. Endpoint sensitivity and the next mechanism

The full rollout defines a terminal soft response

`H_t(z, u) = S W_hat delta_phi(critic(decode(R_t(z, u))))`.

Its Jacobian with respect to a control or editable state perturbation measures the derivative of this composed, differentiable predictor through the remaining numerical trajectory. It incorporates the base flow, remaining time, terminal clean prediction, decoder probabilities, and critic. A gradient of an instantaneous decoder prediction has a different meaning and must be labeled as a local approximation.

The decoded graph uses categorical argmax and is piecewise constant. An accurate soft-response derivative can therefore coexist with no changed identities, wrong signed changes after a boundary crossing, or invalid boundary valence. A useful derivative audit needs paired finite perturbations at several source-selected magnitudes, separate counts for unchanged/changed/invalid decoded structures, and exact descriptor changes for the valid changed pairs. A zero count of informative changed pairs establishes no directional reliability.

Choose one later mechanism from the diagnostic bottleneck. If the existing terminal controller has a favorable success frontier but rollout differentiation dominates its cost, the smallest amortization route is a terminal descriptor head `h_psi(t,z,lead,mask)` on a frozen generator. Train on source prefixes with terminal descriptors from committed valid decoded source rollouts:

`L_endpoint = E_valid ||h_psi(t,z) - stop(descriptor(G_terminal))||^2`.

Use the same fixed normalization as the existing response basis. Optionally add adjacent-prefix consistency along source rollouts, anchored by the terminal descriptor labels. Consistency alone admits constant solutions. Keep a validity label for all prefixes, including invalid terminal attempts, so accuracy on the valid subset is interpretable.

The corresponding constrained intervention objective is

`L_control(delta) = sum_k [g_k - (S W_hat delta_phi(h_psi(t,z + P delta)))_k]_+^2 + alpha ||P delta||^2`.

Both response rows remain in the objective. Projection `P` preserves allowed coordinates; terminal category copying preserves protected categories. Validity remains an evaluated outcome. The base flow continues after the perturbation, and finite rollout checks establish whether the head predicts that actual continuation. Source-held-out Jacobian-vector agreement with the full-rollout teacher can test soft sensitivity, while decoded finite perturbations test chemical usefulness. Value fit by itself does not establish gradient accuracy.

If validity is the main failure, first test a smaller bounded intervention or early stopping using the existing controller. If late conflicting updates are the failure, first test a fixed earlier checkpoint rollback against the same budgets. Implement only the branch supported by the diagnostic, then compare its added training and inference cost with the original B methods. These are proposed experiments; this review makes no priority or novelty claim for the underlying techniques.

## 6. Discussion and collaboration readiness

The completed study supports a focused discussion now: useful evidence steering is measurable, expensive control loses to larger candidate pools, and retained trajectories offer a concrete constrained-revision question. A useful meeting packet contains the one-page result, paired cost curves, one explicit failure example, and the proposed mathematical question.

A substantive collaboration proposal should add reproducible B results under conflicting updates, comparable-cost conditional and sampling alternatives, retained-constraint and validity audits, and one precise mechanism question justified by those results. A favorable result is not required to ask for scientific advice. Claims of a method contribution need a reproducible improvement or an explanatory result with a clearly defined scope. Novelty assessment requires a separate primary-literature review before such claims are made.

The most concrete mathematical question is: when can a terminal sensitivity model support a small intervention that preserves the chemical mask and retained response constraints, while improving final decoded success per unit computation after a conflicting update?

## 7. Operational review checks

Before an allocated-GPU run, check that goal reversal changes the scored goal while preserving physical response; evidence correction preserves the goal; all methods share support and exclusion sets; copied category protection survives a hostile decoder; exact labels arrive after commitment; update prefixes and noise are identical where paired; model fingerprints remain fixed; and charged totals reconstruct from their components. Verify cumulative budget curves use completed attempts and retain zero-output points. After the smoke run, inspect measured timing and all retained failures before increasing the campaign count.

Source context: [ELF sampling study](05_elf_sampling_study.md), [masked-control review](10_masked_edit_control_review.md), and [prior experiment plan](../detail_operation/05_masked_edit_control_experiment_plan.md).
