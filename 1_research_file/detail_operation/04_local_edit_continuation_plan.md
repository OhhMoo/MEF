# Overnight continuation: can known responses control local molecular edits?

The research objective is experiment-responsive molecular redesign: reuse chemical knowledge, use new measurements to choose useful local changes, and preserve required structure and properties. The first implementation's $F+Ba$ factorization is one candidate mechanism. The next experiment must establish local steering when campaign information is already known.

This plan implements the user's approved continuation after the [first evidence-steering results](../result_log/experiment_results_1013.md). Source studies and the original mathematical plan remain as records of the earlier implementation.

## 1. Two failures, two interventions

The previous factorized generators barely responded to evidence, including a diagnostic with the true campaign coefficient. Ordinary conditioning responded, but nearly all property improvements fell outside the lead neighborhood.

A new code-level diagnostic found saturated selector activations in the completed factorized checkpoints. Large pooled trunk features dominated the small task inputs. This motivates a precise normalization repair, separately from the change in molecular representation.

| Initial source model | Question |
|---|---|
| Ordinary conditional flow, pooled lead, true coefficient | Reference for known-context generation |
| Ordinary conditional flow, aligned full lead and residual destination, true coefficient | Does an explicit lead-relative representation supply useful local edits? |
| Original pooled factorized flow with parameter-free pooled-input RMS normalization, true coefficient | Does correcting selector input scale restore use of context? |
| Aligned lead-relative flow, task inputs held constant, trained on all local pairs | Can chemistry alone supply local candidates for explicit response-based control? |

Train each new model with seeds 142 and 143. The normalization repair also has the completed original privileged seed-142 run as a matched historical reference. Preserve its initialization, source episodes, noise stream, objective, and 20,000 updates. Wrap the existing selector without drawing new parameters.

The lead-relative intervention changes full lead conditioning, the initial distribution, and destination parameterization together. Record it as an architectural package; this comparison does not isolate the effect of one of those changes.

## 2. Reuse a controlled task with explicit atom correspondence

Reuse the two chemical families, 294 nine-heavy-atom molecules, source/validation/test campaign splits, and exact response functions from the previous study. Preserve the goals:

$$
\Delta y_1\geq 0.25,
\qquad
\Delta y_2\geq -0.10.
$$

Map each analogue's atoms into its lead's order using chemistry alone. Minimize the number of changed atom categories plus changed bond categories over permutations, with deterministic tie handling. Cache the permutation and exact edit counts. The completed cache covers all 294 family records and round-trips to their canonical identities; observed minimum edits range from zero to four.

Only source query tensors are reordered for training. Support observations remain the same canonical molecules. The frozen encoder supplies full aligned node and pair representations. Molecular identities can recur across campaign splits: this study tests response adaptation on familiar chemistry.

Primary locality stays **Morgan similarity at least 0.4**, with the existing count rule. Exact graph-edit counts are an additional diagnostic, not a replacement chosen after observing results. Lead copies and the entire reserved 64-compound support pool do not count as new candidates, including when only four observations are supplied.

## 3. Lead-relative continuous flow

Let $X_0=E(G_0)$ and $X_1=E(G_1)$ use the aligned atom order. Expose absolute coordinates to the existing sampler:

$$
Z_t=(1-t)(X_0+\epsilon)+tX_1,
\qquad
\widehat X_t=X_0+\Delta_\theta(Z_t-X_0,t,X_0,c).
$$

The reused velocity identity is

$$
v_\theta(Z_t,t)=\frac{\widehat X_t-Z_t}{1-t}.
$$

The new residual heads start at zero, so the initial destination is the lead. Full lead node and pair tensors enter the flow. There is one terminal graph decode, with its clean prediction counted in the 32 flow evaluations. No hard graph edit is inserted during integration.

Conditional runs retain the previous goal-filtered velocity loss, original query denominator, all-query response likelihood, and noisy-clean decoder reconstruction. The chemistry-only prior uses every local pair and constant task inputs. Encoder and normalization remain frozen; the decoder remains trainable during source training.

## 4. Explicit response-based controller

Fit a small differentiable response surrogate on source observations only. Dividing source endpoint 1 by its known source coefficient supplies the unit-coefficient response. Its prediction is a difference from the lead, giving zero predicted change for the lead itself.

At each predicted clean embedding, use the two endpoint Jacobians to find the smallest correction that satisfies the two linearized response inequalities. Enumerate the zero-, one-, and two-active-constraint cases. Use a labelled damped fallback if no numerically feasible candidate exists.

Bound the correction's RMS over active latent coordinates. The two declared validation radii are **0.05 and 0.15**. Clipping and nonlinear response mean that the final decoded molecule must still be evaluated independently.

This diagnostic uses the true campaign coefficient. The surrogate does not access test outcomes for newly proposed molecules. Log clipping, fallback frequency, remaining surrogate deficits, two surrogate forward calls and two backward calls per batch correction, candidate-level correction counts, and host/device timing.

## 5. Measurements and validation gates

For each main validation run, use all eight validation campaigns, 128 attempts per campaign, the same paired Gaussian draws, and the correct versus opposite coefficient. Use the same frozen rank-8 response scorer from the previous study for selection. Evaluate proposals before ranking.

Report:

- Distinct eligible local candidates and distinct local joint-profile successes per **all** attempts.
- Correct-minus-opposite local success, property-only success, validity, and duplicate/lead/support matches.
- Signed descriptor changes under positive and negative coefficients.
- Reconstruction of aligned known analogues.
- Continuous prediction changes, decoder changes, and paired terminal identities.
- Exact edit counts and source/validation feasibility alongside the primary metric.

An initial candidate advances to measured-evidence calibration only if it meets all of these practical validation gates:

1. At least 1% distinct eligible local candidates and 25% valid supported slots in each family and seed.
2. Positive correct-minus-opposite local-profile benefit in both seeds and in each pooled family.
3. The generated descriptor change reverses in the correct direction in both families.

These are operational thresholds for allocating the next experiment. Report known feasible-library counts when a diversity gate fails; a narrowly feasible task can limit distinct output counts. Two families and two seeds support an exploratory comparison. Sampled slots and opposite-sign campaigns are not independent repetitions.

Source runs must complete the matched 20,000 updates to enter this decision. A time-capped or interrupted run is an incomplete run, not a failed scientific hypothesis. Test results are kept closed until the continuation decision is recorded.

If a source run stops just short of 20,000 updates, allow one bounded completion in a new output directory. Restore its full model, optimizer, random-number state, and cumulative update count. Change only the output path and training deadline. This decision uses completion status alone, before any validation comparison. Keep the original partial checkpoint and measurements. Only completed 20,000-update runs enter scientific selection.

## 6. Automatic continuation

### If known-context local steering works

Select the passing architecture with the most correct-evidence local successes on validation; use the declared candidate order to break ties.

Freeze the successful generator, including its evidence encoder. For a learned generator, fit a small source-only head that maps its evidence state to the scalar coefficient used during successful known-context training:

$$
\widehat s=h_\omega(H_\psi(\mathcal D)),
\qquad
b_{\mathrm{flow}}=(\widehat s,0,\ldots,0).
$$

This preserves the input meaning learned by the generator. Train the calibration head on source campaigns only, with support sizes 4, 16, and 64. Verify the generator's entire state dictionary is unchanged.

If the controlled chemistry prior wins, estimate the coefficient by least squares from the support outcomes and frozen surrogate's unit responses. This also changes only the supplied campaign state. Evaluate all three support sizes against the true-context parent. The synthetic source coefficient supervision is stated explicitly.

### If no candidate passes

Continue both anchored conditional seeds for 10,000 source updates in **matched pairs**:

- The original objective, with extra training.
- The same objective plus weight 0.1 categorical supervision on the predicted clean embedding.

Only goal-eligible queries contribute to the added loss, with the original query denominator. Both branches start from identical parent weights and use matching source/noise streams. This separates the effect of the added supervision from simply training longer.

This loss evaluates the denoiser's clean prediction on teacher-interpolated training states. It is not direct supervision of sampled terminal trajectories. The test remains whether independently generated endpoints improve.

If either continuation passes the same validation gates in both seeds, run the frozen calibration stage described above. Preserve every failed branch and its diagnostics.

### Revisability

Evidence-timing experiments follow once local steering and frozen-generator calibration work. This overnight study establishes those prerequisites. It does not interpret a timing change applied to an ineffective correction as evidence for revisability.

## 7. Execution, budget, and records

Use a new study directory: `4_experimental_result/result/local_edit_001`.

The tmux controller owns GPU submissions. It runs an allocated-GPU integration check, up to four parallel source jobs, controller diagnostics, the validation decision, conditional continuations, and final held-out evaluation. Pin code, configuration, input hashes, and resolved continuation choices.

| Stage | Maximum GPU allocation |
|---|---:|
| Original operation check and one corrected repeat | 2 × 10 minutes |
| Eight initial source runs, including validation/trace | 8 × 30 minutes |
| Completion of time-capped source runs, if needed | Up to 4 × 3 minutes |
| Source-surrogate/controller validation | 2 × 15 minutes |
| Matched repair or initial frozen calibration | Up to 4 × 20 minutes |
| Frozen calibration after a successful repair | Up to 2 × 15 minutes |
| Final source/continuation tests | Up to 12 × 4 minutes |
| Final explicit-controller tests | 2 × 10 minutes |
| **Maximum new-study total** | **8 GPU-hours** |

The previous study used 2.773 GPU-hours; it is recorded separately. The combined ceiling is 10.773 GPU-hours. No automatic submission may exceed the new-study cap or four concurrent GPUs.

The original operation check completed training, then stopped because repeated CUDA encodings of an identical lead were not bit-identical. The corrected diagnostic and evaluation reuse one full lead encoding across paired contexts. Preserve the first snapshot, failed job, and manifest in the study's revision record. The revised allocation table pays for the repeat by reducing the two controller-validation caps; the total cap stays unchanged.

Measured source throughput also projected two model types slightly beyond their internal training deadline. Reserve the completion stage by reducing final evaluation caps from five to four minutes. The original source jobs keep their pinned code and inputs; a separately pinned controller revision adopts those jobs and performs any required checkpoint completion. Both amendments remain within the eight GPU-hour cap.

Maintain the live report in [experiment_results_1013.md](../result_log/experiment_results_1013.md), with actual timestamps and links to CSVs, checkpoints, and decisions. Implementation notes go in the same day's implementation log. Summaries explain what was compared, what worked, what failed, and which continuation was selected.
