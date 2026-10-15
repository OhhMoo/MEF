# Overnight plan: can measurements steer reusable chemical directions?

**Status: approved for implementation and overnight execution on 12 October 2026.** The pinned study manifest and daily records track the actual run.

**Question:** With the model weights, starting molecule, goal, and noise fixed, can changing only the measurements change the generated chemistry in the experimentally correct direction?

The first deliverable is a small demonstration of this mechanism, together with a diagnosis of where it fails if it does not work. The main experiment uses familiar chemistry and new simulated response functions. Broader chemical transfer follows once useful steering works.

## 1. What we will build

Build one small task in which the same chemical change helps in one campaign and hurts in another. Both campaigns supply the same measured molecular structures, but different measured outcomes.

Train three versions of the current generator:

| Model | Purpose |
|---|---|
| Factorized flow with 8 directions | Test the current architectural hypothesis |
| Factorized flow with 32 directions | Test whether more correction capacity helps |
| Ordinary conditional flow | Test whether unrestricted conditioning learns the same behavior |

Use two source-training seeds for each version: six training runs. Reuse the frozen encoder, ELF-derived flow backbone, terminal decoder, episode machinery, and attempt ledger. Keep all source-trained weights frozen during main evaluation.

The overnight study has three parts: trace the existing checkpoint, train the reversal task, and reserve one diagnostic training run if evidence-based steering fails. The checkpoint trace runs alongside preparation and training.

## 2. Why this experiment differs from the first study

The completed study combined unfamiliar chemistry, unfamiliar response functions, sparse local analogues, and joint-property optimization. The flow supplied few local candidates, and additional evidence barely changed its output. That combination did not isolate experimental steering.

For this study:

- Use a few densely populated, coherent local chemical families.
- Keep molecular identities familiar during source training; hold out response functions.
- Train on the same fixed improvement/preservation request used in evaluation.
- Measure the complete path from measurements to decoded chemistry.
- Evaluate proposals before response-based ranking.

The first result will establish evidence-driven generation within a controlled task family. It will not establish biological assay transfer.

## 3. Part A: locate the weak link in the existing model

Use the completed factorized checkpoints and their original campaign observations. Keep checkpoint, lead, baseline, context, goal, masks, atom count, and Gaussian draws fixed.

Compare the original nested support sets at sizes 16 and 128. At selected states, also permute outcome rows among the same non-lead support molecules. Preserve the measured lead record. This intervention tests whether the model uses the structure–measurement association.

| Stage | Measurement | Interpretation |
|---|---|---|
| Measurements → evidence state | Change in $b_{\mathcal D}$ | Does the evidence encoder respond? |
| Evidence state → clean prediction | $B_{\mathrm{chem}}(a_{\mathcal D'}-a_{\mathcal D})$ | Does evidence change the actual correction? |
| Evidence → local sensitivity | Singular values of $\partial\widehat X/\partial b_{\mathcal D}$ | How many effective steering directions are used? |
| Continuous prediction → decoder | Atom/bond logits and categorical margins | Is a continuous change large enough to alter decoding? |
| Full trajectory → molecule | Paired canonical identities and valid/local/profile yields | Does the effect survive generation? |
| Evidence → response predictor | Query likelihood, response changes, and ranking changes | Does evidence help prediction even if generation stays unchanged? |

Probe $t=0.1,0.5,0.9$ using one reference trajectory and evaluate alternate evidence at the same states. Then integrate paired trajectories separately using the same noise.

Use 24 matched campaign/seed cases, 16 trajectories per case, and the three probe times. Compute exact Jacobians on a smaller predefined subset of eight cases and two trajectories each. Use float32 for Jacobian calculations and preserve active-coordinate masks.

Report changes in clean prediction when comparing times. The velocity's $1/(1-t)$ factor otherwise inflates sensitivity near the endpoint. Record velocity changes separately.

**Deliverable:** a diagram or table showing the size of the change at each stage, plus exact paired decoded outcomes. These are diagnostics of the existing model.

## 4. Part B: the main exploratory experiment

### Chemistry

Prepare up to three local families around fixed, chemically selected reference leads. Use supported neutral C/N/O/F chemistry and a fixed heavy-atom count within each family.

Build each family by deterministic enumeration of supported same-count atom substitutions and bond changes, using the existing local-search chemistry routines. Require validity, connectivity, a distinct canonical identity, and Morgan similarity of at least 0.4 to the reference lead. Family membership is defined from chemistry before looking at outcomes.

Target at least 128 distinct molecules per family. Preselect leads and their replacement order using chemistry alone. If enumeration cannot supply the planned family size, report that preparation result and the achievable evidence sizes before source training; retain the planned goals.

Use all eligible family analogues as possible training queries rather than the four nearest neighbours. This supplies competing chemical outcomes. Every query keeps its labels outside that episode's support.

Molecular identities may occur in source training and held-out campaign evaluation. That is intentional: the first question concerns transfer of experimental response information with chemical capability held familiar.

### Exact response functions

Choose two fixed descriptor-based scores $u(G)$ and $q(G)$, using source-fitted scales. They are defined independently of the learned embedding and basis.

For reference lead $G_0$, define:

$$
y_1(G;s)=s\,[u(G)-u(G_0)],
\qquad
y_2(G)=-[q(G)-q(G_0)].
$$

The hidden scalar $s$ determines whether increasing or decreasing $u$ is useful. The second endpoint introduces a preservation constraint.

For paired campaigns $+s$ and $-s$, the lead has the same baseline, $y_0=(0,0)$. Context inputs are identical. The model can infer the campaign only from measured outcomes, not from a campaign identifier, coefficient, or changed baseline.

Use a predeclared continuous range $|s|\in[0.5,2]$. Reserve intervals around four magnitudes for validation/test before drawing source coefficients. Use 32 source coefficients, four validation coefficients, and eight test coefficients forming four opposite-sign pairs. Store the exact values and split rules in evaluator metadata. Each reference lead defines a separate centered campaign block.

The first response family is deliberately simple. It establishes whether evidence can steer chemistry at all. Nonlinear response families are a later extension if this mechanism works.

### Goals and feasibility

Use one fixed request, expressed in source-scaled units:

$$
\Delta y_1\geq0.25,
\qquad
\Delta y_2\geq-0.10.
$$

Use this request in both training and evaluation. It is independent of the query outcome.

Before training, the evaluator records how many molecules in each declared family satisfy the request under each campaign. Report coverage for every case, including cases with no satisfying library molecule. The model receives only support observations.

If neither sign has competing feasible unmeasured analogues in any declared family, return a task-construction report before training. A preference-reversal experiment requires actual alternatives.

Do not choose test leads, coefficients, or thresholds after observing which cases are easiest. Report results across all declared cases and separately for the subset with at least one known feasible unmeasured library candidate. Define this stratification before training.

For the primary finite-library diagnostic, reserve the same maximum support pool at every evidence size. Feasibility is counted outside that pool. Valid novel molecules can also be scored exactly, so an empty feasible library does not establish impossibility.

### Training

Train the 8-direction, 32-direction, and ordinary conditional models on the same source graphs, episodes, goals, support/query splits, and random draws. Use support sizes 4, 16, and 64, counting the measured lead. Source training covers all three sizes.

Keep the existing objective structure:

- Goal-eligible queries contribute to velocity matching, with the original query denominator.
- Every query contributes to the full-Gaussian response likelihood.
- Known clean embeddings with small added noise train the decoder.

Start from the same compatible backbone checkpoint. Freeze the encoder and normalization; train the source-stage flow, evidence encoder, response predictor, and decoder.

Target 20,000 updates per run with a 45-minute training cap. Save periodic checkpoints. Compare the last common completed update count across models and seeds if any run hits its cap. Log source feasibility acceptance, losses, gradient norms, correction size, and early validation samples.

Use the ordinary conditional model's parameter budget corresponding to the 32-direction version, and record exact trainable counts. A gain from 32 over 8 directions can reflect additional parameters as well as rank; this first comparison measures capacity, not rank in isolation.

### Evaluation

Freeze all weights. For each family and each of four held-out $+s/-s$ pairs:

1. Use the same support molecule identities and order in both campaigns.
2. Supply their correctly measured outcomes for each campaign.
3. Keep lead, baseline, goal, atom count, and Gaussian draws identical.
4. Generate 256 attempts at each support size, using 32 total flow evaluations including the terminal clean prediction.
5. Repeat generation using the opposite campaign's measurement table while evaluating against the original campaign's truth.
6. Commit proposals before revealing query outcomes.

Changing the measured labels is the only experimental intervention in the paired comparison. Goals stay fixed.

With three families, three support sizes, eight campaigns, three models, two seeds, and two evidence conditions, this is at most 864 batches and 221,184 attempted slots. Cap concurrent GPU jobs at four.

## 5. What will count as useful evidence steering?

The principal comparison uses generated proposals before learned ranking:

$$
\text{steering benefit}
=
\text{joint-profile yield with correct evidence}
-
\text{joint-profile yield with opposite-campaign evidence}.
$$

Compute yield per attempted slot; invalid attempts count as failures. Report local joint-profile yield as the primary redesign measure and profile-only yield separately.

Also report:

| Observable | Question answered |
|---|---|
| Direction of the generated change in $u(G)$ under $+s/-s$ | Does the model reverse its chemical preference correctly? |
| Response sign/ranking accuracy on withheld library candidates | Does it infer the effects from measurements? |
| Validity, uniqueness, neighbourhood eligibility, and profile success | At which stage does it lose usable candidates? |
| Correct-evidence versus opposite-evidence yield | Does the evidence improve generation? |
| Paired decoded changes at fixed noise | Does it actually propose different chemistry? |
| Performance at 4, 16, and 64 observations | How much evidence is needed? |
| Comparison with ordinary conditional flow | Does the factorization contribute? |

After proposals are committed, report top-16 success using the same frozen response scorer across models within each source seed. Use the 8-direction model's scorer, declared before evaluation. Empty slots count as failures. This is secondary to pre-ranking yield.

Record exact oracle calls separately from support observations and selected-candidate budgets. Keep all invalid, duplicate, unsupported, support-matching, and out-of-neighbourhood attempts.

Two seeds and a few reference families support an exploratory result. Report per-family and per-campaign effects alongside aggregate uncertainty; attempts are not independent campaigns.

## 6. Part C: a reserved diagnostic if measurements do not steer

If correct evidence provides no positive pre-ranking benefit across both validation seeds, run one additional 8-direction source model in which the selector receives the true hidden coefficient instead of the inferred evidence state. Keep goals, graphs, training budget, and evaluation otherwise fixed.

This is a simulation-only privileged-context reference, not a deployable method or a mathematical performance upper bound.

| Result | Next question |
|---|---|
| Privileged context steers correctly; measured evidence does not | How should the evidence encoder and selector learn to infer and use the campaign response? |
| Both fail, despite useful decoded analogues being available | Is chemical realization, optimization, or decoding preventing steering? |
| Response predictions adapt; generated molecules do not | Where does evidence-conditioned prediction fail to become generative control? |
| Continuous predictions change; decoded chemistry stays fixed | Are corrections crossing meaningful decoder boundaries? |
| Ordinary conditioning works; factorization fails | Is the restricted correction architecture the bottleneck? |
| Both factorized versions work; conditional flow matches them | Evidence steering works here; a benefit from the factorization is still unestablished |
| Factorization improves yield under matched evidence and costs | Extend the mechanism test to new molecular families and nonlinear campaigns |

The privileged model uses the coefficient only in its separately labelled diagnostic path. The main evidence-based models never receive it.

## 7. How Astra's mathematics will be tested

At fixed state and all fixed non-evidence inputs:

$$
\frac{\partial\widehat X}{\partial b}
=B_{\mathrm{chem}}J_a,
\qquad
\frac{\partial v}{\partial b}
=\frac{B_{\mathrm{chem}}J_a}{1-t},
\qquad
\operatorname{rank}\!\left(\frac{\partial v}{\partial b}\right)\leq r.
$$

Measure the composed Jacobian and correction, not just coefficient magnitudes. Rescaling or rotating the basis can change coefficients without changing the model's prediction.

The rank bound is instantaneous. Once trajectories diverge, their states and correction spaces change. Final molecules need not obey the same rank bound. Record both same-state sensitivity and end-to-end paired outcomes.

The scalar response family tests whether the evidence route works. It gives a deliberately small requirement for correction diversity. Matching performance at ranks 8 and 32 here would not establish that eight directions suffice across broad molecular-design objectives. That stronger question requires richer response families after this mechanism is demonstrated.

A raw clean-minus-base denoising residual contains ordinary denoising uncertainty. Its projection onto the basis is not a valid standalone test of whether task-dependent chemical changes are low-dimensional.

For clarity, the implementation uses a normalized full-Gaussian negative log likelihood for response training and time-weighted velocity matching for the flow. Astra's squared-error formulas were illustrative, not the exact current losses.

## 8. Parallel work and compute

After approval of the plan, use the coordinator plus three agents. The reviewer takes an available slot after the first component handoff.

| Owner | Work | Boundary |
|---|---|---|
| Coordinator | Protocol, contracts, configs, integrated runner, source snapshots, Slurm submission, tmux monitor | Owns shared interfaces and GPU submission |
| Data/chemistry agent | Chemical families, fresh response splits, support/query records, feasibility report | Provides observations to model code and separate evaluator metadata |
| Model agent | Same-state sensitivity instrumentation, rank settings, training logs | Preserves the main F/B/evidence/goal separation |
| Sampling/evaluation agent | Paired evidence/noise replay, proposal commitments, yield and decoder-stage reports | Keeps private outcomes outside generation/ranking |
| Reviewer | Math/code review and an allocated-GPU operational test | Writes tests and reports under 5_test |

Use the existing emf environment and Slurm main L40S allocations. The scheduler currently lists L40S resources; no miyao GPU job was queued or running when this plan was prepared. Queue delay remains outside our control.

Request one L40S, one CPU, and 32 GiB host RAM per GPU job initially, with one Torch CPU thread; keep chemical preparation separate. The previous episodic study reached roughly 20,000 updates in about 30 minutes on this configuration. The new family's throughput must be measured.

| Allocation | Maximum L40S GPU-hours |
|---|---:|
| Existing-checkpoint trace | 0.5 |
| Six main training runs, at most 45 minutes each | 4.5 |
| Paired evaluation | 1.0 |
| One privileged-context diagnostic, if triggered | 1.0 |
| Integrated operation, retries, and timing reserve | 1.0 |
| **Total cap** | **8.0** |

These are proposed allocation caps. Include failed allocations in accounting. Do not launch a second parameter sweep if the mechanism fails; produce its diagnosis.

The reviewer checks support/query isolation, outcome-independent goals, identical paired lead/baseline/context/noise, frozen deployment weights, correct NFE, and oracle timing while exploration proceeds.

## 9. Outputs and morning review

Use separate paths:

- Data: 2_data_storage/processed/evidence_steering_toy/.
- Configs and scripts: 3_model_training/.
- Checkpoints, manifests, attempts, CSV/JSON, and figures: 4_experimental_result/result/evidence_steering_001/.
- Diagnostic control helpers: 4_experimental_result/control/.
- Pytest and operational reports: 5_test/.
- Daily implementation/results: 1_research_file/result_log/, with the run's MMDD suffix and actual timestamps in the body.

The morning report should contain three figures:

1. **Evidence-to-molecule trace:** which stages change, and how much?
2. **Preference reversal:** does the same generator move toward different chemistry when the measured effects reverse?
3. **Success versus measurements:** correct/opposite evidence, rank 8/rank 32/ordinary conditioning, with valid/local/profile stages separated.

The decision is whether we have demonstrated useful evidence steering, located a specific bottleneck, or found that ordinary conditioning already supplies the tested capability. Preserve the existing study and its original criteria. This plan introduces a new mechanism study and fresh response splits; it does not revise the earlier result.

## Related files

- [Architecture explanation](../study_file/09_experimental_evidence_and_vector_flow.md)
- [Core mathematical plan](01_math_plan.md)
- [Original execution plan](02_execution_plan.md)
- [Completed study and diagnosis](../result_log/experiment_results_1012.md)

## Frozen implementation choices for the first run

Two chemical families contain 161 and 133 molecules, all with nine heavy atoms. Their identities are shared across source and held-out campaigns. The fixed scores are source-library-scaled RDKit MolLogP and TPSA. The initial N/O-count construction pilot had too few competing alternatives; its observations are preserved as a discarded preparation pilot. The final descriptor definition and fresh held-out coefficient values were fixed before final feasibility inspection.

Each family has 32 source, four validation, and eight test campaigns. Validation magnitudes are 0.72 and 1.52; test magnitudes are 0.52, 0.92, 1.32, and 1.92, with both signs. Source magnitudes exclude intervals of width 0.04 on either side of those values. All 16 test cases have at least one known feasible molecule outside their fixed 64-compound support reserve; a source case with none outside that reserve is retained.

The source seeds are 142 and 143. The three models train for up to 20,000 updates each, saving checkpoints every 2,000 updates. Evaluation uses the largest completed update count common to all six runs. The privileged diagnostic, if triggered, trains to that same update count.

Primary steering benefit uses distinct unmeasured local successes per attempted slot. Raw successful slots, including repeated identities, are recorded separately. Correct and opposite evidence conditions can replay each other's committed generation because all other inputs match exactly. Reports distinguish logical comparison slots from physical sampling and oracle work.

Training jobs use at most three GPUs concurrently while the old-checkpoint trace uses one. Evaluation uses at most four. The initial Slurm wall-time commitments total 7.167 GPU-hours including the conditional privileged diagnostic, within the eight-hour study cap. Daily logs record actual job IDs and measured allocation time.
