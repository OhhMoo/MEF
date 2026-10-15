# Masked molecular edits and terminal-response control

We test whether preserving the permitted edit region solves the shortage of local analogues, and whether controlling the final response makes experimental evidence useful during generation. The first overnight study uses nine-heavy-atom molecules and an exact simulated response. It tests the mechanism before biological generalization.

## The questions and runs

| Stage | Comparison | What it establishes |
|---|---|---|
| A/B/C | Lead-relative flow; masked absolute graph flow; masked lead-relative flow | Which representation supplies distinct valid local candidates? |
| D1 | Unguided sampling; known-coefficient conditional flow; terminal surrogate control | Can a known campaign response produce useful, structurally permitted changes? |
| D2 | Exact-response search over velocity controls; exact-response ranking of unguided samples | Is the main obstacle response approximation or reachable molecular edits? |
| E | Calibration from 4, 8, and 16 measured compounds | Can a frozen generator use a response inferred from sparse observations? |
| F | Continue after an evidence update; restart; all evidence from the start | Does keeping the intermediate state help, once useful steering works? |

A/B/C each receive 20,000 updates with seeds 142, 143, and 144. Their parameter counts, molecular pair draws, batch sizes, decoder training and evaluation noise are matched. All see the complete aligned lead and the allowed edit mask. A can change every coordinate. B and C clamp the protected coordinates and terminal categories. B starts editable coordinates at Gaussian noise and predicts absolute destinations; A/C start around the lead and predict residual destinations. This compares architecture packages. B and C are coordinate-related models, not evidence that an extra independent edit latent helps.

The best representation is selected from validation structural yield averaged over seeds. Test results do not select representations, controller settings or continuations. An additional 10,000-update conditional flow per selected seed receives the true six response coefficients; this is a known-context diagnostic. The prior remains frozen for control experiments.

## What a mask guarantees

Let $M=1$ on editable atom and bond coordinates. B/C use the training target

$$X^*=(1-M)\odot E(G_0)+M\odot E(G_1).$$

This hybrid is necessary because the frozen encoder is contextual: an unchanged graph category can have a changed embedding when its neighbours change. Training the decoder on the hybrid makes that distinction explicit.

During generation,

$$\dot Z_t=M\odot\left(v_{\mathrm{chem}}(Z_t,t;G_0,M)+u_t\right).$$

The sampler clamps protected latent coordinates, and the decoder copies their original atom/bond categories, including absent bonds. Boundary valence is checked independently. Invalid and disconnected attempts remain in every denominator. The mask is fixed from lead topology before any response or query is observed.

## Data and experimental evidence

The existing 294-molecule, two-family library is split into 182 source, 56 validation and 56 test molecules. Four fixed lead/mask tasks allow changes within either side of each lead. Source training uses only compatible source analogues. The inherited reconstruction/flow initializer may already have seen these structures; the new split establishes exclusion from this study's source updates, not unseen-chemistry pretraining.

Define source-scaled descriptors $u=\mathrm{MolLogP}/s_u$ and $q=\mathrm{TPSA}/s_q$, and

$$\phi(G)=(u,q,uq),\qquad \Delta y=W[\phi(G)-\phi(G_0)],\quad W\in\mathbb R^{2\times3}.$$

Both response rows vary across campaigns. The requested deltas are $\Delta y_1\ge0.25$ and $\Delta y_2\ge-0.1$. Observations contain Gaussian noise with standard deviation 0.05. Training, validation and test coefficient matrices are independent draws. The campaign distribution is conditioned on at least two feasible source analogues; the feasibility rule uses source chemistry only. Empty held-out feasible sets remain in evaluation and are reported.

Support sets are nested at 4, 8 and 16 distinct measured compounds, including the lead. All sizes exclude the same maximal support reserve from candidate success. A ridge response fit uses only support structures and noisy measurements. Its known three-feature basis makes this a calibration diagnostic, not a learned biological response model. Report basis rank, conditioning and response error. Subtracting the noisy lead creates correlated errors; ridge is the initial estimator.

## Controlling the terminal outcome

A frozen, source-trained graph critic predicts scaled descriptors from terminal decoder probabilities. Its data include hard source graphs, decoded hybrid targets, generated terminal states and controlled source perturbations. Labels refer to the actual decoded molecule. Invalid samples are counted, and prediction error is audited on separate source-derived samples. No campaign test outcome trains the critic.

Two time bins parameterize a velocity intervention. Differentiate through the complete rollout to obtain the Jacobian of predicted terminal response with respect to this control. A small sequential quadratic program proposes a minimum-norm correction satisfying linearized response requirements, clips each proposed increment to a declared trust radius, then reruns the full trajectory. Accept a step only when the predicted joint violation decreases. These local surrogate constraints do not guarantee final measured properties.

Every candidate is committed before the independent exact evaluator scores it. The final Euler step includes $(1-T)u_T$. Reports count all rollout calls, terminal predictions, backward traversals and elapsed host chemistry time; intervention strength is measured in velocity coordinates.

Compare correct coefficients/evidence with first-endpoint-reversed coefficients/evidence while retaining the second endpoint and paired initial noise. Also compare with an unguided pool ranked using the same response information. Extra control computation is reported and a larger unguided pool is included where practical.

A separate, explicitly privileged oracle diagnostic searches two-bin velocity coefficients in an eight-dimensional source-edit PCA basis using up to 16 exact terminal-response calls. Its comparator uses the same number of exact calls to rank independent unguided samples, sharing the initial baseline draw. All oracle queries count; this arm is never reported as blind few-shot performance.

## Budgets, metrics and decisions

A/B/C validation uses 256 physical draws per mask and 32 function evaluations per draw. Reusing those molecules across campaigns is recorded as logical rescoring, not extra independent generation. Control begins at 16 evaluations per rollout, three seeds, four masks, and both coefficient orientations. Surrogate control settings are selected using validation only; the selected setting is evaluated once on test campaigns at support sizes 4/8/16. Known-context and few-shot results stay separate.

Primary structural outcomes are distinct valid nonlead molecules respecting the mask per all attempts, exact protected-category preservation, edit distance, known analogue coverage and target/hybrid reconstruction. Primary response outcomes are distinct joint-profile successes excluding the common support reserve, top-16 success and fill rate, and success excluding source compounds. Also report validity, lead copies, diversity, original fingerprint locality and global unique structures. A known local-library selector and direct constrained mutations are strong simple comparisons. No unused top-16 slot counts as a success.

Continuation decisions are fixed before test access:

1. Train all nine source models. If a run is interrupted, preserve it and explicitly complete its remaining update budget; do not compare unequal update counts silently.
2. Fit critics and run known-context surrogate/oracle diagnostics for the best structural representation. Keep all failed arms and prediction audits.
3. If critic error is large, perform one source-only critic refinement with additional source perturbations; keep the original result.
4. Run sparse calibration and simple library/mutation baselines regardless of the steering outcome, to separate prediction from action.
5. Run revision only if correct evidence improves validation decoded joint success over opposite evidence and unguided generation across the selected seeds. Otherwise record why revision was deferred. For revision, use nested evidence, continuation times near 0.25/0.5/0.75, restart and all-evidence references. Record prefix, optimization and remaining-rollout costs; report performance at comparable full compute budgets.

The scheduler uses at most eight simultaneous L40S GPUs initially, with a planned ceiling of 48 allocated GPU hours. This is an operating budget, not a claim that more computation is scientifically useful. Each stage has its own task file, checkpoint, logs and completion marker. Tmux runs the coordinator and live report. Source and input hashes are pinned before long jobs; later integration uses a separately recorded snapshot.

## Records

- Runs and machine-readable results: `4_experimental_result/result/masked_control_001/`.
- Daily implementation and findings: `1_research_file/result_log/implementation_log_1013.md` and `experiment_results_1013.md`, appended without replacing earlier studies.
- Tests and independent CPU/GPU review: `5_test/`.
- Design review: `1_research_file/study_file/10_masked_edit_control_review.md`.

The work uses Astra for the hard design and independent review, 6-sol for model/data/control implementation, and Luna for live reporting. Actual run timestamps accompany the requested daily date label.
