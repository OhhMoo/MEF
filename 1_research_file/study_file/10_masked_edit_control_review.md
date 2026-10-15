# Masked local editing and terminal-response control

This review sets implementation checks for the requested A/B/C exploration. It builds on the pinned ELF model and sampler studies and the completed local-edit experiment. These are mathematical design constraints; experimental outcomes belong in the daily result records.

## 1. One aligned molecular state

Let `P` select editable atom and unordered-pair coordinates, `Q = I-P`, and `L = E(G_lead)`. The protected affine state space is `QZ = QL`. A masked target is

`X* = P E(G_candidate) + QL`.

The encoder is contextual: changing one atom can change embeddings at protected atoms and pairs. Consequently, raw candidate encodings generally lie outside this affine space even when their protected graph categories equal the lead. Train the masked decoder on the hybrid target it will actually receive.

For the three architectural packages, distinguish their initial distributions and clean destinations explicitly:

| Package | Editable initial coordinates | Clean-prediction parameterization | Protection |
|---|---|---|---|
| A: anchored edit | `P(L + epsilon)` with all active coordinates editable | `L + residual_theta` | Existing locality evaluation |
| B: masked absolute graph | `P epsilon` | `P absolute_theta + QL` | Latent projection and terminal category copying |
| C: masked anchored edit | `P(L + epsilon)` | `P(L + residual_theta) + QL` | Latent projection and terminal category copying |

B and C can represent the same endpoint distributions under a coordinate translation. Their finite-training behavior can differ through initial distribution, input representation, initialization, and optimization. Compare them as architectural packages. An additional independent edit latent is unnecessary for this experiment.

## 2. Graph preservation is a terminal category operation

At terminal decoding, copy every protected atom category and protected bond category from the aligned lead before chemistry construction. This establishes exact preservation in the declared graph representation. Decode and sanitize the assembled full graph, including the editable–protected boundary. A preserved protected subgraph can still acquire invalid valence through editable boundary bonds.

Record raw decoder violations before copying, exact protected preservation after copying, disconnected attempts, sanitization failures, and ordinary molecular locality. Keep invalid attempts in the denominator. Specify whether boundary bonds are editable; protecting internal scaffold bonds and protecting every bond incident to a scaffold atom are different constraints.

Mask choice uses the lead and source chemistry. Reuse it across the paired correct/opposite campaign conditions. A mask computed from the withheld candidate's graph differences exposes the destination during generation. For each fixed mask, count known source/validation candidates satisfying category protection, support exclusion, locality, and each response profile. This identifies regions with no feasible targets before interpreting a zero generation yield.

## 3. Project the total vector field

The exact protected-coordinate condition is

`Q(v_base + u) = 0`.

One implementation uses `v = P(v_base + u)` and restores `QZ = QL` after each numerical step. Constraining only `Qu = 0` permits base drift. For a linear property gradient `a` and allowed total derivative `b`, enforce `a^T u >= b - a^T v_base`; the base contribution belongs on the right-hand side.

ELF's pinned PyTorch sampler implements both restoration of conditioned embeddings and zeroing of their velocities in `restore_vx`. Copying protected graph categories is an additional molecular adaptation.

## 4. Steer a terminal response

For a remaining deterministic flow rollout `R(t,z)`, a useful control objective is the source-fitted response surrogate evaluated at its terminal prediction:

`J(t,z;s) = sum_k relu(threshold_k - response_k(R(t,z);s))^2`.

Differentiate through that remaining rollout, including the terminal clean predictor, or train a value surrogate for this objective. A clean-prediction correction at the current time is a myopic approximation and should be named accordingly. Detaching the terminal prediction removes the control gradient; detaching between rollout steps changes it into a truncated approximation.

For quadratic effort and editable control, unconstrained steepest descent gives `u = -eta P grad_z J`. A bounded implementation must state whether it bounds velocity, displacement per step, or total path effort. If `u` is a velocity, its state increment is `dt*u`; counting an unscaled correction at every step changes the intervention with NFE. Check the change in surrogate loss under small positive and negative finite differences before a full run.

The known campaign setting supplies the scalar response coefficient. Molecular gradients come from a surrogate trained on source observations. Exact held-out response functions score committed terminal proposals. This separation preserves the meaning of known-context versus measured-evidence calibration.

## 5. Calibration and revising an active trajectory

Freeze generator and surrogate weights before estimating campaign response parameters from support observations. Store coefficient error and support response energy as diagnostics. Compare calibrated and known coefficients using the same graph masks and Gaussian draws.

A revision experiment becomes interpretable after known-context control changes terminal outcomes. Resume the existing continuous state after new evidence arrives and compare it with a restart using the new evidence. Count the prefix work already spent by the revised trajectory and all new work by both policies. Report final success against full elapsed time and flow evaluations, including rollout evaluations used to compute control gradients.

## 6. Operational acceptance checks

Independent checks should establish protected state invariance at every integration stage, terminal protected category equality under a deliberately hostile decoder, boundary invalidity retention, finite nonzero terminal gradients, agreement with finite differences, source-only calibration, frozen generator state, exact paired Gaussian draws, and complete cost ledgers. The allocated-GPU check exercises these routes at small scale before the overnight source fits.

Count the terminal clean prediction in NFE. Count additional differentiable rollout forward calls and backward calls separately. Include masking, decoding, host chemistry, surrogate fitting, calibration, and failed attempts in the relevant wall-time and allocation budgets. Equal nominal generation NFE does not imply equal work when one method uses lookahead control.

Sources: [ELF model study](03_elf_model_study.md), [ELF sampling study](05_elf_sampling_study.md), [local edit continuation](../detail_operation/04_local_edit_continuation_plan.md), and pinned upstream `0_external_repo/ELF_pytorch/src/utils/sampling_utils.py` (`restore_vx`, `net_out_to_v_x`).
