# How experimental evidence changes the molecular flow

The experimental data enters as an additional input to the flow model. We do not run an experiment at each flow step. We teach the model, during training, how previous measurements should change its direction.

Our thesis is that a molecular generator can reuse learned chemical capabilities and recalibrate their use from a few measurements in a new campaign. The goal is to improve one property while preserving another, without retraining the chemical generator.

## 1. Turn measurements into a campaign summary

Suppose a campaign provides these observations. The numbers below are illustrative, not collected results.

| Measured molecule | Property 1 | Property 2 |
|---|---:|---:|
| A | 0.8 | 0.6 |
| B | 1.2 | 0.3 |
| C | 0.9 | 0.9 |

The molecular encoder describes each molecule's structure. A separate evidence encoder reads that structural description together with the two measured values:

$$
\mathcal D=\{(G_i,y_i)\}_{i=1}^{N},
\qquad
b_{\mathcal D}
=H_\psi\!\left(\{(m(G_i),y_i)\}_{i=1}^{N};c\right).
$$

Here, $m(G_i)$ is the molecule's structural description, $y_i$ contains its two measured outcomes, and $c$ is the shared campaign context. The vector $b_{\mathcal D}$ is a learned summary of the measurement table.

In our implementation, $m$ pools the frozen molecular encoder's atom and bond embeddings and includes the atom count. The evidence encoder processes each record, averages the resulting representations, and combines that average with the number of measurements. Record order does not matter. The requested goal does not enter this encoder.

The summary does not contain an explicit rule such as “add nitrogen.” Its meaning must be learned from training across different campaigns.

## 2. Use that summary to change the flow

At a particular flow step, the chemical branch produces a base prediction and several possible correction directions. The selector receives the measurement summary and the goal, and assigns strengths to those directions:

$$
\widehat X
=F_{\mathrm{chem}}(Z_t,t,G_0;n)
+B_{\mathrm{chem}}(Z_t,t,G_0;n)\,
a_\omega(Z_t,t,G_0,y_0,c,b_{\mathcal D},g,n).
$$

The symbols have the following meanings:

- $Z_t$: the current continuous, noisy molecular representation.
- $t$: generation progress, not physical molecular time.
- $G_0$: the starting molecule, supplied through its pooled description in the current model.
- $y_0$: the starting molecule's measured outcomes.
- $n$: the proposed molecule's atom count.
- $g$: the requested improvement and preservation requirements.
- $F_{\mathrm{chem}}$: the general clean-embedding prediction.
- $B_{\mathrm{chem}}$: the available continuous correction directions.
- $a_\omega$: the strengths used to combine those directions.

The chemical branch does not directly receive the measurement summary or goal. The selector does.

The clean prediction determines the velocity:

$$
v_{\mathcal D}=\frac{\widehat X-Z_t}{1-t},
\qquad t<1.
$$

The connection is therefore:

$$
\boxed{
\text{measured structures and outcomes}
\longrightarrow b_{\mathcal D}
\longrightarrow a_\omega
\longrightarrow \widehat X
\longrightarrow \text{flow direction}.
}
$$

The measurements change the coefficients. The coefficients change the continuous trajectory. No intermediate discrete molecule is required for this connection.

## 3. Teach the model what the measurements mean

During training, we repeatedly give the model a small measurement table from one campaign. We then ask it to handle other molecules from that campaign whose outcomes are withheld from its input.

Two losses teach the connection:

1. **Response loss.** Given the measurement summary, the starting molecule, and a candidate, predict the candidate's measured property changes. This teaches the evidence encoder to extract useful information from the table. Successful and unsuccessful examples both contribute.
2. **Flow loss.** For a known analogue that satisfies the requested goal, predict its clean embedding from a noisy version. This teaches the selector how to use the evidence summary to produce suitable analogues.

The flow training input is constructed from a known analogue's embedding $X$:

$$
Z_t=(1-t)\epsilon+tX,
\qquad
\epsilon\sim\mathcal N(0,I).
$$

Training samples a noise level $t$ and compares the predicted embedding with $X$. It does not need to generate and decode a complete molecule before computing the flow loss.

Goals are drawn independently of the analogue's measured outcomes. The measured outcomes determine whether that example contributes to the goal-conditioned flow loss. The original query count remains the loss denominator.

Both the response loss and flow loss backpropagate into the evidence encoder. The flow loss also trains the selector and chemical branches. A separate reconstruction loss trains the decoder on known molecular embeddings with small added noise. The molecular encoder remains frozen.

There is currently no property-gradient feedback loop inside sampling. The response predictor helps train the evidence representation and ranks final candidates; it does not directly push each intermediate flow state uphill.

## 4. Use new measurements without retraining the generator

After source training, the main deployment setting freezes all model weights.

For a new campaign:

1. Encode its measured compounds and their outcomes into $b_{\mathcal D}$.
2. Supply the starting molecule, its baseline measurements, and the requested goal.
3. Start from continuous noise and integrate the evidence-conditioned flow.
4. Decode the final representation into a molecular graph.
5. Rank candidates using the response predictor. Determine actual success using independent measured outcomes or the exact simulator.

New measurements change the input summary, not the network weights. This is learned adaptation through evidence conditioning.

## 5. The mathematical hypothesis

At the same noisy state, time, starting molecule, baseline, context, atom count, and goal, changing the measurements changes the velocity through:

$$
\boxed{
v_{\mathcal D'}-v_{\mathcal D}
=\frac{B_{\mathrm{chem}}}{1-t}
\left[
a_\omega(b_{\mathcal D'},g,\ldots)
-a_\omega(b_{\mathcal D},g,\ldots)
\right].
}
$$

The omitted arguments are held fixed in this comparison. This equation compares velocities at the same state; it does not assume that trajectories remain at the same state after they move.

This is the architectural restriction: experimental evidence steers generation through the available chemical directions. It does not directly rewrite the chemical branch.

That fits the thesis because a new campaign can select different combinations of existing directions while keeping all trained weights frozen. We test whether this separation needs fewer measurements for useful redesign than ordinary conditional generation or adaptation.

The continuous embedding provides a space in which this steering is differentiable. The factorization provides the proposed structure for transferring it. Neither guarantees that the directions represent useful chemical changes, that nearby embeddings are chemically similar, or that a direction has the same property effect in every campaign.

## 6. What the current prototype establishes

The current prototype implements these connections. It uses real small molecular graphs from QM9, with simulated property outcomes under different hidden campaign response functions. These are controlled computational observations, not biological assay measurements.

The initial study found that generated chemistry barely changed when the evidence changed, and the model rarely produced sufficiently close analogues of the starting molecule. We have built a test of the thesis; we have not demonstrated the proposed transfer.

The current model starts from noise and receives a pooled description of the starting molecule. It does not explicitly anchor the trajectory to the full starting graph. This is a plausible contributor to the locality failure, which requires a further experiment to establish.

## References in this workspace

- [Mathematical plan](../detail_operation/01_math_plan.md)
- [Execution plan](../detail_operation/02_execution_plan.md)
- [Research guide](../research_plan.md)
- [Collected results and diagnosis](../result_log/experiment_results_1012.md)
- [Model and selector implementation](../../3_model_training/emf/redesign.py)
- [Evidence encoder and response predictor](../../3_model_training/emf/response.py)
- [Training objectives](../../3_model_training/emf/redesign_objectives.py)
