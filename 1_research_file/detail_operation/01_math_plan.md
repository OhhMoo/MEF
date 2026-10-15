# Experiment-Calibrated Molecular Flows — core mathematics

**Question:** Can we transfer the ability to redesign molecules by recalibrating the effects of reusable chemical changes from a few measurements?

**Hypothesis:** New campaign evidence can change how a frozen chemical generator selects its available changes. Test the separation between **chemical capability** and **experimental response**.

## 1. Measured changes and the joint design goal

Let $G_0$ be a lead molecule and $c$ the assay context: the two endpoint definitions, targets, and experimental conditions. Write its measured endpoint values as $y_0\in\mathbb R^2$. Orient both endpoints so that larger values are better. Use fixed log scales after common unit conversion for positive concentration-valued endpoints; other endpoint types use a documented common scale. Retain assay context when comparing changes across campaigns.

A measured lead–analogue pair supplies

$$
\rho=(G_0,G_1,c,y_0,y_1),
\qquad
\delta=y_1-y_0.
$$

$\delta_1$ is the change in the endpoint we want to improve. $\delta_2$ is the change in the endpoint we want to preserve. These are changes in measured assay outcomes; their uncertainty includes measurement noise.

The goal $g=(\kappa,\tau)$ requests an improvement of at least $\kappa>0$ in endpoint 1 and an acceptable final value of at least $\tau$ in endpoint 2:

$$
\boxed{
\mathcal A_g(y_0)
=
\left\{\delta:
\delta_1\ge\kappa,\quad
y_{0,2}+\delta_2\ge\tau
\right\}.
}
$$

The lead starts with $y_{0,2}\ge\tau$. Preservation permits a loss only while endpoint 2 stays acceptable. Its absolute threshold requires the measured baseline $y_{0,2}$.

## 2. Evidence calibrates expected effects

A campaign's records share one context $c$ and the same endpoint scales. The first model uses compounds with both endpoints observed. Retain partial or censored records in raw data and report complete-case coverage. A pretrained set encoder infers a response state:

$$
\mathcal D=\{(G_i,c,y_i)\}_{i=1}^{N},
\qquad
\boxed{b_{\mathcal D}=H_\psi(\mathcal D).}
$$

$N$ counts distinct measured compounds, including the lead. $b_{\mathcal D}$ summarizes their chemical responses. $H_\psi$ is invariant to record order. It receives observations and assay context; it receives no requested goal or learned campaign-ID lookup.

A response model predicts the two endpoint changes jointly:

$$
p_\eta\!\left(
\delta\mid G_0,y_0,G,c,b_{\mathcal D}
\right).
$$

This is a normalized joint density, so it can represent dependence between the endpoints. It predicts the outcome of testing candidate $G$ against the fixed observed lead baseline. Its parameters will be learned across source campaigns.

For a fixed candidate, context, baseline, and evidence, there is no computational path from the requested goal to $H_\psi$ or $p_\eta$. Changing that goal therefore leaves the response prediction unchanged.

**Measurements change expectations. Goals change which chemistry we request.** This is an information-routing rule; it does not establish that the response predictions are accurate or calibrated.

## 3. Reusable chemistry inside the denoiser

Retain the existing normalized graph representation and terminal decoder:

$$
X=E(G)\in\mathbb R^{m(n)},
\qquad
D_\phi(X)\in\{\text{molecular graphs}\}\cup\{\bot\}.
$$

$n$ is the candidate's heavy-atom count. $E$ produces atom and unordered-pair embeddings; the pair vocabulary includes the no-bond category. All norms use active coordinates. The count and its mask remain fixed during a trajectory. Draw candidate counts conditional on the lead count:

$$
n_0=|V(G_0)|,
\qquad
n\sim p_{\mathrm{count}}(\,\cdot\mid n_0).
$$

Fit and freeze $p_{\mathrm{count}}$ on source lead–analogue pairs before goal filtering. This is a lead-conditioned proposal law; the flow target below is defined for each fixed $n$.

Reconstruction warm-up establishes $E$ and the decoder. Fit normalization on source training molecules and then freeze $E$ and its normalization. The decoder remains trainable during source-model training.

Collect the supplied query information as

$$
h=(G_0,y_0,c,b_{\mathcal D},g,n).
$$

Draw independent Gaussian noise and construct the continuous path to the measured query molecule:

$$
\varepsilon\sim\mathcal N(0,I),
\qquad
Z_t=(1-t)\varepsilon+tE(G_1),
\qquad
t\sim\mathcal U(0,T),\quad 0<T<1.
$$

The proposed clean prediction is

$$
\boxed{
\widehat X_\Theta(Z_t,t;h)
=
F_\chi(Z_t,t,G_0;n)
+
B_\chi(Z_t,t,G_0;n)\,
a_\omega(Z_t,t;h).
}
$$

Here $F_\chi$ is an unrestricted chemical denoiser, $B_\chi\in\mathbb R^{m(n)\times r}$ supplies $r$ coordinated directions, and $a_\omega\in\mathbb R^r$ selects their strengths.

$F_\chi$ and $B_\chi$ receive no direct campaign measurements, assay context, or requested goal. The selector receives the response state and goal. Its dependence on $b_{\mathcal D}$ is the proposed architectural mechanism for recalibration.

The restriction is local: the task-dependent correction lies in the span of $B_\chi$ at the current state. The whole denoiser is not restricted to rank $r$. Frozen functions can produce different directions at different molecules and times. Their columns need not correspond to named chemical edits.

The current state $Z_t$ can indirectly carry information about the goal during sampling. Excluding direct inputs therefore does not establish statistical independence of the directions from the task.

## 4. Learn the separation across source campaigns

Each training episode draws a support set $\mathcal D$ and a measured query pair $\rho$ from one source campaign. The lead measurement is available; the query analogue's outcome is hidden from $H_\psi$.

Draw support sizes from $\{16,32,64,128\}$ so source training covers deployment.

Split original compound measurements before constructing pairs. A hidden query outcome must not reappear in support through a reversed pair, another pair, or an equivalent assay record. Constructing many pairs does not create more independent measurements.

Draw goals from a fixed prior independently of the query analogue and its outcomes, conditional on the supplied lead information. Retain the pairs that satisfy the request through a flow-loss weight:

$$
g\sim\pi(\,\cdot\mid y_0,c),
\qquad
\operatorname{supp}\pi(\,\cdot\mid y_0,c)
\subseteq
\{(\kappa,\tau):\tau\le y_{0,2}\},
\qquad
I_g(\delta)=\mathbf{1}\{\delta\in\mathcal A_g(y_0)\}.
$$

The prior covers deployment requests and can include signed $\kappa$ during training. Improvement at deployment uses $\kappa>0$. Query outcomes determine $I_g$, while staying outside the support state. The response loss uses all query changes, including unsuccessful modifications.

Write $u=(G_0,y_0,c,b_{\mathcal D},n)$, and let $p_{\mathrm{pair}}$ be the original source-pair law. Where the goal has positive feasibility probability, the weighted flow targets

$$
\boxed{
p_{\mathrm{goal}}(G_1\mid u,g)
=
\frac{
p_{\mathrm{pair}}(G_1\mid u)\,
\Pr_{\mathrm{pair}}\{\delta\in\mathcal A_g(y_0)\mid G_1,u\}
}{
\Pr_{\mathrm{pair}}\{\delta\in\mathcal A_g(y_0)\mid u\}
}.
}
$$

This is the source analogue distribution conditioned on satisfying both requirements. It also handles noisy outcomes. An outcome-derived goal rule would instead weight analogues by its goal density.

Let $\ell_\phi(Y,G)$ be the sum of atom and unordered-pair cross-entropies, each averaged within its group. With independent decoder noise $\xi\sim\mathcal N(0,I)$, train

$$
\begin{aligned}
\mathcal L_{\mathrm{flow}}
&=
\mathbb E\!\left[
I_g(\delta)\,
\frac{
\|\widehat X_\Theta(Z_t,t;h)-E(G_1)\|_2^2
}{
m(n)(1-t)^2
}
\right],
\\
\mathcal L_{\mathrm{response}}
&=
-\mathbb E\!\left[
\log p_\eta\!\left(
\delta\mid G_0,y_0,G_1,c,H_\psi(\mathcal D)
\right)
\right],
\\
\mathcal L_{\mathrm{decode}}
&=
\mathbb E_{G,\xi}\!\left[
\ell_\phi\!\left(E(G)+\sigma\xi,G\right)
\right],
\\[2pt]
\mathcal L
&=
\mathcal L_{\mathrm{flow}}
+
\beta\mathcal L_{\mathrm{response}}
+
\gamma\mathcal L_{\mathrm{decode}},
\qquad \beta,\gamma>0.
\end{aligned}
$$

Learn $\Theta=(\chi,\omega,\psi,\eta,\phi)$ on source data, with $E$ fixed. The support state receives both flow and response learning signals: it must help predict experimental changes and help select chemical directions. These functions can share network components; the notation does not require separately trained modules.

The flow loss is velocity matching because

$$
v_\Theta(Z_t,t;h)
=
\frac{\widehat X_\Theta(Z_t,t;h)-Z_t}{1-t},
\qquad
v_\Theta-(E(G_1)-\varepsilon)
=
\frac{\widehat X_\Theta-E(G_1)}{1-t}.
$$

For a fixed support encoder, the unrestricted clean-prediction optimum is

$$
F^\star(z,t;h)
=
\mathbb E\!\left[
E(G_1)\mid Z_t=z,t,h,\ I_g(\delta)=1
\right].
$$

The proposed factorization may not attain this optimum. Its value must come from more useful transfer under limited evidence, rather than greater unrestricted expressivity.

## 5. Recalibrate a new campaign without retraining

Freeze the pretrained weights, encoder, decoder, and normalization. Infer only the response state from the new measurements:

$$
\Theta=\Theta_{\mathrm{source}},
\qquad
b_{\mathcal D}=H_{\psi_{\mathrm{source}}}(\mathcal D).
$$

The directions, selector, and response model are all pretrained functions. As measurements accumulate, $\mathcal D$ changes and therefore $b_{\mathcal D}$ changes. No campaign-specific weight update is part of this first model.

For each sampled candidate count, generate by the same continuous dynamics:

$$
z(0)\sim\mathcal N(0,I),
\qquad
\dot z(t)=v_\Theta(z(t),t;h),
\quad 0\le t\le T,
\qquad
\widehat G
=
D_\phi\!\left(
\widehat X_\Theta(z(T),T;h)
\right).
$$

Decode the graph once. Count the terminal clean prediction as a flow evaluation. Invalid or disconnected outputs are failed attempts and remain in the attempted-sample budget.

The final prediction satisfies $\widehat X_\Theta(z(T),T;h)=z(T)+(1-T)v_\Theta(z(T),T;h)$. It is one final Euler extrapolation. At the unrestricted optimum it is a conditional mean at $T$, not an exact clean-data draw.

Score each valid proposal by its predicted probability of meeting the complete profile:

$$
S_\eta(G;\mathcal D,G_0,y_0,c,g)
=
\int_{\mathcal A_g(y_0)}
p_\eta\!\left(
\delta\mid G_0,y_0,G,c,b_{\mathcal D}
\right)\,d\delta,
\qquad
\mathcal T_k
=
\operatorname{Top}_k(\mathcal C_{\mathrm{eligible}},S_\eta).
$$

$\mathcal C_{\mathrm{eligible}}$ contains distinct valid connected candidates outside support and satisfying the declared lead-neighbourhood rule, under a fixed attempted-sample budget. $\mathcal T_k$ contains up to $k$ highest-scoring compounds, with a fixed tie rule. An empty candidate set returns an empty selection.

For retrospective evaluation, restrict candidates to a library with both outcomes measured and hidden until selection. An exact simulator can score novel graphs in the controlled computational test. Prospective generated compounds receive truth from independent measurements. Write $\delta_{\mathrm{eval}}(G)$ for that external outcome difference, or $\bot$ if testing fails to supply both outcomes; $\bot$ lies outside $\mathcal A_g(y_0)$ and counts as failure. $p_\eta$ supplies predicted scores.

Define the observed success and efficiency curves on held-out campaigns:

$$
\begin{aligned}
\operatorname{success}(N,k)
&=
\mathbb E_{\mathrm{test}}\!\left[
\frac{1}{k}
\sum_{G\in\mathcal T_k}
\mathbf{1}\{\delta_{\mathrm{eval}}(G)\in\mathcal A_g(y_0)\}
\right],
\\
\operatorname{efficiency}(N,k)
&=
\frac{k\,\operatorname{success}(N,k)}{N+k},
\qquad
N\in\{16,32,64,128\},\quad k\in\mathbb N_{\ge1}.
\end{aligned}
$$

Average over held-out campaigns, leads, and sampling seeds. Unfilled selection slots count as failures. $N+k$ is the allocated compound budget, with the lead counted in $N$; report actual compounds tested and endpoint measurements separately. The toy pilot's full-pool oracle calls are an additional evaluation cost.

**The scientific claim to test:** Evidence-conditioned selection inside a reusable chemical flow improves measured joint-profile success on unfamiliar campaigns. The rank restriction, frozen weights, goal–evidence separation, and response scoring are proposed mechanisms. None alone establishes transfer or novelty.
