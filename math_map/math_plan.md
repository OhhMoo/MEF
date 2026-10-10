# EMF core mathematics

**Hypothesis:** molecular generation improves when continuous transport is trained without intermediate categorical supervision. Test this by changing one coefficient, $\lambda$.

Let $G=(A,B)$ be a molecule. A shared, frozen encoder preserves continuous atom and unordered pair features:

$$
X=E(G)=(X^A,X^B)\in\mathbb{R}^m.
$$

Use active coordinates only; pair features are mirrored to keep bonds symmetric. Sample atom counts from one shared training prior and hold each count fixed through its trajectory.

Draw independent Gaussian noise $\varepsilon,\xi\sim\mathcal{N}(0,I)$ and $t\sim\mathcal{U}(0,T)$, with $0<T<1$. Construct a noisy state and predict its clean destination:

$$
Z_t=(1-t)\varepsilon+tX,
\qquad
\hat{X}_t=F_\theta(Z_t,t).
$$

The destination prediction defines the transport velocity:

$$
v_\theta(Z_t,t)=\frac{\hat{X}_t-Z_t}{1-t},
\qquad
\frac{dZ_t}{dt}=X-\varepsilon.
$$

Let $q_\phi(Y)$ predict atom and bond probabilities from an embedding $Y$. Define the categorical loss:

$$
\ell_\phi(Y,G)
=\operatorname{CE}\big(q_\phi^A(Y),A\big)
+\operatorname{CE}\big(q_\phi^B(Y),B\big).
$$

CE is the average negative log probability of the correct category. Average atoms and unordered pairs separately; the bond vocabulary includes no bond.

Three losses define the first model:

$$
\begin{aligned}
\mathcal{L}_{\mathrm{flow}}
&=\mathbb{E}\left[\frac{1}{m}
\left\|v_\theta(Z_t,t)-(X-\varepsilon)\right\|_2^2\right],\\
\mathcal{L}_{\mathrm{end}}
&=\mathbb{E}\left[\ell_\phi(X+\sigma\xi,G)\right],\\
\mathcal{L}_{\mathrm{mid}}
&=\mathbb{E}\left[\ell_\phi(\hat{X}_t,G)\right].
\end{aligned}
$$

The common corruption scale $\sigma>0$ trains the decoder to tolerate imperfect embeddings. The encoder receives no gradients during flow training.

The entire comparison is one objective:

$$
\boxed{
\mathcal{L}_\lambda
=\mathcal{L}_{\mathrm{flow}}
+\mathcal{L}_{\mathrm{end}}
+\lambda\mathcal{L}_{\mathrm{mid}}.
}
$$

$\lambda=0$ reserves categorical supervision for the decoder branch. $\lambda>0$ also supervises intermediate predictions. The common loss weights above are fixed at one; only $\lambda$ changes in the initial contrast.

Both variants sample by the same continuous dynamics:

$$
z(0)\sim\mathcal{N}(0,I),
\qquad
\dot z(t)=v_\theta(z(t),t),
\qquad
\hat{G}=D_\phi\big(F_\theta(z(T),T)\big).
$$

$D_\phi$ uses $q_\phi$ to choose atom and bond categories once, then checks chemical validity. Invalid outputs return $\bot$ and count as failed attempts. Intermediate CE uses probabilities; it performs no hard graph projection.

The representation must first pass held out reconstruction:

$$
\Pr\big[D_\phi(E(G))\cong G\big]\approx1.
$$

Then compare $\lambda=0$ with tuned positive values using the same frozen encoder, model architectures and initializations, data, corruption, and measured compute budgets. A core observable is

$$
\operatorname{yield}
=\frac{\#\text{distinct valid generated molecules}}
{\text{total generation time}}.
$$

Time includes transport, endpoint prediction, decoding, and validity checks. Compare at a fixed attempted sample count, with novelty and scaffold diversity alongside yield. Either supervision policy may win.
