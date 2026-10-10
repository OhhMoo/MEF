# EMF — Embedded Molecular Flows
## Research concept, rapid feasibility sprint, publication directions, and faculty outreach

**Working research brief | 7 October 2026 | Status: hypothesis, not an established result**

> **Core idea:** Can we learn molecular generative dynamics in an unconstrained continuous representation and postpone atom/bond discretization until a structured final decoding operation? If so, when does this improve chemical validity, diversity, conditioning, or sampling efficiency relative to intermediate discrete supervision?

## 1. Motivation: why this matters

Molecules combine discrete variables (atoms, bonds, charges, stereochemistry) with continuous structure (geometry and physicochemical properties). Molecular generation is usually approached with discrete graph editing, token generation, hybrid categorical/continuous diffusion, or continuous latent generation plus a decoder. Existing methods already address parts of this challenge, so **continuous embeddings or flow matching alone are not novel claims**.

The specific proposed innovation is an experimentally testable *discretization policy*: let continuous denoising/transport evolve with minimal intermediate categorical commitments; use a chemically structured terminal decoder to produce feasible molecular graphs. The motivating use case is **activity-preserving lead optimization and scaffold hopping**, where a useful generated molecule must meet several constraints and may need to move between distant graph structures.

**Key distinction:** The first research question is methodological, *not* whether EMF discovers drugs. Drug-discovery utility requires a later, independent validation stage.

### Inspiration from ELF

The uploaded paper **Hu et al., *ELF: Embedded Language Flows*, arXiv:2605.10938v2 (2026)** performs flow matching in continuous contextual text embeddings, using a shared denoising/decoding network and token decoding only at the end. It reports advantages in language-generation comparisons. Its results **do not establish** comparable benefits in chemistry. Specific transferable choices: x-prediction, a frozen contextual encoder, shared denoiser/decoder modes, separate final-step corruption during decoder training, bottleneck ablations, self-conditioning and stochastic sampling. Paper: https://arxiv.org/abs/2605.10938 ; code: https://github.com/lillian039/ELF .

## 2. One primary research question

**Does delaying discrete atom/bond supervision until terminal decoding improve the quality–efficiency–validity trade-off of molecular graph flow matching, compared with an otherwise matched model trained with intermediate categorical supervision?**

Subquestions (only after the main comparison works):

1. Does it help more as graph constraints become tighter or chemical scaffolds become more diverse?
2. Does terminal-only decoding create a validity bottleneck or distributional bias?
3. Can a learned or scheduled *adaptive discretization* do better than both extremes?
4. Can initial-molecule conditioning improve scaffold hopping and multi-property optimization?

**Falsifiability:** If a matched per-step-supervision baseline consistently outperforms final-only decoding across several seeds and budgets, the core assumption needs revision. A negative result is useful; it should not be relabeled as success.

## 3. Foundation: the smallest EMF worth building

### 3.1 Start with 2D graphs, not a 3D protein-conditioned model

Use small molecular graphs to isolate the architectural effect. First choose a restricted element/charge vocabulary and fixed or padded maximum atom count; define explicit masks and permutation handling. Start with QM9 for debugging (limited chemistry), then move promptly to a filtered, more diverse ZINC subset for a less artificial test. Keep stereochemistry and complex charges out of the initial sprint, and disclose those restrictions.

**Data representations**

- Molecular graph: $G=(A,B)$, atoms $A$, symmetric pairwise bonds $B$, atom-count/padding mask $M$.
- Frozen contextual encoder: $X=E(G)\in\mathbb{R}^{N\times d}$, using a small graph neural network/graph Transformer; ideally pretrain reconstruction or use a suitable pretrained encoder if available. Keep encoder identical across comparisons.
- Flow state: $Z_t=tX+(1-t)\varepsilon$, $\varepsilon\sim\mathcal{N}(0,I)$, with optional normalization.
- Shared graph-aware denoiser: $\hat{X}=F_\theta(Z_t,t,\mathrm{mode})$; start with a small Transformer/GNN, not a foundation-scale network.
- Terminal structured decoder: $D_\phi(\hat{X},M)\rightarrow(\hat{A},\hat{B})$, producing atom/bond logits jointly or using a constraint-aware decoding procedure. Atom and bond argmax **alone** do not guarantee chemically valid structures.

**Continuous training branch**

$$
\mathcal{L}_{\mathrm{flow}}
=\mathbb{E}_{G,t,\varepsilon}\left[
\frac{\left\|F_\theta(Z_t,t,\texttt{denoise})-X\right\|_2^2}{(1-t)^2}
\right]
$$

Use numerically safe clipping/truncation near $t=1$, or another principled loss weighting, and keep it identical across variants.

**Terminal decoder branch**

$$
\mathcal{L}_{\mathrm{decode}}
=\mathcal{L}_{\mathrm{atom\text{-}CE}}
+\lambda_B\mathcal{L}_{\mathrm{bond\text{-}CE}}
+\lambda_V\mathcal{L}_{\mathrm{feasibility}}.
$$

Train terminal decoding on both corrupted encoded molecules and, if needed, realistic denoiser outputs to reduce train/inference mismatch. Consider a separate decoder first as a diagnostic, *then* test ELF-like shared weights. Do not conflate benefits from final-only supervision with benefits from architecture or encoder changes.

**Sampling**: draw continuous noise, integrate a simple Euler/Heun flow for a specified number of function evaluations, perform one terminal graph decoding, then sanitize and measure outcomes. Rejection/repair rates and computational costs **must be reported**, not hidden.

### 3.2 Critically important controls

- **Variant A — Per-step categorical supervision:** same encoder, denoiser, decoder, and compute budget; add atom/bond CE at sampled intermediate times (or equivalent token-aligned projection objective).
- **Variant B — EMF terminal-only:** identical architecture and training allocation, categorical loss only in terminal decoding mode.
- **Variant C — Adaptive/scheduled:** progressively activate supervised decoding later in the flow; only if A/B show an informative difference.
- **External reference:** a credible existing molecular diffusion or graph-flow implementation, using its recommended evaluation protocol. Treat this as an additional external benchmark, not a substitute for controlled A/B tests.

**Fairness problem:** An 80/20 denoise/decode mixture is not automatically budget-matched to per-step CE; explicitly match *network forward passes, training FLOPs, trainable parameters, encoder pretraining budget, data and number of samples*. Report more than one loss weight, because apparently positive results may be artifacts of tuning one baseline poorly.

## 4. Fast toy-model evidence: 7–10-day sprint

### Days 1–2 — implement the simplest credible control

- Data cleaning, canonicalization, graph tensorization, symmetric bonds, atom masking, evaluator.
- Tiny encoder/decoder reconstruction test on held-out molecular graphs; quantify exact reconstruction, validity and graph size failures.
- Baseline flow and numerical sampler work end-to-end.

**Exit criterion:** at least a working graph reconstruction/sampling pipeline and repeatable metrics. If this fails, don't run expensive flow sweeps.

### Days 3–5 — isolate the hypothesis

- Train **A vs B** at one modest model size and several sampling budgets (e.g. 8, 16, 32, 64 network evaluations).
- Log fixed seeds and identical train/validation splits.
- Run a small parameter/loss-weight sweep; at least 3 independent training seeds for a credible initial directional result when practical.

### Days 6–7 — generate outreach-grade results

Make **three figures**:

1. **Validity and valid+unique yield vs number of function evaluations**, counting terminal repair/rejection cost.
2. **Distributional quality vs diversity:** novelty, uniqueness, scaffold diversity, and a chemistry-relevant distributional metric (e.g. FCD) with uncertainty bars.
3. **Ablation isolating discretization:** A vs B, ideally across two datasets or chemical-complexity regimes.

Optional: failure-case montage showing invalid-valence motifs, disconnected graphs, missed chemotypes, and output graph-size distribution.

### Days 8–10 — test whether the effect survives scrutiny

- Replicate promising result on a more diverse subset and/or larger maximum molecular size.
- Measure memory, actual wall-clock time and effective molecule yield per GPU-hour.
- Write a 2-page research memo and tidy public/private repo with instructions, configurations, weights if shareable, and raw results.

**A 7-day sprint is for evidence of feasibility, not proof of chemical usefulness.** Training time depends on available GPUs and the chosen implementation; the calendar is deliberately aggressive.

### Go / pivot / stop

| Observation | Next action |
|---|---|
| B beats A consistently on useful quality–validity–cost frontiers | Seek collaboration and scale carefully. |
| B improves novelty/diversity but decreases valid output yield | Study structured final decoder and adaptive constraint enforcement. |
| B matches A with materially lower measured compute | Investigate efficiency contribution, verify full costs. |
| A wins robustly across tuned controls | Reassess premise, publish/report negative evidence only if methodologically insightful. |
| Results depend entirely on reconstruction errors or baseline tuning | Fix confounder before outreach claims. |

## 5. What the initial model **does not** demonstrate

- That generated molecules are active or selective against any biological target.
- That scaffold-hopped compounds retain biological activity.
- That chemical validity implies synthesizability, stability or safe ADMET.
- That noise-space interpolation corresponds to realistic medicinal-chemistry edits.
- That the model generalizes to previously unseen proteins.

Those require appropriate labeled datasets and, ideally, prospective synthesis and assay validation.

## 6. Two publication narratives

### Track A — ICML 2027: fundamental generative modeling

**Candidate titles**

- *EMF: Delayed Discretization for Flow Matching on Chemically Constrained Graphs*
- *When Should Generative Flows Become Discrete? A Study of Molecular Graph Generation*

**Core claim if validated:** An explicit change in when categorical decisions are enforced yields a better quality/validity/compute frontier in generative modeling of constrained graphs; characterize when it helps and why. The strongest version may propose **adaptive discretization with a measurable criterion** and evaluate outside chemistry (e.g. synthetic constrained graphs) to establish generality.

**Required evidence beyond toy:** strong baselines, matched compute, clear ablations, robust multi-seed results, failure analysis, scaling/generalization, mathematical or analytical characterization of terminal projection error and constraints. A convincing paper should not depend only on a small QM9 improvement.

**Timing:** ICML **2027 submission deadline is not verified as of this brief**. The official ICML 2026 abstract/paper deadlines were January 23/28, 2026; plan provisionally for January 2027 while checking the 2027 call when published. Official ICML 2026: https://icml.cc/Conferences/2026/Dates .

### Track B — Nature Machine Intelligence: AI-driven molecular design

**Candidate titles**

- *Embedded Molecular Flows Enable Activity-Preserving Exploration Across Chemical Scaffolds*
- *Continuous Molecular Transport for Constrained Lead Optimization*

**Core claim if validated:** An ML architecture enables a meaningful advance in medicinal-chemistry decisions: better active and synthesizable scaffold alternatives, multi-parameter optimization under sparse assay data, or higher prospective hit yield at matched experimental budget. The contribution must go beyond a clever molecular representation.

**Evidence beyond ICML-style benchmarking:** target-relevant public experimental assay data; held-out time/series/target splits where appropriate; potency/selectivity and property tradeoffs; synthesis feasibility audited by chemists; diverse nontrivial baselines; uncertainty and reproducibility; and ideally prospective synthesis + biological measurements with collaborators. A toy model alone is **not** a Nature Machine Intelligence result.

**Positioning caution:** Nature Machine Intelligence published a 2026 review of flow matching across bioinformatics and computational biology: https://www.nature.com/articles/s42256-026-01220-0 . Merely applying flow matching to molecules is not sufficient differentiation.

**Relationship between tracks:** Treat these as *alternative initial manuscript narratives*, not simultaneous submissions of overlapping work. A later substantially distinct paper may be possible if the new scientific contribution and venue policies justify it.

## 7. Proposed schedule: early outreach, not months of solo development

| Date (2026–27) | Objective / evidence to show |
|---|---|
| **Oct 7–9** | Narrow question; build graph validity/evaluation pipeline; pick A/B definitions. |
| **Oct 10–13** | Obtain first usable A/B curves; debug controls. |
| **Oct 14–17** | Three-seed confirmation or transparent preliminary results; two-page memo and 2–3 figures. |
| **Oct 16–23** | **First professor/PhD outreach wave**; do not wait for a full paper. Ask for a short critical discussion, not immediately for GPUs. |
| **Oct 24–Nov 3** | Second sprint based on feedback, stronger comparator and/or chemical diversity test. |
| **Early–mid Nov** | If a collaborator is genuinely engaged, agree on scientific direction, authorship expectations, resource needs, and responsibilities. |
| **Nov–Dec** | Strong controlled tests; conditional molecular optimization only if the central hypothesis passes. |
| **Late Dec–Jan** | Decide honestly whether ICML-quality evidence exists; paper writing/submit if ready, otherwise target a later venue. |
| **2027 onward** | With domain/lab collaborators, biological-activity benchmarks and ideally prospective validation for the NMI direction. |

**Faster option:** Start quiet outreach as soon as a *single reproducible pilot plus a meaningful graph* exists (even Oct 12–14). A professor may be particularly helpful in identifying the right negative control. Show all current uncertainty.

### Outreach packet (minimum)

1. A **one-paragraph hypothesis** and explanation of how it differs from standard hybrid discrete/continuous molecular diffusion.
2. A **1-page architecture figure**, highlighting the precise A/B difference.
3. **Two plots with uncertainty or clear preliminary caveats**, baselines and evaluation protocol.
4. Link to a small reproducible repository, experimental configs, and compute usage.
5. A concrete ask: *“Would you be open to a 20-minute discussion? I'm looking for feedback on the novelty and strongest next experiment; if promising, I would be excited to collaborate with a student in your group.”*

Do not offer coauthorship as a reward for mentorship or assume funding/compute is available. Discuss substantive contribution and expectations early.

## 8. Faculty outreach shortlist (research fit, **not** verified openings)

**Strategy:** First select 4–6 best-fit faculty and 1–2 relevant PhD students/postdocs per group after reading their most recent work. Faculty profile listings do not imply that the person accepts unsolicited undergraduate collaborations or has available GPU resources.

| Person / institution | Why EMF might fit | Suggested angle | Public reference |
|---|---|---|---|
| **Tommi Jaakkola — MIT** | Foundational ML for structured objects, molecular graph generation, biomolecular design | Discretization policy, generalization and rigorous controlled baselines | https://people.csail.mit.edu/tommi/research.html |
| **Regina Barzilay — MIT** | Molecular representations, drug discovery, generative chemistry and protein–ligand modeling | Connect architectural change to useful small-molecule optimization | https://www.rbg.mit.edu/ |
| **Connor Coley — MIT** | Molecular design, synthesis constraints, closed-loop discovery | Is the final decoder experimentally actionable or synthetically misleading? | https://coley.mit.edu/research |
| **Aditi Krishnapriyan — UC Berkeley** | Physics/science-motivated generative ML, molecular modeling and flow methods | Generative dynamics, manifold representation, constrained transport | https://chemistry.berkeley.edu/people/aditi-krishnapriyan |
| **Stefanie Jegelka — MIT** | Graph learning, discrete-object distributions, combinatorial optimization | Mathematical case for terminal projection and structural constraint handling | https://www.csail.mit.edu/person/stefanie-jegelka |
| **Quanquan Gu — UCLA** | Diffusion/generative modeling and AI for science | Methodological novelty, efficient generative modeling; geographically accessible | https://web.cs.ucla.edu/~qgu/index.html |
| **Hosea Nelson — Caltech** | Experiment-facing molecular chemistry and automated discovery | Later-stage chemical validation rather than initial flow-method mentorship | https://www.caltech.edu/about/news/nsf-funding-for-caltech-led-cloud-laboratory-leverages-ai-to-explore-chemical-dark-matter-a-conversation-with-hosea-nelson |

**How I would prioritize:** For the architectural question, try **Jaakkola, Krishnapriyan, Jegelka, Gu**. For the drug-discovery relevance and practical constraints, **Barzilay, Coley**. **Nelson** is better suited to a later chemical-validation conversation. This is a fit-based ordering, not a prediction of responsiveness.

**A potentially useful nearby connection:** UC Berkeley's Krishnapriyan gave a Caltech seminar on generative ML and physical dynamics on October 1, 2026. The seminar is evidence of scientific proximity, **not** evidence of a collaboration or introduction channel: https://www.caltech.edu/campus-life-events/calendar/frontiers-in-chemical-engineering-4 .

**Practical contact route:** Start with concise emails to the lab or faculty using the official pages, and identify graduate researchers associated with the most recent related papers. Check current group membership before naming specific students.

## 9. Risks and technical questions an advisor will ask immediately

1. **Novelty:** How is this different from latent diffusion, hybrid molecular diffusion, delayed projection, and constrained discrete flows? Build a related-work matrix before claiming priority.
2. **Atom ordering/permutation:** A graph has no canonical intrinsic atom order. Does the encoder/decoder support permutation equivariance, or does canonicalized ordering drive the result?
3. **Valid graph guarantee:** Does the terminal decoder guarantee validity, reject invalid outputs, or chemically repair them? How are costs and distributional bias handled?
4. **Graph size:** Is atom count known, predicted, or sampled? Are padding masks given unrealistic ground-truth information at inference?
5. **Loss mismatch:** Does final-step decoding actually see off-manifold noisy representations from the trained denoiser?
6. **Meaningful chemistry:** Do valid and novel outputs have sensible chemistry, or are they merely easy-to-sanitize toy molecules?
7. **Compute comparison:** Are all methods fairly matched on parameter count, pretraining and inference function evaluations?
8. **Lead optimization:** Can a model conditioned on a starting active compound actually preserve a function or motif without training labels showing that relationship?

## 10. Immediate decisions

- **Primary venue hypothesis:** ICML, with a methodological framing; Nature Machine Intelligence remains a later or alternative high-impact application direction.
- **First model:** 2D graph contextual-embedding flow + structured terminal decoder; defer equivariant 3D and pocket conditioning.
- **First empirical question:** Intermediate CE vs terminal-only CE at matched compute, ideally with identical encoder and trunk.
- **Time to preliminary evidence:** Target **7 days**, allow **10 days** for clean replication.
- **Outreach:** As soon as one defensible comparative result exists, ideally **Oct 14–23, 2026**.
- **Hard stop:** Don't scale an unresolved toy-model confounder into a large-paper claim.

## References and verification links

- Hu et al., *ELF: Embedded Language Flows* (uploaded manuscript; 2026): https://arxiv.org/abs/2605.10938
- ELF public repository: https://github.com/lillian039/ELF
- *Flow matching for generative modelling in bioinformatics and computational biology*, **Nature Machine Intelligence** (2026): https://www.nature.com/articles/s42256-026-01220-0
- ICML official 2026 schedule (historical comparator, **not** a verified 2027 deadline): https://icml.cc/Conferences/2026/Dates
- ACL 2027 official call (final eligible ARR cycle January 4, 2027; ACL is not the preferred graph-method venue): https://2027.aclweb.org/calls/main/
- Faculty/lab pages linked in the shortlist above. Research fit checked in October 2026; availability for mentorship, new students, and compute is not verified.

---

**Bottom line:** EMF should earn the next level of investment by proving a narrow, transferable claim about *when structural discreteness helps or hurts generative flow*. The first week should produce an honest A/B test; the second should convert the outcome into a substantive faculty conversation. The full molecular-design ambition follows only if the basic phenomenon is real.
