# EMF execution plan — experiment-calibrated redesign

This plan builds the model defined in [01_math_plan.md](01_math_plan.md). The [research guide](../research_plan.md) states the scientific direction; this file states what to build, what to run first, and what each result will tell us.

**First deliverable:** one small, lead-conditioned molecular flow that uses a few observations from an unfamiliar campaign to change its proposals, with all pretrained weights fixed.

## 1. Build on the working generator

The existing package already provides a graph encoder, train-fitted latent normalization, an equivariant node/pair flow, a terminal decoder, checkpoint/resume, and a sampler that records every attempt. The local QM9 preparation contains 20,000 training graphs plus separate validation and test splits.

Reuse this foundation and the pinned ELF primitives. Read the [ELF source studies](../study_file/03_elf_model_study.md) and [adaptation record](../study_file/06_elf_adaptation.md) when modifying the backbone.

| Keep | Add |
|---|---|
| Frozen normalized graph encoder $E$ | Starting-molecule conditioning |
| Existing chemical flow trunk | Shared base-prediction and direction heads $F_\chi,B_\chi$ |
| Trainable source-stage decoder $D_\phi$ | Evidence encoder $H_\psi$ and direction selector $a_\omega$ |
| Continuous sampling and one final decode | Joint response density $p_\eta$ and profile scoring |
| Existing chemistry checks and attempt ledger | Campaign episodes, support/query splits, and response records |

The existing QM9 tensors contain graph labels, counts, identities, and SMILES. Campaign observations and paired endpoint outcomes are new data products. Each campaign shares one endpoint/assay context. The first model uses complete two-endpoint records; retain partial or censored records in raw storage and report their coverage.

## 2. The smallest model

Implement the same prediction as the math plan:

$$
\widehat X
=
F_\chi(Z_t,t,G_0;n)
+
B_\chi(Z_t,t,G_0;n)\,
a_\omega(Z_t,t,G_0,y_0,c,b_{\mathcal D},g,n),
\qquad
b_{\mathcal D}=H_\psi(\mathcal D).
$$

Use one graph trunk for $F_\chi$ and $B_\chi$. Encode the lead with frozen $E$ and pool its active node/pair features into an invariant context vector. This permits different lead and candidate atom counts without assuming an atom correspondence.

Implement $H_\psi$ as a small set encoder: encode each measured compound and its two outcomes, pool the records, and supply the record count. Keep goals outside this computation. The selector combines the current flow state, lead, baseline, actual assay context, response state, and requested goal.

Start $p_\eta$ with a two-dimensional Gaussian having a full covariance matrix. Predict two means and three Cholesky parameters. Use positive diagonal scales and add $10^{-4}I$ to $LL^\top$ in source-normalized endpoint units. This gives a normalized joint density and a well-defined likelihood for the deterministic toy observations.

| Setting | First choice |
|---|---|
| Chemistry | Existing neutral C/N/O/F graphs; at most nine heavy atoms; implicit hydrogens |
| Graph representation | Node width 64; pair width 32 |
| Shared flow trunk | Hidden width 128; four layers |
| Direction count | $r=8$ |
| Evidence state | Width 64 |
| Training batch | Eight campaign episodes, four query pairs per episode |
| Support per training episode | Sample 16, 32, 64, or 128 observed compounds |
| Flow endpoint / decoder noise | $T=0.95$; $\sigma=0.05$ |
| Initial loss weights | $\beta=\gamma=1$, with all three losses logged separately |
| Sampling | 1,000 attempts; 32 total flow evaluations |

These are implementation starting points. Change a setting when its learning curve, gradients, memory use, or samples identify a specific problem.

Initialize the direction heads with small nonzero weights and the selector output at zero. The initial clean prediction then matches the base flow while the selector can receive gradients. Initializing both branches at zero would prevent either branch from learning.

## 3. First exploration: chemistry stays available, effects change

**Question:** Does evidence from a new response function change the flow's chemical proposals in the useful direction?

Use the existing molecular graphs with synthetic campaigns whose responses are exactly computable. Cache graph-derived features such as atom composition, ring counts, polarity descriptors, and interactions between these features. Define each campaign's two outcomes with a hidden nonlinear function of these features.

Use 32 source campaigns, eight validation campaigns, and eight held-out campaigns as a first configuration. Keep the existing molecular splits separate as well. This tests unfamiliar response functions and unfamiliar molecular identities together.

The response functions must include changes that help in one campaign and hurt in another, plus nonlinear interactions. Construct them in graph-feature space independently of the learned latent directions. Their coefficients stay in private evaluator metadata; the model receives observed outcomes and ordinary endpoint context. The oracle must evaluate every valid generated graph in the supported chemistry, including graphs absent from QM9.

These are simulated observations. This pilot measures the learning mechanism with exact computational truth.

### Prepare redesign episodes

1. Select a lead and up to four nearby analogues within the same molecular split. Start with Morgan fingerprint similarity of at least 0.4 and heavy-atom-count difference of at most two. Record how many leads have eligible neighbours. Fit $p_{\mathrm{count}}(n\mid n_0)$ from source lead–analogue pairs before goal filtering; record unsupported lead-count rows.
2. Draw a support set of observed compounds from the campaign, including the measured lead. Keep query analogue outcomes outside support. Every support record counts once; pairs derived from those records do not increase the observation count.
3. Supply the query pair's two measured changes to the supervised training targets. Include favorable and unfavorable changes in the response loss.
4. Draw the goal independently of the query analogue and its outcomes: $\kappa\sim\mathcal U(-s_1,s_1)$ and $\tau\sim\mathcal U(y_{0,2}-s_2,y_{0,2})$, with independent draws and fixed source-training endpoint scales $s_1,s_2>0$. Weight each query's flow error by $\mathbf{1}\{\delta\in\mathcal A_g(y_0)\}$. Use a fixed batch denominator and log the accepted fraction. Batches with no accepted query still train the response model and decoder. The response loss includes every query.
5. Convert positive concentration-valued endpoints to common units and fixed oriented log scales, then fit normalization on source training observations only. Other endpoint types retain a documented common scale. Apply the same transformations to support values, baselines, response targets, and goals. Record log conventions and original units.

Split measurement records before pairing. Keep chemical identities, duplicate assay records, and query outcomes outside the held-out boundaries. Record IDs and campaign IDs support bookkeeping; hidden campaign coefficients and ID lookup embeddings stay outside model inputs.

### Train and generate early

Load the existing representation checkpoint at
[exploration_001/checkpoints/representation.pt](../../4_experimental_result/result/exploration_001/checkpoints/representation.pt)
after checking its encoder, normalization, and data provenance. Reuse its frozen representation for this chemistry. Initialize the base flow from a compatible existing flow checkpoint where practical.

Train $F_\chi,B_\chi,a_\omega,H_\psi,p_\eta,D_\phi$ on source episodes with the three losses in the math plan. Keep $E$ and normalization fixed. Begin sampling once the integrated model produces finite updates; inspect response predictions and molecules during training.

On validation campaigns, examine nested support sizes 16, 32, 64, and 128. For each comparison, keep the lead, goal, source weights, sampled count/noise seeds, and attempted-sample budget fixed. The measured lead is included in the support count. After development choices are recorded, apply the same procedure to held-out campaigns and molecules.

Choose deployment goals from source/validation scales and the observed lead baseline. Start with $\kappa=0.25s_1$ and $\tau=y_{0,2}-0.1s_2$. Keep these choices fixed for test evaluation. Hidden candidate outcomes are used only by the evaluator.

At deployment, freeze every pretrained weight, including the decoder. Only the support records and inferred response state change.

### Read the first result

Save the proposals and response-based ranking before calling the private oracle. Score all distinct valid proposals afterward, up to the 1,000-attempt ceiling. Record actual oracle calls separately from support observations; these evaluation outcomes stay outside the response state.

| Measurement | What it answers |
|---|---|
| Unmeasured joint-profile successes per attempted proposal, before ranking | Does evidence improve what the flow generates? |
| Distinct unmeasured joint-profile successes within the declared lead neighbourhood, per attempt | Does it produce useful local redesigns? |
| $\operatorname{success}(N,k)$ among the top $k=16$ selected proposals | Does response scoring improve the choices we would test? |
| $\operatorname{efficiency}(N,k)$ under the allocated $N+k$ compound budget | How many successes does the full observation-and-testing budget buy? |
| Held-out response likelihood and joint-event probability calibration | Does the evidence state improve expected effects? |
| Changes in selector coefficients and proposals under nested evidence | Does the proposed evidence route affect generation? |
| Validity, duplicates, structural proximity, time, and memory | What practical cost accompanies the behavior? |

Report both profile-only yield and yield satisfying the neighbourhood rule, excluding support rediscoveries from success counts. For paired support-size evaluation, reserve the entire largest nested support pool from candidate eligibility at every size. For the top 16, select distinct valid neighbours outside that pool that differ from the lead; unfilled selection slots count as failures. Retain every original invalid, disconnected, duplicate, support-matching, or out-of-neighbourhood attempt in the ledger.

Use the success and efficiency definitions in the math plan, averaged across held-out campaigns, leads, and seeds. The toy pilot scores the full proposal pool to distinguish generation from ranking; report those extra oracle calls separately. A later prospective campaign uses a declared synthesis-and-assay budget for selected compounds; synthesis or assay failures count as unsuccessful slots. Report actual compounds tested and individual endpoint measurements.

**Deliverable:** a checkpoint, reproducible campaign manifest, proposal ledger, support-size curves, and a plain-language account of how evidence changed the results.

## 4. Build the measured-data path in parallel

Audit one public or collaborator-supplied collection with two endpoints measured on overlapping compounds. The first preferred task is reduced off-target activity while maintaining on-target potency. A potency/developability pair is suitable when its assay records are more coherent.

Start the public-data audit from ChEMBL using a pinned release and documented assay records. The [data and related-work study](../study_file/07_redesign_transfer_study.md) records the sources.

Deliver an audit table containing:

- Endpoint definitions, targets, units/log conventions, shared campaign assay conditions, and source documents.
- Compounds with both endpoints, partial-measurement coverage, replicate/censoring rules, and series membership.
- Available source campaigns, development campaigns, and held-out campaigns.
- Overlap and similarity across splits, including repeated measurements of the same compound.
- Atom counts, elements, charges, stereochemistry, and the fraction supported by our representation.

The audit determines the first biological dataset and representation expansion. Select a tractable chemistry envelope that retains useful series; update the encoder vocabulary and decoder, then repeat source reconstruction training before freezing the new representation.

For retrospective evaluation, restrict candidates to a library with both endpoints measured and withhold those outcomes until selection. Rank or search that library under a fixed evaluation budget, reveal its recorded outcomes, and measure joint-profile success. Generated molecules outside the library remain proposals until independently evaluated. This separates measured decision performance from generative predictions.

Use support sizes 16, 32, 64, and 128 when the selected campaigns support them. Count unique compounds and individual endpoint measurements. Hold out assay/campaign context and chemical series according to the intended transfer claim.

## 5. Compute and run organization

Use the existing isolated environment:
`/hopper/home/miyao/miniconda3/envs/emf`.

The current scheduler configuration provides L40S allocations through `main`, account `standard`, QOS `normal`. Start each training job with one L40S, eight CPUs, 32 GiB host RAM, and a checkpointed four-hour wall-time limit. Keep CPU data preparation separate from GPU training.

The existing 5.05-million-parameter exploration took 289 allocated GPU seconds for reconstruction plus flow training. This is a reference for the backbone; profile the new episodic model before selecting its training duration. Historical measurements are in the [implementation log](../result_log/implementation_log_1009.md).

| Initial exploratory allocation | L40S GPU-hour allowance |
|---|---:|
| Integrated operation and throughput measurement | 1 |
| First source-campaign training and checkpoint inspection | 4 |
| Changes motivated by observed learning or generation failures | 4 |
| Support-size evaluation and reserve | 3 |
| **Total** | **12** |

These are allocation caps, not predicted runtimes. Begin with one model and one seed. Add concurrency when runs answer separate questions. After the first useful behavior appears, use three paired seeds for confirmation before a larger study.

Write new artifacts under `4_experimental_result/result/calibrated_redesign_001/`. Store the run configuration, source checkpoint hashes, dataset/split identities, oracle specification, source-code version, random states, observations revealed to each episode, and actual CPU/GPU timing.

Count the terminal clean prediction in NFE. Sample candidate counts from the frozen source-pair prior $p_{\mathrm{count}}(n\mid n_0)$ using the supplied lead count. Include host chemistry and response scoring in end-to-end timing. Save all attempts, then write summaries in CSV/JSON.

## 6. Parallel agent workflow

The coordinator and three builders can run concurrently. As a builder hands off a working component, use that slot for an independent reviewer while the remaining work continues.

| Role | Owned work | Handoff |
|---|---|---|
| Coordinator | Shared contracts, `train.py`, configs, environment, metadata, Slurm scripts and GPU submission | Episode interface, integrated runner, reproducible source/deployment checkpoints |
| Data/chemistry | `data.py`, `chem.py`, new campaign/response-oracle data modules | Support/query episodes, neighbour pairs, conditional count prior, private oracle, measured-data audit |
| Model/objective | `models.py`, `objectives.py`, `elf_layers.py`, new redesign/response modules | Shared F/B trunk, H/a/p functions, three losses, finite-gradient example |
| Sampling/evaluation | `sampling.py`, `evaluation.py` | Frozen deployment, joint scoring, full proposal ledger and oracle measurements |
| Independent reviewer | Read-only code/math inspection; pytest files and reports under `5_test` | Contract review, leakage checks, math/code consistency and allocated-GPU operational report |

Agree on three interfaces before parallel edits:

- **Campaign episode:** measured support graphs/context/outcomes; measured lead/baseline; query graph/outcome held outside support; goal; candidate count and masks.
- **Prediction:** H maps support to one state; F/B receive state/time/lead/count; a receives the allowed evidence and goal; p receives a fixed candidate and evidence without a goal.
- **Deployment:** a source checkpoint plus support, lead, goal, count/noise seeds, and attempt budget produces proposals and scores without changing weights.

Builders can work against synthetic tensors immediately. Pass campaign conditioning explicitly to each model call. Cache frozen support embeddings and compute each episode's response state once per sampling run. Keep upstream repositories unchanged and preserve ELF's MIT notice in adapted primitives.

Use focused operational checks: support-order invariance, candidate permutation equivariance, positive response covariance, outcome-independent goal sampling and feasibility weighting, normalized conditional count rows, goal-independent response predictions, nonzero selector learning, support/query isolation, correct NFE counting, fixed deployment weights, and checkpoint reproduction. The reviewer exercises training and sampling on an allocated GPU as the integrated model becomes available.

Request fixes from the file owner. Keep each worker's outputs separate.

## 7. Comparisons after exploratory behavior appears

Choose comparisons that explain the observed effect. The principal challenge is a strong few-shot response predictor guiding a standard generator or molecular search.

| Later comparison | Question |
|---|---|
| Ordinary conditional flow with the same evidence encoder and response scorer | Does the explicit reusable-direction structure help? |
| Frozen-generator task conditioning and parameter-efficient adaptation | Does response-state inference transfer efficiently? |
| The same few-shot predictor with standard proposals, retrieval, or Bayesian optimization | Does putting evidence inside the flow add value beyond ranking/search? |
| Fixed or swapped support state within our architecture | Does campaign evidence cause the transfer effect? |

Match source data, target observations, chemical eligibility, attempted proposals, evaluator budget, and tuning effort. Report parameter counts and actual computational cost. Choose settings on validation campaigns, then evaluate whole held-out campaigns/series with paired seeds and uncertainty across those independent units.

Use `4_experimental_result/control/` for later controlled-study code. Preserve the earlier supervision-study artifacts as records of backbone development.

## 8. Milestones and decisions

| Milestone | Concrete result |
|---|---|
| First integrated model | One source-trained checkpoint generates lead-conditioned proposals and joint response scores with frozen deployment |
| First mechanism result | Evidence changes pre-ranking joint-profile yield, local redesign yield, and response predictions on held-out toy campaigns |
| First measured transfer result | Joint-profile selection improves with limited observations on held-out measured campaigns/series |
| Attribution result | The advantage survives strong prediction/search alternatives and an equally resourced conditional flow |
| Collaboration packet | Two-page research note, three central figures, runnable code, and a specific assay/chemistry question |

If response predictions improve while proposals stay unchanged, inspect the selector's learning and evidence route. If gains appear only after ranking, focus the next experiment on coupling response evidence to the flow. If unfamiliar chemistry falls outside the available representation, expand source chemistry before the next transfer study.

Proceed from the first working model to the first informative result. Use that result to choose the next experiment.
