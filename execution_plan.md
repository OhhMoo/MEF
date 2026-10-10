# EMF execution plan

Updated 10 October 2026. Build a working EMF generator, explore its behavior, then design comparisons around what we learn. The core mathematics remains in `math_map/math_plan.md`.

## 1. What we are building

**A small molecular generator that learns continuous atom/pair embeddings and turns Gaussian noise into a molecular graph through one final decoding operation.**

The system has three learned components:

| Component | What it does |
|---|---|
| Encoder $E$ | Converts a training molecule into continuous node and unordered-pair features. |
| Flow model $F_\theta$ | Learns to move noise toward those molecular embeddings. |
| Decoder $q_\phi$ | Converts the predicted final embedding into atom and bond categories. |

The first runnable artifact trains these components, saves checkpoints, and generates a file of molecules from noise. Each generation attempt includes its molecular identity if valid, its failure reason otherwise, and its generation time.

The first research task is to discover what this generator learns, what it generates, and where it fails. These observations will determine the later comparisons.

## 2. First exploratory experiment

### Build one small model

Start with QM9 heavy-atom graphs: C/N/O/F, at most nine heavy atoms, implicit hydrogens, connected graphs, zero formal charge on every atom, and no stereochemical labels. Use one Kekulé bond convention and include “no bond” in the pair vocabulary.

Standardize and deduplicate before making train/validation/test splits. Begin with 10,000–20,000 training molecules. Fit the encoder, normalization, and atom-count prior on training data; use validation molecules for development. Keep the test split for later evaluation.

Use a small permutation-equivariant node-and-pair network. Starting dimensions are node latent width 64, pair width 32, flow hidden width 128, four layers, and batch size 128. These are starting choices that can change during exploration. Explicit pair features carry adjacency information.

### Warm-start the representation, then generate

1. Train $E$ and $q_\phi$ to reconstruct molecules. Inspect a small batch of reconstructions and the learning curve.
2. Freeze $E$ and its train-fitted latent normalization so the flow has a stable target. Keep $q_\phi$ trainable.
3. Train the flow and terminal decoder with the $\lambda=0$ objective from the math plan:

$$
\boxed{
\mathcal L_{\mathrm{explore}}
=
\mathcal L_{\mathrm{flow}}
+
\mathcal L_{\mathrm{end}}.
}
$$

Flow training uses noisy interpolations of encoded molecules. Decoder training uses encoded molecules with the common corruption noise. Start with $T=0.95$ and a corruption scale such as $\sigma=0.05$ after normalization, then adjust from observed reconstruction and generation behavior.

Representation warm-up is part of building this generator. Begin flow training when the pipeline runs, losses are finite, and reconstruction is learning. Measure reconstruction throughout development; use its errors to diagnose the model rather than requiring a near-perfect score before the first generation run.

### Run the complete generation path early

Draw an atom count from the training prior and hold it fixed. Draw Gaussian noise, integrate the learned flow, predict the final embedding, decode atom/bond categories once, and check the resulting molecule.

Start with **1,000 attempted molecules at 32 total flow-model evaluations**: 31 Euler integration calls and one endpoint prediction. Retain every attempt, including invalid and disconnected outputs.

Inspect:

- How many attempts are chemically valid and connected?
- How many distinct molecules appear, and how many are absent from training?
- Which atom, bond, or connectivity errors dominate failures?
- Does generation improve as training continues?
- How much time and memory does the complete pipeline use?

The first deliverable is a checkpoint plus a reproducible generation command, a sample ledger, and a short account of what worked and failed.

## 3. Compute and an exploration budget

The 9 October cluster audit found 60 shared L40S GPUs, with [48 GB memory per GPU](https://www.nvidia.com/en-us/data-center/l40s/). Account `standard` with QOS `normal` had a 32-GPU aggregate ceiling. Access is scheduled through Slurm; queue availability must be checked when launching.

Use **one L40S, eight CPUs, and 32 GiB host RAM per run**. Begin with one training allocation. A second allocation becomes useful when we have a specific alternative to explore. Use short checkpointed jobs, initially capped at four hours.

The existing CUDA-enabled PyTorch environment lacks RDKit. Establish one isolated project environment and verify an actual GPU batch in the first allocation. The coordinator owns installation and job submission.

| Exploration stage | Proposed L40S GPU-hours |
|---|---:|
| Runtime check and representation warm-up | 2–4 |
| First flow/decoder run and early generation | 4–8 |
| Two or three changes motivated by observed failures | 4–8 |
| Sampling, inspection, and interrupted-run reserve | 2–4 |
| **Initial exploration allowance** | **12–24** |

These are spending allowances, not convergence-time predictions. Profile update time and sampling throughput, then choose the run lengths. Decide the next allocation from the first samples and learning curves.

## 4. Parallel agent workflow

Use one coordinator and three workers to build the components of the same generator.

| Agent | Owned paths | What it builds |
|---|---|---|
| Coordinator | `emf/contracts.py`, `emf/train.py`, `configs/`, dependency files, Slurm scripts | Environment, small shared interface, reconstruction/flow training loops, integration, GPU runs |
| Data/chemistry | `emf/data.py`, `emf/chem.py` | QM9 preparation, graph tensors, splits, count prior, graph-to-molecule conversion |
| Model/objective | `emf/models.py`, `emf/objectives.py` | Small encoder, flow model, decoder, reconstruction and flow/decoder losses |
| Sampling/evaluation | `emf/sampling.py`, `emf/evaluation.py` | Euler sampling, terminal decoding, sample ledger, basic generation metrics and timing |

Agree on a short tensor contract before parallel implementation:

- Atom labels `[B,N]`, unordered-pair bond labels `[B,P]`, masks, molecule IDs, and counts; pairs have a fixed $i<j$ ordering.
- `E(batch) -> Latents(nodes, pairs)`; `F(latents, time, masks) -> Latents`; `q(latents, masks) -> atom_logits, bond_logits`.
- F receives noisy latents, time, and masks. Losses use active coordinates; bond symmetry and inactive padding are preserved.
- One chemistry conversion function and one generation-attempt format are shared by the pipeline.

Workers can build against synthetic batches while the dataset is prepared. Use a few practical checks for tensor shapes, gradients, symmetry, and chemistry round trips so the components connect correctly. Each worker returns its functions, a runnable example, and any integration issue.

The coordinator integrates the modules and launches the first model. During training, the workers continue in parallel: inspect molecule failures, investigate model learning, and evaluate successive checkpoints. Keep separate run directories and record configurations, dataset IDs, seeds, and checkpoint versions.

First-wave prompts:

> **Data/chemistry:** Build the restricted QM9 data pipeline and chemistry conversion in your owned files. Deliver a usable training batch, molecule-disjoint splits, unordered-pair tensors, and a training atom-count prior. Work against the shared interface and check a few molecule round trips. Coordinate dependencies with the coordinator.

> **Model/objective:** Build small E, F, q modules and the losses in your owned files. Support reconstruction warm-up followed by a frozen encoder with trainable flow and decoder, using flow loss plus terminal decoder loss. Deliver a synthetic-batch example with finite gradients. Keep the implementation easy to change during exploration.

> **Sampling/evaluation:** Build the complete noise-to-molecule sampler and basic evaluator in your owned files. Count the endpoint call in the model-evaluation budget, decode once through the shared chemistry function, and retain invalid attempts. Deliver a sample ledger and timing/validity/uniqueness summaries.

## 5. The exploration loop

**Build → train → generate → inspect → change one component → run again.**

Prioritize the smallest change suggested by the observed failure:

| Observation | Next exploratory change |
|---|---|
| Reconstruction loses atoms or bonds | Improve the encoder/decoder or its training. |
| Reconstructions work but generated embeddings decode poorly | Inspect flow predictions and decoder corruption tolerance; adjust the weaker component. |
| Valid molecules appear but collapse to duplicates | Inspect training coverage, latent scaling, and flow learning. |
| Results depend strongly on integration length | Compare a few sampling budgets using the existing checkpoint. |
| Small-molecule generation becomes useful | Expand to the full restricted QM9 split, then a filtered diverse subset. |

Keep each change and its results in a short run note. Changing one component at a time makes exploration readable without requiring a full baseline suite.

## 6. Controls after exploration

Once the generator produces behavior worth explaining, choose the comparison that answers the question raised by those observations.

For intermediate categorical supervision, add positive $\lambda$ to the same implementation. If the early comparison is interesting, pretrain and freeze a common encoder and decoder to isolate the transport effect. Then use matched training work, tuned positive weights, multiple paired seeds, and a fixed evaluation protocol. Select settings on validation data before final test evaluation.

Reserve the earlier **40–70 GPU-hour** controlled-study allowance for that later phase, revising it from measured throughput. External baselines and broader dataset replication follow when the observed effect warrants them. Related work such as [molecular latent flow matching with final decoding](https://arxiv.org/abs/2610.06468) informs that comparison and the eventual contribution.

The immediate milestone is **one trained EMF generator, its first molecular samples, and a clear next experiment suggested by those samples**.
