# Supervision diagnostic

**Recorded: 10 October 2026, America/Los_Angeles.**

We tested why adding atom/bond supervision reduced connected validity in the
overnight experiment. The new tests examine training signals and decoder
probabilities. They do not measure an improved molecule generator.

## What we built and ran

We added two small diagnostic programs:
[saved-model diagnostic](../../4_experimental_result/control/supervision_diagnostic.py)
and [exact mixture test](../../4_experimental_result/control/mixture_compatibility.py).
Both use the existing encoder, decoder, data and checkpoints.

The first program inspected all 18 saved models at 50,000 training updates.
It used the same 256 validation molecules, two Gaussian-noise draws and 11
times along the noise path in both decoder settings. Flow-parameter gradients
used the first 64-molecule batch, one noise draw and four selected times.

Slurm array 370552 ran one L40S job per decoder setting. Both completed
successfully. The diagnostics took 36.54 and 37.19 seconds. Slurm allocated
42 and 41 seconds, for a total of 0.0231 GPU-hours. Each job was capped at
20 minutes.

The second program used eight distinct validation molecules with nine heavy
atoms each. For this defined probability distribution, it could calculate the
correct average embedding and atom/bond probabilities exactly. It checked the
three pretrained representations at eight noise times with 32 paired draws.
The CPU run took 2.30 seconds and produced 768 measurement rows.

## What we found

All 18 saved models reconstructed all 256 selected clean validation molecules
exactly. The three pretrained decoders also reconstructed each of the eight
molecules in the exact test. These are subset checks.

At an early time, t=0.1, the model trained with categorical weight 0.1 showed:

| Decoder during flow training | Direction of the categorical signal | Mean size relative to the flow signal |
|---|---|---:|
| Fixed | Opposed the flow signal in all three seeds | 4.70 times |
| Trainable | Agreed with the flow signal in all three seeds | 0.245 times |

These measurements use the raw flow-parameter gradients on one paired batch
per seed. They describe local objectives, before AdamW and gradient clipping.
They do not prove which effect caused the earlier validity difference.

The exact test found a more specific issue. At t=0, all eight molecules were
equally probable. The correct flow prediction was their average embedding.
The decoder assigned inaccurate category probabilities to that average.

The best possible expected atom-plus-bond categorical loss on this case was
0.9001. The three decoders gave losses of 4.77, 5.53 and 5.04. Smaller is better.
These losses use natural logarithms and average atom and pair terms separately.

The categorical-loss gradient at the correct average was nonzero in every
seed: its norm was 0.554, 0.094 and 0.321. Thus, on this specified test, adding
the categorical loss would push an unconstrained prediction away from the
correct flow average. At t=0.03, about five candidates remained plausible on
average and the mismatch persisted. By t=0.2, the posterior was effectively
one candidate and the loss and gradients approached clean reconstruction levels.

This establishes a compatibility failure for the selected eight-molecule
distribution. It does not establish the same mechanism across QM9. The exact
test used CPU FP32 decoder calculations; the training and GPU diagnostic use
CUDA bfloat16 trunks with FP32 heads and losses.

## What this changes in the next experiment

The next pilot should teach the decoder the category probabilities of
averaged embeddings. Fit it on mixtures of training molecules and evaluate
on held-out molecules and mixtures. Preserve complete-molecule reconstruction.

Then test whether categorical supervision works better with that calibrated
decoder. Reuse the existing no-supervision and weight-0.1 checkpoints. Six new
flow runs can test a weak weight and the calibrated decoder across three seeds.
The proposed budget is about 5-6 allocated GPU-hours. This pilot has not been
submitted.

The provisional research question and math are in
[08_supervision_research_plan.md](../detail_operation/08_supervision_research_plan.md).
These findings refine the next experiment; the high-level research guide
keeps its current direction.

## Checks and evidence

Seven saved-model diagnostic math tests and four mixture math tests passed.
The reviewer added and passed two independent analytic mixture checks.
The GPU jobs also passed the seven-test suite before running. Independent
review checked the code, math, real checkpoints, paired inputs, raw rows,
summary calculations and recorded hashes.

Each decoder setting has 152,064 gradient rows, 52,992 decoding rows and
108 flow-parameter rows. Zero-gradient cases have explicit flags and undefined
ratios. The two settings use identical validation inputs and noise.

Raw measurements:
[supervision_diagnostic_20261010](../../4_experimental_result/result/supervision_diagnostic_20261010).
Review evidence:
[saved-model audit](../../5_test/results/supervision_diagnostic_audit.json) and
[finite-mixture audit](../../5_test/results/finite_mixture_audit.json).
