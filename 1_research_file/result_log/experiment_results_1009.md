# First experiment results

The model can now generate small molecular structures from random noise. In our
first overnight comparison, it produced more chemically valid, connected
molecules when we trained the continuous movement without extra guidance about
atom and bond labels along the way.

This gives us initial evidence for our original idea. The explanation for the
effect and its behavior on more varied molecules still need to be investigated.

## What we built

The generator first learns to represent molecules as continuous numbers. It then
learns to move random noise toward those representations. At the end, a final
converter turns the numbers into atom types and bond types.

We built the complete process: preparing molecules, training the model, saving
and restarting training, generating structures, and checking every output. It
was trained on 20,000 small QM9 molecules containing carbon, nitrogen, oxygen
and fluorine, with at most nine heavy atoms.

## What we compared

We compared three ways of training the movement: no extra label guidance, a
small amount, and a larger amount. Extra guidance means asking intermediate
predictions to match the correct atom and bond labels during training. Every
version still makes its final atom and bond choices once, at the end of generation.

We also tried two versions of the final converter: one continued learning during
training; the other was kept fixed. Each comparison was repeated with three
different random starting points. This gave 18 training runs, each with one hour of
measured training time.

## What happened

The table shows the percentage of attempts that produced one connected molecule
and passed our chemistry checks. Broken or disconnected structures count as
failures. Every model received the same generation effort: 32 calls to the
movement model, including its final prediction.

| Extra label guidance during training | Final converter kept learning | Final converter held fixed |
|---|---:|---:|
| None | **84.0%** | **87.4%** |
| Small amount | 80.6% | 60.3% |
| Larger amount | 77.4% | 55.7% |

These are averages across three independently trained models, with 10,000
attempts per model. The settings correspond to λ=0, 0.1 and 1 in the
[experiment plan](../detail_operation/07_overnight_experiment_plan.md).

Within both converter versions, removing the extra guidance produced a higher
success rate in all three repeats and at every generation budget we tested.

For the version with a fixed final converter and no extra guidance, 10,000 attempts produced
an average of **8,547 different valid, connected molecules**. Of those, **7,871
were absent from our selected training set**. Absence from that training subset
does not establish that a molecule is new to chemistry.

## What the result means

**Our idea is worth pursuing: letting the movement learn in continuous space
gave more successful molecular outputs under the tested training-time budget.**

Part of the advantage comes from training speed. Extra label guidance takes
more calculation per update, so those models completed fewer updates in their
one-hour allowance. When we compared the same number of updates, the small
difference for the learning converter largely disappeared. The much larger
difference for the fixed converter remained.

There are also tradeoffs. Extra guidance sometimes produced more varied core
structures or better matched certain properties of the original dataset.
Producing more valid molecules is one useful result; diversity and property
matching remain part of the research question.

This first experiment supports continuing the existing research direction.
The north-star guide remains unchanged.

## Completion and records

All 18 runs finished successfully on 10 October 2026, with the last finishing at
08:51 PDT. The batch took about five hours and fourteen minutes using four GPUs
at a time, and consumed 18.82 L40S GPU-hours. No study training jobs remain active.

The final comparisons contain 900,000 generation attempts. Including the earlier
model check and training diagnostics, the overnight study recorded 1,091,000
attempts, retaining failures. The independent reviewer checked all final records
and separately reconstructed 1,800 sampled graphs; the results agreed.

- [Detailed experiment plan](../detail_operation/07_overnight_experiment_plan.md)
- [Recorded measurements](../../4_experimental_result/result/overnight_20261010/analysis/quality_and_yield_vs_nfe.csv)
- [Independent results review](../../5_test/results/overnight_results_review.md)
