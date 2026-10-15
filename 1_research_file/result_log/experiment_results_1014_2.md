# Updated experiment results — 1014, second record

Recorded **October 14, 2026, at 12:14 PDT**. Study: `revision_efficiency_001`.

**Main result:** continuing a molecular-generation trajectory works reasonably well when we add measurements. It struggles when we reverse the design goal. Hard masking preserves the protected structure exactly, but expensive guidance still reduces chemical validity.

The comparison of generation methods is complete on held-out test campaigns. The revision results below are from completed validation runs. Three final revision test jobs are still running.

## What we tested

We used the saved hard-masked graph generator, called B. Its mask fixes the protected atoms and bonds while allowing changes in the editable region. We compared four ways to choose molecules:

- **Sample and rank:** generate candidates, then rank them using the measured evidence.
- **Ordinary conditional flow:** provide the evidence-derived response information during generation.
- **Terminal control:** repeatedly optimize guidance using predicted final outcomes.
- **Larger sample-and-rank pool:** generate 25 times as many candidates as the small pool.

We also compared continuing an existing trajectory with restarting after an update. One update adds measurements. The other reverses the requested change in the first property while retaining the second-property requirement and the structural mask.

## Completed test: generation success, validity, and cost

These results use 16 measured compounds per campaign, 16 campaigns, and three model seeds. Each campaign/seed case requests 16 selected candidates: **768 requested slots per method**. Success requires meeting both simulated property thresholds and the structural and eligibility rules. Empty slots count as failures.

| Method | Proposals per case | Successful requested slots | Valid connected attempts | Recorded inference and evaluation time |
|---|---:|---:|---:|---:|
| Sample and rank | 64 | 369/768 — **48.0%** | 95.4% | 37.6 s |
| Ordinary conditional flow | 64 | 398/768 — **51.8%** | 95.6% | 37.5 s |
| Terminal control | 64 | 475/768 — **61.8%** | **72.2%** | 738.1 s |
| Larger sample-and-rank pool | 1,600 | 560/768 — **72.9%** | 95.4% | 940.6 s |

Times are summed across cases and seeds. They include decoding and host chemistry, and exclude model training. These methods use different computational budgets.

All four methods preserve protected graph categories on **100% of attempts**. The mask works as intended. Chemical validity remains a separate problem: guidance produces more invalid molecules inside the permitted edit region.

The B-based controller improves selection success from the previous A-based result of **57.3% to 61.8%**. Its validity falls from **81.2% to 72.2%**. The B experiment also uses its own fitted response surrogate, so this comparison does not isolate the effect of masking.

Control takes about **19.6 times** the time of small-pool sampling. The larger pool takes **27.4% more time than control** and achieves higher success. An efficiency advantage for control remains unestablished.

Evidence affects the generated candidates. Correct evidence produces **660 distinct successful identities summed within campaign/seed cases**, compared with **303 under reversed first-property evidence**, at the same proposal count. These totals can include the same molecule in different cases.

## Completed validation: continuing versus restarting

The table shows an update halfway through ordinary conditional generation. Both continuation and restart receive **1,600 proposals per case** and request 16 selections. Restart cost includes the trajectory prefix that it discards.

| Update at approximately 50% progress | Continue: successful slots | Restart: successful slots | Reduction in total charged time from continuing |
|---|---:|---:|---:|
| Increase measurements from 4 to 16; keep the goal | **608/768 — 79.2%** | 618/768 — 80.5% | **27.6%** |
| Reverse the first-property goal; keep the measurements | **121/768 — 15.8%** | 229/768 — 29.8% | **27.6%** |

With additional measurements, continuation loses 10 successful slots while reducing total charged time from **1,383.1 to 1,001.5 seconds**. Time after the update falls from **930.4 to 548.7 seconds**, a **41.0% reduction**. This is a promising descriptive result; statistical equivalence has not been established.

With goal reversal, continuation loses **108 successful slots**. Earlier redirection helps: continuation achieves **151/768 successes** when the goal reverses at 25% progress, versus **121/768** at both 50% and 75%. Restart achieves **229/768**. Conditional generation retains approximately 95% chemical validity, so the main loss here is failure to meet the revised goal.

The expensive controller shows a similar pattern. At the midpoint, continuation versus restart gives **393 versus 409 successes** after additional measurements, and **100 versus 121** after goal reversal. These controller runs use only 32 proposals per case. Their percentages cannot be compared directly with the 1,600-proposal conditional runs as equal-budget results.

Goal reversal also has fewer known feasible examples in the small validation reference library: 8 of 16 campaigns have an eligible reversed-goal example, compared with 13 for the original goal. This finite-library check does not establish that the remaining campaigns have no solution. Continue and restart use the same campaigns and final goals.

## What we learned

**Structural preservation is reliable. Chemical validity under guidance still needs work.** Exact masking protects the fixed region, while the controller can damage the editable region.

**Simple evidence updates can reuse an existing trajectory.** Ordinary conditional generation already provides this benefit. Any new controller must improve on that inexpensive alternative.

**A conflicting goal exposes a useful research problem.** The trajectory may become committed to an earlier objective before generation ends. The results motivate testing this explanation; they do not yet identify its mathematical cause. A focused next experiment could measure when useful redirection is lost and whether a short rollback restores it.

The working research question is: **Can a molecular flow preserve useful progress while retaining enough flexibility to respond to a conflicting design objective?**

## Work still running and scope of the result

At this snapshot, test jobs `375871_0`, `375871_1`, and `375871_2` are active. Their logs report **11/16, 11/16, and 12/16 completed campaigns**, respectively. Final revision results will compare success at common time budgets from **0.0625 to 32 seconds per case**. These budgets were selected from validation timing alone.

The study uses 294 nine-heavy-atom molecules from two families, a known descriptor basis, and synthetic noisy response functions. The original model initializer predates the molecular split. The results concern held-out campaign responses in this setting; transfer to unfamiliar chemical series and real assays remains untested.

## Measurements and related records

- [Static B test measurements](../../4_experimental_result/result/revision_efficiency_001/static_b/).
- [Revision validation measurements](../../4_experimental_result/result/revision_efficiency_001/validation/).
- [Numerical snapshot and source hashes, recorded at 12:10 PDT](../../4_experimental_result/result/revision_efficiency_001/analysis/running_results_1014.json).
- [Live report](../../4_experimental_result/result/revision_efficiency_001/live_report.md).
- [Final test status](../../4_experimental_result/result/revision_efficiency_001/status.json).
- [Experiment plan](../detail_operation/06_revision_efficiency_experiment_plan.md).
- [Earlier October 14 results](experiment_results_1014.md).
