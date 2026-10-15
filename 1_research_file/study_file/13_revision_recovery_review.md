# Saved-state recovery: mathematical and operational review

This review covers the bounded recovery study following `revision_efficiency_001`. The scientific sequence is discovery, diagnosis, and one minimal learned attempt. The final preceding report shows a late goal-reversal penalty for conditional continuation: its three-seed top-16 success is 27.7%, compared with 38.0% for conditional restart at the full attempted pool. Restart uses more post-update computation. Recovery experiments determine whether earlier saved states or bounded late interventions improve this tradeoff.

Source context is the [ELF sampling study](05_elf_sampling_study.md), [adaptation record](06_elf_adaptation.md), [revision review](11_revision_efficiency_review.md), and [completed revision measurements](../../4_experimental_result/result/revision_efficiency_001/live_report.md). The terminal clean prediction belongs in the suffix NFE.

## 1. Three questions and their estimands

**E1: rollback.** Execute the old conditional trajectory to the state nearest time .75, retaining the .25 and .50 states. Reveal the reversed goal and compare continuation, rollback to each earlier checkpoint, restart, and direct constrained mutation. Every strategy in the late-update comparison inherits the full historical cost through .75. Rolling back changes future work; it does not refund work already executed. Show both total cost and cost after revelation.

**E2: empirical recoverability.** For each predefined saved state, draw a bounded set of perturbations on editable latent coordinates, then execute a deterministic suffix under the new context. A useful explicit intervention is `z' = z + P delta`, with a declared per-example editable-coordinate RMS or Euclidean bound. The definition must state which norm is bounded; high-dimensional RMS and Euclidean radii differ substantially. A zero perturbation must reproduce the saved-state continuation. Each nonzero trial starts from the same unmodified saved tensor, with an independently seeded perturbation.

The unchanged deterministic numerical suffix has one output. Perturbation search explores a declared set of altered states. An exact successful terminal graph establishes empirical recovery within that set. Zero successes means no observed recovery at that search budget, with that proposal distribution and radius. It does not establish global unreachability or an ODE uniqueness result.

Keep both all-state recoverability and recovery among deterministic baseline failures. For the latter, success means valid connected chemistry, exact protected categories, common eligibility, the new endpoint-1 requirement, and retained endpoint 2. Report the failed-state denominator, attempts, invalid trials, duplicates, and changed identities. Use identical root-state IDs for baseline and local search. Reusing oracle labels for analysis has zero additional physical query cost, while a deployed oracle-search strategy pays for every queried candidate.

**E3: one small state adjustment.** Freeze the generator and train one small permitted-coordinate adjustment using source campaigns. A gate based on local same-state E2 successes establishes that bounded interventions have targets in the intended regime. Restart successes establish campaign feasibility; they do not establish local recoverability around a saved late state. A same-time restart trajectory is a possible distillation target, whose required displacement must be compared with the intervention bound. State prediction error and exact decoded success answer different questions.

The coordinator locked the prospective source gate: at least eight recoverable baseline-failed source states across at least two source campaigns; seeds may be aggregated. The learned attempt uses hidden width 64, 500 steps, gain one, and joint editable-coordinate trust RMS .5. It saves the fixed final checkpoint. Validation teacher-state error remains a diagnostic. The proposed decoded validation acceptance rule requires positive paired exact-success gain and at most five percentage points lower validity; the coordinator's immutable manifest owns the final rule. These operational thresholds justify one bounded attempt. A failed gate ends E3 with its recorded diagnostic and leaves E1/E2 intact.

## 2. Campaigns, information, and selection

Use fixed campaign IDs from the established splits, chosen by split membership and a deterministic ordering before new outcomes. Preserve all predefined campaigns. A source/library feasibility stratum may be reported alongside the primary all-campaign result; selection based on a generator's new successes would change the target population. Record zero-feasible library cases because finite-library coverage is a concrete property of this chemistry set.

The preceding test results already motivated choosing reversal. New rollouts on those same campaigns measure a prospective implementation on previously examined campaigns. Preserve this provenance when describing generalization. New held-out campaigns require a separate prospective construction and frozen protocol.

Available evidence, support reserve, lead, mask, initial draws, requested goal, and final ranking rule must be paired across deployable arms. Exact simulator coefficients are restricted to post-commitment evaluation or an explicitly privileged oracle diagnostic. Oracle-selected candidates require exact-call-matched restart and direct-search comparators; approximate-critic selection and exact-label selection receive separate labels. Invalid attempts consume the same search budget as valid attempts.

## 3. Accounting and uncertainty

Keep physical work, charged total-strategy work, and incremental work separate. For shared historical prefixes, amortize a single paid prefix over that root state's repeated local search trials. A batch ending after a budget contributes no partial proposals. Include chemistry, deduplication, ranking and synchronization in the measured deployable cost. Record offline exact audit time separately, or include it consistently across all arms. Report oracle search both by complete attempts/exact calls and by wall time.

Bootstrap campaign IDs as clusters and carry seeds, methods, saved times, and budget points together. Form paired method differences before aggregation. Show seed-specific effects and the number of distinct campaigns; perturbation trials from one saved state are correlated and are not independent campaign replicates. Conditional recovery among baseline failures is an explicitly selected diagnostic subgroup; primary population performance remains all-state and all-campaign.

## 4. Review acceptance checks

The independent checks will exercise actual helper and runner contracts as they become available:

1. A zero perturbation reproduces the deterministic suffix, nonzero trials preserve protected and inactive coordinates, and perturbation bounds hold per example without mutating the saved source tensor.
2. Rollback uses the stored earlier state while charging historical execution through the revelation time; terminal clean prediction is counted.
3. Physical response stays unchanged under reversal; independent exact graph evaluation applies endpoint-1 sign and retains endpoint 2 after commitment.
4. Search trials retain invalids and duplicates; oracle selection cannot receive more exact calls than its matched comparators.
5. Model fingerprints remain unchanged; adapter gradients and optimizer membership exclude generator parameters.
6. Completed-batch budgets retain zero-output points and missing selection slots. The paired campaign interval respects common campaign identities across seeds.
7. An allocated-GPU smoke verifies synchronized timing, retained failure records, and reconstruction of cost fields before a full run.

Implementation review and executed checks are appended after the owners expose the new modules. The reviewer owns this note and `5_test/test_revision_recovery_review.py`; fixes to implementation files are requested from their owners.

## 5. Implementation review and executed checks

The recovery builder corrected within-batch root IDs, included lead encoding and Gaussian preparation in charged cost, bounded adaptive perturbations by the declared radius, and distinguished attempted exact-evaluation requests from successful exact-response calls. Root recoverability records retain each deterministic root's failure status and intervention outcome. The adapter implementation contains only small adjustment-network parameters; its training optimizer excludes the generator, generator fingerprints and gradients are checked, and only public support observations enter teacher conditions.

Six independent CPU tests passed in [the reviewer suite](../../5_test/test_revision_recovery_review.py), with [the execution report](../../5_test/results/revision_recovery_review.xml). They check projected perturbation bounds and owned randomness, source-state immutability, cached-prefix and zero-intervention suffix agreement including terminal NFE, indivisible budget pools with historical cost, direct RDKit recomputation of the simulator response from committed identities, the integrated runner's .75 historical charges and root pairing, and source-gate exclusion of duplicates, already successful roots, excessive radii, and test campaigns. Raw candidate validity is reconstructed from all attempted candidates, independently of selected-root validity. These checks use small tensors and a real untrained B runner for operational contracts; they do not estimate scientific success rates.

The allocated L40S operation, parent job `376053`, completed both recovery and two-step adapter training with exit code zero on 15 October UTC (14 October local time). Independent artifact review recomputed 225 exact-response rows across 256 logical rows, verified 208 physical evaluation requests, checked 24 case ledgers and their cost decomposition, and verified frozen-base training and 51 snapshot hashes. The two-step adapter is an operational smoke excluded from the scientific gate and scientific fit. The subsequent L40S rollout, parent job `376055`, also completed with exit code zero: all 80 exact-response rows and 10 cases passed the same audit, and all 16 root-state fingerprints matched across continuation, restart, and adjustment. The machine-readable [GPU review](../../5_test/results/revision_recovery_gpu_review.json) records overall acceptance as passed.

Direct mutation now has a privileged exact-ranked counterpart with the same K-call candidate groups as local search. Local-search diagnostics select one candidate per root before final ranking. Standard direct mutation ranks its full pool. Fresh sampling and mutation references label their zero-history timing explicitly; comparisons of their total-strategy cost to a revised trajectory require the common historical cost, while post-update costs are directly comparable. Full-result review remains pending.

The reporter owner is resolving the final presentation contracts before the scientific snapshot: include adapter-test cases and direct mutation in comparisons, require complete declared-seed pairs for each campaign interval, mark stage completion only after all expected tasks finish, and distinguish fresh-reference total timing. These reporting checks complement the accepted implementation and GPU operation. Scientific source and test campaigns may proceed once the coordinator freezes the final report/workflow checks and manifest.
