# Revision recovery experiment results (1014_3)

Updated 2026-10-14 18:43 PDT.

A saved state, or root, is the intermediate molecular representation captured when the property goal changes. The goal reversal changes endpoint 1 from an increase target to a decrease target while measurements and the endpoint 2 target stay fixed. Strategies share the original noise and prefix; rollback restores an earlier state, while restart uses the initial noise.

The study asks three questions: does rolling back before continuing help; can independent suffix trials recover roots that direct continuation misses, and can an oracle identify promising proposals; can a learned state adapter improve continuation on validation and test campaigns? Each success is one of 16 retained output slots. Raw validity counts every candidate attempt; selected-root validity describes the retained candidate per root.

The full-cap method table uses each method's configured proposal budget, and those budgets differ. The time curves enforce the same wall-time cap per comparison and plot actual charged time; a point below its cap shows time left when the next complete batch did not fit. Full-cap success counts therefore describe each method's configured run, while the curves compare success under matched time limits.

Main test stage: **complete**. Adapter test: **skipped**.

## Final test results

| Stage | Strategy | Audited set | Goal hits / 16-slot opportunities | Raw valid / attempts | Selected-root validity | Total charged seconds |
|---|---|---|---:|---:|---:|---:|
| test | Continue from saved state | all | 153/768 | 11686/12288 | 0.951 | 159.8 |
| test | Continue from saved state | verified both goals | 114/192 | 2837/3072 | 0.924 | 39.9 |
| test | Restart from original input | all | 237/768 | 11659/12288 | 0.949 | 252.7 |
| test | Restart from original input | verified both goals | 144/192 | 2820/3072 | 0.918 | 63.1 |
| test | Conditional fresh-start sampling | all | 292/768 | 72940/76800 | 0.950 | 984.1 |
| test | Conditional fresh-start sampling | verified both goals | 158/192 | 17637/19200 | 0.919 | 245.8 |
| test | Direct mutation search | all | 195/768 | 2431/12288 | 0.198 | 5.2 |
| test | Direct mutation search | verified both goals | 112/192 | 740/3072 | 0.241 | 1.5 |
| test | oracle adaptive suffix | all | 79/768 | 11702/12288 | 0.954 | 77.4 |
| test | oracle adaptive suffix | verified both goals | 65/192 | 2871/3072 | 0.938 | 19.4 |
| test | oracle direct mutation | all | 172/768 | 2431/12288 | 0.734 | 5.2 |
| test | oracle direct mutation | verified both goals | 101/192 | 740/3072 | 0.768 | 1.5 |
| test | oracle independent restart | all | 160/768 | 11687/12288 | 1.000 | 163.2 |
| test | oracle independent restart | verified both goals | 96/192 | 2816/3072 | 1.000 | 40.7 |
| test | oracle select same suffix pool | all | 79/768 | 11699/12288 | 0.954 | 71.2 |
| test | oracle select same suffix pool | verified both goals | 65/192 | 2869/3072 | 0.938 | 17.7 |
| test | Prior-model fresh-start sampling | all | 271/768 | 73319/76800 | 0.955 | 986.8 |
| test | Prior-model fresh-start sampling | verified both goals | 157/192 | 17893/19200 | 0.932 | 246.8 |
| test | randomized suffix | all | 79/768 | 11699/12288 | 0.954 | 71.2 |
| test | randomized suffix | verified both goals | 65/192 | 2869/3072 | 0.938 | 17.7 |
| test | Roll back 25% | all | 187/768 | 11744/12288 | 0.956 | 222.5 |
| test | Roll back 25% | verified both goals | 130/192 | 2883/3072 | 0.938 | 55.8 |
| test | Roll back 50% | all | 153/768 | 11708/12288 | 0.953 | 191.0 |
| test | Roll back 50% | verified both goals | 114/192 | 2848/3072 | 0.927 | 47.7 |

## What the test comparison shows

Across the completed test cases, Conditional fresh-start sampling had 292/768 goal hits, compared with 153/768 for continuing from the saved state. This is the highest observed rate among the deployable test methods in these campaigns and model seeds.
At the largest matched post-update budget (32 seconds), Conditional fresh-start sampling changed the paired success fraction by +0.181 versus continuation (95% campaign-bootstrap interval +0.083 to +0.296).

## Research gates

The source-only training gate did not pass: it found 0 bounded direct-failed roots across 0 training campaigns. This decides whether the adapter is trained; it does not measure test performance.

## Remaining work

All required stages for the recorded gate decisions are complete.

Detailed live report: [live_report.md](../../4_experimental_result/result/revision_recovery_001/live_report.md).
