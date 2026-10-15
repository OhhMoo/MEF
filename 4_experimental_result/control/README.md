# Controlled experiments

The first exploratory generator and overnight comparison completed successfully.
This folder holds
the overnight supervision study:

- `overnight.py`: explicit paired tasks, shared representations, training and sampling grids.
- `study_metrics.py`: validity, useful yield, scaffold diversity and validation distribution comparisons.
- `study_analysis.py`: seed-paired effects, common-update trajectories and scheduler GPU accounting.

The study changes intermediate categorical supervision through λ=0, 0.1 and 1,
under both trainable and common frozen terminal decoders. See the
[experiment plan](../../1_research_file/detail_operation/07_overnight_experiment_plan.md)
for the design and monitoring commands, and the
[experiment results](../../1_research_file/result_log/experiment_results_1009.md)
for the explanation in plain language.
