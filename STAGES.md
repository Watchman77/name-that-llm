# Project stages and current state

| Stage | Status | Evidence / next decision |
|---|---|---|
| 01 — data checks and fixed baseline | Complete | 30.19% balanced accuracy; private offline Kaggle version 1 ran successfully. |
| 02 — word/character comparison | Complete | Word 26.26%, character 30.86%, combined 30.19%. Character lead remains uncertain; retain both references. |
| 03 — error diagnosis and surface-style features | Complete and Kaggle-verified | Character + style: 35.41%; improved all folds; private offline version 2 succeeded. |
| 04 — offline pretrained encoders / complementary models | Conditional | Proceed if simpler controlled experiments justify the added compute and compatible resources are available. |
| 05 — candidate freeze and competition submission | Pending | Verify offline full-size execution, select versions deliberately, then obtain an official score. |
| Publication assessment | Pending | Review related work and establish a contribution with stronger independent/nested evaluation. Ranking alone is insufficient. |

## How each experiment is tracked

Before a run, record the question, data hash, feature/model changes, folds and primary metric. Afterward, retain every declared result, held-out predictions, per-model recall, warnings, timing and package versions. Interpret uncertainty and regressions before selecting the next experiment.

The same five folds are development data. No supplied prompt is now a new untouched holdout. Baseline 01 remains frozen for reference. Kaggle baseline version 1 is preserved. Private version 2 verified the Stage 03 candidate at 35.41%; no competition submission has been made.

Stage 02 plan: `plans/STAGE_02_FEATURE_COMPARISON.md`.
Stage 02 report: `results/stage_02/REPORT.md`.

Stage 03 plan: `plans/STAGE_03_STYLE_FEATURES.md`.
Stage 03 report: `results/stage_03/REPORT.md`.
