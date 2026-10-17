# Project stages and current state

## Competition goal

Target first place in the final private evaluation. At the 17 October 2026 UK-time check, the public leader scored 0.584 versus S01's 0.354, a 23.0-percentage-point gap. This is a time-specific benchmark; the private result can differ. Choose experiments using development evidence rather than repeated public-score tuning.

| Stage | Status | Evidence / next decision |
|---|---|---|
| 01 — data checks and fixed baseline | Complete | 30.19% balanced accuracy; private offline Kaggle version 1 ran successfully. |
| 02 — word/character comparison | Complete | Word 26.26%, character 30.86%, combined 30.19%. Character lead remains uncertain; retain both references. |
| 03 — error diagnosis and surface-style features | Complete and Kaggle-verified | Character + style: 35.41%; improved all folds; private offline version 2 succeeded. |
| 04 — offline pretrained encoders / complementary models | Frozen MPNet trial complete; supervised trial queued | MPNet alone 14.59%; added to character/style 34.12%, below the 35.41% reference. Both rejected. Stage 04B private offline T4 version 1 is queued (359319680); GPU preflight and results pending. |
| 05 — candidate freeze and competition submission | First official submission complete | S01 succeeded: public balanced accuracy 35.4%. Further candidates and final selection remain pending. |
| Publication assessment | Pending | Review related work and establish a contribution with stronger independent/nested evaluation. Ranking alone is insufficient. |

## How each experiment is tracked

Before a run, record the question, data hash, feature/model changes, folds and primary metric. Afterward, retain every declared result, held-out predictions, per-model recall, warnings, timing and package versions. Interpret uncertainty and regressions before selecting the next experiment.

The same five folds are development data. No supplied prompt is now a new untouched holdout. Baseline 01 remains frozen for reference. Kaggle baseline version 1 is preserved. Private version 2 verified the Stage 03 candidate at 35.41% and succeeded in the official hidden-test rerun as S01, with public score 0.354. Public rank was 651 of 914 ranked teams at the recorded check; final private results remain unknown. See `results/submissions/S01_REPORT.md` for the observation time, precision limits and next decision.

Stage 02 plan: `plans/STAGE_02_FEATURE_COMPARISON.md`.
Stage 02 report: `results/stage_02/REPORT.md`.

Stage 03 plan: `plans/STAGE_03_STYLE_FEATURES.md`.
Stage 03 report: `results/stage_03/REPORT.md`.

Stage 04 plan: `plans/STAGE_04_FROZEN_ENCODER.md`.
Stage 04 report: `results/stage_04/REPORT.md`.
Next fixed protocol: `plans/STAGE_04B_SUPERVISED_ENCODER.md` (private GPU trial queued; results pending).
