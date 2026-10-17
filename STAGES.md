# Project stages and current state

## Competition goal

Target first place in the final private evaluation. At the latest 17 October 2026 check, S02 scores **0.549**, the public leader **0.600** (5.1 percentage points ahead), and tenth place **0.559** (1.0 point ahead). Current rank is **181 of 1,212 teams**. S01 remains frozen at 0.354. Public standings are time-specific; the final private result is unknown. The user's request to adopt stronger public methods was completed as one frozen external-baseline reproduction, not an original-method claim or parameter search.

| Stage | Status | Evidence / next decision |
|---|---|---|
| 01 — data checks and fixed baseline | Complete | 30.19% balanced accuracy; private offline Kaggle version 1 ran successfully. |
| 02 — word/character comparison | Complete | Word 26.26%, character 30.86%, combined 30.19%. Character lead remains uncertain; retain both references. |
| 03 — error diagnosis and surface-style features | Complete and Kaggle-verified | Character + style: 35.41%; improved all folds; private offline version 2 succeeded. |
| 04 — offline pretrained encoders / complementary models | Complete; both fixed experiments rejected | Frozen MPNet: 14.59% alone, 34.12% added. Supervised DeBERTa: 22.76% development BA, below the 35.41% reference in every fold; all five folds completed. |
| 05 — candidate freeze and competition submission | S02 succeeded at 54.9% official public BA | S01 remains 35.4%. Unchanged public baseline V1 359332774 reproduced 0.549; rank 177/1204 at observation. Final private evaluation and further original improvements remain pending. |
| 06 — OOF provenance and centering | CPU component audit and one fixed centering test complete | Stage 06B mean component BA 46.57% to 47.25% (+0.68pp, interval +0.09 to +1.30); repeated exact scenario scores. Class regressions and lexical/fp16 limits retained. Stage 06C V1 359405443 queued for actual mixed grouping; no CUDA preflight or result yet. Official 54.9% unchanged. |
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
Completed supervised protocol: `plans/STAGE_04B_SUPERVISED_ENCODER.md`; results: `results/stage_04b/REPORT.md` (22.76%, rejected).

Completed public reproduction protocol: `plans/STAGE_05_PUBLIC_REPRODUCTION.md`; official S02 result: `results/submissions/S02_REPORT.md` (0.549). Full source review: `results/stage_05/SOURCE_REVIEW.md`. The separate 55.2% recipe was excluded because its scored version requires a private checkpoint.

Stage 06 audit: `results/stage_06/ADVICE_AUDIT.md`; next protocol: `plans/STAGE_06_OOF_AND_CENTERING.md`. 70% is a stretch goal (+15.1 percentage points), not a forecast.

Stage 06B completed result: `results/stage_06b/REPORT.md`; fixed CPU protocol: `plans/STAGE_06B_CACHED_CENTERING.md`; next prerequisite and confirmation: `plans/STAGE_06C_MIXED_GROUP_VALIDATION.md`.

Stage 06C: `results/stage_06c/KAGGLE_RUN.json`. Next fixed ensemble endpoint: `plans/STAGE_06D_FIXED_ENSEMBLE_PROXY.md`.
