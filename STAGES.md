# Project stages and current state

## Competition goal

Target first place in the final private evaluation. At an earlier recorded 17 October 2026 check, S02 scored **0.549**, the public leader **0.600** (5.1 percentage points ahead), and tenth place **0.559** (1.0 point ahead). Our rank was **181 of 1,212 teams**. No fresh leaderboard check accompanies this update. S01 remains frozen at 0.354. Public standings are time-specific; the final private result is unknown. The user's request to adopt stronger public methods was completed as one frozen external-baseline reproduction, not an original-method claim or parameter search.

| Stage | Status | Evidence / next decision |
|---|---|---|
| 01 — data checks and fixed baseline | Complete | 30.19% balanced accuracy; private offline Kaggle version 1 ran successfully. |
| 02 — word/character comparison | Complete | Word 26.26%, character 30.86%, combined 30.19%. Character lead remains uncertain; retain both references. |
| 03 — error diagnosis and surface-style features | Complete and Kaggle-verified | Character + style: 35.41%; improved all folds; private offline version 2 succeeded. |
| 04 — offline pretrained encoders / complementary models | Complete; both fixed experiments rejected | Frozen MPNet: 14.59% alone, 34.12% added. Supervised DeBERTa: 22.76% development BA, below the 35.41% reference in every fold; all five folds completed. |
| 05 — candidate freeze and competition submission | S02 succeeded at 54.9% official public BA | S01 remains 35.4%. Unchanged public baseline V1 359332774 reproduced 0.549; rank 177/1204 at observation. Final private evaluation and further original improvements remain pending. |
| 06 — OOF provenance and centering | Audit, lexical test and mixed confirmation complete | Stage 06C mixed-grouping component BA 45.8276% to 46.9518% (+1.1242pp, conditional interval +0.5191 to +1.7179); fixed screen passes. All 2,699 GLM cache rows and five CPU folds complete. Class/context limits remain; official 54.9% unchanged. |
| 06D — fixed available-component ensemble | TECHNICAL_REPRO_FAILURE_NOVALID_ENSEMBLESCORE | V7 359505815 completed 5 folds/15 heads, then failed the strict control gate at 141.6s (151.6s total). Inference paths have zero hard-label changes; 20/32,388 repeated comparisons differ from original C. Call shape ruled out; remaining cause unproven. No valid ensemble/raw screen result; blind refits stopped; official score 54.9%. |
| 06E — canonical CPU evidence | Complete; valid negative development result | V2 359514801 succeeded in 169.5s, all 15 heads/caches verified. Raw proxy 54.8122% to 54.6767% (-0.1356pp), conditional interval -0.5189 to +0.2375; screen not met. Retain original centering; official 54.9% unchanged. Aggregate error analysis identified Gemini 3.7/3.8 as the largest confusion in every context. |
| 06F — fixed Gemini pair diagnostics | Complete; source/replay and independent aggregate validation pass | V4 359521305 succeeded in 20.3s CPU. 275 repeated mutual-confusion observations represent 109 distinct passages; 77 fail in all three contexts. No fits or changed predictions. Original V2 manifest-byte failure preserved; exact-hash constant repair verified by V4. Motivated the fixed G correction; official 54.9% unchanged. |
| 06G — fixed attention-pair complement | Complete; candidate rejected | V2 359531626 succeeded in 22.5s CPU. Raw proxy 54.8122% control to 54.6641% candidate (-0.1481pp; conditional interval -0.6420 to +0.3580), 82 fixed/94 broken repeated decisions. Independent source, numerical and aggregate interval validation pass. Retain control, retire exact rule, inspect representation coverage before a new hypothesis. No official submission; S02 54.9%. |
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

Stage 06D preparation: `results/stage_06d/KAGGLE_RUN.json`; current implementation and limits: `results/stage_06d/REPORT.md`. Exact five-cell packaging was committed in 9db86f28cfd763ccbc3dcc048fb35fe649f6445f; the source-verification block was recovered. Actual Version 3 359500151 failed during input metadata validation before new head fitting; the classes-only safe parser repair passed independent review, was frozen before execution in de19608a, and V5 359502730 completed all five folds but failed strict prediction reproduction. The exact-source-call/diagnostic revision passed saved-source verification in V6 359505487 and V7 359505815 subsequently failed strict reproduction after allfivefolds; diagnostics ruleoutcallshape forhardlabels, but establish no ensemble score.

Stage 06E completed: `results/stage_06e/REPORT.md`; terminal validation: `results/stage_06e/INDEPENDENT_TERMINAL_VALIDATION.json`; source/runtime receipt: `results/stage_06e/KAGGLE_RUN.json`. Pre-execution source freeze ac9d9d4b0487d71b3265b1b2b0f2621803a8b701. The frozen plan preserves its original prospective status; actual V2 completed with a failed improvement screen and no official submission.

Stage 06F completed: `results/stage_06f/REPORT.md`; findings: `results/stage_06f/FINDINGS.md`; terminal validation: `results/stage_06f/INDEPENDENT_TERMINAL_VALIDATION.json`; receipt: `results/stage_06f/KAGGLE_RUN.json`. Initial source freeze baee805c080a06c6d797e68c0a0c1a424756d3a1 and controlled repair freeze 1c9f0650f01d2adc43731876f4fe7e8439f34f48 preserve the failed V2 separately. V4's no-fit diagnostic supplies hypotheses only, with no corrective candidate or new official score.

Stage 06G completed: `results/stage_06g/REPORT.md`; terminal validation: `results/stage_06g/INDEPENDENT_TERMINAL_VALIDATION.json`; source/runtime receipt: `results/stage_06g/KAGGLE_RUN.json`; next prerequisite: `results/stage_06g/REPRESENTATION_COVERAGE.md`. Pre-execution freeze cbdd06c11bb62d72c04781bf7c75bceae08a59fb preceded the single actual candidate. The prospective plan remains unchanged; actual performance failed the fixed screen, and no variant ran.


## Stage 07 — external corpus audit (2026-10-18)

Actual V1 short-corpus assets audited: 24,025 answer cells / 24,015 unique response IDs across 18 labels, with hashes and joins verified. No official-training full-response overlaps under three normalizations; bounded character retrieval found no close matches. Licence/source evidence and prompt independence remain under review. The conservative flag-only subset covers ten labels; SpeechMap covers all eighteen but retains output-terms-review flags. One primary-source Fable check matched 352 cells. Training has not started; the prospective matched two-arm protocol is specified, and its runner is not implemented. Best previously verified official S02 remains 0.549. See [report](results/stage_07/REPORT.md), [protocol](results/stage_07/PROTOCOL.md), and [audit](results/stage_07/audit.json). No paid services, laptop access, submissions or final-selection changes.
