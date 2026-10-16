# Experiment log

## Baseline 01 — 16 October 2026

- Data: supplied official ZIP, 2,699 labelled passages, 150 prompts, 18 models.
- Checks: no missing required values, duplicate IDs, duplicate exact or whitespace/case-normalised passages, or duplicate prompt/model pairs. 149 prompts cover all 18 models; one covers 17.
- Validation: deterministic `GroupKFold(5)` by `prompt_id`, with fold-local TF-IDF. Same folds should be reused for controlled development comparisons.
- Recipe: word 1–2-grams (80,000 feature cap) + character 3–5-grams (120,000 feature cap), LinearSVC C=1, balanced class weights, seed 20261016, dual=True.
- Pooled balanced accuracy: 0.30187670892368884. Fold values and package versions are in `results/baseline_01/`.
- Recognition extremes: Gemini 3.1 Pro Preview recall 74.67%; Claude Fable 5 recall 6.71%.
- Frequent confusion: Gemini 3.8 Flash → 3.7 Flash, 47/150; reverse, 40/150.
- Execution: sequential code cells in a fresh in-process IPython session; saved figures inspected; metric independently reconciled from the confusion matrix; Parquet round trip passed. Ordinary TCP Jupyter kernel startup was unavailable in the local execution environment.
- No paid API usage or hyperparameter search. No leaderboard score established.

## Next controlled experiments

1. Word-only versus character-only versus combined features, on identical folds.
2. Small, interpretable linguistic feature set, with preprocessing fitted inside each fold.
3. A compatible offline pretrained encoder if simpler features plateau.

Repeated tuning makes these folds development data. Reserve fresh prompts or use nested evaluation for strong research claims. Do not infer prize placement from the first local result.

## Kaggle setup

The notebook is kept private with the competition input attached and internet disabled. Removed the preview sample-ID equality check: the hidden test can have different IDs and row counts. Predictions are checked against the actual supplied test file. Model features, parameters and validation folds are unchanged.

## Stage 02 — fixed feature comparison

- Plan recorded before execution: commit aafa46b8ab9fdef40bb4f88062414524f27a5f8b.
- Same data hash and exact saved folds; only the TF-IDF feature branch changed.
- Word-only: 26.2612%; character-only: 30.8553%; original combined reference: 30.1877%.
- Character lead: +0.6677 percentage points, paired prompt-bootstrap 95% interval −0.5194 to +1.9673. Advantage is not established by this comparison.
- All folds completed with no convergence warnings. Combined OOF predictions were reused. No parameter search or extra candidates were added.
- Full outputs: results/stage_02/. Reproduce with python scripts/feature_ablation.py --data-dir data.
- Decision: keep character-only as a provisional development leader and combined as a reference; next inspect errors and test predeclared style features. The private Kaggle baseline stays unchanged.

## Stage 03 — surface-style features

- Prospective plan commit: 09e2fb41f37fd420caf3184d4d2512b864c0e3e7. Two fixed candidates, exact saved folds, no tuning.
- 111 deterministic features; scaling is fitted inside each training fold.
- Style-only: 29.2637%; character plus style: 35.4131%; references: character-only 30.8553%, combined 30.1877%.
- Gain versus character-only: +4.5578 percentage points; paired 95% prompt-bootstrap interval +2.7783 to +6.3379. All five folds improve. These are development comparisons conditional on fitted predictions.
- Improvement occurs in all three fixed length bins, but some per-model regressions remain. Claude Opus 4.8 recall drops from 24.00% to 16.67%.
- All 20 fold records and 2,699 OOF rows saved; metrics independently reconciled; zero convergence warnings.
- Frozen candidate source: notebooks/Name_That_LLM_Stage_03.ipynb and scripts/stage03_candidate.py. It passed private offline Kaggle verification in version 2, reproducing 35.41% (105.7-second successful run); baseline version 1 is preserved.
- Full plan, results and caveats: results/stage_03/REPORT.md. No paid API usage or competition submission.

## S01 — first official competition submission — 16 October 2026

- Frozen Stage 03 candidate, private Kaggle version 2, scriptVersionId 359158002; output `submission.parquet`. Source SHA256: bee527c28fdc4a163c023d2a560d9a45066c52de8017dd69472f8e9cdbdfe294.
- Hidden-test rerun status: **Succeeded**. Public balanced accuracy, exactly as displayed: **0.354 (35.4%)**. Prompt-grouped development score: 0.35413124533929907. Agreement is at display precision; the small arithmetic difference is below that resolution.
- Public rank observed at 2026-10-16T23:11:20Z: **651 of 914 ranked teams**. Tenth-place public score: 0.549; leader: 0.584. The gap to tenth was about 19.5 percentage points. Rank and cutoffs can change.
- Approximately 40% public / 60% private test split as described on the leaderboard; final private score and ranking are unknown. No hidden per-class labels are available.
- Only one submission made. No recipe changes after seeing the score, no paid API usage, and no manual final-selection change.
- Decision: keep S01 frozen and predeclare a compatible offline pretrained-encoder comparison on the same development folds. Choose trials using development evidence rather than repeated leaderboard feedback.
- Full record and prospective feedback protocol: `results/submissions/`.
