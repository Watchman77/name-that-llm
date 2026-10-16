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
