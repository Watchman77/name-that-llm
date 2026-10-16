# Stage 02 — fixed feature comparison

Recorded before running the two new variants. Baseline 01 stays frozen at 0.30187670892368884 pooled balanced accuracy.

## Question
Does either word-only or character-only TF-IDF outperform the existing combined-feature linear SVM on unseen prompts?

## Frozen design

- Source train.csv SHA256: 40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707.
- Reuse the exact five fold assignments in results/baseline_01/baseline_oof.csv. Never split rows randomly or change fold membership after looking at results.
- Fit vocabulary and IDF on each training fold only. Never use id or prompt_id as prediction features.
- Word-only: word 1–2-gram TF-IDF, min_df=2, sublinear_tf=True, max_features=80000.
- Character-only: character 3–5-gram TF-IDF, min_df=3, sublinear_tf=True, max_features=120000.
- Both variants: LinearSVC(C=1.0, class_weight='balanced', random_state=20261016, max_iter=10000, dual=True), all other settings as in baseline 01.
- Combined: reuse saved baseline predictions; do not retrain it simply to get a different number.
- Primary metric: pooled out-of-fold balanced accuracy. Secondary: fold scores, per-model recall, and convergence warnings.
- Calculate paired 95% percentile bootstrap intervals for score differences by resampling the 150 prompt clusters 2,000 times with seed 20261016. These intervals describe variation across existing held-out prompts; they do not capture refitting uncertainty or establish confirmatory significance.

## Interpretation and stop rule

Run exactly the two declared new variants. No parameter tuning, extra feature variants or paid APIs in this stage. The highest development score is the provisional reference for the next stage; report all three scores, including regressions. Do not claim a superior method from overlapping/noisy estimates or equate a development gain with prize placement. If a run fails checks or convergence, record it rather than silently replacing its settings.

These folds are development data. All 150 prompts have already contributed to baseline evaluation, so none can honestly be called a new untouched holdout. Stronger research claims need nested evaluation or genuinely new prompts later.

The existing private Kaggle baseline is preserved during these comparisons. No competition submission is part of this stage.
