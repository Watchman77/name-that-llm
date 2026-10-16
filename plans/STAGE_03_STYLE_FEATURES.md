# Stage 03 — fixed surface-style experiments

## Evidence motivating the stage

Stage 02: character-only 30.8553%, combined 30.1877%, word-only 26.2612% balanced accuracy. The character-only lead is uncertain. Both reference models frequently confuse Gemini 3.7 Flash with 3.8 Flash. These observations motivate testing style features; they do not prove which cues explain the scores.

## Prospective design

Record this plan before running either new candidate. Keep the two earlier references frozen and reuse exactly the five folds from results/baseline_01/baseline_oof.csv.

- Training SHA256: 40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707.
- 111 fixed features in scripts/style_features.py: 15 surface/format statistics, 32 ASCII punctuation frequencies, 7 Unicode punctuation frequencies, and 57 predefined function-word frequencies.
- The 15 statistics are uppercase-letter fraction, digit fraction, whitespace fraction, newline fraction, non-ASCII fraction, mean/std word length, mean/std words per sentence, contraction fraction, bullet/heading/table line fractions, fence markers per nonempty line and blank-line fraction.
- No raw character count, word count, passage ID, prompt ID or model label is a prediction feature. Counts are used only as denominators or to compute the declared ratios/averages.
- Fixed word regex: ASCII words with optional apostrophe contraction. Sentence boundary heuristic: runs of dot/exclamation/question characters. These are simple heuristics, not a linguistic parser.
- Candidate A, style_only: deterministic extraction -> StandardScaler -> L2 row Normalizer -> LinearSVC.
- Candidate B, character_plus_style: concatenate character TF-IDF (3–5 grams, min_df=3, sublinear_tf=True, max_features=120000; default L2) and the scaled/L2-normalized style block, without tuned block weights -> LinearSVC.
- Both candidates: C=1.0, balanced class weights, random_state=20261016, max_iter=10000, dual=True. Vocabulary, IDF and scaling are learned from each training fold only.
- Primary metric: pooled out-of-fold balanced accuracy. Secondary: all five fold scores, per-model recall, convergence warnings and runtime. Reuse saved character-only and combined OOF predictions.
- Report paired prompt-cluster bootstrap 95% percentile intervals for each new candidate versus both references, 2,000 resamples, seed 20261016. This is descriptive development uncertainty conditional on the fitted predictions, without refitting/selection uncertainty or multiplicity-adjusted confirmatory claims.
- Diagnose performance by prespecified word-count bins <=80, 81–150 and >150, with row counts and class coverage. Do not treat a bin with missing classes as a full 18-class score.

## Stop and interpretation rules

Run exactly these two candidates; no extra parameter trials, feature selection, block reweighting or new data in this stage. Record every result and retain original references even if additions fail. A higher development score is provisional and does not establish leaderboard performance or novelty. No supplied prompt is a new untouched holdout. No Kaggle submission, external API use or spending is part of this stage.
