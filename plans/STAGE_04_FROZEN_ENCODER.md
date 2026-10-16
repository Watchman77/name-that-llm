# Stage 04 — fixed frozen-encoder comparison

## Question and motivation

Can a public pretrained contextual representation improve the frozen character/style classifier on unseen prompts? S01 public balanced accuracy is 0.354; the leading public score was 0.584 at the 17 October 2026 UK-time check. These leaderboard observations motivate a larger representation change but do not identify its cause or predict this trial's result.

## Resource and budget

- Encoder: `sentence-transformers/all-mpnet-base-v2`, immutable Hugging Face revision `e8c3b32edf5434bc2275fc9bab85f82640a19130`.
- Model card: https://huggingface.co/sentence-transformers/all-mpnet-base-v2 . It declares Apache-2.0, English, 768-dimensional mean-pooled embeddings and a 384-word-piece default limit. Retain the upstream model card and licence when packaging weights.
- Public, ungated pretrained weights only. No external labelled competition data, generated passages, paid APIs or paid compute.
- Local CPU, float32, up to six torch threads, batch size 8; at most 60 minutes total encoding/evaluation. Stop and record an incomplete trial if resources fail. A failed attempt may resume the identical recipe; no silent model substitution or parameter changes.
- Download resources first, then execute all modelling with `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` and `local_files_only=True`. A later Kaggle candidate requires an attached offline checkpoint and a separate full hidden-rerun check; this experiment is not a submission.

## Fixed design

Commit this plan before computing the new embeddings or fitting new classifiers.

- Official train SHA256: `40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707`; all 2,699 rows, 150 prompts, 18 labels.
- Reuse the exact five saved prompt-grouped folds from Stage 03. Neither IDs nor prompt IDs enter the model.
- Frozen encoder, eval mode, no gradient updates. Fast tokenizer, original text unchanged, dynamic batch padding, truncation at 384 total tokens, attention-mask mean pooling, then L2 row normalization. Report truncation incidence. Embeddings may be computed once for all rows because the encoder is fixed independently of competition data and labels; no corpus-fitted transformation is performed here.
- Candidate A `mpnet_only`: normalized 768-dimensional embeddings -> LinearSVC.
- Candidate B `character_style_mpnet`: concatenate Stage 03 character TF-IDF, fold-fitted scaled/L2-normalized 111 style features, and the unit-normalized MPNet embeddings, with equal block weights -> LinearSVC.
- Both: C=1, balanced class weights, seed 20261016, max_iter=10000, dual=True. Character settings remain 3–5 grams, min_df=3, sublinear_tf=True, max_features=120000 and default L2. Character vocabulary/IDF and style StandardScaler are fitted inside each training fold. No parameter search, reweighting, threshold tuning or additional candidate in this stage.
- Frozen reference: Stage 03 `character_plus_style` OOF predictions (0.35413124533929907). Do not retrain or alter it.

## Evaluation and decision

Primary: pooled OOF balanced accuracy. Save all five fold scores, OOF predictions, per-model recall, length-bin results (<=80, 81–150, >150 words), warnings, execution time, package versions, source/fold/code/weight hashes and the plan commit.

For each candidate versus the reference, report 2,000 paired prompt-cluster bootstrap resamples (seed 20261016), 95% percentile intervals. These describe fixed development predictions; they exclude retraining and model-selection uncertainty. All supplied prompts have already been used in development, so this is not independent confirmation or evidence of publication novelty.

Select a provisional next candidate only after retaining both declared results and examining fold/class regressions. A frozen semantic encoder may emphasize topic rather than model style and may fail. Even a gain is not proof that deep encoders will close the 23.0-point public gap. If it fails, preserve the negative result and design a separate supervised-adaptation stage instead of tuning this trial until it wins. No official submission is part of this stage.
