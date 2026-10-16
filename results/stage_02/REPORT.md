# Stage 02 — feature comparison results

## Decision

Character-only TF-IDF is the provisional development leader at **30.86% balanced accuracy**, compared with **30.19%** for combined features. Its advantage is **0.67 percentage points**; the paired prompt-bootstrap 95% percentile interval is **−0.52 to +1.97 points**. This does not establish a reliable advantage. Keep both character-only and combined models as references. Word-only is weaker here.

## All results

| Variant | Pooled balanced accuracy | Change vs combined | Paired bootstrap 95% interval for change |
|---|---:|---:|---:|
| Character-only | 30.8553% | +0.6677 percentage points | −0.5194 to +1.9673 points |
| Combined baseline | 30.1877% | Reference | — |
| Word-only | 26.2612% | −3.9264 percentage points | −5.2237 to −2.5945 points |

All three use the same saved five prompt-grouped folds and class-weighted linear SVM settings. Word and character TF-IDF settings were fixed before the two new runs. Combined predictions were reused from baseline 01. All runs have zero convergence warnings. The local code checked source hashes, row/label alignment, prompt separation, complete fold coverage and the balanced-accuracy calculation from per-model recalls.

The plan was recorded before execution in commit `aafa46b8ab9fdef40bb4f88062414524f27a5f8b`. No tuning, extra variants, paid APIs or Kaggle submissions were performed in this stage. The saved Kaggle version remains baseline 01.

## Limits

These are development results, not a leaderboard score. The intervals resample 150 prompt clusters 2,000 times with paired predictions and seed 20261016. They are conditional on these fixed OOF predictions and exclude refitting and model-selection uncertainty. All supplied prompts have already contributed to development; none is an untouched final holdout.

## Next stage

Inspect the remaining confusions and test a small, fixed set of surface-style features alongside character TF-IDF: punctuation, capitalisation, whitespace/paragraph structure, markdown and function-word frequencies. Fit any scaling or feature selection inside each training fold. Keep the same folds, predeclare the next candidates, and report all outcomes. Do not infer that punctuation caused this result merely because character features perform better than words here.

## Reproduce

From the project root, extract the original competition data into `data/`, then run:

```bash
python scripts/feature_ablation.py --data-dir data
```

Machine-readable outputs in this directory contain full pooled/fold metrics, per-model recall and all held-out predictions. Raw passage text is excluded.
