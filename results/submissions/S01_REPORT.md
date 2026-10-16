# S01 — first official Kaggle submission

## Outcome

The frozen Stage 03 character-TF-IDF plus 111-style-feature candidate was submitted as notebook **version 2**, scriptVersionId **359158002**, with output `submission.parquet`. Kaggle reran it on the hidden test data and marked the submission **Succeeded**.

| Measure | Observed value |
|---|---:|
| Public balanced accuracy, exactly as displayed | 0.354 |
| Public balanced accuracy as a percentage | 35.4% |
| Prompt-grouped development balanced accuracy | 35.4131% |
| Public rank at this check | 651 of 914 ranked teams |
| Tenth-place public score at this check | 0.549 |
| Leading public score at this check | 0.584 |

The public score and development score agree at the displayed precision. The computed gap based on the displayed public value is about −0.013 percentage points, which is below its display resolution and should not be interpreted as a measured performance change.

The leaderboard says approximately 40% of test data contributes to the public score, with the other 60% used for final results. This is the first official public result, not the final private score. Rank and prize thresholds are time-specific observations. The current gap to the tenth-place public score is about **19.5 percentage points**.

## Interpretation and next decision

The full submission pipeline works on the hidden rerun, and the first public result is consistent with development validation. This does not establish final placement or broad robustness. Hidden labels and per-class counts are unavailable, so no per-class diagnosis of the public score is possible.

The gap to the current prize positions justifies testing a stronger representation, such as a compatible publicly available pretrained text encoder, against the frozen character/style model on the same development folds. Record the exact encoder, licence, compute budget and candidate set before running. Use development evidence to choose trials; avoid repeated tuning to this single public score. No new spending is authorised by this record.

## Evidence

- [Submissions](https://www.kaggle.com/competitions/name-that-llm/submissions)
- [Public leaderboard](https://www.kaggle.com/competitions/name-that-llm/leaderboard)
- [Frozen notebook version](https://www.kaggle.com/code/watchman777/name-that-llm-baseline-01?scriptVersionId=359158002)
- `S01.json` preserves the displayed score, observation times and source fingerprint.

No extra candidate was submitted, no recipe was changed after seeing this score, and the notebook remains private. Final-submission selection was left to Kaggle's existing automatic selection behaviour.
