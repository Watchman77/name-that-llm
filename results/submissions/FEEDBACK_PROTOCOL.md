# Official submission feedback protocol

- S01 uses the frozen Stage 03 candidate, Kaggle version 2 (scriptVersionId 359158002), with development balanced accuracy 0.35413124533929907. The exact code fingerprint and selected output are recorded in S01.json.
- Record the first successful public score exactly as shown, the score gap in percentage points and any visible rank with its observation time. Do not infer a private final score.
- Development validation and the public test use different passages, so the score gap alone does not identify a cause or establish a performance change. Hidden labels and per-class counts are unavailable; do not invent error bars or a per-class explanation for the public score.
- Select future experiments from fixed-fold development evidence, and log every declared result. Do not tune repeatedly against the public leaderboard or treat its current rank as final placement.
- A failed rerun is a technical outcome, not a valid model score. Inspect the stated error before any retry, and preserve the original candidate and failure record.
