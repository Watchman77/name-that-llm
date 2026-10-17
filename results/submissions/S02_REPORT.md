# S02 — official public score 0.549

**Succeeded: 54.9% public balanced accuracy**, observed 2026-10-17T14:02:20.265Z. S01 was 35.4%, so the displayed gain is **19.5 percentage points**. The competition Submissions page shows success and 0.549 for our saved V1, independently confirmed by our leaderboard row.

At this check: **rank 177 of 1,204 teams**, tenth place 55.9%, leader 59.3%. We are 1.0 percentage point behind tenth and 4.4 behind the leader. The public board uses approximately 40% of test data; final private score/rank remain unknown. Current rank is not a final placement.

## Candidate and attribution

This is the unchanged Lucas Boesen **LoRA x Adaptive Centering** V3 (`359126546`), privately reproduced as our V1 (`359332774`). It matches the source author's displayed 0.549. This verifies reproduction at displayed score precision, not identical hidden predictions, a new algorithm, or an independent publication contribution. Original credits, Apache-2.0 licence and resource versions are preserved.

Exact code-cell SHA256: `e3d97306f3185ea5f73d7904d81331b27803e3ceb45970596590ec306969762e`.

Our preview succeeded in 1884.8 seconds / 31m 25s. All stages and both LoRA anchor checks passed. One official S02 was accepted after verification; Kaggle reran the selected notebook with hidden test data. This was not a manual upload of the three preview predictions. No tuning or extra candidate was added after feedback. No paid services or manual final-selection changes were made.

Verification limits: Parquet UI preview is unsupported, so no independent local binary round-trip was done. Executed construction/assertions checked intended columns, test IDs, row count, unique IDs and labels; the official hidden-test rerun has now succeeded. In-sample sanity checks are technical checks, not independent accuracy. Runtime environment label discrepancies are preserved in the source review and launch receipt.

Decision: retain S02 as the current competitive reference and S01 as the earlier frozen reference. Future improvements require separately recorded validation; do not tune repeated variants against public feedback. The separate supervised DeBERTa trial completed at 22.76% development accuracy and is rejected as a replacement for Stage 03.

[Official submissions](https://www.kaggle.com/competitions/name-that-llm/submissions)

[Our scored notebook version](https://www.kaggle.com/code/watchman777/name-that-llm-stage-05-public-reproduction?scriptVersionId=359332774)
