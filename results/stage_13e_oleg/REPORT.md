# Stage 13E: Oleg corpus audit closed the external route

Private CPU audit V1 [368756393](https://www.kaggle.com/code/watchman777/notebook9981b97a7e/log?scriptVersionId=368756393) completed in **27.7 seconds** on 12 November 2026. It used the exact mounted Kaggle Version 1 short-view assets and checked all eight pinned file hashes. The notebook validated the joins, verified its own source hash and read the saved aggregate report back.

**Decision: zero complete 18-model prompt groups, against a minimum of 30.** The competition-like 40–250-word view contained 24,025 answer cells across 12,032 publisher prompt groups. Neither the publisher group IDs nor exact verified user prompt text yielded a single group covering all 18 labels, even before the refusal screen. This applies to the short view, not the unrestricted long-form corpus.

| Diagnostic | Count |
| --- | ---: |
| Short answer cells | 24,025 |
| Publisher prompt groups | 12,032 |
| Complete 18-model groups before screening | 0 |
| Exact complete groups after the refusal-marker screen | 0 |
| Gemini 3.6 Flash cells remaining after the screen | 15 |
| Minimum required complete groups | 30 |

The cell counts alone give an independent, looser upper bound of 15 groups after the fixed refusal-marker screen, even if those cells could be matched across all 18 labels. Gemini 3.8 had 47 SpeechMap cells, all flagged by the screen, and 24 remaining Artificial Analysis cells with unresolved source licence flags. No prompt novelty or official overlap checks were necessary, because they could only shrink a pool already below threshold.

Neither this source nor the [Prvsiyan audit](../stage_13e/REPORT.md) admitted a panel. Close the external validation route for this experiment. Do not present the planned Stage 13C vs. 13D comparison as completed. No models were fitted or scored, and S07 remains the best verified official public balanced accuracy at **65.1%**.

The next bounded development check is to measure whether S07's one-label-per-model assignment has any eligible inferred groups of 12–18 passages on the existing held-out folds. If it has coverage, compare the same frozen proxy probabilities with that rule on versus off, keeping text grouping and Sinkhorn fixed. This is a production-relevant proxy on reused prompts, not full-S07 out-of-fold validation or a new official score.

[Frozen protocol](../../plans/STAGE_13E_OLEG_BOUND.md) · [Aggregate audit](ACTUAL_V1_BOUND.json) · [Kaggle receipt](KAGGLE_V1_RECEIPT.json)
