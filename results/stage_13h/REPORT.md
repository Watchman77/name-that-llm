# Stage 13H: residual errors concentrate within the Gemini family

Private offline CPU Kaggle [V1 368859699](https://www.kaggle.com/code/watchman777/notebook40a2743b57/log?scriptVersionId=368859699) succeeded in **45.7 seconds** on 12 November 2026. The [frozen source](https://github.com/Watchman77/name-that-llm/commit/b6e42fb616bad079ccddbff898f0931ddd160b73), three input versions, and Internet-off CPU configuration were verified on the saved version. The original Stage 13C evidence and Stage 13F geometry and confusion replayed exactly: **61.461844% development balanced accuracy and 1,659 correct**. No new variant was scored.

| Passages | Rows | Correct before assignment | Correct after assignment | Final errors |
| --- | ---: | ---: | ---: | ---: |
| Assigned by the existing 12–18 rule | 1,889 | 899 (47.59%) | 1,274 (67.44%) | 615 |
| Left unassigned | 810 | 385 (47.53%) | 385 (47.53%) | 425 |

The two strata were nearly identical in row accuracy before assignment. The post-assignment gap is descriptive; it does not estimate what would happen if assignment were extended to the unassigned passages.

Of the 425 unassigned errors, **183** ranked the correct model second, **103** ranked it third, and **139** ranked it below third. The top-three total is **286 of 425 (67.29%)**. This is an oracle opportunity count, not a gain forecast. Final balanced accuracy within the unassigned stratum was **46.738270%**, with all 18 labels represented.

**91 of 425 errors (21.41%) were between Gemini variants.** Of the 100 errors on passages actually written by one of the four Gemini variants, 91 stayed in the family and nine went to another family. The largest directed confusion was **Gemini 3.7 Flash → Gemini 3.8 Flash** (18 of 56 unassigned Gemini 3.7 passages, across all five folds; eight in the reverse direction). The four leading directed confusions all stayed within Gemini.

A defensible next test is a separately frozen, fold-local discriminator among the four Gemini variants. Apply it only to unassigned passages already predicted as Gemini and preserve family decisions and the existing 12–18 rule. Report paired corrections and regressions against the unchanged baseline; the audit itself establishes no performance benefit from that intervention.

**S07 remains 65.1% official public balanced accuracy.** The competition test was not read or scored, and no submission or final-selection change was made.

[Protocol](../../plans/STAGE_13H_RESIDUAL_ERROR_AUDIT.md) · [Aggregate report](ACTUAL_V1_REPORT.json) · [Run receipt](KAGGLE_V1_RECEIPT.json) · [Reconciliation](AGGREGATE_RECONCILIATION.json)
