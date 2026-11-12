# Stage 13F: retain S07's dense-group label assignment

Private CPU Kaggle [V2 368776259](https://www.kaggle.com/code/watchman777/fork-of-notebook0770477e92/log?scriptVersionId=368776259) succeeded in **61.8 seconds** on 12 November 2026. The saved V1 source-only snapshot and actual V2 run both matched the [pre-run source freeze](https://github.com/Watchman77/name-that-llm/commit/621361eb044ae3f2fdb2c29fbb7c81f2b56d6d4d), the three input versions, CPU and Internet off. Stage 13C's baseline confusion digest replayed exactly. The notebook read back its saved aggregate report.

**Decision: retain the existing rule.** On the fixed Stage 13C GLM/DeBERTa development proxy, switching only the dense-group assignment on raised pooled balanced accuracy by **+13.896843 percentage points** (exactly 55907/4023 pp; conditional 95% interval **[+11.718218, +16.122660] pp**).

| Assignment | Development balanced accuracy | Correct / 2,699 |
| --- | ---: | ---: |
| Off, highest score per row | 47.565001% | 1,284 |
| On, different labels within eligible inferred groups | 61.461844% | 1,659 |

The rule corrected **502** decisions and broke **127** (net +375; 900 changed overall). Rows outside the eligible groups did not change. All five folds and all 18 class recalls improved; the smallest class gain was **+4.667 pp** for GLM 5.3. All three prespecified criteria passed.

Text-only grouping produced 389 inferred groups, of which **112** met the 12–18 row eligibility rule. They covered **1,889 of 2,699 rows (69.99%)**. All 112 were pure with respect to the true prompt and contained no repeated true model label; those labels were used only after grouping to assess purity. The evidence scores, fold-level Sinkhorn balancing and every other step were identical between arms. The production typography guard was absent from both proxy arms.

The conditional interval resamples previously used true prompts within fixed held-out folds and does not estimate full-pipeline uncertainty or the effect on a new batch. **S07 already uses this assignment.** This test supports keeping it; it supplies no increment to S07's official **65.1%** balanced accuracy. No submission or final-selection change was made.

A useful follow-up is a coverage audit of the unassigned smaller inferred groups before testing any extension. A change should be compared separately against the retained 12–18 rule, and any subsequent production test must preserve the original control.

[Protocol](../../plans/STAGE_13F_ASSIGNMENT_ABLATION.md) · [Aggregate report](ACTUAL_V2_REPORT.json) · [Run receipt](KAGGLE_V2_RECEIPT.json) · [Reconciliation](AGGREGATE_RECONCILIATION.json)
