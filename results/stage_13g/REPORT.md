# Stage 13G: retain the 12–18 group cutoff

Private CPU Kaggle [V4 368807303](https://www.kaggle.com/code/watchman777/notebook219dd00182/log?scriptVersionId=368807303) succeeded in **63.6 seconds** on 12 November 2026. The exact [frozen source](https://github.com/Watchman77/name-that-llm/commit/095201965b0b3a7efd98cfa20f5af97bd9db659d), the three input versions and Internet off were verified in the source-only V3 and in actual V4. V1 was excluded because Kaggle's saved Internet flag was enabled; V2 was a source-only check with the same issue. No score from either version was used.

**Decision: keep the existing 12–18 passage rule.** All eight candidate groups (82 passages) passed the geometry screen: each had one true prompt and no repeated true model label. The extension fixed 16 predictions and broke eight, but it did not pass the prespecified improvement screen.

| Assignment | Development balanced accuracy | Correct / 2,699 |
| --- | ---: | ---: |
| Frozen 12–18 rule | 61.461844% | 1,659 |
| Also assign in 10–11 passage groups | 61.757892% | 1,667 |

The pooled change was **+0.296048 percentage points** (exactly 397/1341 pp; conditional 95% interval **[−0.075565, +0.815312] pp**). It fell below the +0.50 pp threshold and the interval included zero. The class guard passed: worst recall loss was 0.671 pp for Claude Fable 5. Of the 82 passages, 39 predictions changed; every prediction outside the extension was identical. The delta was positive in three folds, negative in one, and zero in the fold with no extension groups.

The notebook replayed Stage 13F's geometry and assignment-on confusion exactly, then scored both arms on the same fixed Stage 13C log scores. Labels assessed group purity only after text-only clustering. The interval is conditional on fixed predictions and reused prompts, only eight of which could contribute a change. Neither this result nor the purity check establishes performance on hidden Kaggle data. **S07 stays at 65.1% official public balanced accuracy.** No submission or final-selection change was made.

[Protocol](../../plans/STAGE_13G_SMALL_GROUP_EXTENSION.md) · [Aggregate report](ACTUAL_V4_REPORT.json) · [Run receipt](KAGGLE_V4_RECEIPT.json) · [Reconciliation](AGGREGATE_RECONCILIATION.json)
