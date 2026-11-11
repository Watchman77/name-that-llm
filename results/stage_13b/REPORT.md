# Stage 13B — fixed balancing screen complete

Verified 11 November 2026. [Private Kaggle V2 368298172](https://www.kaggle.com/code/watchman777/notebook8382801096?scriptVersionId=368298172) succeeded in **23.7 seconds** on CPU. The saved code matched all three cells in the [prospective freeze](https://github.com/Watchman77/name-that-llm/commit/c4c14b2fe623a40d651347291c47aa144b7b25b2), with Internet off and only the pinned private GLM V3 output **368292220**. The source hash, immutable probability cache, original natural-fold baseline and saved-report readback all passed. No model was trained.

**Decision: retain the existing balancing setting.** The fixed no-balancing arm did not pass the reduced proxy screen. Positive deltas favour turning balancing off; metrics are pooled 18-class development balanced accuracy. Both arms use identical rows within each scenario and separate scoring batches for the five held-out folds.

| Scenario | Rows | Uniform | Off | Off − uniform (pp) | Conditional 95% interval (pp) |
|---|---:|---:|---:|---:|---:|
| Original folds | 2,699 | 46.083% | 45.896% | -0.186 | [-1.112, 0.776] |
| Balanced subset | 1,800 | 46.000% | 45.833% | -0.167 | [-1.485, 1.162] |
| Skew A | 1,800 | 45.059% | 45.078% | +0.019 | [-1.324, 1.272] |
| Skew B | 1,800 | 46.929% | 46.277% | -0.652 | [-2.049, 0.712] |

Both skew gains and both positive lower bounds failed the gates. Natural and matched balanced losses stayed within the 0.5-point tolerance; the per-class loss gate failed (worst: **gpt-6-astra**, -10.77 pp in Skew B). All four conditional intervals include zero. This supports stopping this fixed ablation, with no automatic sweep; it does not establish uniform balancing as superior in all settings.

These are exploratory comparisons on the standalone GLM cache. The masks simulate class mixtures and alter which passages are represented. The intervals resample prompts after predictions are fixed; they do not capture refitting or new-batch uncertainty. No full S07 balancing ablation, hidden class-prior claim, new official submission or final-selection change was performed. **S07 remains our best verified official public score at 65.1%.**

- [Prospective protocol](../../plans/STAGE_13B_BALANCING_STRESS.md)
- [Verified run receipt](KAGGLE_V2_RECEIPT.json)
- [Actual aggregate report](ACTUAL_V2_REPORT.json)
- [Aggregate reconciliation](AGGREGATE_RECONCILIATION.json)
