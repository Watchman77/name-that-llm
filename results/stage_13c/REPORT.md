# Stage 13C — fixed DeBERTa complement passes development screen

Verified 11 November 2026. [Private Kaggle V2 368302297](https://www.kaggle.com/code/watchman777/notebook1d8c573cc3?scriptVersionId=368302297) succeeded in **52.4 seconds** on CPU. Both pinned input versions and all three saved code cells match the [prospective freeze](https://github.com/Watchman77/name-that-llm/commit/5de111e93b08cfaafd0f9a68889401cb08953060), with Internet off. The exact GLM cache, frozen pretrained DeBERTa cache, both GLM controls and saved-report readback passed.

**The fixed 75% GLM / 25% DeBERTa probability blend improved pooled development balanced accuracy from 46.082525% to 47.565001% (+1.482476 pp; conditional paired prompt 95% interval [+0.594563, +2.444444] pp).** Both arms used the same uniform balancing within each full held-out fold. All five fold deltas were positive (+1.667, +0.185, +2.778, +1.852, +0.920 pp). Balanced DeBERTa alone scored 45.789709%; the combined result supports complementary errors on this proxy.

The candidate corrected **92** decisions and broke **52**; 276 predictions changed. Pooled correct counts reconcile from **1244 to 1284**. The largest class gain was **claude-sonnet-5** (+6.67 pp), and the largest loss was **gpt-5.6-sol** (-3.33 pp). All three fixed gates passed: gain at least 0.50 pp, positive interval lower bound, and no recall loss greater than 5 pp.

This warrants further validation with a broader held-out reference. The same 150 development prompts were used previously, the interval conditions on fixed predictions, and this is a two-member proxy. S07 already has separate DeBERTa evidence; this result does not establish that changing S07 would improve its leaderboard score. No official submission or final-selection change occurred. **S07 remains the best verified official public score at 65.1%.**

- [Prospective protocol](../../plans/STAGE_13C_DEBERTA_COMPLEMENT.md)
- [Kaggle receipt](KAGGLE_V2_RECEIPT.json)
- [Actual aggregate report](ACTUAL_V2_REPORT.json)
- [Aggregate reconciliation](AGGREGATE_RECONCILIATION.json)
