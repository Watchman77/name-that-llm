# Stage 13 — standalone GLM readout complete

Verified 11 November 2026. [Private Kaggle V3 368292220](https://www.kaggle.com/code/watchman777/name-that-llm-stage-13-glm-readout?scriptVersionId=368292220) succeeded in **152.2 seconds** on CPU, and the saved-report and private probability-cache checks passed. Its three saved code cells match the [prospective freeze](https://github.com/Watchman77/name-that-llm/commit/72e16e20cdac267833946454143d85b9d1eac51a). Both intended inputs (competition 170484 and kernel output 359405443), CPU, Internet off and private visibility were verified. The runner repeated all seven input checks before fitting.

**Pooled development balanced accuracy: 45.896346%** (exact macro recall 61,547/134,100). All five fits converged without warnings. Each fit used 120 training prompts and scored the other 30 with no prompt overlap; all 2,699 rows have one held-out 18-class probability vector. The fold scores were **47.04%, 46.67%, 45.37%, 45.56% and 44.85%**. The 18-class confusion counts independently reconcile to 1,239 correct predictions; ordinary accuracy is a separate metric.

The largest two-way confusions were Gemini 3.7/3.8 Flash (81 errors), GLM 5.3/5.3 Flash (61), and Gemini 3.6/3.7 Flash (58). Lowest recalls were Claude Fable 5 (20.13%), GLM 5.3 (24.67%) and GPT 5.6 Sol (30.00%). These describe this single classifier, not hidden-test or S07 errors.

The exploratory reference uses reused canonical prompt folds and one fixed plain GLM readout. **S07 remains the best verified official submission at 65.1%.** No new submission, balancing, model-weight export or final-selection change occurred. The saved row-level cache remains private; this repo contains the source and aggregates only. The full S07 ensemble and any future batch-coupled test still require separate admission. Freeze one follow-up comparison on matched prompts before evaluating it.

- [Prospective protocol](../../plans/STAGE_13_GLM_READOUT.md)
- [Verified run receipt](KAGGLE_V3_RECEIPT.json)
- [Actual aggregate report from rendered output](ACTUAL_V3_REPORT.json)
- [Independent aggregate reconciliation](INDEPENDENT_AGGREGATE_RECONCILIATION.json)
