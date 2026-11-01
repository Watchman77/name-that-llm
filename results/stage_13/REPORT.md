# Stage 13A — actual cache checks passed; final notebook verification pending

Date: 1 November 2026.

Private CPU **V3 364509756** visibly reported **PASS_FROZEN_BASE_FEATURE_ADMISSION_ONLY** and passed all **seven grouped admission checks** against the real mounted inputs. The logs also printed **STAGE13A_SAVED_REPORT_READBACK PASS_FROZEN_BASE_FEATURE_ADMISSION_ONLY**. The cache covers **2,699 rows, 150 prompts and 18 classes**, with finite float32 features of shape **[2699, 3, 4096]**, exact train/cache hashes and zero fit/validation prompt overlap in each of the five planned partitions. Audit computation took **2.582699 seconds**; this is not total notebook duration.

This admits the frozen-base GLM cache for subsequently frozen fold-local readout fitting. **No model fitting, model inference, held-out predictions, balancing comparison, clean development reference or full-S07 OOF result was produced. There is no new accuracy score.** This cache check does not validate the supervised production branches of S07 as held out.

The last explicit V3 header observed was **Running for 30.8s**. A later Notebook view displayed rounded **Runtime 29s**, which is not an explicit terminal-status check. During the subsequent source/status inspection, browser control stalled; a fresh connection attempt then returned **No connected browser was found for the originating browser instance**. V3's final terminal state, exact whole-notebook duration and final saved-source/settings readback remain **unverified**. Do not relaunch the audit blindly; inspect existing V3 first. The preserved evidence contains selected aggregate fields transcribed from actual visible logs, not a byte-identical download of the output report.

Before execution, recovered source-only **V2 364509282** was independently read back: all three code strings exactly matched the frozen notebook; saved inputs were competition **170484** and kernel version **359405443**, accelerator **None**, Internet **off**. Exactly one actual Save & Run All was then requested for V3 with **Run without an accelerator for this session**. Its actual viewer showed **Accelerator None**, private visibility and the latest container image. V2's saved source/settings evidence does not replace final V3 readback.

Recovery also found that the earlier imported notebook was in Script format and that Kaggle had displayed a generic commit error. It was converted through File > Editor Type > Notebook, and the three canonical code strings, including terminal newlines, were restored and read back before saving V2. The earlier pending V1 is source-only and is not an actual cache-audit execution. The 31 October preparation receipt remains historical.

The source and static ledger remain frozen at **d8d600d5f3151e0f6a4ab131f58fb8db54befa3f**. Independent preparation review passed **18 implementation tests, six supplemental checks, ten notebook structure/source checks and 35 saved-report readback fixtures**. The deterministic notebook rebuild was byte-identical. Local nbformat/nbclient were unavailable; JSON structure and code compilation were checked. Actual Kaggle audit execution is now observed, with the terminal-state limitation above.

**S07 remains the last verified official public best at 0.651 (65.1%), verified 30 October 2026.** No new official submission, paid service or final-selection change occurred. Separately freeze common-prompt-fold readouts next, then generate held-out predictions before admitting any batch-coupled balancing comparison. Reused folds remain exploratory development; do not claim an untouched holdout or add a proxy gain to the official score.

- [Actual V3](https://www.kaggle.com/code/watchman777/name-that-llm-stage-13a-cache-admission?scriptVersionId=364509756)
- [Actual V3 logs](https://www.kaggle.com/code/watchman777/name-that-llm-stage-13a-cache-admission/log?scriptVersionId=364509756)
- [Run receipt](CPU_V3_RUN_RECEIPT.json)
- [Selected actual aggregate log evidence](ACTUAL_CPU_V3_LOG_SUMMARY.json)
- [Independent arithmetic and scope review](INDEPENDENT_CPU_V3_AGGREGATE_REVIEW.json)
- [Historical preparation receipt](KAGGLE_PREPARATION_20261031.json)
- [Static S07 provenance](S07_DEVELOPMENT_PROVENANCE.json)
- [Execution protocol](../../plans/STAGE_13A_CACHE_ADMISSION.md)
