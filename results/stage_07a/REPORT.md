# Stage 07A completed: private GPT-data augmentation pilot

Authorized follow-up: [Stage 07B](../stage_07b/REPORT.md) has now tested the same source addition with matched per-class mass and passed its prospective development screen. It is a diagnostic follow-up on reused folds, not an independent confirmation. The original 07A protocol, failed screen and result below remain unchanged.

**Decision: retain the control.** The fixed candidate has a higher development point estimate, but failed the prospectively frozen uncertainty screen. No new official submission or full-ensemble result was produced.

| Original held-out-prompt development result | Value |
|---|---:|
| Original-only character/style control | 35.376087% |
| Same features/head plus 750 admitted GPT passages | 36.302013% |
| Candidate minus control | +0.925926 percentage points |
| Paired prompt bootstrap 95% interval | [−0.074323, +1.930486] percentage points |
| Corrected / broken original passages | 110 / 85 |
| Changed predictions / all original passages | 466 / 2,699 |
| Fit warnings | 0 |
| Fixed improvement screen | Failed: lower interval endpoint is below zero |

These are **development balanced accuracies for a standalone character/style classifier**, not official public scores and not the full S02 ensemble. The previously verified S02 official public score remains 0.549; this experiment did not recheck or replace it. A 36.30% standalone result cannot be presented as an improvement above that ensemble.

## What actually ran

The runner and prospective protocol were frozen at Git commit `d903e688bd7accd86bc343b728451ec8ea258382` before any head fitting. Frozen runner SHA-256: `bcc2c0a23b67143b8e16532001fc83c7df667829308e5a2579ceafb916ee038d`. All five folds completed without convergence warnings. Total measured runtime, including the repeated preflight and bootstrap, was approximately 64.8 seconds on this execution environment's CPU. No laptop browser/process was accessed and no paid inference service was called.

Source: [prvsiyan's Extended Name That LLM Dataset, V4](https://www.kaggle.com/datasets/prvsiyan/prvsiyans-extended-name-that-llm-dataset), source-declared CC BY 4.0; attribution to prvsiyan, October 2026. We admitted only `new` / clean / direct `openai` rows: 750 passages across 150 complete five-model blocks. The five augmented labels each have 150 examples. Exact request/served IDs and raw responses were checked in the preceding source review. No other provider route was admitted into this fit.

The bounded use assessment relies on the [current OpenAI Services Agreement](https://openai.com/policies/services-agreement/), whose Permitted Exception includes qualifying undistributed classifiers. This experiment trained only a private SVM classifier, did not serialize it, and did not distribute trained weights. This assessment does not authorize distributing a trained competition artifact or a generative fine-tune.

Both arms validate on the same 2,699 original official passages across all 18 labels, using five whole-prompt folds (150 original prompts). The identical character/style feature map is fitted once per fold on original training rows only. Candidate data cannot alter vocabulary, IDF or style scaling. Both heads use the same C, seed, tolerance and iteration limit. Original weights remain unchanged; the candidate adds the frozen external weights. No test passages, hidden labels or leaderboard feedback enter the comparison.

Preflight found no exact/normalized full-response overlap, no whole-passage substring overlap, no candidate with three distinct shared official 8-grams, and no nearest-official character cosine above the frozen 0.85 threshold. Maximum observed cosine was 0.233093; all 750 rows remained. These screens and comparison of inferred topics do not certify semantic independence: the original prompt instructions are unavailable and the published reconstructions are guesses. This limitation prevents promoting the recipe solely on its development score even had the fixed screen passed.

## What changed, and what it does not prove

| Class group | Control mean recall | Candidate mean recall | Difference |
|---|---:|---:|---:|
| Five GPT labels receiving external examples | 34.13% | 41.47% | +7.33 points |
| Other 13 labels | 35.85% | 34.32% | −1.54 points |

The GPT group has 82 corrected and 27 broken passages, net +55. The other classes have 28 corrected and 58 broken, net −30. Overall net gain is 25 correct passages. GPT predictions rise from 770 to 967 while the true GPT count is 750. Qwen recall falls 9.33 points; Claude Opus 5 falls 4.67 points; Kimi falls 3.33 points. All five GPT recalls rise; GPT-5.5 rises 12.67 points. Per-class and prediction-distribution tables retain the complete result, including regressions.

This pattern suggests that the recipe shifts decisions toward the augmented class group. It does **not** distinguish useful extra examples from a changed class preference by itself. The prospectively specified weights preserve each original class's weighted mass and add 50% extra mass to each of the five augmented classes. Consequently this experiment tests the **whole fixed augmentation policy**, including that mass change. A future causal attribution test would need a separately frozen class-mass control or a fixed equal-class-mass design. The present result does not justify attributing all GPT gains to new stylistic information. No extra fit, rescaling, threshold adjustment or alternative weight was tried after observing this result.

The outcome is suggestive but insufficient for the frozen promotion screen. It does not prove the external data has no value. It does rule out describing this exact run as a reliable overall improvement or a competitive breakthrough.

## Validation and retained evidence

The separate validation script aligned all original IDs/classes/prompts, reproduced the five fold assignments, independently recalculated pooled and fold macro recalls, checked all saved artifact hashes, and reproduced the 10,000-draw interval using prompt multiplicities and class-count arithmetic. Corrected/broken counts reconcile. `TERMINAL_VALIDATION.json` records PASS. This is a separate implementation check, not an independent agent review or an independent model refit.

The interval is paired by original prompt group and conditional on the fitted folds; it does not include all retraining uncertainty. Per-class contrasts are exploratory and were not separately multiplicity-adjusted hypothesis tests.

The GitHub repository was freshly verified **public**, correcting earlier private-repository descriptions. Its visibility was not changed. Only source code, public external-data IDs, audit receipts, aggregate metrics/confusion matrices, per-class summaries and fit logs are included in this update. Original row-level labels/predictions and decision-margin arrays remain outside the public repository. No trained model was serialized or uploaded. Their hashes are retained for provenance; frozen code plus authorized original data reproduces the calculation. Intermediate local files can expire and are not represented as durable hosted results.

## Next decision

Keep S02 and the existing control unchanged. Do not submit this standalone candidate, tune its weight against these results, or silently switch the frozen protocol. Before another augmentation fit, separate the class-preference effect from new-example value in a new prospective design. A stronger representation or an ensemble addition is a different experiment and still needs aligned prompt-held-out evidence. The larger all-provider corpus remains a potential resource, not an approved automatic follow-up run.
