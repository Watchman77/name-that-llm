# Stage 06E: valid ensemble comparison, improvement screen not met

Status: **COMPLETE_FIXED_PROXY_PRIMARY_SCREEN_NOT_MET**. Private [V2 359514801](https://www.kaggle.com/code/watchman777/name-that-llm-stage-06e-canonical-cpu-evidence?scriptVersionId=359514801) successfully ran in **169.5 seconds** on CPU, producing 38.58 MB of output. The same-run probability cache and pure cache-consumer gates passed. The primary improvement screen did not pass. **Retain original control centering. Confirmed official S02 remains 54.9%.**

## Primary development result

| Responses per simulated context | Raw control BA | Raw candidate BA | Change |
| --- | ---: | ---: | ---: |
| 1 | 52.6381% | 52.6381% | +0.0000 pp |
| 3 | 55.3067% | 55.3072% | +0.0005 pp |
| 6 | 56.4919% | 56.0848% | -0.4072 pp |
| Mean | **54.8122%** | **54.6767%** | **-0.1356 pp** |

The reported conditional paired-prompt 95% interval for the mean change is **-0.5189 to +0.2375 percentage points**. The point estimate is lower, but the interval includes zero; this is no evidence to adopt the candidate and does not establish a population-level deterioration. The declared positive-mean and positive-interval-lower-bound criteria fail. The six-response regression remains above the predeclared −0.5-point floor.

In the three-response context the candidate corrected 40 passages and broke 40. In the six-response context it corrected 41 and broke 52: eleven fewer correct. One-response predictions were identical. These are paired development counts within each context, not new independent examples across contexts.

The component alone still improves in this new canonical CPU reference: mean 45.8770% to 46.9518% (+1.0748 pp). Its marginal effect in the fixed ensemble is different. The earlier historical Stage 06C component result remains separately recorded.

The balanced diagnostic mean changes by **+0.0250 pp**, with a reported interval spanning zero. It is potentially contaminated by mixed source-fold peer supervision and was excluded from candidate selection. It cannot rescue the failed primary raw screen.

## What ran and what passed

The producer is the literal original Stage 06C script with output-only instrumentation. It preserves all fifteen fits, five prompt folds, seeds, sample weights, grouping and existing inference calls. It captures full probabilities, clipped logs, original component evidence, masks/batches, training partitions and fitted head/scaler snapshots. The static preparation module preserves the existing five public-member audits, including affine readout reconstruction, then writes probability caches. The consumer reads these caches without fitting, estimator invocation, readout reconstruction, regrouping or neural extraction.

All five folds and fifteen head fits completed. Exact stored-array readback and all twelve same-run component prediction/score reconstructions passed. The prospectively fixed Python/package versions, two-thread BLAS/OpenMP and source/input hashes passed before fitting. Fifteen deprecation warnings were recorded; there were no convergence or other fit warnings. Later notebook-renderer syntax warnings are separate from model fitting.

The original C comparison still differs in **20 of 32,388 repeated head/context/row cells, involving twelve unique IDs**. Its cause remains unresolved. This new canonical CPU reference does not retrospectively pass Stage 06D or establish historical runtime parity.

## Independent evidence and limits

The pre-execution freeze is [ac9d9d4b0487d71b3265b1b2b0f2621803a8b701](https://github.com/Watchman77/name-that-llm/commit/ac9d9d4b0487d71b3265b1b2b0f2621803a8b701). All 25 intended files and their remote Git blobs were verified before launch. The local notebook is the frozen launch source; executed outputs are on immutable Kaggle V2.

Before execution, 22 producer checks, eight consumer fixture groups and 31 independent contract tests passed without model fitting or inference. The inert five-cell package validated with nbformat and compilation, and a deterministic rebuild matched exactly. Both metadata-only V1 359514410 and actual V2 359514801 matched all five source strings and all seven pinned input IDs with CPU and internet off. V1's 8.3-second render had no cell execution and was not the experiment.

Independent terminal validation reconciles all 216 printed class recalls to twelve ensemble scores/four means and integer class-correct counts, checks all sixty fold scores for count feasibility/pooled consistency, verifies six corrected-minus-broken comparisons, and checks fold markers, head snapshots, source/runtime bindings and both passed gates.

The full log is `KAGGLE_V2.log`: 127,465 bytes, SHA256 `a267176c4c5e6be764f48381a981fe29acf4c841f6371cc21afe8425d72e3278`. Printed JSON/CSV artifacts are archived locally. The canonical gate and manifest bytes match the runtime consumer's recorded hashes. The local ensemble result, score/count CSVs, cache-consumption receipt and three complete original-control confusion CSVs match the runtime output manifest byte hashes.

The canonical numeric cache, static cache and ensemble arrays remain only in immutable Kaggle V2 output. No local raw-row prediction or bootstrap-draw recomputation is claimed. Recorded cache hashes and paths are in `KAGGLE_RUN.json`, `cache_consumption_gate.json` and `output_manifest.json`. The conditional interval omits refitting/group-recovery/transductive-batch uncertainty.

This remains an available-component development proxy. Missing full S02 components, reused development prompts/source weights, mixed fold schemes and quantized feature caches limit generalization claims. Neither 54.8122% nor 54.6767% is an official competition score.

## Decision and next work

Keep the original centering recipe. Preserve this valid negative result and the unresolved historical discrepancy. Do not tune weights or promote the candidate from its component-only gain.

The completed [error analysis](ERROR_ANALYSIS.md) identifies Gemini 3.7 versus 3.8 as the largest mutual confusion in every context (94 / 87 / 94 errors); GLM 5.3 versus Flash is second. These are repeated predictions of the same passages. Next inspect saved row probabilities, margins and prompt distribution before freezing one complementary separation hypothesis. Any subsequent candidate needs a new prospective comparison; no additional fit, official submission, final-selection change, paid service, GPU execution or new neural training was performed.

## Records

`KAGGLE_RUN.json`, `SAVED_SOURCE_V1.json`, `SAVED_SOURCE_V2.json`, `EXECUTION_FREEZE.json`, `PACKAGING.json`, `INDEPENDENT_IMPLEMENTATION_REVIEW.json`, `INDEPENDENT_TERMINAL_VALIDATION.json`, `CONFUSION_PREVIEW_PROVENANCE.json`, `output_manifest.json`, `ensemble_result.json`, `cache_manifest.json`, `cache_gate.json`, `cache_consumption_gate.json`, the score/count/confusion CSVs and `KAGGLE_V2.log`. The frozen prospective plan and earlier C/D artifacts remain unchanged.
