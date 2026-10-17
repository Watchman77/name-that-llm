# Stage 06C: mixed-grouping confirmation completed

Private Kaggle V1 359405443 completed successfully in **524.8 seconds**, with GPU T4 x2 execution confirmed in the actual logs. Observed on 17 October 2026. Output: 122.82 MB. No new official submission or final-selection change.

## Fixed component result

The size-2/3/6-mixture centered training change passes the predeclared screen using S02's actual mixed GLM24/lexical grouping method. This compares one DeBERTa component, with its fixed plain/centered evidence weights, on reused development prompts. It does not measure marginal benefit in the full S02 ensemble.

| Responses per prompt context | Control development BA | Candidate development BA | Difference (percentage points) |
|---|---:|---:|---:|
| 1 | 43.6416% | 43.6416% | 0.0000 |
| 3 | 44.4188% | 46.4581% | +2.0393 |
| 6 | 49.4223% | 50.7557% | +1.3333 |
| Predeclared mean | **45.8276%** | **46.9518%** | **+1.1242** |

Reported conditional paired prompt-bootstrap 95% interval: **+0.5191 to +1.7179 points**. All five CPU-head folds completed. Zero convergence warnings; 15 identical SciPy deprecation notices preserved. The six-response oracle diagnostic slightly regresses (53.9789% to 53.9036%), so the effect remains context dependent.

Independent checks reconcile all 216 class-recall rows with all 12 declared scenario scores, mean gain and screen logic. Group diagnostics cover each of the 2,699 rows in every context. Three/six-response pair precision is 100% within these supplied batches, with recall 75.80%/74.60% and eligible-row fractions 85.62%/92.48%. These are 30-prompt fold batches; larger-batch false merges remain untested.

Class regressions remain. Averaged over the three contexts, Opus 4.8 falls 2.67 points and Qwen 3.8 falls 2 points. GPT Terra falls 7.33 points in the six-response context. The candidate is not a uniform per-class improvement. See per_class_recall.csv and VALIDATION.json.

## Frozen GLM prerequisite

All 2,699 passages were extracted once into shape (2699, 3, 4096), float32 cache, with fixed S02 layers [8, 3, 24]. Cache SHA256: 2b41585a9482cef825962a8cc096697933c0b1237e8ede0712d1f8ab361d9414. Exact input versions: competition 170484, GLM kernel 358683059 and encoder cache 358745836.

Live manifest preview confirms 338 batches, zero nonfinite events, fp16 backbone throughout, zero truncated passages (44–381 tokens, cap 512), right padding and batch size 8. Model/config/tokenizer/text hashes are recorded; the original full manifest and numeric cache remain in immutable Kaggle V1 output. GLM_MANIFEST_OBSERVATION.json records the visible preview and its limits. No new neural training, network/API use or paid service.

## Interpretation and next step

Retain this centering candidate for the next fixed available-component ensemble comparison. The GPU prerequisite is complete; remaining head-probability reconstruction and fusion can use CPU with this saved cache. Stage 06D's stylegram OOF reference is already complete at 38.4484% component development BA. Stage 06C saved hard predictions, so its identical CPU heads must still retain complete class probabilities for the fusion; no repeat GLM inference is needed.

**Best confirmed official S02 public balanced accuracy remains 54.9%. Do not add the 1.1242-point component gain to that score.** Full S02 OOF reconstruction still lacks several representations and held-out artifacts. The fixed available-component proxy has not yet been run.

Uncertainty limits: historical development reuse; fp16 cached DeBERTa representation differs from fp32 S02 extraction; conditional bootstrap omits refitting/clustering uncertainty and batch coupling; only one frozen centered twin tested. Numeric predictions NPZ and bootstrap were not independently recomputed on the local host. Complete original logs, result JSON, all per-class recalls and group diagnostics are archived here; full numeric outputs remain on Kaggle.

Run: https://www.kaggle.com/code/watchman777/name-that-llm-stage-06c-mixed-grouping?scriptVersionId=359405443
