# Stage 06E control error analysis

The centering candidate failed the primary comparison: **54.8122% → 54.6767% mean raw development balanced accuracy (−0.1356 percentage points)**. Retain the original control. These results do not change the confirmed official S02 score.

This analysis uses the three complete 18×18 raw-control confusion previews from [V2 359514801](https://www.kaggle.com/code/watchman777/name-that-llm-stage-06e-canonical-cpu-evidence/output?scriptVersionId=359514801). Their local bytes and hashes match the preserved runtime output manifest; supports and diagonals reconcile with printed recalls and all three balanced accuracies. Each context predicts the **same 2,699 passages**. Counts averaged or summed across contexts are repeated predictions, never additional independent examples or counts of unique passages. Control errors are 1278, 1206 and 1174 in contexts 1/3/6.

## Weakest classes

Ranked by mean class recall across the fixed contexts:

| True class | Context 1 | Context 3 | Context 6 | Mean recall |
|---|---:|---:|---:|---:|
| claude-fable-5 | 22.82% | 28.19% | 28.19% | 26.40% |
| glm-5.3-flash | 28.67% | 31.33% | 34.67% | 31.56% |
| glm-5.3 | 31.33% | 36.00% | 35.33% | 34.22% |
| gpt-5.6-sol | 34.00% | 34.67% | 35.33% | 34.67% |
| gemini-3.7-flash | 38.00% | 40.00% | 38.00% | 38.67% |
| gpt-5.6-terra | 44.00% | 44.67% | 47.33% | 45.33% |

Fable has 149 passages; every other class has 150. That one-row support difference does not explain its low recall. Fable is frequently confused with both GLM classes, so a same-family-only strategy would miss a material error source.

## Largest confusion pairs

Mutual counts add the two directions within each context:

| Mutual confusion | Mean errors/context | Counts in contexts 1 / 3 / 6 |
|---|---:|---|
| gemini-3.7-flash ↔ gemini-3.8-flash | 91.67 | 94 / 87 / 94 |
| glm-5.3 ↔ glm-5.3-flash | 71.67 | 73 / 69 / 73 |
| gemini-3.6-flash ↔ gemini-3.7-flash | 53.33 | 56 / 52 / 52 |
| claude-fable-5 ↔ glm-5.3 | 50.00 | 46 / 52 / 52 |
| gpt-5.6-sol ↔ gpt-5.6-terra | 47.67 | 50 / 46 / 47 |
| claude-opus-4-8 ↔ claude-sonnet-5 | 45.00 | 45 / 46 / 44 |

The Gemini 3.7/3.8 pair accounts for **7.52% of control errors averaged across contexts**. It remains the largest mutual confusion in all three settings. This is an observed target for investigation, not an estimate of recoverable accuracy.

The largest directed confusions are:

| True → predicted | Mean errors/context | Counts in contexts 1 / 3 / 6 |
|---|---:|---|
| gemini-3.7-flash → gemini-3.8-flash | 50.33 | 50 / 49 / 52 |
| gemini-3.8-flash → gemini-3.7-flash | 41.33 | 44 / 38 / 42 |
| glm-5.3-flash → glm-5.3 | 38.00 | 40 / 37 / 37 |
| glm-5.3 → glm-5.3-flash | 33.67 | 33 / 32 / 36 |
| claude-fable-5 → glm-5.3 | 32.00 | 29 / 32 / 35 |
| gemini-3.7-flash → gemini-3.6-flash | 31.67 | 32 / 31 / 32 |

Full rankings, including zero-count pairs, are in `error_pairs.csv`. For directed rows, `model_a` is the true class and `model_b` the predicted class; undirected rows combine both directions. `repeated_context_error_count` sums three context-specific prediction counts and is not a unique-passage count.

## What the rejected candidate harmed

Raw candidate-minus-control class recall changes:

| Class | Mean recall change | Context 6 change |
|---|---:|---:|
| gemini-3.8-flash | -2.44 pp | -4.00 pp |
| gpt-5.6-luna | -1.78 pp | -2.00 pp |
| gemini-3.6-flash | -1.78 pp | -2.00 pp |
| gpt-5.6-terra | -1.56 pp | -4.67 pp |

The candidate improves GPT-5.6-sol by +2.00 pp mean recall, but its gains do not outweigh the losses. In context 6 it corrects 41 passages and breaks 52: a net loss of 11 correct predictions. The slight balanced diagnostic increase cannot rescue a failed raw endpoint.

## Next hypotheses and evidence needed

1. **Prioritize Gemini 3.7 versus 3.8.** Test whether a dedicated separation signal can address the persistent two-way confusion. Retrieve the already-saved `ensemble_arrays.npz` first to identify row IDs, margins and prompt distribution; the aggregate matrices alone cannot show whether the same passages fail in every context.
2. **Investigate GLM 5.3 versus Flash next.** Both have low recall and a persistent mutual confusion. Inspect cross-confused passages and existing component probabilities before choosing features or a pair classifier. Fable-to-GLM errors deserve a separate check; label names alone do not establish the mechanism.
3. **Require a prospective comparison for any correction.** Freeze one hypothesis and its feature/gating recipe, preserve prompt exclusion in every supervised fit, and evaluate the complete fixed raw ensemble plus every class. Track false entry from other classes into the targeted pair. Reused development errors are hypothesis-generation evidence; they do not create an untouched validation set. A successful pair-only metric is insufficient if total raw balanced accuracy worsens.

No model, fitting, inference or parameter search was run for this analysis. The full raw NPZ files are not locally available, so no row-level overlap, probability-margin, paired-bootstrap or causal error-mechanism claim is made.
