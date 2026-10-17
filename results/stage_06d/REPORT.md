# Stage 06D: technical reproduction failure, no valid ensemble score

Status: **TECHNICAL_REPRO_FAILURE_NOVALID_ENSEMBLESCORE**. Final diagnostic execution [V7 359505815](https://www.kaggle.com/code/watchman777/name-that-llm-stage-06d-fixed-ensemble?scriptVersionId=359505815) completed all five fixed CPU folds/15 heads, then failed the strict original-C prediction gate. The primary raw improvement screen was never evaluated. There is no valid combined-model score or official improvement. Confirmed official S02 remains **54.9%**.

## What the executions established

| Execution | Frozen source commit | Verified outcome |
| --- | --- | --- |
| V3 359500151 | 9db86f28cfd763ccbc3dcc048fb35fe649f6445f | Exact style transfer and five-member/47-file audit passed. Object-typed class metadata failed before new head fits at 20.7 log seconds; log ended at 27.7 seconds. |
| V5 359502730 | de19608a672015fdb84e7921ed2aa1482c495a18 | Classes-only static reader passed the actual cache. All five folds/15 heads completed; strict control argmax reproduction failed at 164.0 seconds. Platform total 175.5 seconds. |
| V7 359505815 | 15969f9cd696c7a4ce51be92680db16cbb42ea0d | Exact original-C eligible-row inference and all-row diagnosis produced identical hard labels. All five folds/15 heads completed; strict control gate failed at 141.6 seconds. Platform total 151.6 seconds; output 13.19 MB. |

These are technical failures before ensemble scoring. Neither the candidate nor its primary raw screen passed or failed on accuracy. No official submission or final-selection change occurred.

The earlier browser/save failure was resolved before V3. Metadata-only saved-source gates were separately verified: V2 359499684 for the original five-cell source, V4 359502142 for the classes-only repair, and V6 359505487 for the source-path diagnostic revision. V7's actual immutable export also matched all five c600 source cells and all seven pinned inputs with CPU and internet off. The first four cells remained unchanged; V7's final cell had 69,909 characters. A Save & Run All completion toast meant commit completion, not terminal computation; earlier Running observations are historical launch records in `EXPERIMENTS.md`.

## V7 diagnostic evidence

The full original log is `KAGGLE_V7_FAILED.log`, 28,581 bytes, SHA256 `c397e1eafde8fffd8335cd0f3262c6c11a678d967512201cd6d37a0b9c9ed492`. All five fold markers occurred at 53.9 / 74.5 / 98.4 / 119.8 / 140.7 seconds. Diagnostics were saved and printed before the unchanged gate failed. No reproduction-pass or `ENSEMBLE_RESULT` marker appeared.

| Component evidence | 1-response mismatches | 3-response mismatches | 6-response mismatches |
| --- | ---: | ---: | ---: |
| Control | 1 | 5 | 4 |
| Candidate | 1 | 0 | 0 |
| Oracle control | 1 | 4 | 2 |
| Oracle candidate | 1 | 0 | 1 |

These are **20 mismatches across 32,388 repeated head/context/row comparisons**, not 20 unique affected passages. Source-eligible and all-row inference produced zero hard-label changes relative to each other. The largest eligible log-probability difference was **3.774758283725532e-15**. Thus inference call shape is ruled out as the cause of the observed hard-label differences in V7; the remaining cause is unproven.

Oracle one-response contexts had zero eligible centered rows but retained one mismatch each. Some discrepancy therefore persists without centered evidence. This is a clue about the plain prediction pathway, not identification of a fitting/runtime cause. Matching aggregate balanced accuracy in some cases did not prove identical predictions.

`INDEPENDENT_V7_VALIDATION.json` reconciles all 12 reported frozen balanced accuracies with original C, all 60 fold/context/head summaries, inferred eligibility counts against original C grouping diagnostics, five ordered fold markers and 15 deprecation warnings with zero convergence warnings. Reported source-path and all-row scores/counts agree. No extra fitting occurred during validation.

Actual V7 runtime: Python 3.13.15, NumPy 2.1.3, pandas 2.3.3, SciPy 1.16.3, scikit-learn 1.6.1 and threadpoolctl 3.6.0. Both OpenBLAS pools reported Haswell architecture and two threads. The live UI reported Latest Container Image; no editor-pinned environment parity is inferred.

The full numeric `reproduction_diagnostics.npz` remains only in immutable Kaggle V7 output, reported SHA256 `cf0ba543ab468329ee25cc6e01238dd04a8941f522ea61470a6b76f7435e86a5`. It contains predictions, probabilities, margins, masks, batches/groups and 15 head snapshots. It was **not downloaded or recomputed locally**. The independent audit validates summary/log consistency, not raw arrays, unique mismatch identities or coefficient differences. Original C did not retain head coefficients/scalers for direct comparison.

## Scope of the repairs

The classes-only repair interprets a bounded, restricted NumPy serialization grammar as inert symbols. It checks exactly 18 unique class strings in official order without unpickling or invoking constructors. Numeric, ID and prompt arrays retain `allow_pickle=False`. Source SHA256: `b27925f978b0c87938c92adc36bcbb52e6536aa79fbe9330217daec7fecbae51`. Focused and independent fixtures are retained in `CLASS_METADATA_REPAIR_FIXTURES.json` and `SAFE_CLASSES_INDEPENDENT_REVIEW.json`.

The source-path revision uses the same fitted heads, retaining all-row outputs only for diagnosis and overwriting active rows using C's exact eligible-only call. It archives pre-gate predictions, margins, runtime and head snapshots before the unchanged twelve-array and score-atol-1e-12 checks. No old-path result is selectable in fusion. Source SHA256: `c6005196a73100676141d18f67eb299c1bf686a547fbb077b91825d2bd2611b3`; combined runner SHA256: `9185a208679ea99c318bb11feb9abd773661886523621e268052ca037e613893`. Review and packaging: `INFERENCE_PATH_INDEPENDENT_REVIEW.json`, `SOURCE_PATH_REPAIR_FIXTURES.json`, `SOURCE_PATH_REPAIR_PACKAGING.json`.

Both revisions preserved fitting, folds, seeds, grouping, weights and strict acceptance gates. The one evidence-based diagnostic retry is complete. **No further blind Stage 06D refits or gate relaxation are planned.** Historical protocols and manifests remain unchanged.

## Completed stylegram prerequisite

The original fixed stylegram OOF run completed before ensemble assembly:

- Development balanced accuracy: **0.3844842157593836 (38.4484%)**.
- 2,699 rows, 150 prompts, 18 classes; 1,038 correct and 1,661 errors.
- All five archived prompt folds completed in 42.48 CPU seconds, with zero warnings.
- Fold balanced accuracies: 35.7407%, 36.2963%, 40.0000%, 40.9259%, 39.2848%.
- Every vocabulary/IDF fit excluded its validation prompts. Full float64 probabilities, score softmax, class recalls, confusion counts and fold assignments independently reconcile in `stylegram_oof/VALIDATION.json`.

Style implementation was frozen before fitting in 4afba147c4290ae24ddf4f1ea078fe8a8fc729d1; source SHA256 `73b8c397733075ea7cbb183964d3b59cbfc24881a71cc043859532edf380acf6`. Its local environment was Python 3.12.14, NumPy 2.3.5, pandas 2.2.3, SciPy 1.17.0 and scikit-learn 1.8.0. Later Kaggle runs transferred its exact probability bytes rather than refitting it: raw SHA256 `5e2e05577b7dc632f38f67d47958d7f3d069e8b758ead4664050f0d984803bad`, 388,656 bytes. SVM score softmax is not calibrated probability.

This component result remains valid as recorded. It is not a full ensemble or official score. Its largest errors, including Gemini 3.7/3.8 and cross-family Fable/GLM-flash confusions, do not identify the full system's remaining errors.

## Next distinct prospective stage

The independently reviewed [Stage 06E canonical CPU evidence plan](../../plans/STAGE_06E_CANONICAL_CPU_EVIDENCE.md) proposes output-only instrumentation of the literal original C script, capturing its existing inference calls once into a verified immutable cache. A separate consumer would assemble fixed evidence without refitting or translating classifiers. Historical C differences remain a separate diagnostic; this would be a new development reference, not a retrospective pass for Stage 06D.

**Stage 06E implementation and fitting have not started.** Its original-to-instrumented diff, schema, consumer and saved-source/runtime checks require review and a pre-execution freeze before the single planned CPU execution. Plan review: `../stage_06e/PLAN_REVIEW.json`.

The proposed system remains an available-component proxy. Missing centered LoRA, other LLM representations, LightGBM and fine-tuned encoder evidence prevent a full S02 claim. Reused development prompts/source-fitted weights and conditional bootstrap limitations remain. Mixed-fold balancing is potentially contaminated and cannot select the candidate. No current result establishes an official gain or a route to 70%.

## Preserved records

Current receipt: `KAGGLE_RUN.json`. Original failed logs and parsed records: `KAGGLE_V3_FAILED.log` / `V3_FAILURE.json`, `KAGGLE_V5_FAILED.log` / `V5_FAILURE.json`, `KAGGLE_V7_FAILED.log` / `V7_FAILURE.json`. Final diagnostic summary: `reproduction_diagnostics.json`; independent audit: `INDEPENDENT_V7_VALIDATION.json`. Earlier source/packaging and review JSON files remain historical snapshots.

Browser screenshots remain scratch proof outside the repository. The pre-refresh save error and final V7 diagnostics screenshot do not substitute for raw-array validation. No screenshot binary, passage text, paid service, GPU retry, new neural training, submission or message to another person was part of Stage 06D ensemble work.
