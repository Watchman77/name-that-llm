# Stage 06E: canonical CPU evidence from the original Stage 06C arithmetic

Status: prospective plan only. No Stage 06E fitting, evidence generation or ensemble evaluation has run. This is a distinct experiment; it does not change Stage 06D's failed reproduction gates or give that stage a retrospective pass.

## Evidence motivating this experiment

Stage 06D V7 (`359505815`) completed all fifteen CPU head fits and then failed its strict historical Stage 06C prediction gate. Its source-path and prior all-row-path predictions were identical in all twelve head/context comparisons. The largest eligible-row log-probability difference between those paths was approximately `3.77e-15`. There were twenty historical hard-prediction mismatches across 32,388 repeated comparison cells: four heads times three contexts times 2,699 rows. These counts do not establish twenty unique affected passages. Matching aggregate accuracy in some comparisons did not imply matching predictions.

The call-shape revision did not explain those hard-label differences. The remaining cause is unestablished. Neither environment drift nor another mechanism is declared the cause. The failed run computed no combined ensemble score and did not evaluate its primary raw screen. Preserve `results/stage_06d/V7_FAILURE.json`, its complete log and the pre-gate diagnostics as failed-run evidence.

The new question is whether the literal original C arithmetic can generate a self-consistent, fully archived CPU probability reference that ensemble assembly consumes without another translation or refit. This reference is new development evidence; it is not automatically an exact reproduction of historical C.

## Source and pre-execution freeze

Use `scripts/stage06c_mixed_confirmation.py` directly as the computational source. Its SHA256 is `5c27fcdffde2bcfd512d20fb4edec9bd7133430e79ccd71e31875151d235a23d`. Add output-only instrumentation and input/output path binding. Do not translate its heads, grouping or inference into another implementation.

Before any fit:

1. Commit this plan, the instrumented source, its original-to-instrumented diff, the cache schema and the separate cache-consuming fusion implementation. Record their exact hashes.
2. Independently inspect the diff: permitted changes are provenance checks, path binding, capturing existing values, writing artifacts, numeric/string metadata serialization and internal cache verification. Fitting, centering, batch definitions, random-number consumption, arithmetic expressions, eligibility, classifier parameters and source inference call shapes must remain unchanged.
3. Verify the private immutable notebook's complete saved source, all seven attached input-version IDs, CPU accelerator and disabled internet. Do not rely on an editor prefix or notebook title.
4. Record the actual runtime and resolved estimator parameters before the first fit. The prospective numeric environment is the observed V7 CPU environment: Python `3.13.15`, NumPy `2.1.3`, pandas `2.3.3`, SciPy `1.16.3`, scikit-learn `1.6.1`, threadpoolctl `3.6.0`; two threads for BLAS/OpenMP. Record the OpenBLAS versions, architecture and loaded library paths. A package-version or thread-limit mismatch must stop before fitting and be reviewed as a preflight discrepancy, not trigger a search through environments for a favorable result.

Use fresh output directories and fail closed on existing trial output. Preserve failed artifacts. Do not overwrite or recategorize the old D experiment. There is one planned canonical CPU execution after this freeze; another failure requires examining its preserved evidence before proposing further fitting.

## Immutable data and fixed arithmetic

Use the same seven pinned inputs as Stage 06D:

| Input | Immutable version |
| --- | ---: |
| Official competition | 170484 |
| Frozen encoder cache | 358745836 |
| Public LoRA1 | 358831649 |
| Public LoRA2 | 358911769 |
| Qwen token attention | 358928013 |
| DeBERTa token attention | 358948428 |
| Completed own Stage 06C output, including frozen GLM features | 359405443 |

Required original hashes:

- Official `train.csv`: `40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707`.
- `enc_debv3l2_train.npz`: `11eed58762ad128f97457dde9cc4ab03c7932e1b4b04845ec6664c834b1b1dda`.
- `glm_features.npz`: `2b41585a9482cef825962a8cc096697933c0b1237e8ede0712d1f8ab361d9414`.

Verify all other used input files against the established D provenance or their published completion manifests, retaining the existing limits where a source supplied no hashes. Record current file hashes, ordered IDs, class strings and prompt mappings. Bind the original C GLM path to the exact completed cache by an I/O-only change or byte-identical copy; do not perform GLM inference.

Retain C's 2,699 rows, eighteen classes, archived five prompt folds, frozen DeBERTa mean layers 24+6, fp16-cache-to-fp32 conversion, mixed GLM24/TF-IDF grouping, linkage threshold, all seeds and common 1/3/6-response batches. Retain full-prompt control centering, the existing size-2/3/6 mixture candidate, sample weights, scaler fitting, LogisticRegression parameters and all eligibility rules. There remain exactly fifteen head fits: plain, control and candidate in each of five folds. No new model, representation, hyperparameter, grouping threshold, calibration or candidate is introduced.

## Capture existing calls, without additional inference

Capture the values already produced by the original C computation:

- Plain-head probabilities and log-probabilities from the existing per-batch `predict_proba` calls after C's exact batch normalization.
- Control/candidate probabilities and log-probabilities from the existing eligible-only centered calls, for both inferred and oracle groups. Introduce named temporaries around those existing results where needed; never call a classifier again to populate a cache.
- The exact original component log-evidence, hard predictions and scores. Keep the original arithmetic order, clipping floor `1e-9` and eligible-row masks.
- Each of the fifteen heads' scaler mean/scale/variance/sample count, classifier coefficients/intercepts/classes/iteration counts, resolved parameters and warnings. Copy these attributes after fitting; do not use capture operations to mutate an estimator or its input arrays.
- Actual row/fold coverage, training-prompt exclusion, candidate training assignments, batch membership, inferred/oracle cluster labels, cluster sizes and eligibility.

Centered predictions do not exist on inactive rows in C's original computation. Represent those rows with an explicit inactive mask and zero centered **log-evidence contribution**. Do not describe zero-filled inactive arrays as normalized class probabilities. Keep active-row probabilities/log-probabilities together with their row indices or equivalent full-shape masked storage. Do not run an all-row head merely to manufacture unused inactive probabilities.

Save numeric and Unicode arrays only; serialize class strings as Unicode without changing the internal computational class ordering. Use `allow_pickle=False` for every later load. Do not persist passage text. Record array dtypes/shapes and file hashes. Read every saved array back and require exact equality to the corresponding captured in-memory value.

## Internal cache gate and separate historical comparison

Before ensemble assembly, require:

- Unique, aligned IDs/classes; complete once-only scored-row coverage; exact archived prompt folds and whole-prompt exclusion.
- Finite active probabilities/log-probabilities, normalized active probabilities, exact probability-floor/log relationship and mask-consistent zero inactive centered contributions.
- The complete saved component evidence reconstructs all twelve in-memory original-script prediction arrays exactly. Reconstructed component balanced accuracies match the same run's in-memory scores at `rtol=0, atol=1e-12`; independently reconcile confusion counts and class recalls.
- No non-deprecation fit warnings, preserving every warning record. Do not relabel an unrecognized warning as harmless to obtain a pass.
- Runtime, source, input and serialized-array provenance checks pass.

Any internal gate failure preserves diagnostics and blocks all combined-model evaluation. It must not produce a successful ensemble result marker.

Report historical C comparison separately: per-head/context/fold hard-label mismatches, aggregate and per-class scores, and the number of unique affected IDs if it can actually be computed from the arrays. Keep repeated-cell counts distinct from unique-passage counts. Historical mismatch does not become a retrospective D pass and is not dismissed by an unchanged mean score. If the new cache differs, label it explicitly as the new canonical CPU development reference, with the historical discrepancy still unresolved. Do not choose a runtime, seed, source variation or reference cache based on agreement with historical scores or on the candidate's measured gain.

## One frozen cache, one fixed ensemble consumption

After the internal cache gate passes, freeze its manifest and consume those immutable arrays directly. Ensemble assembly must not refit heads, invoke captured heads, rerun passage extraction, rebuild groups, change batches or recompute probabilities through a different classifier implementation. Consume captured clipped log-probabilities directly; do not reconstruct them through an exp/log round trip.

Reuse the exact frozen style probability transfer and audited plain LoRA1/LoRA2 and two attention outputs, with their established ID/class/probability/fold checks. No style or neural refitting is included. Use the original D fixed weights without renormalizing them:

| Evidence member | Weight |
| --- | ---: |
| Stylegram | 1.516 |
| Plain LoRA1 | 0.177 |
| Plain LoRA2 | 0.113 |
| Qwen token attention | 0.153 |
| DeBERTa token attention | 0.185 |
| Frozen DeBERTa plain | 0.050 |
| Eligible frozen DeBERTa centered | 0.250 |

The paired control/candidate comparison changes only the corresponding canonical centered-head contribution. All other evidence, IDs, batches, masks and weights remain identical. The old D implementation `scripts/stage06d_fixed_ensemble.py` at SHA256 `c6005196a73100676141d18f67eb299c1bf686a547fbb077b91825d2bd2611b3` and `plans/STAGE_06D_FIXED_ENSEMBLE_PROXY.md` at SHA256 `40376f918da741b3a7667e68a9e56fdd1b0fa2d8265780f7086f62536ed30032` define the unchanged fixed fusion recipe and endpoint calculations. Reuse their verified fusion mathematics only after removing any refitting/regrouping responsibility from the cache consumer; do not alter those archived files or their failed gates.

The primary endpoint remains mean **raw rowwise fixed-fusion balanced accuracy** across contexts 1/3/6. Use the unchanged paired prompt bootstrap: 3,000 draws, seed `20261017`, resample the 150 prompts with replacement, aggregate class-specific corrected-minus-broken counts and supports, average class recall differences and the three contexts exactly as D. The fixed screen remains mean gain >0, conditional 95% interval lower bound >0, every context gain >=`-0.005`, and no non-deprecation fit warning. Report per-class regressions, fold scores and corrected/broken/changed predictions; do not add a post-hoc class veto or select new weights.

Report exact S02 alpha-0.5, fifty-iteration marginal balancing per existing simulated batch only as a **potentially contaminated diagnostic**. Mixed supervised fold schemes allow peer predictions to have trained on the target prompt. Balanced scores cannot determine acceptance or support clean generalization claims. Do not pool distinct batches or silently intersect fold maps to change the experiment.

## Interpretation and stop conditions

This remains the available-component proxy, not full S02. Missing centered LoRA, other centered LLM representations, LightGBM and fine-tuned encoder evidence prevent a full S02 claim. Historical source weights and reused development prompts also prevent an untouched generalization claim. The conditional bootstrap does not include model refitting, recovered-group or transductive-batch variability.

A passed raw screen would support further matched ensemble investigation only. A failed screen is a retained negative result. Neither outcome establishes an official public score or a route to 70%. The confirmed official S02 score remains a separate result.

No GPU execution, API, paid service, new neural training, official submission, final-selection change or messages to others are part of Stage 06E. This file records the next distinct prospective experiment. Implementation and execution remain unstarted at preparation; no fits or model code were executed while preparing it.
