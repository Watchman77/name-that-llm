# Stage 06E canonical CPU evidence schema

Schema version: `stage06e-canonical-cpu-v1`. This is a new same-run CPU development reference. Passing its internal gates never converts Stage 06D's historical reproduction failure into a pass. The historical comparison is reported separately and cannot choose the runtime or source. No Stage 06E experiment has been executed while preparing these files.

## Literal producer and runtime binding

`scripts/stage06e_capture_original.py` is the literal original `scripts/stage06c_mixed_confirmation.py` with I/O and output-only instrumentation. The original remains unchanged at SHA256 `5c27fcdffde2bcfd512d20fb4edec9bd7133430e79ccd71e31875151d235a23d`. The complete unified diff is `results/stage_06e/ORIGINAL_TO_CAPTURE.diff`.

The producer must execute as a real saved file, with the original C source and `stage06e_cache_checks.py` in its directory. Set `NTL_STAGE06E_FREEZE` to a JSON file containing `capture_source_sha256`, `cache_checks_source_sha256`, `original_source_sha256`, `plan_sha256`, the seven ordered `input_version_ids`, `accelerator: "none"` and `internet_enabled: false`. Root packaging can include extra frozen module hashes and provenance. No source contains its own hash. Source/helper/original and all read input hashes are checked again before completion.

`NTL_STAGE06E_INPUT` defaults to `/kaggle/input`; `NTL_STAGE06E_OUTPUT` defaults to `/kaggle/working/stage06e_cache`. The output directory must be absent or empty. The GLM cache is loaded directly from the unique attached `glm_features.npz`; no GLM inference occurs. Runtime preflight records package versions, resolved unfitted estimator parameters and complete threadpool library details before the first fit, and rejects any prospective environment or thread count discrepancy. See the frozen plan for exact versions.

The notebook wrapper must preserve failures explicitly (IPython need not invoke `sys.excepthook`):

```python
import runpy
import stage06e_cache_checks as cache_checks
try:
    runpy.run_path(capture_source_path, run_name="__main__")
except BaseException as error:
    cache_checks.preserve_active_failure(error)
    raise
```

Capture methods only copy existing values. There remain fifteen original estimator fits; each original `predict_proba` is called once with its original inputs and call shape. The original batch standardization, clipping, arithmetic, fold/group operations, seeds and bootstrap remain unchanged. A `warnings.showwarning` observer additionally records emitted scaler warnings during fits; original LogisticRegression warning capture and convergence logic are retained.

## Numeric/Unicode NPZ

`canonical_cpu_evidence.npz` has no object arrays. Every read uses `allow_pickle=False`. `N=2699`, `C=18`, and the first dimension `S=3` follows `scenarios=[1,3,6]`. IDs, classes and prompt strings are Unicode; no passage text is persisted. Original class order is preserved by string serialization.

| Array | Shape | Meaning |
| --- | --- | --- |
| `ids`, `groups` | `(N,)` | Ordered official row IDs and prompt IDs |
| `classes` | `(C,)` | Original sorted model labels |
| `y`, `fold` | `(N,)` | Integer class and archived one-based prompt fold |
| `scenarios` | `(3,)` | `[1,3,6]` |
| `plain_probability`, `plain_log_probability` | `(S,N,C)` | Original plain calls after their batch standardization; exact clipped logs |
| `plain_coverage`, `context_coverage` | `(S,N)` | Must be exactly one |
| `batches` | `(S,N)` | Literal C local batch IDs, scoped to a fold |
| `global_batches` | `(S,N)` | `batches + fold*1000000`; same membership with no cross-fold pooling |
| `inferred`, `oracle` | `(S,N)` | Literal C cluster labels, scoped to a fold |
| `cluster_size_inferred`, `cluster_size_oracle` | `(S,N)` | Cluster row count in that fold/context |
| `eligible_inferred`, `eligible_oracle` | `(S,N)` | Boolean source eligibility, cluster size 2 through 18 |
| `centered_probability_{key}` | `(S,N,C)` | Active source probabilities; inactive storage is zero and is **not** a probability distribution |
| `centered_log_probability_{key}` | `(S,N,C)` | Exact clipped source log probabilities on active rows; zero inactive log-evidence contribution |
| `component_log_evidence_{key}` | `(S,N,C)` | Original weighted evidence, `.05*plain_log`, then in-place `.25*centered_log` addition only on eligible rows |
| `prediction_{key}` | `(S,N)` | Original in-memory argmax |
| `component_coverage_{key}` | `(S,N)` | Must be exactly one |

`{key}` is one of `control`, `candidate`, `oracle_control`, `oracle_candidate`. Oracle keys use the oracle mask; the others use inferred eligibility. Cached clipped logs must be consumed directly without an exp/log round trip.

For each zero-based fold `f=0..4` and `label=plain/control/candidate`, prefix `head_f{f}_{label}__` contains `scaler_mean`, `scaler_scale`, `scaler_var`, `scaler_n_samples_seen`, `clf_coef`, `clf_intercept`, `clf_classes`, `clf_n_iter`, `fit_rows`, `valid_rows`. Snapshots are copied after fits; they are evidence, never a consumer inference path. The same state is persisted early in individual `head_f{f}_{label}.npz` files to retain completed fits if a later step fails. Resolved parameters are archived in `head_parameters.json`.

For each zero-based fold and training size 2/3/6, `training_f{f}_size{size}__rows` and `__assignments` capture the original candidate training partition, without RNG consumption or changed grouping. Training/validation prompt exclusion is also archived in `fold_and_training_proof.json`.

## Internal gate and manifests

`cache_gate.json` must have `status: "PASS"`, matching `canonical_cache_sha256`, `all_saved_arrays_exact_readback: true`, `component_predictions_exact: true`, `component_scores_atol_1e_12: true`, and `warnings_non_deprecation: 0`. No combined-model evaluation is permitted before this gate.

The gate checks numeric/Unicode types, unique aligned coordinates, exact archived folds, whole-prompt exclusion, complete once-only coverage, local/global batch membership, cluster sizes and eligibility, normalized finite active probabilities, exact clip/log relationships replayed on the original per-batch/plain and per-fold/eligible-centered call boundaries, zero inactive centered contributions, exact original arithmetic replay for all twelve predictions, and balanced accuracy reconciled independently through confusion/support/recall counts. Every serialized array must exactly equal its in-memory capture. No non-deprecation fit warning is accepted.

`cache_manifest.json` binds schema version, `cache_file`, compressed-file `cache_sha256`, `internal_gate_file` and its hash, actual producer `source_sha256`, `capture_helper_sha256`/`cache_checks_source_sha256`, original/plan/freeze hashes, the seven version IDs, `input_hashes` (`train_sha256`, `deb_cache_sha256`, `glm_cache_sha256`), full input artifact records, runtime, estimator parameters, warning records, `component_scores`, and every array's dtype/shape/raw C-order bytes SHA256. The consumer independently verifies file/source/gate/schema bindings and replays cached evidence.

`historical_comparison.json` is separate. It compares the new predictions to original C for each head/context/fold, including confusion counts, class recalls, repeated comparison mismatches and unique affected IDs. Historical object class metadata is never unpickled; original class order is bound by the exact official train, original source and completed C provenance. A hard-label mismatch remains an unresolved historical discrepancy and does not fail a valid same-run new cache.

On any producer failure, the notebook wrapper calls `preserve_active_failure`. It preserves `FAILED.json`, `FAILED_partial_capture.npz`, partial head states and status; partial evidence is never a valid cache. No success ensemble marker is written. Existing trial output is never overwritten.
