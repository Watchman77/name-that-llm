"""Fit one fixed GLM readout per canonical prompt fold; Kaggle CPU only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_info, threadpool_limits

import stage13a_cache_admission as admission


COMPLETE = "COMPLETE_GLM_PLAIN_GROUPED_DEVELOPMENT_REFERENCE"
SCOPE = "glm_only_prompt_heldout_development"
REQUIRED_SKLEARN = "1.6.1"
THREADS = 2
HEAD = dict(C=0.0025, solver="lbfgs", max_iter=5000, tol=1e-4,
            fit_intercept=True, random_state=0)
PROBABILITY_TOL = 1e-12
AUDIT_CHECKS = ["fresh_output_directory", "unique_required_inputs", "exact_input_file_hashes",
                "official_train_axis_contract", "numeric_unicode_npz_headers",
                "glm_values_alignment_and_finiteness", "canonical_future_prompt_fold_partitions"]


def validate_probabilities(values, rows, classes):
    admission.check(values.shape == (rows, classes) and values.dtype == np.dtype("float64"),
                    "probability_shape_or_dtype")
    admission.check(np.isfinite(values).all() and (values > 0).all() and (values <= 1).all(),
                    "invalid_probability_values")
    error = float(np.max(np.abs(values.sum(axis=1) - 1.0)))
    admission.check(error <= PROBABILITY_TOL, "probability_rows_not_normalized")
    return error


def metrics(targets, probabilities, classes):
    predicted = probabilities.argmax(axis=1)
    counts = np.bincount(targets * classes + predicted, minlength=classes**2).reshape(classes, classes)
    supports = counts.sum(axis=1)
    admission.check((supports > 0).all() and counts.sum() == len(targets), "confusion_count_reconciliation")
    recall = counts.diagonal() / supports
    return float(recall.mean()), counts, supports, recall


def fit_grouped(features, targets, groups, folds, classes, n_folds, *, on_fold=None):
    """Use the same feature/estimator contract in the synthetic and production paths."""
    rows, class_count = len(targets), len(classes)
    admission.check(features.ndim == 2 and len(features) == rows and features.dtype == np.float64
                    and features.flags.c_contiguous and np.isfinite(features).all(), "invalid_readout_features")
    admission.check(targets.shape == groups.shape == folds.shape == (rows,)
                    and targets.dtype.kind in "iu"
                    and np.array_equal(np.unique(targets), np.arange(class_count)), "invalid_readout_axes")
    _, summaries = admission.validate_fold_partition(groups, folds, n_folds)
    # Check all folds before the first fit, so partial coverage cannot yield a misleading score.
    for number in range(1, n_folds + 1):
        mask = folds == number
        for subset in (mask, ~mask):
            admission.check(np.array_equal(np.unique(targets[subset]), np.arange(class_count)),
                            "fold_missing_class")
    probabilities = np.full((rows, class_count), np.nan, dtype=np.float64)
    filled = np.zeros(rows, dtype=np.int8)
    with threadpool_limits(limits=THREADS):
        pools = [{"api": item["user_api"], "threads": item["num_threads"]} for item in threadpool_info()]
        admission.check(all(0 < p["threads"] <= THREADS for p in pools), "thread_limit_not_applied")
        for summary in summaries:
            started = time.monotonic()
            mask = folds == summary["fold"]
            scaler = StandardScaler().fit(features[~mask])
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                classifier = LogisticRegression(**HEAD).fit(scaler.transform(features[~mask]), targets[~mask])
            admission.check(np.array_equal(classifier.classes_, np.arange(class_count)), "readout_class_order")
            iterations = [int(i) for i in classifier.n_iter_]
            admission.check(len(iterations) == 1 and 0 < iterations[0] < HEAD["max_iter"], "readout_iteration_limit")
            values = classifier.predict_proba(scaler.transform(features[mask]))
            validate_probabilities(values, int(mask.sum()), class_count)
            admission.check((filled[mask] == 0).all(), "duplicate_heldout_predictions")
            probabilities[mask], filled[mask] = values, 1
            score, _, support, _ = metrics(targets[mask], values, class_count)
            summary.update(balanced_accuracy=score, iterations=iterations[0],
                           seconds=round(time.monotonic() - started, 6), valid_class_supports=support.tolist())
            if on_fold is not None:
                on_fold(dict(summary))
    admission.check((filled == 1).all(), "incomplete_heldout_predictions")
    max_error = validate_probabilities(probabilities, rows, class_count)
    return probabilities, summaries, pools, max_error


def save_private_cache(path, *, ids, groups, targets, folds, classes, probabilities):
    arrays = dict(ids=ids, prompt_ids=groups, targets=targets, folds=folds,
                  classes=np.asarray(classes, dtype=str), probabilities=probabilities)
    admission.check(all(not value.dtype.hasobject for value in arrays.values()), "cache_object_dtype")
    tmp = path.with_name(".heldout_probabilities.tmp")
    try:
        with tmp.open("xb") as stream:
            np.savez(stream, **arrays)
        with np.load(tmp, allow_pickle=False) as saved:
            admission.check(set(saved.files) == set(arrays), "cache_readback_schema")
            for key, value in arrays.items():
                readback = saved[key]
                admission.check(readback.dtype == value.dtype and np.array_equal(readback, value),
                                "cache_readback_values")
        digest = admission.sha256(tmp)
        tmp.replace(path)
        return digest
    finally:
        tmp.unlink(missing_ok=True)


def run(input_root, output_root, *, on_fold=None):
    started = time.monotonic()
    root, out = Path(input_root), Path(output_root)
    report = dict(schema_version=1, status="FAILED_GLM_READOUT", scope=SCOPE, folds=[],
                  fit_attempted=False, fit_performed=False, heldout_predictions_produced=False, balancing_performed=False,
                  official_score_estimated=False, full_s07_oof_admitted=False,
                  stage13b_batch_coupled_comparison_admitted=False, heldout_cache_readback_passed=False,
                  raw_rows_or_predictions_exported=False, official_submission_started=False,
                  model_weights_saved=False, expected_input_versions={"competition": 170484, "own_glm_output": 359405443},
                  estimator=HEAD.copy(), classifier_layers=[8, 3], thread_limit=THREADS)
    write_report = False
    try:
        admission.check(not out.exists() or (out.is_dir() and not any(out.iterdir())), "fresh_readout_output_directory")
        out.mkdir(parents=True, exist_ok=True)
        write_report = True
        audited = admission.audit_inputs(root, out / "input_admission")
        report["runtime_versions"] = audited["runtime_versions"]
        report["admission_checks_passed"] = audited["checks_passed_count"]
        report["admission_status"] = audited["status"]
        if audited["status"] != "PASS_FROZEN_BASE_FEATURE_ADMISSION_ONLY":
            report["admission_failure_code"] = audited.get("failure_code")
        admission.check(audited["status"] == "PASS_FROZEN_BASE_FEATURE_ADMISSION_ONLY"
                        and audited["contract_is_production"] is True and audited["report_written"] is True
                        and audited["checks_passed"] == AUDIT_CHECKS, "real_input_admission_failed")
        admission.check(sklearn_version == REQUIRED_SKLEARN, "scikit_learn_version_mismatch")
        report.update(input_files=audited["files"], id_axis_sha256=audited["id_axis_sha256"],
                      label_axis_sha256=audited["label_axis_sha256"],
                      canonical_prompt_fold_mapping_sha256=audited["canonical_prompt_fold_mapping_sha256"])
        train_path = admission.unique_file(root, "train.csv")
        cache_path = admission.unique_file(root, "glm_features.npz")
        admission.check(admission.sha256(train_path) == admission.TRAIN_SHA256
                        and admission.sha256(cache_path) == admission.GLM_SHA256, "post_admission_input_hashes")
        train = pd.read_csv(train_path, dtype=str, usecols=["id", "prompt_id", "model"])
        ids, groups = train.id.to_numpy(dtype=str), train.prompt_id.to_numpy(dtype=str)
        classes = admission.CLASSES
        targets = train.model.map({name: i for i, name in enumerate(classes)}).to_numpy(dtype=np.int64)
        with np.load(cache_path, allow_pickle=False) as cache:
            admission.check(np.array_equal(cache["ids"], ids) and np.array_equal(cache["layers"], [8, 3, 24]),
                            "readout_feature_axis_binding")
            values = cache["features"]
            features = np.empty((len(ids), 8192), dtype=np.float64, order="C")
            features[:, :4096], features[:, 4096:] = values[:, 0, :], values[:, 1, :]
            del values
        folds = np.zeros(len(ids), dtype=np.int64)
        for number, (_, valid) in enumerate(GroupKFold(5).split(features, targets, groups), 1):
            admission.check((folds[valid] == 0).all(), "overlapping_outer_fold_indices")
            folds[valid] = number
        mapping, summaries = admission.validate_fold_partition(groups, folds, 5)
        admission.check(admission.json_hash(mapping) == admission.FOLD_MAP_SHA256
                        and summaries == audited["fold_partitions"], "readout_fold_map_mismatch")
        admission.check(admission.json_hash(ids.tolist()) == audited["id_axis_sha256"], "readout_id_axis_hash")
        report.update(rows=len(ids), prompts=len(mapping), classes=list(classes), fit_attempted=True,
                      feature_shape=list(features.shape))

        def record_fold(summary):
            report["folds"].append(summary)
            report["fit_performed"] = True
            if on_fold is not None:
                on_fold(summary)

        probs, summaries, pools, error = fit_grouped(features, targets, groups, folds, classes, 5, on_fold=record_fold)
        ba, counts, supports, recalls = metrics(targets, probs, len(classes))
        admission.check(supports.tolist() == [audited["class_supports"][c] for c in classes], "class_support_mismatch")
        admission.check(len(summaries) == 5 and sum(f["valid_rows"] for f in summaries) == len(ids),
                        "heldout_fold_count_mismatch")
        cache_hash = save_private_cache(out / "heldout_probabilities.npz", ids=ids, groups=groups,
                                        targets=targets, folds=folds, classes=classes, probabilities=probs)
        report.update(status=COMPLETE, fit_performed=True, heldout_predictions_produced=True,
                      heldout_cache_readback_passed=True, heldout_cache_sha256=cache_hash,
                      probability_shape=list(probs.shape), probability_dtype=str(probs.dtype),
                      probability_values_sha256=hashlib.sha256(probs.astype("<f8", copy=False).tobytes()).hexdigest(),
                      probability_max_row_sum_error=error, pooled_balanced_accuracy=ba,
                      confusion_counts=counts.tolist(), class_supports=supports.tolist(), class_recalls=recalls.tolist(),
                      convergence_warnings=0, observed_thread_pools=pools)
    except admission.AdmissionFailure as error:
        report.update(failure_code=str(error), failure_type="AdmissionFailure")
    except ConvergenceWarning:
        report.update(failure_code="readout_convergence_warning", failure_type="ConvergenceWarning")
    except Exception as error:
        # Exception strings may contain paths or row values; publish only the type.
        report.update(failure_code="readout_runtime_failure", failure_type=type(error).__name__)
    report["seconds"] = round(time.monotonic() - started, 6)
    if write_report:
        try:
            tmp = out / ".readout_report.tmp"
            tmp.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            tmp.replace(out / "readout_report.json")
        except Exception as error:
            report.update(status="FAILED_GLM_READOUT", failure_code="readout_report_write_failed",
                          failure_type=type(error).__name__)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default="/kaggle/input")
    parser.add_argument("--output-root", default="/kaggle/working/stage13_glm_readout")
    args = parser.parse_args()

    def progress(fold):
        print("STAGE13_GLM_FOLD", json.dumps(fold, sort_keys=True, allow_nan=False), flush=True)

    report = run(args.input_root, args.output_root, on_fold=progress)
    print("STAGE13_GLM_READOUT_JSON_BEGIN", flush=True)
    print(json.dumps(report, sort_keys=True, allow_nan=False), flush=True)
    print("STAGE13_GLM_READOUT_JSON_END", flush=True)
    return 0 if report["status"] == COMPLETE else 2


if __name__ == "__main__":
    raise SystemExit(main())
