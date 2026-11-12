"""Stage 13F: test S07's dense-group assignment on a reduced held-out proxy."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy import __version__ as scipy_version
from sklearn import __version__ as sklearn_version
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

import stage13b_balancing_stress as reference
import stage13c_deberta_complement as prior
import stage13f_assignment_core as assignment


COMPLETE_ZERO = "COMPLETE_ZERO_COVERAGE"
COMPLETE_PAIRED = "COMPLETE_PAIRED_ASSIGNMENT_PROXY"
FAILED = "FAILED_STAGE13F_ASSIGNMENT_AUDIT"
SCOPE = "s07_assignment_rule_on_reused_prompt_heldout_glm_deberta_proxy"
TRAIN_VERSION = 170484
TRAIN_SHA = "40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707"
TEXT_SHA = "6496071874da8f911fafebd9eec6191e56134a97f00226dd8c78805d31f07e36"
STAGE13C_CM_SHA = "c076ee1a507753660bd7356baa3083d277c8fc8e61bedc81bae9192986caad9b"
STAGE13C_BA = 95677 / 201150
CLASSES = (
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5",
    "claude-sonnet-5", "gemini-3.1-pro-preview", "gemini-3.6-flash",
    "gemini-3.7-flash", "gemini-3.8-flash", "glm-5.3", "glm-5.3-flash",
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra",
    "gpt-6-astra", "kimi-k3", "qwen3.8-max",
)
REPORT_NAME = "assignment_audit_report.json"
require = reference.require


def train_inputs(path):
    """Recover the canonical fold axes from official bytes without opening a cache."""
    with path.open("rb") as stream:
        require(hashlib.file_digest(stream, "sha256").hexdigest() == TRAIN_SHA,
                "stage13f_official_train_hash")
        stream.seek(0)
        frame = pd.read_csv(stream, dtype=str, encoding="utf-8")
    require(len(frame) == reference.ROWS and set(frame.columns) == {"id", "prompt_id", "model", "text"}
            and not frame.isna().any().any()
            and all(frame[c].str.strip().ne("").all() for c in frame.columns), "stage13f_train_schema")
    ids, prompts, texts = (frame[c].to_numpy(dtype=str) for c in ("id", "prompt_id", "text"))
    labels = np.asarray(CLASSES, dtype=str)
    require(reference.digest_json(labels.tolist()) == reference.LABEL_SHA and set(frame.model) == set(CLASSES),
            "stage13f_train_classes")
    require(len(np.unique(ids)) == reference.ROWS and len(np.unique(prompts)) == reference.PROMPTS
            and reference.digest_json(ids.tolist()) == reference.ID_SHA, "stage13f_train_id_axis")
    require(reference.digest_json([hashlib.sha256(x.encode("utf-8")).hexdigest() for x in texts]) == TEXT_SHA,
            "stage13f_ordered_text_hash")
    targets = frame.model.map({label: i for i, label in enumerate(CLASSES)}).to_numpy(dtype=np.int64)
    require(np.bincount(targets).tolist() == [149] + [150] * 17, "stage13f_train_class_supports")
    require(not frame.duplicated(["prompt_id", "model"]).any(), "stage13f_duplicate_true_model_in_prompt")
    folds = np.zeros(reference.ROWS, dtype=np.int64)
    dummy = np.zeros((reference.ROWS, 1), dtype=np.uint8)
    for f, (_, selected) in enumerate(GroupKFold(reference.FOLDS).split(dummy, targets, prompts), 1):
        require((folds[selected] == 0).all(), "stage13f_fold_duplicate_rows")
        folds[selected] = f
    masks = prior.fold_masks(prompts, targets, folds)
    fold_map = {}
    for group, fold in zip(prompts, folds):
        require(fold_map.setdefault(str(group), int(fold)) == int(fold), "stage13f_prompt_crosses_folds")
    require(reference.digest_json(fold_map) == reference.FOLD_SHA, "stage13f_canonical_fold_map_hash")
    return ids, prompts, texts, targets, folds, labels, masks


def replay_stage13c(glm, deb, y, masks):
    """Apply unchanged Stage 13C mixing and balancing; verify its frozen confusion."""
    evidence = np.full((reference.ROWS, reference.K), np.nan, dtype=np.float64)
    residuals = []
    for val in masks:
        mixed = 0.75 * glm[val] + 0.25 * deb[val]
        evidence[val], residual = reference.evidence(mixed, True)
        residuals.append(residual)
    require(np.isfinite(evidence).all(), "stage13f_stage13c_evidence_coverage")
    control = reference.confusion(y, evidence.argmax(axis=1))
    digest = reference.digest_json(control.tolist())
    require(digest == STAGE13C_CM_SHA and abs(float(reference.recalls(control).mean()) - STAGE13C_BA) < 1e-15,
            "stage13f_stage13c_confusion_replay")
    return evidence, digest, residuals


def run(input_root, output_root):
    started = time.monotonic()
    root, out = Path(input_root), Path(output_root)
    report = dict(schema_version=1, status=FAILED, scope=SCOPE, fit_attempted=False, fit_performed=False,
                  cached_probabilities_loaded=False, stage13c_confusion_replayed=False,
                  paired_comparison_performed=False, full_s07_oof_admitted=False,
                  official_score_estimated=False, official_submission_created=False,
                  competition_test_read=False, raw_rows_or_predictions_exported=False,
                  uses_reused_development_prompts=True, s07_typography_guard_used=False,
                  train_sha256=TRAIN_SHA, train_ordered_text_sha256=TEXT_SHA,
                  canonical_prompt_fold_mapping_sha256=reference.FOLD_SHA,
                  expected_input_version_ids=[TRAIN_VERSION, reference.SOURCE_VERSION, prior.DEBERTA_VERSION],
                  expected_heldout_cache_sha256=reference.CACHE_SHA,
                  expected_deberta_cache_sha256=prior.DEBERTA_SHA,
                  sklearn_version=sklearn_version, numpy_version=np.__version__, scipy_version=scipy_version)
    writable = False
    try:
        require(not out.exists() or (out.is_dir() and not any(out.iterdir())), "fresh_output_directory")
        out.mkdir(parents=True, exist_ok=True)
        writable = True
        require(sklearn_version == prior.SKLEARN_VERSION, "stage13f_sklearn_version")
        ids, prompts, texts, y, folds, classes, masks = train_inputs(reference.one(root, "train.csv"))
        with threadpool_limits(limits=prior.THREADS):
            geometry, cluster_ids = assignment.geometry(texts, folds, prompts, y)
        eligible_rows = geometry["eligible_rows"]
        eligible_clusters = geometry["eligible_clusters"]
        require(type(eligible_rows) is int and type(eligible_clusters) is int and
                0 <= eligible_clusters * 12 <= eligible_rows <= min(eligible_clusters * 18, reference.ROWS),
                "stage13f_geometry_coverage")
        require(len(geometry["folds"]) == reference.FOLDS and cluster_ids.shape == folds.shape
                and cluster_ids.dtype.kind in "iu", "stage13f_geometry_folds_or_cluster_axis")
        report.update(geometry=geometry, rows=reference.ROWS, prompts=reference.PROMPTS, classes=classes.tolist())
        if eligible_rows == 0:
            report["status"] = COMPLETE_ZERO
        else:
            saved_ids, saved_prompts, saved_y, saved_folds, saved_classes, glm, producer = reference.cache_inputs(root)
            require(all(np.array_equal(a, b) for a, b in zip((ids, prompts, y, folds, classes),
                                                            (saved_ids, saved_prompts, saved_y, saved_folds, saved_classes))),
                    "stage13f_heldout_cache_row_or_fold_alignment")
            report["cached_probabilities_loaded"] = True
            prior.replay_control(glm, y, masks, producer)
            report["glm_controls_replayed"] = True
            features, id_mode = prior.load_features(reference.one(root, "enc_debv3l2_train.npz"), ids)
            report["fit_attempted"] = True
            deb, fit_details, pools = prior.fit_oof(features, y, masks)
            report.update(fit_performed=True, deberta_fit_folds=fit_details, thread_pools=pools,
                          deberta_id_verification=id_mode, feature_layers=[24, 6])
            evidence, digest, residuals = replay_stage13c(glm, deb, y, masks)
            report.update(stage13c_confusion_replayed=True, stage13c_confusion_sha256=digest,
                          mixture_weights={"glm": 0.75, "deberta": 0.25},
                          balancing={"alpha": 1, "iterations": 100, "fold_local": True,
                                     "maximum_column_mass_errors": residuals})
            evidence_sha = hashlib.sha256(evidence.astype("<f8", copy=False).tobytes()).hexdigest()
            comparison = assignment.compare(evidence, cluster_ids, folds, prompts, y)
            require(comparison["eligible_rows"] == eligible_rows
                    and comparison["eligible_clusters"] == eligible_clusters
                    and comparison["assignment_off_confusion_sha256"] == digest,
                    "stage13f_comparison_coverage_or_control_replay")
            require(hashlib.sha256(evidence.astype("<f8", copy=False).tobytes()).hexdigest() == evidence_sha,
                    "stage13f_evidence_mutated")
            require(hashlib.sha256(glm.astype("<f8", copy=False).tobytes()).hexdigest() == reference.PROB_SHA,
                    "stage13f_glm_cache_mutated")
            report.update(status=COMPLETE_PAIRED, paired_comparison_performed=True, comparison=comparison)
    except reference.GateError as error:
        report.update(failure_code=str(error))
    except Exception as error:
        report.update(failure_code="runtime_failure", failure_type=type(error).__name__)
    report["seconds"] = round(time.monotonic() - started, 6)
    if writable:
        try:
            saved = out / REPORT_NAME
            tmp = out / ("." + REPORT_NAME + ".tmp")
            tmp.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            tmp.replace(saved)
            require(json.loads(saved.read_text(encoding="utf-8")) == report, "stage13f_saved_report_mismatch")
        except Exception as error:
            report.update(status=FAILED, failure_code="report_write_or_readback_failure", failure_type=type(error).__name__)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default="/kaggle/input")
    parser.add_argument("--output-root", default="/kaggle/working/stage13f_assignment_audit")
    args = parser.parse_args()
    result = run(args.input_root, args.output_root)
    print("STAGE13F_ASSIGNMENT_AUDIT_JSON " + json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0 if result["status"] in (COMPLETE_ZERO, COMPLETE_PAIRED) else 2


if __name__ == "__main__":
    raise SystemExit(main())
