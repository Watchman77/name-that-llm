"""Replay frozen Stage 13F and diagnose its residual errors; no variant."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy import __version__ as scipy_version
from sklearn import __version__ as sklearn_version
from threadpoolctl import threadpool_limits

import stage13b_balancing_stress as reference
import stage13c_deberta_complement as stage13c
import stage13f_assignment_audit as stage13f
import stage13f_assignment_core as assignment
import stage13h_residual_core as residual


COMPLETE = "COMPLETE_STAGE13H_RESIDUAL_DIAGNOSTIC"
FAILED = "FAILED_STAGE13H_RESIDUAL_DIAGNOSTIC"
SCOPE = "stage13f_residual_errors_on_reused_prompt_heldout_glm_deberta_proxy"
REPORT_NAME = "residual_audit_report.json"
STAGE13F_ANCHOR_VERSION = 368776259
STAGE13F_CONTROL_CM_SHA = "6b3a2cd77f8b7d5c24f343dd685e3be63aaa63b3635d59b6ccefc2528d1ae696"
STAGE13F_CONTROL_BA = 0.6146184439473029
STAGE13F_GEOMETRY_SHA = "6df9d668c3ccfb3140939a752ae1612f7d7fce775291877285581c0c9370c3f1"
CONTROL_SIZES = list(range(12, 19))
CONTROL_ROWS, CONTROL_CLUSTERS, CONTROL_CORRECT = 1889, 112, 1659
RESIDUAL_ROWS = 810
require = reference.require


def baseline_geometry_digest(geometry):
    """Use the unchanged Stage 13G projection of the Stage 13F geometry."""
    folds = [[fold["fold"], fold["rows"], fold["true_prompts"], fold["singleton_fallback"],
              fold["eligible_rows"], fold["eligible_clusters"],
              sorted((int(size), count) for size, count in fold["cluster_size_histogram"].items())]
             for fold in geometry["folds"]]
    return reference.digest_json(dict(folds=folds, rows=geometry["eligible_rows"],
                                      clusters=geometry["eligible_clusters"]))


def verify_geometry(geometry, cluster_ids, folds):
    """Lock the fold-local grouping before loading either score source."""
    digest = baseline_geometry_digest(geometry)
    require(digest == STAGE13F_GEOMETRY_SHA, "stage13h_stage13f_geometry_replay")
    require(geometry["eligible_rows"] == CONTROL_ROWS and geometry["eligible_clusters"] == CONTROL_CLUSTERS
            and geometry["eligible_mixed_prompt_clusters"] == 0
            and geometry["eligible_clusters_with_duplicate_true_label"] == 0,
            "stage13h_stage13f_baseline_coverage")
    require(len(geometry["folds"]) == reference.FOLDS and cluster_ids.shape == folds.shape
            and cluster_ids.dtype.kind in "iu" and (cluster_ids >= 0).all(), "stage13h_geometry_axes")
    require(sum(int(size) * count for size, count in geometry["cluster_size_histogram"].items()
                if int(size) < 12) == reference.ROWS - CONTROL_ROWS == RESIDUAL_ROWS,
            "stage13h_residual_coverage")
    return digest


def verify_diagnostic(diagnostic):
    """Independently reconcile the frozen baseline and its diagnostic strata."""
    assigned, unassigned = diagnostic["assigned"], diagnostic["unassigned"]
    require(assigned["rows"] == CONTROL_ROWS and unassigned["rows"] == RESIDUAL_ROWS,
            "stage13h_diagnostic_coverage")
    matrix = np.asarray(diagnostic["baseline_confusion_counts"])
    a = np.asarray(assigned["confusion_counts"])
    u = np.asarray(unassigned["confusion_counts"])
    require(all(m.shape == (reference.K, reference.K) and m.dtype.kind in "iu" and (m >= 0).all()
                for m in (matrix, a, u)), "stage13h_diagnostic_matrices")
    require(int(a.sum()) == CONTROL_ROWS and int(u.sum()) == RESIDUAL_ROWS
            and np.array_equal(a + u, matrix), "stage13h_diagnostic_confusion_reconciliation")
    require(int(np.trace(matrix)) == diagnostic["baseline_correct"] == CONTROL_CORRECT
            and abs(float(reference.recalls(matrix).mean()) - STAGE13F_CONTROL_BA) < 1e-15
            and abs(diagnostic["baseline_balanced_accuracy"] - STAGE13F_CONTROL_BA) < 1e-15
            and reference.digest_json(matrix.tolist()) == diagnostic["baseline_confusion_sha256"]
            == STAGE13F_CONTROL_CM_SHA, "stage13h_stage13f_control_confusion_replay")
    require(all(s["correct"] == int(np.trace(m)) and s["errors"] == s["rows"] - s["correct"]
                for s, m in ((assigned, a), (unassigned, u))), "stage13h_diagnostic_count_reconciliation")


def run(input_root, output_root):
    started = time.monotonic()
    root, out = Path(input_root), Path(output_root)
    report = dict(schema_version=1, status=FAILED, scope=SCOPE, fit_attempted=False,
                  fit_performed=False, cached_probabilities_loaded=False,
                  stage13c_confusion_replayed=False, stage13f_geometry_replayed=False,
                  stage13f_control_confusion_replayed=False, residual_diagnostic_performed=False,
                  new_variant_scored=False, full_s07_oof_admitted=False, official_score_estimated=False,
                  official_submission_created=False, competition_test_read=False,
                  raw_rows_or_predictions_exported=False, uses_reused_development_prompts=True,
                  s07_typography_guard_used=False, control_assignment_sizes=CONTROL_SIZES,
                  stage13f_anchor_version=STAGE13F_ANCHOR_VERSION,
                  expected_stage13f_control_confusion_sha256=STAGE13F_CONTROL_CM_SHA,
                  expected_stage13f_geometry_sha256=STAGE13F_GEOMETRY_SHA,
                  train_sha256=stage13f.TRAIN_SHA, train_ordered_text_sha256=stage13f.TEXT_SHA,
                  canonical_prompt_fold_mapping_sha256=reference.FOLD_SHA,
                  expected_input_version_ids=[stage13f.TRAIN_VERSION, reference.SOURCE_VERSION,
                                              stage13c.DEBERTA_VERSION],
                  expected_heldout_cache_sha256=reference.CACHE_SHA,
                  expected_deberta_cache_sha256=stage13c.DEBERTA_SHA,
                  sklearn_version=sklearn_version, numpy_version=np.__version__, scipy_version=scipy_version)
    writable = False
    try:
        require(not out.exists() or (out.is_dir() and not any(out.iterdir())), "fresh_output_directory")
        out.mkdir(parents=True, exist_ok=True)
        writable = True
        require(sklearn_version == stage13c.SKLEARN_VERSION, "stage13h_sklearn_version")
        ids, prompts, texts, y, folds, classes, masks = stage13f.train_inputs(reference.one(root, "train.csv"))
        with threadpool_limits(limits=stage13c.THREADS):
            geometry, cluster_ids = assignment.geometry(texts, folds, prompts, y)
        geometry_sha = verify_geometry(geometry, cluster_ids, folds)
        report.update(geometry=geometry, rows=reference.ROWS, prompts=reference.PROMPTS,
                      classes=classes.tolist(), stage13f_geometry_replayed=True,
                      stage13f_geometry_sha256=geometry_sha, previously_untouched_rows=RESIDUAL_ROWS)

        saved_ids, saved_prompts, saved_y, saved_folds, saved_classes, glm, producer = reference.cache_inputs(root)
        require(all(np.array_equal(a, b) for a, b in zip((ids, prompts, y, folds, classes),
                                                        (saved_ids, saved_prompts, saved_y, saved_folds, saved_classes))),
                "stage13h_heldout_cache_row_or_fold_alignment")
        report["cached_probabilities_loaded"] = True
        stage13c.replay_control(glm, y, masks, producer)
        report["glm_controls_replayed"] = True
        features, id_mode = stage13c.load_features(reference.one(root, "enc_debv3l2_train.npz"), ids)
        report["fit_attempted"] = True
        deb, fit_details, pools = stage13c.fit_oof(features, y, masks)
        report.update(fit_performed=True, deberta_fit_folds=fit_details, thread_pools=pools,
                      deberta_id_verification=id_mode, feature_layers=[24, 6])
        evidence, anchor_digest, residuals = stage13f.replay_stage13c(glm, deb, y, masks)
        report.update(stage13c_confusion_replayed=True, stage13c_confusion_sha256=anchor_digest,
                      mixture_weights={"glm": 0.75, "deberta": 0.25},
                      balancing={"alpha": 1, "iterations": 100, "fold_local": True,
                                 "maximum_column_mass_errors": residuals})
        evidence_sha = hashlib.sha256(evidence.astype("<f8", copy=False).tobytes()).hexdigest()
        diagnostic = residual.diagnose(evidence, cluster_ids, folds, prompts, y,
                                       expected_baseline_confusion_sha256=STAGE13F_CONTROL_CM_SHA)
        verify_diagnostic(diagnostic)
        require(hashlib.sha256(evidence.astype("<f8", copy=False).tobytes()).hexdigest() == evidence_sha,
                "stage13h_evidence_mutated")
        require(hashlib.sha256(glm.astype("<f8", copy=False).tobytes()).hexdigest() == reference.PROB_SHA,
                "stage13h_glm_cache_mutated")
        report.update(status=COMPLETE, residual_diagnostic_performed=True, diagnostic=diagnostic,
                      stage13f_control_confusion_replayed=True,
                      stage13f_control_confusion_sha256=diagnostic["baseline_confusion_sha256"])
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
            require(json.loads(saved.read_text(encoding="utf-8")) == report, "stage13h_saved_report_mismatch")
        except Exception as error:
            report.update(status=FAILED, failure_code="report_write_or_readback_failure", failure_type=type(error).__name__)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default="/kaggle/input")
    parser.add_argument("--output-root", default="/kaggle/working/stage13h_residual_audit")
    args = parser.parse_args()
    result = run(args.input_root, args.output_root)
    print("STAGE13H_RESIDUAL_AUDIT_JSON " + json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0 if result["status"] == COMPLETE else 2


if __name__ == "__main__":
    raise SystemExit(main())
