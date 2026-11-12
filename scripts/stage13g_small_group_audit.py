"""Stage 13G: test a fixed 10–11 row extension to Stage 13F assignment."""
from __future__ import annotations

import argparse
from fractions import Fraction
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
import stage13g_small_group_core as extension


COMPLETE_ZERO = "COMPLETE_NO_SMALL_GROUP_COVERAGE"
COMPLETE_NO_GO = "COMPLETE_SMALL_GROUP_GEOMETRY_NO_GO"
COMPLETE_PAIRED = "COMPLETE_SMALL_GROUP_PAIRED_PROXY"
FAILED = "FAILED_STAGE13G_SMALL_GROUP_AUDIT"
SCOPE = "s07_small_group_extension_on_reused_prompt_heldout_glm_deberta_proxy"
REPORT_NAME = "small_group_audit_report.json"
STAGE13F_ANCHOR_VERSION = 368776259
STAGE13F_CONTROL_CM_SHA = "6b3a2cd77f8b7d5c24f343dd685e3be63aaa63b3635d59b6ccefc2528d1ae696"
STAGE13F_CONTROL_BA = 0.6146184439473029
STAGE13F_GEOMETRY_SHA = "6df9d668c3ccfb3140939a752ae1612f7d7fce775291877285581c0c9370c3f1"
CONTROL_SIZES = list(range(12, 19))
EXTENSION_SIZES = [10, 11]
CONTROL_ROWS, CONTROL_CLUSTERS = 1889, 112
require = reference.require


def baseline_geometry_digest(geometry):
    """Checksum the original Stage 13F cluster histogram and coverage by fold."""
    folds = [[fold["fold"], fold["rows"], fold["true_prompts"], fold["singleton_fallback"],
              fold["eligible_rows"], fold["eligible_clusters"],
              sorted((int(size), count) for size, count in fold["cluster_size_histogram"].items())]
             for fold in geometry["folds"]]
    return reference.digest_json(dict(folds=folds, rows=geometry["eligible_rows"],
                                      clusters=geometry["eligible_clusters"]))


def verify_geometry(geometry, cluster_ids, folds):
    """Require exact Stage 13F replay, then reconcile the new candidate scope."""
    base, added = geometry["base"], geometry["extension"]
    digest = baseline_geometry_digest(base)
    require(digest == STAGE13F_GEOMETRY_SHA, "stage13g_stage13f_geometry_replay")
    require(base["eligible_rows"] == CONTROL_ROWS and base["eligible_clusters"] == CONTROL_CLUSTERS
            and base["eligible_mixed_prompt_clusters"] == 0
            and base["eligible_clusters_with_duplicate_true_label"] == 0,
            "stage13g_stage13f_baseline_coverage")
    require(cluster_ids.shape == folds.shape and cluster_ids.dtype.kind in "iu"
            and (cluster_ids >= 0).all(), "stage13g_cluster_axis")
    histogram = base["cluster_size_histogram"]
    expected_n = sum(histogram.get(str(n), 0) for n in EXTENSION_SIZES)
    expected_rows = sum(n * histogram.get(str(n), 0) for n in EXTENSION_SIZES)
    require(added["clusters"] == expected_n and added["rows"] == expected_rows
            and sum(int(n) * count for n, count in histogram.items() if int(n) < 12)
            == geometry["prior_untouched_rows"] == reference.ROWS - CONTROL_ROWS,
            "stage13g_candidate_geometry_coverage")
    for size in range(1, 12):
        group = geometry["untouched_by_size"][str(size)]
        require(group["clusters"] == histogram.get(str(size), 0) and group["rows"] == size * group["clusters"],
                "stage13g_untouched_size_coverage")
    for original, candidate in zip(base["folds"], geometry["folds"], strict=True):
        sizes = original["cluster_size_histogram"]
        require(original["fold"] == candidate["fold"]
                and candidate["extension"]["clusters"] == sum(sizes.get(str(n), 0) for n in EXTENSION_SIZES)
                and candidate["extension"]["rows"] == sum(n * sizes.get(str(n), 0) for n in EXTENSION_SIZES),
                "stage13g_candidate_fold_coverage")
    require(0 <= added["mixed_prompt_clusters"] <= expected_n
            and 0 <= added["duplicate_true_label_clusters"] <= expected_n,
            "stage13g_candidate_geometry_counts")
    pure = added["mixed_prompt_clusters"] == 0
    unique = added["duplicate_true_label_clusters"] == 0
    require(geometry["passes_extension_geometry_gate"] is (expected_n > 0 and pure and unique),
            "stage13g_candidate_geometry_gate_reconciliation")
    return digest, expected_n, expected_rows, pure, unique


def run(input_root, output_root):
    started = time.monotonic()
    root, out = Path(input_root), Path(output_root)
    report = dict(schema_version=1, status=FAILED, scope=SCOPE, fit_attempted=False,
                  fit_performed=False, cached_probabilities_loaded=False,
                  stage13c_confusion_replayed=False, stage13f_geometry_replayed=False,
                  stage13f_control_confusion_replayed=False, paired_comparison_performed=False,
                  full_s07_oof_admitted=False, official_score_estimated=False,
                  official_submission_created=False, competition_test_read=False,
                  raw_rows_or_predictions_exported=False, uses_reused_development_prompts=True,
                  s07_typography_guard_used=False, control_assignment_sizes=CONTROL_SIZES,
                  extension_assignment_sizes=EXTENSION_SIZES,
                  stage13f_anchor_version=STAGE13F_ANCHOR_VERSION,
                  expected_stage13f_control_confusion_sha256=STAGE13F_CONTROL_CM_SHA,
                  expected_stage13f_geometry_sha256=STAGE13F_GEOMETRY_SHA,
                  train_sha256=stage13f.TRAIN_SHA, train_ordered_text_sha256=stage13f.TEXT_SHA,
                  canonical_prompt_fold_mapping_sha256=reference.FOLD_SHA,
                  expected_input_version_ids=[stage13f.TRAIN_VERSION, reference.SOURCE_VERSION,
                                              stage13c.DEBERTA_VERSION],
                  expected_heldout_cache_sha256=reference.CACHE_SHA,
                  expected_deberta_cache_sha256=stage13c.DEBERTA_SHA,
                  sklearn_version=sklearn_version, numpy_version=np.__version__,
                  scipy_version=scipy_version)
    writable = False
    try:
        require(not out.exists() or (out.is_dir() and not any(out.iterdir())), "fresh_output_directory")
        out.mkdir(parents=True, exist_ok=True)
        writable = True
        require(sklearn_version == stage13c.SKLEARN_VERSION, "stage13g_sklearn_version")
        ids, prompts, texts, y, folds, classes, masks = stage13f.train_inputs(reference.one(root, "train.csv"))
        with threadpool_limits(limits=stage13c.THREADS):
            geometry, cluster_ids = extension.geometry(texts, folds, prompts, y)
        digest, n_groups, n_rows, pure, unique = verify_geometry(geometry, cluster_ids, folds)
        report.update(geometry=geometry, rows=reference.ROWS, prompts=reference.PROMPTS,
                      classes=classes.tolist(), stage13f_geometry_replayed=True,
                      stage13f_geometry_sha256=digest,
                      previously_untouched_rows=reference.ROWS - CONTROL_ROWS,
                      geometry_gate=dict(nonempty=n_groups > 0, all_candidate_groups_prompt_pure=pure,
                                         all_candidate_groups_have_distinct_true_labels=unique,
                                         passed=n_groups > 0 and pure and unique))
        if n_groups == 0:
            report.update(status=COMPLETE_ZERO, extension_screen_passed=False,
                          extension_decision="KEEP_12_TO_18_NO_CANDIDATE_GROUPS")
        elif not (pure and unique):
            report.update(status=COMPLETE_NO_GO, extension_screen_passed=False,
                          extension_decision="KEEP_12_TO_18_GEOMETRY_GATE_FAILED")
        else:
            saved_ids, saved_prompts, saved_y, saved_folds, saved_classes, glm, producer = reference.cache_inputs(root)
            require(all(np.array_equal(a, b) for a, b in zip((ids, prompts, y, folds, classes),
                                                            (saved_ids, saved_prompts, saved_y, saved_folds, saved_classes))),
                    "stage13g_heldout_cache_row_or_fold_alignment")
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
            # The comparison verifies the expected control digest before applying the extension.
            comparison = extension.compare(evidence, cluster_ids, folds, prompts, y,
                                           expected_baseline_confusion_sha256=STAGE13F_CONTROL_CM_SHA)
            control_digest = comparison["baseline_confusion_sha256"]
            require(control_digest == STAGE13F_CONTROL_CM_SHA
                    and abs(comparison["baseline_ba"] - STAGE13F_CONTROL_BA) < 1e-15,
                    "stage13g_stage13f_control_confusion_replay")
            report.update(stage13f_control_confusion_replayed=True,
                          stage13f_control_confusion_sha256=control_digest)
            require(comparison["extension_rows"] == n_rows and comparison["extension_clusters"] == n_groups
                    and comparison["baseline_confusion_sha256"] == control_digest
                    and comparison["outside_extension_unchanged"] and comparison["fixed_log_evidence_unchanged"],
                    "stage13g_comparison_coverage_or_baseline_replay")
            require(hashlib.sha256(evidence.astype("<f8", copy=False).tobytes()).hexdigest() == evidence_sha,
                    "stage13g_evidence_mutated")
            require(hashlib.sha256(glm.astype("<f8", copy=False).tobytes()).hexdigest() == reference.PROB_SHA,
                    "stage13g_glm_cache_mutated")
            gates = dict(gain_at_least_half_pp=Fraction(comparison["delta_pp_exact"]) >= Fraction(1, 2),
                         ci_lower_positive=comparison["conditional_delta_ci95_pp"][0] > 0,
                         no_class_loss_over_five_pp=min(comparison["class_delta_pp"]) >= -5)
            report.update(status=COMPLETE_PAIRED, paired_comparison_performed=True, comparison=comparison,
                          decision_gates=gates, extension_screen_passed=all(gates.values()),
                          extension_decision="ELIGIBLE_FOR_SEPARATE_FOLLOW_UP_ONLY" if all(gates.values())
                          else "KEEP_12_TO_18_PAIRED_GATE_FAILED")
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
            require(json.loads(saved.read_text(encoding="utf-8")) == report, "stage13g_saved_report_mismatch")
        except Exception as error:
            report.update(status=FAILED, failure_code="report_write_or_readback_failure", failure_type=type(error).__name__)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default="/kaggle/input")
    parser.add_argument("--output-root", default="/kaggle/working/stage13g_small_group_audit")
    args = parser.parse_args()
    result = run(args.input_root, args.output_root)
    print("STAGE13G_SMALL_GROUP_AUDIT_JSON " + json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0 if result["status"] in (COMPLETE_ZERO, COMPLETE_NO_GO, COMPLETE_PAIRED) else 2


if __name__ == "__main__":
    raise SystemExit(main())
