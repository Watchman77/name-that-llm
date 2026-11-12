"""Test one frozen four-class Gemini specialist on the Stage 13F control."""
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
import stage13f_assignment_core as assignment
import stage13h_residual_core as residual
import stage13h_residual_audit as stage13h
import stage13i_gemini_core as specialist


COMPLETE = "COMPLETE_STAGE13I_GEMINI_SPECIALIST"
FAILED = "FAILED_STAGE13I_GEMINI_SPECIALIST"
SCOPE = "gemini_specialist_on_reused_prompt_heldout_glm_deberta_proxy"
REPORT_NAME = "gemini_audit_report.json"
STAGE13F_CONTROL_CM_SHA = stage13h.STAGE13F_CONTROL_CM_SHA
STAGE13F_CONTROL_BA = stage13h.STAGE13F_CONTROL_BA
STAGE13H_REPORT_COMMIT = "8b45ab4d2ea32061891f8ca7a2199af51d026d54"
STAGE13H_ANCHOR_VERSION = 368859699
GEMINI = ("gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash")
# Nonzero rows of Stage 13H's frozen unassigned confusion, restricted to the
# four Gemini prediction columns, in canonical class order. Counts only.
ELIGIBLE_CM_ROWS = ((2, 1, 0, 0, 0), (5, 47, 5, 2, 2), (6, 11, 31, 8, 5),
                    (7, 6, 12, 18, 18), (8, 3, 11, 8, 32), (12, 0, 0, 0, 1),
                    (15, 0, 1, 0, 0), (17, 0, 0, 0, 1))
EXPECTED_ELIGIBLE = dict(eligible_rows=223, eligible_true_gemini_rows=219,
                         eligible_true_non_gemini_rows=4, eligible_baseline_errors=95,
                         eligible_within_family_baseline_errors=91, eligible_baseline_correct=128)
require = reference.require


def replay_residual_gate(evidence, cluster_ids, folds, prompts, y, classes):
    """Replay the full Stage 13F control and Stage 13H gated cohort before fit."""
    _, baseline, covered, _, _, cm, digest = residual._replay(
        evidence, cluster_ids, folds, prompts, y, STAGE13F_CONTROL_CM_SHA)
    require(digest == STAGE13F_CONTROL_CM_SHA and int(np.trace(cm)) == stage13h.CONTROL_CORRECT
            and abs(float(reference.recalls(cm).mean()) - STAGE13F_CONTROL_BA) < 1e-15,
            "stage13i_stage13f_control_replay")
    labels = tuple(classes)
    require(tuple(labels[i] for i in range(5, 9)) == GEMINI, "stage13i_gemini_class_axis")
    family = np.arange(5, 9)
    eligible = ~covered & np.isin(baseline, family)
    gated_cm = reference.confusion(y[eligible], baseline[eligible])
    expected = np.zeros((reference.K, reference.K), dtype=np.int64)
    for i, *counts in ELIGIBLE_CM_ROWS:
        expected[i, 5:9] = counts
    require(int((~covered).sum()) == stage13h.RESIDUAL_ROWS
            and np.array_equal(gated_cm, expected), "stage13i_stage13h_eligible_confusion_replay")
    return cm, gated_cm, family


def verify_comparison(comparison, replayed, gated, family):
    """Verify paired reconciliation and immutable eligible anchor independently."""
    require(all(comparison[k] == value for k, value in EXPECTED_ELIGIBLE.items()),
            "stage13i_eligible_counts_replay")
    require(all(comparison[k] is True for k in (
        "outside_eligible_unchanged", "predicted_family_unchanged", "fixed_features_and_log_evidence_unchanged",
        "gate_uses_only_inferred_assignment_and_baseline_prediction")) and comparison["row_level_outputs"] is False,
        "stage13i_fixed_scope_invariants")
    baseline = np.asarray(comparison["baseline_confusion_counts"])
    candidate = np.asarray(comparison["candidate_confusion_counts"])
    old = np.asarray(comparison["eligible_baseline_confusion_counts"])
    new = np.asarray(comparison["eligible_candidate_confusion_counts"])
    shape = (reference.K, reference.K)
    require(all(x.shape == shape and x.dtype.kind in "iu" and (x >= 0).all()
                for x in (baseline, candidate, old, new)), "stage13i_comparison_matrix_schema")
    require(np.array_equal(baseline, replayed) and np.array_equal(old, gated)
            and np.array_equal(candidate - baseline, new - old)
            and np.array_equal(old.sum(axis=1), new.sum(axis=1)), "stage13i_paired_matrix_reconciliation")
    outside = np.setdiff1d(np.arange(reference.K), family)
    require(not old[:, outside].any() and not new[:, outside].any()
            and int(new[outside].sum()) == EXPECTED_ELIGIBLE["eligible_true_non_gemini_rows"]
            and int(np.trace(new)) == comparison["eligible_candidate_correct"], "stage13i_family_gate")
    def family_matrix(cm):
        axes = (outside, family)
        return [[int(cm[np.ix_(i, j)].sum()) for j in axes] for i in axes]

    require(family_matrix(baseline) == family_matrix(candidate)
            == comparison["baseline_family_confusion_counts"] == comparison["candidate_family_confusion_counts"],
            "stage13i_family_confusion_reconciliation")
    supports = baseline.sum(axis=1)
    require(np.array_equal(supports, candidate.sum(axis=1)) and (supports > 0).all(),
            "stage13i_class_supports")
    class_deltas = [Fraction(100 * int(candidate[i, i] - baseline[i, i]), int(supports[i]))
                    for i in range(reference.K)]
    exact_delta = sum(class_deltas) / reference.K
    delta = Fraction(comparison["delta_pp_exact"])
    require(delta == exact_delta
            and abs(comparison["baseline_ba"] - STAGE13F_CONTROL_BA) < 1e-15
            and abs(comparison["candidate_ba"] - float(reference.recalls(candidate).mean())) < 1e-15
            and abs(comparison["delta_pp"] - float(exact_delta)) < 1e-15,
            "stage13i_balanced_accuracy_reconciliation")
    correct_delta = int(np.trace(candidate) - np.trace(baseline))
    require(correct_delta == comparison["corrected"] - comparison["broken"]
            and comparison["corrected"] + comparison["broken"] <= comparison["changed"]
            <= EXPECTED_ELIGIBLE["eligible_rows"],
            "stage13i_changed_decision_reconciliation")
    fold_reports = comparison["folds"]
    require([f["fold"] for f in fold_reports] == list(range(1, reference.FOLDS + 1))
            and all(f["fit_class_counts"] == [120] * 4 and f["fit_rows"] == 480
                    and f["fit_prompts"] == 120 and f["heldout_prompts"] == 30 and f["prompt_overlap"] == 0
                    for f in fold_reports)
            and all(sum(f[k] for f in fold_reports) == comparison[k]
                    for k in ("eligible_rows", "corrected", "broken", "changed")),
            "stage13i_fold_fit_and_pair_coverage")
    low, high = comparison["conditional_delta_ci95_pp"]
    require(np.isfinite([low, high]).all() and low <= high, "stage13i_ci_schema")
    gates = dict(gain_at_least_half_pp=exact_delta >= Fraction(1, 2), ci_lower_positive=low > 0,
                 no_class_loss_over_five_pp=min(class_deltas) >= -5)
    require(comparison["passes_development_screen"] is all(gates.values()),
            "stage13i_development_gate_reconciliation")
    return gates


def run(input_root, output_root):
    started = time.monotonic()
    root, out = Path(input_root), Path(output_root)
    report = dict(schema_version=1, status=FAILED, scope=SCOPE, fit_attempted=False,
                  fit_performed=False, cached_probabilities_loaded=False,
                  stage13c_confusion_replayed=False, stage13f_geometry_replayed=False,
                  stage13f_control_confusion_replayed=False, stage13h_eligible_confusion_replayed=False,
                  specialist_fit_attempted=False, specialist_fit_performed=False,
                  new_variant_scored=False, full_s07_oof_admitted=False, official_score_estimated=False,
                  official_submission_created=False, competition_test_read=False,
                  raw_rows_or_predictions_exported=False, uses_reused_development_prompts=True,
                  s07_typography_guard_used=False, control_assignment_sizes=stage13h.CONTROL_SIZES,
                  stage13f_anchor_version=stage13h.STAGE13F_ANCHOR_VERSION,
                  stage13h_anchor_version=STAGE13H_ANCHOR_VERSION,
                  stage13h_anchor_commit=STAGE13H_REPORT_COMMIT,
                  gate="Unassigned under Stage 13F and its frozen prediction belongs to Gemini",
                  specialist_labels=list(GEMINI), specialist_head=stage13c.HEAD,
                  expected_stage13f_control_confusion_sha256=STAGE13F_CONTROL_CM_SHA,
                  expected_stage13f_geometry_sha256=stage13h.STAGE13F_GEOMETRY_SHA,
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
        require(sklearn_version == stage13c.SKLEARN_VERSION, "stage13i_sklearn_version")
        ids, prompts, texts, y, folds, classes, masks = stage13f.train_inputs(reference.one(root, "train.csv"))
        with threadpool_limits(limits=stage13c.THREADS):
            geometry, cluster_ids = assignment.geometry(texts, folds, prompts, y)
        geometry_sha = stage13h.verify_geometry(geometry, cluster_ids, folds)
        report.update(geometry=geometry, rows=reference.ROWS, prompts=reference.PROMPTS,
                      classes=classes.tolist(), stage13f_geometry_replayed=True,
                      stage13f_geometry_sha256=geometry_sha, previously_untouched_rows=stage13h.RESIDUAL_ROWS)

        saved_ids, saved_prompts, saved_y, saved_folds, saved_classes, glm, producer = reference.cache_inputs(root)
        require(all(np.array_equal(a, b) for a, b in zip((ids, prompts, y, folds, classes),
                                                        (saved_ids, saved_prompts, saved_y, saved_folds, saved_classes))),
                "stage13i_heldout_cache_row_or_fold_alignment")
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
        baseline, gated, family = replay_residual_gate(evidence, cluster_ids, folds, prompts, y, classes)
        report.update(stage13f_control_confusion_replayed=True,
                      stage13f_control_confusion_sha256=STAGE13F_CONTROL_CM_SHA,
                      stage13h_eligible_confusion_replayed=True, specialist_fit_attempted=True)
        comparison = specialist.compare(features, evidence, cluster_ids, folds, prompts, y, classes,
                                        expected_baseline_confusion_sha256=STAGE13F_CONTROL_CM_SHA)
        gates = verify_comparison(comparison, baseline, gated, family)
        require(hashlib.sha256(evidence.astype("<f8", copy=False).tobytes()).hexdigest() == evidence_sha,
                "stage13i_evidence_mutated")
        require(hashlib.sha256(glm.astype("<f8", copy=False).tobytes()).hexdigest() == reference.PROB_SHA,
                "stage13i_glm_cache_mutated")
        report.update(status=COMPLETE, specialist_fit_performed=True, new_variant_scored=True,
                      comparison=comparison, decision_gates=gates,
                      specialist_screen_passed=all(gates.values()),
                      specialist_decision="ELIGIBLE_FOR_SEPARATE_FOLLOW_UP_ONLY" if all(gates.values())
                      else "KEEP_FROZEN_CONTROL_PAIRED_GATE_FAILED")
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
            require(json.loads(saved.read_text(encoding="utf-8")) == report, "stage13i_saved_report_mismatch")
        except Exception as error:
            report.update(status=FAILED, failure_code="report_write_or_readback_failure", failure_type=type(error).__name__)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default="/kaggle/input")
    parser.add_argument("--output-root", default="/kaggle/working/stage13i_gemini_audit")
    args = parser.parse_args()
    result = run(args.input_root, args.output_root)
    print("STAGE13I_GEMINI_AUDIT_JSON " + json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0 if result["status"] == COMPLETE else 2


if __name__ == "__main__":
    raise SystemExit(main())
