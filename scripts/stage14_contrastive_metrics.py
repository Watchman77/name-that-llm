"""Compare three fixed Stage 14 model arms on the same held-prompt predictions.

The primary readout is row-independent. The secondary readout applies the
unchanged Stage 13F text grouping, fold-local Sinkhorn, and dense assignment
to each arm. This module returns aggregates only.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy.special import logsumexp

import stage13b_balancing_stress as reference
import stage13f_assignment_core as assignment


K = reference.K
BOOTSTRAP_DRAWS = 4000
BOOTSTRAP_SEED = 141401
HEAD_WEIGHT = 0.20
MIN_CONTROL_GAIN_PP = Fraction(1)
MAX_CLASS_REGRESSION_PP = Fraction(5)
ARM_NAMES = ("control", "ce_blend", "contrast_blend")
require = reference.require


def _class_state(y, p):
    cm = reference.confusion(y, p)
    recall = reference.recalls(cm)
    return {"balanced_accuracy": float(recall.mean()), "correct_rows": int(cm.trace()),
            "class_recalls": recall.tolist(), "confusion_counts": cm.tolist(),
            "confusion_sha256": reference.digest_json(cm.tolist())}


def _blend(control, logits):
    require(logits.shape == control.shape and logits.dtype.kind == "f"
            and np.isfinite(logits).all(), "stage14_head_logit_axis")
    z = logits.astype(np.float64, copy=False)
    lp = (1 - HEAD_WEIGHT) * np.log(control) + HEAD_WEIGHT * (z - logsumexp(z, axis=1, keepdims=True))
    lp -= logsumexp(lp, axis=1, keepdims=True)
    out = np.exp(lp)
    reference.validate_p(out)
    return out


def _bootstrap_plan(y, prompts, folds):
    """Precompute common prompt multiplicities and class support for all arms."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    support = np.zeros((BOOTSTRAP_DRAWS, K), dtype=np.int64)
    plan = []
    for fold in np.unique(folds):
        rows = np.flatnonzero(folds == fold)
        unique_prompts, index = np.unique(prompts[rows], return_inverse=True)
        n = len(unique_prompts)
        counts = np.zeros((n, K), dtype=np.int64)
        np.add.at(counts, (index, y[rows]), 1)
        require((counts.sum(axis=0) > 0).all(), "stage14_fold_class_support")
        weights = rng.multinomial(n, np.full(n, 1 / n), size=BOOTSTRAP_DRAWS)
        support += weights @ counts
        plan.append((int(fold), rows, index, weights))
    require((support > 0).all(), "stage14_bootstrap_class_support")
    return plan, support


def _paired(y, p, q, plan, support, assigned, reference_evidence):
    counts = assignment.paired_counts(y, p, q)
    before, after = p == y, q == y
    diff = after.astype(np.int64) - before
    supports = np.bincount(y, minlength=K)
    class_diff = np.zeros(K, dtype=np.int64)
    np.add.at(class_diff, y, diff)
    class_pp = [Fraction(int(d) * 100, int(s)) for d, s in zip(class_diff, supports, strict=True)]
    delta = sum(class_pp) / K
    draw_diff = np.zeros_like(support)
    fold_reports = []
    for fold, rows, index, weights in plan:
        prompt_diff = np.zeros((weights.shape[1], K), dtype=np.int64)
        np.add.at(prompt_diff, (index, y[rows]), diff[rows])
        draw_diff += weights @ prompt_diff
        fold_support = np.bincount(y[rows], minlength=K)
        fold_delta = (prompt_diff.sum(axis=0) / fold_support).mean() * 100
        fold_reports.append({"fold": fold, "rows": len(rows), "delta_pp": float(fold_delta),
                             **assignment.paired_counts(y[rows], p[rows], q[rows])})
    require(all(sum(x[k] for x in fold_reports) == v for k, v in counts.items()),
            "stage14_pair_reconciliation")
    low, high = np.percentile((draw_diff / support).mean(axis=1) * 100, [2.5, 97.5])
    true_scores = reference_evidence[np.arange(len(y)), y][:, None]
    higher = ((reference_evidence > true_scores) | ((reference_evidence == true_scores)
              & (np.arange(K)[None, :] < y[:, None]))).sum(axis=1)
    hard = (higher >= 3) & ~assigned & ~before
    result = {**counts, "delta_pp": float(delta), "delta_pp_exact": str(delta),
              "conditional_paired_ci95_pp": [float(low), float(high)],
              "class_delta_pp": [float(d) for d in class_pp],
              "worst_class_delta_pp": float(min(class_pp)),
              "class_regression_cap_pass": min(class_pp) >= -MAX_CLASS_REGRESSION_PP,
              "unassigned": assignment.paired_counts(y[~assigned], p[~assigned], q[~assigned]),
              "unassigned_prior_errors_true_rank_below_3": int(hard.sum()),
              "unassigned_prior_errors_true_rank_below_3_corrected": int((hard & after).sum()),
              "folds": fold_reports}
    return result, delta


def _secondary(folds, clusters, probabilities):
    evidence = np.empty_like(probabilities)
    residuals = []
    for fold in np.unique(folds):
        mask = folds == fold
        evidence[mask], residual = reference.evidence(probabilities[mask], uniform=True)
        residuals.append(residual)
    buckets = assignment.buckets(clusters)
    require(max(map(len, buckets)) <= assignment.CAP
            and all(len(np.unique(folds[rows])) == 1 for rows in buckets), "stage14_assignment_cluster_scope")
    covered = np.zeros(len(folds), dtype=bool)
    off = evidence.argmax(axis=1)
    on = off.copy()
    for rows in assignment.eligible(buckets):
        r, c = assignment.linear_sum_assignment(-evidence[rows])
        require(np.array_equal(r, np.arange(len(rows))) and len(np.unique(c)) == len(rows)
                and not covered[rows].any(), "stage14_assignment_unique_coverage")
        on[rows[r]], covered[rows] = c, True
    require(np.array_equal(on[~covered], off[~covered]), "stage14_assignment_outside_group")
    return evidence, on, covered, residuals


def evaluate(texts, prompts, y, folds, classes, control_probs, ce_logits, contrast_logits):
    """Evaluate frozen arms; all arrays must share the canonical official row axis."""
    texts, prompts, y, folds, classes = map(np.asarray, (texts, prompts, y, folds, classes))
    require(folds.ndim == 1 and folds.dtype.kind in "iu" and
            np.unique(folds).tolist() in (list(range(5)), list(range(1, 6))), "stage14_fold_axis")
    # The old geometry helper expects strictly positive fold labels.
    folds = folds.astype(np.int64, copy=False)
    if folds.min() == 0:
        folds = folds + 1
    require(classes.shape == (K,) and classes.dtype.kind == "U"
            and reference.digest_json(classes.tolist()) == reference.LABEL_SHA, "stage14_class_axis")
    assignment.validate_truth(folds, prompts, y)
    y = y.astype(np.int64, copy=False)
    control = np.asarray(control_probs)
    require(control.shape == (len(y), K), "stage14_control_axis")
    reference.validate_p(control)
    ce, contrast = np.asarray(ce_logits), np.asarray(contrast_logits)
    probs = (control, _blend(control, ce), _blend(control, contrast))
    # Geometry is fitted independently in each scoring fold with texts only.
    # The frozen helper consults prompts/labels afterwards solely for diagnostics.
    geometry, clusters = assignment.geometry(texts, folds, prompts, y)
    plan, support = _bootstrap_plan(y, prompts, folds)
    secondary = [_secondary(folds, clusters, p) for p in probs]
    assigned = secondary[0][2]
    require(int(assigned.sum()) == geometry["eligible_rows"]
            and all(np.array_equal(assigned, s[2]) for s in secondary), "stage14_fixed_geometry")
    primary_p = [p.argmax(axis=1) for p in probs]
    assigned_p = [s[1] for s in secondary]
    comparisons = (("ce_vs_control", 0, 1), ("contrast_vs_control", 0, 2), ("contrast_vs_ce", 1, 2))
    primary_pairs, secondary_pairs, delta = {}, {}, {}
    for name, i, j in comparisons:
        primary_pairs[name], delta[name] = _paired(y, primary_p[i], primary_p[j], plan, support, assigned, probs[i])
        secondary_pairs[name], _ = _paired(y, assigned_p[i], assigned_p[j], plan, support, assigned, secondary[i][0])
    cp, ep = primary_pairs["contrast_vs_control"], primary_pairs["contrast_vs_ce"]
    gates = {"at_least_1_pp_vs_control": delta["contrast_vs_control"] >= MIN_CONTROL_GAIN_PP,
             "control_gain_paired_lower_ci_above_0": cp["conditional_paired_ci95_pp"][0] > 0,
             "gain_vs_ce_positive": delta["contrast_vs_ce"] > 0,
             "ce_gain_paired_lower_ci_above_0": ep["conditional_paired_ci95_pp"][0] > 0,
             "no_class_loses_over_5_pp_vs_control": cp["class_regression_cap_pass"],
             "no_class_loses_over_5_pp_vs_ce": ep["class_regression_cap_pass"]}
    return {"class_order": classes.tolist(), "class_supports": np.bincount(y, minlength=K).tolist(),
            "head_geometric_weight": HEAD_WEIGHT, "group_geometry": geometry,
            "primary": {"scope": "row_independent", "arms": {n: _class_state(y, p) for n, p in
                         zip(ARM_NAMES, primary_p, strict=True)}, "paired": primary_pairs},
            "standalone_raw_heads": {"ce": _class_state(y, ce.argmax(axis=1)),
                                     "contrast": _class_state(y, contrast.argmax(axis=1))},
            "secondary": {"scope": "same_fold_local_sinkhorn_and_frozen_stage13f_text_assignment",
                          "arms": {n: {**_class_state(y, s[1]), "sinkhorn_column_mass_residual_by_fold": s[3]}
                                   for n, s in zip(ARM_NAMES, secondary, strict=True)}, "paired": secondary_pairs},
            "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED,
                          "scope": "paired true-prompt resampling within fixed folds on fixed predictions",
                          "limitation": "does not refit or regroup; inferred groups may couple prompts"},
            "promotion": {"exploratory_only": True, "passes_predeclared_primary_gate": all(gates.values()),
                          "checks": gates}, "raw_rows_or_predictions_exported": False}
