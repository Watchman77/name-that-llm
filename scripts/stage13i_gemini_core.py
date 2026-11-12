"""Test one fold-local four-way Gemini readout behind a fixed family gate."""
from __future__ import annotations

from fractions import Fraction
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import stage13b_balancing_stress as reference
import stage13c_deberta_complement as deberta
import stage13f_assignment_core as assignment
import stage13h_residual_core as residual


GEMINI_CLASSES = ("gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash")
K = reference.K
require = reference.require


def eligible_gate(covered, baseline, gemini):
    """Only the existing assignment coverage and baseline prediction gate scoring."""
    return ~covered & np.isin(baseline, gemini)


def _fit_and_predict(train_x, train_y, score_x, gemini):
    require(np.array_equal(np.unique(train_y), gemini), "stage13i_specialist_fit_class_axis")
    scaler = StandardScaler().fit(train_x)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model = LogisticRegression(**deberta.HEAD).fit(scaler.transform(train_x), train_y)
    require(np.array_equal(model.classes_, gemini), "stage13i_specialist_model_class_axis")
    require(len(model.n_iter_) == 1 and 0 < model.n_iter_[0] < deberta.HEAD["max_iter"],
            "stage13i_specialist_iteration_limit")
    if len(score_x):
        probabilities = model.predict_proba(scaler.transform(score_x))
        require(probabilities.shape == (len(score_x), len(gemini))
                and np.isfinite(probabilities).all() and ((probabilities >= 0) & (probabilities <= 1)).all()
                and np.allclose(probabilities.sum(axis=1), 1, atol=1e-12, rtol=0), "stage13i_specialist_probabilities")
        prediction = gemini[probabilities.argmax(axis=1)]
    else:
        prediction = np.empty(0, dtype=np.int64)
    return prediction, int(model.n_iter_[0])


def _conditional_draws(prompts, y, baseline, candidate, rng):
    unique, prompt_axis = np.unique(prompts, return_inverse=True)
    n = len(unique)
    support = np.zeros((n, K), dtype=np.int64)
    difference = np.zeros_like(support)
    np.add.at(support, (prompt_axis, y), 1)
    np.add.at(difference, (prompt_axis, y), (candidate == y).astype(np.int64) - (baseline == y).astype(np.int64))
    weights = rng.multinomial(n, np.full(n, 1 / n), size=reference.DRAWS)
    return weights @ support, weights @ difference


def _family_matrix(y, predictions, gemini):
    return np.bincount(2 * np.isin(y, gemini) + np.isin(predictions, gemini), minlength=4).reshape(2, 2)


def compare(features, evidence, cluster_ids, folds, prompts, y, classes, *, expected_baseline_confusion_sha256):
    """Compare one fixed specialist with an exactly replayed Stage 13F control."""
    labels = np.asarray(classes)
    require(labels.shape == (K,) and labels.dtype.kind == "U"
            and reference.digest_json(labels.tolist()) == reference.LABEL_SHA, "stage13i_class_axis")
    gemini = np.flatnonzero(np.isin(labels, GEMINI_CLASSES))
    require(tuple(labels[gemini]) == GEMINI_CLASSES, "stage13i_gemini_class_axis")
    require(features.shape == (len(folds), 2048) and features.dtype == np.float64
            and np.isfinite(features).all(), "stage13i_feature_axis")
    score_sha = residual._digest(evidence)
    feature_sha = residual._digest(features)
    _, baseline, covered, _, _, cm0, digest = residual._replay(
        evidence, cluster_ids, folds, prompts, y, expected_baseline_confusion_sha256)
    eligible = eligible_gate(covered, baseline, gemini)
    require(eligible.any() and not np.any(eligible & covered), "stage13i_no_valid_gate_rows")
    candidate = baseline.copy()
    y = y.astype(np.int64, copy=False)
    true_gemini = np.isin(y, gemini)
    rows_scored = np.zeros(len(y), dtype=bool)
    fold_reports = []
    rng = np.random.default_rng(reference.SEED)
    support_draws = np.zeros((reference.DRAWS, K), dtype=np.int64)
    difference_draws = np.zeros_like(support_draws)
    with threadpool_limits(limits=deberta.THREADS):
        for fold in np.unique(folds):
            hold = folds == fold
            train = ~hold & true_gemini
            score = hold & eligible
            fit_prompts = np.unique(prompts[train])
            held_prompts = np.unique(prompts[hold])
            require(np.intersect1d(fit_prompts, held_prompts).size == 0
                    and not rows_scored[score].any(), "stage13i_fold_isolation")
            predictions, iterations = _fit_and_predict(features[train], y[train], features[score], gemini)
            require(len(predictions) == int(score.sum()) and np.isin(predictions, gemini).all(),
                    "stage13i_specialist_gate_predictions")
            candidate[score], rows_scored[score] = predictions, True
            yt, p, q = y[hold], baseline[hold], candidate[hold]
            fcm0, fcm1 = reference.confusion(yt, p), reference.confusion(yt, q)
            b0, b1 = float(reference.recalls(fcm0).mean()), float(reference.recalls(fcm1).mean())
            fold_reports.append(dict(fold=int(fold), rows=int(hold.sum()), eligible_rows=int(score.sum()),
                                     fit_rows=int(train.sum()), fit_class_counts=[int((y[train] == i).sum()) for i in gemini],
                                     fit_prompts=len(fit_prompts), heldout_prompts=len(held_prompts), prompt_overlap=0,
                                     specialist_iterations=iterations, baseline_ba=b0, candidate_ba=b1,
                                     delta_pp=100 * (b1 - b0), **assignment.paired_counts(yt, p, q)))
            support, difference = _conditional_draws(prompts[hold], yt, p, q, rng)
            support_draws += support
            difference_draws += difference

    require(np.array_equal(rows_scored, eligible) and np.array_equal(candidate[~eligible], baseline[~eligible])
            and np.array_equal(np.isin(candidate, gemini), np.isin(baseline, gemini))
            and residual._digest(evidence) == score_sha and residual._digest(features) == feature_sha,
            "stage13i_frozen_scores_or_routing_changed")
    cm1 = reference.confusion(y, candidate)
    support = cm0.sum(axis=1)
    base_recall, cand_recall = reference.recalls(cm0), reference.recalls(cm1)
    deltas = [Fraction(int(cm1[i, i]) - int(cm0[i, i]), int(support[i])) * 100 for i in range(K)]
    delta = sum(deltas) / K
    counts = assignment.paired_counts(y, baseline, candidate)
    base_family, cand_family = _family_matrix(y, baseline, gemini), _family_matrix(y, candidate, gemini)
    ecm0, ecm1 = reference.confusion(y[eligible], baseline[eligible]), reference.confusion(y[eligible], candidate[eligible])
    n_eligible = int(eligible.sum())
    require((support_draws > 0).all() and np.array_equal(support, cm1.sum(axis=1))
            and np.array_equal(base_family, cand_family) and np.array_equal(cm1 - cm0, ecm1 - ecm0)
            and int(ecm0.sum()) == int(ecm1.sum()) == n_eligible == sum(f["eligible_rows"] for f in fold_reports)
            and counts["changed"] <= n_eligible and all(counts[k] == sum(f[k] for f in fold_reports) for k in counts),
            "stage13i_paired_reconciliation")
    interval = np.percentile((difference_draws / support_draws).mean(axis=1) * 100, [2.5, 97.5]).tolist()
    criteria = dict(pooled_gain_at_least_half_pp=delta >= Fraction(1, 2), conditional_ci_lower_positive=interval[0] > 0,
                    every_class_recall_loss_le_five_pp=min(deltas) >= -5)
    true_eligible = int((eligible & true_gemini).sum())
    correct_base, correct_candidate = int(ecm0.trace()), int(ecm1.trace())
    return dict(baseline_ba=float(base_recall.mean()), candidate_ba=float(cand_recall.mean()), delta_pp=float(delta),
                delta_pp_exact=str(delta), conditional_delta_ci95_pp=interval, class_supports=support.tolist(),
                class_delta_pp=[float(d) for d in deltas], baseline_class_recall=base_recall.tolist(),
                candidate_class_recall=cand_recall.tolist(), baseline_confusion_counts=cm0.tolist(),
                candidate_confusion_counts=cm1.tolist(), baseline_confusion_sha256=digest,
                candidate_confusion_sha256=reference.digest_json(cm1.tolist()),
                eligible_rows=n_eligible, eligible_true_gemini_rows=true_eligible,
                eligible_true_non_gemini_rows=n_eligible - true_eligible, eligible_baseline_errors=n_eligible - correct_base,
                eligible_within_family_baseline_errors=true_eligible - correct_base,
                eligible_baseline_correct=correct_base, eligible_candidate_correct=correct_candidate,
                eligible_baseline_confusion_counts=ecm0.tolist(), eligible_candidate_confusion_counts=ecm1.tolist(),
                gemini_classes=list(GEMINI_CLASSES), family_confusion_axis=["not_gemini", "gemini"],
                baseline_family_confusion_counts=base_family.tolist(), candidate_family_confusion_counts=cand_family.tolist(),
                folds=fold_reports, gate_uses_only_inferred_assignment_and_baseline_prediction=True,
                outside_eligible_unchanged=True, predicted_family_unchanged=True,
                fixed_features_and_log_evidence_unchanged=True, row_level_outputs=False,
                specialist=dict(scaler="StandardScaler", estimator="multinomial logistic regression", parameters=deberta.HEAD.copy(),
                                layers=[24, 6], fit_scope="true Gemini training rows only, outside the scoring prompt fold",
                                application="replace variant by four-class argmax on gate rows only"),
                bootstrap=dict(draws=reference.DRAWS, seed=reference.SEED,
                               scope="conditional paired true-prompt resampling on fixed out-of-fold predictions"),
                criteria=criteria, passes_development_screen=all(criteria.values()), **counts)
