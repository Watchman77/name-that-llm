"""Audit and test a fixed 10–11 row extension of the retained S07 rule."""
from __future__ import annotations

from collections import Counter
from fractions import Fraction
import hashlib

import numpy as np
from scipy.optimize import linear_sum_assignment

import stage13b_balancing_stress as reference
import stage13f_assignment_core as prior


EXTENSION_SIZES = (10, 11)
K = reference.K
require = reference.require


def _validated_buckets(cluster_ids, folds, groups, targets):
    prior.validate_folds(folds)
    prior.validate_truth(folds, groups, targets)
    require(cluster_ids.shape == folds.shape and cluster_ids.dtype.kind in "iu"
            and (cluster_ids >= 0).all(), "stage13g_cluster_axis")
    inferred = prior.buckets(cluster_ids)
    require(max(map(len, inferred)) <= prior.CAP
            and all(np.unique(folds[idx]).size == 1 for idx in inferred),
            "stage13g_cluster_boundaries")
    return inferred


def _statistics(buckets, groups, targets):
    rows = sum(map(len, buckets))
    prompt_counts = [Counter(groups[idx]) for idx in buckets]
    majority_rows = sum(max(c.values()) for c in prompt_counts)
    duplicate_excess = [len(idx) - len(np.unique(targets[idx])) for idx in buckets]
    return dict(clusters=len(buckets), rows=rows,
                mixed_prompt_clusters=sum(len(c) > 1 for c in prompt_counts),
                duplicate_true_label_clusters=sum(n > 0 for n in duplicate_excess),
                duplicate_true_label_excess_rows=sum(duplicate_excess),
                majority_prompt_rows=majority_rows,
                row_weighted_prompt_purity=majority_rows / rows if rows else None)


def _smaller_statistics(inferred, groups, targets):
    by_size = {str(n): _statistics([idx for idx in inferred if len(idx) == n], groups, targets)
               for n in range(1, prior.MIN_GROUP)}
    extension = _statistics([idx for idx in inferred if len(idx) in EXTENSION_SIZES], groups, targets)
    prior_untouched_rows = sum(stats["rows"] for stats in by_size.values())
    require(prior_untouched_rows + sum(map(len, prior.eligible(inferred))) == sum(map(len, inferred)),
            "stage13g_untouched_row_partition")
    require(all(extension[key] == sum(by_size[str(n)][key] for n in EXTENSION_SIZES)
                for key in ("clusters", "rows", "mixed_prompt_clusters", "duplicate_true_label_clusters",
                            "duplicate_true_label_excess_rows", "majority_prompt_rows")),
            "stage13g_extension_size_reconciliation")
    return dict(untouched_by_size=by_size, prior_untouched_rows=prior_untouched_rows, extension=extension)


def extension_geometry(cluster_ids, folds, groups, targets):
    """Use truth only to audit groups fixed by the unchanged text algorithm."""
    inferred = _validated_buckets(cluster_ids, folds, groups, targets)
    summary = _smaller_statistics(inferred, groups, targets)
    per_fold = []
    for fold in np.unique(folds):
        subset = [idx for idx in inferred if folds[idx[0]] == fold]
        per_fold.append(dict(fold=int(fold), **_smaller_statistics(subset, groups, targets)))
    extension = summary["extension"]
    require(all(extension[key] == sum(f["extension"][key] for f in per_fold)
                for key in ("clusters", "rows", "mixed_prompt_clusters", "duplicate_true_label_clusters",
                            "duplicate_true_label_excess_rows", "majority_prompt_rows"))
            and summary["prior_untouched_rows"] == sum(f["prior_untouched_rows"] for f in per_fold),
            "stage13g_fold_geometry_reconciliation")
    return dict(**summary, folds=per_fold, extension_sizes=list(EXTENSION_SIZES),
                passes_extension_geometry_gate=bool(extension["clusters"]
                    and not extension["mixed_prompt_clusters"] and not extension["duplicate_true_label_clusters"]))


def geometry(texts, folds, groups, targets):
    """Reuse the exact Stage 13F geometry and audit all previously unassigned sizes."""
    base, cluster_ids = prior.geometry(texts, folds, groups, targets)
    return dict(base=base, **extension_geometry(cluster_ids, folds, groups, targets)), cluster_ids


def _assigned(evidence, decisions, eligible):
    assigned = decisions.copy()
    covered = np.zeros(len(decisions), dtype=bool)
    for idx in eligible:
        # Exact source objective and row order; no label-based filtering.
        r, c = linear_sum_assignment(-evidence[idx])
        require(np.array_equal(r, np.arange(len(idx))) and len(np.unique(c)) == len(idx)
                and not covered[idx].any(), "stage13g_assignment_unique_coverage")
        assigned[idx[r]], covered[idx] = c, True
    return assigned, covered


def compare(balanced_log_evidence, cluster_ids, folds, groups, targets, *, expected_baseline_confusion_sha256=None):
    """Compare exact 12–18 assignment to adding only the fixed 10–11 groups."""
    inferred = _validated_buckets(cluster_ids, folds, groups, targets)
    require(balanced_log_evidence.shape == (len(folds), K)
            and balanced_log_evidence.dtype == np.float64 and np.isfinite(balanced_log_evidence).all(),
            "stage13g_log_evidence_axis")
    require(np.max(np.abs(np.exp(balanced_log_evidence).sum(axis=1) - (1 + K * 1e-12))) <= 2e-12,
            "stage13g_log_evidence_normalization")
    dense = prior.eligible(inferred)
    small = [idx for idx in inferred if len(idx) in EXTENSION_SIZES]
    g = _statistics(small, groups, targets)
    require(dense and small and not g["mixed_prompt_clusters"] and not g["duplicate_true_label_clusters"],
            "stage13g_extension_geometry_gate")
    sha = lambda: hashlib.sha256(balanced_log_evidence.astype("<f8", copy=False).tobytes()).hexdigest()
    frozen = sha()
    baseline, base_coverage = _assigned(balanced_log_evidence, balanced_log_evidence.argmax(axis=1), dense)
    y = targets.astype(np.int64, copy=False)
    cm0 = reference.confusion(y, baseline)
    baseline_sha = reference.digest_json(cm0.tolist())
    require(expected_baseline_confusion_sha256 is None or baseline_sha == expected_baseline_confusion_sha256,
            "stage13g_stage13f_on_confusion_replay")
    candidate, extended_coverage = _assigned(balanced_log_evidence, baseline, small)
    require(not np.any(base_coverage & extended_coverage)
            and np.array_equal(baseline[~extended_coverage], candidate[~extended_coverage])
            and sha() == frozen, "stage13g_changed_fixed_evidence_or_outside_extension")
    cm1 = reference.confusion(y, candidate)
    supports = cm0.sum(axis=1)
    recall0, recall1 = reference.recalls(cm0), reference.recalls(cm1)
    deltas = [Fraction(int(cm1[i, i]) - int(cm0[i, i]), int(supports[i])) * 100 for i in range(K)]
    delta = sum(deltas) / K
    fold_reports = []
    rng = np.random.default_rng(reference.SEED)
    support_draws = np.zeros((reference.DRAWS, K), dtype=np.int64)
    diff_draws = np.zeros_like(support_draws)
    for fold in np.unique(folds):
        mask = folds == fold
        yt, p, q = y[mask], baseline[mask], candidate[mask]
        a, b = reference.confusion(yt, p), reference.confusion(yt, q)
        b0, b1 = float(reference.recalls(a).mean()), float(reference.recalls(b).mean())
        gp, idx = np.unique(groups[mask], return_inverse=True)
        prompt_support = np.zeros((len(gp), K), dtype=np.int64)
        prompt_diff = np.zeros_like(prompt_support)
        np.add.at(prompt_support, (idx, yt), 1)
        np.add.at(prompt_diff, (idx, yt), (q == yt).astype(np.int64) - (p == yt).astype(np.int64))
        weights = rng.multinomial(len(gp), np.full(len(gp), 1 / len(gp)), size=reference.DRAWS)
        support_draws += weights @ prompt_support
        diff_draws += weights @ prompt_diff
        fold_reports.append(dict(fold=int(fold), rows=int(mask.sum()),
                                 extension_rows=int((extended_coverage & mask).sum()),
                                 baseline_ba=b0, candidate_ba=b1, delta_pp=100 * (b1 - b0),
                                 **prior.paired_counts(yt, p, q)))
    counts = prior.paired_counts(y, baseline, candidate)
    require((support_draws > 0).all() and np.array_equal(supports, cm1.sum(axis=1))
            and all(counts[key] == sum(f[key] for f in fold_reports) for key in counts)
            and counts["changed"] <= g["rows"] == int(extended_coverage.sum())
            and sum(f["extension_rows"] for f in fold_reports) == g["rows"], "stage13g_paired_reconciliation")
    ci = np.percentile((diff_draws / support_draws).mean(axis=1) * 100, [2.5, 97.5]).tolist()
    return dict(baseline_ba=float(recall0.mean()), candidate_ba=float(recall1.mean()),
                delta_pp=float(delta), delta_pp_exact=str(delta), conditional_delta_ci95_pp=ci,
                class_supports=supports.tolist(), class_delta_pp=[float(d) for d in deltas],
                baseline_confusion_counts=cm0.tolist(), candidate_confusion_counts=cm1.tolist(),
                baseline_confusion_sha256=baseline_sha,
                candidate_confusion_sha256=reference.digest_json(cm1.tolist()),
                baseline_rows=int(base_coverage.sum()), baseline_clusters=len(dense),
                extension_rows=g["rows"], extension_clusters=g["clusters"], folds=fold_reports,
                outside_extension_unchanged=True, fixed_log_evidence_unchanged=True,
                bootstrap=dict(draws=reference.DRAWS, seed=reference.SEED,
                               scope="conditional true-prompt resampling on fixed scores, clusters and decisions"),
                **counts)
