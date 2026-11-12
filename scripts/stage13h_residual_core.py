"""Describe the frozen Stage 13F decisions without changing any prediction."""
from __future__ import annotations

import hashlib

import numpy as np

import stage13b_balancing_stress as reference
import stage13f_assignment_core as assignment


K = reference.K
MARGIN_CUTS = (0.10, 0.25, 0.50, 1.0, 2.0)
PAIR_LIMIT = 15
require = reference.require


def _digest(evidence):
    return hashlib.sha256(evidence.astype("<f8", copy=False).tobytes()).hexdigest()


def _replay(evidence, cluster_ids, folds, prompts, targets, expected_digest):
    assignment.validate_folds(folds)
    assignment.validate_truth(folds, prompts, targets)
    require(cluster_ids.shape == folds.shape and cluster_ids.dtype.kind in "iu"
            and (cluster_ids >= 0).all(), "stage13h_cluster_axis")
    require(evidence.shape == (len(folds), K) and evidence.dtype == np.float64
            and np.isfinite(evidence).all(), "stage13h_log_evidence_axis")
    require(np.max(np.abs(np.exp(evidence).sum(axis=1) - (1 + K * 1e-12))) <= 2e-12,
            "stage13h_log_evidence_normalization")
    inferred = assignment.buckets(cluster_ids)
    require(max(map(len, inferred)) <= assignment.CAP
            and all(np.unique(folds[idx]).size == 1 for idx in inferred),
            "stage13h_cluster_boundaries")
    dense = assignment.eligible(inferred)
    require(dense, "stage13h_no_assignment_eligible_groups")
    off = evidence.argmax(axis=1)
    on = off.copy()
    covered = np.zeros(len(folds), dtype=bool)
    sizes = np.zeros(len(folds), dtype=np.int64)
    for idx in inferred:
        sizes[idx] = len(idx)
    # Same solver, objective, group membership and original row order as Stage 13F.
    for idx in dense:
        r, c = assignment.linear_sum_assignment(-evidence[idx])
        require(np.array_equal(r, np.arange(len(idx))) and len(np.unique(c)) == len(idx)
                and not covered[idx].any(), "stage13h_assignment_unique_coverage")
        on[idx[r]], covered[idx] = c, True
    require(np.array_equal(off[~covered], on[~covered]) and (sizes > 0).all()
            and np.array_equal(covered, sizes >= assignment.MIN_GROUP), "stage13h_assignment_partition")
    matrix = reference.confusion(targets.astype(np.int64, copy=False), on)
    digest = reference.digest_json(matrix.tolist())
    require(digest == expected_digest, "stage13h_stage13f_on_confusion_replay")
    return off, on, covered, sizes, inferred, matrix, digest


def _scores(matrix):
    support = matrix.sum(axis=1)
    correct = np.diag(matrix)
    represented = support > 0
    recalls = [int(c) / int(n) if n else None for c, n in zip(correct, support, strict=True)]
    rows, n_correct = int(support.sum()), int(correct.sum())
    return dict(rows=rows, correct=n_correct, errors=rows - n_correct,
                row_accuracy=n_correct / rows if rows else None,
                balanced_accuracy=float(np.mean(correct[represented] / support[represented]))
                if represented.any() else None,
                represented_classes=int(represented.sum()), class_supports=support.tolist(),
                class_correct=correct.tolist(), class_recall=recalls)


def _pair_counts(matrix, y, decision, folds):
    indices = [(i, j) for i in range(K) for j in range(K) if i != j and matrix[i, j]]
    indices.sort(key=lambda ij: (-int(matrix[ij]), *ij))
    out = []
    for i, j in indices[:PAIR_LIMIT]:
        selected = (y == i) & (decision == j)
        n, supported = int(matrix[i, j]), int(matrix[i].sum())
        out.append(dict(true_class_index=i, predicted_class_index=j, errors=n,
                        true_class_support=supported, true_class_error_rate=n / supported,
                        folds_with_error=int(np.unique(folds[selected]).size)))
    return out


def _stratum(y, off, on, ranks, margin_bins, folds, *, detail):
    cm = reference.confusion(y, on)
    post = _scores(cm)
    pre = _scores(reference.confusion(y, off))
    misses = on != y
    report = dict(rows=post["rows"], correct=post["correct"], errors=post["errors"],
                  row_accuracy=post["row_accuracy"], balanced_accuracy=post["balanced_accuracy"],
                  represented_classes=post["represented_classes"],
                  pre_assignment_correct=pre["correct"], pre_assignment_errors=pre["errors"],
                  pre_assignment_row_accuracy=pre["row_accuracy"],
                  pre_assignment_balanced_accuracy=pre["balanced_accuracy"],
                  evidence_true_top2_rows=int((ranks <= 2).sum()),
                  evidence_true_top3_rows=int((ranks <= 3).sum()),
                  errors_with_true_top2=int((misses & (ranks <= 2)).sum()),
                  errors_with_true_top3=int((misses & (ranks <= 3)).sum()))
    if detail:
        margin = []
        edges = (0.0, *MARGIN_CUTS)
        for idx, lower in enumerate(edges):
            in_bin = margin_bins == idx
            n, error = int(in_bin.sum()), int((in_bin & misses).sum())
            margin.append(dict(lower_inclusive=lower, upper_exclusive=edges[idx + 1] if idx < len(edges) - 1 else None,
                               rows=n, correct=n - error, errors=error,
                               errors_with_true_top2=int((in_bin & misses & (ranks <= 2)).sum()),
                               errors_with_true_top3=int((in_bin & misses & (ranks <= 3)).sum())))
        report.update(class_supports=post["class_supports"], class_correct=post["class_correct"],
                      class_recall=post["class_recall"], confusion_counts=cm.tolist(),
                      top_directed_error_pairs=_pair_counts(cm, y, on, folds),
                      evidence_rank_counts=np.bincount(ranks, minlength=K + 1)[1:].tolist(),
                      error_evidence_rank_counts=np.bincount(ranks[misses], minlength=K + 1)[1:].tolist(),
                      evidence_margin_bins=margin)
        require(sum(b["rows"] for b in margin) == post["rows"]
                and sum(b["errors"] for b in margin) == post["errors"]
                and sum(report["evidence_rank_counts"]) == post["rows"]
                and sum(report["error_evidence_rank_counts"]) == post["errors"],
                "stage13h_margin_or_rank_reconciliation")
    return report


def diagnose(evidence, cluster_ids, folds, prompts, y, *, expected_baseline_confusion_sha256):
    """Return only aggregate diagnostics after exact Stage 13F-on confusion replay."""
    snapshot = _digest(evidence)
    off, on, covered, sizes, inferred, baseline, digest = _replay(
        evidence, cluster_ids, folds, prompts, y, expected_baseline_confusion_sha256)
    y = y.astype(np.int64, copy=False)
    rank_order = np.argsort(-evidence, axis=1, kind="stable")
    require(np.array_equal(rank_order[:, 0], off), "stage13h_rank_tie_order")
    ranks = (rank_order == y[:, None]).argmax(axis=1) + 1
    top = evidence[np.arange(len(folds))[:, None], rank_order[:, :2]]
    gap = top[:, 0] - top[:, 1]
    require((gap >= 0).all() and np.isfinite(gap).all(), "stage13h_margin_domain")
    margin_bins = np.searchsorted(MARGIN_CUTS, gap, side="right")

    def summary(mask, *, detail=False):
        return _stratum(y[mask], off[mask], on[mask], ranks[mask], margin_bins[mask], folds[mask], detail=detail)

    assigned, unassigned = summary(covered, detail=True), summary(~covered, detail=True)
    by_fold = [dict(fold=int(fold), assigned=summary((folds == fold) & covered),
                    unassigned=summary((folds == fold) & ~covered)) for fold in np.unique(folds)]
    by_size = []
    for size in np.unique(sizes):
        mask = sizes == size
        part = summary(mask)
        n_groups = sum(len(idx) == int(size) for idx in inferred)
        require(n_groups * int(size) == part["rows"], "stage13h_size_reconciliation")
        by_size.append(dict(cluster_size=int(size), clusters=n_groups,
                            assignment_applied=bool(size >= assignment.MIN_GROUP), **part))
    for key in ("rows", "correct", "errors", "pre_assignment_correct", "errors_with_true_top2", "errors_with_true_top3"):
        require(all(sum(row[name][key] for row in by_fold) == part[key]
                    and sum(row[key] for row in by_size if row["assignment_applied"] == (name == "assigned")) == part[key]
                    for name, part in (("assigned", assigned), ("unassigned", unassigned))),
                "stage13h_partition_reconciliation")
    require(np.array_equal(np.asarray(assigned["confusion_counts"]) + np.asarray(unassigned["confusion_counts"]), baseline)
            and unassigned["correct"] == unassigned["pre_assignment_correct"]
            and unassigned["error_evidence_rank_counts"][0] == 0
            and _digest(evidence) == snapshot, "stage13h_baseline_reconciliation")
    base = _scores(baseline)
    return dict(baseline_confusion_sha256=digest, baseline_confusion_counts=baseline.tolist(),
                baseline_balanced_accuracy=base["balanced_accuracy"], baseline_correct=base["correct"],
                assigned=assigned, unassigned=unassigned, by_fold=by_fold, by_cluster_size=by_size,
                evidence_margin_definition="Highest minus second-highest frozen balanced natural log score, before assignment. Null upper bound means infinity.",
                rank_definition="True model rank in frozen balanced scores before assignment, with ties by original class order. Rank count vectors are 1-based.",
                balanced_accuracy_definition="Mean recall over classes represented in the stratum; absent class recall is null.",
                fixed_log_evidence_unchanged=True, predictions_outside_assignment_unchanged=True,
                row_level_outputs=False)
