"""Isolate S07's dense-group assignment on a held-out development proxy."""
from __future__ import annotations

from collections import Counter
from fractions import Fraction
import hashlib

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import TfidfVectorizer

import stage13b_balancing_stress as reference


K = reference.K
THRESHOLD, CAP, MIN_GROUP = 0.15, 18, 12
TFIDF = dict(sublinear_tf=True, stop_words="english", min_df=2, max_df=0.05)
require = reference.require


def capped_agglo(S, cap=18, thr=0.02):
    """S07's float32 average linkage, including its row-order tie policy."""
    n = len(S); A = S.astype(np.float32).copy(); np.fill_diagonal(A, -np.inf)
    size = np.ones(n, np.int32); alive = np.ones(n, bool); lab = np.arange(n)
    while True:
        ok = (size[:, None] + size[None, :] <= cap) & alive[:, None] & alive[None, :]
        M = np.where(ok, A, -np.inf); k = int(np.argmax(M)); i, j = divmod(k, n)
        if not np.isfinite(M[i, j]) or M[i, j] < thr: break
        new = (size[i] * A[i] + size[j] * A[j]) / (size[i] + size[j])
        A[i] = new; A[:, i] = new; A[i, i] = -np.inf; A[j] = -np.inf; A[:, j] = -np.inf
        size[i] += size[j]; alive[j] = False; lab[lab == j] = i
    return lab


def infer_groups(texts):
    """One scoring batch only; reproduce the source's singleton fallback."""
    try:
        features = TfidfVectorizer(**TFIDF).fit_transform(texts)
        clusters = capped_agglo((features @ features.T).toarray(), CAP, THRESHOLD)
    except ValueError:
        return np.arange(len(texts), dtype=np.int64), True
    return clusters, False


def validate_folds(folds):
    require(folds.ndim == 1 and len(folds) > 0 and folds.dtype.kind in "iu"
            and (folds > 0).all(), "stage13f_fold_axis")


def validate_truth(folds, groups, targets):
    require(groups.shape == targets.shape == folds.shape and groups.dtype.kind == "U"
            and np.char.str_len(groups).min() > 0 and targets.dtype.kind in "iu"
            and ((targets >= 0) & (targets < K)).all(), "stage13f_truth_axes")
    assigned = {}
    for prompt, fold in zip(groups, folds, strict=True):
        require(assigned.setdefault(prompt, fold) == fold, "stage13f_prompt_crosses_folds")


def buckets(clusters):
    mapping = {}
    for index, label in enumerate(clusters):
        mapping.setdefault(int(label), []).append(index)
    return [np.asarray(mapping[label], dtype=np.int64) for label in sorted(mapping)]


def eligible(cluster_buckets):
    return [rows for rows in cluster_buckets if MIN_GROUP <= len(rows) <= CAP]


def group_statistics(cluster_buckets, groups, targets):
    selected = eligible(cluster_buckets)
    rows = sum(map(len, selected))
    prompt_counters = [Counter(groups[idx]) for idx in selected]
    majority_rows = sum(max(c.values()) for c in prompt_counters)
    duplicate_counts = [len(idx) - len(np.unique(targets[idx])) for idx in selected]
    all_sizes = Counter(map(len, cluster_buckets))
    require(not all_sizes or max(all_sizes) <= CAP, "stage13f_cluster_size_cap")
    return dict(inferred_clusters=len(cluster_buckets),
                cluster_size_histogram={str(s): all_sizes[s] for s in sorted(all_sizes)},
                eligible_clusters=len(selected), eligible_rows=rows,
                eligible_mixed_prompt_clusters=sum(len(c) > 1 for c in prompt_counters),
                eligible_clusters_with_duplicate_true_label=sum(d > 0 for d in duplicate_counts),
                eligible_duplicate_true_label_excess_rows=sum(duplicate_counts),
                eligible_majority_prompt_rows=majority_rows,
                eligible_row_weighted_prompt_purity=(majority_rows / rows if rows else None))


def geometry(texts, folds, groups, targets):
    """Cluster text independently in each fold; only then inspect truth."""
    validate_folds(folds)
    require(texts.shape == folds.shape and texts.dtype.kind == "U", "stage13f_text_axis")
    cluster_ids = np.full(len(folds), -1, dtype=np.int64)
    entries = []
    offset = 0
    for fold in np.unique(folds):
        index = np.flatnonzero(folds == fold)
        local, fallback = infer_groups(texts[index])
        require(local.shape == index.shape and local.dtype.kind in "iu"
                and ((0 <= local) & (local < len(index))).all(), "stage13f_inferred_axis")
        cluster_ids[index] = local + offset
        offset += len(index)
        entries.append((int(fold), index, local, fallback))
    require(offset == len(folds) and (cluster_ids >= 0).all(), "stage13f_cluster_coverage")
    validate_truth(folds, groups, targets)
    reports = [dict(fold=fold, rows=len(index), true_prompts=len(np.unique(groups[index])),
                    singleton_fallback=fallback, **group_statistics(buckets(local), groups[index], targets[index]))
               for fold, index, local, fallback in entries]
    pooled = group_statistics(buckets(cluster_ids), groups, targets)
    require(all(pooled[key] == sum(r[key] for r in reports) for key in
                ("inferred_clusters", "eligible_clusters", "eligible_rows", "eligible_majority_prompt_rows")),
            "stage13f_cluster_pool_reconciliation")
    return (dict(**pooled, folds=reports, tfidf=TFIDF.copy(), cluster_cap=CAP,
                 average_cosine_cut=THRESHOLD, assignment_size_min=MIN_GROUP,
                 assignment_size_max=CAP, grouping_uses_only_batch_text=True), cluster_ids)


def paired_counts(targets, off, on):
    co, cn = off == targets, on == targets
    fixed, broken = int((cn & ~co).sum()), int((co & ~cn).sum())
    changed = int((on != off).sum())
    require(fixed + broken <= changed and fixed - broken == int(cn.sum() - co.sum()), "stage13f_paired_counts")
    return dict(corrected=fixed, broken=broken, changed=changed)


def compare(balanced_log_evidence, cluster_ids, folds, groups, targets):
    """Toggle just the source's assignment on the same fixed log evidence."""
    validate_folds(folds)
    validate_truth(folds, groups, targets)
    require(cluster_ids.shape == folds.shape and cluster_ids.dtype.kind in "iu"
            and (cluster_ids >= 0).all(), "stage13f_cluster_axis")
    require(balanced_log_evidence.shape == (len(folds), K) and balanced_log_evidence.dtype == np.float64
            and np.isfinite(balanced_log_evidence).all(), "stage13f_log_evidence_axis")
    require(np.max(np.abs(np.exp(balanced_log_evidence).sum(axis=1) - (1 + K * 1e-12))) <= 2e-12,
            "stage13f_log_evidence_normalization")
    before = hashlib.sha256(balanced_log_evidence.astype("<f8", copy=False).tobytes()).hexdigest()
    inferred = buckets(cluster_ids)
    require(max(map(len, inferred)) <= CAP and all(np.unique(folds[idx]).size == 1 for idx in inferred),
            "stage13f_assignment_cluster_boundaries")
    dense = eligible(inferred)
    require(dense, "stage13f_no_assignment_eligible_groups")
    off = balanced_log_evidence.argmax(axis=1)
    on = off.copy()
    covered = np.zeros(len(folds), dtype=bool)
    for idx in dense:
        r, c = linear_sum_assignment(-balanced_log_evidence[idx])
        require(np.array_equal(r, np.arange(len(idx))) and len(np.unique(c)) == len(idx) and not covered[idx].any(),
                "stage13f_assignment_unique_coverage")
        on[idx[r]], covered[idx] = c, True
    require(np.array_equal(off[~covered], on[~covered])
            and hashlib.sha256(balanced_log_evidence.astype("<f8", copy=False).tobytes()).hexdigest() == before,
            "stage13f_assignment_changed_fixed_evidence_or_outside_cluster")
    targets = targets.astype(np.int64, copy=False)
    cm0, cm1 = reference.confusion(targets, off), reference.confusion(targets, on)
    recall0, recall1 = reference.recalls(cm0), reference.recalls(cm1)
    supports = cm0.sum(axis=1)
    deltas = [Fraction(int(cm1[i, i]) - int(cm0[i, i]), int(supports[i])) * 100 for i in range(K)]
    delta = sum(deltas) / K
    fold_reports = []
    rng = np.random.default_rng(reference.SEED)
    support_draws = np.zeros((reference.DRAWS, K), dtype=np.int64)
    diff_draws = np.zeros_like(support_draws)
    for fold in np.unique(folds):
        mask = folds == fold
        yt, pf, qf = targets[mask], off[mask], on[mask]
        c0, c1 = reference.confusion(yt, pf), reference.confusion(yt, qf)
        groups_f = groups[mask]
        gp, idx = np.unique(groups_f, return_inverse=True)
        prompt_support = np.zeros((len(gp), K), dtype=np.int64)
        prompt_diff = np.zeros_like(prompt_support)
        np.add.at(prompt_support, (idx, yt), 1)
        np.add.at(prompt_diff, (idx, yt), (qf == yt).astype(np.int64) - (pf == yt).astype(np.int64))
        w = rng.multinomial(len(gp), np.full(len(gp), 1 / len(gp)), size=reference.DRAWS)
        support_draws += w @ prompt_support
        diff_draws += w @ prompt_diff
        off_ba, on_ba = float(reference.recalls(c0).mean()), float(reference.recalls(c1).mean())
        fold_reports.append(dict(fold=int(fold), rows=int(mask.sum()), eligible_rows=int((covered & mask).sum()),
                                 assignment_off_ba=off_ba, assignment_on_ba=on_ba,
                                 delta_pp=100 * (on_ba - off_ba), **paired_counts(yt, pf, qf)))
    require((support_draws > 0).all(), "stage13f_bootstrap_missing_class")
    counts = paired_counts(targets, off, on)
    require(all(counts[key] == sum(f[key] for f in fold_reports) for key in counts)
            and counts["changed"] <= int(covered.sum())
            and np.array_equal(cm0.sum(1), cm1.sum(1)), "stage13f_assignment_pair_reconciliation")
    interval = np.percentile((diff_draws / support_draws).mean(axis=1) * 100, [2.5, 97.5]).tolist()
    return dict(assignment_off_ba=float(recall0.mean()), assignment_on_ba=float(recall1.mean()),
                delta_pp=float(delta), delta_pp_exact=str(delta), conditional_delta_ci95_pp=interval,
                class_supports=supports.tolist(), class_delta_pp=[float(d) for d in deltas],
                assignment_off_confusion_counts=cm0.tolist(), assignment_on_confusion_counts=cm1.tolist(),
                assignment_off_confusion_sha256=reference.digest_json(cm0.tolist()),
                assignment_on_confusion_sha256=reference.digest_json(cm1.tolist()),
                eligible_clusters=len(dense), eligible_rows=int(covered.sum()), folds=fold_reports,
                outside_eligible_clusters_unchanged=True, fixed_log_evidence_unchanged=True,
                bootstrap=dict(draws=reference.DRAWS, seed=reference.SEED,
                               scope="conditional true-prompt resampling on fixed scores, clusters and decisions",
                               limitation="inferred groups may couple true prompts; not pipeline-level uncertainty"),
                **counts)
