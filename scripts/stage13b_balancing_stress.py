"""Run one predeclared balancing diagnostic on the private GLM OOF cache."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np


COMPLETE = "COMPLETE_GLM_SINKHORN_STRESS_ONLY"
SCOPE = "glm_only_paired_fold_local_balancing_stress"
SOURCE_VERSION = 368292220
CACHE_SHA = "565df6f8e652a9850ff11cf4e8b6a81fcccbc7db593186ff7f53358e7190bd16"
PROB_SHA = "41d80bc9350ab44d5aabe02b24e0174881a307bb1eb962f04196682e7faa94fe"
ID_SHA = "707edf645b0f09bba8f0983b5bc9f522595c95e8ca5b763b38babeb38200368d"
LABEL_SHA = "088b14188775e41a314f269da3af74f96144a56caa242ccbeacc9e3c817a7e08"
FOLD_SHA = "a55b035686cb7b31f6fbe852ac990668709aec4b913c5d0928d89d843bdbfe62"
EXPECTED_RECALL = 61547 / 134100
K, ROWS, PROMPTS, FOLDS = 18, 2699, 150, 5
SCENARIOS = ("natural_full", "matched_balanced", "skew_a", "skew_b")
DRAWS, SEED = 5000, 20261111
ATOL = 1e-12


class GateError(Exception):
    pass


def require(condition, code):
    if not bool(condition):
        raise GateError(code)


def digest_json(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def salted_hash(salt, value):
    return hashlib.sha256((salt + "\0" + str(value)).encode("utf-8")).digest()


def one(root, name):
    matches = [p for p in root.rglob(name) if p.is_file()]
    require(len(matches) == 1, "unique_" + name.replace(".", "_"))
    return matches[0]


def validate_p(p):
    require(p.ndim == 2 and p.shape[1] == K and p.dtype == np.float64 and len(p) > 0,
            "probability_shape_or_dtype")
    require(np.isfinite(p).all() and (p > 0).all() and (p <= 1).all(), "invalid_probability_values")
    require(np.max(np.abs(p.sum(axis=1) - 1.0)) <= ATOL, "probability_row_sums")


def evidence(p, uniform):
    """Use the source's fixed 100 updates and branch threshold on existing P."""
    validate_p(p)
    if not uniform or len(p) < K:
        return np.log(p + 1e-12), None
    q = p.copy()
    with np.errstate(divide="raise", invalid="raise", over="raise"):
        for _ in range(100):
            q *= (len(q) / 18) / q.sum(0, keepdims=True)
            q /= q.sum(1, keepdims=True)
    validate_p(q)
    residual = float(np.max(np.abs(q.sum(axis=0) - len(q) / K)))
    require(np.isfinite(residual), "nonfinite_column_mass_error")
    return np.log(q + 1e-12), residual


def cache_inputs(root):
    cache = one(root, "heldout_probabilities.npz")
    report = one(root, "readout_report.json")
    require(cache.parent == report.parent, "cache_report_parent_mismatch")
    with cache.open("rb") as f:
        require(hashlib.file_digest(f, "sha256").hexdigest() == CACHE_SHA, "cache_byte_hash_mismatch")
    r = json.loads(report.read_text(encoding="utf-8"))
    require(r.get("status") == "COMPLETE_GLM_PLAIN_GROUPED_DEVELOPMENT_REFERENCE"
            and r.get("scope") == "glm_only_prompt_heldout_development"
            and r.get("heldout_cache_sha256") == CACHE_SHA
            and r.get("probability_values_sha256") == PROB_SHA
            and r.get("admission_checks_passed") == 7, "producer_report_contract")
    for flag in ("fit_performed", "heldout_predictions_produced", "heldout_cache_readback_passed"):
        require(r.get(flag) is True, "producer_report_missing_required_flag")
    for flag in ("balancing_performed", "official_score_estimated", "full_s07_oof_admitted", "raw_rows_or_predictions_exported"):
        require(r.get(flag) is False, "producer_report_scope")
    keys = ("ids", "prompt_ids", "targets", "folds", "classes", "probabilities")
    with np.load(cache, allow_pickle=False) as z:
        require(len(z.files) == len(keys) and set(z.files) == set(keys), "cache_schema")
        ids, prompts, targets, folds, classes, p = (z[k] for k in keys)
    require(ids.shape == prompts.shape == targets.shape == folds.shape == (ROWS,), "cache_row_axes")
    require(ids.dtype.kind == prompts.dtype.kind == classes.dtype.kind == "U"
            and targets.dtype == folds.dtype == np.int64 and classes.shape == (K,), "cache_axis_dtypes")
    require(len(np.unique(ids)) == ROWS and len(np.unique(prompts)) == PROMPTS, "cache_unique_axes")
    require(digest_json(ids.tolist()) == ID_SHA and digest_json(classes.tolist()) == LABEL_SHA, "cache_axis_hashes")
    require(np.array_equal(np.unique(targets), np.arange(K)) and np.array_equal(np.unique(folds), np.arange(1, 6)),
            "cache_class_or_fold_indices")
    require(np.bincount(targets).tolist() == [149] + [150] * 17, "original_class_supports")
    validate_p(p)
    require(len(p) == ROWS and hashlib.sha256(p.astype("<f8", copy=False).tobytes()).hexdigest() == PROB_SHA,
            "cache_probability_hash")
    mapping = {}
    for group, fold in zip(prompts, folds):
        require(mapping.setdefault(str(group), int(fold)) == int(fold), "prompt_crosses_fold")
    require(digest_json(mapping) == FOLD_SHA and r.get("canonical_prompt_fold_mapping_sha256") == FOLD_SHA,
            "canonical_fold_map_hash")
    require(r.get("classes") == classes.tolist() and len(r.get("folds", [])) == FOLDS, "producer_fold_axis")
    for f, item in enumerate(r["folds"], 1):
        n = 540 if f < 5 else 539
        mask = folds == f
        require(item.get("fold") == f and int(mask.sum()) == n and len(np.unique(prompts[mask])) == 30
                and [item.get(k) for k in ("fit_rows", "valid_rows", "fit_prompts", "valid_prompts", "prompt_overlap")]
                == [ROWS - n, n, 120, 30, 0], "producer_fold_exclusion_contract")
    return ids, prompts, targets, folds, classes, p, r


def mask_indices(ids, targets, classes, indices):
    """Rank once per class. Use matched subsets for all paired scoring arms."""
    ranked = sorted(range(K), key=lambda c: (salted_hash("stage13b-class-v1", classes[c]), str(classes[c]).encode("utf-8")))
    a_major = set(ranked[:9])
    buckets = [sorted(indices[targets[indices] == c].tolist(),
                      key=lambda i: (salted_hash("stage13b-row-v1", ids[i]), str(ids[i]).encode("utf-8"))) for c in range(K)]
    require(all(len(rows) >= 27 for rows in buckets), "insufficient_mask_class_support")
    masks = {"natural_full": indices}
    for name in SCENARIOS[1:]:
        counts = [20] * K if name == "matched_balanced" else [
            27 if ((c in a_major) == (name == "skew_a")) else 13 for c in range(K)]
        chosen = np.asarray(sorted(i for c, rows in enumerate(buckets) for i in rows[:counts[c]]), dtype=np.int64)
        require(len(chosen) == len(np.unique(chosen)) == 360 and np.bincount(targets[chosen], minlength=K).tolist() == counts,
                "matched_mask_supports")
        masks[name] = chosen
    return masks


def confusion(y, pred):
    return np.bincount(y * K + pred, minlength=K*K).reshape(K, K)


def recalls(counts):
    totals = counts.sum(axis=1)
    require((totals > 0).all(), "class_missing_from_score")
    return counts.diagonal() / totals


def bootstrap_delta(prompt_index, targets, correct_off, correct_uniform, weights):
    supports = np.zeros((30, K), dtype=np.int64)
    differences = np.zeros_like(supports)
    np.add.at(supports, (prompt_index, targets), 1)
    np.add.at(differences, (prompt_index, targets), correct_off.astype(np.int64) - correct_uniform.astype(np.int64))
    return weights @ supports, weights @ differences


def diagnostics(ids, prompts, targets, folds, classes, p, producer):
    # Freeze the shared paired resampling before applying either arm.
    rng = np.random.default_rng(SEED)
    weights = [rng.multinomial(30, np.full(30, 1/30), size=DRAWS) for _ in range(FOLDS)]
    stores = {n: {"off": np.zeros((K, K), dtype=np.int64), "uniform": np.zeros((K, K), dtype=np.int64),
                  "support_draws": np.zeros((DRAWS, K), dtype=np.int64), "delta_draws": np.zeros((DRAWS, K), dtype=np.int64),
                  "fixed": 0, "broken": 0, "changed": 0, "batches": []} for n in SCENARIOS}
    digest_before = hashlib.sha256(p.astype("<f8", copy=False).tobytes()).hexdigest()
    for f in range(1, 6):
        indices = np.flatnonzero(folds == f)
        ordered_groups = np.unique(prompts[indices])
        require(len(ordered_groups) == 30, "scoring_batch_prompt_count")
        for name, selected in mask_indices(ids, targets, classes, indices).items():
            require(len(np.unique(prompts[selected])) == 30 and (folds[selected] == f).all(), "masked_batch_fold_or_prompt_count")
            y, pp = targets[selected], p[selected]
            log_off, _ = evidence(pp, False)
            log_uniform, residual = evidence(pp, True)
            pred_off, pred_uniform = log_off.argmax(1), log_uniform.argmax(1)
            off, uniform = confusion(y, pred_off), confusion(y, pred_uniform)
            co, cu = pred_off == y, pred_uniform == y
            fixed, broken = int((co & ~cu).sum()), int((cu & ~co).sum())
            changed = int((pred_off != pred_uniform).sum())
            require(fixed + broken <= changed and int(co.sum() - cu.sum()) == fixed - broken, "paired_counts")
            b = stores[name]
            b["off"] += off
            b["uniform"] += uniform
            b["fixed"] += fixed
            b["broken"] += broken
            b["changed"] += changed
            b["batches"].append(dict(fold=f, rows=len(y), prompts=30, class_supports=off.sum(1).tolist(),
                                     off_ba=float(recalls(off).mean()), uniform_ba=float(recalls(uniform).mean()),
                                     uniform_max_column_mass_error=residual))
            ds, dd = bootstrap_delta(np.searchsorted(ordered_groups, prompts[selected]), y, co, cu, weights[f-1])
            b["support_draws"] += ds
            b["delta_draws"] += dd
    natural = stores["natural_full"]["off"]
    require(np.array_equal(natural, np.asarray(producer["confusion_counts"]))
            and abs(float(recalls(natural).mean()) - EXPECTED_RECALL) <= 1e-15
            and abs(float(producer["pooled_balanced_accuracy"]) - EXPECTED_RECALL) <= 1e-15, "original_plain_reference_replay")
    require(hashlib.sha256(p.astype("<f8", copy=False).tobytes()).hexdigest() == digest_before, "mutated_upstream_probabilities")
    results = {}
    for name, b in stores.items():
        off, uniform = b["off"], b["uniform"]
        ro, ru = recalls(off), recalls(uniform)
        require(np.array_equal(off.sum(1), uniform.sum(1)) and (b["support_draws"] > 0).all(), "paired_support_or_bootstrap_coverage")
        intervals = np.percentile((b["delta_draws"] / b["support_draws"]).mean(axis=1) * 100, [2.5, 97.5])
        results[name] = dict(rows=int(off.sum()), batches=b["batches"], off_ba=float(ro.mean()), uniform_ba=float(ru.mean()),
                             delta_pp=float((ro.mean() - ru.mean()) * 100), conditional_delta_ci95_pp=intervals.tolist(),
                             class_supports=off.sum(1).tolist(), class_delta_pp=((ro - ru) * 100).tolist(),
                             off_class_recalls=ro.tolist(), uniform_class_recalls=ru.tolist(),
                             off_confusion_counts=off.tolist(), uniform_confusion_counts=uniform.tolist(),
                             corrected_by_off=b["fixed"], broken_by_off=b["broken"], changed_by_off=b["changed"])
    require([results[n]["rows"] for n in SCENARIOS] == [2699, 1800, 1800, 1800], "scenario_row_counts")
    criteria = dict(
        both_skew_point_gains_ge_half_pp=all(results[n]["delta_pp"] >= 0.5 for n in ("skew_a", "skew_b")),
        both_skew_conditional_ci_lowers_positive=all(results[n]["conditional_delta_ci95_pp"][0] > 0 for n in ("skew_a", "skew_b")),
        natural_and_matched_losses_le_half_pp=all(results[n]["delta_pp"] >= -0.5 for n in ("natural_full", "matched_balanced")),
        every_class_recall_loss_le_five_pp=all(min(b["class_delta_pp"]) >= -5.0 - 1e-12 for b in results.values()),
    )
    return results, criteria


def run(input_root, output_root):
    t = time.monotonic()
    root, out = Path(input_root), Path(output_root)
    report = dict(schema_version=1, status="FAILED_GLM_SINKHORN_STRESS", scope=SCOPE, source_version_id=SOURCE_VERSION,
                  heldout_cache_sha256=CACHE_SHA, fit_performed=False, official_score_estimated=False,
                  full_s07_oof_admitted=False, raw_rows_or_predictions_exported=False)
    writable = False
    try:
        require(not out.exists() or (out.is_dir() and not any(out.iterdir())), "fresh_output_directory")
        out.mkdir(parents=True, exist_ok=True)
        writable = True
        ids, groups, targets, folds, classes, p, prior = cache_inputs(root)
        results, criteria = diagnostics(ids, groups, targets, folds, classes, p, prior)
        report.update(status=COMPLETE, scenarios=results, classes=classes.tolist(), decision_criteria=criteria,
                      passes_proxy_screen=all(criteria.values()), probabilities_unchanged=True,
                      original_plain_reference_replayed=True, actual_folds=5, prompts=150,
                      bootstrap=dict(draws=DRAWS, seed=SEED, scope="fixed-prediction stratified prompt resampling"),
                      control=dict(alpha=1, iterations=100, minimum_batch_rows=18), candidate=dict(alpha=0),
                      delta_direction="off_minus_uniform", numpy_version=np.__version__)
    except GateError as error:
        report.update(failure_code=str(error))
    except Exception as error:
        report.update(failure_code="runtime_failure", failure_type=type(error).__name__)
    report["seconds"] = round(time.monotonic() - t, 6)
    if writable:
        try:
            tmp = out / ".balancing_stress_report.tmp"
            tmp.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            tmp.replace(out / "balancing_stress_report.json")
        except Exception as error:
            report.update(status="FAILED_GLM_SINKHORN_STRESS", failure_code="report_write_failure", failure_type=type(error).__name__)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default="/kaggle/input")
    parser.add_argument("--output-root", default="/kaggle/working/stage13b_balancing_stress")
    args = parser.parse_args()
    result = run(args.input_root, args.output_root)
    print("STAGE13B_GLM_STRESS_JSON_BEGIN", flush=True)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    print("STAGE13B_GLM_STRESS_JSON_END", flush=True)
    return 0 if result["status"] == COMPLETE else 2


if __name__ == "__main__":
    raise SystemExit(main())
