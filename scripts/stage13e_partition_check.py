"""Stage 13E: one alternate-partition stress check on reused development prompts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GroupKFold

import stage13_glm_readout as glm_readout
import stage13a_cache_admission as admission
import stage13b_balancing_stress as reference
import stage13c_deberta_complement as deberta
import stage13d_broader_complement as character


COMPLETE = "COMPLETE_ALTERNATE_PARTITION_STRESS_REUSED_PROMPTS"
FAILED = "FAILED_ALTERNATE_PARTITION_STRESS_REUSED_PROMPTS"
SCOPE = "exploratory_alternate_partition_on_reused_development_prompts"
SALT = "stage13e-split-stress-v1-20261111"
BLOCK = 30
K, ROWS, PROMPTS, FOLDS = reference.K, reference.ROWS, reference.PROMPTS, reference.FOLDS
INPUT_VERSIONS = {"competition": 170484, "own_glm_features": 359405443,
                  "pretrained_deberta_features": deberta.DEBERTA_VERSION}
WEIGHTS = {"G": {"glm": 1.0}, "C": {"glm": 0.75, "deberta": 0.25},
           "D": {"glm": 0.50, "char": 0.25, "deberta": 0.25}}
REPORT_NAME = "alternate_partition_report.json"
require = reference.require


def rank_prompts(prompts):
    """Rank only prompt IDs, with no score, class, or row-count inputs."""
    unique = np.unique(prompts).tolist()
    require(len(unique) == PROMPTS == BLOCK * FOLDS, "alternate_prompt_count")
    ranked = sorted(unique, key=lambda key: (reference.salted_hash(SALT, key), key.encode("utf-8")))
    return {key: offset // BLOCK + 1 for offset, key in enumerate(ranked)}


def alternate_partition(prompts, targets, canonical):
    require(prompts.shape == targets.shape == canonical.shape == (ROWS,)
            and prompts.dtype.kind == "U" and targets.dtype.kind in "iu", "partition_row_axes")
    old_map, _ = admission.validate_fold_partition(prompts, canonical, FOLDS)
    mapping = rank_prompts(prompts)
    folds = np.fromiter((mapping[key] for key in prompts), dtype=np.int64, count=ROWS)
    checked_map, details = admission.validate_fold_partition(prompts, folds, FOLDS)
    require(checked_map == mapping, "alternate_partition_mapping_reconciliation")
    old_blocks = {frozenset(k for k, f in old_map.items() if f == i) for i in range(1, FOLDS + 1)}
    new_blocks = {frozenset(k for k, f in mapping.items() if f == i) for i in range(1, FOLDS + 1)}
    require(old_blocks != new_blocks, "partition_only_renumbered_canonical_folds")
    require(sorted(item["valid_rows"] for item in details) == [539, 540, 540, 540, 540],
            "alternate_partition_heldout_row_counts")
    masks = [folds == i for i in range(1, FOLDS + 1)]
    require((np.asarray(masks).sum(axis=0) == 1).all(), "alternate_partition_unique_oof_coverage")
    for item, val in zip(details, masks):
        require((item["fit_prompts"], item["valid_prompts"], item["prompt_overlap"]) == (120, BLOCK, 0),
                "alternate_partition_prompt_exclusion")
        require(all(np.array_equal(np.unique(targets[subset]), np.arange(K)) for subset in (val, ~val)),
                "alternate_partition_class_coverage")
    return folds, masks, details, admission.json_hash(mapping)


def arm_probabilities(glm, deb, char):
    for p in (glm, deb, char):
        require(p.shape == (ROWS, K), "alternate_oof_axes")
        reference.validate_p(p)
    return {"G": glm, "C": WEIGHTS["C"]["glm"] * glm + WEIGHTS["C"]["deberta"] * deb,
            "D": WEIGHTS["D"]["glm"] * glm + WEIGHTS["D"]["char"] * char + WEIGHTS["D"]["deberta"] * deb}


def compare_arms(glm, deb, char, prompts, y, masks, folds):
    preds, mass_errors = {}, {}
    for arm, p in arm_probabilities(glm, deb, char).items():
        preds[arm], mass_errors[arm] = character.uniform_decisions(p, masks)

    rng = np.random.default_rng(reference.SEED)
    supports = np.zeros((reference.DRAWS, K), dtype=np.int64)
    cg_draws, dc_draws = np.zeros_like(supports), np.zeros_like(supports)
    for f, val in enumerate(masks):
        yt = y[val]
        groups = np.unique(prompts[val])
        require(len(groups) == BLOCK, "bootstrap_prompt_axis")
        weights = rng.multinomial(BLOCK, np.full(BLOCK, 1 / BLOCK), size=reference.DRAWS)
        index = np.searchsorted(groups, prompts[val])
        hit = {arm: pred[val] == yt for arm, pred in preds.items()}
        s1, d1 = reference.bootstrap_delta(index, yt, hit["C"], hit["G"], weights)
        s2, d2 = reference.bootstrap_delta(index, yt, hit["D"], hit["C"], weights)
        require(np.array_equal(s1, s2), "bootstrap_pair_support_mismatch")
        supports += s1
        cg_draws += d1
        dc_draws += d2
        fold_ba = {arm: float(reference.recalls(reference.confusion(yt, pred[val])).mean())
                   for arm, pred in preds.items()}
        folds[f].update(balanced_accuracy=fold_ba,
                        column_mass_error={arm: errors[f] for arm, errors in mass_errors.items()},
                        delta_pp={"C_minus_G": (fold_ba["C"] - fold_ba["G"]) * 100,
                                  "D_minus_C": (fold_ba["D"] - fold_ba["C"]) * 100},
                        corrected={"C_minus_G": int((hit["C"] & ~hit["G"]).sum()),
                                   "D_minus_C": int((hit["D"] & ~hit["C"]).sum())},
                        broken={"C_minus_G": int((hit["G"] & ~hit["C"]).sum()),
                                "D_minus_C": int((hit["C"] & ~hit["D"]).sum())})

    cg = character.paired_summary(y, preds["G"], preds["C"], supports, cg_draws)
    dc = character.paired_summary(y, preds["C"], preds["D"], supports, dc_draws)
    require(cg["treatment_confusion_counts"] == dc["control_confusion_counts"], "shared_control_reconciliation")
    for key, result in (("C_minus_G", cg), ("D_minus_C", dc)):
        require(all(result[k] == sum(fold[k][key] for fold in folds) for k in ("corrected", "broken")),
                "paired_fold_count_reconciliation")
    cg_gates = dict(gain_at_least_half_pp=cg["gain_at_least_half_pp"], ci_lower_positive=cg["ci_lower_positive"])
    dc_gates = dict(gain_at_least_half_pp=dc["gain_at_least_half_pp"], ci_lower_positive=dc["ci_lower_positive"],
                    class_loss_within_five_pp=dc["class_loss_within_five_pp"])
    return dict(balanced_accuracy={"G": cg["control_ba"], "C": cg["treatment_ba"], "D": dc["treatment_ba"]},
                C_minus_G=cg, D_minus_C=dc, comparator_gates=cg_gates, marginal_gates=dc_gates,
                comparator_admitted=all(cg_gates.values()),
                passes_split_stress_screen=all(cg_gates.values()) and all(dc_gates.values()))


def run(input_root, output_root, *, progress=None):
    started = time.monotonic()
    root, out = Path(input_root), Path(output_root)
    report = dict(schema_version=1, status=FAILED, scope=SCOPE, reused_prompts=True,
                  untouched_prompt_validation=False, official_score_estimated=False,
                  full_s07_oof_admitted=False, raw_rows_or_predictions_exported=False,
                  fit_attempted=False, fit_performed=False, comparator_admitted=False,
                  passes_split_stress_screen=False, canonical_oof_probabilities_used=False,
                  expected_input_versions=INPUT_VERSIONS, input_versions_verified_by_filesystem=False,
                  expected_input_sha256={"train": admission.TRAIN_SHA256, "glm": admission.GLM_SHA256,
                                         "deberta": deberta.DEBERTA_SHA},
                  prompt_partition_salt=SALT, arm_weights=WEIGHTS,
                  limits=["All 150 prompts were reused from prior development stages.",
                          "A pass supports alternate-partition robustness only.",
                          "The interval conditions on fixed fitted and balanced predictions; no refitting in bootstrap.",
                          "No official Kaggle score or full S07 effect is estimated."])
    writable = False
    try:
        require(not out.exists() or (out.is_dir() and not any(out.iterdir())), "fresh_output_directory")
        out.mkdir(parents=True, exist_ok=True)
        writable = True
        require(sklearn_version == glm_readout.REQUIRED_SKLEARN == deberta.SKLEARN_VERSION, "sklearn_version")
        audited = admission.audit_inputs(root, out / "input_admission")
        report.update(input_admission_status=audited["status"], input_admission_checks=audited["checks_passed"])
        if audited["status"] != "PASS_FROZEN_BASE_FEATURE_ADMISSION_ONLY":
            report["input_admission_failure_code"] = audited.get("failure_code")
        require(audited["status"] == "PASS_FROZEN_BASE_FEATURE_ADMISSION_ONLY"
                and audited["contract_is_production"] is True and audited["report_written"] is True
                and audited["checks_passed"] == glm_readout.AUDIT_CHECKS, "production_input_admission")
        train_path = admission.unique_file(root, "train.csv")
        glm_path = admission.unique_file(root, "glm_features.npz")
        with train_path.open("rb") as stream:
            require(hashlib.file_digest(stream, "sha256").hexdigest() == admission.TRAIN_SHA256,
                    "post_admission_train_hash")
            stream.seek(0)
            train = pd.read_csv(stream, dtype=str, encoding="utf-8", usecols=["id", "prompt_id", "model"])
        ids, prompts = train.id.to_numpy(dtype=str), train.prompt_id.to_numpy(dtype=str)
        classes = np.asarray(admission.CLASSES, dtype=str)
        y = train.model.map({name: i for i, name in enumerate(classes)}).to_numpy(dtype=np.int64)
        require(admission.json_hash(ids.tolist()) == audited["id_axis_sha256"] == reference.ID_SHA
                and admission.json_hash(classes.tolist()) == audited["label_axis_sha256"] == reference.LABEL_SHA,
                "post_admission_axes")
        texts = character.load_texts(train_path, ids, prompts, y, classes)
        canonical = np.zeros(ROWS, dtype=np.int64)
        for fold, (_, valid) in enumerate(GroupKFold(FOLDS).split(np.zeros((ROWS, 1), dtype=np.uint8), y, prompts), 1):
            require((canonical[valid] == 0).all(), "canonical_fold_overlap")
            canonical[valid] = fold
        canonical_map, _ = admission.validate_fold_partition(prompts, canonical, FOLDS)
        require(admission.json_hash(canonical_map) == admission.FOLD_MAP_SHA256 == reference.FOLD_SHA,
                "canonical_audit_fold_digest")
        folds, masks, details, digest = alternate_partition(prompts, y, canonical)
        report.update(rows=ROWS, prompts=PROMPTS, classes=classes.tolist(), folds=details,
                      canonical_partition_sha256=admission.FOLD_MAP_SHA256, alternate_partition_sha256=digest,
                      alternate_membership_differs=True, official_train_ordered_text_sha256=character.TEXT_SHA)
        if progress is not None:
            progress("STAGE13E_FIXED_ALTERNATE_PARTITION", {"sha256": digest, "reused_prompts": True})

        with glm_path.open("rb") as stream:
            # Validate the bytes and load the same stream, with no canonical OOF input.
            require(hashlib.file_digest(stream, "sha256").hexdigest() == admission.GLM_SHA256,
                    "post_admission_glm_hash")
            stream.seek(0)
            with np.load(stream, allow_pickle=False) as cache:
                require(np.array_equal(cache["ids"], ids) and np.array_equal(cache["layers"], [8, 3, 24]),
                        "glm_feature_axis_binding")
                values = cache["features"]
                require(values.shape == (ROWS, 3, 4096) and values.dtype == np.float32, "glm_feature_shape_dtype")
                gx = np.empty((ROWS, 8192), dtype=np.float64, order="C")
                gx[:, :4096], gx[:, 4096:] = values[:, 0, :], values[:, 1, :]
                del values
        dx, id_mode = deberta.load_features(reference.one(root, "enc_debv3l2_train.npz"), ids)
        report["fit_attempted"] = True

        def glm_progress(detail):
            if progress is not None:
                progress("STAGE13E_GLM_FOLD", {key: detail[key] for key in ("fold", "iterations", "seconds")})

        gp, g_details, g_pools, max_error = glm_readout.fit_grouped(
            gx, y, prompts, folds, classes, FOLDS, on_fold=glm_progress)
        del gx
        dp, d_details, d_pools = deberta.fit_oof(dx, y, masks)
        del dx
        if progress is not None:
            progress("STAGE13E_DEBERTA_FOLDS_COMPLETE", {"folds": len(d_details)})
        cp, c_details, c_pools = character.fit_char_oof(
            texts, y, masks, on_fold=None if progress is None else lambda d: progress("STAGE13E_CHAR_FOLD", d))
        report["fit_performed"] = True
        for detail, g, d, c in zip(details, g_details, d_details, c_details, strict=True):
            require(all(g[key] == d[key] == detail[key] for key in (
                "fold", "fit_rows", "valid_rows", "fit_prompts", "valid_prompts", "prompt_overlap"))
                    and c["fold"] == detail["fold"], "head_fold_reconciliation")
            detail.update(glm_fit_iterations=g["iterations"], glm_fit_seconds=g["seconds"],
                          deberta_fit_iterations=d["iterations"], deberta_fit_seconds=d["seconds"],
                          **{key: value for key, value in c.items() if key != "fold"})
        scores = compare_arms(gp, dp, cp, prompts, y, masks, details)
        require(scores["C_minus_G"]["class_supports"] == [audited["class_supports"][label] for label in classes],
                "official_class_support_reconciliation")
        report.update(status=COMPLETE, **scores, deberta_id_verification=id_mode,
                      estimators={"glm": glm_readout.HEAD, "deberta": deberta.HEAD, "char": character.CHAR_HEAD},
                      glm_layers=[8, 3], deberta_layers=[24, 6],
                      char_vectorizer={**character.CHAR_TFIDF, "ngram_range": [3, 5], "dtype": "float64"},
                      thread_pools={"glm": g_pools, "deberta": d_pools, "char": c_pools},
                      glm_probability_max_row_sum_error=max_error, sklearn_version=sklearn_version,
                      numpy_version=np.__version__, pandas_version=pd.__version__,
                      balancing={"alpha": 1, "iterations": 100, "minimum_batch_rows": K, "fold_local": True},
                      bootstrap={"draws": reference.DRAWS, "seed": reference.SEED,
                                 "scope": "shared paired within-fold prompt draws on fixed predictions"})
    except (reference.GateError, admission.AdmissionFailure) as error:
        report.update(failure_code=str(error), failure_type=type(error).__name__)
    except ConvergenceWarning:
        report.update(failure_code="convergence_warning", failure_type="ConvergenceWarning")
    except Exception as error:
        report.update(failure_code="runtime_failure", failure_type=type(error).__name__)

    report["seconds"] = round(time.monotonic() - started, 6)
    if writable:
        tmp = out / ".alternate_partition_report.tmp"
        try:
            tmp.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            require(json.loads(tmp.read_text(encoding="utf-8")) == report, "aggregate_report_readback")
            tmp.replace(out / REPORT_NAME)
        except Exception as error:
            tmp.unlink(missing_ok=True)
            report.update(status=FAILED, comparator_admitted=False, passes_split_stress_screen=False,
                          failure_code="aggregate_report_write_or_readback", failure_type=type(error).__name__)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default="/kaggle/input")
    parser.add_argument("--output-root", default="/kaggle/working/stage13e_partition_check")
    args = parser.parse_args()

    def progress(marker, payload):
        print(marker, json.dumps(payload, sort_keys=True, allow_nan=False), flush=True)

    report = run(args.input_root, args.output_root, progress=progress)
    print("STAGE13E_PARTITION_CHECK_JSON_BEGIN", flush=True)
    print(json.dumps(report, sort_keys=True, allow_nan=False), flush=True)
    print("STAGE13E_PARTITION_CHECK_JSON_END", flush=True)
    return 0 if report["status"] == COMPLETE else 2


if __name__ == "__main__":
    raise SystemExit(main())
