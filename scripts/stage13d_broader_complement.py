"""Stage 13D: test a fixed DeBERTa swap after adding fold-local char evidence."""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_info, threadpool_limits

import stage13b_balancing_stress as reference
import stage13c_deberta_complement as prior


COMPLETE = "COMPLETE_BROADER_DEBERTA_COMPLEMENT_SCREEN"
FAILED = "FAILED_BROADER_DEBERTA_COMPLEMENT_SCREEN"
SCOPE = "glm_char_deberta_paired_fold_local_development"
TRAIN_VERSION = 170484
TRAIN_SHA = "40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707"
TEXT_SHA = "6496071874da8f911fafebd9eec6191e56134a97f00226dd8c78805d31f07e36"
STAGE13C_CM_SHA = "c076ee1a507753660bd7356baa3083d277c8fc8e61bedc81bae9192986caad9b"
STAGE13C_BA = 95677 / 201150
CHAR_TFIDF = dict(analyzer="char", lowercase=False, ngram_range=(3, 5), min_df=2,
                  max_features=50000, sublinear_tf=True, norm="l2")
CHAR_HEAD = dict(C=10, solver="lbfgs", max_iter=1000, tol=1e-4, random_state=0)
WEIGHTS = {"broader_control": {"glm": 0.75, "char": 0.25},
           "candidate": {"glm": 0.50, "char": 0.25, "deberta": 0.25}}
K, ROWS, FOLDS = reference.K, reference.ROWS, reference.FOLDS
require = reference.require


def load_texts(path, ids, prompts, y, classes):
    """Admit exact official bytes and require alignment without reordering."""
    with path.open("rb") as stream:
        require(hashlib.file_digest(stream, "sha256").hexdigest() == TRAIN_SHA,
                "stage13d_official_train_hash")
        stream.seek(0)
        frame = pd.read_csv(stream, dtype=str, encoding="utf-8")
    require(set(frame.columns) == {"id", "text", "prompt_id", "model"}
            and len(frame) == ROWS and not frame.isna().any().any()
            and all(frame[c].str.len().gt(0).all() for c in frame.columns), "stage13d_train_schema")
    require(np.array_equal(frame.id.to_numpy(dtype=str), ids), "stage13d_train_id_axis")
    require(np.array_equal(frame.prompt_id.to_numpy(dtype=str), prompts), "stage13d_train_prompt_axis")
    require(np.array_equal(frame.model.to_numpy(dtype=str), classes[y]), "stage13d_train_model_axis")
    texts = frame.text.to_numpy(dtype=str)
    require(reference.digest_json([hashlib.sha256(x.encode("utf-8")).hexdigest() for x in texts]) == TEXT_SHA,
            "stage13d_train_ordered_text_hash")
    return texts


def fit_char_oof(texts, y, masks, *, on_fold=None):
    require(texts.shape == y.shape == (ROWS,) and texts.dtype.kind == "U"
            and len(masks) == FOLDS and (np.asarray(masks).sum(axis=0) == 1).all(),
            "stage13d_char_oof_input_axes")
    p = np.full((ROWS, K), np.nan, dtype=np.float64)
    filled = np.zeros(ROWS, dtype=np.int8)
    details = []
    with threadpool_limits(limits=prior.THREADS):
        pools = [{"api": item["user_api"], "threads": item["num_threads"]} for item in threadpool_info()]
        require(all(0 < pool["threads"] <= prior.THREADS for pool in pools), "stage13d_char_thread_limit")
        for fold, val in enumerate(masks, 1):
            start = time.monotonic()
            vectorizer = TfidfVectorizer(**CHAR_TFIDF, dtype=np.float64)
            fit_x = vectorizer.fit_transform(texts[~val])
            require(0 < fit_x.shape[1] <= CHAR_TFIDF["max_features"]
                    and fit_x.shape[0] == int((~val).sum())
                    and np.isfinite(fit_x.data).all(), "stage13d_char_fitting_matrix")
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                estimator = LogisticRegression(**CHAR_HEAD).fit(fit_x, y[~val])
            require(np.array_equal(estimator.classes_, np.arange(K)), "stage13d_char_estimator_classes")
            iterations = [int(x) for x in estimator.n_iter_]
            require(len(iterations) == 1 and 0 < iterations[0] < CHAR_HEAD["max_iter"],
                    "stage13d_char_iteration_limit")
            score_x = vectorizer.transform(texts[val])
            require(score_x.shape == (int(val.sum()), fit_x.shape[1])
                    and np.isfinite(score_x.data).all(), "stage13d_char_scoring_matrix")
            scores = estimator.predict_proba(score_x)
            reference.validate_p(scores)
            require(len(scores) == int(val.sum()) and (filled[val] == 0).all(),
                    "stage13d_char_oof_write")
            p[val], filled[val] = scores, 1
            details.append(dict(fold=fold, char_iterations=iterations[0],
                                char_vocabulary_size=int(fit_x.shape[1]),
                                char_zero_feature_heldout_rows=int((np.diff(score_x.indptr) == 0).sum()),
                                char_seconds=round(time.monotonic() - start, 6)))
            if on_fold is not None:
                on_fold(dict(fold=fold, feature_count=int(fit_x.shape[1]), seconds=details[-1]["char_seconds"]))
    require((filled == 1).all(), "stage13d_char_oof_complete")
    reference.validate_p(p)
    return p, details, pools


def uniform_decisions(p, masks):
    pred = np.full(ROWS, -1, dtype=np.int64)
    residuals = []
    for val in masks:
        require((pred[val] == -1).all(), "stage13d_duplicate_scoring_rows")
        scores, residual = reference.evidence(p[val], True)
        require(residual is not None, "stage13d_balancing_skipped")
        pred[val] = scores.argmax(axis=1)
        residuals.append(residual)
    require((pred >= 0).all(), "stage13d_predictions_incomplete")
    return pred, residuals


def replay_stage13c(glm, deb, y, masks):
    pred, _ = uniform_decisions(0.75 * glm + 0.25 * deb, masks)
    cm = reference.confusion(y, pred)
    require(reference.digest_json(cm.tolist()) == STAGE13C_CM_SHA
            and int(cm.trace()) == 1284
            and abs(float(reference.recalls(cm).mean()) - STAGE13C_BA) < 1e-15,
            "stage13d_stage13c_candidate_confusion_mismatch")


def paired_summary(y, control, treatment, supports_draws, delta_draws):
    c0, c1 = reference.confusion(y, control), reference.confusion(y, treatment)
    supports = c0.sum(axis=1)
    require((supports > 0).all() and np.array_equal(supports, c1.sum(axis=1))
            and (supports_draws > 0).all(), "stage13d_pair_support")
    delta_class = [Fraction(int(c1[i, i]) - int(c0[i, i]), int(supports[i])) * 100 for i in range(K)]
    delta = sum(delta_class) / K
    interval = np.percentile((delta_draws / supports_draws).mean(axis=1) * 100, [2.5, 97.5]).tolist()
    c0good, c1good = control == y, treatment == y
    fixed, broken = int((c1good & ~c0good).sum()), int((c0good & ~c1good).sum())
    changed = int((control != treatment).sum())
    require(fixed + broken <= changed and fixed - broken == int(c1.trace() - c0.trace())
            and np.isfinite(interval).all(), "stage13d_pair_counts")
    return dict(control_ba=float(reference.recalls(c0).mean()), treatment_ba=float(reference.recalls(c1).mean()),
                delta_pp=float(delta), delta_pp_exact=str(delta), conditional_delta_ci95_pp=interval,
                class_supports=supports.tolist(), class_delta_pp=[float(x) for x in delta_class],
                control_class_recalls=reference.recalls(c0).tolist(), treatment_class_recalls=reference.recalls(c1).tolist(),
                control_confusion_counts=c0.tolist(), treatment_confusion_counts=c1.tolist(),
                corrected=fixed, broken=broken, changed=changed,
                gain_at_least_half_pp=delta >= Fraction(1, 2),
                ci_lower_positive=interval[0] > 0, class_loss_within_five_pp=min(delta_class) >= -5)


def compare_arms(glm, char, deb, prompts, y, masks, glm_pred, glm_residuals, folds):
    # Keep the char share fixed; the candidate swaps 25 points of GLM for DeBERTa.
    broad_pred, broad_residuals = uniform_decisions(0.75 * glm + 0.25 * char, masks)
    cand_pred, cand_residuals = uniform_decisions(0.50 * glm + 0.25 * char + 0.25 * deb, masks)
    rng = np.random.default_rng(reference.SEED)
    supports = np.zeros((reference.DRAWS, K), dtype=np.int64)
    broad_delta_draws, cand_delta_draws = np.zeros_like(supports), np.zeros_like(supports)
    for f, val in enumerate(masks):
        yt = y[val]
        g, b, c = glm_pred[val], broad_pred[val], cand_pred[val]
        gg, bg, cg = g == yt, b == yt, c == yt
        groups = np.unique(prompts[val])
        require(len(groups) == 30, "stage13d_paired_prompt_axis")
        weights = rng.multinomial(30, np.full(30, 1 / 30), size=reference.DRAWS)
        index = np.searchsorted(groups, prompts[val])
        s1, d1 = reference.bootstrap_delta(index, yt, bg, gg, weights)
        s2, d2 = reference.bootstrap_delta(index, yt, cg, bg, weights)
        require(np.array_equal(s1, s2), "stage13d_pair_draw_support_mismatch")
        supports += s1
        broad_delta_draws += d1
        cand_delta_draws += d2
        folds[f].update(glm_control_ba=float(reference.recalls(reference.confusion(yt, g)).mean()),
                        broader_control_ba=float(reference.recalls(reference.confusion(yt, b)).mean()),
                        candidate_ba=float(reference.recalls(reference.confusion(yt, c)).mean()),
                        glm_column_mass_error=glm_residuals[f], broader_column_mass_error=broad_residuals[f],
                        candidate_column_mass_error=cand_residuals[f],
                        broader_corrected=int((bg & ~gg).sum()), broader_broken=int((gg & ~bg).sum()),
                        candidate_corrected=int((cg & ~bg).sum()), candidate_broken=int((bg & ~cg).sum()))
    broad = paired_summary(y, glm_pred, broad_pred, supports, broad_delta_draws)
    candidate = paired_summary(y, broad_pred, cand_pred, supports, cand_delta_draws)
    require(broad["treatment_confusion_counts"] == candidate["control_confusion_counts"]
            and broad["corrected"] == sum(f["broader_corrected"] for f in folds)
            and broad["broken"] == sum(f["broader_broken"] for f in folds)
            and candidate["corrected"] == sum(f["candidate_corrected"] for f in folds)
            and candidate["broken"] == sum(f["candidate_broken"] for f in folds),
            "stage13d_pair_fold_reconciliation")
    broad_gates = dict(pooled_gain_at_least_half_pp=broad["gain_at_least_half_pp"],
                       conditional_ci_lower_positive=broad["ci_lower_positive"])
    cand_gates = dict(pooled_gain_at_least_half_pp=candidate["gain_at_least_half_pp"],
                      conditional_ci_lower_positive=candidate["ci_lower_positive"],
                      every_class_recall_loss_le_five_pp=candidate["class_loss_within_five_pp"])
    admitted = all(broad_gates.values())
    return dict(glm_control_ba=broad["control_ba"], broader_control_ba=broad["treatment_ba"],
                candidate_ba=candidate["treatment_ba"], broader_delta_pp=broad["delta_pp"],
                delta_pp=candidate["delta_pp"], broader_conditional_ci95_pp=broad["conditional_delta_ci95_pp"],
                conditional_delta_ci95_pp=candidate["conditional_delta_ci95_pp"],
                broader_control_admitted=admitted, passes_proxy_screen=admitted and all(cand_gates.values()),
                broader_control_criteria=broad_gates, decision_criteria=cand_gates,
                broader_vs_glm=broad, candidate_vs_broader=candidate)


def run(input_root, output_root, *, progress=None):
    started = time.monotonic()
    root, out = Path(input_root), Path(output_root)
    report = dict(schema_version=1, status=FAILED, scope=SCOPE, fit_attempted=False, fit_performed=False,
                  control_replayed=False, stage13c_replayed=False, broader_control_admitted=False,
                  passes_proxy_screen=False, official_score_estimated=False,
                  raw_rows_or_predictions_exported=False, full_s07_oof_admitted=False,
                  expected_input_version_ids=[TRAIN_VERSION, reference.SOURCE_VERSION, prior.DEBERTA_VERSION],
                  train_sha256=TRAIN_SHA, heldout_cache_sha256=reference.CACHE_SHA,
                  deberta_cache_sha256=prior.DEBERTA_SHA)
    writable = False
    try:
        require(not out.exists() or (out.is_dir() and not any(out.iterdir())), "fresh_output_directory")
        out.mkdir(parents=True, exist_ok=True)
        writable = True
        require(sklearn_version == prior.SKLEARN_VERSION, "stage13d_sklearn_version")
        ids, prompts, y, folds, classes, glm, producer = reference.cache_inputs(root)
        texts = load_texts(reference.one(root, "train.csv"), ids, prompts, y, classes)
        masks = prior.fold_masks(prompts, y, folds)
        glm_pred, _, glm_residuals = prior.replay_control(glm, y, masks, producer)
        report["control_replayed"] = True
        x, id_mode = prior.load_features(reference.one(root, "enc_debv3l2_train.npz"), ids)
        report["fit_attempted"] = True
        deb, details, deb_pools = prior.fit_oof(x, y, masks)
        del x
        replay_stage13c(glm, deb, y, masks)
        report["stage13c_replayed"] = True
        if progress is not None:
            progress("STAGE13D_STAGE13C_REPLAY_PASS", {})
        char, char_details, char_pools = fit_char_oof(
            texts, y, masks,
            on_fold=None if progress is None else lambda detail: progress("STAGE13D_CHAR_FOLD", detail))
        report["fit_performed"] = True
        for detail, extra in zip(details, char_details):
            require(detail["fold"] == extra["fold"], "stage13d_fitted_fold_order")
            detail.update(extra)
        scores = compare_arms(glm, char, deb, prompts, y, masks, glm_pred, glm_residuals, details)
        require(hashlib.sha256(glm.astype("<f8", copy=False).tobytes()).hexdigest() == reference.PROB_SHA,
                "stage13d_glm_mutated")
        report.update(status=COMPLETE, rows=ROWS, prompts=reference.PROMPTS, classes=classes.tolist(), folds=details,
                      train_ordered_text_sha256=TEXT_SHA, stage13c_candidate_confusion_sha256=STAGE13C_CM_SHA,
                      control_uniform_confusion_sha256=prior.UNIFORM_CONTROL_CM_SHA,
                      deberta_id_verification=id_mode, deberta_estimator=prior.HEAD, char_estimator=CHAR_HEAD,
                      char_vectorizer={**CHAR_TFIDF, "dtype": "float64"}, blend_weights=WEIGHTS,
                      thread_pools={"char": char_pools, "deberta": deb_pools}, sklearn_version=sklearn_version,
                      numpy_version=np.__version__, pandas_version=pd.__version__,
                      balancing={"alpha": 1, "iterations": 100, "minimum_batch_rows": 18, "fold_local": True},
                      bootstrap={"draws": reference.DRAWS, "seed": reference.SEED,
                                 "scope": "shared paired within-fold prompt resampling on fixed predictions"},
                      **scores)
    except reference.GateError as error:
        report.update(failure_code=str(error))
    except Exception as error:
        report.update(failure_code="runtime_failure", failure_type=type(error).__name__)
    report["seconds"] = round(time.monotonic() - started, 6)
    if writable:
        try:
            tmp = out / ".broader_complement_report.tmp"
            tmp.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            tmp.replace(out / "broader_complement_report.json")
        except Exception as error:
            report.update(status=FAILED, failure_code="report_write_failure", failure_type=type(error).__name__)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default="/kaggle/input")
    parser.add_argument("--output-root", default="/kaggle/working/stage13d_broader_complement")
    args = parser.parse_args()
    def progress(marker, payload):
        print(marker, json.dumps(payload, sort_keys=True, allow_nan=False), flush=True)

    report = run(args.input_root, args.output_root, progress=progress)
    print("STAGE13D_BROADER_COMPLEMENT_JSON_BEGIN", flush=True)
    print(json.dumps(report, sort_keys=True, allow_nan=False), flush=True)
    print("STAGE13D_BROADER_COMPLEMENT_JSON_END", flush=True)
    return 0 if report["status"] == COMPLETE else 2


if __name__ == "__main__":
    raise SystemExit(main())
