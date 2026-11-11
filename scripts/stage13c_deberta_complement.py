"""Stage 13C: one fixed DeBERTa readout and paired, fold-local GLM comparison."""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import io
import json
from pathlib import Path
import pickletools
import time
import warnings
import zipfile

import numpy as np
from sklearn import __version__ as sklearn_version
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_info, threadpool_limits

import stage13b_balancing_stress as reference


COMPLETE = "COMPLETE_GLM_DEBERTA_COMPLEMENT_SCREEN"
FAILED = "FAILED_GLM_DEBERTA_COMPLEMENT_SCREEN"
SCOPE = "glm_deberta_paired_fold_local_development"
DEBERTA_VERSION = 358745836
DEBERTA_SHA = "11eed58762ad128f97457dde9cc4ab03c7932e1b4b04845ec6664c834b1b1dda"
DEBERTA_BYTES = 99537639
RAW_CONTROL_CM_SHA = "fa3ced3a896ce9389bf0427a42a30d8e7a30089c792cb92b8988953204b98747"
UNIFORM_CONTROL_CM_SHA = "e81a80974d67dc60dc93faf3144dff56ab2055a327abcf7805dbb3556f364b0b"
UNIFORM_CONTROL_BA = 0.4608252547849864
SKLEARN_VERSION = "1.6.1"
THREADS = 2
HEAD = dict(C=0.005, solver="lbfgs", max_iter=5000, tol=1e-4, fit_intercept=True, random_state=0)
K, ROWS, FOLDS = reference.K, reference.ROWS, reference.FOLDS
require = reference.require


def header(stream):
    """Check the NPY metadata before array materialization."""
    version = np.lib.format.read_magic(stream)
    require(version in ((1, 0), (2, 0)), "unsupported_deberta_npy_version")
    read = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
    return read(stream, max_header_size=4096)


def checked_ids(blob, expected):
    """Verify the frozen ID member without deserializing any object array."""
    stream = io.BytesIO(blob)
    shape, fortran, dtype = header(stream)
    require(shape == expected.shape and not fortran, "deberta_ids_shape_or_order")
    if not dtype.hasobject:
        require(dtype.kind in "SU", "deberta_ids_numeric_or_invalid_dtype")
        actual = np.load(io.BytesIO(blob), allow_pickle=False).astype(str)
        mode = "numpy_nonobject"
    else:
        require(dtype == np.dtype(object), "deberta_ids_object_dtype")
        payload = stream.read()
        id_set = set(expected.tolist())
        literals = []
        # The byte hash is checked before opening the archive. genops parses
        # bytecode, never imports symbols or executes a pickle instruction.
        last = None
        for n, (op, value, pos) in enumerate(pickletools.genops(payload), 1):
            require(n <= 100000, "deberta_ids_opcode_limit")
            last = (op.name, pos)
            if isinstance(value, bytes):
                try:
                    value = value.decode("ascii")
                except UnicodeDecodeError:
                    continue
            if isinstance(value, str) and value in id_set:
                literals.append(value)
        require(last == ("STOP", len(payload) - 1), "deberta_ids_pickle_trailing_data")
        actual = np.asarray(literals, dtype=str)
        mode = "object_npy_static_unique_ID_literals_no_unpickling"
    require(np.array_equal(actual, expected), "deberta_ids_order_mismatch")
    return mode


def load_features(path, ids):
    with path.open("rb") as f:
        require(path.stat().st_size == DEBERTA_BYTES, "deberta_cache_size_mismatch")
        require(hashlib.file_digest(f, "sha256").hexdigest() == DEBERTA_SHA, "deberta_cache_hash_mismatch")
        f.seek(0)
        with zipfile.ZipFile(f) as z:
            for key, limit in (("ids.npy", 2_000_000), ("layers.npy", 4096),
                               ("emb.npy", ROWS * 9 * 1024 * 2 + 4096)):
                require(z.namelist().count(key) == 1 and z.getinfo(key).file_size < limit,
                        "deberta_cache_member_bounds")
            mode = checked_ids(z.read("ids.npy"), ids)
            with z.open("emb.npy") as member:
                shape, _, dtype = header(member)
                require(shape == (ROWS, 9, 1024) and dtype == np.dtype(np.float16), "deberta_emb_shape_or_dtype")
            with z.open("layers.npy") as member:
                shape, _, dtype = header(member)
                require(shape == (9,) and dtype.kind in "iu", "deberta_layers_shape_or_dtype")
        f.seek(0)
        with np.load(f, allow_pickle=False) as z:
            layers, emb = z["layers"], z["emb"]
            require(layers.shape == (9,) and layers.dtype.kind in "iu" and len(np.unique(layers)) == 9,
                    "deberta_layers_invalid")
            require(all((layers == layer).sum() == 1 for layer in (24, 6)), "deberta_missing_fixed_layers")
            require(emb.shape == (ROWS, 9, 1024) and emb.dtype == np.float16 and np.isfinite(emb).all(),
                    "deberta_emb_validation")
            x = np.ascontiguousarray(np.concatenate([emb[:, np.flatnonzero(layers == layer).item()]
                                                     for layer in (24, 6)], axis=1), dtype=np.float64)
    require(x.shape == (ROWS, 2048) and np.isfinite(x).all(), "deberta_selected_features")
    return x, mode


def fold_masks(prompts, y, folds):
    """Validate all fitting/scoring boundaries before fitting any head."""
    require(prompts.shape == y.shape == folds.shape == (ROWS,), "complement_axis_shape")
    masks = []
    for f in range(1, FOLDS + 1):
        val = folds == f
        fit_prompts, val_prompts = np.unique(prompts[~val]), np.unique(prompts[val])
        require(len(fit_prompts) == 120 and len(val_prompts) == 30
                and len(np.intersect1d(fit_prompts, val_prompts)) == 0
                and int(val.sum()) == (540 if f < 5 else 539), "complement_fold_exclusion")
        require(all(np.array_equal(np.unique(y[m]), np.arange(K)) for m in (val, ~val)),
                "complement_fold_class_coverage")
        masks.append(val)
    require(np.asarray(masks).sum(axis=0).tolist() == [1] * ROWS, "complement_fold_cover")
    return masks


def fit_oof(x, y, masks):
    require(x.ndim == 2 and len(x) == ROWS and x.dtype == np.float64 and np.isfinite(x).all(),
            "complement_fit_features")
    p = np.full((ROWS, K), np.nan, dtype=np.float64)
    filled = np.zeros(ROWS, dtype=np.int8)
    details = []
    with threadpool_limits(limits=THREADS):
        pools = [{"api": item["user_api"], "threads": item["num_threads"]} for item in threadpool_info()]
        require(all(0 < pool["threads"] <= THREADS for pool in pools), "complement_thread_limit")
        for fold, val in enumerate(masks, 1):
            started = time.monotonic()
            scaler = StandardScaler().fit(x[~val])
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                estimator = LogisticRegression(**HEAD).fit(scaler.transform(x[~val]), y[~val])
            require(np.array_equal(estimator.classes_, np.arange(K)), "complement_estimator_class_axis")
            niter = [int(n) for n in estimator.n_iter_]
            require(len(niter) == 1 and 0 < niter[0] < HEAD["max_iter"], "complement_iteration_limit")
            scores = estimator.predict_proba(scaler.transform(x[val]))
            reference.validate_p(scores)
            require(len(scores) == val.sum() and (filled[val] == 0).all(), "complement_duplicate_or_missing_rows")
            p[val], filled[val] = scores, 1
            details.append(dict(fold=fold, iterations=niter[0], fit_rows=int((~val).sum()),
                                valid_rows=int(val.sum()), fit_prompts=120, valid_prompts=30,
                                prompt_overlap=0, seconds=round(time.monotonic() - started, 6)))
    require((filled == 1).all() and len(details) == FOLDS, "complement_oof_cover")
    reference.validate_p(p)
    return p, details, pools


def replay_control(p, y, masks, producer):
    """Require the full prior confusion evidence before fitting anything."""
    predicted = np.full(ROWS, -1, dtype=np.int64)
    residuals = []
    for mask in masks:
        scores, residual = reference.evidence(p[mask], True)
        predicted[mask] = scores.argmax(axis=1)
        residuals.append(residual)
    require((predicted >= 0).all(), "control_oof_cover")
    raw = reference.confusion(y, reference.evidence(p, False)[0].argmax(axis=1))
    uniform = reference.confusion(y, predicted)
    require(raw.tolist() == producer["confusion_counts"]
            and reference.digest_json(raw.tolist()) == RAW_CONTROL_CM_SHA
            and abs(float(reference.recalls(raw).mean()) - reference.EXPECTED_RECALL) < 1e-15,
            "complement_raw_control_confusion_mismatch")
    require(reference.digest_json(uniform.tolist()) == UNIFORM_CONTROL_CM_SHA
            and abs(float(reference.recalls(uniform).mean()) - UNIFORM_CONTROL_BA) < 1e-15,
            "complement_uniform_control_confusion_mismatch")
    return predicted, uniform, residuals


def paired_scores(glm, deb, prompts, y, masks, control, control_cm, residuals, folds):
    """Apply the same frozen balancing to full, separate scoring folds."""
    candidate = np.full(ROWS, -1, dtype=np.int64)
    alone = np.full(ROWS, -1, dtype=np.int64)
    rng = np.random.default_rng(reference.SEED)
    bootstrap_support = np.zeros((reference.DRAWS, K), dtype=np.int64)
    bootstrap_diff = np.zeros_like(bootstrap_support)
    for f, mask in enumerate(masks):
        pp = 0.75 * glm[mask] + 0.25 * deb[mask]
        scores, candidate_residual = reference.evidence(pp, True)
        candidate[mask] = scores.argmax(axis=1)
        alone_scores, _ = reference.evidence(deb[mask], True)
        alone[mask] = alone_scores.argmax(axis=1)
        yt, gc, mix = y[mask], control[mask], candidate[mask]
        cg, cc = reference.confusion(yt, gc), reference.confusion(yt, mix)
        cgood, ggood = mix == yt, gc == yt
        ordered = np.unique(prompts[mask])
        weights = rng.multinomial(30, np.full(30, 1 / 30), size=reference.DRAWS)
        support, diff = reference.bootstrap_delta(np.searchsorted(ordered, prompts[mask]), yt, cgood, ggood, weights)
        bootstrap_support += support
        bootstrap_diff += diff
        folds[f].update(control_ba=float(reference.recalls(cg).mean()),
                        candidate_ba=float(reference.recalls(cc).mean()),
                        control_column_mass_error=residuals[f],
                        candidate_column_mass_error=candidate_residual,
                        corrected_by_candidate=int((cgood & ~ggood).sum()),
                        broken_by_candidate=int((ggood & ~cgood).sum()))
    require((bootstrap_support > 0).all(), "complement_bootstrap_missing_class")
    require((candidate >= 0).all() and (alone >= 0).all(), "complement_predictions_incomplete")
    cand_cm, alone_cm = reference.confusion(y, candidate), reference.confusion(y, alone)
    raw_deb_cm = reference.confusion(y, reference.evidence(deb, False)[0].argmax(axis=1))
    controls, candidates = reference.recalls(control_cm), reference.recalls(cand_cm)
    supports = control_cm.sum(axis=1)
    require(np.array_equal(supports, cand_cm.sum(axis=1)), "complement_support_mismatch")
    exact = [Fraction(int(cand_cm[i, i]) - int(control_cm[i, i]), int(supports[i])) * 100 for i in range(K)]
    delta = sum(exact) / K
    ci = np.percentile((bootstrap_diff / bootstrap_support).mean(axis=1) * 100, [2.5, 97.5]).tolist()
    corrected = int(((candidate == y) & (control != y)).sum())
    broken = int(((control == y) & (candidate != y)).sum())
    changed = int((candidate != control).sum())
    require(corrected + broken <= changed
            and corrected - broken == int(cand_cm.trace() - control_cm.trace())
            and corrected == sum(f["corrected_by_candidate"] for f in folds)
            and broken == sum(f["broken_by_candidate"] for f in folds), "complement_paired_count_mismatch")
    criteria = dict(pooled_gain_at_least_half_pp=delta >= Fraction(1, 2),
                    conditional_ci_lower_positive=ci[0] > 0,
                    every_class_recall_loss_le_five_pp=min(exact) >= -5)
    return dict(control_ba=float(controls.mean()), candidate_ba=float(candidates.mean()),
                deberta_alone_ba=float(reference.recalls(alone_cm).mean()),
                deberta_alone_raw_ba=float(reference.recalls(raw_deb_cm).mean()),
                delta_pp=float(delta), delta_pp_exact=str(delta), conditional_delta_ci95_pp=ci,
                class_supports=supports.tolist(), class_delta_pp=[float(v) for v in exact],
                control_class_recalls=controls.tolist(), candidate_class_recalls=candidates.tolist(),
                control_confusion_counts=control_cm.tolist(), candidate_confusion_counts=cand_cm.tolist(),
                deberta_alone_confusion_counts=alone_cm.tolist(),
                deberta_alone_raw_confusion_counts=raw_deb_cm.tolist(),
                corrected_by_candidate=corrected, broken_by_candidate=broken, changed_by_candidate=changed,
                decision_criteria=criteria, passes_proxy_screen=all(criteria.values()))


def run(input_root, output_root):
    started = time.monotonic()
    root, out = Path(input_root), Path(output_root)
    report = dict(schema_version=1, status=FAILED, scope=SCOPE, fit_attempted=False, fit_performed=False,
                  raw_rows_or_predictions_exported=False, official_score_estimated=False,
                  full_s07_oof_admitted=False, expected_input_version_ids=[reference.SOURCE_VERSION, DEBERTA_VERSION],
                  heldout_cache_sha256=reference.CACHE_SHA, deberta_cache_sha256=DEBERTA_SHA)
    writable = False
    try:
        require(not out.exists() or (out.is_dir() and not any(out.iterdir())), "fresh_output_directory")
        out.mkdir(parents=True, exist_ok=True)
        writable = True
        require(sklearn_version == SKLEARN_VERSION, "complement_sklearn_version")
        ids, prompts, y, folds, classes, glm, producer = reference.cache_inputs(root)
        masks = fold_masks(prompts, y, folds)
        control, cm, residuals = replay_control(glm, y, masks, producer)
        report["control_replayed"] = True
        features, id_mode = load_features(reference.one(root, "enc_debv3l2_train.npz"), ids)
        report["fit_attempted"] = True
        deb, details, pools = fit_oof(features, y, masks)
        report["fit_performed"] = True
        scores = paired_scores(glm, deb, prompts, y, masks, control, cm, residuals, details)
        require(hashlib.sha256(glm.astype("<f8", copy=False).tobytes()).hexdigest() == reference.PROB_SHA,
                "complement_glm_mutated")
        report.update(status=COMPLETE, rows=ROWS, prompts=reference.PROMPTS, classes=classes.tolist(), folds=details,
                      estimator=HEAD, thread_pools=pools, feature_layers=[24, 6], feature_dtype="float16_cached",
                      deberta_id_verification=id_mode, sklearn_version=sklearn_version, numpy_version=np.__version__,
                      control_raw_confusion_sha256=RAW_CONTROL_CM_SHA, control_uniform_confusion_sha256=UNIFORM_CONTROL_CM_SHA,
                      mixture_weights={"glm": 0.75, "deberta": 0.25},
                      balancing={"alpha": 1, "iterations": 100, "minimum_batch_rows": 18, "fold_local": True},
                      bootstrap={"draws": reference.DRAWS, "seed": reference.SEED,
                                 "scope": "conditional stratified paired prompt resampling on fixed predictions"},
                      **scores)
    except reference.GateError as error:
        report.update(failure_code=str(error))
    except Exception as error:
        report.update(failure_code="runtime_failure", failure_type=type(error).__name__)
    report["seconds"] = round(time.monotonic() - started, 6)
    if writable:
        try:
            tmp = out / ".complement_report.tmp"
            tmp.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            tmp.replace(out / "complement_report.json")
        except Exception as error:
            report.update(status=FAILED, failure_code="report_write_failure", failure_type=type(error).__name__)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default="/kaggle/input")
    parser.add_argument("--output-root", default="/kaggle/working/stage13c_deberta_complement")
    args = parser.parse_args()
    result = run(args.input_root, args.output_root)
    print("STAGE13C_GLM_DEBERTA_JSON_BEGIN", flush=True)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    print("STAGE13C_GLM_DEBERTA_JSON_END", flush=True)
    return 0 if result["status"] == COMPLETE else 2


if __name__ == "__main__":
    raise SystemExit(main())
