#!/usr/bin/env python3
"""Frozen Stage 12A CPU component screen; real fitting requires a Git freeze.

This is an exploratory prompt-disjoint SVC diagnostic, not full S05 OOF.
No original text, IDs, predictions, margins or fitted states are published.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time
import warnings

# Set before importing NumPy/SciPy/sklearn in either parent or child.
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_key] = "1"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT_ROOT.parent
EXPECTED_TRAIN_SHA256 = "40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707"
EXPECTED_SPLIT_SHA256 = "72d7d43119367ef100511d0f8174b9483ec86bf28bfc7b7e61d8fb599ee1209d"
SOURCE_VERSION_ID = 362014737
LABELS = ["gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra", "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5", "claude-fable-5", "qwen3.8-max", "glm-5.3", "glm-5.3-flash", "kimi-k3"]
SMALL_WORDS = set("a an the and or but if while although because as so than that which who whom whose this these those it its they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from with without for of by over under between through into about is are was were be been being have has had do does did can could will would shall should may might must not no nor all any some each every both either neither other another such only also just very too more most less least here there when where why how what whether then now still yet even much many few little".split())
VIEWS = [("char", 3, 5, 250000, 1.0), ("word", 1, 3, 180000, 0.7), ("shape", 2, 6, 60000, 0.5), ("small", 1, 3, 50000, 0.5)]
CONFIG = {
    "C": 1.0, "class_weight": "balanced", "min_df": 2,
    "max_iter": 10000, "tol": 1e-4, "random_state": 42,
    "topic_threshold": 0.85, "bootstrap_draws": 10000,
    "bootstrap_seed": 12012, "hard_deadline_seconds": 900,
    "screen_minimum_gain_percentage_points": 0.5,
    "screen_maximum_class_loss_percentage_points": 5.0,
}


def numeric_imports():
    global np, pd, softmax, TfidfVectorizer
    import numpy as np
    import pandas as pd
    from scipy.special import softmax
    from sklearn.feature_extraction.text import TfidfVectorizer


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def shape_text(t):
    """Every run of letters becomes X or x, so only the rhythm is left."""
    return re.sub(r"[^\W\d_]+", lambda m: "X" if m[0].isupper() else "x", t)


def small_words_text(t):
    """Keep the small words and the punctuation, and replace every other word by CONTENT."""
    return " ".join(w.lower() if w.lower() in SMALL_WORDS or not w.isalnum() else "CONTENT"
                    for w in re.findall(r"\w+|[^\w\s]", t))


def topic_groups(texts,threshold=.85):
    from scipy.cluster.hierarchy import linkage,fcluster
    from scipy.spatial.distance import squareform
    if len(texts)<2:return np.ones(len(texts),dtype=np.int64)
    x=TfidfVectorizer(stop_words="english",sublinear_tf=True).fit_transform(texts)
    distance=np.maximum(1-(x@x.T).toarray(),0)
    np.fill_diagonal(distance,0)
    return fcluster(linkage(squareform(distance,checks=False),method="average"),threshold,criterion="distance").astype(np.int64)


def clipped_logp(logits):
    # Match the released reference's finite probability floor exactly.
    return np.log(np.clip(softmax(np.asarray(logits,dtype=np.float64),axis=1),1e-9,1))


def adaptive_svc_topic_correction(total,svc_scores,groups):
    """Roman Tamrazov .628 v9: gated removal of SVC's per-topic class bias.

    No test labels or one-per-class assignment are used. Return the unchanged
    score on every row outside topic clusters of size8..18.
    """
    scores=np.asarray(svc_scores,dtype=np.float64)
    centered=scores.copy()
    sizes=np.bincount(groups)[groups]
    valid=(sizes>=8)&(sizes<=18)
    for group in np.unique(groups[valid]):
        rows=groups==group
        centered[rows]-=scores[rows].mean(0,keepdims=True)
    delta=clipped_logp(centered)-clipped_logp(scores)
    delta[~valid]=0
    ordered=np.sort(total,axis=1)
    gate=np.clip((2.-(ordered[:,-1]-ordered[:,-2]))/1.6,.15,1.)
    gate[~valid]=0
    return total+.62*delta*gate[:,None]


def function_ast(source, name):
    function = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == name)
    # Docstrings/comments do not alter the numerical recipe.
    if function.body and isinstance(function.body[0], ast.Expr) and isinstance(function.body[0].value, ast.Constant) and isinstance(function.body[0].value.value, str):
        function.body = function.body[1:]
    return ast.dump(function, include_attributes=False)


def source_parity():
    notebook = json.loads((PROJECT_ROOT / "notebooks/Stage_11_Public_642_Reproduction.ipynb").read_text())
    cell = notebook["cells"][5]["source"]
    cell = "".join(cell) if isinstance(cell, list) else cell
    source = "\n".join(cell.splitlines()[1:])
    own = Path(__file__).read_text()
    names = ["shape_text", "small_words_text", "topic_groups", "clipped_logp", "adaptive_svc_topic_correction"]
    checks = {name: function_ast(source, name) == function_ast(own, name) for name in names}
    def assigned_value(text, name):
        return next(node.value for node in ast.parse(text).body if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name for target in node.targets))
    constants = {name: ast.dump(assigned_value(source, name), include_attributes=False) == ast.dump(assigned_value(own, name), include_attributes=False) for name in ["VIEWS", "SMALL_WORDS"]}
    if not all(checks.values()) or not all(constants.values()):
        raise ValueError("Source mathematical recipe mismatch: " + str({**checks, **constants}))
    return {"source_version": SOURCE_VERSION_ID, "source_code_cell": 5, "function_ast_matches": checks, "view_and_function_word_constants_match": constants}


def validate_frames(train, split):
    if list(train.columns) != ["id", "prompt_id", "text", "model"]:
        raise ValueError("Official training columns differ")
    if len(train) != 2699 or train.isna().any().any():
        raise ValueError("Training size or null contract differs")
    if train.id.duplicated().any() or split.id.duplicated().any():
        raise ValueError("Duplicate original IDs")
    if set(train.id) != set(split.id) or len(split) != len(train):
        raise ValueError("Canonical split ID coverage differs")
    aligned = split.set_index("id").loc[train.id].reset_index()
    if not np.array_equal(train.prompt_id.to_numpy(), aligned.prompt_id.to_numpy()) or not np.array_equal(train.model.to_numpy(), aligned.model.to_numpy()):
        raise ValueError("Canonical split prompt/model labels differ")
    folds = aligned.fold.astype(int).to_numpy()
    if set(folds) != set(range(1, 6)) or set(train.model) != set(LABELS):
        raise ValueError("Expected five folds and exact 18 classes")
    prompt_folds = pd.DataFrame({"prompt": train.prompt_id, "fold": folds}).groupby("prompt").fold.nunique()
    if len(prompt_folds) != 150 or not (prompt_folds == 1).all():
        raise ValueError("Prompt leakage across canonical folds")
    y = np.array([LABELS.index(x) for x in train.model], dtype=np.int64)
    for fold in range(1, 6):
        tr, va = folds != fold, folds == fold
        if set(y[tr]) != set(range(18)) or set(y[va]) != set(range(18)):
            raise ValueError("Missing class in a training or validation fold")
        if set(train.prompt_id[tr]) & set(train.prompt_id[va]):
            raise ValueError("Outer prompt leakage")
    return aligned, folds, y


def preflight(args):
    numeric_imports()
    actual = {"train_sha256": sha256(args.train), "split_sha256": sha256(args.splits)}
    if actual["train_sha256"] != EXPECTED_TRAIN_SHA256 or actual["split_sha256"] != EXPECTED_SPLIT_SHA256:
        raise ValueError("Frozen data or canonical split SHA256 mismatch")
    train = pd.read_csv(args.train, dtype=str, keep_default_na=False)
    split = pd.read_csv(args.splits, usecols=["id", "prompt_id", "model", "fold"], dtype=str, keep_default_na=False)
    aligned, folds, y = validate_frames(train, split)
    if (train.text.str.len() == 0).any():
        raise ValueError("Unexpected empty original passage")
    result = {"status": "PASS_INPUT_PREFLIGHT_ONLY", **actual, "rows": len(train), "prompts": train.prompt_id.nunique(), "classes": 18,
              "folds": [{"fold": int(f), "train_rows": int(sum(folds != f)), "validation_rows": int(sum(folds == f)), "validation_prompts": int(train.prompt_id[folds == f].nunique())} for f in range(1, 6)],
              "split_prediction_columns_read": False, "training_or_inference_started": False, "source_parity": source_parity()}
    return train, aligned, folds, y, result


def grouping_diagnostics(inferred, known):
    # Known prompt IDs enter only this evaluation function, never correction.
    inferred, known = np.asarray(inferred), np.asarray(known)
    _, counts = np.unique(inferred, return_counts=True)
    sizes = np.bincount(inferred)[inferred]
    eligible = (sizes >= 8) & (sizes <= 18)
    table = pd.crosstab(inferred, known).to_numpy(dtype=np.int64)
    pairs = lambda a: (a * (a - 1) // 2).sum()
    joint = int(pairs(table)); predicted = int(pairs(table.sum(1))); actual = int(pairs(table.sum(0)))
    subset_table = pd.crosstab(inferred[eligible], known[eligible]).to_numpy(dtype=np.int64) if eligible.any() else None
    return {"rows": len(inferred), "inferred_clusters": len(counts), "eligible_clusters_8_to_18": int(sum((counts >= 8) & (counts <= 18))),
            "eligible_rows": int(eligible.sum()), "eligible_fraction": float(eligible.mean()), "singleton_clusters": int(sum(counts == 1)), "oversized_clusters": int(sum(counts > 18)),
            "majority_purity_all_rows": float(table.max(1).sum() / len(inferred)),
            "majority_purity_eligible_rows": float(subset_table.max(1).sum() / eligible.sum()) if eligible.any() else None,
            "pairwise_precision": joint / predicted if predicted else None, "pairwise_recall": joint / actual if actual else None,
            "purity_interpretation": "Purity includes trivial singleton clusters; always read with coverage and pairwise recall."}


def paired_prompt_bootstrap(y, control, candidate, prompts, draws=10000, seed=12012):
    groups, inverse = np.unique(np.asarray(prompts), return_inverse=True)
    counts = np.zeros((len(groups), 18), dtype=np.int64)
    control_correct = np.zeros_like(counts); candidate_correct = np.zeros_like(counts)
    np.add.at(counts, (inverse, y), 1)
    np.add.at(control_correct, (inverse, y), (control == y).astype(np.int64))
    np.add.at(candidate_correct, (inverse, y), (candidate == y).astype(np.int64))
    rng = np.random.default_rng(seed); differences = []
    for start in range(0, draws, 500):
        selected = rng.integers(0, len(groups), size=(min(500, draws-start), len(groups)))
        denominator = counts[selected].sum(1)
        if (denominator <= 0).any():
            raise ValueError("Bootstrap draw lost a class; the frozen scheme requires every class")
        delta = (candidate_correct[selected].sum(1) - control_correct[selected].sum(1)) / denominator
        differences.extend(delta.mean(1).tolist())
    interval = np.quantile(np.asarray(differences), [.025, .975]) * 100
    return {"draws": draws, "seed": seed, "resampling_unit": "known prompt, jointly across all classes and both arms", "confidence_percent": 95,
            "difference_percentage_points_interval": interval.tolist(), "interpretation": "Conditional paired prompt percentile interval on reused development folds; not independent confirmation."}


def fit_views(train_texts, validation_texts):
    from scipy.sparse import hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    train_views, validation_views, dimensions = [], [], {}
    for view, lo, hi, maximum, weight in VIEWS:
        prep = shape_text if view == "shape" else small_words_text if view == "small" else str
        analyzer = "char" if view == "shape" else "word" if view == "small" else view
        options = dict(analyzer=analyzer, ngram_range=(lo, hi), max_features=maximum, min_df=CONFIG["min_df"], lowercase=False, sublinear_tf=True, dtype=np.float32)
        if analyzer == "word":
            options["token_pattern"] = r"(?u)\b\w+\b|[^\w\s]"
        vectorizer = TfidfVectorizer(**options)
        fitted = vectorizer.fit_transform([prep(t) for t in train_texts])
        held_out = vectorizer.transform([prep(t) for t in validation_texts])
        train_views.append(fitted * weight); validation_views.append(held_out * weight)
        dimensions[view] = int(fitted.shape[1])
    return hstack(train_views, format="csr"), hstack(validation_views, format="csr"), dimensions


def worker(args):
    if os.environ.get("STAGE12_FREEZE_COMMIT") != args.freeze_commit or not re.fullmatch(r"[0-9a-f]{40}", args.freeze_commit or ""):
        raise ValueError("Real fitting is locked until a frozen Git commit is supplied by the parent runner")
    from sklearn.svm import LinearSVC
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score
    from sklearn.exceptions import ConvergenceWarning
    began = time.monotonic()
    public = Path(args.public_dir).resolve(); private = Path(args.private_dir).resolve()
    if private.is_relative_to(PROJECT_ROOT):
        raise ValueError("Row-level output directory must be outside the repository")
    public.mkdir(parents=True, exist_ok=True); private.mkdir(parents=True, exist_ok=True)
    train, aligned, folds, y, audit = preflight(args)
    write_json(public / "PREFLIGHT.json", audit)
    events = []
    def event(stage, **data):
        record = {"stage": stage, "elapsed_seconds": time.monotonic()-began, **data}; events.append(record)
        print(json.dumps(record, allow_nan=False), flush=True)
        write_json(private / "RUN_EVENTS.json", events)
    event("INPUTS_PASSED", rows=len(train), prompts=150)
    control = np.full(len(y), -1, dtype=np.int64); candidate = control.copy()
    control_scores = np.full((len(y), 18), np.nan); candidate_scores = control_scores.copy()
    fold_results = []; grouping = []; convergence_count = 0
    for fold in range(1, 6):
        tr, va = folds != fold, folds == fold; tick = time.monotonic()
        Xtr, Xva, dimensions = fit_views(train.text[tr].tolist(), train.text[va].tolist())
        feature_seconds = time.monotonic()-tick; tick = time.monotonic()
        classifier = LinearSVC(C=CONFIG["C"], class_weight=CONFIG["class_weight"], dual="auto", max_iter=CONFIG["max_iter"], tol=CONFIG["tol"], random_state=CONFIG["random_state"])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            classifier.fit(Xtr, y[tr])
        warning_count = sum(issubclass(w.category, ConvergenceWarning) for w in caught)
        convergence_count += warning_count
        if not np.array_equal(classifier.classes_, np.arange(18)):
            raise ValueError("Fitted classifier class axis differs")
        margins = np.asarray(classifier.decision_function(Xva), np.float64)
        if margins.shape != (int(va.sum()), 18) or not np.isfinite(margins).all():
            raise ValueError("Nonfinite or malformed validation margins")
        baseline = clipped_logp(margins)
        inferred = topic_groups(train.text[va].tolist(), threshold=CONFIG["topic_threshold"])
        corrected = adaptive_svc_topic_correction(baseline, margins, inferred)
        if not np.isfinite(corrected).all():
            raise ValueError("Nonfinite candidate correction")
        control[va] = baseline.argmax(1); candidate[va] = corrected.argmax(1)
        control_scores[va] = baseline; candidate_scores[va] = corrected
        fit_and_score_seconds = time.monotonic()-tick
        diag = {"fold": fold, **grouping_diagnostics(inferred, train.prompt_id[va])}; grouping.append(diag)
        item = {"fold": fold, "training_rows": int(tr.sum()), "validation_rows": int(va.sum()), "validation_prompts": int(train.prompt_id[va].nunique()),
                "feature_dimensions": dimensions, "feature_seconds": feature_seconds, "fit_and_score_seconds": fit_and_score_seconds,
                "control_balanced_accuracy": float(balanced_accuracy_score(y[va], control[va])), "candidate_balanced_accuracy": float(balanced_accuracy_score(y[va], candidate[va])),
                "convergence_warnings": warning_count, "iterations": int(classifier.n_iter_), "one_shared_fit_for_both_arms": True}
        fold_results.append(item); event("FOLD_COMPLETE", **item)
        del Xtr, Xva, classifier
    if (control < 0).any() or (candidate < 0).any() or not np.isfinite(control_scores).all() or not np.isfinite(candidate_scores).all():
        raise ValueError("Incomplete OOF coverage")
    base_ba = float(balanced_accuracy_score(y, control)); candidate_ba = float(balanced_accuracy_score(y, candidate))
    base_recall = recall_score(y, control, labels=np.arange(18), average=None, zero_division=0)
    candidate_recall = recall_score(y, candidate, labels=np.arange(18), average=None, zero_division=0)
    bootstrap = paired_prompt_bootstrap(y, control, candidate, train.prompt_id, CONFIG["bootstrap_draws"], CONFIG["bootstrap_seed"])
    gain = (candidate_ba-base_ba)*100
    largest_loss = float(np.max(base_recall-candidate_recall)*100)
    gates = {"gain_at_least_half_percentage_point": gain >= CONFIG["screen_minimum_gain_percentage_points"],
             "paired_prompt_interval_lower_above_zero": bootstrap["difference_percentage_points_interval"][0] > 0,
             "maximum_class_recall_loss_at_most_five_points": largest_loss <= CONFIG["screen_maximum_class_loss_percentage_points"],
             "no_convergence_warnings": convergence_count == 0}
    result = {"stage": "12A", "status": "COMPLETED_COMPONENT_SCREEN", "decision": "ADMIT_TO_FURTHER_ENSEMBLE_VALIDATION" if all(gates.values()) else "REJECT_FIXED_COMPONENT_CANDIDATE",
              "freeze_commit": args.freeze_commit, "runner_sha256": sha256(__file__), "input_sha256": {"train": audit["train_sha256"], "canonical_splits": audit["split_sha256"]},
              "config": CONFIG, "views": VIEWS, "rows": len(y), "prompts": 150, "classes": 18, "folds": 5,
              "control_development_balanced_accuracy": base_ba, "candidate_development_balanced_accuracy": candidate_ba, "difference_percentage_points": gain,
              "bootstrap": bootstrap, "screen_gates": gates, "corrected_rows": int(((control != y) & (candidate == y)).sum()), "broken_rows": int(((control == y) & (candidate != y)).sum()),
              "largest_class_recall_loss_percentage_points": largest_loss, "convergence_warnings": convergence_count,
              "fit_scope": "Outer prompt-disjoint official training folds; one SVC fit shared across arms; inferred validation topics use only validation text",
              "C_and_min_df_scope": "Fixed diagnostic C=1 and min_df=2; original S05 SVC fit hyperparameters not verified. This is source-style, not an exact S05 SVC reproduction.",
              "official_public_score": None, "full_S05_oof": False, "GPU_used": False, "external_training_rows": 0, "paid_services_used": False,
              "row_level_outputs_published": False, "fitted_states_serialized": False, "elapsed_worker_seconds": time.monotonic()-began,
              "environment": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scipy", "scikit-learn"]},
              "limitations": ["Component-only screen, not a measured gain to full S05 or a public leaderboard score.", "Canonical development folds have been reused; prospective settings do not make this independent confirmation.", "Other S05 members already use centering, so any isolated gain may not complement the full ensemble.", "No test-time class balancing, external pool, full-fit supervised checkpoint or hyperparameter search is included."]}
    for name, predictions in [("control", control), ("candidate", candidate)]:
        pd.DataFrame(confusion_matrix(y, predictions, labels=np.arange(18)), index=LABELS, columns=LABELS).rename_axis("true_model").to_csv(public / (name + "_confusion.csv"))
    pd.DataFrame({"model": LABELS, "support": np.bincount(y, minlength=18), "control_recall": base_recall, "candidate_recall": candidate_recall, "difference_percentage_points": (candidate_recall-base_recall)*100}).to_csv(public / "per_class_recall.csv", index=False)
    write_json(public / "FOLDS.json", fold_results); write_json(public / "GROUP_DIAGNOSTICS.json", grouping)
    private_oof = aligned[["id", "prompt_id", "model"]].copy(); private_oof["fold"] = folds
    private_oof["control"] = np.asarray(LABELS)[control]; private_oof["candidate"] = np.asarray(LABELS)[candidate]
    private_oof.to_csv(private / "OOF_predictions.csv", index=False)
    np.savez_compressed(private / "OOF_component_scores.npz", ids=train.id.to_numpy(dtype=str), labels=np.asarray(LABELS), folds=folds, control=control_scores, candidate=candidate_scores)
    write_json(public / "RESULT.json", result)
    write_json(public / "STATUS.json", {"stage": "12A", "phase": "COMPLETED_COMPONENT_SCREEN", "decision": result["decision"], "development_result": "RESULT.json", "official_submission": False, "GPU_used": False, "full_S05_oof": False, "row_level_outputs_published": False})
    event("COMPLETE", decision=result["decision"], control_development_balanced_accuracy=base_ba, candidate_development_balanced_accuracy=candidate_ba, difference_percentage_points=gain)


def terminate_group(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


def run_with_deadline(args):
    if not re.fullmatch(r"[0-9a-f]{40}", args.freeze_commit or ""):
        raise ValueError("--run requires the actual pre-execution --freeze-commit SHA")
    private = Path(args.private_dir).resolve()
    if private.is_relative_to(PROJECT_ROOT):
        raise ValueError("Private row outputs must remain outside the repository")
    private.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--freeze-commit", args.freeze_commit, "--train", str(Path(args.train).resolve()), "--splits", str(Path(args.splits).resolve()), "--public-dir", str(Path(args.public_dir).resolve()), "--private-dir", str(private)]
    env = dict(os.environ, STAGE12_FREEZE_COMMIT=args.freeze_commit)
    began = time.monotonic(); process = subprocess.Popen(command, env=env, start_new_session=True)
    timed_out = False
    try:
        process.wait(timeout=CONFIG["hard_deadline_seconds"])
    except subprocess.TimeoutExpired:
        timed_out = True; terminate_group(process)
    except BaseException:
        terminate_group(process); raise
    receipt = {"stage": "12A", "freeze_commit": args.freeze_commit, "runner_sha256": sha256(__file__), "hard_deadline_seconds": CONFIG["hard_deadline_seconds"], "elapsed_parent_seconds": time.monotonic()-began, "timed_out": timed_out, "returncode": process.returncode, "process_group_terminated": timed_out, "status": "SUCCEEDED" if process.returncode == 0 and not timed_out else "TIMEOUT" if timed_out else "FAILED"}
    write_json(Path(args.public_dir) / "EXECUTION.json", receipt)
    if receipt["status"] != "SUCCEEDED":
        write_json(Path(args.public_dir) / "STATUS.json", {"stage": "12A", "phase": receipt["status"], "complete_development_result": False, "partial_folds_not_a_result": True, "new_official_submission": False})
    print(json.dumps(receipt), flush=True)
    return 0 if receipt["status"] == "SUCCEEDED" else 2


def self_test():
    numeric_imports(); checks = {}
    parity = source_parity(); checks["five_source_function_mathematical_AST_matches"] = all(parity["function_ast_matches"].values())
    scores = np.arange(27*18, dtype=float).reshape(27, 18) / 100
    baseline = clipped_logp(scores)
    groups = np.array([1]*7 + [2]*8 + [3]*12)
    corrected = adaptive_svc_topic_correction(baseline, scores, groups)
    checks["ineligible_seven_row_cluster_unchanged"] = np.array_equal(corrected[:7], baseline[:7])
    centered = scores[7:15] - scores[7:15].mean(0)
    gate = np.clip((2-(np.sort(baseline[7:15], axis=1)[:, -1]-np.sort(baseline[7:15], axis=1)[:, -2]))/1.6, .15, 1)
    manual = baseline[7:15] + .62*(clipped_logp(centered)-baseline[7:15])*gate[:, None]
    checks["eligible_eight_row_correction_matches_independent_formula"] = np.allclose(corrected[7:15], manual, atol=1e-14)
    nineteen = np.ones(19, dtype=int)
    checks["nineteen_row_cluster_unchanged"] = np.array_equal(adaptive_svc_topic_correction(baseline[:19], scores[:19], nineteen), baseline[:19])
    checks["singleton_group_unchanged"] = np.array_equal(adaptive_svc_topic_correction(baseline, scores, np.arange(1, 28)), baseline)
    checks["finite_probability_floor_for_extreme_logits"] = np.isfinite(clipped_logp(np.array([[1e4, -1e4]+[0]*16]))).all()
    y = np.tile(np.arange(18), 3); prompts = np.repeat(np.arange(3), 18)
    perfect = paired_prompt_bootstrap(y, y, y, prompts, draws=100, seed=4)
    checks["paired_bootstrap_identical_arms_zero_interval"] = perfect["difference_percentage_points_interval"] == [0., 0.]
    wrong = (y+1)%18
    lift = paired_prompt_bootstrap(y, wrong, y, prompts, draws=100, seed=4)
    checks["paired_bootstrap_perfect_lift_hundred_points"] = np.allclose(lift["difference_percentage_points_interval"], [100, 100])
    diag = grouping_diagnostics(np.repeat([1, 2], 9), np.repeat(["a", "b"], 9))
    checks["grouping_pure_pairwise_and_coverage"] = diag["majority_purity_all_rows"] == diag["pairwise_precision"] == diag["pairwise_recall"] == diag["eligible_fraction"] == 1
    mixed = grouping_diagnostics(np.ones(18, dtype=int), np.repeat(["a", "b"], 9))
    checks["grouping_impurity_is_diagnostic_not_hidden"] = mixed["majority_purity_all_rows"] == .5 and mixed["pairwise_precision"] < 1
    try:
        empty = pd.DataFrame(columns=["id", "prompt_id", "text", "model"])
        validate_frames(empty, empty)
    except ValueError:
        checks["reject_incomplete_canonical_frame"] = True
    else:
        checks["reject_incomplete_canonical_frame"] = False
    synthetic = pd.DataFrame({"id": [f"synthetic-{i}" for i in range(2699)], "prompt_id": [f"prompt-{i//18}" for i in range(2699)], "text": ["synthetic text"]*2699, "model": [LABELS[i%18] for i in range(2699)]})
    split = synthetic[["id", "prompt_id", "model"]].copy()
    split["fold"] = [str(1+(i//18)%5) for i in range(2699)]
    validate_frames(synthetic, split)
    checks["accept_complete_synthetic_prompt_disjoint_canonical_frame"] = True
    for name, mutation in [
        ("reject_prompt_crossing_outer_folds", lambda x: x.__setitem__("fold", ["2"]+x.fold.tolist()[1:])),
        ("reject_mismatched_canonical_label", lambda x: x.__setitem__("model", [LABELS[1]]+x.model.tolist()[1:])),
        ("reject_duplicate_canonical_id", lambda x: x.__setitem__("id", [x.id.iloc[1]]+x.id.tolist()[1:])),
    ]:
        corrupt = split.copy(); mutation(corrupt)
        try:
            validate_frames(synthetic, corrupt)
        except ValueError:
            checks[name] = True
        else:
            checks[name] = False
    try:
        run_with_deadline(argparse.Namespace(freeze_commit=None))
    except ValueError:
        checks["real_run_locked_without_freeze_commit"] = True
    else:
        checks["real_run_locked_without_freeze_commit"] = False
    checks["thread_limits_one"] = all(os.environ[k] == "1" for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"])
    # Actual process-group termination check; no classifier or real data is fitted.
    child = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(30)"], start_new_session=True)
    terminate_group(child)
    checks["watchdog_process_group_is_terminated"] = child.poll() is not None
    result = {"stage": "12A", "status": "PASS" if all(checks.values()) else "FAIL", "passed": sum(bool(x) for x in checks.values()), "total": len(checks), "checks": {k: bool(v) for k,v in checks.items()}, "real_data_fitted": False, "toy_classifier_fitted": False, "upstream_pipeline_executed": False, "source_parity": parity}
    if not all(checks.values()):
        raise AssertionError(json.dumps(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true"); mode.add_argument("--self-test", action="store_true"); mode.add_argument("--run", action="store_true"); mode.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--freeze-commit")
    parser.add_argument("--train", default=str(WORKSPACE / "stage07_private/train.csv"))
    parser.add_argument("--splits", default=str(PROJECT_ROOT / "results/stage_07b/oof_predictions.csv"))
    parser.add_argument("--public-dir", default=str(PROJECT_ROOT / "results/stage_12a"))
    parser.add_argument("--private-dir", default=str(WORKSPACE / "stage12_private/stage12a"))
    parser.add_argument("--receipt", help="Optional aggregate preflight or self-test JSON destination")
    args = parser.parse_args()
    if args.worker:
        worker(args); return 0
    if args.run:
        return run_with_deadline(args)
    result = self_test() if args.self_test else preflight(args)[-1]
    if args.receipt:
        write_json(args.receipt, result)
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
