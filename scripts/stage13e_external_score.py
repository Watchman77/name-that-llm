"""Score a source-qualified external Name That LLM holdout, without exporting rows.

The caller must freeze/admit the holdout before running this scorer: verify
source/model identities and prompt disjointness, then bind each text, target,
group and frozen-model embedding to the same row. This module validates axes
and fits only on the exact official training passages. A result concerns the
external panel and the reduced Stage 13 arms, not the Kaggle hidden set or S07.
"""
from __future__ import annotations

import hashlib
import time
import warnings

import numpy as np
from sklearn import __version__ as sklearn_version
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_info, threadpool_limits

import stage13_glm_readout as glm_reference
import stage13b_balancing_stress as ref
import stage13c_deberta_complement as pair_reference
import stage13d_broader_complement as triple_reference

COMPLETE = "COMPLETE_STAGE13E_EXTERNAL_SCREEN"
SCOPE = "external_prompt_holdout_fixed_reduced_complement"
PROMPTS_PER_BLOCK = 30
K = ref.K
ROWS_PER_BLOCK = K * PROMPTS_PER_BLOCK
ARMS = ("glm", "pair", "triple")
WEIGHTS = {
    "glm": {"glm": 1.0},
    "pair": {"glm": 0.75, "deberta": 0.25},
    "triple": {"glm": 0.50, "deberta": 0.25, "character": 0.25},
}
require = ref.require


def _strings(x, n, code):
    require(isinstance(x, np.ndarray) and x.shape == (n,) and x.dtype.kind == "U"
            and (np.char.str_len(x) > 0).all(), code)


def _integers(x, n, code):
    require(isinstance(x, np.ndarray) and x.shape == (n,) and x.dtype.kind in "iu", code)


def _features(x, rows, width, code):
    require(isinstance(x, np.ndarray) and x.shape == (rows, width)
            and x.dtype == np.float64 and np.isfinite(x).all(), code)


def _holdout_layout(y, prompts, blocks):
    """Return full disjoint scoring masks after validating all prompt blocks."""
    require(isinstance(y, np.ndarray) and y.ndim == 1
            and len(y) >= ROWS_PER_BLOCK and len(y) % ROWS_PER_BLOCK == 0,
            "stage13e_holdout_row_count")
    n = len(y)
    _integers(y, n, "stage13e_holdout_target_axis")
    require((y >= 0).all() and (y < K).all(), "stage13e_holdout_target_index")
    y = y.astype(np.int64, copy=False)
    _strings(prompts, n, "stage13e_holdout_prompt_axis")
    _integers(blocks, n, "stage13e_holdout_block_axis")
    expected = np.arange(1, 1 + n // ROWS_PER_BLOCK)
    require(np.array_equal(np.unique(blocks), expected), "stage13e_holdout_block_index")
    _, inverse = np.unique(prompts, return_inverse=True)
    n_groups = n // K
    require(int(inverse.max()) + 1 == n_groups, "stage13e_holdout_prompt_count")
    memberships = np.bincount(inverse * K + y, minlength=n_groups * K)
    require((memberships == 1).all(), "stage13e_prompt_missing_or_repeated_class")
    seen = set()
    masks = []
    for block in expected:
        selected = blocks == block
        groups = set(inverse[selected].tolist())
        require(int(selected.sum()) == ROWS_PER_BLOCK and len(groups) == PROMPTS_PER_BLOCK
                and not seen.intersection(groups), "stage13e_holdout_prompt_crosses_block")
        seen.update(groups)
        masks.append(selected)
    require(len(seen) == n_groups, "stage13e_holdout_prompt_coverage")
    return y, masks


def _fit_head(fit_x, train_y, head, branch, score_x):
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        estimator = LogisticRegression(**head).fit(fit_x, train_y)
    require(np.array_equal(estimator.classes_, np.arange(K)), f"stage13e_{branch}_classes")
    steps = np.asarray(estimator.n_iter_)
    require(steps.shape == (1,) and 0 < int(steps[0]) < head["max_iter"],
            f"stage13e_{branch}_iterations")
    p = estimator.predict_proba(score_x)
    ref.validate_p(p)
    require(len(p) == score_x.shape[0], f"stage13e_{branch}_prediction_coverage")
    return p, int(steps[0])


def _dense_branch(train, y, holdout, head, branch):
    # Only the official training features are used to fit the scaler and estimator.
    scaler = StandardScaler().fit(train)
    return _fit_head(scaler.transform(train), y, head, branch, scaler.transform(holdout))


def _character_branch(train, y, holdout):
    vectorizer = TfidfVectorizer(**triple_reference.CHAR_TFIDF, dtype=np.float64)
    fit_x = vectorizer.fit_transform(train)
    score_x = vectorizer.transform(holdout)
    require(0 < fit_x.shape[1] <= triple_reference.CHAR_TFIDF["max_features"]
            and fit_x.shape[0] == len(train) and score_x.shape == (len(holdout), fit_x.shape[1])
            and np.isfinite(fit_x.data).all() and np.isfinite(score_x.data).all(),
            "stage13e_character_matrices")
    p, iterations = _fit_head(fit_x, y, triple_reference.CHAR_HEAD, "character", score_x)
    return p, {"iterations": iterations, "vocabulary_size": int(fit_x.shape[1]),
               "holdout_rows_with_no_features": int((np.diff(score_x.indptr) == 0).sum())}


def _compare(glm, deberta, character, y, prompts, masks):
    """Use the same matched fixed predictions in both paired bootstraps."""
    p = (glm, 0.75 * glm + 0.25 * deberta, 0.50 * glm + 0.25 * character + 0.25 * deberta)
    decisions = [np.full(len(y), -1, dtype=np.int64) for _ in ARMS]
    rng = np.random.default_rng(ref.SEED)
    support = np.zeros((ref.DRAWS, K), dtype=np.int64)
    pair_change, triple_change = np.zeros_like(support), np.zeros_like(support)
    blocks = []
    for i, mask in enumerate(masks, 1):
        by_arm = {}
        target = y[mask]
        correctness = []
        for arm, pp, out in zip(ARMS, p, decisions):
            require((out[mask] == -1).all(), "stage13e_duplicate_predictions")
            log, error = ref.evidence(pp[mask], True)
            require(error is not None, "stage13e_balancing_skipped")
            preds = log.argmax(axis=1)
            out[mask] = preds
            by_arm[arm] = dict(
                balanced_accuracy=float(ref.recalls(ref.confusion(target, preds)).mean()),
                max_column_mass_error=error,
            )
            correctness.append(preds == target)
        groups = np.unique(prompts[mask])
        require(len(groups) == PROMPTS_PER_BLOCK, "stage13e_bootstrap_group_axis")
        index = np.searchsorted(groups, prompts[mask])
        weights = rng.multinomial(
            PROMPTS_PER_BLOCK, np.full(PROMPTS_PER_BLOCK, 1 / PROMPTS_PER_BLOCK), size=ref.DRAWS
        )
        glm_ok, pair_ok, triple_ok = correctness
        s1, d1 = ref.bootstrap_delta(index, target, pair_ok, glm_ok, weights)
        s2, d2 = ref.bootstrap_delta(index, target, triple_ok, pair_ok, weights)
        require(np.array_equal(s1, s2) and (s1 == PROMPTS_PER_BLOCK).all(),
                "stage13e_bootstrap_support_mismatch")
        support += s1
        pair_change += d1
        triple_change += d2
        blocks.append({
            "block": i, "rows": int(mask.sum()), "prompts": len(groups), "arms": by_arm,
            "pair_corrected": int((pair_ok & ~glm_ok).sum()), "pair_broken": int((glm_ok & ~pair_ok).sum()),
            "triple_corrected": int((triple_ok & ~pair_ok).sum()), "triple_broken": int((pair_ok & ~triple_ok).sum()),
        })
    require(all((x >= 0).all() for x in decisions), "stage13e_prediction_coverage")
    g, c, d = decisions
    pair = triple_reference.paired_summary(y, g, c, support, pair_change)
    triple = triple_reference.paired_summary(y, c, d, support, triple_change)
    require(pair["treatment_confusion_counts"] == triple["control_confusion_counts"]
            and pair["corrected"] == sum(b["pair_corrected"] for b in blocks)
            and pair["broken"] == sum(b["pair_broken"] for b in blocks)
            and triple["corrected"] == sum(b["triple_corrected"] for b in blocks)
            and triple["broken"] == sum(b["triple_broken"] for b in blocks),
            "stage13e_pair_reconciliation")
    pair_gates = {"gain_ge_half_pp": pair["gain_at_least_half_pp"],
                  "conditional_ci_lower_positive": pair["ci_lower_positive"]}
    triple_gates = {"gain_ge_half_pp": triple["gain_at_least_half_pp"],
                    "conditional_ci_lower_positive": triple["ci_lower_positive"],
                    "every_class_loss_le_five_pp": triple["class_loss_within_five_pp"]}
    pair_admitted, candidate_marginal_pass = all(pair_gates.values()), all(triple_gates.values())
    return dict(glm_ba=pair["control_ba"], pair_ba=pair["treatment_ba"], triple_ba=triple["treatment_ba"],
                pair_vs_glm=pair, triple_vs_pair=triple, blocks=blocks,
                stronger_comparator_criteria=pair_gates, candidate_criteria=triple_gates,
                stronger_comparator_admitted=pair_admitted, candidate_marginal_pass=candidate_marginal_pass,
                passes_external_screen=pair_admitted and candidate_marginal_pass)


def score_external(*, classes, train_texts, train_targets, train_glm, train_deberta,
                   holdout_texts, holdout_targets, holdout_glm, holdout_deberta,
                   holdout_prompts, holdout_blocks, progress=None):
    """Return only aggregate diagnostics for three fixed arms on an admitted panel.

    Arrays must be row aligned: non-object NumPy Unicode for text/prompt/classes,
    integer labels in canonical class order, and finite float64 matrices of
    widths 8192 (GLM L8/L3) and 2048 (DeBERTa L24/L6). Block indices must be
    contiguous from 1, with exactly 30 complete 18-model prompt groups per
    block. No holdout text, embeddings, groups, or labels are supplied to fit.
    """
    started = time.monotonic()
    require(sklearn_version == pair_reference.SKLEARN_VERSION, "stage13e_sklearn_version")
    _strings(classes, K, "stage13e_class_axis")
    require(ref.digest_json(classes.tolist()) == ref.LABEL_SHA, "stage13e_class_order")
    _strings(train_texts, ref.ROWS, "stage13e_train_text_axis")
    require(ref.digest_json([hashlib.sha256(t.encode("utf-8")).hexdigest() for t in train_texts])
            == triple_reference.TEXT_SHA, "stage13e_ordered_train_text_digest")
    _integers(train_targets, ref.ROWS, "stage13e_train_target_axis")
    require((train_targets >= 0).all() and (train_targets < K).all(), "stage13e_train_target_index")
    train_y = train_targets.astype(np.int64, copy=False)
    require(np.bincount(train_y, minlength=K).tolist() == [149] + [150] * (K - 1),
            "stage13e_train_class_support")
    holdout_y, masks = _holdout_layout(holdout_targets, holdout_prompts, holdout_blocks)
    _strings(holdout_texts, len(holdout_y), "stage13e_holdout_text_axis")
    require(len(set(holdout_texts.tolist())) == len(holdout_y)
            and not set(train_texts.tolist()).intersection(holdout_texts.tolist()),
            "stage13e_exact_text_overlap")
    for name, width, train, held in (("glm", 8192, train_glm, holdout_glm),
                                     ("deberta", 2048, train_deberta, holdout_deberta)):
        _features(train, ref.ROWS, width, f"stage13e_{name}_train_features")
        _features(held, len(holdout_y), width, f"stage13e_{name}_holdout_features")

    def emit(marker, values):
        if progress is not None:
            progress(marker, values)

    emit("STAGE13E_SCORING_INPUTS_PASS", {"train_rows": ref.ROWS, "holdout_rows": len(holdout_y),
                                         "holdout_blocks": len(masks)})
    with threadpool_limits(limits=pair_reference.THREADS):
        pools = [{"api": x["user_api"], "threads": x["num_threads"]} for x in threadpool_info()]
        require(all(0 < p["threads"] <= pair_reference.THREADS for p in pools), "stage13e_thread_limit")
        glm, glm_steps = _dense_branch(train_glm, train_y, holdout_glm, glm_reference.HEAD, "glm")
        emit("STAGE13E_GLM_HEAD_COMPLETE", {"iterations": glm_steps})
        deberta, deb_steps = _dense_branch(
            train_deberta, train_y, holdout_deberta, pair_reference.HEAD, "deberta"
        )
        emit("STAGE13E_DEBERTA_HEAD_COMPLETE", {"iterations": deb_steps})
        char, char_detail = _character_branch(train_texts, train_y, holdout_texts)
        emit("STAGE13E_CHARACTER_HEAD_COMPLETE", dict(char_detail))
        result = _compare(glm, deberta, char, holdout_y, holdout_prompts, masks)
    return dict(
        schema_version=1, status=COMPLETE, scope=SCOPE, classes=classes.tolist(),
        official_train_rows=ref.ROWS, external_rows=len(holdout_y), external_prompts=len(holdout_y) // K,
        external_blocks=len(masks), fit_performed=True, raw_rows_or_predictions_exported=False,
        official_score_estimated=False, full_s07_oof_admitted=False,
        blend_weights={arm: dict(weights) for arm, weights in WEIGHTS.items()},
        glm_estimator=dict(glm_reference.HEAD), deberta_estimator=dict(pair_reference.HEAD),
        character_estimator=dict(triple_reference.CHAR_HEAD),
        character_vectorizer={**triple_reference.CHAR_TFIDF, "dtype": "float64"},
        head_fit={"glm": {"iterations": glm_steps}, "deberta": {"iterations": deb_steps},
                  "character": char_detail},
        balancing={"alpha": 1, "iterations": 100, "minimum_batch_rows": K, "block_local": True},
        bootstrap={"draws": ref.DRAWS, "seed": ref.SEED,
                   "scope": "shared paired within-block prompt resampling on fixed predictions"},
        thread_pools=pools, sklearn_version=sklearn_version, numpy_version=np.__version__,
        seconds=round(time.monotonic() - started, 6), **result,
    )
