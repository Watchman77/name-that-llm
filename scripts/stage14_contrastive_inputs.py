"""Admit exact frozen features and refit the sparse Stage 13C control on Q8 prompt folds.

Only the published split declarations are used from Q8. Its weights, native
scores and reference archive are neither opened nor required by this module.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import stage13a_cache_admission as glm_admission
import stage13_glm_readout as glm_readout
import stage13c_deberta_complement as deberta_readout
from stage14_native_input_audit import stable_hash
from stage14_native_oof_audit import ID_SHA, ROWS, PROMPTS, FOLDS, split_axes


GLM_OUTPUT_VERSION = 359405443
DEBERTA_OUTPUT_VERSION = 358745836
COMPETITION_ID = 170484
Q8_HUB_VERSION_ORDINAL = 22
Q8_TREE = "ntl-native-q8-multidepth-fivefold-0679"
SPLIT_FILES_SHA = "bf073280a69cf3d427b14e2c057dd5c7b5199051cdbdc8e3c49fcf0926cd4bad"
PROMPT_FOLD_SHA = "449f2facc165d51d1898d1225eba8a30fa4a280d126722376a6704e6950d4b36"
CLASSES = glm_admission.CLASSES
CLASS_SHA = "088b14188775e41a314f269da3af74f96144a56caa242ccbeacc9e3c817a7e08"
CONTROL_WEIGHTS = (0.75, 0.25)
GLM_WIDTH, DEBERTA_WIDTH = 8192, 2048
THREADS = 2
check = glm_admission.check


@dataclass(frozen=True)
class Bundle:
    x: np.ndarray
    ids: np.ndarray
    y: np.ndarray
    prompts: np.ndarray
    texts: np.ndarray
    folds: np.ndarray
    classes: tuple[str, ...]
    evidence: dict

    @property
    def glm(self):
        return self.x[:, :GLM_WIDTH]

    @property
    def deberta(self):
        return self.x[:, GLM_WIDTH:]


def unique_mount(root, name, *, directory=False):
    candidates = {p.resolve() for p in root.rglob(name)
                  if (p.is_dir() if directory else p.is_file())}
    check(len(candidates) == 1, "stage14_unique_" + name.replace(".", "_"))
    return next(iter(candidates))


def read_train(path):
    with path.open("rb") as handle:
        check(hashlib.file_digest(handle, "sha256").hexdigest() == glm_admission.TRAIN_SHA256,
              "stage14_official_train_sha256")
        handle.seek(0)
        train = pd.read_csv(handle, dtype=str)
    check(train.shape == (ROWS, 4) and set(train.columns) == {"id", "prompt_id", "model", "text"}
          and not train.isna().any().any(), "stage14_official_train_schema")
    ids, prompts, texts, labels = (train[c].to_numpy(dtype=str) for c in ("id", "prompt_id", "text", "model"))
    check(all((np.char.str_len(axis) > 0).all() for axis in (ids, prompts, texts, labels))
          and len(np.unique(ids)) == ROWS and len(np.unique(prompts)) == PROMPTS,
          "stage14_official_train_axes")
    check(glm_admission.json_hash(ids.tolist()) == ID_SHA and set(labels) == set(CLASSES)
          and stable_hash(list(CLASSES)) == CLASS_SHA and len(set(zip(prompts, labels))) == ROWS,
          "stage14_official_train_axis_identity")
    lookup = {label: i for i, label in enumerate(CLASSES)}
    y = np.fromiter((lookup[label] for label in labels), dtype=np.int64, count=ROWS)
    check(np.bincount(y, minlength=len(CLASSES)).tolist() == [149] + [150] * 17,
          "stage14_official_class_supports")
    text_hashes = np.asarray([hashlib.sha256(t.encode("utf-8")).hexdigest() for t in texts], dtype=str)
    check(glm_admission.json_hash(text_hashes.tolist()) == glm_admission.ORDERED_TEXT_SHA256,
          "stage14_ordered_train_text_sha256")
    return ids, prompts, texts, y, text_hashes


def read_folds(assets, prompts, y):
    split_files = {}
    folds, mapping, counts, official_axes, canonical = split_axes(assets, prompts, split_files)
    check(stable_hash(split_files) == SPLIT_FILES_SHA and stable_hash(mapping) == PROMPT_FOLD_SHA
          and official_axes == [True] * FOLDS and canonical is False, "stage14_q8_split_pin")
    check([row["validation_rows"] for row in counts] == [540, 540, 540, 539, 540],
          "stage14_q8_split_support")
    _, summaries = glm_admission.validate_fold_partition(prompts, folds + 1, FOLDS)
    for f, row in enumerate(summaries):
        row["fold"] = f
        check(row["fit_prompts"] == 120 and row["valid_prompts"] == 30
              and row["valid_rows"] == counts[f]["validation_rows"], "stage14_fold_counts")
        for mask in (folds == f, folds != f):
            check(np.array_equal(np.unique(y[mask]), np.arange(len(CLASSES))), "stage14_fold_class_coverage")
    return folds, summaries


def load_glm(path, ids, text_hashes, output):
    contract = glm_admission.PRODUCTION
    with path.open("rb") as handle:
        check(hashlib.file_digest(handle, "sha256").hexdigest() == contract.glm_sha256,
              "stage14_glm_cache_sha256")
        handle.seek(0)
        glm_admission.npz_headers(handle, contract)
        handle.seek(0)
        with np.load(handle, allow_pickle=False) as cache:
            check(np.array_equal(cache["ids"], ids) and np.array_equal(cache["text_sha256"], text_hashes)
                  and cache["train_sha256"].item() == contract.train_sha256
                  and np.array_equal(cache["layers"], contract.layers), "stage14_glm_cache_axis_binding")
            lengths = cache["token_lengths"]
            check(((lengths > 0) & (lengths <= 512)).all(), "stage14_glm_cache_token_lengths")
            feature = cache["features"]
            check(all(np.isfinite(feature[i:i + 256]).all() for i in range(0, ROWS, 256)),
                  "stage14_glm_cache_feature_finite")
            output[:, :4096], output[:, 4096:] = feature[:, 0, :], feature[:, 1, :]


def load(input_root):
    root = Path(input_root)
    check(root.is_dir(), "stage14_input_root")
    train = unique_mount(root, "train.csv")
    glm = unique_mount(root, "glm_features.npz")
    deberta = unique_mount(root, "enc_debv3l2_train.npz")
    assets = unique_mount(root, Q8_TREE, directory=True)
    ids, prompts, texts, y, text_hashes = read_train(train)
    folds, fold_summary = read_folds(assets, prompts, y)
    x = np.empty((ROWS, GLM_WIDTH + DEBERTA_WIDTH), dtype=np.float32)
    load_glm(glm, ids, text_hashes, x[:, :GLM_WIDTH])
    deb, id_mode = deberta_readout.load_features(deberta, ids)
    x[:, GLM_WIDTH:] = deb
    del deb
    # The GLM source is f32 and DeBERTa f16, so this layout keeps every input
    # value exact; control fitting promotes its slices to f64 as before.
    for axis in (x, ids, y, prompts, texts, folds):
        axis.flags.writeable = False
    evidence = dict(rows=ROWS, prompts=PROMPTS, classes=list(CLASSES),
                    class_sha256=CLASS_SHA, official_id_sha256=ID_SHA,
                    train_sha256=glm_admission.TRAIN_SHA256,
                    glm_cache_sha256=glm_admission.GLM_SHA256,
                    deberta_cache_sha256=deberta_readout.DEBERTA_SHA,
                    glm_output_version=GLM_OUTPUT_VERSION, deberta_output_version=DEBERTA_OUTPUT_VERSION,
                    q8_hub_version_ordinal=Q8_HUB_VERSION_ORDINAL, q8_split_files_sha256=SPLIT_FILES_SHA,
                    q8_prompt_fold_map_sha256=PROMPT_FOLD_SHA, q8_prediction_archive_read=False,
                    q8_weights_read=False, supervised_fit_data="official_competition_train_only",
                    stage13_canonical_folds_reused=False, official_prompts_previously_used=True,
                    glm_layers=[8, 3], deberta_layers=[24, 6], feature_shape=list(x.shape),
                    feature_dtype=str(x.dtype), deberta_id_admission_mode=id_mode,
                    scikit_learn_version=sklearn_version, folds=fold_summary)
    return Bundle(x, ids, y, prompts, texts, folds, CLASSES, evidence)


def _head(x, y, mask, head):
    fit = np.ascontiguousarray(x[~mask], dtype=np.float64)
    score = np.ascontiguousarray(x[mask], dtype=np.float64)
    scaler = StandardScaler(copy=False)
    fit = scaler.fit_transform(fit)
    score = scaler.transform(score)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        estimator = LogisticRegression(**head).fit(fit, y[~mask])
    check(np.array_equal(estimator.classes_, np.arange(len(CLASSES))), "stage14_control_class_order")
    nit = estimator.n_iter_
    check(len(nit) == 1 and 0 < nit[0] < head["max_iter"], "stage14_control_convergence")
    p = estimator.predict_proba(score)
    glm_readout.validate_probabilities(p, int(mask.sum()), len(CLASSES))
    return p, int(nit[0])


def baseline(bundle, fold):
    """Return the existing 75:25 probability mixture, freshly fit on this split."""
    check(type(fold) is int and 0 <= fold < FOLDS, "stage14_control_fold_index")
    mask = bundle.folds == fold
    check(set(bundle.prompts[mask]).isdisjoint(bundle.prompts[~mask])
          and len(np.unique(bundle.prompts[mask])) == 30
          and len(np.unique(bundle.prompts[~mask])) == 120, "stage14_control_fold_prompt_exclusion")
    with threadpool_limits(limits=THREADS):
        gp, gi = _head(bundle.glm, bundle.y, mask, glm_readout.HEAD)
        dp, di = _head(bundle.deberta, bundle.y, mask, deberta_readout.HEAD)
    p = CONTROL_WEIGHTS[0] * gp + CONTROL_WEIGHTS[1] * dp
    glm_readout.validate_probabilities(p, int(mask.sum()), len(CLASSES))
    return p, {"glm_iterations": gi, "deberta_iterations": di, "probability_weights": list(CONTROL_WEIGHTS)}
