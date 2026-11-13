"""Audit a mounted native Q8 reference without exporting rows or fitting models.

Default is a structure-only check. Only an independently reviewed and pinned
provenance receipt may authorize the single aggregate raw plain-head score.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
from itertools import permutations
import json
import math
from pathlib import Path
import re
import sys
import zipfile

import numpy as np

from stage14_native_input_audit import (
    AuditFailure, CLASSES, TRAIN_SHA256, audit as audit_task_assets, need,
    read_json, stable_hash,
)

ROWS, PROMPTS, FOLDS = 2699, 150, 5
ID_SHA = "707edf645b0f09bba8f0983b5bc9f522595c95e8ca5b763b38babeb38200368d"
STAGE13F_FOLD_SHA = "a55b035686cb7b31f6fbe852ac990668709aec4b913c5d0928d89d843bdbfe62"
REFERENCE = "oof_reference.npz"
REVIEW_STAGES = {"external_and_synthetic_training", "official_supervised_training",
                 "checkpoint_selection_and_calibration"}


def sha_string(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def stage13_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def train_axes(path):
    with path.open("rb") as raw:
        need(hashlib.file_digest(raw, "sha256").hexdigest() == TRAIN_SHA256, "official train fingerprint")
        raw.seek(0)
        reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8", newline=""))
        need(len(reader.fieldnames or []) == 4 and set(reader.fieldnames or []) ==
             {"id", "prompt_id", "model", "text"}, "official train schema")
        rows = [(r["id"], r["prompt_id"], r["model"]) for r in reader]
    ids, prompts, targets = [list(column) for column in zip(*rows)]
    need(len(rows) == len(set(ids)) == ROWS and all(ids) and stage13_digest(ids) == ID_SHA,
         "official train row axis")
    need(len(set(prompts)) == PROMPTS and all(prompts) and set(targets) == CLASSES
         and len(set(zip(prompts, targets))) == ROWS, "official train prompt or class axis")
    return ids, prompts, targets


def index_list(values, gate):
    need(isinstance(values, list) and 0 < len(values) <= ROWS
         and all(type(v) is int and 0 <= v < ROWS for v in values)
         and len(set(values)) == len(values), gate)
    return np.asarray(values, dtype=np.int64)


def split_axes(assets, prompts, files):
    prompts = np.asarray(prompts)
    folds = np.full(ROWS, -1, dtype=np.int64)
    groups, counts, has_train_axes = {}, [], []
    for fold in range(FOLDS):
        key = f"fold{fold}/split.json"
        data = read_json(assets / key, key, files)
        validation = index_list(data.get("validation_rows"), key + ": invalid validation rows")
        need((folds[validation] == -1).all(), "validation rows shared across folds")
        folds[validation] = fold
        held = set(prompts[validation])
        need(all(groups.setdefault(str(prompt), fold) == fold for prompt in held),
             "official prompt shared across validation folds")
        train_keys = set(data) & {"train_rows", "training_rows"}
        need(len(train_keys) <= 1, key + ": ambiguous training rows")
        if train_keys:
            train = index_list(data[next(iter(train_keys))], key + ": invalid training rows")
            need(not held.intersection(prompts[train]), key + ": training prompt overlaps validation")
        has_train_axes.append(bool(train_keys))
        need(len(held) == PROMPTS // FOLDS, key + ": expected 30 held-out prompts")
        counts.append({"validation_rows": len(validation), "validation_prompts": len(held)})
    need((folds >= 0).all() and len(groups) == PROMPTS, "validation folds do not partition official rows")
    # Fold numbering is arbitrary; only the assignment of prompts matters.
    canonical = any(stage13_digest({p: perm[f] for p, f in groups.items()}) == STAGE13F_FOLD_SHA
                    for perm in permutations(range(1, FOLDS + 1)))
    return folds, groups, counts, has_train_axes, canonical


def load_reference(path, ids, classes, folds):
    # Fixed small numeric archive; never load pickled or arbitrarily large members.
    limit = 16 * 1024 * 1024
    need(0 < path.stat().st_size <= limit, "reference compressed size limit")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    need(0 < len(raw) <= limit, "reference compressed size limit")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        info = archive.infolist()
        names = [x.filename for x in info]
        need(len(names) <= 16 and len(set(names)) == len(names)
             and all(re.fullmatch(r"[a-zA-Z0-9_-]+\.npy", n) for n in names)
             and sum(x.file_size for x in info) <= limit, "unsafe reference members")
        need({"plain.npy", "centered.npy"} <= set(names), "reference head arrays missing")
        for key, shape, kinds in (("plain", (ROWS, len(classes)), "f"),
                                  ("centered", (ROWS, len(classes)), "f"),
                                  ("ids", (ROWS,), "US"), ("classes", (len(classes),), "US"),
                                  ("folds", (ROWS,), "iu")):
            name = key + ".npy"
            if name not in names:
                continue
            with archive.open(name) as stream:
                version = np.lib.format.read_magic(stream)
                readers = {(1, 0): np.lib.format.read_array_header_1_0,
                           (2, 0): np.lib.format.read_array_header_2_0}
                need(version in readers, "unsupported NumPy format: " + key)
                actual_shape, _, dtype = readers[version](stream, max_header_size=4096)
                need(actual_shape == shape and dtype.kind in kinds and 0 < dtype.itemsize <= 1024
                     and stream.tell() + math.prod(shape) * dtype.itemsize == archive.getinfo(name).file_size,
                     "unsafe reference array header: " + key)
    with np.load(io.BytesIO(raw), allow_pickle=False, max_header_size=4096) as data:
        plain, centered = data["plain"], data["centered"]
        need(all(a.shape == (ROWS, len(classes)) and a.dtype in (np.dtype("float32"), np.dtype("float64"))
                 and np.isfinite(a).all() for a in (plain, centered)), "reference shape, dtype, or finiteness")
        for key, expected in (("ids", ids), ("classes", classes), ("folds", folds.tolist())):
            if key in data.files:
                a = data[key]
                values = a.astype("U").tolist() if a.dtype.kind == "S" else a.tolist()
                need(values == expected,
                     "reference embedded " + key + " axis mismatch")
        present = {key: key in data.files for key in ("ids", "classes", "folds")}
    return plain, hashlib.sha256(raw).hexdigest(), present


def review_provenance(receipt_path, receipt_sha, contract):
    """Bind explicit human review evidence; never infer exclusion from filenames."""
    files = {}
    data = read_json(receipt_path, "provenance receipt", files, limit=256 * 1024)
    need(files["provenance receipt"]["sha256"] == receipt_sha, "provenance receipt fingerprint mismatch")
    need(type(data.get("schema_version")) is int and data["schema_version"] == 1
         and data.get("contract") == contract and data.get("reference_row_and_class_order_verified") is True
         and data.get("each_reference_row_scored_only_by_its_excluding_fold") is True
         and data.get("all_supervised_training_and_selection_sources_enumerated") is True,
         "provenance receipt incomplete or not bound to this reference")
    stages = data.get("reviewed_sources")
    need(isinstance(stages, dict) and set(stages) == REVIEW_STAGES, "provenance stages incomplete")
    for stage, sources in stages.items():
        need(isinstance(sources, list) and len(sources) > 0, "provenance sources absent: " + stage)
        for item in sources:
            need(isinstance(item, dict) and sha_string(item.get("source_manifest_sha256"))
                 and sha_string(item.get("exclusion_evidence_sha256"))
                 and item.get("heldout_prompt_overlaps_by_fold") == [0] * FOLDS
                 and all(type(x) is int for x in item["heldout_prompt_overlaps_by_fold"]),
                 "provenance source exclusion missing: " + stage)


def run(assets, train_csv, expected_tree_sha, receipt=None, receipt_sha=None):
    need(sha_string(expected_tree_sha), "real task asset tree fingerprint required")
    need((receipt is None and receipt_sha is None) or
         (receipt is not None and sha_string(receipt_sha)), "receipt and exact receipt fingerprint required together")
    task = audit_task_assets(assets, expected_tree_sha)
    ids, prompts, targets = train_axes(train_csv)
    files = {}
    classes = read_json(assets / "manifest.json", "manifest.json", files)["classes"]
    need(stable_hash(classes) == task["class_order_sha256"], "task class order changed")
    folds, groups, counts, training_rows_present, canonical = split_axes(assets, prompts, files)
    plain, ref_sha, embedded = load_reference(assets / REFERENCE, ids, classes, folds)
    contract = {"task_tree_sha256": expected_tree_sha, "reference_sha256": ref_sha,
                "official_train_sha256": TRAIN_SHA256, "ordered_official_id_sha256": ID_SHA,
                "class_order_sha256": stable_hash(classes), "ordered_fold_files_sha256": stable_hash(files),
                "prompt_fold_map_sha256": stable_hash(groups)}
    score = None
    if receipt is not None:
        review_provenance(receipt, receipt_sha, contract)
        lookup = {label: index for index, label in enumerate(classes)}
        y = np.asarray([lookup[label] for label in targets])
        pred = plain.argmax(axis=1)
        score = float(np.mean([np.mean(pred[y == c] == c) for c in range(len(classes))]))
    return {"status": "Q8_REFERENCE_DEVELOPMENT_AGGREGATE" if score is not None
            else "Q8_REFERENCE_STRUCTURAL_PREFLIGHT_ONLY", "schema_version": 1,
            "contract": contract, "rows": ROWS, "prompts": PROMPTS, "classes": len(classes),
            "fold_counts": counts, "reference_embedded_axes": embedded,
            "declared_official_train_rows_checked_by_fold": training_rows_present,
            "validation_prompt_partition_verified": True,
            "stage13f_prompt_partition_matches_up_to_fold_numbering": canonical,
            "stage13f_paired_comparison_performed": False, "centered_scores_interpreted": False,
            "training_exclusion_independently_reviewed_receipt_sha256": receipt_sha,
            "training_source_provenance": "bound_to_independent_receipt" if receipt is not None else
            "unverified_external_official_selection_and_calibration_inputs",
            "raw_plain_development_balanced_accuracy": score,
            "reused_official_train_prompts": True, "raw_rows_or_predictions_exported": False,
            "official_score_measured": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--expected-tree-sha256", required=True)
    parser.add_argument("--provenance-receipt", type=Path)
    parser.add_argument("--expected-provenance-receipt-sha256")
    args = parser.parse_args(argv)
    try:
        result = run(args.assets, args.train_csv, args.expected_tree_sha256,
                     args.provenance_receipt, args.expected_provenance_receipt_sha256)
    except AuditFailure as exc:
        result = {"status": "BLOCK_Q8_REFERENCE_AUDIT", "gate": str(exc)}
    except (OSError, ValueError, IndexError, KeyError, TypeError, OverflowError, RecursionError,
            zipfile.BadZipFile, csv.Error) as exc:
        result = {"status": "BLOCK_Q8_REFERENCE_AUDIT", "gate": type(exc).__name__}
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return int(result["status"] == "BLOCK_Q8_REFERENCE_AUDIT")


if __name__ == "__main__":
    sys.exit(main())
