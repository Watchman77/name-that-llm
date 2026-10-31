"""Stage13A: read-only admission of frozen-base GLM features, never OOF scoring.

Run on CPU with Internet off. Only aggregate evidence is written. No fitting,
prediction, Torch/custom-code import, object deserialization or balancing occurs.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform
import time
import zipfile

import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.model_selection import GroupKFold


TRAIN_SHA256 = "40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707"
GLM_SHA256 = "2b41585a9482cef825962a8cc096697933c0b1237e8ede0712d1f8ab361d9414"
ORDERED_TEXT_SHA256 = "6496071874da8f911fafebd9eec6191e56134a97f00226dd8c78805d31f07e36"
FOLD_SOURCE_SHA256 = "5c27fcdffde2bcfd512d20fb4edec9bd7133430e79ccd71e31875151d235a23d"
FOLD_MAP_SHA256 = "a55b035686cb7b31f6fbe852ac990668709aec4b913c5d0928d89d843bdbfe62"
CLASSES = (
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5",
    "claude-sonnet-5", "gemini-3.1-pro-preview", "gemini-3.6-flash",
    "gemini-3.7-flash", "gemini-3.8-flash", "glm-5.3", "glm-5.3-flash",
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra",
    "gpt-6-astra", "kimi-k3", "qwen3.8-max",
)
CACHE_KEYS = {"features", "ids", "layers", "text_sha256", "token_lengths", "train_sha256"}


@dataclass(frozen=True)
class Contract:
    """The CLI always uses PRODUCTION; small contracts exist for fixtures only."""
    train_sha256: str
    glm_sha256: str
    ordered_text_sha256: str
    fold_map_sha256: str
    rows: int = 2699
    prompts: int = 150
    hidden: int = 4096
    classes: tuple[str, ...] = CLASSES
    layers: tuple[int, ...] = (8, 3, 24)
    folds: int = 5


PRODUCTION = Contract(TRAIN_SHA256, GLM_SHA256, ORDERED_TEXT_SHA256, FOLD_MAP_SHA256)


class AdmissionFailure(Exception):
    """Only a fixed, sanitized check code may leave the audit."""


def check(condition, code):
    if not bool(condition):
        raise AdmissionFailure(code)


def json_hash(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def unique_file(root, name):
    matches = [path for path in root.rglob(name) if path.is_file()]
    check(len(matches) == 1, "unique_" + name.replace(".", "_"))
    return matches[0]


def npz_headers(path, contract):
    """Inspect object/oversize headers before NumPy can allocate or deserialize."""
    expected_names = {key + ".npy" for key in CACHE_KEYS}
    result = {}
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        check(len(names) == len(set(names)) and set(names) == expected_names, "glm_cache_schema")
        max_size = contract.rows * len(contract.layers) * contract.hidden * 4 + 2_000_000
        check(sum(info.file_size for info in archive.infolist()) <= max_size, "glm_cache_size_bound")
        for name in names:
            with archive.open(name) as stream:
                version = np.lib.format.read_magic(stream)
                check(version in ((1, 0), (2, 0)), "npy_header_version")
                reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                          else np.lib.format.read_array_header_2_0)
                shape, fortran, dtype = reader(stream)
            check(not dtype.hasobject, "npz_object_dtype_rejected")
            result[name[:-4]] = (shape, dtype)
    check(result["features"] == ((contract.rows, len(contract.layers), contract.hidden), np.dtype("float32")),
          "glm_feature_shape_dtype")
    for name in ("ids", "text_sha256"):
        shape, dtype = result[name]
        check(shape == (contract.rows,) and dtype.kind == "U" and dtype.itemsize <= 512,
              "glm_" + name + "_header")
    check(result["layers"] == ((len(contract.layers),), np.dtype("int64")), "glm_layers_header")
    check(result["token_lengths"] == ((contract.rows,), np.dtype("int64")), "glm_token_lengths_header")
    shape, dtype = result["train_sha256"]
    check(shape == () and dtype.kind == "U" and dtype.itemsize <= 512, "glm_train_digest_header")
    return result


def validate_fold_partition(groups, folds, n_folds):
    """No prompt may cross a future fit/scoring partition; no model is fitted."""
    groups, folds = np.asarray(groups), np.asarray(folds)
    check(folds.shape == groups.shape and folds.dtype.kind in "iu", "fold_shape_dtype")
    check(set(folds.tolist()) == set(range(1, n_folds + 1)), "fold_coverage")
    mapping = {}
    for group, fold in zip(groups, folds):
        key = str(group)
        check(key not in mapping or mapping[key] == int(fold), "prompt_crosses_folds")
        mapping[key] = int(fold)
    summaries = []
    for number in range(1, n_folds + 1):
        valid = folds == number
        fit = ~valid
        fit_groups, valid_groups = set(groups[fit]), set(groups[valid])
        check(valid.any() and fit.any() and not (fit_groups & valid_groups), "fold_prompt_exclusion")
        summaries.append({"fold": number, "fit_rows": int(fit.sum()), "valid_rows": int(valid.sum()),
                          "fit_prompts": len(fit_groups), "valid_prompts": len(valid_groups),
                          "prompt_overlap": 0})
    return mapping, summaries


def audit_inputs(input_root, output_root, *, contract=PRODUCTION):
    """Return sanitized admission evidence. Existing output directories stay intact."""
    started = time.monotonic()
    root, out = Path(input_root), Path(output_root)
    production = contract == PRODUCTION
    report = {
        "schema_version": 1, "status": "FAILED_INPUT_ADMISSION", "contract_is_production": production,
        "scope": "frozen_base_feature_admission_only", "fit_performed": False,
        "model_inference_performed": False, "balancing_performed": False,
        "held_out_predictions_produced": False, "development_reference_admitted": False,
        "stage13b_batch_coupled_comparison_admitted": False, "full_s07_oof_admitted": False,
        "eligible_for_future_fold_local_readout_fitting": False,
        "official_score_estimated": False, "raw_rows_or_predictions_exported": False,
        "object_deserialization_performed": False, "checks_passed": [],
        "expected_input_versions": {"competition": 170484, "own_glm_output": 359405443},
        "input_versions_verified_by_filesystem": False,
        "runtime_versions": {"python": platform.python_version(), "numpy": np.__version__,
                             "pandas": pd.__version__, "scikit_learn": sklearn_version},
        "limits": [
            "Feature admission is not fitted held-out predictions, a development score or a full-S07 reference.",
            "Future readouts and learned preprocessing must exclude every scoring-batch prompt.",
            "Mixed supervised source folds are not admitted for coupled balancing.",
            "DeBERTa, adapted/full-fit S07 members and other branches are not admitted by this GLM-only audit.",
            "Recorded cache/extractor provenance is not an independent audit of general pretrained-weight history.",
            "Immutable input-version bindings require separate saved-UI verification.",
            "Canonical prompt folds were previously used and remain exploratory development data.",
        ],
    }
    can_write = False

    def passed(code):
        report["checks_passed"].append(code)

    try:
        check(not out.exists() or (out.is_dir() and not any(out.iterdir())), "fresh_output_directory")
        out.mkdir(parents=True, exist_ok=True)
        can_write = True
        passed("fresh_output_directory")
        check(root.is_dir(), "input_root_directory")
        train_path, cache_path = unique_file(root, "train.csv"), unique_file(root, "glm_features.npz")
        passed("unique_required_inputs")
        train_digest, cache_digest = sha256(train_path), sha256(cache_path)
        check(train_digest == contract.train_sha256, "official_train_sha256")
        check(cache_digest == contract.glm_sha256, "glm_cache_sha256")
        report["files"] = {"official_train": {"sha256": train_digest, "bytes": train_path.stat().st_size},
                           "glm_cache": {"sha256": cache_digest, "bytes": cache_path.stat().st_size}}
        passed("exact_input_file_hashes")
        train = pd.read_csv(train_path, dtype=str)
        check(set(train.columns) == {"id", "text", "prompt_id", "model"}, "official_train_schema")
        check(len(train) == contract.rows and not train.isna().any().any(), "official_train_rows_nonnull")
        check(all(train[col].str.len().gt(0).all() for col in train.columns), "official_train_nonempty_strings")
        check(train.id.is_unique, "official_train_unique_ids")
        ids = train.id.to_numpy(dtype=str)
        groups = train.prompt_id.to_numpy(dtype=str)
        classes = tuple(np.sort(train.model.unique()).tolist())
        check(classes == contract.classes and len(np.unique(groups)) == contract.prompts, "official_classes_prompts")
        passed("official_train_axis_contract")
        text_hashes = np.asarray([hashlib.sha256(value.encode("utf-8")).hexdigest()
                                  for value in train.text], dtype=str)
        text_digest = json_hash(text_hashes.tolist())
        check(text_digest == contract.ordered_text_sha256, "ordered_text_digest")
        npz_headers(cache_path, contract)
        passed("numeric_unicode_npz_headers")
        with np.load(cache_path, allow_pickle=False) as cache:
            check(np.array_equal(cache["ids"], ids), "glm_exact_official_id_order")
            check(np.array_equal(cache["text_sha256"], text_hashes), "glm_exact_text_hash_order")
            check(cache["train_sha256"].item() == contract.train_sha256, "glm_bound_train_digest")
            check(np.array_equal(cache["layers"], contract.layers), "glm_exact_layer_order")
            lengths = cache["token_lengths"]
            check(bool(((lengths > 0) & (lengths <= 512)).all()), "glm_recorded_token_lengths")
            features = cache["features"]
            for start in range(0, contract.rows, 256):
                check(np.isfinite(features[start:start + 256]).all(), "glm_features_finite")
            del features
        passed("glm_values_alignment_and_finiteness")
        labels = {label: i for i, label in enumerate(classes)}
        y = train.model.map(labels).to_numpy(dtype=int)
        folds = np.full(contract.rows, -1, dtype=np.int64)
        dummy = np.zeros((contract.rows, 1), dtype=np.uint8)
        for number, (_, valid) in enumerate(GroupKFold(contract.folds).split(dummy, y, groups), 1):
            folds[valid] = number
        mapping, fold_summary = validate_fold_partition(groups, folds, contract.folds)
        fold_hash = json_hash(mapping)
        check(fold_hash == contract.fold_map_sha256, "canonical_prompt_fold_mapping")
        passed("canonical_future_prompt_fold_partitions")
        report.update(
            status=("PASS_FROZEN_BASE_FEATURE_ADMISSION_ONLY" if production
                    else "PASS_SYNTHETIC_FEATURE_ADMISSION_ONLY"),
            eligible_for_future_fold_local_readout_fitting=True,
            rows=contract.rows, prompts=contract.prompts, classes=list(classes),
            class_supports={label: int((train.model == label).sum()) for label in classes},
            feature_shape=[contract.rows, len(contract.layers), contract.hidden], feature_dtype="float32",
            layers=list(contract.layers), classifier_layers=[8, 3], grouping_layer=24,
            all_feature_values_finite=True, ordered_text_sha256=text_digest,
            id_axis_sha256=json_hash(ids.tolist()), label_axis_sha256=json_hash(list(classes)),
            fold_source_path="scripts/stage06c_mixed_confirmation.py", fold_source_sha256=FOLD_SOURCE_SHA256,
            canonical_prompt_fold_mapping_sha256=fold_hash, fold_partitions=fold_summary,
            recorded_token_length_min=int(lengths.min()), recorded_token_length_max=int(lengths.max()),
        )
    except AdmissionFailure as error:
        report.update(failure_code=str(error), failure_type="AdmissionFailure")
    except Exception as error:
        # Never include library exceptions: they can contain row values or private paths.
        report.update(failure_code="reader_or_environment_failure", failure_type=type(error).__name__)
    report["checks_passed_count"] = len(report["checks_passed"])
    report["seconds"] = round(time.monotonic() - started, 6)
    report["report_written"] = can_write
    if can_write:
        try:
            (out / "stage13a_input_admission.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        except Exception as error:
            report.update(status="FAILED_INPUT_ADMISSION", failure_code="report_write_failure",
                          failure_type=type(error).__name__, report_written=False,
                          eligible_for_future_fold_local_readout_fitting=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default="/kaggle/input")
    parser.add_argument("--output-root", default="/kaggle/working/stage13a_admission")
    arguments = parser.parse_args()
    report = audit_inputs(arguments.input_root, arguments.output_root)
    print("STAGE13A_INPUT_ADMISSION_JSON_BEGIN", flush=True)
    print(json.dumps(report, sort_keys=True), flush=True)
    print("STAGE13A_INPUT_ADMISSION_JSON_END", flush=True)
    return 0 if report["status"] == "PASS_FROZEN_BASE_FEATURE_ADMISSION_ONLY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
