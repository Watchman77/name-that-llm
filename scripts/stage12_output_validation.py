"""Read-only saved-output checks for frozen Dhruv nlm-v27 V1 (362013743).

This checks the saved parquet and complementary logits. The six source cells do
not save main probabilities or final scores, so a pass is not a complete-model,
mixture, runtime, accuracy, or expert-coverage verification. Review full logs.
"""
from __future__ import annotations

import hashlib
import json
import math
import numbers
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd


SOURCE_VERSION = 362013743
LABELS = [
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5",
    "claude-sonnet-5", "gemini-3.1-pro-preview", "gemini-3.6-flash",
    "gemini-3.7-flash", "gemini-3.8-flash", "glm-5.3", "glm-5.3-flash",
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra",
    "gpt-6-astra", "kimi-k3", "qwen3.8-max",
]
MAIN_GROUPS = ["q14v5", "q8v5", "ll8v5", "n14b", "q25_14", "deb"]
MANUAL_LOG_CHECKS = [
    "COMP STACK DONE and all expected complementary member contributions",
    "no missing-checkpoint SKIP for the four complementary LoRA members",
    "all six main groups completed: " + ", ".join(MAIN_GROUPS),
    "comp stack loaded; no COMP SKIPPED fallback",
    "ATLAS selection and successful mixing; no ATLAS SKIPPED fallback",
    "final wrote submission.parquet and complete execution without failure",
    "nonfinite-row warnings, replacements, retries and in-sample checks reviewed",
]
LIMITATIONS = [
    "Main member probabilities, aggregate scores and postprocessing arrays are unsaved.",
    "Finite saved complementary logits do not prove finite main internals.",
    "No independent main/COMP/ATLAS mixture or final-argmax reconstruction is possible.",
    "COMP/ATLAS fallback and missing-member SKIP branches require full own-log review.",
    "Shell pipeline success does not prove complementary-worker success or expert coverage.",
    "Three-row source execution is real inference, not a format-only shortcut.",
    "Small-input in-sample checks are not development or out-of-fold accuracy.",
    "No official public/private score, complete runtime, or GPU admission is established here.",
]


class OutputValidationError(RuntimeError):
    """A sanitized check failure; report contains aggregate information only."""

    def __init__(self, code, report):
        super().__init__(code)
        self.report = report


def _sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _axis_hash(values):
    encoded = json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _positive_seconds(value):
    return (isinstance(value, numbers.Real) and not isinstance(value, (bool, np.bool_))
            and math.isfinite(float(value)) and float(value) > 0)


def validate_output(work_dir, test_path, *, namespace, report_path=None):
    """Read actual saved files; never load or fit models, alter outputs, or tune.

    `namespace` must be the frozen notebook's `globals()`. It supplies the source's
    complementary start/wall time, ATLAS budget and six configured model groups.
    Saved files must be fresh relative to that source start. All row identifiers,
    texts and per-row values stay out of the JSON/stdout report.
    """
    root, test_path = Path(work_dir), Path(test_path)
    report_path = Path(report_path) if report_path else root / "stage12_output_validation.json"
    report = {
        "schema_version": 1, "source_script_version_id": SOURCE_VERSION,
        "status": "RUNNING", "guard_passed": False,
        "validation_scope": "saved_parquet_and_complementary_logit_contract_only",
        "parquet_readback": "pandas.read_parquet_actual_saved_file",
        "source_has_format_only_preview_shortcut": False,
        "full_main_finiteness_verified": False,
        "full_mixture_verified": False, "final_argmax_reconstructed": False,
        "expert_coverage_verified": False, "hidden_runtime_verified": False,
        "development_accuracy_verified": False, "official_score_verified": False,
        "manual_log_review_required": True,
        "manual_log_checks": MANUAL_LOG_CHECKS, "limitations": LIMITATIONS,
        "checks_passed": [],
    }

    def check(condition, code):
        if not condition:
            raise OutputValidationError(code, report)
        report["checks_passed"].append(code)

    try:
        check(isinstance(namespace, dict), "notebook_namespace_present")
        for name in ("NTL_SMOKE", "NTL_FAKE_TEST", "NTL_TINY"):
            check(os.environ.get(name) in (None, ""), "diagnostic_override_absent_" + name)
        check(os.environ.get("NTL_DROP_Q35") == "1", "frozen_intentional_q35_omission")
        started, comp_wall = namespace.get("_c0"), namespace.get("COMP_SEC")
        check(_positive_seconds(started), "source_complement_start_recorded")
        check(_positive_seconds(comp_wall), "source_complement_wall_seconds_valid")
        budget = namespace.get("ATLAS_BUDGET")
        check(isinstance(budget, numbers.Integral) and not isinstance(budget, (bool, np.bool_))
              and int(budget) == int(8.3 * 3600 - float(comp_wall) - 600),
              "source_atlas_budget_matches_frozen_formula")
        specs = namespace.get("MODELS")
        check(isinstance(specs, list) and all(isinstance(s, dict) for s in specs)
              and [s.get("name") for s in specs] == MAIN_GROUPS,
              "source_six_main_group_names")
        now = time.time()
        check(float(started) <= now + 1, "source_start_not_in_future")

        check(test_path.is_file(), "test_csv_present")
        test = pd.read_csv(test_path, dtype={"id": "string"})
        check(list(test.columns) == ["id", "text"], "test_exact_column_order")
        check(len(test) > 0, "test_nonempty")
        check(test.id.notna().all(), "test_ids_nonnull")
        ids = test.id.tolist()
        check(all(isinstance(i, str) and len(i) > 0 for i in ids), "test_ids_nonempty_strings")
        check(test.id.is_unique, "test_ids_unique")
        count = len(ids)
        report.update(row_count=count, input_is_three_rows=(count == 3),
                      test_id_axis_sha256=_axis_hash(ids),
                      canonical_label_axis_sha256=_axis_hash(LABELS))

        paths = {"submission": root / "submission.parquet",
                 "comp_logit": root / "comp_logit.npy",
                 "comp_meta": root / "comp_meta.json"}
        for key, path in paths.items():
            check(path.is_file() and path.stat().st_size > 0, key + "_present_nonempty")
            check(float(started) - 1 <= path.stat().st_mtime <= now + 1,
                  key + "_fresh_for_this_execution")
        check(paths["submission"].stat().st_mtime + 1 >=
              max(paths["comp_logit"].stat().st_mtime, paths["comp_meta"].stat().st_mtime),
              "submission_written_after_complement_artifacts")

        sub = pd.read_parquet(paths["submission"])
        check(list(sub.columns) == ["id", "model"], "submission_exact_column_order")
        check(len(sub) == count, "submission_row_count_matches_test")
        check(not sub.isna().to_numpy().any(), "submission_no_nulls")
        check(all(isinstance(i, str) and len(i) > 0 for i in sub.id), "submission_string_ids")
        check(sub.id.is_unique, "submission_ids_unique")
        check(sub.id.tolist() == ids, "submission_exact_test_id_order")
        check(all(isinstance(m, str) for m in sub.model), "submission_string_models")
        check(sub.model.isin(LABELS).all(), "submission_allowed_18_labels")

        meta = json.loads(paths["comp_meta"].read_text(encoding="utf-8"))
        check(isinstance(meta, dict) and set(meta) == {"labels", "ids", "seconds"},
              "comp_metadata_exact_schema")
        check(isinstance(meta["ids"], list) and meta["ids"] == ids,
              "comp_exact_test_id_order")
        check(isinstance(meta["labels"], list) and meta["labels"] == LABELS,
              "comp_named_class_axis_matches_frozen_sorted_order")
        check(_positive_seconds(meta["seconds"]), "comp_worker_seconds_positive_finite")
        check(float(meta["seconds"]) <= float(comp_wall) + 2,
              "comp_worker_seconds_within_notebook_wall_time")
        logits = np.load(paths["comp_logit"], allow_pickle=False)
        check(isinstance(logits, np.ndarray), "comp_saved_array_is_ndarray")
        check(logits.shape == (count, len(LABELS)), "comp_exact_n_by_18_shape")
        check(logits.dtype.kind == "f", "comp_logits_real_floating_dtype")
        check(np.isfinite(logits).all(), "comp_all_saved_logits_finite")
        report.update(
            submission_class_counts={k: int(v) for k, v in sorted(sub.model.value_counts().items())},
            comp_shape=list(logits.shape), comp_dtype=str(logits.dtype),
            comp_values_are="raw_weighted_log_evidence_not_normalized_probabilities",
            comp_worker_seconds=float(meta["seconds"]),
            complementary_notebook_wall_seconds=float(comp_wall),
            atlas_budget_seconds=int(budget), atlas_budget_positive=bool(budget > 0),
            files={k: {"bytes": p.stat().st_size, "sha256": _sha256_file(p)}
                   for k, p in paths.items()},
            status="PASS_SAVED_OUTPUT_CONTRACT", guard_passed=True,
        )
    except Exception as exc:
        report.update(status="FAILED_OUTPUT_VALIDATION", guard_passed=False,
                      failure_code=str(exc) if isinstance(exc, OutputValidationError)
                      else "reader_or_io_failure", failure_type=type(exc).__name__)
    report["checks_passed_count"] = len(report["checks_passed"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("STAGE12_OUTPUT_VALIDATION_JSON_BEGIN")
    print(json.dumps(report, sort_keys=True))
    print("STAGE12_OUTPUT_VALIDATION_JSON_END")
    if not report["guard_passed"]:
        raise OutputValidationError(report["failure_code"], report)
    return report
