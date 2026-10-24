"""CPU-only aggregate audit of the public V20 old-stack CSV merge.

Copies only load_extra_passages()'s pandas/glob operations. It does not execute
the captured public notebook, fit models, load weights, use a network, or emit
passages, prompt IDs, or row-level predictions. Input files remain read-only.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path
import re

import pandas as pd


SOURCE_VERSION = 360088024
EXPECTED_ROWS, EXPECTED_PROMPTS = 7644, 509
LABELS = sorted([
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
    "qwen3.8-max", "glm-5.3", "glm-5.3-flash", "kimi-k3",
])
SOURCES = [
    ("**/ntl_extra_passages_claude.csv", "model", None),
    ("**/gen_passages.csv", "model", None),
    ("**/ntl_generations.csv", "label", "finish_reason"),
    ("**/ntl_proxy_generations.csv", "label", None),
    ("**/ntl_generated*.csv", "model", "finish"),
    ("**/passages_clean.csv", "model", None),
]


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def receipt(path, root):
    return {"path": str(path.relative_to(root)), "bytes": path.stat().st_size,
            "sha256": sha256(path)}


class MissingColumns(ValueError):
    def __init__(self, path, columns):
        self.path = path
        self.columns = sorted(columns)


def load_extra_passages(root, membership):
    """Exact source merge, with aggregate read receipts and explicit schema errors.

    Preserve pattern/file order and pandas default CSV parsing. In particular,
    preserve keep-first deduplication and the pv_ prefix for passages_clean.csv.
    No cross-source prompt-bank repairs or new overlap filtering are applied.
    """
    parts = []
    for pattern, label_col, finish_col in SOURCES:
        for filename in sorted(glob.glob(str(root / pattern), recursive=True)):
            path = Path(filename)
            d = pd.read_csv(path)
            missing = {"prompt_id", label_col, "text"} - set(d.columns)
            entry = {**receipt(path, root), "pattern": pattern,
                     "raw_rows": len(d), "columns": [str(c) for c in d.columns],
                     "label_column": label_col, "finish_column": finish_col,
                     "finish_filter_applied": bool(finish_col and finish_col in d)}
            membership.append(entry)
            if missing:
                raise MissingColumns(entry["path"], missing)
            if finish_col and finish_col in d:
                d = d[d[finish_col].eq("stop")]
            entry["rows_after_finish_filter"] = len(d)
            d = d.rename(columns={label_col: "model"})[["prompt_id", "model", "text"]].copy()
            d["prompt_id"] = ("pv_" if filename.endswith("passages_clean.csv") else "") + d["prompt_id"].astype(str)
            parts.append(d)
    if not parts:
        return pd.DataFrame(columns=["prompt_id", "model", "text"])
    x = pd.concat(parts, ignore_index=True)
    x = x[x["model"].isin(LABELS)].copy()
    x["text"] = x["text"].astype(str).map(lambda t: re.sub(r"\s+", " ", re.sub(r"^\*\*(.*)\*\*$", r"\1", t.strip())).strip())
    x = x[x["text"].str.split().str.len().between(30, 300)].drop_duplicates(subset=["model", "text"])
    x = x.drop_duplicates(subset=["prompt_id", "model"])
    x = x[x.groupby("prompt_id")["model"].transform("nunique") >= 12].reset_index(drop=True)
    return x


def checkpoint_meta_summaries(root):
    """Report selected safe metadata fields only; the existing checker owns gates."""
    output = {}
    for name in ["lora8b_v1", "lora14b_v1"]:
        matches = sorted(root.glob("**/" + name + "/meta.json"))
        info = {"matching_meta_files": len(matches), "records": []}
        output[name] = info
        for path in matches:
            meta = json.loads(path.read_text())
            cfg = meta.get("cfg", {})
            adapter_path = path.parent / "adapter/adapter_config.json"
            adapter = json.loads(adapter_path.read_text()) if adapter_path.is_file() else {}
            selected = {k: cfg.get(k) for k in ["keep", "layers", "max_len"]}
            selected_adapter = {k: adapter.get(k) for k in ["r", "lora_alpha", "lora_dropout", "bias", "task_type", "target_modules", "fan_in_fan_out"]}
            # A metadata summary reports values; it is not an identity certificate.
            info["records"].append({"meta": receipt(path, root), "cfg": selected,
                                    "label_order_matches_outer_source": meta.get("labels") == [
                                        "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
                                        "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
                                        "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5", "claude-fable-5",
                                        "qwen3.8-max", "glm-5.3", "glm-5.3-flash", "kimi-k3"],
                                    "adapter_config": receipt(adapter_path, root) if adapter_path.is_file() else None,
                                    "adapter_fields": selected_adapter})
    return output


def run(root, expected_rows=EXPECTED_ROWS, expected_prompts=EXPECTED_PROMPTS):
    root = root.resolve()
    membership = []
    report = {"stage": "08_extra_merge_audit", "public_source_version": SOURCE_VERSION,
              "read_only": True, "model_code_executed": False, "new_fitting": False,
              "new_inference": False, "network_used": False, "membership": membership}
    try:
        frame = load_extra_passages(root, membership)
        rows, prompts = len(frame), frame["prompt_id"].nunique()
        checks = {"source_files_present": bool(membership), "author_row_count_matches": rows == expected_rows,
                  "author_prompt_count_matches": prompts == expected_prompts}
        report.update({"status": "PASS_EXTRA_MERGE_COUNTS" if all(checks.values()) else "FAIL_EXTRA_MERGE_COUNTS",
                       "checks": checks, "observed": {"rows": rows, "prompts": int(prompts),
                           "class_counts": {label: int(frame["model"].eq(label).sum()) for label in LABELS},
                           "labels_per_prompt_counts": {str(k): int(v) for k, v in frame.groupby("prompt_id")["model"].nunique().value_counts().sort_index().items()}},
                       "expected": {"rows": expected_rows, "prompts": expected_prompts},
                       "new_checkpoint_metadata": checkpoint_meta_summaries(root)})
    except MissingColumns as exc:
        report.update({"status": "FAIL_EXTRA_MERGE_SCHEMA", "failed_file": exc.path,
                       "missing_columns": exc.columns})
    except Exception as exc:
        report.update({"status": "FAIL_EXTRA_MERGE_EXCEPTION", "exception_type": type(exc).__name__,
                       "raw_exception_message_omitted": True})
    report["limits"] = [
        "Matching 7644 rows and 509 prompts does not prove identical merged text, source versions, provider identity, or training provenance.",
        "The copied source assumes prompt-bank identity except for passages_clean.csv; this audit preserves that assumption rather than repairing it.",
        "Official response overlap and the separate 34940-row pool are handled by the separate input checker, not this merge audit.",
        "No passage text, individual prompt ID, model tensor, or row-level prediction is emitted.",
    ]
    return report


def self_test():
    import tempfile
    tests = {}
    def text(key, words=30):
        return " ".join([key] + ["word"] * (words - 1))
    def block(prompt, start=0, count=12, words=30):
        return [{"prompt_id": prompt, "model": label, "text": text(f"p{prompt}x{i+start}", words)}
                for i, label in enumerate(LABELS[:count])]
    with tempfile.TemporaryDirectory(prefix="stage08_extra_audit_test_") as temp:
        root = Path(temp)
        csv = root / "gen_passages.csv"
        pd.DataFrame(block(1)).to_csv(csv, index=False)
        report = run(root, 12, 1)
        tests["valid_12_label_prompt_passes"] = report["status"] == "PASS_EXTRA_MERGE_COUNTS"
        tests["file_hash_receipt_recorded"] = report["membership"][0]["sha256"] == sha256(csv)
        tests["wrong_expected_counts_fail"] = run(root, 13, 1)["status"] == "FAIL_EXTRA_MERGE_COUNTS"
        csv.unlink()
        tests["missing_inputs_fail_closed"] = run(root)["status"] == "FAIL_EXTRA_MERGE_COUNTS"

        pd.DataFrame(block(1, count=11)).to_csv(csv, index=False)
        tests["under_12_labels_removed"] = load_extra_passages(root, [])["text"].empty
        # First duplicate prompt/model wins; model/text duplicates disappear across prompts.
        rows = block(1) + [{"prompt_id": 1, "model": LABELS[0], "text": text("later-different")}]
        rows += [{"prompt_id": 2, "model": LABELS[0], "text": rows[0]["text"]}]
        pd.DataFrame(rows).to_csv(csv, index=False)
        frame = load_extra_passages(root, [])
        tests["keep_first_prompt_model_and_model_text"] = len(frame) == 12 and frame.iloc[0]["text"] == rows[0]["text"]

        rows = block(1)
        rows[0]["text"] = "  **" + text("bold").replace(" ", "  ") + "**  "
        rows += [{"prompt_id": 1, "model": "unknown", "text": text("unknown")}]
        pd.DataFrame(rows).to_csv(csv, index=False)
        frame = load_extra_passages(root, [])
        tests["bold_whitespace_and_unknown_label_protocol"] = len(frame) == 12 and frame.iloc[0]["text"] == text("bold")

        rows = block(1, words=30) + block(2, words=300) + block(3, words=29) + block(4, words=301)
        pd.DataFrame(rows).to_csv(csv, index=False)
        frame = load_extra_passages(root, [])
        tests["word_limits_30_and_300_inclusive"] = len(frame) == 24 and set(frame.prompt_id) == {"1", "2"}

        csv.unlink()
        rows = block(1)
        pd.DataFrame(rows).to_csv(root / "passages_clean.csv", index=False)
        pd.DataFrame(block(1, start=50)).to_csv(csv, index=False)
        frame = load_extra_passages(root, [])
        tests["passages_clean_uses_separate_pv_bank"] = len(frame) == 24 and set(frame.prompt_id) == {"1", "pv_1"}

        for file in root.glob("*.csv"):
            file.unlink()
        rows = block(1) + block(2)
        for row in rows:
            row["label"] = row.pop("model")
            row["finish_reason"] = "stop" if row["prompt_id"] == 1 else "length"
        pd.DataFrame(rows).to_csv(root / "ntl_generations.csv", index=False)
        tests["finish_reason_stop_and_label_rename"] = len(load_extra_passages(root, [])) == 12
        (root / "ntl_generations.csv").unlink()
        rows = block(1)
        pd.DataFrame(rows).to_csv(root / "ntl_generated_trial.csv", index=False)
        tests["absent_optional_finish_column_preserves_source_behavior"] = len(load_extra_passages(root, [])) == 12
        (root / "ntl_generated_trial.csv").unlink()
        pd.DataFrame({"prompt_id": [1], "model": [LABELS[0]]}).to_csv(csv, index=False)
        tests["missing_required_column_fails_closed"] = run(root)["status"] == "FAIL_EXTRA_MERGE_SCHEMA"
    return {"stage": "08_extra_merge_synthetic_tests", "status": "PASS" if all(tests.values()) else "FAIL",
            "tests": tests, "real_inputs_read": False, "model_code_executed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    report = self_test() if args.self_test else run(args.input_root)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["status"].startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
