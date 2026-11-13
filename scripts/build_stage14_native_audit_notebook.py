"""Build the CPU-only Kaggle audit after the native hub version is resolved."""

import argparse
import hashlib
import json
from pathlib import Path


def build(dataset_version_id, expected_sha256=None):
    if type(dataset_version_id) is not int or dataset_version_id <= 0:
        raise ValueError("An explicit Kaggle dataset-version ID is required")
    if expected_sha256 is not None and (
        len(expected_sha256) != 64 or any(c not in "0123456789abcdef" for c in expected_sha256)
    ):
        raise ValueError("Expected a lowercase SHA-256 fingerprint")
    script = Path(__file__).with_name("stage14_native_input_audit.py").read_text(encoding="utf-8")
    digest = hashlib.sha256(script.encode("utf-8")).hexdigest()
    cell = f'''from pathlib import Path
import hashlib
import subprocess
import sys

source = {script!r}
sha = {digest!r}
input_root = Path("/kaggle/input")
slug = "ntl-olmo2-native24-final-inference-assets"
tree = "ntl-native-q8-multidepth-fivefold-0679"
paths = [input_root / "datasets" / "lightningv08" / slug / tree,
         input_root / slug / tree]
mounts = {{p.resolve() for p in paths if p.is_dir()}}
train_mounts = {{p.resolve() for p in (input_root / "competitions/name-that-llm/train.csv",
                                      input_root / "name-that-llm/train.csv") if p.is_file()}}
if len(mounts) != 1 or len(train_mounts) != 1:
    raise RuntimeError("Expected one exact hub mount and one competition train mount")
out = Path("/kaggle/working/stage14_native_input_audit")
out.mkdir(parents=True, exist_ok=True)
runner = out / "audit.py"
runner.write_bytes(source.encode("utf-8"))
if hashlib.sha256(runner.read_bytes()).hexdigest() != sha:
    raise RuntimeError("Audit code fingerprint mismatch")
cmd = [sys.executable, "-B", str(runner), "--assets", str(next(iter(mounts))),
       "--train-csv", str(next(iter(train_mounts)))]
expected = {expected_sha256!r}
if expected is not None:
    cmd.extend(["--expected-tree-sha256", expected])
result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
print(result.stdout, end="", flush=True)
result.check_returncode()
(out / "audit.json").write_text(result.stdout, encoding="utf-8")
print("STAGE14_NATIVE_CPU_AUDIT_COMPLETE", sha, flush=True)
'''
    compile(cell, "stage14_native_input_audit_cell.py", "exec")
    return {"nbformat": 4, "nbformat_minor": 5, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "kaggle": {"accelerator": "none", "dataSources": [
            {"sourceType": "competition", "sourceId": 170484},
            {"sourceType": "datasetVersion", "sourceId": dataset_version_id},
        ], "isInternetEnabled": False, "isGpuEnabled": False,
            "language": "python", "sourceType": "notebook"},
    }, "cells": [{"cell_type": "markdown", "id": "stage14-native-scope", "metadata": {},
                    "source": ["# Stage 14: Native Qwen CPU input audit\n",
                               "Structural and numeric checks only. No predictions or submission.\n"]},
                  {"cell_type": "code", "id": "stage14-native-audit", "metadata": {},
                   "execution_count": None, "outputs": [], "source": cell.splitlines(keepends=True)}]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-version-id", type=int, required=True,
                        help="numeric Kaggle datasetVersion source ID, not the displayed version ordinal")
    parser.add_argument("--expected-tree-sha256")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.write_text(json.dumps(build(args.dataset_version_id, args.expected_tree_sha256), indent=2) + "\n",
                        encoding="utf-8")
