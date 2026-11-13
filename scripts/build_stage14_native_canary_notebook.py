"""Build a private, offline two-T4 NativeQ8 notebook from reviewed local source.

The generated notebook includes the supplied third-party worker for a private
canary. Keep the generated notebook outside the repo; commit this builder only.
"""

import argparse
import hashlib
import json
from pathlib import Path

from stage14_native_canary import sha256


COMPETITION_ID = 170484
SCRIPTS = ("stage14_native_input_audit.py", "stage14_native_canary.py")


def build(dataset_version_id, model_version_id, task_sha256, worker_path, worker_sha256):
    for name, value in (("datasetVersion", dataset_version_id), ("modelInstanceVersion", model_version_id)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"An explicit numeric Kaggle {name} source ID is required")
    cell = build_cell(task_sha256, worker_path, worker_sha256)
    return {"nbformat": 4, "nbformat_minor": 5, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "kaggle": {"accelerator": "nvidiaTeslaT4", "dataSources": [
            {"sourceType": "competition", "sourceId": COMPETITION_ID},
            {"sourceType": "datasetVersion", "sourceId": dataset_version_id},
            {"sourceType": "modelInstanceVersion", "sourceId": model_version_id},
        ], "isInternetEnabled": False, "isGpuEnabled": True, "isPrivate": True,
            "language": "python", "sourceType": "notebook"},
    }, "cells": [
        {"cell_type": "markdown", "id": "stage14-native-gpu-scope", "metadata": {},
         "source": ["# Stage 14: Native Qwen GPU feasibility\n",
                    "Keep the notebook private, Internet off, and accelerator GPU T4 x2.\n",
                    "Isolated inference on 32 official train rows. No predictions, score, or submission.\n"]},
        {"cell_type": "code", "id": "stage14-native-gpu-canary", "metadata": {},
         "execution_count": None, "outputs": [], "source": cell.splitlines(keepends=True)},
    ]}


def build_cell(task_sha256, worker_path, worker_sha256):
    """Build the cell for a notebook whose exact mounts have been pinned in the editor."""
    sha256(task_sha256)
    sha256(worker_sha256)
    if not 0 < worker_path.stat().st_size <= 100_000:
        raise ValueError("Unexpected reviewed worker size")
    worker = worker_path.read_bytes()
    if hashlib.sha256(worker).hexdigest() != worker_sha256:
        raise ValueError("Reviewed worker fingerprint mismatch")
    files = {name: Path(__file__).with_name(name).read_bytes() for name in SCRIPTS}
    files["reviewed_native_q8.py"] = worker
    payloads = {}
    for name, data in files.items():
        compile(data, name, "exec")
        payloads[name] = (data.decode("utf-8"), hashlib.sha256(data).hexdigest())
    cell = f'''import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile

inputs = Path("/kaggle/input")
slug = "ntl-olmo2-native24-final-inference-assets"
tree = "ntl-native-q8-multidepth-fivefold-0679"

def select(paths, predicate, label):
    hits = {{p.resolve() for p in paths if predicate(p)}}
    if len(hits) != 1:
        raise RuntimeError(f"Expected exactly one {{label}} mount, found {{len(hits)}}")
    return hits.pop()

assets = select((p / name / tree for p in (inputs / "datasets/lightningv08", inputs)
                 for name in (slug, "name-that-llm-artifacts")),
                Path.is_dir, "native adapter tree")
train = select((inputs / "competitions/name-that-llm/train.csv", inputs / "name-that-llm/train.csv"),
               Path.is_file, "competition train.csv")
base = select((inputs / "models/qwen-lm/qwen-3-embedding/transformers/8b/1",
               inputs / "qwen-3-embedding/transformers/8b/1"), Path.is_dir, "Qwen3-Embedding-8B V1")
report = Path("/kaggle/working/native_q8_canary.json")
temp_root = Path("/kaggle/temp")
temp_root.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory(prefix="stage14-native-code-", dir=temp_root) as temp:
    root = Path(temp)
    for name, (source, sha) in {payloads!r}.items():
        target = root / name
        target.write_bytes(source.encode("utf-8"))
        if hashlib.sha256(target.read_bytes()).hexdigest() != sha:
            raise RuntimeError(f"Source fingerprint mismatch: {{name}}")
    command = [sys.executable, "-B", "-u", str(root / "stage14_native_canary.py"),
               "--assets", str(assets), "--base", str(base), "--train-csv", str(train),
               "--worker", str(root / "reviewed_native_q8.py"), "--work", temp,
               "--report", str(report), "--task-sha256", {task_sha256!r},
               "--worker-sha256", {worker_sha256!r}]
    subprocess.run(command, check=True, timeout=1800)
'''
    compile(cell, "stage14_native_canary_notebook_cell.py", "exec")
    return cell


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-version-id", type=int, required=True)
    parser.add_argument("--model-version-id", type=int, required=True)
    parser.add_argument("--task-sha256", type=sha256, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--worker-sha256", type=sha256, required=True)
    parser.add_argument("--out", type=Path, required=True, help="private scratch notebook, outside the repo")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parent.parent
    if args.out.resolve().is_relative_to(repo):
        parser.error("write generated private notebook outside the repo")
    notebook = build(args.dataset_version_id, args.model_version_id, args.task_sha256,
                     args.worker, args.worker_sha256)
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(notebook, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
