"""Build private CPU notebooks for outcome-blind admission or reused-prompt stress.

This script never opens competition data or runs an experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def source_cell(root, names, runner, arguments, timeout):
    sources = {}
    for name in names:
        data = (root / "scripts" / name).read_bytes()
        decoded = data.decode("utf-8")
        compile(decoded, name, "exec")
        sources[name] = (decoded, hashlib.sha256(data).hexdigest())
    cell = f'''from pathlib import Path
import hashlib
import subprocess
import sys

SOURCES = {sources!r}
root = Path("/kaggle/working/_stage13e_source")
root.mkdir(parents=True, exist_ok=True)
for name, (source, expected) in SOURCES.items():
    path = root / name
    data = source.encode("utf-8")
    if hashlib.sha256(data).hexdigest() != expected:
        raise RuntimeError("Stage 13E source checksum mismatch")
    path.write_bytes(data)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise RuntimeError("Stage 13E written source checksum mismatch")
print("STAGE13E_SOURCE_CHECKS_PASS", SOURCES[{runner!r}][1], flush=True)
subprocess.run([sys.executable, "-B", str(root / {runner!r}), *{arguments!r}],
               check=True, timeout={timeout!r})
'''
    compile(cell, "stage13e_notebook_cell.py", "exec")
    return cell


def notebook(root, mode, dataset_version, dataset_source_id):
    inputs = [{"sourceType": "competition", "sourceId": 170484}]
    if mode == "audit":
        if not dataset_version or not dataset_source_id:
            raise ValueError("Audit requires explicit observed corpus version and Kaggle input ID")
        inputs.append({"sourceType": "datasetVersion", "sourceId": dataset_source_id})
        runner = "stage13e_prvsiyan_admission.py"
        names = [runner]
        args = ["--dataset-version", str(dataset_version), "--out", "/kaggle/working/stage13e_prvsiyan_admission.json"]
        timeout = 300
        title = "Stage 13E — external panel audit only"
        purpose = "No fit, inference, accuracy, row-level output, or official submission."
    else:
        inputs += [{"sourceType": "kernelVersion", "sourceId": v} for v in (359405443, 358745836)]
        runner = "stage13e_partition_check.py"
        names = [runner, "stage13_glm_readout.py", "stage13a_cache_admission.py", "stage13b_balancing_stress.py",
                 "stage13c_deberta_complement.py", "stage13d_broader_complement.py"]
        args, timeout = [], 1200
        title = "Stage 13E — alternate split stress check"
        purpose = "Exploratory refits on the same 150 reused prompts. No fresh holdout or official score."
    code = source_cell(root, names, runner, args, timeout)
    return dict(nbformat=4, nbformat_minor=5, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "kaggle": {"accelerator": "none", "dataSources": inputs, "isInternetEnabled": False,
                   "language": "python", "sourceType": "notebook", "isGpuEnabled": False},
    }, cells=[
        dict(cell_type="markdown", id="stage13e-scope", metadata={}, source=[f"# {title}\n", "\n", purpose + "\n"]),
        dict(cell_type="code", id="stage13e-run", metadata={}, execution_count=None, outputs=[],
             source=code.splitlines(keepends=True)),
    ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("audit", "stress"))
    parser.add_argument("--dataset-version", type=int)
    parser.add_argument("--dataset-source-id", type=int)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    nb = notebook(root, args.mode, args.dataset_version, args.dataset_source_id)
    args.out.write_text(json.dumps(nb, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
