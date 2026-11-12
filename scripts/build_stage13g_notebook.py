"""Build Stage 13G's private CPU notebook without reading data or running it."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


RUNNER = "stage13g_small_group_audit.py"
LOCKS = {
    "stage13b_balancing_stress.py": "f4981f385074f60ee034b979d4aa6672c7d28eaa408c374d5caf55a7c446fbd9",
    "stage13c_deberta_complement.py": "b9b6f523b8a6b1f9f1dae556c863f7fee34f56fddfb27adb0946021e1f17181e",
    "stage13f_assignment_audit.py": "be418f7603ec61a1920061bd063494eb17d609ad5c64495b3c404672adf0f720",
    "stage13f_assignment_core.py": "9d4a64db8fb3f2853234c10d98617f0219184d3269b54e0bba87b9883bdc8e8c",
}
INPUTS = [{"sourceType": "competition", "sourceId": 170484},
          {"sourceType": "kernelVersion", "sourceId": 368292220},
          {"sourceType": "kernelVersion", "sourceId": 358745836}]


def build(root):
    sources = {}
    for name in (RUNNER, "stage13g_small_group_core.py", *LOCKS):
        data = (Path(root) / "scripts" / name).read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        if name in LOCKS and sha != LOCKS[name]:
            raise ValueError(f"Frozen dependency changed: {name}")
        code = data.decode("utf-8")
        compile(code, name, "exec")
        sources[name] = (code, sha)
    cell = f'''from pathlib import Path
import hashlib
import subprocess
import sys

SOURCES = {sources!r}
source_dir = Path("/kaggle/working/_stage13g_sources")
source_dir.mkdir(parents=True, exist_ok=True)
for name, (code, sha) in SOURCES.items():
    data = code.encode("utf-8")
    path = source_dir / name
    if hashlib.sha256(data).hexdigest() != sha:
        raise RuntimeError("Stage 13G source checksum mismatch")
    path.write_bytes(data)
    if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
        raise RuntimeError("Stage 13G written source checksum mismatch")
print("STAGE13G_SOURCE_CHECKS_PASS", SOURCES[{RUNNER!r}][1], flush=True)
subprocess.run([sys.executable, "-B", str(source_dir / {RUNNER!r}),
                "--input-root", "/kaggle/input", "--output-root", "/kaggle/working/stage13g_small_group_audit"],
               check=True, timeout=900)
'''
    compile(cell, "stage13g_notebook_cell.py", "exec")
    return dict(nbformat=4, nbformat_minor=5, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "kaggle": {"accelerator": "none", "dataSources": INPUTS, "isInternetEnabled": False,
                   "language": "python", "sourceType": "notebook", "isGpuEnabled": False},
    }, cells=[
        dict(cell_type="markdown", id="stage13g-scope", metadata={}, source=[
            "# Stage 13G — small-group extension\n", "\n",
            "Audit the 810 previously untouched rows. Test only groups of 10 or 11 if the fixed geometry gate passes.\n",
            "Stage 13F assignment-on is the frozen control. Reused development prompts only.\n",
        ]),
        dict(cell_type="code", id="stage13g-run", metadata={}, execution_count=None, outputs=[],
             source=cell.splitlines(keepends=True)),
    ])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    root = Path(__file__).resolve().parent.parent
    args.out.write_text(json.dumps(build(root), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
