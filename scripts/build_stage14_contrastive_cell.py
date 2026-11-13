"""Package the frozen offline Stage 14B training cell; does not run training."""

import argparse
import base64
import hashlib
from pathlib import Path
import zlib


FILES = (
    "stage14_contrastive_run.py", "stage14_contrastive_head.py", "stage14_contrastive_inputs.py",
    "stage14_contrastive_metrics.py", "stage14_native_oof_audit.py", "stage14_native_input_audit.py",
    "stage13a_cache_admission.py", "stage13_glm_readout.py", "stage13c_deberta_complement.py",
    "stage13b_balancing_stress.py", "stage13f_assignment_core.py",
)


def build():
    root = Path(__file__).resolve().parent
    sources = {}
    for name in FILES:
        data = (root / name).read_bytes()
        compile(data, name, "exec")
        sources[name] = (base64.b64encode(zlib.compress(data, 9)).decode("ascii"), hashlib.sha256(data).hexdigest())
    cell = f'''from pathlib import Path
import base64
import hashlib
import subprocess
import sys
import zlib

SOURCES = {sources!r}
source_dir = Path("/kaggle/temp/stage14b_training_sources")
source_dir.mkdir(parents=True, exist_ok=True)
for name, (payload, sha) in SOURCES.items():
    data = zlib.decompress(base64.b64decode(payload, validate=True))
    if hashlib.sha256(data).hexdigest() != sha:
        raise RuntimeError("Stage 14B payload identity mismatch")
    dest = source_dir / name
    dest.write_bytes(data)
    if hashlib.sha256(dest.read_bytes()).hexdigest() != sha:
        raise RuntimeError("Stage 14B source identity mismatch")
print("STAGE14B_SOURCE_VERIFIED", SOURCES["stage14_contrastive_run.py"][1], flush=True)
subprocess.run([sys.executable, "-B", "-u", str(source_dir / "stage14_contrastive_run.py"),
                "--input-root", "/kaggle/input",
                "--report", "/kaggle/working/stage14b/paired_report.json"],
               check=True, timeout=1800)
'''
    compile(cell, "stage14b_training_cell.py", "exec")
    return cell


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.write_bytes(build().encode("utf-8"))
