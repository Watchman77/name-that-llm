"""Package an offline CPU check of the eight pinned prefit vectorizers."""

import argparse
import base64
import hashlib
import json
from pathlib import Path
import zlib


def build(root):
    files = ("stage14_tfidf_safe_load.py", "stage14_fullpath_executable_inventory.py")
    sources = {}
    for name in files:
        raw = (root / "scripts" / name).read_bytes()
        compile(raw, name, "exec")
        sources[name] = (base64.b64encode(zlib.compress(raw, 9)).decode(), hashlib.sha256(raw).hexdigest())
    inventory = json.loads((root / "results/stage_14/V25_REPAIRED_STATIC_INVENTORY.json").read_bytes())
    prefit = inventory["static_inventory"]["checks"]["prefit"]
    vectors = [row for rows in prefit["active_readout_targets"].values() for row in rows
               if row["path"].endswith(".joblib")]
    if len(vectors) != 8:
        raise ValueError("expected the eight observed prefit vectorizers")
    cell = f'''import base64, hashlib, json, sys, tempfile, zipfile, zlib
from pathlib import Path

Path('/kaggle/temp').mkdir(exist_ok=True)
with tempfile.TemporaryDirectory(prefix='stage14c_prefit_cpu_', dir='/kaggle/temp') as tmp:
    root = Path(tmp)
    for name, (payload, digest) in {sources!r}.items():
        raw = zlib.decompress(base64.b64decode(payload, validate=True))
        if hashlib.sha256(raw).hexdigest() != digest:
            raise RuntimeError('Source fingerprint mismatch')
        (root / name).write_bytes(raw)
    sys.path.insert(0, tmp)
    from stage14_fullpath_executable_inventory import Segments
    from stage14_tfidf_safe_load import inspect_prefit_vectorizer
    mount = Path('/kaggle/input')
    parts = []
    for expected in {prefit['parts']!r}:
        file = mount / expected['path']
        with file.open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        if file.stat().st_size != expected['bytes'] or digest != expected['sha256']:
            raise RuntimeError('Pinned archive part changed')
        parts.append(file)
    result = {{}}
    with Segments(parts) as stream, zipfile.ZipFile(stream) as archive:
        for expected in {vectors!r}:
            rel = expected['path']
            info = archive.getinfo(rel)
            if info.file_size != expected['bytes']:
                raise RuntimeError('Unexpected vectorizer byte size')
            data = archive.read(info)
            if hashlib.sha256(data).hexdigest() != expected['sha256']:
                raise RuntimeError('Pinned vectorizer changed')
            local = root / rel
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(data)
            result[rel] = inspect_prefit_vectorizer(local)
            print('STAGE14C_PREFIT_VECTOR', rel, json.dumps(result[rel], sort_keys=True), flush=True)
    report = {{'status': 'STAGE14C_RESTRICTED_PREFIT_CPU_PASS', 'vectorizers': result,
               'ordinary_joblib_deserialization_performed': False, 'official_score_measured': False}}
    (Path('/kaggle/working') / 'stage14c_restricted_prefit_cpu.json').write_text(json.dumps(report, sort_keys=True))
    print(report['status'], len(result), flush=True)
'''
    compile(cell, "stage14c_prefit_cpu_cell.py", "exec")
    return cell


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.write_text(build(Path(__file__).resolve().parent.parent), encoding="utf-8")
