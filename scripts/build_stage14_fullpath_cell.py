"""Build a private operational canary from exact, locally provided Alex V25 cells.

The generated cell contains the third-party source. Keep it private. Neither
this builder nor the notebook creates a competition submission.
"""

import argparse
import base64
import hashlib
import json
from pathlib import Path
import re
import zlib

from stage14_fullpath_guard import (V25_CANARY_BOOTSTRAP, V25_CANARY_FINISH, fix_v25_qwen_checkpoint_selection,
                                   overlay_v25, replace_exact)
from build_stage14_lora_anchor_cell import ADAPTER_PATHS, CAUSAL_BASES, extract


def causal_input_lock(receipt):
    """Require a passed three-row GPU anchor receipt before building the repaired variant."""
    if (receipt['status'] != 'PASS_V25_LORA_ANCHORS' or receipt['threshold_strictly_below'] != 0.02
            or receipt['rows'] != 3 or set(receipt['anchors']) != set(ADAPTER_PATHS)
            or any(receipt[k] is not False for k in ('official_accuracy_measured', 'test_rows_read', 'full_ensemble_run'))):
        raise ValueError('A passing private anchor receipt with the original threshold is required')
    locked = {f'{base}/{name}': (None, sha) for base, files in CAUSAL_BASES.values() for name, sha in files.items()}
    for member, path in ADAPTER_PATHS.items():
        anchor = receipt['anchors'][member]
        base_key = ('qwen2', 3584) if member == 'lora20_q25_instyle' else ('qwen3', 4096)
        head_tensor = 'norms.safetensors' if member in ('lora24_seed7', 'lora20_q25_instyle') else 'heads.safetensors'
        anchor_features = 'valid_features.npz' if head_tensor == 'norms.safetensors' else 'anchor_features.npz'
        expected = {'COMPLETE.json', 'adapter/adapter_config.json', 'adapter/adapter_model.safetensors',
                    'tokenizer/tokenizer.json', 'tokenizer/tokenizer_config.json', head_tensor, anchor_features}
        files = anchor['files_sha256']
        if (anchor['adapter'] != path or anchor['base'] != CAUSAL_BASES[base_key][0] or anchor['passes'] is not True
                or anchor['rows'] != 3 or not 0 <= anchor['mean_rel'] < 0.02 or set(files) != expected
                or not all(re.fullmatch(r'[a-f0-9]{64}', digest) for digest in files.values())):
            raise ValueError(f'Invalid passed anchor receipt for {member}')
        locked.update({f'{path}/{name}': (None, sha) for name, sha in files.items()})
    return locked


def build(root, sources_dir, anchor_receipt=None):
    originals = [(sources_dir / f"stage14c_alex_v25_cell_{i}.txt").read_text(encoding="utf-8") for i in range(20)]
    cells = overlay_v25(originals)
    finish_cell = V25_CANARY_FINISH
    adapter_bindings = {}
    if anchor_receipt is not None:
        config_sha = CAUSAL_BASES[('qwen3', 4096)][1]['config.json']
        comp = cells[16].split('\n', 1)[1]
        adapter_bindings = {marker: f'{ADAPTER_PATHS[key]}/COMPLETE.json'
                            for _, _, specs in extract(comp)['LORA_RUNS'] for key, marker, *_ in specs}
        cells[16] = fix_v25_qwen_checkpoint_selection(cells[16], config_sha)
        finish_cell = replace_exact(finish_cell, 'Path.cwd(), pd.read_parquet)',
                                                  'Path.cwd(), pd.read_parquet, causal_checkpoint=True)')
    # Diagnostic-only: rebuild the verified vectorizers from restricted data.
    cells[7] = replace_exact(cells[7], "import joblib\n", "from stage14_tfidf_safe_load import load_prefit_vectorizer\n")
    cells[7] = replace_exact(cells[7], "joblib.load(self.folder / e['file'])", "load_prefit_vectorizer(self.folder / e['file'])")
    cells = [V25_CANARY_BOOTSTRAP, *cells[1:], finish_cell]  # no notebook chart in operational probe
    code = []
    for i, cell in enumerate(cells):
        if cell.startswith("%%writefile "):
            name, body = cell.removeprefix("%%writefile ").split("\n", 1)
            if Path(name).name != name or not name.endswith(".py"):
                raise ValueError("unexpected write target")
            compile(body, name, "exec")
            cell = f"from pathlib import Path\nPath({name!r}).write_text({body!r}, encoding='utf-8')\n"
        compile(cell, f"v25-canary-{i}", "exec")
        code.append(cell)
    inventory = json.loads((root / "results/stage_14/V25_REPAIRED_STATIC_INVENTORY.json").read_bytes())
    original = json.loads((root / "results/stage_14/V25_ORIGINAL_MOUNT_AUDIT.json").read_bytes())["report"]
    sections = inventory["static_inventory"]["checks"]
    records = [sections["prefit"]["manifest"], *sections["prefit"]["parts"],
               *sections["main_torch_heads"].values(), *sections["qwen25"]["metadata"],
               sections["gsq"]["final"], sections["gsq"]["manifest"], sections["gsq"]["complete"],
               original["official_train_identity"], *original["llama_mirror"]["metadata"].values()]
    records += [x for v in original["offline_wheels"].values() for x in v]
    records += [x for v in original["dataset_inventory"].values() for x in v]
    records += [x for f in sections["ultrabert"] for x in (*f["python"], f["config"], f["tokenizer_config"])]
    locked = {x["path"]: (x["bytes"], x["sha256"]) for x in records}
    if len(locked) < 60:
        raise ValueError("incomplete active input lock")
    if anchor_receipt is not None:
        for key, (size, digest) in causal_input_lock(anchor_receipt).items():
            if key in locked and locked[key][1] != digest:
                raise ValueError(f'Input lock disagrees with anchor probe for {key}')
            locked.setdefault(key, (size, digest))
    extras = {}
    for name in ("stage14_fullpath_guard.py", "stage14_native_input_audit.py", "stage14_tfidf_safe_load.py"):
        raw = (root / "scripts" / name).read_bytes()
        compile(raw, name, "exec")
        extras[name] = raw.decode("utf-8")
    encoded = base64.b64encode(zlib.compress(json.dumps([code, extras, locked], ensure_ascii=False).encode(), 9)).decode()
    wheels = {x["path"] for v in original["offline_wheels"].values() for x in v}
    native_digest = json.loads((root / "results/stage_14/CPU_KAGGLE_V1.json").read_bytes())["results"][0]["task_asset_tree_sha256"]
    rendered = f'''import base64, glob, hashlib, json, os, sys, tempfile, zlib
from pathlib import Path

if os.getenv('KAGGLE_IS_COMPETITION_RERUN') is not None:
    raise RuntimeError('Private diagnostic cannot run as a competition submission')
cells, helpers, locked = json.loads(zlib.decompress(base64.b64decode({encoded!r}, validate=True)))
input_root = Path('/kaggle/input')
observed_wheels = {{Path(p).relative_to(input_root).as_posix() for pattern in (
    '**/bitsandbytes-*.whl', '**/name-that-llm-artifacts/qwen38-offline-linux/*.whl')
    for p in glob.glob(str(input_root / pattern), recursive=True)}}
if observed_wheels != set({sorted(wheels)!r}):
    raise RuntimeError('Active wheel selection changed')
for relative, (size, digest) in locked.items():
    path = input_root / relative
    if not path.is_file() or (size is not None and path.stat().st_size != size):
        raise RuntimeError(f'Pinned input unavailable: {{relative}}')
    with path.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != digest:
            raise RuntimeError(f'Pinned input changed: {{relative}}')
for pattern, relative in {adapter_bindings!r}.items():
    resolved = {{Path(p).resolve() for p in glob.glob(str(input_root / pattern), recursive=True) if Path(p).is_file()}}
    if resolved != {{(input_root / relative).resolve()}}:
        raise RuntimeError(f'Anchor adapter mount changed: {{relative}}')
print('STAGE14C_LOCKED_INPUTS_VERIFIED', len(locked), flush=True)
root = Path('/kaggle/temp')
root.mkdir(exist_ok=True)
modules = Path(tempfile.mkdtemp(prefix='stage14c_helpers_', dir=root))
for name, source in helpers.items():
    (modules / name).write_text(source, encoding='utf-8')
sys.path.insert(0, str(modules))
os.environ['PYTHONPATH'] = str(modules)
os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '1'
from stage14_native_input_audit import audit as audit_native
native_roots = {{p.resolve() for p in input_root.rglob('ntl-native-q8-multidepth-fivefold-0679') if p.is_dir()}}
if len(native_roots) != 1:
    raise RuntimeError('Unexpected native mount selection')
audit_native(next(iter(native_roots)), {native_digest!r})
print('STAGE14C_NATIVE_INPUT_IDENTITY_VERIFIED', flush=True)
scope = {{'__name__': '__main__'}}
for i, source in enumerate(cells):
    print('STAGE14C_CANARY_CELL', i, flush=True)
    exec(compile(source, f'v25-canary-{{i}}', 'exec'), scope)
'''
    compile(rendered, "v25-canary-cell", "exec")
    return rendered


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--anchor-receipt", type=Path, help="Kaggle aggregate output from the passed private GPU probe")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    if args.out.resolve().is_relative_to(root):
        parser.error("write the private generated cell outside the repository")
    receipt = json.loads(args.anchor_receipt.read_bytes()) if args.anchor_receipt is not None else None
    args.out.write_bytes(build(root, args.source_dir, receipt).encode("utf-8"))
