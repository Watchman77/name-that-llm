"""Build a private V25 adapter parity probe; retain released source locally.

The generated cell embeds exact reviewed portions of the public notebook.
Keep it private. This does not fit any head or read competition test data.
"""

import argparse
import ast
import hashlib
from pathlib import Path

from stage14_fullpath_guard import V25_CODE_SHA256, normalize_rendered_cell, replace_exact
from stage14_native_input_audit import TRAIN_SHA256


CAUSAL_BASES = {
    ("qwen3", 4096): ("models/qwen-lm/qwen-3/transformers/8b-base/1", {
        "config.json": "3bd01d7ad7a2e203ecbbe84e24087a51c6d2a108ee4bcc42d0016bf49564983a",
        "model.safetensors.index.json": "cd39d5e44c5aadfd1954b598883aea1c4a98822a2f7723d214ef2278b7fec2e4",
        "tokenizer.json": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
        "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
        "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
        "tokenizer_config.json": "5a7303fcb1a27ede63134a2cbd61d5282c247ca6d769ce4746d4ffa124aedd63",
    }),
    ("qwen2", 3584): ("models/qwen-lm/qwen2.5/transformers/7b/1", {
        "config.json": "267ce68584c5f24c3b267d934db2de68dd21d1ca677fb78ed809eb60067f7642",
        "model.safetensors.index.json": "624bf7c47cd12468fdc16e38a47cf4f19e0415b859a223ba3c027eed2f0e1028",
        "tokenizer.json": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
        "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
        "merges.txt": "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
        "tokenizer_config.json": "c91efca15ceff6e9ee9424db58a6f59cd41294e550a86cbd07e3c1fb500b34f9",
    }),
}

ADAPTER_PATHS = {
    "lora24_public": "ntl-olmo2-native24-final-inference-assets/notebook-inputs/notebook-inputs/ntl-lora24-fivefold-training/lora24/final",
    "lora24_seed7": "notebooks/lucasboesen/name-that-llm-train-lora24-s7/lora_out/job_final",
    "lora24_claude": "datasets/zbigniewkirchner/ntl-qwen3-8b-lora-claude-head/final",
    "lora20_q25_instyle": "notebooks/vnton37nguyn/ntl-train-lora-qwen25-instyle/lora_out/job_final",
}

KNOWN_ADAPTER_SHA256 = {
    "lora24_seed7": (
        "e92a525842d44c1d4a7f3c55bbcdfdc46ed96a545b90b80717fc86f6cb22c2d5",
        "3b786d83737e6890ccb19a2cddbaee5c8b1aae1bc2c13840161970909165711c",
        "a045cbdc521b957736e828a3d8f1eee4a844e5ce1a8c2a17e1b0531f86212682",
        "709a8214889291cc9d779928bbdf3ba4849d97e023c005f6ba11f9b704d75126",
        "b3744d3112b8c3d04760859bfdbf9a7ce00d5539512b5fc5d5bfb3f76eec0d9d",
        "2e3bd3cb5055f7a903a54e7062b2fd14d21b70f4174c3ffcd9fc81ec50a25631",
    ),
    "lora24_claude": (
        "4a9f1ee827240c221c2e83bc0686df644aa3d16c2cef08e0b5e1e71c463b2a54",
        "e37c54313ad1d0886f0f4562e82e7934cea51b9433f7465fca9df9f765fe6176",
        "9aed9517bf5954c90c14ad22a95af64b90c3ad23407de2910c05bc18c804c505",
        "79fcbbc4b313450b4580565d5addba4cd769e80f10d5d68a2068e694b50619aa",
        "b3744d3112b8c3d04760859bfdbf9a7ce00d5539512b5fc5d5bfb3f76eec0d9d",
        "2e3bd3cb5055f7a903a54e7062b2fd14d21b70f4174c3ffcd9fc81ec50a25631",
    ),
    "lora20_q25_instyle": (
        "ed24f3ec9e73878b99ee97730591ad17ebdff69e4f59e3f3dee90ea29d8ded8d",
        "01f4767094380c58a8092eae287325c7b97f47905132076776d60afbe8e6af8e",
        "509626893eb960bf6ce0fc9075c7a3c196ffa4994aba46dcf44508b8ab8c646b",
        "29b8845f58771d6a0756cf66e3b303110b53e7e8f5939013706abf93e2db50ec",
        "0bf86e7bd1b719430e67169321519a1ba441a691eb763fd7abb0dc305eb41120",
        "5f007d04324afa93876e2a484190e868136c69dfe0e1a29a1063eae583df9d32",
    ),
}


def source_cell(source_dir, i, filename):
    source = normalize_rendered_cell((source_dir / f"stage14c_alex_v25_cell_{i}.txt").read_text(encoding="utf-8"))
    heading = f"%%writefile {filename}\n"
    if hashlib.sha256(source.encode()).hexdigest() != V25_CODE_SHA256[i] or not source.startswith(heading):
        raise ValueError(f"public source fingerprint mismatch for cell {i}")
    return source[len(heading):]


def extract(source):
    tree = ast.parse(source)
    leaves = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {"LORA_RUNS", "LORA_MAX_LEN"}:
                    leaves[target.id] = ast.literal_eval(node.value)
                    if target.id == "LORA_MAX_LEN":
                        leaves[target.id] = leaves[target.id][0]
                elif isinstance(target, ast.Tuple) and any(isinstance(e, ast.Name) and e.id == "LORA_MAX_LEN" for e in target.elts):
                    leaves["LORA_MAX_LEN"] = ast.literal_eval(node.value)[0]
        elif isinstance(node, ast.FunctionDef) and node.name == "score_lora_checkpoints":
            lines = source.splitlines(keepends=True)
            first = min(x.lineno for x in node.decorator_list)
            leaves[node.name] = "".join(lines[first - 1:node.end_lineno])
    if set(leaves) != {"LORA_RUNS", "LORA_MAX_LEN", "score_lora_checkpoints"}:
        raise ValueError("parity probe source bindings differ")
    if {spec[0] for x in leaves["LORA_RUNS"] for spec in x[2]} != set(ADAPTER_PATHS):
        raise ValueError("released adapter list differs")
    return leaves


def build(source_dir):
    public_pipe = source_cell(source_dir, 3, "pipe.py")
    pieces = extract(source_cell(source_dir, 16, "comp_stack.py"))
    public_fn = pieces["score_lora_checkpoints"]
    load_candidate = '''        hits = sorted(glob.glob('/kaggle/input/' + marker, recursive=True))
        if not hits:
            print(f'  SKIP {key}: checkpoint not attached ({marker})', flush=True); continue
        ck = pathlib.Path(os.path.dirname(hits[0]))'''
    public_fn = replace_exact(public_fn, load_candidate, "        ck, provenance = check_adapter(key, marker, norms_file, anchor_file, base)")
    public_fn = replace_exact(public_fn, "    cfg = AutoConfig.from_pretrained(base)", "    cfg = AutoConfig.from_pretrained(base, local_files_only=True, trust_remote_code=False)")
    public_fn = replace_exact(public_fn, "dtype=torch.float16, device_map='auto').eval()", "dtype=torch.float16, device_map='auto', local_files_only=True, trust_remote_code=False, use_safetensors=True).eval()")
    public_fn = replace_exact(public_fn, "AutoTokenizer.from_pretrained(ck / 'tokenizer')", "AutoTokenizer.from_pretrained(ck / 'tokenizer', local_files_only=True, trust_remote_code=False)")
    public_fn = replace_exact(public_fn, "PeftModel.from_pretrained(trunk, ck / 'adapter').eval()", "PeftModel.from_pretrained(trunk, ck / 'adapter', local_files_only=True).eval()")
    public_fn = replace_exact(public_fn, "        ro = np.load(ck / 'readouts.npz')\n        assert list(ro['classes']) == LABELS and list(ro['pool_layers']) == pool_layers\n", "")
    public_fn = replace_exact(public_fn, "            saved = np.load(ck / anchor_file)['features'].astype(np.float32)", "            with np.load(ck / anchor_file, allow_pickle=False) as data:\n                saved = data['features'].astype(np.float32)")
    public_fn = replace_exact(public_fn, "            rel = float(np.abs(ours - saved).mean() / (np.abs(saved).mean() + 1e-9))", "            require(ours.shape == saved.shape == (3, H * len(pool_layers)) and np.isfinite(ours).all() and np.isfinite(saved).all(), 'anchor feature schema: ' + key)\n            rel = float(np.abs(ours - saved).mean() / (np.abs(saved).mean() + 1e-9))")
    start = public_fn.index("            assert rel < 0.02,")
    end = public_fn.index("        trunk = lm.unload()", start)
    public_fn = public_fn[:start] + '''            records[key] = {**provenance, 'max_abs': float(np.abs(ours - saved).max()), 'mean_rel': rel, 'passes': rel < 0.02, 'rows': 3}
            print('STAGE14C_LORA_ANCHOR', key, json.dumps(records[key], allow_nan=False, sort_keys=True), flush=True)
''' + public_fn[end:]
    try_start = public_fn.index("        ck, provenance = check_adapter")
    try_stop = public_fn.index("        trunk = lm.unload()", try_start)
    try_block = "".join("    " + line if line.strip() else line for line in public_fn[try_start:try_stop].splitlines(keepends=True))
    public_fn = public_fn[:try_start] + "        lm, provenance = None, {}\n        try:\n" + try_block + '''        except Exception as exc:
            records[key] = {**provenance, 'passes': False, 'error': f'{type(exc).__name__}: {exc}'}
            print('STAGE14C_LORA_ANCHOR_ERROR', key, json.dumps(records[key], sort_keys=True), flush=True)
            if lm is None and any('.lora_' in name for name, _ in trunk.named_parameters()):
                raise RuntimeError('adapter partially loaded; original trunk no longer assured') from exc
        finally:
            if lm is not None:
                trunk = lm.unload()
                del lm
            gc.collect()
            torch.cuda.empty_cache()
    del trunk
    gc.collect()
    torch.cuda.empty_cache()
'''
    cell = rf'''import csv, gc, glob, hashlib, itertools, json, os, pathlib, re, types, zipfile
from pathlib import Path

ROOT = Path('/kaggle/input')
BASES = {CAUSAL_BASES!r}
ADAPTERS = {ADAPTER_PATHS!r}
KNOWN_SHA = {KNOWN_ADAPTER_SHA256!r}

def require(value, message):
    if not value:
        raise RuntimeError(message)

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def file_json(path):
    require(path.is_file() and 0 < path.stat().st_size < 8 * 1024 * 1024, 'missing or invalid JSON: ' + path.name)
    return json.loads(path.read_bytes())

def find_checkpoint(model_type, hidden):
    rel, expected = BASES[(model_type, hidden)]
    path = ROOT / rel
    for key, digest in expected.items():
        require(sha(path / key) == digest, 'official base fingerprint: ' + rel + '/' + key)
    c = file_json(path / 'config.json')
    require((c['model_type'], c['hidden_size']) == (model_type, hidden) and not c.get('auto_map'), 'unexpected base configuration')
    require(c.get('architectures') == [{{'qwen3': 'Qwen3ForCausalLM', 'qwen2': 'Qwen2ForCausalLM'}}[model_type]], 'not a causal base release')
    shards = set(file_json(path / 'model.safetensors.index.json')['weight_map'].values())
    require(bool(shards) and all(re.fullmatch(r'model-\d{{5}}-of-\d{{5}}\.safetensors', n) and (path / n).is_file() for n in shards), 'missing official base shards')
    print('STAGE14C_LORA_CAUSAL_BASE', rel, len(shards), flush=True)
    return str(path)

def check_adapter(key, marker, norms_file, anchor_file, base):
    path = ROOT / ADAPTERS[key]
    found = {{Path(p).resolve() for p in glob.glob(str(ROOT / marker), recursive=True) if Path(p).is_file()}}
    require(found == {{(path / 'COMPLETE.json').resolve()}}, 'ambiguous or changed adapter binding: ' + key)
    names = ('COMPLETE.json', 'adapter/adapter_config.json', 'adapter/adapter_model.safetensors', norms_file, 'tokenizer/tokenizer.json', 'tokenizer/tokenizer_config.json', anchor_file)
    limits = (2**20, 2**20, 300*2**20, 20*2**20, 30*2**20, 2**20, 4*2**20)
    require(all((path / n).is_file() and 0 < (path / n).stat().st_size <= max_size for n, max_size in zip(names, limits)), 'missing or oversized adapter artifacts: ' + key)
    hashes = [sha(path / n) for n in names]
    if key in KNOWN_SHA:
        require(hashes[:6] == list(KNOWN_SHA[key]), 'known adapter bytes changed: ' + key)
    with zipfile.ZipFile(path / anchor_file) as z:
        features = z.getinfo('features.npy')
        require(0 < features.file_size <= 2*2**20 and z.namelist().count('features.npy') == 1, 'unexpected anchor member: ' + key)
    cfg = file_json(path / names[1])
    expected_mapping = {{'qwen3': ('Qwen3Model', 'transformers.models.qwen3.modeling_qwen3'), 'qwen2': ('Qwen2Model', 'transformers.models.qwen2.modeling_qwen2')}}[file_json(Path(base) / 'config.json')['model_type']]
    mapping = cfg.get('auto_mapping') or {{}}
    require(cfg.get('peft_type') == 'LORA' and (not mapping or (mapping.get('base_model_class'), mapping.get('parent_library')) == expected_mapping), 'unexpected adapter type: ' + key)
    require(not (path/'adapter'/'adapter_model.bin').exists(), 'unexpected pickle adapter: ' + key)
    token_cfg = file_json(path/'tokenizer'/'tokenizer_config.json')
    require(not token_cfg.get('auto_map'), 'unexpected custom tokenizer: ' + key)
    bundled_config = path/'base_config'/'config.json'
    bundled = file_json(bundled_config) if bundled_config.is_file() else {{}}
    return path, {{'adapter': ADAPTERS[key], 'base': str(Path(base).relative_to(ROOT)), 'files_sha256': dict(zip(names, hashes)), 'known_s07_bytes_checked': key in KNOWN_SHA, 'adapter_reported_base': cfg.get('base_model_name_or_path'), 'adapter_auto_mapping': mapping, 'bundled_base_config': {{k: bundled.get(k) for k in ('_name_or_path', 'model_type', 'architectures', 'num_hidden_layers', 'hidden_size')}} if bundled else None, 'bundled_base_config_sha256': sha(bundled_config) if bundled else None}}

require(os.getenv('KAGGLE_IS_COMPETITION_RERUN') is None, 'private diagnostic only')
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', NTL_PIPE_WORKERS='1', TORCH_FORCE_WEIGHTS_ONLY_LOAD='1')
import numpy as np
import torch
import transformers
from transformers import AutoConfig, AutoModel, AutoTokenizer
require(torch.cuda.is_available() and torch.cuda.device_count() == 2 and all('T4' in torch.cuda.get_device_name(i) for i in range(2)), 'requires two T4 GPUs')
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
found_train = {{p.resolve() for p in (ROOT/'name-that-llm/train.csv', ROOT/'competitions/name-that-llm/train.csv') if p.is_file()}}
require(len(found_train) == 1, 'competition training mount mismatch')
train = next(iter(found_train))
require(sha(train) == {TRAIN_SHA256!r}, 'competition training fingerprint mismatch')
with train.open(encoding='utf-8', newline='') as source:
    TR = [row['text'] for row in itertools.islice(csv.DictReader(source), 3)]
require(len(TR) == 3 and all(isinstance(x, str) and x for x in TR), 'missing official anchor rows')

pipe = types.ModuleType('_reviewed_v25_pipe')
exec(compile({public_pipe!r}, '_reviewed_v25_pipe', 'exec'), pipe.__dict__)
pipelined, spans_devices = pipe.pipelined, pipe.spans_devices
LORA_MAX_LEN = {pieces['LORA_MAX_LEN']!r}
LORA_RUNS = {pieces['LORA_RUNS']!r}
records = {{}}
exec(compile({public_fn!r}, '_reviewed_v25_anchor_only', 'exec'))
for trunk_spec, pool_layers, checkpoints in LORA_RUNS:
    try:
        score_lora_checkpoints(trunk_spec, pool_layers, checkpoints)
    except Exception as exc:
        for key, *_ in checkpoints:
            records.setdefault(key, {{'passes': False, 'error': f'{{type(exc).__name__}}: {{exc}}'}})
        print('STAGE14C_LORA_TRUNK_ERROR', trunk_spec, str(exc), flush=True)
        gc.collect()
        torch.cuda.empty_cache()
require(set(records) == set(ADAPTERS), 'missing adapter anchors')
passed = all(result['passes'] for result in records.values())
report = {{'status': 'PASS_V25_LORA_ANCHORS' if passed else 'FAIL_V25_LORA_ANCHORS', 'threshold_strictly_below': 0.02,
          'rows': 3, 'official_accuracy_measured': False, 'test_rows_read': False, 'full_ensemble_run': False,
          'torch_version': torch.__version__, 'transformers_version': transformers.__version__, 'anchors': records}}
out = Path('/kaggle/working/stage14c_lora_anchor_gpu.json')
out.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + '\n', encoding='utf-8')
print('STAGE14C_LORA_ANCHOR_COMPLETE', json.dumps(report, sort_keys=True, allow_nan=False), flush=True)
require(passed, 'one or more V25 checkpoint anchor comparisons failed')
'''
    compile(cell, "stage14c_lora_anchor_cell.py", "exec")
    return cell


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.resolve().is_relative_to(Path(__file__).resolve().parent.parent):
        parser.error("write the private generated cell outside the repository")
    args.out.write_bytes(build(args.source_dir).encode("utf-8"))
