"""Private branch-execution helper for a reviewed Name That LLM notebook.

This never imports third-party notebook code. `stage` writes only a temporary
canary with new IDs and no labels. `check` publishes aggregates only. A caller
must separately review the exact notebook, freeze its input versions, and wire
all test readers and outputs to the same private canary directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import tempfile
import zipfile

from stage14_native_input_audit import CLASSES, TRAIN_SHA256, need, stable_hash


N_CANARY = 72
ID_PREFIX = "stage14c-canary-"
MAX_LOG_BYTES = 32 * 1024 * 1024
MAX_NATIVE_BYTES = 8 * 1024 * 1024

# SHA-256 of each rendered V25 code cell, numbered among code cells only.
V25_CODE_SHA256 = tuple("""
67fc7aed2c6170ba2129e268caf4ef7f42dac9e9055360c0ab60d9ff4103efea
e25a3ae1b64bb7f3020c1f88b89bb2ad39a68118685152410a22968c373af530
76518140dbda44efdc41043bdcd3ed85ec2bf40bbea0bb8d9b38e54284c836e4
9d6cd28f87f60c086fd32bd78628a11ba746e4ea4486fd47f89cce6de10804a3
b11ad8b9a5f2c9be3584e37c4a2d68fa7c2b71ca328a7ca80cd17d3a155ee9db
a93fac558059f7e7b3d3cfa7144487f0bdc8d11e509524bbbdfe042cc45fd1b2
ce7555ba42a6fa5f47466aab0e596b20413efb7f3898e92e1c6be238c61f3b0b
0976c135dac02fcd9994d72b18eeaab368d7b4386ec1f0ea55cab70dab421162
8abba2d27f9687bb4bebf214f9ef6cd75d0f95243e2e6b035d5a846581c71513
a88cb6d1e1a61c0a6eefc5a7428cdf35236e2face5dd74470852fdbc7f734feb
1ba068037be943f2731972745e2a1327173b7254bc8702240013d7ffbcd44e2b
a559797d817ddba0fcf9e0a7ecdec89fa1302041f0c603da3bfa88dbae197cd4
822de2923f3af71c7723b562541a172f359970659d075ed00a8b8b2783571e3b
b254e7b0398e014a08a04c24b440fd7db1f38afd7d87eaa4f0f99a62836c9cb7
4040feffe9bc48e01f8ce1c833b97941027b8a7aad4a41fe46daac8583988fd6
deb86d8ecd29e730ec0322a2c7b86a2fe3f4fc3bd713999e9f5528a91dbfffbe
3da698827ce95cc1dc37cb000cd543842147576c7e8ea40b5538e7564f62b37a
b3def3130b4e86725a9fc4c45f476173bbbc3d8ca18a5e02e01df00420bb77e7
ccfe93c86561a0267563396ae918729060f618211fa77437012713dd80f9bfae
e1cd485d0a90ccc6e3de833d653fafe0cbad7b9526f18eea2b5ab60d1a61fb49
""".split())

V25_BRANCHES = {
    "required": {
        "full_path": ["train (2699, 4)  test (72, 2)  gpus 2  smoke False", "train 2699 test 72"],
        "main_members": ["members failed: none", "q14v5 done", "q8v5 done", "ll8v5 done",
                         "n14b_clean done", "q25_14 done", "deb done"],
        "qwen_grouping": ["qwen groups"],
        "comp": ["COMP STACK DONE", "comp stack loaded", "+ stylegram", "+ ultrabert"],
        "gsq": ["GSQ 27B scored 72 rows", "wrote blend with GSQ 27B"],
        "native": ["native folds [0, 1, 2, 3, 4] (worker exit 0)", "rows without native 0",
                   "wrote final blend with native q8"],
    },
    "forbidden": {"fallback": ["NATIVE SKIPPED:", "GSQ SKIPPED:", "COMP SKIPPED:", "QWEN GROUPS SKIPPED",
                                "MEMBER FAILED", "MEMBER SKIPPED (time):", "prefit core votes smoke ok",
                                "SKIP lora24_claude:", "SKIP lora24_seed7:", "SKIP lora20_q25_instyle:"]},
}


def normalize_rendered_cell(text):
    return text.replace("\r\n", "\n").rstrip("\n") + "\n"


def replace_exact(text, old, new):
    need(text.count(old) == 1, "canary overlay anchor did not match exactly once")
    return text.replace(old, new, 1)


def overlay_v25(cells):
    """Private canary-only path overlay; reject any other model source version."""
    cells = [normalize_rendered_cell(x) for x in cells]
    need(tuple(hashlib.sha256(x.encode()).hexdigest() for x in cells) == V25_CODE_SHA256,
         "not the reviewed 20-cell Alex V25 source")
    cells[2] = replace_exact(cells[2], "DATA = next(p for p in [",
                             "_ORIGINAL_DATA = next(p for p in [")
    cells[2] = replace_exact(cells[2], "FAST_COMMIT = ",
                             "DATA = str(stage14c_fixture(Path(_ORIGINAL_DATA) / 'train.csv', Path.cwd()))\n"
                             "os.environ['STAGE14C_DATA'] = DATA\nFAST_COMMIT = ")
    cells[16] = replace_exact(cells[16],
        "DATA_DIR = next(p for p in [pathlib.Path('/kaggle/input/competitions/name-that-llm'), pathlib.Path('/kaggle/input/name-that-llm')]\n"
        "                    if (p / 'train.csv').exists())",
        "DATA_DIR = pathlib.Path(os.environ['STAGE14C_DATA'])")
    cells[17] = replace_exact(cells[17], "            print(line, end='')",
                              "            print(line, end='', flush=True)\n"
                              "            with open('stage14c_branch.log', 'a', encoding='utf-8') as _log:\n"
                              "                _log.write(line)")
    cells[17] = replace_exact(cells[17], "    COMP_SEC = time.time() - t0",
                             "    if code != 0:\n        raise RuntimeError('canary complement subprocess failed')\n"
                             "    COMP_SEC = time.time() - t0")
    cells[18] = replace_exact(cells[18], "'/kaggle/working/submission.parquet'", "'canary_predictions.parquet'")
    cells[18] = replace_exact(cells[18], "'--comp_path', ",
                              "'--native_work', 'stage14c_native', '--gsq_work', 'stage14c_gsq',\n"
                              "       '--group_work', 'stage14c_groups', '--comp_path', ")
    cells[18] = replace_exact(cells[18],
        "    print('infer exit code', run(cmd, env=dict(os.environ, NTL_PIPE_WORKERS='1')))",
        "    _infer_status = run(cmd, env=dict(os.environ, NTL_PIPE_WORKERS='1'))\n"
        "    if _infer_status != 0:\n        raise RuntimeError('canary primary inference subprocess failed')\n"
        "    print('infer exit code', _infer_status)")
    for i in (2, 16, 18, 19):
        # The notebook CWD is newly created below /kaggle/temp. Each replaced
        # absolute write path in these cells becomes a scratch-relative path.
        cells[i] = cells[i].replace("/kaggle/working/", "")
    return cells


def fix_v25_qwen_checkpoint_selection(comp_source, config_sha256):
    """Pin only COMP's generic Qwen3 4096 selector to the causal base."""
    need(re.fullmatch(r"[0-9a-f]{64}", config_sha256) is not None, "invalid causal checkpoint pin")
    anchor = "def find_checkpoint(model_type, hidden=None, dir_name=None):\n"
    pin = f'''    if dir_name is None and (model_type, hidden) == ('qwen3', 4096):
        import hashlib
        home = pathlib.Path('/kaggle/input/models/qwen-lm/qwen-3/transformers/8b-base/1')
        cfg = home / 'config.json'
        if hashlib.sha256(cfg.read_bytes()).hexdigest() != {config_sha256!r}:
            raise RuntimeError('Pinned causal Qwen3 checkpoint mismatch')
        print('STAGE14C_COMP_CAUSAL_CHECKPOINT', home, flush=True)
        return str(home)
'''
    return replace_exact(comp_source, anchor, anchor + pin)


V25_CANARY_BOOTSTRAP = """from pathlib import Path
import os, tempfile
from stage14_fullpath_guard import stage as stage14c_fixture
assert os.getenv('KAGGLE_IS_COMPETITION_RERUN') is None, 'private canary cannot enter competition rerun'
assert not any(k in os.environ for k in ('NTL_SMOKE', 'NTL_FAKE_TEST', 'NTL_POOL', 'NTL_TINY'))
assert not Path('/kaggle/working/submission.parquet').exists(), 'unexpected submission artifact'
assert not any(os.path.lexists(p) for p in ('/tmp/llama-3.1', '/kaggle/temp/ntl666-prefit',
                                            '/kaggle/temp/ntl666-prefit.zip')), 'fresh Kaggle runtime required'
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
Path('/kaggle/temp').mkdir(exist_ok=True)
os.chdir(tempfile.mkdtemp(prefix='stage14c_alex_canary_', dir='/kaggle/temp'))
"""

V25_CANARY_FINISH = """import json
from pathlib import Path
import pandas as pd
from stage14_fullpath_guard import report_v25
assert not Path('/kaggle/working/submission.parquet').exists(), 'unexpected submission artifact'
print('STAGE14C_V25_FULL_PATH', json.dumps(report_v25(Path(DATA), Path.cwd(), pd.read_parquet), sort_keys=True))
"""


def sha256_file(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def stable_key(scope, value):
    return hashlib.sha256(f"stage14c:{scope}:{value}".encode("utf-8")).hexdigest()


def fixture_rows(train_csv: Path):
    """Choose four full prompts across byte-length quartiles; do not use labels."""
    need(sha256_file(train_csv) == TRAIN_SHA256, "canary requires exact official train CSV")
    prompts, seen = {}, set()
    with train_csv.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        need(reader.fieldnames is not None and set(reader.fieldnames) == {"id", "prompt_id", "text", "model"},
             "official train schema differs")
        for row in reader:
            key, prompt, passage = row["id"], row["prompt_id"], row["text"]
            need(bool(key and prompt and passage) and key not in seen, "invalid official train row")
            seen.add(key)
            prompts.setdefault(prompt, []).append((key, passage))
    need(len(seen) == 2699 and len(prompts) == 150, "official train geometry differs")
    complete = [(prompt, values) for prompt, values in prompts.items() if len(values) == 18]
    need(len(complete) == 149, "expected 149 complete official prompt groups")
    complete.sort(key=lambda group: (sum(len(v.encode("utf-8")) for _, v in group[1]),
                                     stable_key("prompt", group[0])))
    picks = [complete[(2 * q + 1) * len(complete) // 8] for q in range(3)] + [complete[-1]]
    shuffled = sorted((row for _, group in picks for row in group), key=lambda x: stable_key("row", x[0]))
    result = [(f"{ID_PREFIX}{i:03d}", passage) for i, (_, passage) in enumerate(shuffled)]
    need(len(result) == N_CANARY, "wrong canary count")
    return result


def stage(train_csv: Path, directory: Path):
    """Make a fresh data directory below an existing private scratch directory."""
    need(directory.is_dir() and not directory.is_symlink(), "private scratch parent must already exist")
    rows = fixture_rows(train_csv)
    target = Path(tempfile.mkdtemp(prefix="stage14c_data_", dir=directory))
    (target / "train.csv").symlink_to(train_csv.resolve())
    with (target / "test.csv").open("x", encoding="utf-8", newline="") as dest:
        writer = csv.writer(dest, lineterminator="\n")
        writer.writerow(("id", "text"))
        writer.writerows(rows)
    return target


def canary_ids(query: Path):
    with query.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        need(reader.fieldnames == ["id", "text"], "only id and text allowed in canary query")
        ids = []
        for row in reader:
            need(bool(row["text"]) and None not in row, "invalid canary text field")
            ids.append(row["id"])
    need(ids == [f"{ID_PREFIX}{i:03d}" for i in range(N_CANARY)], "wrong canary ID axis")
    return ids


def _ordered_labels(values):
    return isinstance(values, list) and len(values) == len(CLASSES) and all(isinstance(x, str) for x in values) and set(values) == CLASSES


def guard_prediction_frame(frame, ids):
    """Exact ordered readback, which also prevents fallback to the real fixture."""
    need(list(frame.columns) == ["id", "model"] and len(frame) == len(ids), "wrong parquet schema")
    need(frame["id"].tolist() == ids, "wrong parquet ID order or coverage")
    need(frame["model"].notna().all() and frame["model"].isin(CLASSES).all(), "invalid parquet class label")
    return int(frame["model"].nunique())


def guard_complement(path: Path, metadata: Path, ids):
    import numpy as np

    need(path.stat().st_size <= 128 * 1024 and metadata.stat().st_size <= 128 * 1024, "complement artifact size")
    meta = json.loads(metadata.read_bytes())
    need(meta.get("ids") == ids and _ordered_labels(meta.get("labels")), "complement axes differ")
    values = np.load(path, allow_pickle=False)
    need(values.shape == (len(ids), len(CLASSES)) and values.dtype.kind == "f" and np.isfinite(values).all(),
         "invalid complement logits")
    return list(values.shape)


def guard_native(path: Path, ids):
    import numpy as np

    members = {key + ".npy" for key in ("ids", "labels", "finite", "folds", "plain", "centered", "centered_bias")}
    need(path.stat().st_size <= MAX_NATIVE_BYTES, "oversized native cache")
    with zipfile.ZipFile(path) as zf:
        need(len(zf.infolist()) == len(members) and set(zf.namelist()) == members and
             sum(f.file_size for f in zf.infolist()) <= MAX_NATIVE_BYTES, "unexpected native archive contents")
    with np.load(path, allow_pickle=False) as values:
        need(values["ids"].tolist() == ids and _ordered_labels(values["labels"].tolist()), "native axes differ")
        need(values["folds"].dtype.kind in "iu" and values["folds"].tolist() == list(range(5)), "missing native fold")
        need(values["finite"].dtype.kind == "b" and values["finite"].shape == (len(ids),) and values["finite"].all(),
             "native invalid-row mask")
        for key, shape in (("plain", (5, len(ids), 18)), ("centered", (5, len(ids), 18)), ("centered_bias", (5, 18))):
            part = values[key]
            need(part.shape == shape and part.dtype.kind == "f" and np.isfinite(part).all(), "invalid native array: " + key)
    return 5


def checked_branches(log: Path, required, forbidden):
    """Require reviewed, positive evidence; log successful exit is insufficient."""
    need(0 < log.stat().st_size <= MAX_LOG_BYTES, "unexpected log size")
    need(isinstance(required, dict) and bool(required) and isinstance(forbidden, dict) and bool(forbidden),
         "a positive and failure branch contract is required")
    with log.open(encoding="utf-8", errors="replace") as source:
        data = source.read(MAX_LOG_BYTES + 1)
    keys = {}
    for kind, rules in (("positive", required), ("failure", forbidden)):
        for name, markers in rules.items():
            need(re.fullmatch(r"[a-z][a-z0-9_]{0,60}", name) is not None and isinstance(markers, list)
                 and markers and all(isinstance(x, str) and 3 <= len(x) <= 400 for x in markers),
                 "invalid branch marker contract")
            hits = [m in data for m in markers]
            keys[name] = all(hits) if kind == "positive" else not any(hits)
            need(keys[name], f"branch evidence {kind} check failed: {name}")
    return keys


def audit_dependencies(root: Path, locked):
    """Resolve every required path exactly once; verify sizes and optional digests.

    This is a byte-level mount check. Dataset revision selection in Kaggle's
    editor is a separate requirement. Use independently known digests where
    available; size-only checks do not establish byte identity.
    """
    need(root.is_dir() and isinstance(locked, dict) and bool(locked), "missing dependency lock")
    result = {}
    for name, spec in locked.items():
        need(re.fullmatch(r"[a-z][a-z0-9_]{0,60}", name) is not None and isinstance(spec, dict) and
             {"glob", "bytes"} <= set(spec) <= {"glob", "bytes", "sha256"}, "invalid dependency contract")
        glob, size = spec["glob"], spec["bytes"]
        need(isinstance(glob, str) and 0 < len(glob) < 400 and "\\" not in glob and
             not PurePosixPath(glob).is_absolute() and ".." not in PurePosixPath(glob).parts and
             type(size) is int and size > 0, "invalid dependency pattern or size")
        matches = [p for p in root.glob(glob) if p.is_file()]
        need(len(matches) == 1, f"missing or ambiguous dependency: {name}; matches={len(matches)}")
        path = matches[0]
        need(path.stat().st_size == size, f"dependency size mismatch: {name}")
        expected = spec.get("sha256")
        if expected is not None:
            need(isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected) is not None,
                 "invalid dependency digest")
            need(sha256_file(path) == expected, f"dependency digest mismatch: {name}")
        result[name] = {"bytes": size, "sha256_verified": expected is not None}
    return {"status": "DEPENDENCY_CONTRACT_PASS", "locked_files": len(result), "checks": result,
            "contract_sha256": stable_hash(locked), "kaggle_attachment_versions_verified": False}


def report(query, parquet, comp, comp_meta, native, log, branch, read_parquet):
    ids = canary_ids(query)
    need(isinstance(branch, dict) and set(branch) == {"required", "forbidden"}, "bad branch contract")
    need(0 < parquet.stat().st_size <= 128 * 1024, "oversized canary parquet")
    branch_ok = checked_branches(log, branch["required"], branch["forbidden"])
    model_count = guard_prediction_frame(read_parquet(parquet), ids)
    comp_shape = guard_complement(comp, comp_meta, ids)
    native_count = guard_native(native, ids)
    return {"status": "PRIVATE_FULL_PATH_CANARY_PASS", "rows": len(ids), "distinct_predicted_labels": model_count,
            "complement_shape": comp_shape, "native_folds": native_count, "branch_checks": branch_ok,
            "fixture_sha256": sha256_file(query), "branch_contract_sha256": stable_hash(branch),
            "accuracy_measured": False, "hidden_runtime_measured": False, "raw_rows_exported": False}


def report_v25(data, run_dir, read_parquet, *, causal_checkpoint=False):
    branch = V25_BRANCHES
    if causal_checkpoint:
        branch = {
            'required': {**branch['required'], 'causal_and_lora': [
                'STAGE14C_COMP_CAUSAL_CHECKPOINT /kaggle/input/models/qwen-lm/qwen-3/transformers/8b-base/1',
                *[f'+ {name:20s} weight' for name in (
                    'lora24_public', 'lora24_seed7', 'lora24_claude', 'lora20_q25_instyle')],
            ]},
            'forbidden': {**branch['forbidden'], 'fallback': [*branch['forbidden']['fallback'], 'SKIP lora24_public:']},
        }
    return report(data / "test.csv", run_dir / "canary_predictions.parquet", run_dir / "comp_logit.npy",
                  run_dir / "comp_meta.json", run_dir / "stage14c_native/native.npz", run_dir / "stage14c_branch.log",
                  branch, read_parquet)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    stage_parser = commands.add_parser("stage")
    stage_parser.add_argument("--train", required=True, type=Path)
    stage_parser.add_argument("--scratch", required=True, type=Path)
    mount = commands.add_parser("inputs")
    mount.add_argument("--root", required=True, type=Path)
    mount.add_argument("--lock", required=True, type=Path)
    check = commands.add_parser("check")
    for flag in ("query", "parquet", "comp", "comp-meta", "native", "log", "branches"):
        check.add_argument("--" + flag, required=True, type=Path)
    args = parser.parse_args()
    if args.mode == "stage":
        print(stage(args.train, args.scratch))
        return
    if args.mode == "inputs":
        result = audit_dependencies(args.root, json.loads(args.lock.read_bytes()))
    else:
        import pandas as pd

        branch = json.loads(args.branches.read_bytes())
        result = report(args.query, args.parquet, args.comp, args.comp_meta, args.native, args.log, branch, pd.read_parquet)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
