"""Read-only, CPU input inventory for public Name That LLM V20 (360088024).

Prints aggregate JSON to stdout; does not fit, infer, execute public source,
import Torch/custom model code, access a network, or write real input files.
Use --self-test for bounded synthetic failure-gate checks in a temporary dir.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path
import struct
import unicodedata

import numpy as np
import pandas as pd


VERSION = 360088024
TRAIN_SHA = "40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707"
POOL_ROWS = 34940
LABELS = ["gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
          "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
          "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5", "claude-fable-5",
          "qwen3.8-max", "glm-5.3", "glm-5.3-flash", "kimi-k3"]
OLD_LABELS = sorted(LABELS)
NEW = {
    "lora8b_v1": {"model_type": "qwen3", "hidden_size": 4096, "num_hidden_layers": 36, "keep": 24},
    "lora14b_v1": {"model_type": "qwen3", "hidden_size": 5120, "num_hidden_layers": 40, "keep": 27},
}
OLD_LORAS = {
    "lora24_public": ("**/lora24/final/COMPLETE.json", "heads.safetensors", [2, 4, 6, 8, 12, 16, 20, 24]),
    "lora24_seed7": ("**/name-that-llm-train-lora24-s7/lora_out/job_final/COMPLETE.json", "norms.safetensors", [2, 4, 6, 8, 12, 16, 20, 24]),
    "lora24_claude": ("**/ntl-qwen3-8b-lora-claude-head/**/final/COMPLETE.json", "heads.safetensors", [2, 4, 6, 8, 12, 16, 20, 24]),
    "lora20_q25_instyle": ("**/ntl-train-lora-qwen25-instyle/lora_out/job_final/COMPLETE.json", "norms.safetensors", [2, 4, 6, 8, 10, 12, 16, 20]),
}
EXTRA_PATTERNS = ["**/ntl_extra_passages_claude.csv", "**/gen_passages.csv", "**/ntl_generations.csv",
                  "**/ntl_proxy_generations.csv", "**/ntl_generated*.csv", "**/passages_clean.csv"]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


class Audit:
    def __init__(self, root: Path):
        self.root = root
        self.checks = []
        self.inventory = {}

    def check(self, name, passed, **counts):
        self.checks.append({"name": name, "passed": bool(passed), **counts})
        return bool(passed)

    def relative(self, path):
        try:
            return str(path.relative_to(self.root))
        except ValueError:
            return str(path)

    def receipt(self, path, digest=False):
        out = {"path": self.relative(path), "bytes": path.stat().st_size}
        if digest:
            out["sha256"] = sha(path)
        return out

    def unique(self, name, pattern):
        matches = sorted({Path(p) for p in glob.glob(str(self.root / pattern), recursive=True) if Path(p).is_file()})
        self.inventory[name] = {"pattern": pattern, "matches": [self.receipt(p) for p in matches]}
        return matches[0] if self.check(name + ":one_file", len(matches) == 1, count=len(matches)) else None

    def file(self, name, path, digest=False):
        if not self.check(name + ":exists_nonempty", path.is_file() and path.stat().st_size > 0):
            return False
        self.inventory[name] = self.receipt(path, digest)
        return True

    def json_file(self, name, path, digest=True):
        if not self.file(name, path, digest):
            return None
        try:
            value = json.loads(path.read_text())
            if not isinstance(value, dict):
                raise TypeError()
            return value
        except Exception as e:
            self.check(name + ":json_object", False, exception_type=type(e).__name__)
            return None


def tensor_header(path):
    """Read safetensors shape/header metadata only, never tensor weights."""
    size = path.stat().st_size
    with path.open("rb") as f:
        prefix = f.read(8)
        if len(prefix) != 8:
            raise ValueError("short_header")
        length = struct.unpack("<Q", prefix)[0]
        if length > 32 * 1024 * 1024 or length > size - 8:
            raise ValueError("invalid_header_length")
        header = json.loads(f.read(length))
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    if not tensors:
        raise ValueError("empty_tensors")
    dtype_bytes = {"BOOL": 1, "U8": 1, "I8": 1, "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
                   "I32": 4, "U32": 4, "F32": 4, "I64": 8, "U64": 8, "F64": 8}
    for spec in tensors.values():
        shape, offsets = spec["shape"], spec["data_offsets"]
        if not isinstance(shape, list) or not all(type(x) is int and x >= 0 for x in shape):
            raise ValueError("invalid_shape")
        if len(offsets) != 2 or not all(type(x) is int for x in offsets):
            raise ValueError("invalid_offsets")
        lo, hi = offsets
        if lo < 0 or hi < lo or hi > size - 8 - length:
            raise ValueError("out_of_bounds_tensor")
        if spec["dtype"] not in dtype_bytes:
            raise ValueError("unsupported_dtype")
        if hi - lo != int(np.prod(shape, dtype=object)) * dtype_bytes[spec["dtype"]]:
            raise ValueError("tensor_size_mismatch")
    return tensors


def read_tensor_header(a, name, path, digest=False):
    if not a.file(name, path, digest):
        return None
    try:
        value = tensor_header(path)
        a.check(name + ":header_bounds", True, tensor_count=len(value))
        return value
    except Exception as e:
        a.check(name + ":header_bounds", False, exception_type=type(e).__name__)
        return None


def validate_meta(a, name, meta, spec):
    if meta is None:
        return None
    a.check(name + ":label_order", meta.get("labels") == LABELS)
    cfg = meta.get("cfg")
    if not a.check(name + ":cfg_object", isinstance(cfg, dict)):
        return None
    layers = cfg.get("layers")
    good_layers = (isinstance(layers, list) and len(layers) == 8
                   and all(type(x) is int and 0 < x <= spec["keep"] for x in layers)
                   and layers == sorted(set(layers)))
    a.check(name + ":kept_depth", type(cfg.get("keep")) is int and cfg["keep"] == spec["keep"])
    a.check(name + ":pooled_layers_schema", good_layers)
    a.check(name + ":token_limit_schema", type(cfg.get("max_len")) is int and cfg["max_len"] > 0)
    a.inventory[name + ":cfg_contract"] = {"keep": cfg.get("keep"), "layers": layers if good_layers else None,
                                           "max_len": cfg.get("max_len") if type(cfg.get("max_len")) is int else None}
    return cfg if good_layers else None


def validate_pool(a, pool, train, expected_rows=POOL_ROWS):
    """Aggregate-only data gates; parameterized row count is used by synthetic tests."""
    a.check("pool:row_count", len(pool) == expected_rows, rows=len(pool), expected_rows=expected_rows)
    if not a.check("pool:required_columns", {"text", "model"} <= set(pool.columns)):
        return
    strings = all(isinstance(x, str) and bool(x.strip()) for x in pool["text"]) and all(isinstance(x, str) for x in pool["model"])
    a.check("pool:nonempty_string_text_and_string_labels", strings)
    nulls = int(pool[["text", "model"]].isna().sum().sum())
    a.check("pool:required_values_nonnull", nulls == 0, null_values=nulls)
    known = set(pool["model"].dropna().astype(str))
    a.check("pool:exact_18_class_set", known == set(LABELS), classes=len(known), unknown_classes=len(known - set(LABELS)))
    numeric = pool.select_dtypes(include="number")
    try:
        finite = bool(np.isfinite(numeric.to_numpy(dtype=float, na_value=np.nan)).all())
    except Exception:
        finite = False
    a.check("pool:numeric_metadata_finite", finite, numeric_columns=len(numeric.columns))
    a.inventory["pool_aggregates"] = {
        "rows": len(pool), "columns": [str(c) for c in pool.columns],
        "class_counts": {label: int(pool["model"].eq(label).sum()) for label in LABELS},
    }
    if not strings:
        return
    exact = pool["text"].tolist()
    norm = [normalized(x) for x in exact]
    original = set(train["text"].astype(str))
    original_norm = {normalized(x) for x in original}
    hits = sum(x in original for x in exact)
    norm_hits = sum(x in original_norm for x in norm)
    a.check("pool:no_exact_official_fullresponse_overlap", hits == 0, overlapping_rows=hits)
    a.check("pool:no_normalized_official_fullresponse_overlap", norm_hits == 0, overlapping_rows=norm_hits)
    a.inventory["pool_aggregates"].update({
        "duplicate_exact_text_rows_after_first": len(exact) - len(set(exact)),
        "duplicate_normalized_text_rows_after_first": len(norm) - len(set(norm)),
        "duplicate_model_exact_text_rows_after_first": int(pool.duplicated(["model", "text"]).sum()),
        "duplicate_model_normalized_text_rows_after_first": len(norm) - len(set(zip(pool["model"], norm))),
        "exact_text_values_shared_by_multiple_labels": sum(len(v) > 1 for v in _labels_by_text(exact, pool["model"]).values()),
        "normalized_text_values_shared_by_multiple_labels": sum(len(v) > 1 for v in _labels_by_text(norm, pool["model"]).values()),
        "duplicate_counts_are_report_only": True,
    })


def _labels_by_text(texts, labels):
    out = {}
    for text, label in zip(texts, labels):
        out.setdefault(text, set()).add(label)
    return out


def model_files(a, name, directory, require_tokenizer=True):
    """Inventory base shards; only small configuration/index files are hashed."""
    small = []
    for filename in ["config.json", "tokenizer_config.json", "special_tokens_map.json", "generation_config.json"]:
        p = directory / filename
        if p.is_file():
            small.append(a.receipt(p, digest=p.stat().st_size <= 16 * 1024 * 1024))
    weights = sorted(set(directory.glob("*.safetensors")) | set(directory.glob("pytorch_model*.bin")))
    index_files = sorted(set(directory.glob("*.safetensors.index.json")) | set(directory.glob("pytorch_model*.bin.index.json")))
    for index in index_files:
        data = a.json_file(name + ":" + index.name, index)
        if data is None:
            continue
        wm = data.get("weight_map")
        if a.check(name + ":index_weight_map", isinstance(wm, dict) and bool(wm)):
            values = list(wm.values())
            safe = all(isinstance(v, str) and Path(v).name == v for v in values)
            a.check(name + ":index_safe_filenames", safe)
            if safe:
                required = {directory / v for v in values}
                a.check(name + ":all_index_shards_present", all(p.is_file() and p.stat().st_size > 0 for p in required), required_shards=len(required))
                weights = sorted(set(weights) | {p for p in required if p.is_file()})
    a.check(name + ":weight_files_present", bool(weights) and all(p.stat().st_size > 0 for p in weights), shard_count=len(weights))
    if require_tokenizer:
        a.check(name + ":tokenizer_files_present", (directory / "tokenizer.json").is_file() or (directory / "vocab.json").is_file()
                or (directory / "vocab.txt").is_file() or (directory / "spm.model").is_file() or (directory / "sentencepiece.bpe.model").is_file())
    a.inventory[name + ":model_files"] = {"small_configs": small, "weight_shards": [a.receipt(p) for p in weights], "weight_shards_hashed": False}


def inspect_new(a, name, spec):
    marker = a.unique(name, "**/" + name + "/meta.json")
    if marker is None:
        return
    meta = a.json_file(name + ":meta", marker)
    cfg = validate_meta(a, name, meta, spec)
    adapter = marker.parent / "adapter"
    acfg = a.json_file(name + ":adapter_config", adapter / "adapter_config.json")
    ah = read_tensor_header(a, name + ":adapter_weights", adapter / "adapter_model.safetensors", digest=True)
    hh = read_tensor_header(a, name + ":heads", marker.parent / "heads.safetensors", digest=True)
    if acfg is not None:
        rank = acfg.get("r")
        a.check(name + ":adapter_rank_and_scale", type(rank) is int and rank > 0 and isinstance(acfg.get("lora_alpha"), (int, float)) and np.isfinite(acfg["lora_alpha"]))
        if ah is not None and type(rank) is int:
            pairs = [(key, key.replace(".lora_A.", ".lora_B.")) for key in ah if ".lora_A." in key]
            good = bool(pairs) and all(b in ah and len(ah[x]["shape"]) == 2 and len(ah[b]["shape"]) == 2
                                       and ah[x]["shape"][0] == rank and ah[b]["shape"][1] == rank for x, b in pairs)
            a.check(name + ":adapter_pair_shapes", good, pairs=len(pairs))
    if cfg is not None and hh is not None:
        width = len(cfg["layers"]) * spec["hidden_size"]
        shapes = {"plain.weight": [18, width], "plain.bias": [18], "centered.weight": [18, width], "centered.bias": [18]}
        for i in range(len(cfg["layers"])):
            shapes.update({f"norms.{i}.weight": [spec["hidden_size"]], f"norms.{i}.bias": [spec["hidden_size"]]})
        a.check(name + ":head_shapes", all(k in hh and hh[k]["shape"] == shape for k, shape in shapes.items()))


def inspect_old_lora(a, name, contract):
    marker, norms, layers = contract
    p = a.unique(name, marker)
    if p is None:
        return
    d = p.parent
    a.json_file(name + ":completion", p)
    a.json_file(name + ":adapter_config", d / "adapter/adapter_config.json")
    a.file(name + ":adapter_weights", d / "adapter/adapter_model.safetensors")
    read_tensor_header(a, name + ":norms", d / norms)
    a.check(name + ":tokenizer_directory", (d / "tokenizer").is_dir())
    tok = d / "tokenizer"
    tok_assets = [tok / f for f in ["tokenizer.json", "vocab.json", "vocab.txt", "spm.model", "sentencepiece.bpe.model"]]
    a.check(name + ":tokenizer_assets", any(p.is_file() and p.stat().st_size > 0 for p in tok_assets))
    a.inventory[name + ":tokenizer_configs"] = [a.receipt(tok / f, True) for f in ["tokenizer_config.json", "special_tokens_map.json"] if (tok / f).is_file()]
    rp = d / "readouts.npz"
    if a.file(name + ":readouts", rp):
        try:
            with np.load(rp, allow_pickle=False) as z:
                a.check(name + ":readout_columns_and_layers", list(z["classes"]) == OLD_LABELS and list(z["pool_layers"]) == layers)
                names = ["plain_coef", "plain_intercept"] + (["center_coef", "center_intercept"] if name in {"lora24_public", "lora24_claude"} else [])
                a.check(name + ":readout_numeric_finite", all(k in z.files and np.issubdtype(z[k].dtype, np.number) and np.isfinite(z[k]).all() for k in names))
        except Exception as e:
            a.check(name + ":readout_safe_load", False, exception_type=type(e).__name__)


def run(root):
    a = Audit(root)
    configs = []
    for p in sorted(root.rglob("config.json")):
        try:
            c = json.loads(p.read_text())
            if isinstance(c, dict):
                configs.append((p, c))
        except Exception:
            pass
    # The two source competition locations are authoritative; no broad train.csv guessing.
    train_paths = [root / "competitions/name-that-llm/train.csv", root / "name-that-llm/train.csv"]
    found = [p for p in train_paths if p.is_file()]
    train = None
    if a.check("official:one_competition_train", len(found) == 1, count=len(found)):
        p = found[0]
        a.file("official_train", p, digest=True)
        a.check("official:train_hash", sha(p) == TRAIN_SHA)
        try:
            train = pd.read_csv(p, keep_default_na=False)
            needed = {"id", "text", "model", "prompt_id"}
            schema = needed <= set(train.columns)
            a.check("official:schema", schema)
            if schema:
                a.check("official:rows_classes_prompts", len(train) == 2699 and set(train.model) == set(LABELS) and train.prompt_id.nunique() == 150)
                a.check("official:unique_ids", train.id.is_unique)
            else:
                train = None
        except Exception as e:
            a.check("official:readable", False, exception_type=type(e).__name__)
        a.file("official_test", p.parent / "test.csv", digest=True)
    pool_path = a.unique("pool", "**/pool_v1.parquet")
    if pool_path is not None:
        a.file("pool_file", pool_path, digest=True)
        try:
            pool = pd.read_parquet(pool_path)
            if train is not None:
                validate_pool(a, pool, train)
            else:
                a.check("pool:official_overlap_check_available", False)
        except Exception as e:
            a.check("pool:readable", False, exception_type=type(e).__name__)
    for name, spec in NEW.items():
        inspect_new(a, name, spec)
    specs = {"base_new8b": {k: v for k, v in NEW["lora8b_v1"].items() if k != "keep"},
             "base_new14b": {k: v for k, v in NEW["lora14b_v1"].items() if k != "keep"},
             "base_old_qwen8b": {"model_type": "qwen3", "hidden_size": 4096},
             "base_old_qwen25_7b": {"model_type": "qwen2", "hidden_size": 3584},
             "base_old_qwen4b": {"model_type": "qwen3", "hidden_size": 2560},
             "base_glm": {"model_type": "glm4", "hidden_size": 4096},
             "base_qwen35": {"model_type": "qwen3_5"}}
    for name, spec in specs.items():
        matches = [p for p, c in configs if all(c.get(k) == v for k, v in spec.items())]
        a.inventory[name + ":config_matches"] = [a.receipt(p, True) for p in matches]
        if a.check(name + ":one_matching_config", len(matches) == 1, count=len(matches)):
            model_files(a, name, matches[0].parent)
    for name in ["deberta-v3-large", "electra-large-discriminator", "desklib-detector"]:
        p = a.unique(name, "**/" + name + "/config.json")
        if p is not None:
            a.json_file(name + ":config", p)
            model_files(a, name, p.parent)
    for name, contract in OLD_LORAS.items():
        inspect_old_lora(a, name, contract)
    for filename in ["tokattn_final.pt", "tokattn_debl23_final.pt"]:
        p = a.unique(filename, "**/" + filename)
        if p is not None:
            a.file(filename + ":checkpoint", p)
    ultra = [(p, c) for p, c in configs if p.parent.name.startswith("fold") and p.parent.parent.name == "ultrabert-380m-ntl" and c.get("model_type") == "ultrabert"]
    roots = {p.parent.parent for p, _ in ultra}
    if a.check("ultrabert:one_root_exactly_five_folds", len(roots) == 1 and len(ultra) == 5, roots=len(roots), folds=len(ultra)):
        for i, (p, _) in enumerate(ultra):
            model_files(a, "ultrabert:" + p.parent.name, p.parent, require_tokenizer=i == 0)
        custom = sorted({p for parent in [next(iter(roots))] + [p.parent for p, _ in ultra] for p in parent.glob("*.py")})
        a.inventory["ultrabert_custom_code"] = [a.receipt(p, True) for p in custom]
        a.check("ultrabert:custom_code_present", bool(custom))
    extras = sorted({Path(p) for pattern in EXTRA_PATTERNS for p in glob.glob(str(root / pattern), recursive=True) if Path(p).is_file()})
    a.inventory["old_extra_csvs"] = [a.receipt(p, True) for p in extras]
    a.check("old_extra_csvs:present", bool(extras), files=len(extras))
    return {"stage": "08_input_preflight", "public_source_version": VERSION,
            "status": "PASS_INPUT_CONTRACT" if all(c["passed"] for c in a.checks) else "FAIL_INPUT_CONTRACT",
            "read_only": True, "new_fitting": False, "new_inference": False, "network_used": False,
            "public_source_executed": False, "checks": a.checks, "inventory": a.inventory,
            "limits": ["A pass inventories present bytes; it does not prove that these are the author's exact scored input versions until receipts are compared with a source-version manifest.",
                       "No adapter training-data provenance, provider identity, usage-terms admission, semantic prompt independence or claimed CV result is independently established here.",
                       "Base/encoder weight shards are listed by size/path, not read or hashed; Torch checkpoint contents and model/custom Python code are not executed.",
                       "The known pool count is author-described, not an independent provenance guarantee. Duplicate counts are descriptive; any official full-response overlap fails.",
                       "Extra CSV membership affects the old style branch; presence alone does not establish the author's exact 7644-row merge, prompt-bank compatibility or versions.",
                       "This does not establish GPU compatibility, prediction parity, a completed preview/hidden run, or an official score."]}


def self_test():
    import tempfile
    checks = {}
    with tempfile.TemporaryDirectory(prefix="stage08_preflight_test_") as d:
        root = Path(d)
        a = Audit(root)
        checks["missing_critical_file_rejected"] = a.unique("x", "**/x.json") is None and not a.checks[-1]["passed"]
        for folder in ["a", "b"]:
            (root / folder).mkdir()
            (root / folder / "x.json").write_text("{}")
        a = Audit(root)
        checks["ambiguous_critical_file_rejected"] = a.unique("x", "**/x.json") is None and not a.checks[-1]["passed"]
        train = pd.DataFrame({"text": ["official unique passage"]})
        pool = pd.DataFrame({"model": LABELS * 2, "text": [f"synthetic passage {i}" for i in range(36)]})
        def passed(frame, expected=36):
            au = Audit(root); validate_pool(au, frame, train, expected); return all(c["passed"] for c in au.checks)
        checks["valid_pool_contract_passes"] = passed(pool)
        checks["pool_row_count_rejected"] = not passed(pool.iloc[:-1])
        checks["pool_schema_rejected"] = not passed(pool.drop(columns="text"))
        bad = pool.copy(); bad.loc[0, "model"] = "unknown"
        checks["unknown_pool_label_rejected"] = not passed(bad)
        bad = pool.copy(); bad.loc[0, "text"] = "  OFFICIAL   unique passage  "
        checks["normalized_fullresponse_overlap_rejected"] = not passed(bad)
        bad = pool.copy(); bad["number"] = np.nan
        checks["nonfinite_numeric_metadata_rejected"] = not passed(bad)
        bad = pool.copy(); bad.loc[0, "text"] = None
        checks["null_text_rejected"] = not passed(bad)
        au = Audit(root)
        validate_meta(au, "new", {"labels": list(reversed(LABELS)), "cfg": {"keep": 24, "layers": [2, 4, 6, 8, 12, 16, 20, 24], "max_len": 447}}, NEW["lora8b_v1"])
        checks["checkpoint_label_order_rejected"] = not all(c["passed"] for c in au.checks)
        p = root / "bad.safetensors"; p.write_bytes(struct.pack("<Q", 99999))
        try:
            tensor_header(p); checks["invalid_tensor_header_rejected"] = False
        except ValueError:
            checks["invalid_tensor_header_rejected"] = True
        checks["empty_input_manifest_fails_closed"] = run(root)["status"] == "FAIL_INPUT_CONTRACT"
    return {"stage": "08_input_preflight_synthetic_tests", "status": "PASS" if all(checks.values()) else "FAIL",
            "tests": checks, "real_inputs_read": False, "model_code_executed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        report = self_test() if args.self_test else run(args.input_root)
    except Exception as e:
        report = {"stage": "08_input_preflight", "status": "FAIL_PREFLIGHT_EXCEPTION", "exception_type": type(e).__name__,
                  "raw_exception_message_omitted": True, "public_source_executed": False}
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["status"].startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
