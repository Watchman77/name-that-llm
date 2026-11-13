"""Read-only structural/numeric audit of the published native Q8 task bundle.

The worker/fusion contract comes from the frozen v43 mirror, cells 51 and 53,
of lbjyang's .684 mirror. No model loading, training, or held-out claim.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

import numpy as np


SOURCE_BLOB = "26fa9032d4bfcfe69a95674f273baaa1eae62e38"
UPSTREAM_VERSION = 365653660
TREE_NAME = "ntl-native-q8-multidepth-fivefold-0679"
TRAIN_SHA256 = "40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707"
CLASSES = frozenset("""claude-fable-5 claude-opus-4-6 claude-opus-4-8 claude-opus-5 claude-sonnet-5
gemini-3.1-pro-preview gemini-3.6-flash gemini-3.7-flash gemini-3.8-flash glm-5.3 glm-5.3-flash
gpt-5.5 gpt-5.6-luna gpt-5.6-sol gpt-5.6-terra gpt-6-astra kimi-k3 qwen3.8-max""".split())
DEPTHS = (3, 6, 9, 12, 18, 24, 30, 36)
BASE_DIMS = dict(model_type="qwen3", num_hidden_layers=36, hidden_size=4096,
                 intermediate_size=12288, head_dim=128, num_attention_heads=32,
                 num_key_value_heads=8)
CHUNK = 4 * 1024 * 1024


class AuditFailure(ValueError):
    """Fail-closed gate, with a diagnostic safe for the aggregate report."""


def need(condition, gate):
    if not condition:
        raise AuditFailure(gate)


def no_duplicates(items):
    result = {}
    for key, value in items:
        need(key not in result, "duplicate_json_key")
        result[key] = value
    return result


def bad_constant(_):
    raise AuditFailure("invalid_json_constant")


def decode_json(data, gate):
    try:
        value = json.loads(data, object_pairs_hook=no_duplicates, parse_constant=bad_constant)
    except (UnicodeError, ValueError) as exc:
        raise AuditFailure(gate + ": invalid or ambiguous JSON") from exc
    need(isinstance(value, dict), gate + ": expected object")
    return value


def require_file(path, key):
    need(path.is_file() and path.stat().st_size > 0, key + ": missing or empty")


def hash_file(path, key):
    require_file(path, key)
    digest, count = hashlib.sha256(), 0
    with path.open("rb") as handle:
        while block := handle.read(CHUNK):
            digest.update(block)
            count += len(block)
    need(path.stat().st_size == count, key + ": file size changed")
    return {"bytes": count, "sha256": digest.hexdigest()}


def read_json(path, key, files, limit=32 * 1024 * 1024):
    require_file(path, key)
    need(path.stat().st_size <= limit, key + ": JSON too large")
    with path.open("rb") as handle:
        raw = handle.read(limit + 1)
    need(0 < len(raw) <= limit and path.stat().st_size == len(raw), key + ": file size changed")
    data = decode_json(raw, key)
    files[key] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    return data


def expected_tensors(config):
    need(all(type(config.get(k)) is type(v) and config[k] == v for k, v in BASE_DIMS.items()),
         "base/config.json: Qwen3-Embedding-8B geometry mismatch")
    h, mlp = config["hidden_size"], config["intermediate_size"]
    q = config["head_dim"] * config["num_attention_heads"]
    kv = config["head_dim"] * config["num_key_value_heads"]
    specs = {"self_attn.q_proj": (h, q), "self_attn.k_proj": (h, kv),
             "self_attn.v_proj": (h, kv), "self_attn.o_proj": (q, h),
             "mlp.gate_proj": (h, mlp), "mlp.up_proj": (h, mlp), "mlp.down_proj": (mlp, h)}
    expected = {}
    for layer in range(config["num_hidden_layers"]):
        for projection, (input_dim, output_dim) in specs.items():
            prefix = f"encoder.backbone.base_model.model.layers.{layer}.{projection}.lora_"
            expected[prefix + "A.default.weight"] = [16, input_dim]
            expected[prefix + "B.default.weight"] = [output_dim, 16]
    for pos in range(len(DEPTHS)):
        for param in ("weight", "bias"):
            expected[f"norms.{pos}.{param}"] = [h]
    for head in ("plain", "centered"):
        expected[f"{head}.weight"] = [18, h * len(DEPTHS)]
        expected[f"{head}.bias"] = [18]
    need(len(expected) == 524, "internal expected tensor count")
    return expected


def check_checkpoint(path, key, expected):
    """Check the safetensors wire format without instantiating the model."""
    require_file(path, key)
    size, digest = path.stat().st_size, hashlib.sha256()
    with path.open("rb") as handle:
        length_bytes = handle.read(8)
        need(len(length_bytes) == 8, key + ": truncated header length")
        length = struct.unpack("<Q", length_bytes)[0]
        need(2 <= length <= min(2 * 1024 * 1024, size - 8), key + ": invalid header length")
        raw = handle.read(length)
        need(len(raw) == length and raw.startswith(b"{"), key + ": invalid header bytes")
        header = decode_json(raw, key)
        meta = header.pop("__metadata__", {})
        need(isinstance(meta, dict) and all(isinstance(v, str) for v in meta.values()),
             key + ": invalid safetensors metadata")
        need(set(header) == set(expected), key + ": unexpected or missing tensor keys")
        spans = []
        for name, shape in expected.items():
            info = header[name]
            need(isinstance(info, dict) and set(info) == {"dtype", "shape", "data_offsets"},
                 key + ": invalid tensor entry")
            need(info["dtype"] == "F32" and isinstance(info["shape"], list)
                 and all(type(x) is int for x in info["shape"]) and info["shape"] == shape,
                 key + ": tensor dtype or shape mismatch")
            pair = info["data_offsets"]
            need(isinstance(pair, list) and len(pair) == 2 and all(type(x) is int for x in pair),
                 key + ": invalid tensor offsets")
            lo, hi = pair
            need(lo >= 0 and hi - lo == 4 * math.prod(shape), key + ": invalid tensor span")
            spans.append((lo, hi))
        end = 0
        for lo, hi in sorted(spans):
            need(lo == end, key + ": overlapping or uncovered tensor spans")
            end = hi
        need(end == size - 8 - length, key + ": payload length mismatch")
        digest.update(length_bytes)
        digest.update(raw)
        remaining = end
        while remaining:
            chunk_size = min(CHUNK, remaining)
            block = handle.read(chunk_size)
            need(len(block) == chunk_size, key + ": truncated payload")
            need(np.isfinite(np.frombuffer(block, dtype="<f4")).all(), key + ": nonfinite tensor values")
            digest.update(block)
            remaining -= chunk_size
        need(not handle.read(1) and path.stat().st_size == size, key + ": file size changed")
    return {"bytes": size, "sha256": digest.hexdigest(), "f32_tensors": len(expected)}


def validate_calibration(data, key):
    models = data.get("models")
    state = models.get("qwen8") if isinstance(models, dict) else None
    need(isinstance(state, dict) and state.get("outer_labels_used") is False
         and state.get("centered_head_trained") is True, key + ": Q8 calibration flags mismatch")
    for name in ("plain_scale", "centered_scale", "selected_centered_weight"):
        value = state.get(name)
        need(type(value) in (int, float) and math.isfinite(value), key + ": invalid calibration value")
        need(0 <= value <= 0.5 if name == "selected_centered_weight" else value > 0,
             key + ": calibration value out of bounds")


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def check_tokenizer(assets, files):
    key = "base/tokenizer.json"
    tokenizer = read_json(assets / key, key, files)
    model, added = tokenizer.get("model"), tokenizer.get("added_tokens")
    need(isinstance(model, dict) and model.get("type") == "BPE" and isinstance(added, list),
         key + ": expected Qwen BPE tokenizer structure")
    terminal = [item for item in added if isinstance(item, dict) and item.get("id") == 151643]
    need(len(terminal) == 1 and terminal[0].get("content") == "<|endoftext|>"
         and terminal[0].get("special") is True, key + ": terminal token ID mismatch")
    key = "base/tokenizer_config.json"
    tokenizer_cfg = read_json(assets / key, key, files)
    decoder = tokenizer_cfg.get("added_tokens_decoder")
    entry = decoder.get("151643") if isinstance(decoder, dict) else None
    need(isinstance(entry, dict) and entry.get("content") == "<|endoftext|>"
         and entry.get("special") is True, key + ": terminal token ID mismatch")
    key = "base/vocab.json"
    vocab = read_json(assets / key, key, files)
    need(len(vocab) == 151643 and vocab == model.get("vocab")
         and all(type(v) is int for v in vocab.values()) and set(vocab.values()) == set(range(151643)),
         key + ": BPE vocabulary mismatch")
    key = "base/merges.txt"
    files[key] = hash_file(assets / key, key)


def check_base_index(assets, files):
    key = "base/model.safetensors.index.json"
    index = read_json(assets / key, key, files)
    weight_map = index.get("weight_map")
    shard_names = {f"model-{i:05d}-of-00004.safetensors" for i in range(1, 5)}
    need(isinstance(weight_map, dict) and weight_map and all(isinstance(v, str) for v in weight_map.values())
         and set(weight_map.values()) == shard_names, key + ": expected four Q8 base shard references")


def audit(assets, expected_tree_sha256=None, train_csv=None):
    need(assets.is_dir(), "missing Q8 asset directory")
    files = {}
    manifest = read_json(assets / "manifest.json", "manifest.json", files)
    classes = manifest.get("classes")
    need(isinstance(classes, list) and len(classes) == 18 and all(isinstance(c, str) for c in classes)
         and set(classes) == CLASSES, "manifest.json: class axis mismatch")
    config = read_json(assets / "base/config.json", "base/config.json", files)
    expected = expected_tensors(config)
    check_base_index(assets, files)
    check_tokenizer(assets, files)
    for fold in range(5):
        key = f"calibration/fold{fold}.json"
        validate_calibration(read_json(assets / key, key, files), key)
        key = f"fold{fold}/trainables.safetensors"
        files[key] = check_checkpoint(assets / key, key, expected)
    tree_hash = stable_hash(files)
    if expected_tree_sha256 is not None:
        need(tree_hash == expected_tree_sha256, "Q8 tree fingerprint mismatch")
    official_hash_matched = None
    if train_csv is not None:
        official_hash_matched = hash_file(train_csv, "official train.csv")["sha256"] == TRAIN_SHA256
        need(official_hash_matched, "official train.csv: fingerprint mismatch")
    return {"status": "PASS_Q8_TASK_ASSET_INPUT_AUDIT_ONLY", "schema_version": 1,
            "source_notebook_blob_sha1": SOURCE_BLOB, "upstream_script_version_id": UPSTREAM_VERSION,
            "tree_name": TREE_NAME, "task_asset_tree_sha256": tree_hash,
            "external_tree_fingerprint_matched": expected_tree_sha256 is not None,
            "class_order_sha256": stable_hash(classes), "classes": len(classes), "folds": 5,
            "depths": list(DEPTHS), "tensors_per_fold": len(expected), "official_train_hash_matched": official_hash_matched,
            "task_assets": files, "task_base_shard_index_checked": True, "base_model_shards_checked": False,
            "tokenizer_execution_parity_checked": False, "training_prompt_exclusion_verified": False,
            "oof_predictions_admitted": False, "model_or_gpu_execution_performed": False,
            "official_score_measured": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True, help=f"directory containing {TREE_NAME} contents")
    parser.add_argument("--expected-tree-sha256", help="independently pinned exact task asset fingerprint")
    parser.add_argument("--train-csv", type=Path, help="optionally verify the known official train.csv fingerprint")
    args = parser.parse_args(argv)
    try:
        report = audit(args.assets, args.expected_tree_sha256, args.train_csv)
    except AuditFailure as exc:
        print(json.dumps({"status": "BLOCK_Q8_TASK_ASSET_INPUT_AUDIT", "gate": str(exc)}, sort_keys=True))
        return 1
    except (OSError, OverflowError, RecursionError) as exc:
        print(json.dumps({"status": "BLOCK_Q8_TASK_ASSET_INPUT_AUDIT", "gate": type(exc).__name__}, sort_keys=True))
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
