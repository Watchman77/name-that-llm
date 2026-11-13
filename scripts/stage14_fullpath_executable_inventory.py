"""Inspect Alex V25 serialized/runtime artifacts without loading or executing them."""
from __future__ import annotations

import bisect
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import stat
import struct
import zipfile

LIMIT = 4 * 1024 * 1024
LABELS = {
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "glm-5.3", "glm-5.3-flash", "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra",
    "gpt-6-astra", "kimi-k3", "qwen3.8-max",
}
GSQ_REVISION = "b5ce0b76f60020a875dee4f6ec9d934cca4121e4"
Q25_OFFICIAL_CONFIG_SHA = "267ce68584c5f24c3b267d934db2de68dd21d1ca677fb78ed809eb60067f7642"
REVIEWED_ULTRA_PY = {
    "configuration_ultrabert.py": "823d10bda19f5164da4f79f8f9c01e5ebd7a75569abce78c0a26b7674fc78cd9",
    "modeling_ultrabert.py": "bfca1bd08af72cd170e58b5a048c67b52bc9d06a2a42847c7a387751f527f333",
}
GSQ_BASE_BYTES = {"config.json": 5038, "merges.txt": 3353259, "model.safetensors.index.json": 198539,
                  "tokenizer.json": 12809320, "tokenizer_config.json": 17928, "vocab.json": 6722759,
                  "model-00001-of-00003.safetensors": 2272369784,
                  "model-00002-of-00003.safetensors": 4777704120,
                  "model-00003-of-00003.safetensors": 4777704376}


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def json_load(data):
    def pairs(items):
        out = {}
        for key, val in items:
            require(key not in out, "duplicate_json_key")
            out[key] = val
        return out
    return json.loads(data, object_pairs_hook=pairs)


def read_json(path):
    require(path.is_file() and 0 < path.stat().st_size <= LIMIT, "json_missing_or_size:" + path.name)
    value = json_load(path.read_bytes())
    require(isinstance(value, dict), "json_not_object:" + path.name)
    return value


def info(path, root, hash_bytes=True):
    require(path.is_file(), "file_missing:" + path.name)
    o = {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size}
    if hash_bytes:
        with path.open("rb") as f:
            o["sha256"] = hashlib.file_digest(f, "sha256").hexdigest()
    return o


def unique(paths, label):
    found = {}
    for p in paths:
        if p.is_file():
            found.setdefault(p.resolve(), p)
    require(len(found) == 1, label + "_missing_or_ambiguous")
    return next(iter(found.values()))


class Segments(io.RawIOBase):
    """Seekable view of a split ZIP; never writes or extracts it."""
    def __init__(self, paths):
        super().__init__()
        self.parts = paths
        self.ends = [0]
        for p in paths:
            self.ends.append(self.ends[-1] + p.stat().st_size)
        self.pos = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        offset += {0: 0, 1: self.pos, 2: self.ends[-1]}[whence]
        require(offset >= 0, "negative_archive_offset")
        self.pos = offset
        return offset

    def read(self, size=-1):
        remaining = max(0, self.ends[-1] - self.pos)
        size = remaining if size < 0 else min(remaining, size)
        require(size <= 64 * 1024 * 1024, "oversize_segment_read")
        chunks = []
        while size > 0:
            i = bisect.bisect_right(self.ends, self.pos) - 1
            n = min(size, self.ends[i + 1] - self.pos)
            with self.parts[i].open("rb") as f:
                f.seek(self.pos - self.ends[i])
                data = f.read(n)
            require(len(data) == n, "truncated_archive_segment")
            chunks.append(data)
            self.pos += n
            size -= n
        return b"".join(chunks)


def zip_members(archive, cap):
    entries = archive.infolist()
    names = {}
    require(0 < len(entries) <= cap, "zip_member_count")
    for e in entries:
        p = PurePosixPath(e.filename)
        require(not e.is_dir() and len(p.parts) >= 2 and not p.is_absolute()
                and ".." not in p.parts and "\\" not in e.filename and p.as_posix() == e.filename
                and e.filename not in names and stat.S_IFMT(e.external_attr >> 16) in {0, stat.S_IFREG}
                and not e.flag_bits & 1, "zip_unsafe_member")
        names[e.filename] = e
    return names


def zip_bytes(z, names, key, limit=LIMIT):
    require(key in names and names[key].file_size <= limit, "zip_member_missing_or_size:" + key)
    return z.read(names[key])


def zip_sha(z, e):
    with z.open(e, "r") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def prefit(root, expected):
    path = unique(sorted(root.rglob("ntl666-cache-parts.json")), "prefit_manifest")
    m = read_json(path)
    require(m.get("format") == "ntl666-cache-parts-v1" and isinstance(m.get("parts"), list)
            and 0 < len(m["parts"]) <= 100, "prefit_manifest_shape")
    paths, receipts = [], []
    for i, entry in enumerate(m["parts"]):
        name = f"ntl666-prefit.part{i:03d}.bin"
        p = path.parent / name
        require(entry.get("name") == name and p.is_file() and p.stat().st_size == entry.get("bytes"),
                "prefit_part_length_or_name")
        record = info(p, root)
        for wanted in (entry.get("sha256"), expected.get(name)):
            require(wanted is None or record["sha256"] == wanted, "prefit_part_fingerprint:" + name)
        paths.append(p)
        receipts.append(record)
    require(sum(x["bytes"] for x in receipts) == m.get("archive_bytes"), "prefit_total_bytes")
    with Segments(paths) as stream, zipfile.ZipFile(stream) as z:
        names = zip_members(z, 10000)
        require(len(names) == m.get("files") and all(n.startswith("ntl666-prefit/") for n in names),
                "prefit_member_count_or_prefix")
        require(zip_bytes(z, names, "ntl666-prefit/COMPLETE") == b"ntl666-v14-data-numeric-v1\n",
                "prefit_complete")
        p = json_load(zip_bytes(z, names, "ntl666-prefit/provenance.json"))
        require(p.get("status") == "COMPLETE" and p.get("pool_rows") == 190588
                and p.get("dataset_version") == 13, "prefit_provenance")
        targets = {}
        for head in ("svc4", "tfidf", "distort"):
            folder = "ntl666-prefit/core/" + head
            meta = f"{folder}/metadata.json"
            d = json_load(zip_bytes(z, names, meta))
            require(d.get("labels") == sorted(LABELS) and isinstance(d.get("views"), list)
                    and 0 < len(d["views"]) <= 16, "prefit_head_class_or_view_axis:" + head)
            checks = [meta, f"{folder}/model.npz"]
            for view in d["views"]:
                item = view.get("file")
                require(isinstance(item, str) and bool(re.fullmatch(r"[A-Za-z0-9_.-]+[.]joblib", item))
                        and not item.startswith("."), "prefit_vectorizer_target:" + head)
                checks.append(f"{folder}/{item}")
            require(len(set(checks)) == len(checks) and all(k in names for k in checks),
                    "prefit_duplicate_or_missing_target:" + head)
            targets[head] = [{"path": k, "bytes": names[k].file_size,
                              "sha256": zip_sha(z, names[k])} for k in checks]
        inventory = [[n, names[n].file_size, f"{names[n].CRC:08x}"] for n in sorted(names)]
        digest = hashlib.sha256(json.dumps(inventory, separators=(",", ":")).encode()).hexdigest()
    return {"manifest": info(path, root), "parts": receipts, "parts_bound_to_prior_receipt": bool(expected),
            "archive_members": len(inventory), "archive_member_inventory_sha256": digest,
            "active_readout_targets": targets, "pickle_loaded": False, "archive_extracted": False}


def torch_archive(path, root):
    with zipfile.ZipFile(path) as z:
        names = zip_members(z, 20000)
        pickles = [n for n in names if n.endswith("/data.pkl")]
        require(len(pickles) == 1, "torch_head_pickle_inventory:" + path.name)
        prefix = pickles[0][:-len("data.pkl")]
        members = [[n[len(prefix):], e.file_size, f"{e.CRC:08x}"] for n,e in sorted(names.items())
                   if n.startswith(prefix)]
        require(len(members) == len(names) and any(re.fullmatch(r"data/[0-9]+", x[0]) for x in members),
                "torch_head_member_tree:" + path.name)
        payload = zip_bytes(z, names, pickles[0], 16 * 1024 * 1024)
    return {**info(path, root), "members": len(members),
            "tensor_storages": sum(bool(re.fullmatch(r"data/[0-9]+", x[0])) for x in members),
            "pickle_bytes": len(payload), "pickle_sha256": hashlib.sha256(payload).hexdigest(),
            "member_inventory_sha256": hashlib.sha256(json.dumps(members, separators=(",", ":")).encode()).hexdigest(),
            "pickle_loaded": False}


def safetensor_header(path, root):
    with path.open("rb") as f:
        raw = f.read(8)
        require(len(raw) == 8, "safetensors_header_size")
        n, = struct.unpack("<Q", raw)
        require(2 <= n <= LIMIT and n + 8 < path.stat().st_size, "safetensors_header_range")
        raw = f.read(n)
    d = json_load(raw)
    d.pop("__metadata__", None)
    spans = sorted(v["data_offsets"] for v in d.values())
    end = 0
    for pair in spans:
        require(len(pair) == 2 and all(type(x) is int for x in pair) and pair[0] == end and pair[1] >= end,
                "safetensors_payload_gaps")
        end = pair[1]
    require(end + n + 8 == path.stat().st_size, "safetensors_payload_bounds")
    return {**info(path, root, False), "tensors": len(d), "header_sha256": hashlib.sha256(raw).hexdigest()}, set(d)


def base_model(path, root):
    cfg = read_json(path)
    folder = path.parent
    metadata = [info(folder / n, root) for n in ("config.json", "tokenizer.json", "tokenizer_config.json",
                                                "vocab.json", "merges.txt", "model.safetensors.index.json")]
    index = read_json(folder / "model.safetensors.index.json").get("weight_map")
    require(isinstance(index, dict) and index, "model_index_empty")
    shards = []
    for n in sorted(set(index.values())):
        require(bool(re.fullmatch(r"model-[0-9]+-of-[0-9]+[.]safetensors", n)), "model_shard_name")
        item, keys = safetensor_header(folder / n, root)
        require(keys == {k for k,v in index.items() if v == n}, "model_shard_tensor_index")
        shards.append(item)
    return {"metadata": metadata, "config_matches_s07_q25": metadata[0]["sha256"] == Q25_OFFICIAL_CONFIG_SHA,
            "model_type": cfg.get("model_type"), "hidden_size": cfg.get("hidden_size"), "shards": shards,
            "weight_payload_hashed": False}


def qwen25(root):
    hit = []
    for cfg in sorted(root.rglob("config.json")):
        try:
            d = read_json(cfg)
        except (ValueError, OSError, TypeError):
            continue
        if (d.get("model_type"), d.get("hidden_size")) == ("qwen2", 3584):
            hit.append(cfg)
    path = unique(hit, "qwen2_3584_comp_7b")
    return base_model(path, root)


def main_heads(root):
    out = {}
    for group in ("q14v5", "q8v5", "ll8v5"):
        for name in [f"{group}_f{i}" for i in range(5)] + [group]:
            path = unique(sorted(root.glob(f"**/{name}/heads.pt")), name)
            out[name] = torch_archive(path, root)
    attn = unique(sorted(root.rglob("tokattn_debl23_final.pt")), "deberta_attention")
    out["deberta_attention"] = torch_archive(attn, root)
    return out


def ultrabert(root):
    configs = [p for p in sorted(root.glob("**/ultrabert-380m-ntl/fold*/config.json"))
               if read_json(p).get("model_type") == "ultrabert"]
    require(len({p.resolve() for p in configs}) == 5, "ultrabert_five_folds")
    out = []
    for p in configs:
        cfg = read_json(p)
        tokenizer = read_json(p.parent / "tokenizer_config.json")
        files = sorted(p.parent.rglob("*.py"))
        records = [info(f, root) for f in files]
        require({Path(x["path"]).name: x["sha256"] for x in records} == REVIEWED_ULTRA_PY
                and len(records) == len(REVIEWED_ULTRA_PY), "ultrabert_code_tree_drift")
        maps = [x.get("auto_map", {}) for x in (cfg, tokenizer)]
        targets = []
        for mapping in maps:
            require(isinstance(mapping, dict), "ultrabert_auto_map_shape")
            for v in mapping.values():
                for target in v if isinstance(v, list) else [v]:
                    if target is None:
                        continue
                    require(isinstance(target, str) and re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*[.][A-Za-z_][A-Za-z_0-9]*", target)
                            and target.split(".")[0] + ".py" in REVIEWED_ULTRA_PY,
                            "ultrabert_auto_map_unreviewed_target")
                    targets.append(target)
        require(targets and "modeling_ultrabert.py" in {x.split(".")[0] + ".py" for x in targets},
                "ultrabert_auto_map_missing_model")
        out.append({"config": info(p, root), "tokenizer_config": info(p.parent / "tokenizer_config.json", root),
                    "local_auto_map_targets": sorted(set(targets)), "python": records})
    return out


def gsq(root):
    final = unique(sorted(root.glob("**/gsq120k-trained/gsq120k-inference/gsq/final.safetensors")), "gsq_final")
    meta, complete = final.with_name("final.json"), final.with_name("COMPLETE.json")
    m, c = read_json(meta), read_json(complete)
    require(c.get("status") == "TRAINED_FULLTEXT_GSQ3_PUBLIC119899_OFFICIAL2_EPOCHS"
            and c.get("rows_seen") == {"external": 119899, "official": 5398}
            and c.get("contract_hash") == m.get("checkpoint_contract_hash")
            and c.get("final_artifact", {}).get("path") == final.name
            and c.get("final_artifact", {}).get("bytes") == final.stat().st_size
            and m.get("source_revision") == GSQ_REVISION and m.get("rank") == 16 and m.get("alpha") == 32
            and m.get("depths") == [8, 16, 24, 32, 40, 48, 56, 64]
            and m.get("lora_target_layer_indices") == list(range(56, 64))
            and m.get("executed_decoder_layers") == 64 and m.get("text_truncation") is False
            and len(m.get("labels", [])) == 18 and set(m["labels"]) == LABELS, "gsq_final_contract")
    final_head, _ = safetensor_header(final, root)
    final_head.update(info(final, root))
    declared = c["final_artifact"].get("sha256")
    require(declared is None or final_head["sha256"] == declared, "gsq_final_fingerprint")
    base = unique(sorted(root.glob("**/gsq3-base/qwen38-27b-gsq3/config.json")), "gsq_base")
    files = []
    for n, size in GSQ_BASE_BYTES.items():
        item = info(base.parent / n, root, not n.endswith(".safetensors"))
        require(item["bytes"] == size, "gsq_base_file_size:" + n)
        files.append(item)
    return {"final": final_head, "manifest": info(meta, root), "complete": info(complete, root),
            "final_matches_declared_sha": declared is not None, "base_inventory": files}


def audit(input_root, expected_prefit_sha256=None):
    root = Path(input_root).absolute()
    expected = expected_prefit_sha256 or {}
    report = {"checks": {}, "failures": {}, "executed_upstream_code": False, "pickle_loaded": False,
              "models_loaded": False, "test_data_read": False}
    for name, fn in (("prefit", lambda: prefit(root, expected)), ("qwen25", lambda: qwen25(root)),
                     ("main_torch_heads", lambda: main_heads(root)), ("ultrabert", lambda: ultrabert(root)),
                     ("gsq", lambda: gsq(root))):
        try:
            report["checks"][name] = fn()
        except ValueError as e:
            report["failures"][name] = str(e)
        except (OSError, zipfile.BadZipFile, TypeError, KeyError, RuntimeError) as e:
            report["failures"][name] = type(e).__name__
    report["status"] = "BLOCK_STATIC_V25_EXECUTABLE_INVENTORY" if report["failures"] else "PASS_STATIC_V25_EXECUTABLE_INVENTORY_ONLY"
    return report
