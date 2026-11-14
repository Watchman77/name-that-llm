"""Read-only Stage 14C mount audit for Alex Chilton V25.

Does not import upstream modules, install wheels, unpickle, read test passages
or fit/infer. All paths in the JSON report are relative to the mounted root.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

SLUGS = (
    "name-that-llm-desklib-weights", "name-that-llm-electra-large-weights",
    "name-that-llm-extract-encoders2", "name-that-llm-glm4-9b-trunc25",
    "name-that-llm-open-pool", "name-that-llm-train-lora24-s7", "name-that-llm-train-tokattn",
    "name-that-llm-train-tokattn2", "nlm-l24-adapters", "nlm-l24-adapters-v5",
    "nlm-matched-clean", "nlm-offline-wheels", "ntl-extra-api-passages",
    "ntl-free-proxy-generations", "ntl-generations", "ntl-handwriting-blend",
    "ntl-olmo2-native24-final-inference-assets", "ntl-proxy-generations", "ntl-q14-lora",
    "ntl-q14-lora-v4", "ntl-qwen3-8b-lora-claude-head", "ntl-train-lora-qwen25-instyle",
    "ntl-two-stage-lora", "prvsiyans-extended-name-that-llm-dataset", "ultrabert-380m-ntl",
)
MODELS = (
    "llama-3-1-8b-base/transformers/default/1", "qwen-3-embedding/transformers/8b/1",
    "qwen-3/transformers/14b-base/1", "qwen-3/transformers/14b/1",
    "qwen-3/transformers/4b-base/1", "qwen-3/transformers/8b-base/1",
    "qwen-3/transformers/8b/1", "qwen2.5/transformers/14b/1",
)
META_HASHES = {
    "config.json": "54acfad3cffe057640904ca8a1e83525e6551c70c7a04c641f5a9eda0bbf64bd",
    "tokenizer.json": "76e48799b099d43365bd24ccd8ecc5aedac831718da780552f03b0a6eb4412aa",
    "tokenizer_config.json": "8004530facf809ac432114de2a4dcc65fcb632da5ec16d666091aeb6a2ee444a",
    "special_tokens_map.json": "462d91939dbc37178aa5a3eae7068d1990ccc92e09f288cc71f42cdf139d69cc",
    "model.safetensors.index.json": "146776fce3f6db1103aa6f249e65ee5544c5923ce6f971b092eee79aa6e5d37b",
}
LLAMA_SHARDS = (4976698672, 4999802720, 4915916176, 1168138808)
PY_HASHES = {
    "configuration_ultrabert.py": "823d10bda19f5164da4f79f8f9c01e5ebd7a75569abce78c0a26b7674fc78cd9",
    "modeling_ultrabert.py": "bfca1bd08af72cd170e58b5a048c67b52bc9d06a2a42847c7a387751f527f333",
}
ATTN_SHA = "ed384fd1af244ad1e9dca42194e423ad7fc2cce7e1926820344bffec6de3c561"
A_BATCH = {"b001", "b001g", "b003", "b003a", "d1028", "d1029", "d1030", "pilot", "pilot2", "pilot3", "pilotpro"}
L_BATCH = {f"b{i}" for i in (1, 2, *range(5, 16))} | {f"v2b{i:02d}" for i in range(1, 21)}
PRVS_VERSIONS = {1.0, 2.0, 4.0, 5.0}
TRAIN_SHA = "40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707"


def file_info(path, root):
    with path.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    return {"path": path.relative_to(root).as_posix(), "sha256": digest, "bytes": path.stat().st_size}


def read_json(path, limit=2**24):
    if not 0 < path.stat().st_size <= limit:
        raise ValueError("unexpected JSON length")
    with path.open("rb") as f:
        return json.load(f)


def same_file_candidates(paths):
    return list({p.resolve(): p for p in paths if p.is_file()}.values())


def numeric_in(x, wanted):
    try:
        return float(x) in wanted
    except (ValueError, TypeError):
        return False


def csv_inventory(path, root):
    filters = {
        "ntl_generations.csv": ("batch", A_BATCH, "finish_reason"),
        "gen_passages.csv": ("batch", L_BATCH, None),
        "passages_clean.csv": ("dataset_version_added", PRVS_VERSIONS, None),
        "ntl_generated*.csv": (None, (), "finish"),
    }
    key = "ntl_generated*.csv" if re.fullmatch("ntl_generated.*[.]csv", path.name) else path.name
    fld, wanted, finish = filters.get(key, (None, (), None))
    total, retained = 0, 0
    csv.field_size_limit(2**24)
    with path.open(encoding="utf-8", newline="") as f:
        rows = csv.DictReader(f)
        schema = rows.fieldnames or []
        for row in rows:
            total += 1
            passed = (not fld or fld not in schema
                      or (numeric_in(row[fld], wanted) if fld == "dataset_version_added" else row[fld] in wanted))
            retained += int(passed and (not finish or finish not in schema or row[finish] == "stop"))
    return {**file_info(path, root), "rows": total, "retained_before_text_filters": retained,
            "required_selector_column_present": fld is None or fld in schema}


def audit(root):
    root = Path(root).absolute()
    failures, notes = [], []
    dirs = [p for p in root.glob("*") if p.is_dir()]
    dirs += [p for p in root.glob("*/*") if p.is_dir()]
    dirs += [p for p in root.glob("*/*/*") if p.is_dir()]
    sources = {}
    for slug in SLUGS:
        sources[slug] = sorted({p.relative_to(root).as_posix() for p in dirs if p.name == slug})
        if not sources[slug]:
            failures.append("source_missing:" + slug)
        elif len({(root / p).resolve() for p in sources[slug]}) != 1:
            failures.append("source_ambiguous:" + slug)
    models = {}
    for suffix in MODELS:
        hit = [root / suffix, *root.glob(f"models/*/{suffix}")]
        models[suffix] = sorted({p.relative_to(root).as_posix() for p in hit if (p / "config.json").is_file()})
        if len({(root / p).resolve() for p in models[suffix]}) != 1:
            failures.append("model_missing_or_ambiguous:" + suffix)

    # V25 sorts full path strings, not pathlib's component tuples. In particular,
    # qwen-3-embedding sorts before qwen-3/ and can shadow the causal backbone.
    configs = sorted(root.rglob("config.json"), key=str)
    selectors = {
        "qwen2_3584_comp_7b": ("qwen2", 3584), "qwen3_4096": ("qwen3", 4096),
        "qwen3_2560": ("qwen3", 2560), "glm4_4096": ("glm4", 4096),
    }
    hits = {k: [] for k in selectors}
    for config in configs:
        try:
            d = read_json(config, 1024**2)
            shape = (d.get("model_type"), d.get("hidden_size"))
        except (ValueError, OSError, AttributeError):
            continue
        for key, expect in selectors.items():
            if shape == expect:
                hits[key].append(file_info(config, root))
    if not hits["qwen2_3584_comp_7b"]:
        failures.append("qwen2_3584_comp_7b_missing")
    if len({(root / x["path"]).resolve() for x in hits["qwen2_3584_comp_7b"]}) > 1:
        failures.append("qwen2_3584_comp_7b_ambiguous")
    for name, paths in hits.items():
        if not paths:
            failures.append("checkpoint_selector_missing:" + name)

    llama = {}
    alias = models[MODELS[0]]
    if alias:
        path = root / alias[0]
        static = {name: file_info(path / name, root) for name in META_HASHES if (path / name).is_file()}
        for name, d in static.items():
            d["matches_s07_first_party"] = d["sha256"] == META_HASHES[name]
        files = {}
        for i, size in enumerate(LLAMA_SHARDS, 1):
            name = f"model-{i:05d}-of-00004.safetensors"
            actual = (path / name).stat().st_size if (path / name).is_file() else None
            files[name] = {"bytes": actual, "matches_s07_first_party_size": actual == size}
        if not all(f["matches_s07_first_party_size"] for f in files.values()):
            failures.append("llama_shard_layout_changed")
        legal = []
        for item in sorted(path.iterdir()):
            if item.is_file() and re.match(r"^(license|notice|readme|use.policy)", item.name, re.I) and item.stat().st_size < 1024**2:
                t = item.read_text(encoding="utf-8", errors="replace").casefold()
                legal.append({**file_info(item, root), "mentions_llama_3_1": "llama 3.1" in t,
                              "mentions_built_with_llama": "built with llama" in t})
        llama = {"metadata": static, "shards": files, "license_readme_notice": legal,
                 "weight_payload_hashes_compared": False}
        if not any(Path(x["path"]).name.casefold().startswith("license") and x["mentions_llama_3_1"] for x in legal):
            notes.append("llama_license_copy_not_seen_inside_model_mount; review exact public model card")

    custom = {}
    for name, sha in PY_HASHES.items():
        paths = sorted(root.rglob(name))
        custom[name] = [{**file_info(p, root), "matches_s07_reviewed": file_info(p, root)["sha256"] == sha}
                        for p in paths if p.is_file() and "ultrabert-380m-ntl" in p.parts]
        if len({(root / d["path"]).resolve() for d in custom[name]}) != 5 or not all(d["matches_s07_reviewed"] for d in custom[name]):
            failures.append("ultrabert_python_missing_or_drift:" + name)
    attn = same_file_candidates(sorted(root.rglob("tokattn_debl23_final.pt")))
    attention = [file_info(p, root) for p in attn]
    if len(attention) != 1 or attention[0]["sha256"] != ATTN_SHA:
        failures.append("deberta_token_attention_missing_or_drift")

    trains = same_file_candidates(root / p for p in ("competitions/name-that-llm/train.csv", "name-that-llm/train.csv"))
    train = file_info(trains[0], root) if len(trains) == 1 else None
    if train is None or train["sha256"] != TRAIN_SHA:
        failures.append("competition_train_missing_or_drift")

    datasets = {}
    for glob in ("ntl_extra_passages_claude.csv", "gen_passages.csv", "ntl_generations.csv",
                 "ntl_proxy_generations.csv", "ntl_generated*.csv", "passages_clean.csv",
                 "ntl_open_pool.parquet", "nlm_matched_clean.parquet"):
        files = same_file_candidates(sorted(root.rglob(glob)))
        datasets[glob] = [csv_inventory(p, root) if p.suffix == ".csv" else file_info(p, root) for p in files]
        if glob in {"ntl_open_pool.parquet", "nlm_matched_clean.parquet"} and len(files) != 1:
            failures.append("required_pool_missing_or_ambiguous:" + glob)
        if any(not d.get("required_selector_column_present", True) for d in datasets[glob]):
            failures.append("csv_pin_filter_column_missing:" + glob)

    wheels = {}
    for name, paths in (("bitsandbytes", sorted(root.rglob("bitsandbytes-*.whl"))),
                        ("gsq", sorted(root.glob("**/name-that-llm-artifacts/qwen38-offline-linux/*.whl")))):
        wheels[name] = [file_info(p, root) for p in same_file_candidates(paths)]
    if len(wheels["gsq"]) < 6:
        failures.append("gsq_wheel_set_missing")

    prefit = {}
    pp = same_file_candidates(sorted(root.rglob("ntl666-cache-parts.json")))
    if len(pp) == 1:
        m = read_json(pp[0])
        parts = m.get("parts", [])
        manifest_ok = (m.get("format") == "ntl666-cache-parts-v1" and isinstance(parts, list) and
                       0 < len(parts) < 100 and all(isinstance(p, dict) for p in parts))
        parts_report = []
        if manifest_ok:
            for i, row in enumerate(parts):
                name = f"ntl666-prefit.part{i:03d}.bin"
                p = pp[0].parent / name
                if row.get("name") != name or not p.is_file() or p.stat().st_size != row.get("bytes"):
                    manifest_ok = False
                    break
                # Record per-part bytes, without extracting or deserializing the archive.
                info = file_info(p, root)
                expected = row.get("sha256")
                if expected is not None and expected != info["sha256"]:
                    manifest_ok = False
                parts_report.append({**info, "matches_declared_sha256": expected == info["sha256"] if expected is not None else None})
            manifest_ok &= sum(d["bytes"] for d in parts_report) == m.get("archive_bytes")
        if not manifest_ok:
            failures.append("prefit_archive_contract")
        prefit = {"manifest": file_info(pp[0], root), "parts": parts_report,
                  "archive_member_count_declared": m.get("files"), "archive_opened": False}
    else:
        failures.append("prefit_manifest_missing_or_ambiguous")

    return {"status": "BLOCK_STATIC_V25_MOUNT" if failures else "PASS_STATIC_V25_MOUNT_ONLY",
            "failures": failures, "notes": notes, "nonmodel_mounts": sources, "model_mounts": models,
            "checkpoint_selectors_source_sort": hits, "llama_mirror": llama, "ultrabert_code": custom,
            "deberta_attention": attention, "official_train_identity": train, "dataset_inventory": datasets,
            "offline_wheels": wheels, "prefit_parts": prefit, "test_data_read": False,
            "upstream_code_run": False, "pickle_loaded": False, "model_weights_loaded": False,
            "official_score_measured": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = audit(args.input_root)
    if args.out:
        with args.out.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    print("STAGE14C_STATIC_MOUNT_AUDIT", json.dumps(report, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
