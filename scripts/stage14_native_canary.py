"""Exercise an audited NativeQ8 worker on a fixed official-train smoke sample.

Requires the public task asset fingerprint from the CPU audit, the exact reviewed
worker fingerprint, the pinned Qwen3-Embedding-8B base mount, and two T4s. This is
an inference feasibility check only. It emits no labels, predictions, or scores.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import struct
import tempfile
import time
import types

import numpy as np

from stage14_native_input_audit import audit, decode_json, need, stable_hash


SAMPLE_QUARTILES = 4
ROWS_PER_QUARTILE = 8
BATCH_SIZE = 4
FOLDS = tuple(range(5))


def pinned_worker(path, expected_hash):
    need(0 < path.stat().st_size <= 100_000, "unexpected worker source size")
    source = path.read_bytes()
    need(len(source) <= 100_000, "unexpected worker source size")
    need(hashlib.sha256(source).hexdigest() == expected_hash, "worker fingerprint mismatch")
    module = types.ModuleType("_reviewed_native_q8")
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


def check_base(base, assets):
    """Validate exact expected shard inventories and capture header identities."""
    idx = json.loads((assets / "base/model.safetensors.index.json").read_bytes())
    shards = {}
    for name, shard in idx["weight_map"].items():
        shards.setdefault(shard, set()).add(name)
    report = {}
    for shard, tensor_names in sorted(shards.items()):
        path = base / shard
        need(path.is_file(), f"missing pinned base shard {shard}")
        size = path.stat().st_size
        with path.open("rb") as handle:
            raw_size = handle.read(8)
            need(len(raw_size) == 8, f"invalid base shard header {shard}")
            length = struct.unpack("<Q", raw_size)[0]
            need(2 <= length <= min(size - 8, 2 * 1024 * 1024), f"invalid base shard header {shard}")
            raw = handle.read(length)
        need(len(raw) == length, f"short base shard header {shard}")
        header = decode_json(raw, shard)
        header.pop("__metadata__", None)
        need(set(header) == tensor_names, f"base shard inventory mismatch {shard}")
        ranges = sorted((tuple(v["data_offsets"]) for v in header.values()))
        end = 0
        for lo, hi in ranges:
            need(lo == end and hi > lo, f"base shard tensor coverage mismatch {shard}")
            end = hi
        need(end == size - 8 - length, f"base shard size mismatch {shard}")
        report[shard] = {"bytes": size, "tensors": len(header),
                         "header_sha256": hashlib.sha256(raw).hexdigest()}
    return report


def sample_train(path, tokenizer, windows):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [(r["id"], r["text"]) for r in csv.DictReader(handle)]
    need(len(rows) == 2699 and len({r[0] for r in rows}) == 2699, "official train row identity mismatch")
    ranked = []
    for pos in range(0, len(rows), 64):
        part = rows[pos:pos + 64]
        seqs, owners, weights = windows(tokenizer, [t for _, t in part])
        need(len(seqs) == len(owners) == len(weights) and all(0 < len(s) <= 512 for s in seqs),
             "worker windows exceed the original token limit")
        lengths = np.bincount(owners, weights=weights, minlength=len(part))
        n_windows = np.bincount(owners, minlength=len(part))
        need(lengths.shape == n_windows.shape == (len(part),) and (lengths > 0).all()
             and (n_windows > 0).all(), "worker window owner coverage mismatch")
        for (row_id, text), length, count in zip(part, lengths, n_windows):
            tie = hashlib.sha256(("stage14-native-canary-v1:" + row_id).encode()).hexdigest()
            ranked.append((int(length), tie, int(count), text))
    ranked.sort(key=lambda x: (x[0], x[1]))
    selected, summary = [], []
    for q in range(SAMPLE_QUARTILES):
        lo = q * len(ranked) // SAMPLE_QUARTILES
        hi = (q + 1) * len(ranked) // SAMPLE_QUARTILES
        picks = [ranked[lo + i * (hi - lo - 1) // (ROWS_PER_QUARTILE - 1)]
                 for i in range(ROWS_PER_QUARTILE)]
        selected.extend(picks)
        summary.append({"quartile": q + 1, "rows": len(picks),
                        "min_weighted_tokens": picks[0][0], "max_weighted_tokens": picks[-1][0],
                        "windows": sum(row[2] for row in picks),
                        "multiwindow_rows": sum(row[2] > 1 for row in picks)})
    return [row[3] for row in selected], stable_hash([row[1] for row in selected]), summary


def cuda_snapshot(torch):
    rows = []
    for i in range(2):
        torch.cuda.synchronize(i)
        rows.append({"device": i, "allocated_peak_bytes": torch.cuda.max_memory_allocated(i),
                     "reserved_peak_bytes": torch.cuda.max_memory_reserved(i)})
    return rows


def check_outputs(results, finite, folds, rows):
    need(np.array_equal(results["folds"], folds), "unexpected worker fold order or incomplete run")
    need(finite.dtype == np.bool_ and finite.shape == (rows,) and finite.all(),
         "worker found nonfinite intermediate output")
    for key in ("plain", "centered", "centered_bias"):
        expected = (len(folds), 18) if key == "centered_bias" else (len(folds), rows, 18)
        array = results[key]
        need(array.shape == expected and np.isfinite(array).all(), f"invalid {key} output")


def run(args):
    started = time.monotonic()
    preflight = audit(args.assets, args.task_sha256, args.train_csv)
    shard_meta = check_base(args.base, args.assets)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    need(torch.cuda.is_available() and torch.cuda.device_count() == 2, "requires exactly two CUDA devices")
    props = [torch.cuda.get_device_properties(i) for i in range(2)]
    need(all(p.name.upper().endswith("T4") for p in props), "requires a two-T4 allocation")
    worker = pinned_worker(args.worker, args.worker_sha256)
    dependencies = {package: importlib.metadata.version(package)
                    for package in ("torch", "transformers", "accelerate", "safetensors", "numpy")}
    records = []
    args.work.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="stage14-native-", dir=args.work) as directory:
        t0 = time.monotonic()
        model = worker.NativeQ8(args.base, args.assets, directory, gpus=2,
                                log=lambda *msg: print(*msg, flush=True))
        setup_seconds = time.monotonic() - t0
        placement = {0: 0, 1: 0}
        for p in model.model.parameters():
            need(p.device.type == "cuda" and p.device.index in placement, "unexpected backbone device")
            placement[p.device.index] += p.numel() * p.element_size()
        need(all(placement.values()), "backbone not split across the two T4s")
        setup_peaks = cuda_snapshot(torch)
        need(stable_hash(list(model.labels)) == preflight["class_order_sha256"], "worker class axis mismatch")
        texts, sample_digest, sample_bins = sample_train(args.train_csv, model.tokenizer, worker.windows)
        for i in range(2):
            torch.cuda.reset_peak_memory_stats(i)
        fold_started = time.monotonic()

        def on_fold(results, finite):
            nonlocal fold_started
            done = len(records) + 1
            check_outputs(results, finite, FOLDS[:done], len(texts))
            record = {"fold": done - 1, "cuda": cuda_snapshot(torch),
                      "seconds_including_adapter_load": round(time.monotonic() - fold_started, 3)}
            records.append(record)
            print("STAGE14_NATIVE_GPU_FOLD", json.dumps(record, sort_keys=True), flush=True)
            for i in range(2):
                torch.cuda.reset_peak_memory_stats(i)
            fold_started = time.monotonic()

        results, finite = model.predict(texts, folds=FOLDS, batch_size=BATCH_SIZE, on_fold=on_fold)
        check_outputs(results, finite, FOLDS, len(texts))
        need(len(records) == 5, "did not observe all fold callbacks")
        states = [json.loads((args.assets / f"calibration/fold{k}.json").read_bytes())["models"]["qwen8"]
                  for k in FOLDS]
        groups = np.arange(len(texts)) // 4  # synthetic groups exercise arithmetic only
        for centered in (False, True):
            calibrated = worker.calibrate(results["plain"], results["centered"], results["centered_bias"],
                                          groups, states, use_centered=centered)
            need(calibrated.shape == (5, len(texts), 18) and np.isfinite(calibrated).all()
                 and np.allclose(np.exp(calibrated).sum(axis=-1), 1, rtol=0, atol=1e-12),
                 "calibrated outputs are not finite normalized probabilities")
    return {"status": "PASS_NATIVE_Q8_GPU_FEASIBILITY_ONLY", "worker_sha256": args.worker_sha256,
            "task_asset_tree_sha256": preflight["task_asset_tree_sha256"],
            "class_order_sha256": preflight["class_order_sha256"], "official_train_sha256_verified": True,
            "sample_sha256": sample_digest, "sample_rows": len(texts), "length_quartiles": sample_bins,
            "batch_size": BATCH_SIZE, "classes": 18, "base_shard_headers": shard_meta,
            "base_shard_payload_hashes_checked": False, "synthetic_group_calibration_checked": True,
            "device_names": [p.name for p in props], "device_total_bytes": [p.total_memory for p in props],
            "backbone_parameter_bytes_per_device": placement, "setup_cuda_peaks": setup_peaks,
            "dependency_versions": dependencies,
            "model_setup_seconds": round(setup_seconds, 3), "elapsed_seconds": round(time.monotonic() - started, 3),
            "folds": records, "accuracy_measured": False, "full_ensemble_run": False}


def sha256(value):
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise argparse.ArgumentTypeError("expected a lowercase SHA-256 fingerprint")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("assets", "base", "train-csv", "worker", "work", "report"):
        parser.add_argument("--" + name, required=True, type=Path)
    for name in ("task-sha256", "worker-sha256"):
        parser.add_argument("--" + name, required=True, type=sha256)
    args = parser.parse_args()
    need(not args.report.exists(), "refusing to replace an existing report")
    result = run(args)
    rendered = json.dumps(result, sort_keys=True)
    with args.report.open("x", encoding="utf-8") as handle:
        handle.write(rendered + "\n")
    print("STAGE14_NATIVE_GPU_CANARY", rendered, flush=True)


if __name__ == "__main__":
    main()
