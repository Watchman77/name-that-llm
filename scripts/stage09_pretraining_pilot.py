"""Stage 09: frozen, narrow external-data pretraining pilot; no test inference.

The two arms compare the staged procedure, including extra pretraining compute.
They do not isolate a compute-matched data effect or test the S03 ensemble.
Torch/Transformers are imported only inside run(); --self-test is CPU-only.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import random
import time
import traceback

import numpy as np
import pandas as pd

SEED = 20261024
MODEL_ID = "microsoft/deberta-v3-base"
REVISION = "8ccc9b6f36199bec6961081d44eb72fb3f7353f3"
FOLD = 1
OFFICIAL_SHA = "40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707"
MAX_LENGTH, BATCH_SIZE, ACCUMULATION = 384, 4, 4
PRETRAIN_EPOCHS, OFFICIAL_EPOCHS = 2, 3
PRETRAIN_ENCODER_LR, PRETRAIN_HEAD_LR = 1.5e-5, 1e-3
OFFICIAL_ENCODER_LR, OFFICIAL_HEAD_LR = 1e-5, 5e-4
GPU_BUDGET_SECONDS = 3 * 60 * 60
PROTOCOL = {
    "stage": "09", "seed": SEED, "fold": FOLD,
    "model_id": MODEL_ID, "revision": REVISION,
    "architecture": "AutoModel attention-mask mean of last hidden state + single Linear(hidden,18); no added dropout",
    "max_length": MAX_LENGTH, "batch_size": BATCH_SIZE,
    "gradient_accumulation": ACCUMULATION,
    "pretrain_epochs": PRETRAIN_EPOCHS, "official_epochs": OFFICIAL_EPOCHS,
    "pretrain_encoder_lr": PRETRAIN_ENCODER_LR, "pretrain_head_lr": PRETRAIN_HEAD_LR,
    "official_encoder_lr": OFFICIAL_ENCODER_LR, "official_head_lr": OFFICIAL_HEAD_LR,
    "official_train_rows": 2159, "official_validation_rows": 540,
    "external_rows": 750, "external_classes": 5,
    "pretrain_sampler": "all 750 admitted rows shuffled without replacement each epoch; exact 150 examples per each of five GPT classes",
    "official_loss": "inverse-class-support weights normalized to mean one; sum weighted CE / accumulation-group actual rows",
    "head_reset": "identical initial linear head in both official arms",
    "optimizer": "fresh AdamW each phase; betas=(0.9,0.999), eps=1e-8, weight_decay=0.01 except biases/LayerNorm",
    "scheduler": "linear decay, ceil(10% updates) warmup; new scheduler each phase",
    "amp": "float32 parameters with fp16 autocast and GradScaler; any overflow fails phase",
    "gradient_clip_norm": 1.0, "validation": "once, after fixed final official epoch; no early stopping",
    "bootstrap": {"resamples": 10000, "unit": "official prompt_id", "seed": SEED},
    "gpu_budget_seconds": GPU_BUDGET_SECONDS,
    "required_allocated_gpus": 2,
    "budget_unit": "allocated CUDA device count times wall seconds; cumulative across resumes",
    "scope": "reused-development-fold pilot; narrow V4 direct-OpenAI five-GPT subset; not 83k upstream reproduction",
    "comparison": "whole staged procedure including extra pretraining compute; not compute-matched data effect or S03 ensemble gain",
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def inverse_class_weights(y, classes):
    support = np.bincount(np.asarray(y, dtype=int), minlength=classes)
    if len(support) != classes or (support == 0).any():
        raise ValueError("Official fit data must support every declared class")
    weights = 1.0 / support
    weights /= weights.mean()
    return support, weights


def require_balanced_external(y):
    counts = Counter(np.asarray(y, dtype=int).tolist())
    if len(counts) != 5 or set(counts.values()) != {150}:
        raise ValueError("Frozen external subset must contain exactly 150 rows in each of five classes")
    return counts


def accumulation_groups(row_count, batch_size=BATCH_SIZE, accumulation=ACCUMULATION):
    sizes = [min(batch_size, row_count - start) for start in range(0, row_count, batch_size)]
    return [sizes[start:start + accumulation] for start in range(0, len(sizes), accumulation)]


def token_length_summary(lengths):
    lengths = np.asarray(lengths, dtype=np.int64)
    if lengths.ndim != 1 or len(lengths) == 0 or (lengths < 1).any():
        raise ValueError("Token lengths must be a nonempty positive vector")
    truncated = int((lengths > MAX_LENGTH).sum())
    return {"rows": len(lengths), "truncated_rows": truncated,
            "truncated_fraction": truncated / len(lengths),
            "untruncated_token_length_quantiles": {
                str(q): float(np.quantile(lengths, q)) for q in [0, .5, .9, .95, 1]},
            "maximum_encoded_length": int(np.minimum(lengths, MAX_LENGTH).max())}


def validate_probabilities(probabilities, actual_ids, expected_ids, classes):
    p = np.asarray(probabilities)
    actual, expected = list(map(str, actual_ids)), list(map(str, expected_ids))
    if len(set(actual)) != len(actual) or actual != expected:
        raise ValueError("Validation row IDs must be unique and preserve frozen official order")
    if (not np.issubdtype(p.dtype, np.number) or not np.isrealobj(p) or
            p.shape != (len(expected), classes) or not np.isfinite(p).all()):
        raise ValueError("Probability shape/finiteness check failed")
    if (p < 0).any() or (p > 1).any() or not np.allclose(p.sum(1), 1, rtol=1e-5, atol=1e-6):
        raise ValueError("Probabilities must lie in [0,1] and sum to one")
    return p


def validate_prediction_payload(payload, expected_ids, labels):
    if list(map(str, payload["labels"])) != list(labels):
        raise ValueError("Saved probability columns have a different canonical label order")
    p = validate_probabilities(payload["probabilities"], payload["ids"], expected_ids, len(labels))
    logits = np.asarray(payload["logits"])
    if (not np.issubdtype(logits.dtype, np.number) or not np.isrealobj(logits) or
            logits.shape != p.shape or not np.isfinite(logits).all()):
        raise ValueError("Saved logits have invalid shape/type or nonfinite values")
    exponents = np.exp(logits.astype(np.float64) - logits.max(1, keepdims=True))
    expected = exponents / exponents.sum(1, keepdims=True)
    if not np.allclose(p, expected, rtol=1e-5, atol=1e-6):
        raise ValueError("Saved probabilities differ from softmax of saved logits")
    return p.copy()


def summarize_predictions(y, groups, control, candidate, labels):
    """Aggregate only. Original IDs and row predictions never enter this result."""
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix
    y, groups = np.asarray(y), np.asarray(groups)
    if len(y) != len(groups) or set(np.unique(y)) != set(range(len(labels))):
        raise ValueError("Validation support/alignment mismatch")
    preds = {"control": np.asarray(control), "candidate": np.asarray(candidate)}
    result, cms = {}, {}
    for arm, pred in preds.items():
        if pred.shape != y.shape or not np.isin(pred, np.arange(len(labels))).all():
            raise ValueError("Prediction alignment/label mismatch")
        cm = confusion_matrix(y, pred, labels=np.arange(len(labels)))
        cms[arm] = cm
        result[arm] = {"development_balanced_accuracy": float(balanced_accuracy_score(y, pred)),
                       "confusion_matrix": cm.tolist(),
                       "per_class": [{"label": lab, "support": int(cm[i].sum()),
                                      "recall": float(cm[i, i] / cm[i].sum())}
                                     for i, lab in enumerate(labels)]}
    prompt_values = sorted(np.unique(groups).tolist())
    gi = np.asarray([prompt_values.index(g) for g in groups])
    total = np.zeros((len(prompt_values), len(labels)), dtype=np.int64)
    np.add.at(total, (gi, y), 1)
    draws = np.random.default_rng(SEED).multinomial(
        len(prompt_values), np.full(len(prompt_values), 1 / len(prompt_values)), size=10000)
    denominators = draws @ total
    if (denominators == 0).any():
        raise ValueError("Bootstrap draw omitted a class; do not silently change estimand")
    values = {}
    for arm, pred in preds.items():
        correct = np.zeros_like(total)
        np.add.at(correct, (gi, y), (pred == y).astype(np.int64))
        values[arm] = ((draws @ correct) / denominators).mean(1)
    delta = result["candidate"]["development_balanced_accuracy"] - result["control"]["development_balanced_accuracy"]
    low, high = np.quantile(values["candidate"] - values["control"], [.025, .975])
    good_c, good_t = preds["control"] == y, preds["candidate"] == y
    result.update({"labels": labels, "validation_rows": len(y), "validation_prompts": len(prompt_values),
                   "delta_percentage_points": 100 * delta,
                   "paired_conditional_95_interval_percentage_points": [100 * float(low), 100 * float(high)],
                   "bootstrap_resamples": 10000, "bootstrap_seed": SEED,
                   "corrected": int((~good_c & good_t).sum()), "broken": int((good_c & ~good_t).sum()),
                   "changed_predictions": int((preds["control"] != preds["candidate"]).sum()),
                   "interpretation": "Reused fixed development fold, conditional on fitted models. No refitting/selection uncertainty. No automatic promotion, S03 ensemble gain, or official score claim.",
                   "comparison_scope": PROTOCOL["comparison"]})
    result["per_class_delta_percentage_points"] = [
        {"label": lab, "delta": 100 * (result["candidate"]["per_class"][i]["recall"] - result["control"]["per_class"][i]["recall"])}
        for i, lab in enumerate(labels)]
    result["fixed_pilot_screen_pass"] = bool(delta > 0 and low > 0)
    result["pilot_screen_scope"] = "Positive development BA difference and conditional paired 95% lower endpoint above zero earns broader grouped evaluation only. All class regressions retained; no automatic submission or ensemble claim."
    return result


class GpuBudget:
    def __init__(self, ledger_path, device_count, started=None):
        self.path, self.devices = Path(ledger_path), int(device_count)
        self.previous = json.loads(self.path.read_text())["gpu_seconds_consumed"] if self.path.exists() else 0.0
        self.started = time.perf_counter() if started is None else started
        if self.devices < 1:
            raise ValueError("No allocated GPU")

    def consumed(self):
        return self.previous + self.devices * (time.perf_counter() - self.started)

    def check(self, reserve_wall_seconds=120):
        consumed = self.consumed()
        save_json(self.path, {"gpu_seconds_consumed": consumed, "gpu_seconds_cap": GPU_BUDGET_SECONDS,
                             "allocated_cuda_devices": self.devices, "updated_utc": utc_now(),
                             "unit": PROTOCOL["budget_unit"]})
        if consumed >= GPU_BUDGET_SECONDS - self.devices * reserve_wall_seconds:
            raise TimeoutError("Frozen 3 GPU-hour budget guard reached; phase remains incomplete, schedule is not shortened")


def verify_resources(resource_dir, expected_manifest_sha):
    resource_dir = Path(resource_dir)
    manifest_path = resource_dir / "resources_manifest.json"
    if file_sha(manifest_path) != expected_manifest_sha:
        raise ValueError("Untouched vendor resource manifest SHA differs from prospective freeze")
    receipt = json.loads(manifest_path.read_text())
    if receipt.get("model_id") != MODEL_ID or receipt.get("revision") != REVISION:
        raise ValueError("Wrong vendor model or revision")
    model_dir = resource_dir / "deberta-v3-base"
    expected = receipt["model_files_sha256"]
    if not expected or "config.json" not in expected:
        raise ValueError("Resource manifest omits model config")
    if not any(name.endswith((".safetensors", ".bin")) for name in expected):
        raise ValueError("Resource manifest omits weights")
    for name, digest in expected.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or not model_dir.joinpath(relative).is_file():
            raise ValueError("Unsafe/missing model resource path")
        if file_sha(model_dir / relative) != digest:
            raise ValueError(f"Vendor resource changed: {name}")
    # The original snapshot manifest intentionally excludes Hub cache metadata.
    # Preserve that same inventory rule; cache receipts are not model inputs.
    model_assets = {str(p.relative_to(model_dir)) for p in model_dir.rglob("*")
                    if p.is_file() and ".cache" not in p.relative_to(model_dir).parts}
    if model_assets != set(expected):
        raise ValueError("Model asset inventory differs; no extra adapters/checkpoints permitted")
    return model_dir, receipt


def run(train_path, source_dir, resource_dir, out_dir, freeze_path, resume=False, allocation_started=None):
    # Offline inference APIs only; no Hub lookup, test file, paid API, or competition submission.
    allocation_started = time.perf_counter() if allocation_started is None else allocation_started
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", USE_TF="0", USE_FLAX="0",
                      TOKENIZERS_PARALLELISM="false", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    import importlib.metadata
    import torch
    from torch import nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
    from stage09_data_audit import audit, frozen_split

    train_path, source_dir, resource_dir = map(Path, (train_path, source_dir, resource_dir))
    out_dir, freeze_path = Path(out_dir), Path(freeze_path)
    if out_dir.exists() and not resume:
        raise FileExistsError("Preserve previous output; use explicit boundary-only --resume")
    out_dir.mkdir(parents=True, exist_ok=True)
    public, private = out_dir / "aggregate", out_dir / "private"
    public.mkdir(exist_ok=True)
    private.mkdir(exist_ok=True)
    freeze = json.loads(freeze_path.read_text())
    if freeze.get("status") != "FROZEN_BEFORE_GPU_TRAINING" or freeze.get("protocol") != PROTOCOL:
        raise ValueError("Prospective source freeze missing or protocol differs")
    if freeze.get("runner_sha256") != file_sha(__file__):
        raise ValueError("Runner source differs from prospective freeze")
    import stage09_data_audit
    if freeze.get("audit_script_sha256") != file_sha(stage09_data_audit.__file__):
        raise ValueError("Data-audit source differs from prospective freeze")
    freeze_sha = file_sha(freeze_path)
    status_path = public / "STATUS.json"
    prior_state = json.loads(status_path.read_text()) if status_path.exists() else {}
    if prior_state and prior_state.get("freeze_sha256") != freeze_sha:
        raise ValueError("Resume freeze differs from original attempt")
    assert torch.cuda.is_available(), "Included T4 CUDA GPU required; no CPU fallback"
    allocated = torch.cuda.device_count()
    assert allocated == PROTOCOL["required_allocated_gpus"], "Frozen hardware requires two allocated T4 GPUs"
    assert all("T4" in torch.cuda.get_device_name(i) for i in range(allocated)), "Hardware amendment required"
    budget = GpuBudget(private / "budget_ledger.json", allocated, started=allocation_started)
    state = {"stage": "09", "status": "PREFLIGHT", "started_utc": prior_state.get("started_utc", utc_now()),
             "freeze_sha256": freeze_sha, "completed_phases": [],
             "actual_gpu_training": bool(prior_state.get("actual_gpu_training", False))}
    save_json(status_path, state)
    model = None

    def reset_seed():
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.use_deterministic_algorithms(True, warn_only=True)

    class MeanClassifier(nn.Module):
        def __init__(self, model_dir):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(model_dir, local_files_only=True).float()
            self.head = nn.Linear(self.encoder.config.hidden_size, 18)

        def forward(self, **inputs):
            hidden = self.encoder(**inputs).last_hidden_state.float()
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            return self.head(pooled)

    class TokenRows(Dataset):
        def __init__(self, encoded, y, indices):
            self.encoded, self.y, self.indices = encoded, y, np.asarray(indices)

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, pos):
            i = int(self.indices[pos])
            return {"inputs": {k: v[i] for k, v in self.encoded.items()}, "label": int(self.y[i]), "index": i}

    def collate(rows):
        values = tokenizer.pad([r["inputs"] for r in rows], padding=True, pad_to_multiple_of=8, return_tensors="pt")
        return values, torch.tensor([r["label"] for r in rows]), [r["index"] for r in rows]

    def checkpoint(phase, current_model, metadata):
        # Only complete phases are resumable. Never load an incomplete or unverified checkpoint.
        target = private / f"{phase}_COMPLETE"
        if target.exists():
            raise FileExistsError(f"Completed phase already exists: {phase}")
        target.mkdir()
        from safetensors.torch import save_file
        # Independent clones also serialize correctly if a future upstream
        # encoder exposes aliased state entries; no keys or tensors are omitted.
        cpu_state = {k: v.detach().cpu().contiguous().clone() for k, v in current_model.state_dict().items()}
        save_file(cpu_state, str(target / "model.safetensors"))
        del cpu_state
        save_json(target / "COMPLETE.json", {"phase": phase, "freeze_sha256": freeze_sha,
                                            "model_sha256": file_sha(target / "model.safetensors"), **metadata})
        return target

    def load_complete(phase, current_model):
        target = private / f"{phase}_COMPLETE"
        if not target.exists():
            return None
        meta = json.loads((target / "COMPLETE.json").read_text())
        if meta["phase"] != phase or meta["freeze_sha256"] != freeze_sha or file_sha(target / "model.safetensors") != meta["model_sha256"]:
            raise ValueError("Completed phase checkpoint contract failed")
        from safetensors.torch import load_file
        current_model.load_state_dict(load_file(str(target / "model.safetensors"), device="cpu"), strict=True)
        return meta

    def parameter_groups(current_model, encoder_lr, head_lr):
        groups = []
        for prefix, module, lr in (("encoder", current_model.encoder, encoder_lr), ("head", current_model.head, head_lr)):
            for no_decay in (False, True):
                params = [p for name, p in module.named_parameters() if p.requires_grad and
                          ((name.endswith("bias") or "LayerNorm" in name or "layer_norm" in name) == no_decay)]
                if params:
                    groups.append({"params": params, "lr": lr, "weight_decay": 0.0 if no_decay else 0.01})
        assert sum(len(g["params"]) for g in groups) == sum(p.requires_grad for p in current_model.parameters())
        return groups

    def train_phase(phase, current_model, encoded, y, indices, epochs, encoder_lr, head_lr, pretraining=False):
        existing = load_complete(phase, current_model)
        if existing is not None:
            print("RESUME_COMPLETE_PHASE", phase, flush=True)
            return existing
        reset_seed()
        budget.check()
        current_model.to("cuda:0")
        data = TokenRows(encoded, y, indices)
        generator = torch.Generator().manual_seed(SEED)
        if pretraining:
            require_balanced_external(y[indices])
            loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, generator=generator,
                                num_workers=0, collate_fn=collate)
            loss_weights = None
        else:
            support, weights = inverse_class_weights(y[indices], 18)
            loss_weights = torch.tensor(weights, dtype=torch.float32, device="cuda:0")
            loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, generator=generator,
                                num_workers=0, collate_fn=collate)
        total_steps = epochs * math.ceil(len(loader) / ACCUMULATION)
        warmup = math.ceil(total_steps * .1)
        optimizer = torch.optim.AdamW(parameter_groups(current_model, encoder_lr, head_lr), betas=(.9, .999), eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup, total_steps)
        scaler = torch.amp.GradScaler("cuda", init_scale=1024.0, growth_interval=2000)
        rows_log, updates = [], 0
        phase_started = time.perf_counter()
        for epoch in range(1, epochs + 1):
            current_model.train()
            iterator, epoch_rows, loss_total = iter(loader), 0, 0.0
            sampled_labels = Counter()
            sampled_indices = []
            while True:
                batches = list(itertools.islice(iterator, ACCUMULATION))
                if not batches:
                    break
                budget.check()
                actual_rows = sum(len(batch[1]) for batch in batches)
                optimizer.zero_grad(set_to_none=True)
                for inputs, targets, batch_indices in batches:
                    targets = targets.to("cuda:0")
                    inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        logits = current_model(**inputs)
                        loss_sum = F.cross_entropy(logits.float(), targets, weight=loss_weights, reduction="sum")
                        loss = loss_sum / actual_rows
                    if not torch.isfinite(loss) or not torch.isfinite(logits).all():
                        raise FloatingPointError("Nonfinite training loss/logits; phase incomplete")
                    if state["status"] != "RUNNING":
                        first_started = state.get("first_training_backward_utc", utc_now())
                        state.update(status="RUNNING", actual_gpu_training=True,
                                     first_training_backward_utc=first_started)
                        save_json(status_path, state)
                    scaler.scale(loss).backward()
                    loss_total += float(loss_sum.detach().cpu())
                    epoch_rows += len(targets)
                    sampled_labels.update(targets.detach().cpu().tolist())
                    sampled_indices.extend(batch_indices)
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(current_model.parameters(), 1.0)
                prior_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() < prior_scale or not torch.isfinite(norm):
                    raise FloatingPointError("AMP overflow/nonfinite gradient; no skipped step or automatic recipe change")
                scheduler.step()
                updates += 1
                if updates % 25 == 0:
                    print(f"phase={phase} epoch={epoch}/{epochs} update={updates}/{total_steps} gpu_seconds={budget.consumed():.1f}", flush=True)
            assert epoch_rows == len(indices)
            assert sorted(sampled_indices) == sorted(map(int, indices))
            if pretraining:
                assert sampled_labels == require_balanced_external(y[indices])
            row = {"phase": phase, "epoch": epoch, "rows": epoch_rows,
                   "sampled_class_counts": {labels[k]: v for k, v in sorted(sampled_labels.items())},
                   "sample_order_sha256": canonical_sha(sampled_indices),
                   "mean_weighted_loss": loss_total / epoch_rows, "cumulative_updates": updates,
                   "validation_performed": False}
            rows_log.append(row)
            save_json(public / f"{phase}_training_log.json", rows_log)
            print("EPOCH_COMPLETE", json.dumps(row), flush=True)
        assert updates == total_steps
        assert all(torch.isfinite(p).all() for p in current_model.parameters())
        metadata = {"epochs": epochs, "optimizer_steps": updates, "warmup_steps": warmup,
                    "seconds": time.perf_counter() - phase_started, "training_rows_per_epoch": len(indices),
                    "encoder_lr": encoder_lr, "head_lr": head_lr, "amp_overflow_steps": 0}
        checkpoint(phase, current_model, metadata)
        budget.check()
        del optimizer, scheduler, scaler
        return metadata

    def evaluate_once(arm, current_model):
        target = private / f"{arm}_validation.npz"
        receipt_path = private / f"{arm}_validation.json"
        expected_ids = train.iloc[valid_idx].id.astype(str).tolist()
        if target.exists():
            meta = json.loads(receipt_path.read_text())
            if meta["freeze_sha256"] != freeze_sha or meta["sha256"] != file_sha(target):
                raise ValueError("Saved evaluation receipt differs")
            with np.load(target, allow_pickle=False) as values:
                return validate_prediction_payload(values, expected_ids, labels)
        current_model.to("cuda:0").eval()
        loader = DataLoader(TokenRows(official_encoded, y, valid_idx), batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=0, collate_fn=collate)
        logits_values, indices = [], []
        for inputs, _, batch_indices in loader:
            budget.check()
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = current_model(**{k: v.to("cuda:0") for k, v in inputs.items()})
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Nonfinite final validation logits")
            logits_values.append(logits.float().cpu().numpy())
            indices.extend(batch_indices)
        assert indices == valid_idx.tolist()
        logits = np.concatenate(logits_values)
        e = np.exp(logits - logits.max(1, keepdims=True))
        probabilities = e / e.sum(1, keepdims=True)
        validate_probabilities(probabilities, train.iloc[indices].id.tolist(), expected_ids, 18)
        np.savez_compressed(target, ids=np.asarray(expected_ids), labels=np.asarray(labels),
                            logits=logits, probabilities=probabilities)
        save_json(receipt_path, {"freeze_sha256": freeze_sha, "sha256": file_sha(target),
                                 "rows": len(expected_ids), "fixed_final_epoch": OFFICIAL_EPOCHS})
        with np.load(target, allow_pickle=False) as values:
            return validate_prediction_payload(values, expected_ids, labels)

    try:
        budget.check()
        train, external, audit_receipt = audit(train_path, source_dir, public / "data_audit", persist=True)
        if audit_receipt.get("status") != "PASS_FROZEN_PILOT" or audit_receipt.get("admission_fingerprint") != freeze["data_admission_fingerprint"]:
            raise ValueError("Real CPU data-admission receipt differs from prospective freeze")
        if file_sha(train_path) != OFFICIAL_SHA:
            raise ValueError("Official train hash differs")
        train_idx, valid_idx, split_receipt = frozen_split(train)
        train_idx, valid_idx = np.asarray(train_idx), np.asarray(valid_idx)
        assert len(train_idx) == 2159 and len(valid_idx) == 540
        assert set(train.iloc[train_idx].prompt_id).isdisjoint(train.iloc[valid_idx].prompt_id)
        labels = sorted(train.model.unique().tolist())
        assert len(labels) == 18 and set(train.iloc[valid_idx].model) == set(train.iloc[train_idx].model) == set(labels)
        y = train.model.map({label: i for i, label in enumerate(labels)}).to_numpy(dtype=int)
        ey = external.model.map({label: i for i, label in enumerate(labels)}).to_numpy(dtype=int)
        assert len(external) == 750 and len(np.unique(ey)) == 5 and not np.isnan(ey).any()
        model_dir, resource_receipt = verify_resources(resource_dir, freeze["resource_manifest_sha256"])
        environment = {"python": platform.python_version(),
                       "packages": {name: importlib.metadata.version(name) for name in ["torch", "transformers", "sentencepiece", "numpy", "pandas", "scikit-learn", "safetensors"]},
                       "cuda": torch.version.cuda, "allocated_gpu_count": allocated,
                       "gpu_names": [torch.cuda.get_device_name(i) for i in range(allocated)],
                       "training_device": "cuda:0", "all_allocated_devices_count_towards_budget": True,
                       "determinism": "Seeded; deterministic algorithms warn-only; no bitwise GPU guarantee"}
        assert environment["packages"]["transformers"] == "4.56.2", "Frozen transformer version required"
        assert environment["packages"]["sentencepiece"] == "0.2.1", "Frozen sentencepiece required"
        assert tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2]) >= (2, 6)
        save_json(public / "environment.json", environment)
        save_json(public / "protocol.json", PROTOCOL)
        save_json(public / "resources_manifest.json", resource_receipt)
        save_json(public / "label_order.json", labels)
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, use_fast=False)
        tokenizer.truncation_side = "right"
        official_full_lengths = np.asarray([len(ids) for ids in tokenizer(
            train.text.tolist(), truncation=False, padding=False)["input_ids"]])
        external_full_lengths = np.asarray([len(ids) for ids in tokenizer(
            external.text.tolist(), truncation=False, padding=False)["input_ids"]])
        tokenization_receipt = {
            "max_length": MAX_LENGTH, "use_fast": False, "truncation_side": "right",
            "padding": "dynamic to multiple of 8", "special_tokens_included": True,
            "official_training": token_length_summary(official_full_lengths[train_idx]),
            "official_validation": token_length_summary(official_full_lengths[valid_idx]),
            "external_pretraining": token_length_summary(external_full_lengths),
        }
        save_json(public / "tokenization.json", tokenization_receipt)
        official_encoded = tokenizer(train.text.tolist(), truncation=True, max_length=MAX_LENGTH, padding=False)
        external_encoded = tokenizer(external.text.tolist(), truncation=True, max_length=MAX_LENGTH, padding=False)
        assert max(map(len, official_encoded["input_ids"])) <= MAX_LENGTH
        assert max(map(len, external_encoded["input_ids"])) <= MAX_LENGTH
        reset_seed()
        model = MeanClassifier(model_dir)
        initial_head = {k: v.detach().clone() for k, v in model.head.state_dict().items()}
        initial_head_sha = hashlib.sha256(b"".join(initial_head[k].numpy().tobytes() for k in sorted(initial_head))).hexdigest()
        save_json(public / "initial_head.json", {"sha256": initial_head_sha, "reset_before_both_official_arms": True})
        # Exercise the real graph on two arbitrary fixtures before any fitting.
        # No optimizer exists or is stepped, and fixture loss is not an accuracy result.
        budget.check()
        gpu_preflight = {"status": "RUNNING", "fixture_rows": 2, "no_optimizer_step": True,
                         "actual_model_fitting": False, "initial_head_sha256": initial_head_sha}
        save_json(public / "GPU_PREFLIGHT.json", gpu_preflight)
        try:
            model.to("cuda:0").eval()
            model.zero_grad(set_to_none=True)
            fixture = tokenizer(["Offline graph fixture one.", "A second brief graph fixture."],
                                padding=True, pad_to_multiple_of=8, return_tensors="pt")
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                fixture_logits = model(**{k: v.to("cuda:0") for k, v in fixture.items()})
                fixture_loss = F.cross_entropy(fixture_logits.float(), torch.tensor([0, 1], device="cuda:0"))
            assert fixture_logits.shape == (2, 18) and torch.isfinite(fixture_logits).all()
            assert torch.isfinite(fixture_loss)
            fixture_loss.backward()
            used_gradients = [(name, p.grad) for name, p in model.named_parameters() if p.grad is not None]
            assert used_gradients and all(torch.isfinite(g).all() for _, g in used_gradients)
            assert any(name.startswith("encoder.") for name, _ in used_gradients)
            assert any(name.startswith("head.") for name, _ in used_gradients)
            head_after = model.head.state_dict()
            head_after_sha = hashlib.sha256(b"".join(head_after[k].detach().cpu().numpy().tobytes()
                                                     for k in sorted(head_after))).hexdigest()
            assert head_after_sha == initial_head_sha, "No-step preflight changed classifier weights"
            gpu_preflight.update(status="PASS_GRAPH_NO_OPTIMIZER_STEP", logit_shape=[2, 18],
                                 logits_finite=True, loss_finite=True, used_gradient_tensors=len(used_gradients),
                                 all_used_gradients_finite=True, head_after_sha256=head_after_sha,
                                 head_unchanged=True, environment=environment)
            model.zero_grad(set_to_none=True)
            del fixture, fixture_logits, fixture_loss, used_gradients, head_after
            torch.cuda.empty_cache()
            save_json(public / "GPU_PREFLIGHT.json", gpu_preflight)
            print("GPU_PREFLIGHT_PASS", json.dumps(gpu_preflight), flush=True)
        except Exception as preflight_error:
            gpu_preflight.update(status="FAIL_NO_FITTING", error_type=type(preflight_error).__name__,
                                 error=str(preflight_error))
            save_json(public / "GPU_PREFLIGHT.json", gpu_preflight)
            raise
        state["status"] = "PREFLIGHT_PASSED_READY_FOR_FITTING"
        save_json(status_path, state)
        phase_records = {}
        phase_records["candidate_pretrain"] = train_phase("candidate_pretrain", model, external_encoded, ey,
            np.arange(len(external)), PRETRAIN_EPOCHS, PRETRAIN_ENCODER_LR, PRETRAIN_HEAD_LR, pretraining=True)
        state["completed_phases"].append("candidate_pretrain")
        save_json(status_path, state)
        del model
        model = None
        gc.collect()
        torch.cuda.empty_cache()
        reset_seed()
        model = MeanClassifier(model_dir)
        model.head.load_state_dict(initial_head, strict=True)
        phase_records["control_official"] = train_phase("control_official", model, official_encoded, y,
            train_idx, OFFICIAL_EPOCHS, OFFICIAL_ENCODER_LR, OFFICIAL_HEAD_LR)
        control_probabilities = evaluate_once("control", model)
        state["completed_phases"].append("control_official")
        save_json(status_path, state)
        del model
        model = None
        gc.collect()
        torch.cuda.empty_cache()
        reset_seed()
        model = MeanClassifier(model_dir)
        assert load_complete("candidate_pretrain", model) is not None
        model.head.load_state_dict(initial_head, strict=True)
        phase_records["candidate_official"] = train_phase("candidate_official", model, official_encoded, y,
            train_idx, OFFICIAL_EPOCHS, OFFICIAL_ENCODER_LR, OFFICIAL_HEAD_LR)
        candidate_probabilities = evaluate_once("candidate", model)
        state["completed_phases"].append("candidate_official")
        result = summarize_predictions(y[valid_idx], train.iloc[valid_idx].prompt_id.to_numpy(),
            control_probabilities.argmax(1), candidate_probabilities.argmax(1), labels)
        budget.check(reserve_wall_seconds=0)
        result.update({"status": "COMPLETE_FIXED_FOLD_PILOT", "completed_utc": utc_now(), "protocol": PROTOCOL,
                       "freeze_sha256": freeze_sha, "phase_records": phase_records,
                       "gpu_seconds_consumed": budget.consumed(), "official_submission_made": False,
                       "test_file_accessed": False, "actual_gpu_training": state["actual_gpu_training"],
                       "tokenization": tokenization_receipt})
        save_json(public / "RESULT.json", result)
        pd.DataFrame({"id": train.iloc[valid_idx].id, "prompt_id": train.iloc[valid_idx].prompt_id,
                      "model": train.iloc[valid_idx].model,
                      "control": np.asarray(labels)[control_probabilities.argmax(1)],
                      "candidate": np.asarray(labels)[candidate_probabilities.argmax(1)]}).to_csv(private / "validation_predictions.csv", index=False)
        state.update(status="COMPLETE_FIXED_FOLD_PILOT", completed_utc=utc_now())
        save_json(status_path, state)
        print("STAGE09_COMPLETE", json.dumps(result, allow_nan=False), flush=True)
        return result
    except Exception as error:
        state.update(status="INCOMPLETE_NO_RESULT", error_type=type(error).__name__, error=str(error), failed_utc=utc_now())
        save_json(status_path, state)
        (private / "failure_traceback.txt").write_text(traceback.format_exc())
        budget.check(reserve_wall_seconds=0) if budget.consumed() < GPU_BUDGET_SECONDS else None
        print("STAGE09_INCOMPLETE", json.dumps(state), flush=True)
        raise
    finally:
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()


def self_test():
    checks = {}
    groups = accumulation_groups(2159)
    checks["official_exact_update_count_405"] = len(groups) * OFFICIAL_EPOCHS == 405
    checks["official_last_accumulation_group_15_rows"] = groups[-1] == [4, 4, 4, 3]
    checks["pretrain_exact_update_count_94"] = len(accumulation_groups(750)) * PRETRAIN_EPOCHS == 94
    checks["pretrain_last_accumulation_group_14_rows"] = accumulation_groups(750)[-1] == [4, 4, 4, 2]
    values = np.arange(1, 16, dtype=float) ** 2
    reconstructed = sum(values[sum(groups[-1][:i]):sum(groups[-1][:i + 1])].sum() / 15 for i in range(4))
    checks["partial_group_sum_loss_normalization_matches_example_mean"] = np.isclose(reconstructed, values.mean())
    # Analytic scalar linear-model gradient, independent of the implementation's Torch path.
    x, target, coef = np.arange(1., 16.), np.arange(15.) / 3, .7
    gradient = 2 * (coef * x - target) * x
    accumulated = sum(gradient[start:start + size].sum() / 15 for start, size in zip([0, 4, 8, 12], groups[-1]))
    checks["partial_group_analytic_gradient_matches_full_batch"] = np.isclose(accumulated, gradient.mean())
    y = np.repeat(np.arange(5), 150)
    order = np.random.default_rng(SEED).permutation(len(y))
    checks["external_full_shuffle_exact_balanced_class_exposure"] = require_balanced_external(y[order]) == Counter({i: 150 for i in range(5)})
    checks["external_full_shuffle_no_duplicate_or_omitted_rows"] = sorted(order.tolist()) == list(range(750))
    try:
        require_balanced_external(y[:-1])
    except ValueError:
        checks["external_imbalanced_subset_rejected"] = True
    else:
        checks["external_imbalanced_subset_rejected"] = False
    support, official_weights = inverse_class_weights(np.repeat(np.arange(18), np.arange(1, 19)), 18)
    checks["official_weight_mass_balanced"] = np.allclose(support * official_weights, support[0] * official_weights[0])
    checks["official_weights_mean_one"] = np.isclose(official_weights.mean(), 1)
    lengths = token_length_summary([2, MAX_LENGTH, MAX_LENGTH + 1])
    checks["truncation_incidence_uses_strict_boundary_including_specials"] = (
        lengths["rows"] == 3 and lengths["truncated_rows"] == 1 and
        lengths["truncated_fraction"] == 1 / 3 and lengths["maximum_encoded_length"] == MAX_LENGTH)
    ids = ["a", "b", "c"]
    valid = np.full((3, 18), 1 / 18)
    checks["finite_aligned_probability_table_accepted"] = validate_probabilities(valid, ids, ids, 18).shape == (3, 18)
    for name, p, actual in [("reordered_ids_rejected", valid, ids[::-1]),
                            ("duplicate_ids_rejected", valid, ["a", "a", "c"]),
                            ("nan_probability_rejected", np.full((3, 18), np.nan), ids),
                            ("wrong_row_count_rejected", valid[:2], ids),
                            ("unnormalized_probability_rejected", valid * 2, ids)]:
        try:
            validate_probabilities(p, actual, ids, 18)
        except ValueError:
            checks[name] = True
        else:
            checks[name] = False
    payload = {"ids": ids, "labels": [f"label{i}" for i in range(18)],
               "probabilities": valid, "logits": np.zeros((3, 18))}
    checks["saved_payload_validated_including_logits"] = validate_prediction_payload(payload, ids, payload["labels"]).shape == (3, 18)
    for name, field, value in [
        ("saved_payload_column_order_rejected", "labels", payload["labels"][::-1]),
        ("saved_payload_nonfinite_logits_rejected", "logits", np.full((3, 18), np.inf)),
        ("saved_payload_probability_logit_mismatch_rejected", "logits", np.tile(np.arange(18), (3, 1))),
    ]:
        try:
            validate_prediction_payload({**payload, field: value}, ids, payload["labels"])
        except ValueError:
            checks[name] = True
        else:
            checks[name] = False
    # Every prompt has every class: exact 0-to-1 gain must remain +100 points in all bootstrap draws.
    truth = np.tile(np.arange(18), 3)
    summary = summarize_predictions(truth, np.repeat(["g0", "g1", "g2"], 18), (truth + 1) % 18, truth,
                                    [f"label{i}" for i in range(18)])
    checks["paired_bootstrap_known_100_point_gain"] = summary["delta_percentage_points"] == 100 and summary["paired_conditional_95_interval_percentage_points"] == [100, 100]
    checks["paired_corrected_broken_known_contrast"] = summary["corrected"] == 54 and summary["broken"] == 0
    checks = {name: bool(value) for name, value in checks.items()}
    report = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
              "passed": sum(checks.values()), "total": len(checks),
              "actual_gpu_training_executed": False,
              "limits": ["CPU algebra/schema checks only; no Torch import, autograd, model load, AMP, CUDA, or GPU training executed.",
                         "Real corpus admission is separately mandatory; synthetic checks do not establish data independence."]}
    if not all(checks.values()):
        raise AssertionError(json.dumps(report))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--train-path", type=Path)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--resource-dir", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--freeze-path", type=Path)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    if arguments.self_test:
        print(json.dumps(self_test(), indent=2))
    else:
        required = [arguments.train_path, arguments.source_dir, arguments.resource_dir, arguments.out_dir, arguments.freeze_path]
        if any(value is None for value in required):
            parser.error("Training requires all five explicit paths; no automatic test/input discovery")
        run(*required, resume=arguments.resume)
