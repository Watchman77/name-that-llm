"""Run one prospectively fixed comparison on the native prompt splits."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "2"

import numpy as np
from threadpoolctl import threadpool_limits

from stage14_contrastive_head import fit_pair
from stage14_contrastive_inputs import baseline, load
from stage14_contrastive_metrics import evaluate


SEED = 141400
FILES = (
    "stage14_contrastive_run.py", "stage14_contrastive_head.py", "stage14_contrastive_inputs.py",
    "stage14_contrastive_metrics.py", "stage14_native_oof_audit.py", "stage14_native_input_audit.py",
    "stage13a_cache_admission.py", "stage13_glm_readout.py", "stage13c_deberta_complement.py",
    "stage13b_balancing_stress.py", "stage13f_assignment_core.py",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise ValueError("refusing to overwrite a prior experiment")
    started = time.monotonic()
    bundle = load(args.input_root)
    print("STAGE14B_INPUTS_ADMITTED", flush=True)
    n_rows, n_classes = len(bundle.y), len(bundle.classes)
    controls, ce, contrast = (np.full((n_rows, n_classes), np.nan) for _ in range(3))
    fold_reports = []
    with threadpool_limits(limits=2):
        for fold in range(5):
            fold_start = time.monotonic()
            score = bundle.folds == fold
            fit = ~score
            fit_prompts, score_prompts = set(bundle.prompts[fit]), set(bundle.prompts[score])
            if len(fit_prompts) != 120 or len(score_prompts) != 30 or fit_prompts & score_prompts:
                raise ValueError("not an exact prompt-held-out split")
            print("STAGE14B_FOLD_START", fold, flush=True)
            controls[score], iterations = baseline(bundle, fold)
            ce[score], contrast[score] = fit_pair(bundle.x[fit], bundle.y[fit], bundle.prompts[fit],
                                                 bundle.x[score], SEED + fold)
            record = {"fold": fold, "fit_rows": int(fit.sum()), "score_rows": int(score.sum()),
                      "fit_prompts": len(fit_prompts), "score_prompts": len(score_prompts),
                      "baseline_iterations": iterations, "elapsed_seconds": round(time.monotonic() - fold_start, 3)}
            fold_reports.append(record)
            print("STAGE14B_FOLD_DONE", json.dumps(record, sort_keys=True), flush=True)
        results = evaluate(bundle.texts, bundle.prompts, bundle.y, bundle.folds, bundle.classes,
                           controls, ce, contrast)
    root = Path(__file__).parent
    report = {"status": "STAGE14B_PAIRED_DEVELOPMENT_COMPLETE", "scope": "reused_official_training_prompts",
              "official_score_measured": False, "raw_predictions_exported": False,
              "native_reference_logits_read": False, "rows": n_rows, "classes": n_classes,
              "seed_base": SEED, "folds": fold_reports, "inputs": bundle.evidence, "results": results,
              "dependencies": {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "scikit-learn", "torch")},
              "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in FILES},
              "elapsed_seconds": round(time.monotonic() - started, 3)}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    summary = {k: results[k] for k in ("primary", "secondary", "promotion")}
    print("STAGE14B_PAIRED_DEVELOPMENT_COMPLETE", json.dumps(summary, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
