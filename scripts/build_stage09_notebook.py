"""Build a portable, frozen Kaggle Stage 09 pilot notebook without fitting a model."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import nbformat

RESOURCE_MANIFEST_SHA = "f8e60471afe49ec5f883bf141fc100d84081b2fe6a0eb28b1f9236486b7532bf"


def load_runner(path):
    spec = importlib.util.spec_from_file_location("stage09_build_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build(repo, enable_training=False):
    repo = Path(repo).resolve()
    runner_path = repo / "scripts/stage09_pretraining_pilot.py"
    audit_path = repo / "scripts/stage09_data_audit.py"
    runner = load_runner(runner_path)
    audit_receipt_path = repo / "results/stage_09/DATA_AUDIT.json"
    audit_receipt = json.loads(audit_receipt_path.read_text())
    if audit_receipt.get("status") != "PASS_FROZEN_PILOT":
        raise ValueError("Real CPU audit has not passed")
    freeze = {
        "stage": "09", "status": "FROZEN_BEFORE_GPU_TRAINING",
        "runner_sha256": runner.file_sha(runner_path),
        "audit_script_sha256": runner.file_sha(audit_path),
        "data_audit_receipt_sha256": runner.file_sha(audit_receipt_path),
        "data_admission_fingerprint": audit_receipt["admission_fingerprint"],
        "resource_manifest_sha256": RESOURCE_MANIFEST_SHA,
        "resource_source_script_version_id": 359317729,
        "resource_source_url": "https://www.kaggle.com/code/watchman777/name-that-llm-stage-04b-resources?scriptVersionId=359317729",
        "protocol": runner.PROTOCOL,
        "training_started": False,
        "source_freeze_note": "Commit this source and receipt before any GPU fitting. Notebook and runner refuse source/input changes. Build time is not a training result.",
    }
    freeze_path = repo / "results/stage_09/SOURCE_FREEZE.json"
    runner.save_json(freeze_path, freeze)
    tests = runner.self_test()
    runner.save_json(repo / "results/stage_09/RUNNER_SYNTHETIC_VALIDATION.json", tests)
    code = {"stage09_data_audit.py": audit_path.read_text(), "stage09_pretraining_pilot.py": runner_path.read_text()}
    code_hashes = {name: runner.canonical_sha(value) for name, value in code.items()}
    setup = '''# The budget clock starts before dependency setup, not after model loading.
from pathlib import Path
import hashlib, json, os, subprocess, sys, time
stage09_allocation_started = time.perf_counter()
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", USE_TF="0", USE_FLAX="0",
                  TOKENIZERS_PARALLELISM="false", CUBLAS_WORKSPACE_CONFIG=":4096:8")
def unique_path(pattern):
    matches = list(Path("/kaggle/input").rglob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one frozen attachment asset {pattern}; found {len(matches)}")
    return matches[0]
resource_manifest_path = unique_path("resources_manifest.json")
if hashlib.sha256(resource_manifest_path.read_bytes()).hexdigest() != RESOURCE_MANIFEST_SHA:
    raise RuntimeError("Attach the exact private Stage04B Resources V1 (359317729)")
resource_dir = resource_manifest_path.parent
resource_manifest = json.loads(resource_manifest_path.read_text())
if sys.version_info[:2] != (3, 13):
    raise RuntimeError("Pinned offline sentencepiece wheel requires Python 3.13; do not silently substitute")
wheel_paths = []
for name, expected in resource_manifest["wheel_files_sha256"].items():
    paths = list(resource_dir.rglob(name))
    if len(paths) != 1 or hashlib.sha256(paths[0].read_bytes()).hexdigest() != expected:
        raise RuntimeError("Pinned offline dependency asset mismatch: " + name)
    wheel_paths.append(str(paths[0]))
subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", *wheel_paths], check=True)
train_path = unique_path("train.csv")
source_dir = unique_path("passages_clean.csv").parent
print("Pinned offline dependencies and unique required input paths resolved; no test file opened.")
'''
    materialize = '''# Embedded source is checksum-bound to the prospective receipt.
import sys
runtime_dir = Path("/kaggle/working/stage09_frozen_code")
runtime_dir.mkdir(exist_ok=True)
for filename, source in CODE_SOURCES.items():
    target = runtime_dir / filename
    target.write_text(source)
    expected = FREEZE["audit_script_sha256" if filename == "stage09_data_audit.py" else "runner_sha256"]
    if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
        raise RuntimeError("Embedded frozen source mismatch: " + filename)
freeze_path = runtime_dir / "SOURCE_FREEZE.json"
freeze_path.write_text(json.dumps(FREEZE, indent=2, sort_keys=True, allow_nan=False) + "\\n")
sys.path.insert(0, str(runtime_dir))
import stage09_pretraining_pilot as pilot
if pilot.PROTOCOL != FREEZE["protocol"]:
    raise RuntimeError("Embedded protocol mismatch")
print(json.dumps(pilot.self_test(), indent=2))
'''
    training = f'''RUN_TRAINING = {bool(enable_training)!r}
# The notebook must remain PRIVATE: checkpoints and original-row probabilities are private outputs.
if RUN_TRAINING:
    result = pilot.run(train_path, source_dir, resource_dir,
                       Path("/kaggle/working/stage09_pilot"), freeze_path,
                       resume=False, allocation_started=stage09_allocation_started)
    print("Fixed-fold development comparison completed; inspect aggregate/RESULT.json.")
else:
    print("PREPARED_ONLY: GPU training has not been enabled or executed.")
'''
    source_payload = "CODE_SOURCES = " + repr(code) + "\nFREEZE = " + repr(freeze) + "\n"
    setup = "RESOURCE_MANIFEST_SHA = " + repr(RESOURCE_MANIFEST_SHA) + "\n" + setup
    notebook = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_markdown_cell("""# Stage 09 — narrow public-data pretraining pilot

## tl;dr

**Prepared; GPU results are not yet available.** This notebook compares an official-only control with two epochs of admitted external authorship pretraining followed by the identical official training schedule. Both start from the same untouched pinned DeBERTa-v3-base vendor checkpoint; the single classification head is reset identically before both official phases.

The official S03 result of 62.0% public balanced accuracy is a separate result. This pilot uses a smaller encoder, 750 external examples from five GPT labels and one reused development fold. It is not the upstream 83k training reproduction and cannot establish an ensemble gain.

Keep this notebook and its outputs **private**. Only the `aggregate/` folder is suitable for later public review after inspection; `private/` contains original-row predictions and fitted checkpoints.
"""),
        nbformat.v4.new_markdown_cell("""## Context & Methods

The question is whether the staged procedure improves fixed-fold development balanced accuracy. Extra pretraining compute is part of the procedure, so this is not a compute-matched estimate of the value of external examples alone.

### Key Assumptions

- The corpus is frozen V4 `new`/direct-OpenAI only: 750 responses in 150 complete five-label prompt blocks. Every external epoch uses each row once, shuffled; each label has exactly 150 examples.
- Official fold 1 has 2,159 training rows and 540 validation rows, all 18 labels; whole original prompts remain disjoint. Admission is conservative, but original prompt wording is unavailable, so semantic independence is not certified.
- Architecture: attention-mask mean of the untouched encoder's last hidden state plus a single linear 18-label head. Seed 20261024, sequence length 384, batch 4, accumulation 4.
- External phase: 2 epochs, encoder LR 1.5e-5/head LR 1e-3. Official phases: 3 epochs, encoder LR 1e-5/head LR 5e-4; fresh identical head and official optimizer/scheduler. Final epoch is fixed, with no validation-based selection.
- Two allocated T4 GPUs; training uses CUDA device 0. Both allocated devices count toward a cumulative three GPU-hour cap, including setup. No paid service is used.

### 1. Verify pinned offline dependencies

Attach the official competition, prvsiyan's Extended Name That LLM Dataset **V4**, and our private Stage04B Resources **V1 359317729**. Use Python 3.13 and keep Internet off. No model or package is downloaded by this notebook.
"""),
        nbformat.v4.new_code_cell(setup),
        nbformat.v4.new_markdown_cell("""## Data

### 2. Materialize the frozen audit and runner

The embedded source is deliberately explicit and portable. The CPU audit verifies immutable input hashes, public prompt/call associations, the exact 750-row admission, whole-prompt overlap screening and frozen official membership before fitting. The vendor resource manifest and every model asset are verified again.

The checks below validate accumulation arithmetic, class exposure, prediction alignment and known bootstrap contrasts on synthetic data. They do not execute Torch, AMP or a model.
"""),
        nbformat.v4.new_code_cell(source_payload + materialize),
        nbformat.v4.new_markdown_cell("""## Results

### 3. Execute the predeclared comparison

This is the only fitting cell. The runner creates completed phase-boundary checkpoints; it does not resume partial optimizer states or silently shorten a phase. A budget, AMP, numerical or contract failure leaves an explicit incomplete status.

Each arm is evaluated only after its fixed third official epoch. `aggregate/RESULT.json` retains development balanced accuracy, every class's recall and regression, confusion matrices, paired corrected/broken counts and the conditional paired prompt-bootstrap interval. Positive BA difference and a positive interval lower endpoint only earn broader grouped evaluation.

Original-row probabilities, logits, IDs and model checkpoints stay under `private/`. There is no test loading, submission or final-selection action.
"""),
        nbformat.v4.new_code_cell(training),
        nbformat.v4.new_markdown_cell("""## Takeaways

No GPU score is populated in this prepared notebook. A completed fixed-fold result remains exploratory development evidence because these prompt groups have been reused. Public S03 accuracy, private leaderboard performance and an ensemble improvement are separate claims requiring their own evidence.

Source and data provenance: [Nishant Dahal part two V1](https://www.kaggle.com/code/nishantdahal/two-stage-lora-part-two?scriptVersionId=361311282) motivated the staged mean-pooling encoder architecture/rates; [prvsiyan's Extended Dataset V4](https://www.kaggle.com/datasets/prvsiyan/prvsiyans-extended-name-that-llm-dataset/versions/4), source-declared CC BY 4.0, supplies the narrow externally admitted corpus; [our pinned vendor resource preparation](https://www.kaggle.com/code/watchman777/name-that-llm-stage-04b-resources?scriptVersionId=359317729) supplies the untouched MIT-licensed DeBERTa-v3-base revision. No upstream adapted checkpoint initializes these development arms.
"""),
    ], metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python", "version": "3.13"},
                 "stage09": {"prepared_only": True, "run_training_enabled": bool(enable_training),
                             "runner_sha256": freeze["runner_sha256"], "audit_sha256": freeze["audit_script_sha256"],
                             "source_payload_sha256": code_hashes}})
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            compile(cell.source, "Stage_09_Pretraining_Pilot.ipynb", "exec")
    target = repo / "notebooks/Stage_09_Pretraining_Pilot.ipynb"
    target.parent.mkdir(exist_ok=True)
    nbformat.write(notebook, target)
    # Readback validation catches accidental serialization/embedding changes.
    reopened = nbformat.read(target, as_version=4)
    nbformat.validate(reopened)
    return {"notebook": str(target), "source_freeze": str(freeze_path),
            "runner_sha256": freeze["runner_sha256"], "audit_script_sha256": freeze["audit_script_sha256"],
            "notebook_sha256": runner.file_sha(target), "training_enabled": bool(enable_training),
            "actual_gpu_training_executed": False, "synthetic_tests": tests["passed"],
            "validation": "nbformat schema/readback and all code-cell compilation pass; GPU notebook execution and rendered UI inspection pending"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--enable-training", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(args.repo, enable_training=args.enable_training), indent=2))
