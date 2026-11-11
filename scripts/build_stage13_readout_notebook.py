"""Package the reviewed Stage 13 readout as an inert Kaggle notebook."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ADMISSION_SHA256 = "4b30488d126bc6213d8e1d128d50d7f0641b33e90b13192af55abe413ae49629"
NOTEBOOK_PATH = "notebooks/Stage_13_GLM_Readout.ipynb"
INPUTS = [
    {"sourceType": "competition", "sourceId": 170484},
    {"sourceType": "kernelVersion", "sourceId": 359405443},
]


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build(root):
    root = Path(root)
    admission = (root / "scripts/stage13a_cache_admission.py").read_bytes().decode("utf-8")
    runner = (root / "scripts/stage13_glm_readout.py").read_bytes().decode("utf-8")
    admission_sha, runner_sha = digest(admission), digest(runner)
    if admission_sha != ADMISSION_SHA256:
        raise ValueError("Stage13A admission source differs from the previously verified freeze")
    compile(admission, "<embedded-admission>", "exec")
    compile(runner, "<embedded-readout>", "exec")

    cells = []

    def markdown(cell_id, source):
        cells.append(dict(cell_type="markdown", id=cell_id, metadata={}, source=source.splitlines(keepends=True)))

    def code(cell_id, source):
        compile(source, "<notebook-cell>", "exec")
        cells.append(dict(cell_type="code", id=cell_id, metadata={}, execution_count=None,
                          outputs=[], source=source.splitlines(keepends=True)))

    markdown("scope", "# Stage 13 — frozen GLM prompt-held-out classifier\n\n"
             "Fit one fixed CPU classifier recipe on five whole-prompt training folds, then collect "
             "held-out probabilities for the corresponding validation folds. The canonical folds "
             "were used in prior experiments, so this is exploratory development evidence. "
             "The job does not balance predictions, reconstruct S07 or make a competition submission.\n")
    markdown("inputs", "## Private CPU run inputs\n\n"
             "| Input | Required version |\n|---|---:|\n"
             "| Name That LLM competition | 170484 |\n"
             "| Own Stage 06C feature-cache notebook output | 359405443 |\n\n"
             "Verify the two exact bindings, accelerator None, Internet off and private visibility "
             "in the saved Kaggle editor before running. The first step of the runner repeats "
             "the frozen Stage 13A input and fold checks.\n")
    code("verified-sources", (
        "from pathlib import Path\nimport hashlib\nimport json\nimport math\nimport subprocess\nimport sys\n\n"
        + "_admission_source = " + repr(admission) + "\n"
        + "_runner_source = " + repr(runner) + "\n"
        + "_admission_sha256 = " + repr(admission_sha) + "\n"
        + "_runner_sha256 = " + repr(runner_sha) + "\n"
        + "_source_dir = Path('/kaggle/working/_stage13_glm_readout_source')\n"
        + "_runner_path = _source_dir / 'stage13_glm_readout.py'\n"
        + "_admission_path = _source_dir / 'stage13a_cache_admission.py'\n"
        + "for _source, _expected in [(_admission_source, _admission_sha256), (_runner_source, _runner_sha256)]:\n"
        + "    if hashlib.sha256(_source.encode('utf-8')).hexdigest() != _expected:\n"
        + "        raise RuntimeError('Stage13 embedded source hash mismatch')\n"
        + "_source_dir.mkdir(parents=True, exist_ok=True)\n"
        + "for _path, _source, _expected in [(_admission_path, _admission_source, _admission_sha256), "
          "(_runner_path, _runner_source, _runner_sha256)]:\n"
        + "    _path.write_bytes(_source.encode('utf-8'))\n"
        + "    if hashlib.sha256(_path.read_bytes()).hexdigest() != _expected:\n"
        + "        raise RuntimeError('Stage13 written source hash mismatch')\n"
        + "print('STAGE13_GLM_READOUT_SOURCE_CHECKS_PASS')\n"
    ))
    markdown("execute", "## Fit and evaluate the frozen recipe\n\n"
             "A failed input, fold or model check stops the run. The row-level cache must remain in "
             "private Kaggle output. Logs contain only aggregate evidence and digests.\n")
    code("train", "subprocess.run([sys.executable, str(_runner_path), '--input-root', '/kaggle/input', "
         "'--output-root', '/kaggle/working/stage13_glm_readout'], check=True, timeout=1800)\n")
    markdown("readback", "## Check the saved result\n\n"
             "Verify the aggregate report, fold scope and private cache digest before accepting this "
             "development reference. No cache rows are displayed.\n")
    code("saved-result", (
        "_result_dir = Path('/kaggle/working/stage13_glm_readout')\n"
        "_report = json.loads((_result_dir / 'readout_report.json').read_text(encoding='utf-8'))\n"
        "def _require(condition, code):\n"
        "    if not condition:\n"
        "        raise RuntimeError('Stage13 saved report check failed: ' + code)\n"
        "_require(_report.get('status') == 'COMPLETE_GLM_PLAIN_GROUPED_DEVELOPMENT_REFERENCE', 'status')\n"
        "_require(_report.get('scope') == 'glm_only_prompt_heldout_development', 'scope')\n"
        "_require((_report.get('rows'), _report.get('prompts'), _report.get('admission_checks_passed')) "
        "== (2699, 150, 7), 'admission_axes')\n"
        "_require(_report.get('probability_shape') == [2699, 18], 'probability_shape')\n"
        "for _key in ['fit_performed', 'heldout_predictions_produced', 'heldout_cache_readback_passed']:\n"
        "    _require(_report.get(_key) is True, _key)\n"
        "for _key in ['balancing_performed', 'official_score_estimated', 'full_s07_oof_admitted', "
        "'stage13b_batch_coupled_comparison_admitted']:\n"
        "    _require(_report.get(_key) is False, _key)\n"
        "_folds = _report.get('folds')\n"
        "_require(isinstance(_folds, list) and len(_folds) == 5, 'fold_count')\n"
        "for _number, _fold in enumerate(_folds, 1):\n"
        "    _valid_rows = 540 if _number <= 4 else 539\n"
        "    _require(isinstance(_fold, dict) and _fold.get('fold') == _number, 'fold_order')\n"
        "    _require((_fold.get('fit_rows'), _fold.get('valid_rows'), _fold.get('fit_prompts'), "
        "_fold.get('valid_prompts'), _fold.get('prompt_overlap')) == "
        "(2699 - _valid_rows, _valid_rows, 120, 30, 0), 'fold_partition')\n"
        "_score = _report.get('pooled_balanced_accuracy')\n"
        "_require(type(_score) in (float, int) and math.isfinite(_score) and 0 <= _score <= 1, 'development_score')\n"
        "with (_result_dir / 'heldout_probabilities.npz').open('rb') as _stream:\n"
        "    _actual_cache_sha256 = hashlib.file_digest(_stream, 'sha256').hexdigest()\n"
        "_require(_actual_cache_sha256 == _report.get('heldout_cache_sha256'), 'cache_hash')\n"
        "print('STAGE13_GLM_READOUT_SAVED_REPORT_READBACK_PASS', "
        "json.dumps({'rows': _report['rows'], 'folds': len(_folds), 'development_balanced_accuracy': _score, "
        "'heldout_cache_sha256': _actual_cache_sha256}, sort_keys=True))\n"
    ))

    nb = dict(nbformat=4, nbformat_minor=5, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "kaggle": {"accelerator": "none", "dataSources": INPUTS, "isInternetEnabled": False,
                   "language": "python", "sourceType": "notebook", "isGpuEnabled": False},
        "stage13_glm_readout": {"scope": "glm_only_prompt_heldout_development",
                                "admission_source_sha256": admission_sha, "runner_sha256": runner_sha,
                                "training_executed_by_build": False},
    }, cells=cells)
    encoded = json.dumps(nb, indent=2, ensure_ascii=False) + "\n"
    reread = json.loads(encoded)
    assert reread == nb and len({cell["id"] for cell in cells}) == len(cells)
    assert sum(cell["cell_type"] == "code" for cell in cells) == 3
    target = root / NOTEBOOK_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(encoded.encode("utf-8"))
    return {"notebook_path": NOTEBOOK_PATH, "notebook_sha256": digest(encoded), "runner_sha256": runner_sha,
            "admission_source_sha256": admission_sha, "code_cells": 3, "total_cells": len(cells),
            "training_executed_by_build": False}


if __name__ == "__main__":
    print(json.dumps(build(Path(__file__).resolve().parents[1]), sort_keys=True))
