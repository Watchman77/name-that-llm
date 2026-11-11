"""Package the frozen Stage 13B stress runner as an inert Kaggle notebook."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


RUNNER_PATH = "scripts/stage13b_balancing_stress.py"
NOTEBOOK_PATH = "notebooks/Stage_13B_Balancing_Stress.ipynb"
SOURCE_VERSION = 368292220
CACHE_SHA256 = "565df6f8e652a9850ff11cf4e8b6a81fcccbc7db593186ff7f53358e7190bd16"
SCOPE = "glm_only_paired_fold_local_balancing_stress"
EXPECTED_ROWS = {"natural_full": 2699, "matched_balanced": 1800, "skew_a": 1800, "skew_b": 1800}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def build(root):
    root = Path(root)
    runner_bytes = (root / RUNNER_PATH).read_bytes()
    runner = runner_bytes.decode("utf-8")
    runner_sha = sha256(runner_bytes)
    compile(runner, "<stage13b-embedded-runner>", "exec")

    cells = []

    def markdown(cell_id, source):
        cells.append(dict(cell_type="markdown", id=cell_id, metadata={}, source=source.splitlines(keepends=True)))

    def code(cell_id, source):
        compile(source, "<stage13b-notebook-cell>", "exec")
        cells.append(dict(cell_type="code", id=cell_id, metadata={}, execution_count=None,
                          outputs=[], source=source.splitlines(keepends=True)))

    markdown("scope", "# Stage 13B — GLM balancing stress screen\n\n"
             "Compare the frozen uniform Sinkhorn rule with no balancing on the same rows within each fold. "
             "All probabilities come from the existing prompt-held-out GLM cache. These are development "
             "stress scenarios; they do not estimate hidden-test performance or change S07.\n")
    markdown("input", "## Private CPU input\n\n"
             "Bind only the private Stage 13 GLM readout V3 output (kernel version **368292220**). "
             "Keep the notebook private, with accelerator None and Internet off. The runner verifies the "
             "frozen cache before scoring. Row-level predictions and identifiers stay inside the private input.\n")
    code("verified-source", (
        "from pathlib import Path\nimport hashlib\nimport json\nimport math\nimport subprocess\nimport sys\n\n"
        + "_runner_source = " + repr(runner) + "\n"
        + "_runner_sha256 = " + repr(runner_sha) + "\n"
        + "_source_dir = Path('/kaggle/working/_stage13b_balancing_stress_source')\n"
        + "_runner_path = _source_dir / 'stage13b_balancing_stress.py'\n"
        + "_runner_bytes = _runner_source.encode('utf-8')\n"
        + "if hashlib.sha256(_runner_bytes).hexdigest() != _runner_sha256:\n"
        + "    raise RuntimeError('Stage13B embedded runner hash mismatch')\n"
        + "_source_dir.mkdir(parents=True, exist_ok=True)\n"
        + "_runner_path.write_bytes(_runner_bytes)\n"
        + "if hashlib.sha256(_runner_path.read_bytes()).hexdigest() != _runner_sha256:\n"
        + "    raise RuntimeError('Stage13B written runner hash mismatch')\n"
        + "print('STAGE13B_SOURCE_CHECKS_PASS', _runner_sha256)\n"
    ))
    markdown("execute", "## Run the frozen stress comparison\n\n"
             "Scoring stops if any cache, fold, arithmetic, or scenario check fails. No model is fitted.\n")
    code("run", "subprocess.run([sys.executable, str(_runner_path), '--input-root', '/kaggle/input', "
         "'--output-root', '/kaggle/working/stage13b_balancing_stress'], check=True, timeout=180)\n")
    markdown("readback", "## Verify the saved aggregate report\n\n"
             "Check the report scope, all four scenario row counts, and five batches per scenario. "
             "Display only the aggregate differences and the proxy-screen decision.\n")
    code("saved-result", (
        "_result_dir = Path('/kaggle/working/stage13b_balancing_stress')\n"
        "_report = json.loads((_result_dir / 'balancing_stress_report.json').read_text(encoding='utf-8'))\n"
        "def _require(condition, code):\n"
        "    if not condition:\n"
        "        raise RuntimeError('Stage13B saved report check failed: ' + code)\n"
        "_require(type(_report) is dict, 'report_type')\n"
        "_require(_report.get('status') == 'COMPLETE_GLM_SINKHORN_STRESS_ONLY', 'status')\n"
        "_require(_report.get('scope') == 'glm_only_paired_fold_local_balancing_stress', 'scope')\n"
        + f"_require(_report.get('source_version_id') == {SOURCE_VERSION}, 'source_version_id')\n"
        + f"_require(_report.get('heldout_cache_sha256') == {CACHE_SHA256!r}, 'cache_sha256')\n"
        + "_require(_report.get('probabilities_unchanged') is True, 'probabilities_unchanged')\n"
        + "for _key in ['fit_performed', 'official_score_estimated', 'full_s07_oof_admitted', "
        "'raw_rows_or_predictions_exported']:\n"
        "    _require(_report.get(_key) is False, _key)\n"
        "_require(type(_report.get('passes_proxy_screen')) is bool, 'passes_proxy_screen')\n"
        + "_expected_rows = " + repr(EXPECTED_ROWS) + "\n"
        + "_scenarios = _report.get('scenarios')\n"
        + "_require(type(_scenarios) is dict and set(_scenarios) == set(_expected_rows), 'scenario_names')\n"
        + "_delta_pp = {}\n"
        + "for _name, _rows in _expected_rows.items():\n"
        + "    _scenario = _scenarios[_name]\n"
        + "    _require(type(_scenario) is dict and type(_scenario.get('rows')) is int and "
          "_scenario['rows'] == _rows, 'scenario_rows_' + _name)\n"
        + "    _batches = _scenario.get('batches')\n"
        + "    _require(type(_batches) is list and len(_batches) == 5, 'batch_count_' + _name)\n"
        + "    _delta = _scenario.get('delta_pp')\n"
        + "    _require(type(_delta) in (int, float) and math.isfinite(_delta) and -100 <= _delta <= 100, "
          "'delta_pp_' + _name)\n"
        + "    _delta_pp[_name] = _delta\n"
        + "print('STAGE13B_SAVED_REPORT_READBACK_PASS', json.dumps({"
          "'scenario_delta_pp': _delta_pp, 'passes_proxy_screen': _report['passes_proxy_screen']}, sort_keys=True))\n"
    ))

    nb = dict(nbformat=4, nbformat_minor=5, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "kaggle": {"accelerator": "none", "dataSources": [{"sourceType": "kernelVersion", "sourceId": SOURCE_VERSION}],
                   "isInternetEnabled": False, "language": "python", "sourceType": "notebook", "isGpuEnabled": False},
        "stage13b_balancing_stress": {"scope": SCOPE, "runner_sha256": runner_sha,
                                      "heldout_cache_sha256": CACHE_SHA256, "evaluation_executed_by_build": False},
    }, cells=cells)
    encoded = (json.dumps(nb, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    if json.loads(encoded) != nb or len({cell["id"] for cell in cells}) != len(cells):
        raise ValueError("Stage13B notebook serialization check failed")
    code_cells = sum(cell["cell_type"] == "code" for cell in cells)
    if code_cells != 3:
        raise ValueError("Stage13B notebook must contain exactly three code cells")
    target = root / NOTEBOOK_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(encoded)
    return {"notebook_path": NOTEBOOK_PATH, "notebook_sha256": sha256(encoded), "runner_sha256": runner_sha,
            "source_kernel_version": SOURCE_VERSION, "code_cells": code_cells, "total_cells": len(cells),
            "evaluation_executed_by_build": False}


if __name__ == "__main__":
    print(json.dumps(build(Path(__file__).resolve().parents[1]), sort_keys=True))
