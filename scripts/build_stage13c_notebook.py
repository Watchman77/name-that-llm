"""Package the frozen Stage 13C complement runner as an inert Kaggle notebook."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


RUNNER_PATH = "scripts/stage13c_deberta_complement.py"
DEPENDENCY_PATH = "scripts/stage13b_balancing_stress.py"
DEPENDENCY_SHA256 = "f4981f385074f60ee034b979d4aa6672c7d28eaa408c374d5caf55a7c446fbd9"
NOTEBOOK_PATH = "notebooks/Stage_13C_DeBERTa_Complement.ipynb"
SOURCE_VERSIONS = [368292220, 358745836]
HELDOUT_CACHE_SHA256 = "565df6f8e652a9850ff11cf4e8b6a81fcccbc7db593186ff7f53358e7190bd16"
DEBERTA_CACHE_SHA256 = "11eed58762ad128f97457dde9cc4ab03c7932e1b4b04845ec6664c834b1b1dda"
CONTROL_BA = 0.4608252547849864
SCOPE = "glm_deberta_paired_fold_local_development"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def build(root):
    root = Path(root)
    sources = {}
    for relative_path in (RUNNER_PATH, DEPENDENCY_PATH):
        source_bytes = (root / relative_path).read_bytes()
        source = source_bytes.decode("utf-8")
        name = Path(relative_path).name
        compile(source, f"<stage13c-embedded:{name}>", "exec")
        sources[name] = (source, sha256(source_bytes))
    if sources[Path(DEPENDENCY_PATH).name][1] != DEPENDENCY_SHA256:
        raise ValueError("Stage13B module differs from the verified freeze")
    runner_sha = sources[Path(RUNNER_PATH).name][1]

    cells = []

    def markdown(cell_id, source):
        cells.append(dict(cell_type="markdown", id=cell_id, metadata={}, source=source.splitlines(keepends=True)))

    def code(cell_id, source):
        compile(source, "<stage13c-notebook-cell>", "exec")
        cells.append(dict(cell_type="code", id=cell_id, metadata={}, execution_count=None,
                          outputs=[], source=source.splitlines(keepends=True)))

    markdown("scope", "# Stage 13C — frozen GLM and DeBERTa complement screen\n\n"
             "Replay the fixed GLM control and compare it with a fixed DeBERTa complement on the same "
             "five prompt-held-out folds. This is exploratory development evidence, not an official "
             "Kaggle score or an out-of-fold replay of the full S07 ensemble.\n")
    markdown("inputs", "## Private CPU inputs\n\n"
             "| Input | Required kernel version |\n|---|---:|\n"
             "| Private Stage 13 GLM readout V3 | 368292220 |\n"
             "| Frozen pretrained DeBERTa feature output | 358745836 |\n\n"
             "Bind exactly these versions. Keep the notebook private, accelerator None and Internet off. "
             "The runner checks both cache digests and alignment before scoring. No row-level data is exported.\n")
    code("verified-sources", (
        "from pathlib import Path\nimport hashlib\nimport json\nimport math\nimport subprocess\nimport sys\n\n"
        + "_sources = " + repr(sources) + "\n"
        + "_source_dir = Path('/kaggle/working/_stage13c_deberta_complement_source')\n"
        + "_source_dir.mkdir(parents=True, exist_ok=True)\n"
        + "for _name, (_source, _expected) in _sources.items():\n"
        + "    _source_bytes = _source.encode('utf-8')\n"
        + "    if hashlib.sha256(_source_bytes).hexdigest() != _expected:\n"
        + "        raise RuntimeError('Stage13C embedded source hash mismatch')\n"
        + "    _path = _source_dir / _name\n"
        + "    _path.write_bytes(_source_bytes)\n"
        + "    if hashlib.sha256(_path.read_bytes()).hexdigest() != _expected:\n"
        + "        raise RuntimeError('Stage13C written source hash mismatch')\n"
        + "_runner_path = _source_dir / 'stage13c_deberta_complement.py'\n"
        + "print('STAGE13C_SOURCE_CHECKS_PASS', _sources[_runner_path.name][1])\n"
    ))
    markdown("execute", "## Run the frozen comparison\n\n"
             "The job stops if an input, fold, control replay, or numerical check fails. "
             "The runner saves aggregate evidence only.\n")
    code("run", "subprocess.run([sys.executable, '-B', str(_runner_path), '--input-root', '/kaggle/input', "
         "'--output-root', '/kaggle/working/stage13c_deberta_complement'], check=True, timeout=600)\n")
    markdown("readback", "## Verify the saved aggregate report\n\n"
             "Check the report scope, inputs, control replay and score arithmetic. "
             "Display only the aggregate scores and the proxy-screen decision.\n")
    code("saved-result", f"""_result_dir = Path('/kaggle/working/stage13c_deberta_complement')
_report = json.loads((_result_dir / 'complement_report.json').read_text(encoding='utf-8'))
def _require(condition, code):
    if not condition:
        raise RuntimeError('Stage13C saved report check failed: ' + code)
_require(type(_report) is dict, 'report_type')
_require(_report.get('status') == 'COMPLETE_GLM_DEBERTA_COMPLEMENT_SCREEN', 'status')
_require(_report.get('scope') == {SCOPE!r}, 'scope')
_require(type(_report.get('rows')) is int and _report['rows'] == 2699, 'rows')
_require(type(_report.get('prompts')) is int and _report['prompts'] == 150, 'prompts')
_classes = _report.get('classes')
_require(type(_classes) is list and len(_classes) == 18 and
         all(type(_label) is str and _label for _label in _classes) and len(set(_classes)) == 18, 'classes')
_folds = _report.get('folds')
_require(type(_folds) is list and len(_folds) == 5 and all(type(_fold) is dict for _fold in _folds), 'folds')
_require(_report.get('expected_input_version_ids') == {SOURCE_VERSIONS!r}, 'expected_input_version_ids')
_require(_report.get('heldout_cache_sha256') == {HELDOUT_CACHE_SHA256!r}, 'heldout_cache_sha256')
_require(_report.get('deberta_cache_sha256') == {DEBERTA_CACHE_SHA256!r}, 'deberta_cache_sha256')
for _flag in ('fit_performed', 'control_replayed'):
    _require(_report.get(_flag) is True, _flag)
for _flag in ('official_score_estimated', 'full_s07_oof_admitted', 'raw_rows_or_predictions_exported'):
    _require(_report.get(_flag) is False, _flag)
_require(type(_report.get('passes_proxy_screen')) is bool, 'passes_proxy_screen')
_scores = {{}}
for _key in ('control_ba', 'candidate_ba', 'deberta_alone_ba'):
    _score = _report.get(_key)
    _require(type(_score) in (int, float) and 0 <= _score <= 1 and math.isfinite(_score), _key)
    _scores[_key] = _score
_require(math.isclose(_scores['control_ba'], {CONTROL_BA!r}, abs_tol=1e-12, rel_tol=0), 'control_replay')
_delta = _report.get('delta_pp')
_require(type(_delta) in (int, float) and -100 <= _delta <= 100 and math.isfinite(_delta), 'delta_pp')
_require(math.isclose(_delta, (_scores['candidate_ba'] - _scores['control_ba']) * 100,
                      abs_tol=1e-10, rel_tol=0), 'delta_arithmetic')
_ci = _report.get('conditional_delta_ci95_pp')
_require(type(_ci) is list and len(_ci) == 2 and
         all(type(_bound) in (int, float) and -100 <= _bound <= 100 and math.isfinite(_bound) for _bound in _ci)
         and _ci[0] <= _ci[1], 'conditional_delta_ci95_pp')
_scores.update(delta_pp=_delta, conditional_delta_ci95_pp=_ci, passes_proxy_screen=_report['passes_proxy_screen'])
print('STAGE13C_SAVED_REPORT_READBACK_PASS', json.dumps(_scores, sort_keys=True, allow_nan=False))
""")

    nb = dict(nbformat=4, nbformat_minor=5, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "kaggle": {"accelerator": "none",
                   "dataSources": [{"sourceType": "kernelVersion", "sourceId": version} for version in SOURCE_VERSIONS],
                   "isInternetEnabled": False, "language": "python", "sourceType": "notebook", "isGpuEnabled": False},
        "stage13c_deberta_complement": {"scope": SCOPE, "runner_sha256": runner_sha,
                                        "dependency_sha256": DEPENDENCY_SHA256,
                                        "heldout_cache_sha256": HELDOUT_CACHE_SHA256,
                                        "deberta_cache_sha256": DEBERTA_CACHE_SHA256,
                                        "evaluation_executed_by_build": False},
    }, cells=cells)
    encoded = (json.dumps(nb, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    if json.loads(encoded) != nb or len({cell["id"] for cell in cells}) != len(cells):
        raise ValueError("Stage13C notebook serialization check failed")
    code_cells = sum(cell["cell_type"] == "code" for cell in cells)
    if code_cells != 3:
        raise ValueError("Stage13C notebook must contain exactly three code cells")
    target = root / NOTEBOOK_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(encoded)
    return {"notebook_path": NOTEBOOK_PATH, "notebook_sha256": sha256(encoded), "runner_sha256": runner_sha,
            "dependency_sha256": DEPENDENCY_SHA256, "source_kernel_versions": SOURCE_VERSIONS,
            "code_cells": code_cells, "total_cells": len(cells), "evaluation_executed_by_build": False}


if __name__ == "__main__":
    print(json.dumps(build(Path(__file__).resolve().parents[1]), sort_keys=True))
