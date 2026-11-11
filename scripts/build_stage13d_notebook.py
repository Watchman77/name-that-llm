"""Package the frozen Stage 13D broader complement runner as an inert Kaggle notebook."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


RUNNER_PATH = "scripts/stage13d_broader_complement.py"
DEPENDENCIES = {
    "scripts/stage13c_deberta_complement.py": "b9b6f523b8a6b1f9f1dae556c863f7fee34f56fddfb27adb0946021e1f17181e",
    "scripts/stage13b_balancing_stress.py": "f4981f385074f60ee034b979d4aa6672c7d28eaa408c374d5caf55a7c446fbd9",
}
NOTEBOOK_PATH = "notebooks/Stage_13D_Broader_Complement.ipynb"
SOURCE_VERSIONS = [368292220, 358745836]
COMPETITION_ID = 170484
TRAIN_CSV_SHA256 = "40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707"
HELDOUT_CACHE_SHA256 = "565df6f8e652a9850ff11cf4e8b6a81fcccbc7db593186ff7f53358e7190bd16"
DEBERTA_CACHE_SHA256 = "11eed58762ad128f97457dde9cc4ab03c7932e1b4b04845ec6664c834b1b1dda"
CONTROL_BA = 0.4608252547849864
STAGE13C_CONFUSION_SHA256 = "c076ee1a507753660bd7356baa3083d277c8fc8e61bedc81bae9192986caad9b"
BLEND_WEIGHTS = {"broader_control": {"glm": 0.75, "char": 0.25},
                 "candidate": {"glm": 0.50, "char": 0.25, "deberta": 0.25}}
SCOPE = "glm_char_deberta_paired_fold_local_development"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def build(root):
    root = Path(root)
    sources = {}
    for relative_path in (RUNNER_PATH, *DEPENDENCIES):
        source_bytes = (root / relative_path).read_bytes()
        source = source_bytes.decode("utf-8")
        name = Path(relative_path).name
        compile(source, f"<stage13d-embedded:{name}>", "exec")
        sources[name] = (source, sha256(source_bytes))
    for path, expected in DEPENDENCIES.items():
        if sources[Path(path).name][1] != expected:
            raise ValueError(f"{path} differs from the verified freeze")
    runner_sha = sources[Path(RUNNER_PATH).name][1]

    cells = []

    def markdown(cell_id, source):
        cells.append(dict(cell_type="markdown", id=cell_id, metadata={}, source=source.splitlines(keepends=True)))

    def code(cell_id, source):
        compile(source, "<stage13d-notebook-cell>", "exec")
        cells.append(dict(cell_type="code", id=cell_id, metadata={}, execution_count=None,
                          outputs=[], source=source.splitlines(keepends=True)))

    markdown("scope", "# Stage 13D — frozen broader complement screen\n\n"
             "Replay the fixed GLM baseline, then test the character-feature control and the DeBERTa complement "
             "on the same five prompt-held-out folds. This is exploratory development evidence, not an official "
             "Kaggle score or an out-of-fold replay of the full S07 ensemble.\n")
    markdown("inputs", "## Private CPU inputs\n\n"
             "| Input | Required ID |\n|---|---:|\n"
             "| Name That LLM competition | 170484 |\n"
             "| Private Stage 13 GLM readout V3 | 368292220 |\n"
             "| Frozen pretrained DeBERTa feature output | 358745836 |\n\n"
             "Bind exactly these versions. Keep the notebook private, accelerator None and Internet off. "
             "The runner checks exact input digests and alignment before scoring. No row-level data is exported.\n")
    code("verified-sources", (
        "from pathlib import Path\nimport hashlib\nimport json\nimport math\nimport subprocess\nimport sys\n\n"
        + "_sources = " + repr(sources) + "\n"
        + "_source_dir = Path('/kaggle/working/_stage13d_broader_complement_source')\n"
        + "_source_dir.mkdir(parents=True, exist_ok=True)\n"
        + "for _name, (_source, _expected) in _sources.items():\n"
        + "    _source_bytes = _source.encode('utf-8')\n"
        + "    if hashlib.sha256(_source_bytes).hexdigest() != _expected:\n"
        + "        raise RuntimeError('Stage13D embedded source hash mismatch')\n"
        + "    _path = _source_dir / _name\n"
        + "    _path.write_bytes(_source_bytes)\n"
        + "    if hashlib.sha256(_path.read_bytes()).hexdigest() != _expected:\n"
        + "        raise RuntimeError('Stage13D written source hash mismatch')\n"
        + "_runner_path = _source_dir / 'stage13d_broader_complement.py'\n"
        + "print('STAGE13D_SOURCE_CHECKS_PASS', _sources[_runner_path.name][1])\n"
    ))
    markdown("execute", "## Run the frozen comparison\n\n"
             "The job stops if an input, fold, control replay, or numerical check fails. "
             "The runner saves aggregate evidence only.\n")
    code("run", "subprocess.run([sys.executable, '-B', str(_runner_path), '--input-root', '/kaggle/input', "
         "'--output-root', '/kaggle/working/stage13d_broader_complement'], check=True, timeout=900)\n")
    markdown("readback", "## Verify the saved aggregate report\n\n"
             "Check the report scope, inputs, control replay and score arithmetic. "
             "Display only the aggregate scores and the proxy-screen decision.\n")
    code("saved-result", f"""_result_dir = Path('/kaggle/working/stage13d_broader_complement')
_report = json.loads((_result_dir / 'broader_complement_report.json').read_text(encoding='utf-8'))
def _require(condition, code):
    if not condition:
        raise RuntimeError('Stage13D saved report check failed: ' + code)
_require(type(_report) is dict, 'report_type')
_require(_report.get('status') == 'COMPLETE_BROADER_DEBERTA_COMPLEMENT_SCREEN', 'status')
_require(_report.get('scope') == {SCOPE!r}, 'scope')
_require(type(_report.get('rows')) is int and _report['rows'] == 2699, 'rows')
_require(type(_report.get('prompts')) is int and _report['prompts'] == 150, 'prompts')
_classes = _report.get('classes')
_require(type(_classes) is list and len(_classes) == 18 and
         all(type(_label) is str and _label for _label in _classes) and len(set(_classes)) == 18, 'classes')
_folds = _report.get('folds')
_require(type(_folds) is list and len(_folds) == 5 and all(type(_fold) is dict for _fold in _folds), 'folds')
_require(_report.get('expected_input_version_ids') == {[COMPETITION_ID, *SOURCE_VERSIONS]!r}, 'expected_input_version_ids')
_require(_report.get('train_sha256') == {TRAIN_CSV_SHA256!r}, 'train_sha256')
_require(_report.get('heldout_cache_sha256') == {HELDOUT_CACHE_SHA256!r}, 'heldout_cache_sha256')
_require(_report.get('deberta_cache_sha256') == {DEBERTA_CACHE_SHA256!r}, 'deberta_cache_sha256')
_require(_report.get('stage13c_candidate_confusion_sha256') == {STAGE13C_CONFUSION_SHA256!r}, 'stage13c_candidate_confusion_sha256')
_require(_report.get('blend_weights') == {BLEND_WEIGHTS!r}, 'blend_weights')
for _flag in ('fit_performed', 'control_replayed', 'stage13c_replayed'):
    _require(_report.get(_flag) is True, _flag)
for _flag in ('official_score_estimated', 'full_s07_oof_admitted', 'raw_rows_or_predictions_exported'):
    _require(_report.get(_flag) is False, _flag)
_scores = {{}}
for _key in ('glm_control_ba', 'broader_control_ba', 'candidate_ba'):
    _score = _report.get(_key)
    _require(type(_score) in (int, float) and 0 <= _score <= 1 and math.isfinite(_score), _key)
    _scores[_key] = _score
_require(math.isclose(_scores['glm_control_ba'], {CONTROL_BA!r}, abs_tol=1e-12, rel_tol=0), 'control_replay')
for _delta_key, _ci_key, _candidate, _baseline in (
        ('broader_delta_pp', 'broader_conditional_ci95_pp', 'broader_control_ba', 'glm_control_ba'),
        ('delta_pp', 'conditional_delta_ci95_pp', 'candidate_ba', 'broader_control_ba')):
    _delta = _report.get(_delta_key)
    _require(type(_delta) in (int, float) and -100 <= _delta <= 100 and math.isfinite(_delta), _delta_key)
    _require(math.isclose(_delta, (_scores[_candidate] - _scores[_baseline]) * 100,
                          abs_tol=1e-10, rel_tol=0), _delta_key + '_arithmetic')
    _ci = _report.get(_ci_key)
    _require(type(_ci) is list and len(_ci) == 2 and
             all(type(_bound) in (int, float) and -100 <= _bound <= 100 and math.isfinite(_bound) for _bound in _ci)
             and _ci[0] <= _ci[1], _ci_key)
    _scores.update({{_delta_key: _delta, _ci_key: _ci}})
for _key in ('broader_control_admitted', 'passes_proxy_screen'):
    _require(type(_report.get(_key)) is bool, _key)
    _scores[_key] = _report[_key]
_candidate_pair = _report.get('candidate_vs_broader')
_require(type(_candidate_pair) is dict, 'candidate_vs_broader')
_class_deltas = _candidate_pair.get('class_delta_pp')
_require(type(_class_deltas) is list and len(_class_deltas) == 18 and
         all(type(_x) in (int, float) and -100 <= _x <= 100 and math.isfinite(_x) for _x in _class_deltas),
         'candidate_class_delta_pp')
_require(math.isclose(sum(_class_deltas) / 18, _scores['delta_pp'], abs_tol=1e-10, rel_tol=0),
         'candidate_class_delta_mean')
_broader_criteria = dict(pooled_gain_at_least_half_pp=_scores['broader_delta_pp'] >= 0.50,
                        conditional_ci_lower_positive=_scores['broader_conditional_ci95_pp'][0] > 0)
_candidate_criteria = dict(pooled_gain_at_least_half_pp=_scores['delta_pp'] >= 0.50,
                           conditional_ci_lower_positive=_scores['conditional_delta_ci95_pp'][0] > 0,
                           every_class_recall_loss_le_five_pp=min(_class_deltas) >= -5)
for _key, _expected in (('broader_control_criteria', _broader_criteria), ('decision_criteria', _candidate_criteria)):
    _actual = _report.get(_key)
    _require(type(_actual) is dict and set(_actual) == set(_expected) and
             all(_actual[_gate] is _passes for _gate, _passes in _expected.items()), _key)
_broader_admits = all(_broader_criteria.values())
_require(_scores['broader_control_admitted'] == _broader_admits, 'broader_control_gate')
_require(_scores['passes_proxy_screen'] == (_broader_admits and all(_candidate_criteria.values())), 'proxy_screen_gate')
print('STAGE13D_SAVED_REPORT_READBACK_PASS', json.dumps(_scores, sort_keys=True, allow_nan=False))
""")

    nb = dict(nbformat=4, nbformat_minor=5, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "kaggle": {"accelerator": "none",
                   "dataSources": [{"sourceType": "competition", "sourceId": COMPETITION_ID},
                                   *[{"sourceType": "kernelVersion", "sourceId": version} for version in SOURCE_VERSIONS]],
                   "isInternetEnabled": False, "language": "python", "sourceType": "notebook", "isGpuEnabled": False},
        "stage13d_broader_complement": {"scope": SCOPE, "runner_sha256": runner_sha,
                                        "dependency_sha256": DEPENDENCIES, "train_sha256": TRAIN_CSV_SHA256,
                                        "heldout_cache_sha256": HELDOUT_CACHE_SHA256,
                                        "deberta_cache_sha256": DEBERTA_CACHE_SHA256,
                                        "evaluation_executed_by_build": False},
    }, cells=cells)
    encoded = (json.dumps(nb, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    if json.loads(encoded) != nb or len({cell["id"] for cell in cells}) != len(cells):
        raise ValueError("Stage13D notebook serialization check failed")
    code_cells = sum(cell["cell_type"] == "code" for cell in cells)
    if code_cells != 3:
        raise ValueError("Stage13D notebook must contain exactly three code cells")
    target = root / NOTEBOOK_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(encoded)
    return {"notebook_path": NOTEBOOK_PATH, "notebook_sha256": sha256(encoded), "runner_sha256": runner_sha,
            "dependency_sha256": DEPENDENCIES, "source_kernel_versions": SOURCE_VERSIONS, "competition_id": COMPETITION_ID,
            "code_cells": code_cells, "total_cells": len(cells), "evaluation_executed_by_build": False}


if __name__ == "__main__":
    print(json.dumps(build(Path(__file__).resolve().parents[1]), sort_keys=True))
