"""Build the Stage 06F five-cell diagnostic notebook without running it.

The builder reads source text, validates Python/nbformat, and writes a
deterministic package. It never imports the diagnostic module, opens numeric
caches, fits a model, runs inference, or interacts with Kaggle.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
INPUT_VERSION_IDS = [170484, 359514801]
MODULE_PATHS = {
    'diagnostic': 'scripts/stage06f_pair_diagnostics.py',
    'plan': 'plans/STAGE_06F_PAIR_DIAGNOSTICS.md',
}
EXPECTED_SOURCE_INPUTS = {
    'ensemble_arrays.npz': '9348eeed149ad48bdd828d8055e945f57e384d8917c654125c2ebbe8d4b99519',
    'ensemble_result.json': '9717bcb12f0c730b4af631067621969dcdbf239e6bed2f84304b470a38458018',
    'train.csv': '40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707',
    'output_manifest.json': 'a5fcffaef27f4942af8a679370c30c247b8d872b4b6d711eccc0101ea7006dbd',
    'raw_control_scenario1_confusion.csv': 'a48b61f2a7ffb50764d44bb060adb925e13122abd63fed4730c9ba097685c595',
    'raw_control_scenario3_confusion.csv': 'fae2720b0f34b0bb701bfdffaace99051f0add84d52c91e6973aa6f124299463',
    'raw_control_scenario6_confusion.csv': '938f3bf5a1ea51fdda8b8a75512810c5c74002063e741feef0d8e5dd614c7513',
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')


def source_texts():
    result = {}
    for key, relative_path in MODULE_PATHS.items():
        data = (ROOT / relative_path).read_bytes()
        text = data.decode('utf-8')
        assert text.encode('utf-8') == data
        if relative_path.endswith('.py'):
            ast.parse(text, filename=relative_path)
            compile(text, relative_path, 'exec')
        result[key] = text
    return result


def build_cells(sources, freeze):
    files = {Path(MODULE_PATHS[key]).name: value for key, value in sources.items()}
    files['EXECUTION_FREEZE.json'] = json_bytes(freeze).decode('utf-8')
    file_hashes = {name: sha256(text.encode('utf-8')) for name, text in files.items()}
    setup = '''# Stage 06F: cached Gemini pair diagnosis. No fits or new predictions.
import os
os.environ.update({
    'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
    'NUMEXPR_NUM_THREADS': '2',
})
import hashlib
import json
import runpy
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
'''
    scope = '''# Fixed scope: Stage 06E V2 saved raw-control evidence and official train.
# The same passages recur in three contexts; contexts are not independent data.
# This diagnostic changes no prediction and cannot increase the official score.
STAGE06F_INPUT_VERSION_IDS = [170484, 359514801]
STAGE06F_ROOTS = {
    'sources': Path('/kaggle/working/stage06f_sources'),
    'pipeline': Path('/kaggle/working/stage06f_pipeline'),
    'diagnostics': Path('/kaggle/working/stage06f_diagnostics'),
}
STAGE06F_STARTED = time.monotonic()
STAGE06F_PHASE = 'FRESH_ROOT_PREFLIGHT'
STAGE06F_PIPELINE_CREATED = False
'''
    receipt_helpers = '''# Preserve phase and failure receipts; never overwrite a prior trial.
def stage06f_pipeline_json(name, value):
    path = STAGE06F_ROOTS['pipeline'] / name
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\\n'
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(text, encoding='utf-8')
    temporary.replace(path)

def stage06f_pipeline_status(phase, **extra):
    global STAGE06F_PHASE
    STAGE06F_PHASE = phase
    value = dict(status=phase, observed_utc=datetime.now(timezone.utc).isoformat(),
                 elapsed_seconds=time.monotonic() - STAGE06F_STARTED, **extra)
    stage06f_pipeline_json('status.json', value)
    print('STAGE06F_PIPELINE_PHASE', json.dumps(value, allow_nan=False), flush=True)
'''
    materialization = (
        '# These are inert reviewed text literals until verified materialization.\n'
        'STAGE06F_SOURCE_TEXT = ' + repr(files) + '\n'
        'STAGE06F_FILE_SHA256 = ' + repr(file_hashes) + '\n'
    )
    execution = '''# Execute one diagnostic only after all frozen source bytes match.
try:
    assert all(not root.exists() for root in STAGE06F_ROOTS.values()), (
        'Fresh Stage 06F source and output roots required; preserve any prior trial',
        [str(root) for root in STAGE06F_ROOTS.values() if root.exists()])
    STAGE06F_ROOTS['pipeline'].mkdir(parents=True, exist_ok=False)
    STAGE06F_PIPELINE_CREATED = True
    stage06f_pipeline_status('MATERIALIZING_REVIEWED_SOURCES')
    STAGE06F_ROOTS['sources'].mkdir(parents=True, exist_ok=False)
    for name, source in STAGE06F_SOURCE_TEXT.items():
        assert Path(name).name == name
        encoded = source.encode('utf-8')
        assert hashlib.sha256(encoded).hexdigest() == STAGE06F_FILE_SHA256[name]
        path = STAGE06F_ROOTS['sources'] / name
        with path.open('xb') as stream:
            stream.write(encoded)
        assert path.read_bytes() == encoded
        assert hashlib.sha256(path.read_bytes()).hexdigest() == STAGE06F_FILE_SHA256[name]
    assert {path.name for path in STAGE06F_ROOTS['sources'].iterdir()} == set(STAGE06F_SOURCE_TEXT)
    stage06f_pipeline_json('materialized_sources.json', dict(
        files=STAGE06F_FILE_SHA256, exact_text_bytes_verified=True,
        input_version_ids=STAGE06F_INPUT_VERSION_IDS,
        accelerator='none', internet_enabled=False,
        source_verification_scope='Runtime local materialization; saved private notebook and attached versions require separate verification.'))
    os.environ.update({
        'NTL_STAGE06F_INPUT': '/kaggle/input',
        'NTL_STAGE06F_OUTPUT': str(STAGE06F_ROOTS['diagnostics']),
        'NTL_STAGE06F_FREEZE': str(STAGE06F_ROOTS['sources'] / 'EXECUTION_FREEZE.json'),
    })
    assert all(os.environ[key] == '2' for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'))
    stage06f_pipeline_status('VERIFY_AND_DIAGNOSE_SAVED_EVIDENCE')
    runpy.run_path(str(STAGE06F_ROOTS['sources'] / 'stage06f_pair_diagnostics.py'), run_name='__main__')
    stage06f_terminal = json.loads((STAGE06F_ROOTS['diagnostics'] / 'status.json').read_text())
    stage06f_result = json.loads((STAGE06F_ROOTS['diagnostics'] / 'pair_result.json').read_text())
    assert stage06f_terminal['status'] == 'COMPLETE_FIXED_PAIR_DIAGNOSTICS' and stage06f_terminal['valid'] is True
    assert stage06f_result['status'] == 'COMPLETE_FIXED_PAIR_DIAGNOSTICS'
    assert stage06f_result['source_script_version_id'] == 359514801 and stage06f_result['primary_source'] == 'raw_control'
    assert all(stage06f_result[key] == 0 for key in ('fits', 'model_inference_calls', 'regrouping_calls', 'changed_predictions'))
    assert stage06f_result['corrective_candidate_evaluated'] is False and stage06f_result['official_submission'] is False
    assert (STAGE06F_ROOTS['diagnostics'] / 'output_manifest.json').is_file()
    stage06f_pipeline_status('COMPLETE_FIXED_PAIR_DIAGNOSTICS',
        model_fits=0, model_inference_calls=0, changed_predictions=0,
        official_submission_performed=False)
    print('STAGE06F_PIPELINE_COMPLETE', flush=True)
except BaseException as stage06f_error:
    failure = dict(status='FAILED_PRESERVE_OUTPUTS', failed_phase=STAGE06F_PHASE,
        exception_type=type(stage06f_error).__name__, exception_message=str(stage06f_error),
        traceback=traceback.format_exc(), observed_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.monotonic() - STAGE06F_STARTED,
        completed_diagnostic=False, official_submission_performed=False,
        source_sha256=STAGE06F_FILE_SHA256)
    if STAGE06F_PIPELINE_CREATED:
        stage06f_pipeline_json('failure.json', failure)
        stage06f_pipeline_json('status.json', failure)
    print('STAGE06F_PIPELINE_FAILED', json.dumps(failure, allow_nan=False), flush=True)
    raise
'''
    return [setup, scope, receipt_helpers, materialization, execution], file_hashes


def main():
    sources = source_texts()
    diagnostic_literals = {}
    for node in ast.parse(sources['diagnostic']).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id == 'EXPECTED_HASHES':
                diagnostic_literals['EXPECTED_HASHES'] = ast.literal_eval(node.value)
    assert diagnostic_literals['EXPECTED_HASHES'] == EXPECTED_SOURCE_INPUTS
    source_hashes = {key: sha256(value.encode('utf-8')) for key, value in sources.items()}
    freeze = dict(
        schema_version='stage06f-execution-freeze-v1',
        status='FROZEN_SOURCE_CANDIDATE_NOT_EXECUTED',
        diagnostic_source_sha256=source_hashes['diagnostic'],
        plan_sha256=source_hashes['plan'],
        source_path='/kaggle/working/stage06f_sources/stage06f_pair_diagnostics.py',
        plan_path='/kaggle/working/stage06f_sources/STAGE_06F_PAIR_DIAGNOSTICS.md',
        input_version_ids=INPUT_VERSION_IDS,
        stage06e_script_version_id=359514801,
        expected_source_input_sha256=EXPECTED_SOURCE_INPUTS,
        accelerator='none', internet_enabled=False,
        expected_thread_limits={'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'},
        planned_model_fits=0, planned_model_inference_calls=0,
        planned_changed_predictions=0, official_submission_performed=False,
        limitation='Preparation only. Saved private Kaggle source, pinned inputs, CPU/offline settings and independent review must be verified before execution.',
    )
    cells, materialized_hashes = build_cells(sources, freeze)
    assert len(cells) == 5 and all(len(source) < 190000 for source in cells)
    for i, source in enumerate(cells):
        ast.parse(source, filename=f'Stage_06F_cell_{i}')
        compile(source, f'Stage_06F_cell_{i}', 'exec')
    literals = {}
    for node in ast.parse(cells[3]).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ('STAGE06F_SOURCE_TEXT','STAGE06F_FILE_SHA256'):
                literals[node.targets[0].id] = ast.literal_eval(node.value)
    assert set(literals) == {'STAGE06F_SOURCE_TEXT','STAGE06F_FILE_SHA256'}
    assert literals['STAGE06F_FILE_SHA256'] == materialized_hashes
    assert set(literals['STAGE06F_SOURCE_TEXT']) == set(materialized_hashes)
    for name, text in literals['STAGE06F_SOURCE_TEXT'].items():
        assert sha256(text.encode('utf-8')) == materialized_hashes[name]
    assert json.loads(literals['STAGE06F_SOURCE_TEXT']['EXECUTION_FREEZE.json']) == freeze
    metadata = dict(
        kernelspec=dict(display_name='Python 3', language='python', name='python3'),
        language_info=dict(name='python', file_extension='.py', mimetype='text/x-python', pygments_lexer='ipython3'),
        kaggle=dict(accelerator='none', isGpuEnabled=False, isInternetEnabled=False,
                    dataSources=[dict(sourceId=value, sourceType='competition' if i == 0 else 'kernelVersion')
                                 for i, value in enumerate(INPUT_VERSION_IDS)]),
        stage06f=dict(mode='fixed_cached_evidence_diagnostic', model_fits=0, model_inference_calls=0,
                      source_paths=MODULE_PATHS, source_sha256=source_hashes),
    )
    notebook = nbformat.v4.new_notebook(metadata=metadata)
    notebook.cells = [nbformat.v4.new_code_cell(source=source, id=f'stage06f-{i}')
                      for i, source in enumerate(cells)]
    nbformat.validate(notebook)
    for cell in notebook.cells:
        assert cell.execution_count is None and cell.outputs == []
    result_dir = ROOT / 'results/stage_06f'
    result_dir.mkdir(parents=True, exist_ok=True)
    freeze_path = result_dir / 'EXECUTION_FREEZE.json'
    freeze_path.write_bytes(json_bytes(freeze))
    notebook_path = ROOT / 'notebooks/Stage_06F_Pair_Diagnostics.ipynb'
    nbformat.write(notebook, notebook_path)
    runner_path = ROOT / 'scripts/kaggle_stage06f_runner.py'
    runner_path.write_text(''.join(cells), encoding='utf-8')
    saved = nbformat.read(notebook_path, as_version=4)
    nbformat.validate(saved)
    assert [cell.source for cell in saved.cells] == cells
    assert runner_path.read_text(encoding='utf-8') == ''.join(cell.source for cell in saved.cells)
    compile(runner_path.read_text(encoding='utf-8'), str(runner_path), 'exec')
    assert materialized_hashes['EXECUTION_FREEZE.json'] == sha256(freeze_path.read_bytes())
    manifest = dict(
        status='PACKAGED_NOT_EXECUTED', package_schema='stage06f-five-cell-package-v1',
        builder_path='scripts/build_stage06f_package.py', builder_sha256=sha256(Path(__file__).read_bytes()),
        source_paths=MODULE_PATHS, source_sha256=source_hashes,
        execution_freeze_path='results/stage_06f/EXECUTION_FREEZE.json', execution_freeze_sha256=sha256(freeze_path.read_bytes()),
        notebook_path='notebooks/Stage_06F_Pair_Diagnostics.ipynb', notebook_sha256=sha256(notebook_path.read_bytes()),
        linear_runner_path='scripts/kaggle_stage06f_runner.py', linear_runner_sha256=sha256(runner_path.read_bytes()),
        materialized_files=materialized_hashes,
        cells=[dict(index=i, characters=len(source), utf8_bytes=len(source.encode('utf-8')),
                    sha256=sha256(source.encode('utf-8'))) for i, source in enumerate(cells)],
        validation=dict(nbformat_schema=True, all_five_cells_compile=True, diagnostic_source_compiles=True,
                        linear_source_equals_exact_cell_concatenation=True, maximum_cell_characters=190000,
                        empty_outputs_and_null_execution_counts=True, exact_materialized_literal_hashes=True,
                        no_self_referential_freeze_hash=True),
        prospective_kaggle_metadata=metadata['kaggle'], input_version_ids=INPUT_VERSION_IDS,
        accelerator='none', internet_enabled=False,
        diagnostic_executions_during_packaging=0, model_fits_during_packaging=0,
        model_inference_calls_during_packaging=0, browser_actions_performed=False,
        official_submission_performed=False,
        limitations=['Unexecuted package; no new diagnostic result exists yet.',
                     'Saved immutable source/settings and attached versions need independent pre-execution verification.',
                     'Regenerate after any diagnostic or plan change; final manifest hashes define launchable source.'],
    )
    (result_dir / 'PACKAGING.json').write_bytes(json_bytes(manifest))
    print(json.dumps(dict(status=manifest['status'], source_sha256=source_hashes,
                         freeze_sha256=manifest['execution_freeze_sha256'],
                         notebook_sha256=manifest['notebook_sha256'], runner_sha256=manifest['linear_runner_sha256'],
                         cell_characters=[len(source) for source in cells], diagnostic_executions=0,
                         model_fits=0, model_inference_calls=0), indent=2))


if __name__ == '__main__':
    main()
