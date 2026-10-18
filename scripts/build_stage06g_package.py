"""Build Stage 06G's inert five-cell CPU package without evaluating it.

Only source text is read. AST/compile, literal-hash and nbformat checks do not
import either experiment module or open probability caches. The builder does
not call Kaggle, fit a model, run inference or evaluate the fixed candidate.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
INPUT_VERSION_IDS = [170484, 359514801]
FROZEN_GATE_SHA256 = 'a8bb68181087640fb563539870840117531a4d57e529cbeaf69e73d4c0a05be7'
FROZEN_PLAN_SHA256 = 'ece986eb6244bd322564cd7fd7a437239e5f74635a2cea2b9a39968bd891b927'
MODULE_PATHS = {
    'implementation': 'scripts/stage06g_pair_complement.py',
    'source_gate': 'scripts/stage06f_pair_diagnostics.py',
    'plan': 'plans/STAGE_06G_PAIR_COMPLEMENT.md',
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
    for key, relative in MODULE_PATHS.items():
        data = (ROOT / relative).read_bytes()
        text = data.decode('utf-8')
        assert text.encode('utf-8') == data
        if relative.endswith('.py'):
            ast.parse(text, filename=relative)
            compile(text, relative, 'exec')
        result[key] = text
    assert sha256(result['source_gate'].encode('utf-8')) == FROZEN_GATE_SHA256
    assert sha256(result['plan'].encode('utf-8')) == FROZEN_PLAN_SHA256
    return result


def build_cells(sources, freeze):
    files = {Path(MODULE_PATHS[key]).name: value for key, value in sources.items()}
    files['EXECUTION_FREEZE.json'] = json_bytes(freeze).decode('utf-8')
    file_hashes = {name: sha256(text.encode('utf-8')) for name, text in files.items()}
    setup = '''# Stage 06G: one fixed, no-fit pair-complement development comparison.
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
    scope = '''# Immutable Stage 06E V2 probabilities; no inference, training or search.
# Reused-development comparison, not full S02 or an official Kaggle score.
STAGE06G_INPUT_VERSION_IDS = [170484, 359514801]
STAGE06G_ROOTS = {
    'sources': Path('/kaggle/working/stage06g_sources'),
    'pipeline': Path('/kaggle/working/stage06g_pipeline'),
    'complement': Path('/kaggle/working/stage06g_complement'),
}
STAGE06G_STARTED = time.monotonic()
STAGE06G_PHASE = 'FRESH_ROOT_PREFLIGHT'
STAGE06G_PIPELINE_CREATED = False
STAGE06G_COMPARISON_COMPLETED = False
'''
    helpers = '''# Fresh roots and phase receipts preserve a failure without overwriting it.
def stage06g_pipeline_json(name, value):
    path = STAGE06G_ROOTS['pipeline'] / name
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\\n'
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(text, encoding='utf-8')
    temporary.replace(path)

def stage06g_pipeline_status(phase, **extra):
    global STAGE06G_PHASE
    STAGE06G_PHASE = phase
    value = dict(status=phase, observed_utc=datetime.now(timezone.utc).isoformat(),
                 elapsed_seconds=time.monotonic() - STAGE06G_STARTED, **extra)
    stage06g_pipeline_json('status.json', value)
    print('STAGE06G_PIPELINE_PHASE', json.dumps(value, allow_nan=False), flush=True)
'''
    materialization = (
        '# Inert reviewed source literals; no module is evaluated in this cell.\n'
        'STAGE06G_SOURCE_TEXT = ' + repr(files) + '\n'
        'STAGE06G_FILE_SHA256 = ' + repr(file_hashes) + '\n'
    )
    execution = '''# Materialize exact frozen bytes, then evaluate exactly one candidate.
try:
    assert all(not root.exists() for root in STAGE06G_ROOTS.values()), (
        'Fresh Stage 06G source and output roots required; preserve any prior trial',
        [str(root) for root in STAGE06G_ROOTS.values() if root.exists()])
    STAGE06G_ROOTS['pipeline'].mkdir(parents=True, exist_ok=False)
    STAGE06G_PIPELINE_CREATED = True
    stage06g_pipeline_status('MATERIALIZING_REVIEWED_SOURCES')
    STAGE06G_ROOTS['sources'].mkdir(parents=True, exist_ok=False)
    for name, source in STAGE06G_SOURCE_TEXT.items():
        assert Path(name).name == name
        encoded = source.encode('utf-8')
        assert hashlib.sha256(encoded).hexdigest() == STAGE06G_FILE_SHA256[name]
        path = STAGE06G_ROOTS['sources'] / name
        with path.open('xb') as stream:
            stream.write(encoded)
        assert path.read_bytes() == encoded
        assert hashlib.sha256(path.read_bytes()).hexdigest() == STAGE06G_FILE_SHA256[name]
    assert {path.name for path in STAGE06G_ROOTS['sources'].iterdir()} == set(STAGE06G_SOURCE_TEXT)
    stage06g_pipeline_json('materialized_sources.json', dict(
        files=STAGE06G_FILE_SHA256, exact_text_bytes_verified=True,
        input_version_ids=STAGE06G_INPUT_VERSION_IDS,
        accelerator='none', internet_enabled=False,
        source_verification_scope='Runtime local materialization; saved private notebook and attached versions require separate verification.'))
    os.environ.update({
        'NTL_STAGE06G_INPUT': '/kaggle/input',
        'NTL_STAGE06G_OUTPUT': str(STAGE06G_ROOTS['complement']),
        'NTL_STAGE06G_FREEZE': str(STAGE06G_ROOTS['sources'] / 'EXECUTION_FREEZE.json'),
    })
    assert all(os.environ[key] == '2' for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'))
    stage06g_pipeline_status('VERIFY_AND_EVALUATE_ONE_FIXED_COMPLEMENT')
    runpy.run_path(str(STAGE06G_ROOTS['sources'] / 'stage06g_pair_complement.py'), run_name='__main__')
    stage06g_terminal = json.loads((STAGE06G_ROOTS['complement'] / 'status.json').read_text())
    stage06g_result = json.loads((STAGE06G_ROOTS['complement'] / 'pair_complement_result.json').read_text())
    assert stage06g_terminal['status'] == 'COMPLETE_FIXED_ATTENTION_PAIR_COMPLEMENT' and stage06g_terminal['valid'] is True
    assert stage06g_result['status'] == 'COMPLETE_FIXED_ATTENTION_PAIR_COMPLEMENT'
    assert stage06g_result['source_script_version_id'] == 359514801
    assert all(stage06g_result[key] == 0 for key in ('fits', 'model_inference_calls', 'regrouping_calls'))
    assert stage06g_result['single_candidate_evaluated'] is True and stage06g_result['official_submission'] is False
    assert (STAGE06G_ROOTS['complement'] / 'output_manifest.json').is_file()
    STAGE06G_COMPARISON_COMPLETED = True
    stage06g_pipeline_status('RENDERING_VERIFIED_COMPARISON_OUTPUTS', completed_comparison=True)
    # Persist a compact review table and the immutable output figure in the notebook.
    import pandas as pd
    from IPython.display import Image, display
    stage06g_summary = pd.read_csv(STAGE06G_ROOTS['complement'] / 'context_summary.csv')
    display(stage06g_summary[['scenario', 'source_balanced_accuracy', 'candidate_balanced_accuracy',
                             'gain', 'corrected', 'broken', 'changed_both_wrong', 'scope_rows']])
    display(Image(filename=str(STAGE06G_ROOTS['complement'] / 'context_gain.png')))
    stage06g_pipeline_status('COMPLETE_FIXED_ATTENTION_PAIR_COMPLEMENT',
        model_fits=0, model_inference_calls=0, candidate_count=1,
        official_submission_performed=False)
    print('STAGE06G_PIPELINE_COMPLETE', flush=True)
except BaseException as stage06g_error:
    failure = dict(status='FAILED_PRESERVE_OUTPUTS', failed_phase=STAGE06G_PHASE,
        exception_type=type(stage06g_error).__name__, exception_message=str(stage06g_error),
        traceback=traceback.format_exc(), observed_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.monotonic() - STAGE06G_STARTED,
        completed_comparison=STAGE06G_COMPARISON_COMPLETED, official_submission_performed=False,
        source_sha256=STAGE06G_FILE_SHA256)
    if STAGE06G_PIPELINE_CREATED:
        stage06g_pipeline_json('failure.json', failure)
        stage06g_pipeline_json('status.json', failure)
    print('STAGE06G_PIPELINE_FAILED', json.dumps(failure, allow_nan=False), flush=True)
    raise
'''
    return [setup, scope, helpers, materialization, execution], file_hashes


def main():
    sources = source_texts()
    gate_literals = {}
    for node in ast.parse(sources['source_gate']).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id == 'EXPECTED_HASHES':
                gate_literals['EXPECTED_HASHES'] = ast.literal_eval(node.value)
    assert gate_literals['EXPECTED_HASHES'] == EXPECTED_SOURCE_INPUTS
    source_hashes = {key: sha256(value.encode('utf-8')) for key, value in sources.items()}
    freeze = dict(
        schema_version='stage06g-execution-freeze-v1',
        status='FROZEN_SOURCE_CANDIDATE_NOT_EXECUTED',
        implementation_source_sha256=source_hashes['implementation'],
        source_gate_source_sha256=source_hashes['source_gate'],
        plan_sha256=source_hashes['plan'],
        implementation_path='/kaggle/working/stage06g_sources/stage06g_pair_complement.py',
        source_gate_path='/kaggle/working/stage06g_sources/stage06f_pair_diagnostics.py',
        plan_path='/kaggle/working/stage06g_sources/STAGE_06G_PAIR_COMPLEMENT.md',
        input_version_ids=INPUT_VERSION_IDS,
        stage06e_script_version_id=359514801,
        expected_source_input_sha256=EXPECTED_SOURCE_INPUTS,
        pair=['gemini-3.7-flash', 'gemini-3.8-flash'],
        candidate_count=1, member_coefficients={'tokattn': 1.0, 'tokattn_deb': 1.0},
        contexts=[1, 3, 6], bootstrap_draws=3000, bootstrap_seed=20261017,
        accelerator='none', internet_enabled=False,
        expected_thread_limits={'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'},
        planned_model_fits=0, planned_model_inference_calls=0, planned_regrouping_calls=0,
        official_submission_performed=False,
        limitation='Preparation only. Saved private Kaggle source, pinned inputs, CPU/offline settings and independent review must be verified before execution.',
    )
    cells, materialized_hashes = build_cells(sources, freeze)
    assert len(cells) == 5 and all(len(source) < 190000 for source in cells)
    for i, source in enumerate(cells):
        ast.parse(source, filename=f'Stage_06G_cell_{i}')
        compile(source, f'Stage_06G_cell_{i}', 'exec')
    literals = {}
    for node in ast.parse(cells[3]).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ('STAGE06G_SOURCE_TEXT','STAGE06G_FILE_SHA256'):
                literals[node.targets[0].id] = ast.literal_eval(node.value)
    assert set(literals) == {'STAGE06G_SOURCE_TEXT','STAGE06G_FILE_SHA256'}
    assert literals['STAGE06G_FILE_SHA256'] == materialized_hashes
    assert set(literals['STAGE06G_SOURCE_TEXT']) == set(materialized_hashes)
    for name, text in literals['STAGE06G_SOURCE_TEXT'].items():
        assert sha256(text.encode('utf-8')) == materialized_hashes[name]
    assert json.loads(literals['STAGE06G_SOURCE_TEXT']['EXECUTION_FREEZE.json']) == freeze
    metadata = dict(
        kernelspec=dict(display_name='Python 3', language='python', name='python3'),
        language_info=dict(name='python', file_extension='.py', mimetype='text/x-python', pygments_lexer='ipython3'),
        kaggle=dict(accelerator='none', isGpuEnabled=False, isInternetEnabled=False,
                    dataSources=[dict(sourceId=value, sourceType='competition' if i == 0 else 'kernelVersion')
                                 for i, value in enumerate(INPUT_VERSION_IDS)]),
        stage06g=dict(mode='one_fixed_no_fit_pair_complement', model_fits=0, model_inference_calls=0,
                      candidate_count=1, source_paths=MODULE_PATHS, source_sha256=source_hashes),
    )
    notebook = nbformat.v4.new_notebook(metadata=metadata)
    notebook.cells = [nbformat.v4.new_code_cell(source=source, id=f'stage06g-{i}')
                      for i, source in enumerate(cells)]
    nbformat.validate(notebook)
    assert all(cell.execution_count is None and cell.outputs == [] for cell in notebook.cells)
    result_dir = ROOT / 'results/stage_06g'
    result_dir.mkdir(parents=True, exist_ok=True)
    freeze_path = result_dir / 'EXECUTION_FREEZE.json'
    freeze_path.write_bytes(json_bytes(freeze))
    notebook_path = ROOT / 'notebooks/Stage_06G_Pair_Complement.ipynb'
    nbformat.write(notebook, notebook_path)
    runner_path = ROOT / 'scripts/kaggle_stage06g_runner.py'
    runner_path.write_text(''.join(cells), encoding='utf-8')
    saved = nbformat.read(notebook_path, as_version=4)
    nbformat.validate(saved)
    assert [cell.source for cell in saved.cells] == cells
    assert runner_path.read_text(encoding='utf-8') == ''.join(cell.source for cell in saved.cells)
    compile(runner_path.read_text(encoding='utf-8'), str(runner_path), 'exec')
    assert materialized_hashes['EXECUTION_FREEZE.json'] == sha256(freeze_path.read_bytes())
    manifest = dict(
        status='PACKAGED_NOT_EXECUTED', package_schema='stage06g-five-cell-package-v1',
        builder_path='scripts/build_stage06g_package.py', builder_sha256=sha256(Path(__file__).read_bytes()),
        source_paths=MODULE_PATHS, source_sha256=source_hashes,
        execution_freeze_path='results/stage_06g/EXECUTION_FREEZE.json', execution_freeze_sha256=sha256(freeze_path.read_bytes()),
        notebook_path='notebooks/Stage_06G_Pair_Complement.ipynb', notebook_sha256=sha256(notebook_path.read_bytes()),
        linear_runner_path='scripts/kaggle_stage06g_runner.py', linear_runner_sha256=sha256(runner_path.read_bytes()),
        materialized_files=materialized_hashes,
        cells=[dict(index=i, characters=len(source), utf8_bytes=len(source.encode('utf-8')),
                    sha256=sha256(source.encode('utf-8'))) for i, source in enumerate(cells)],
        validation=dict(nbformat_schema=True, all_five_cells_compile=True, both_module_sources_compile=True,
                        linear_source_equals_exact_cell_concatenation=True, maximum_cell_characters=190000,
                        empty_outputs_and_null_execution_counts=True, exact_materialized_literal_hashes=True,
                        unchanged_frozen_source_gate_and_plan=True, no_self_referential_freeze_hash=True),
        prospective_kaggle_metadata=metadata['kaggle'], input_version_ids=INPUT_VERSION_IDS,
        accelerator='none', internet_enabled=False,
        candidate_evaluations_during_packaging=0, source_gate_executions_during_packaging=0,
        model_fits_during_packaging=0, model_inference_calls_during_packaging=0,
        browser_actions_performed=False, official_submission_performed=False,
        limitations=['Unexecuted package; no Stage 06G result exists yet.',
                     'Saved immutable source/settings and attached versions need independent pre-execution verification.',
                     'Regenerate after any implementation change; final manifest hashes define launchable source.'],
    )
    (result_dir / 'PACKAGING.json').write_bytes(json_bytes(manifest))
    print(json.dumps(dict(status=manifest['status'], source_sha256=source_hashes,
                         freeze_sha256=manifest['execution_freeze_sha256'],
                         notebook_sha256=manifest['notebook_sha256'], runner_sha256=manifest['linear_runner_sha256'],
                         cell_characters=[len(source) for source in cells], candidate_evaluations=0,
                         model_fits=0, model_inference_calls=0), indent=2))


if __name__ == '__main__':
    main()
