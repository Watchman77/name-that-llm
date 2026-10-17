"""Build, but never execute, the frozen five-cell Stage 06E Kaggle package.

The first four cells are byte-for-byte Stage 06D style payload cells. The fifth
keeps its exact transfer functions, materializes reviewed modules as inert text,
then defines the one planned CPU pipeline. Running this builder performs no
model fitting, inference, public-artifact audit, or Kaggle action.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_SHA256 = '5c27fcdffde2bcfd512d20fb4edec9bd7133430e79ccd71e31875151d235a23d'
PLAN_SHA256 = '89dbf531820b965f09fab07157475b199f929f4d297bd268e0f141c765a80f32'
D_SHA256 = 'c6005196a73100676141d18f67eb299c1bf686a547fbb077b91825d2bd2611b3'
INPUT_VERSION_IDS = [170484, 358745836, 358831649, 358911769, 358928013, 358948428, 359405443]
EXPECTED_RUNTIME = {'python': '3.13.15', 'numpy': '2.1.3', 'pandas': '2.3.3',
                    'scipy': '1.16.3', 'scikit-learn': '1.6.1', 'threadpoolctl': '3.6.0'}
MODULE_PATHS = {
    'producer': 'scripts/stage06e_capture_original.py',
    'capture_helper': 'scripts/stage06e_cache_checks.py',
    'static_preparation': 'scripts/stage06e_static_members.py',
    'consumer': 'scripts/stage06e_cached_ensemble.py',
    'original_C': 'scripts/stage06c_mixed_confirmation.py',
    'plan': 'plans/STAGE_06E_CANONICAL_CPU_EVIDENCE.md',
}
D_PAYLOAD_CELL_HASHES = [
    '86e5a3d50cd1850e65139b3fc7c8c39a77bd8ab8508f3208d101fe194a357fda',
    'bd765f02846b89e7537a6d92655cafa7f7664ba8199463048aca422ad9f04bac',
    'e7f8d7e91ad9179852c6606710d490c30fa510175d9e0d99101c29b6f2847504',
    '6c17db3c11cc0e1d4ad51710aa4874e16b1831701f608a3f54048b2cb2f06a29',
]


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def json_bytes(value) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')


def module_texts():
    result = {}
    for key, path in MODULE_PATHS.items():
        data = (ROOT / path).read_bytes()
        text = data.decode('utf-8')
        assert text.encode('utf-8') == data
        if path.endswith('.py'):
            ast.parse(text, filename=path)
            compile(text, path, 'exec')
        result[key] = text
    assert sha256(result['original_C'].encode()) == ORIGINAL_SHA256
    assert sha256(result['plan'].encode()) == PLAN_SHA256
    assert sha256((ROOT / 'scripts/stage06d_fixed_ensemble.py').read_bytes()) == D_SHA256
    return result


def build_runner(sources, freeze):
    """Return source text only. Its body is not invoked by package construction."""
    source_files = {Path(MODULE_PATHS[key]).name: value for key, value in sources.items()}
    source_files['EXECUTION_FREEZE.json'] = json_bytes(freeze).decode('utf-8')
    source_hashes = {name: sha256(value.encode('utf-8')) for name, value in source_files.items()}
    expected_sources = {key: sha256(value.encode('utf-8')) for key, value in sources.items()}
    expected_sources['freeze'] = source_hashes['EXECUTION_FREEZE.json']
    constants = (
        '\n# Exact reviewed files are text literals until their verified materialization.\n'
        + 'STAGE06E_SOURCE_TEXT = ' + repr(source_files) + '\n'
        + 'STAGE06E_FILE_SHA256 = ' + repr(source_hashes) + '\n'
        + 'STAGE06E_EXPECTED_SOURCES = ' + repr(expected_sources) + '\n'
    )
    pipeline = '''
import runpy
import sys
import time
import traceback
from datetime import datetime, timezone

STAGE06E_ROOTS = {
    'sources': Path('/kaggle/working/stage06e_sources'),
    'style': Path('/kaggle/working/stage06e_stylegram'),
    'static': Path('/kaggle/working/stage06e_static_members'),
    'canonical': Path('/kaggle/working/stage06e_cache'),
    'ensemble': Path('/kaggle/working/stage06e_ensemble'),
    'pipeline': Path('/kaggle/working/stage06e_pipeline'),
}
STAGE06E_STARTED = time.monotonic()
STAGE06E_PHASE = 'FRESH_ROOT_PREFLIGHT'
STAGE06E_PIPELINE_CREATED = False

def stage06e_pipeline_json(name, value):
    path = STAGE06E_ROOTS['pipeline'] / name
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\\n'
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(text)
    temporary.replace(path)

def stage06e_pipeline_status(phase, **extra):
    global STAGE06E_PHASE
    STAGE06E_PHASE = phase
    value = dict(status=phase, observed_utc=datetime.now(timezone.utc).isoformat(),
                 elapsed_seconds=time.monotonic() - STAGE06E_STARTED, **extra)
    stage06e_pipeline_json('status.json', value)
    print('STAGE06E_PIPELINE_PHASE', json.dumps(value, allow_nan=False), flush=True)

try:
    # Existing empty directories also count as prior trials. Never overwrite them.
    assert all(not root.exists() for root in STAGE06E_ROOTS.values()), (
        'Fresh Stage 06E source and output roots required; preserve any prior trial',
        [str(root) for root in STAGE06E_ROOTS.values() if root.exists()])
    STAGE06E_ROOTS['pipeline'].mkdir(parents=True, exist_ok=False)
    STAGE06E_PIPELINE_CREATED = True
    stage06e_pipeline_status('MATERIALIZING_REVIEWED_SOURCES')
    STAGE06E_ROOTS['sources'].mkdir(parents=True, exist_ok=False)
    for name, source in STAGE06E_SOURCE_TEXT.items():
        assert Path(name).name == name
        encoded = source.encode('utf-8')
        assert hashlib.sha256(encoded).hexdigest() == STAGE06E_FILE_SHA256[name]
        path = STAGE06E_ROOTS['sources'] / name
        with path.open('xb') as stream:
            stream.write(encoded)
        assert path.read_bytes() == encoded
        assert hashlib.sha256(path.read_bytes()).hexdigest() == STAGE06E_FILE_SHA256[name]
    assert {p.name for p in STAGE06E_ROOTS['sources'].iterdir()} == set(STAGE06E_SOURCE_TEXT)
    stage06e_pipeline_json('materialized_sources.json', dict(
        expected_sources=STAGE06E_EXPECTED_SOURCES,
        files=STAGE06E_FILE_SHA256,
        exact_text_bytes_verified=True,
        input_version_ids=[170484,358745836,358831649,358911769,358928013,358948428,359405443],
        accelerator='none', internet_enabled=False,
        source_verification_scope='Runtime local materialization; immutable notebook and attached versions require separate saved-source verification.'))
    os.environ.update({
        'NTL_STAGE06E_INPUT': '/kaggle/input',
        'NTL_STAGE06E_OUTPUT': str(STAGE06E_ROOTS['canonical']),
        'NTL_STAGE06E_STYLE_ROOT': str(STAGE06E_ROOTS['style']),
        'NTL_STAGE06E_STATIC_OUTPUT': str(STAGE06E_ROOTS['static']),
        'NTL_STAGE06E_ENSEMBLE_OUTPUT': str(STAGE06E_ROOTS['ensemble']),
        'NTL_STAGE06E_FREEZE': str(STAGE06E_ROOTS['sources'] / 'EXECUTION_FREEZE.json'),
    })
    assert all(os.environ[key] == '2' for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'))
    sys.path.insert(0, str(STAGE06E_ROOTS['sources']))
    for name in ('stage06e_cache_checks','stage06e_static_members','stage06e_cached_ensemble'):
        assert name not in sys.modules, ('Stale Stage 06E imported module', name)
    stage06e_pipeline_status('TRANSFER_FROZEN_STYLE_PROBABILITIES')
    STYLE_P, STYLE_CLASSES, STYLE_IDS = transfer_style_reference(
        input_root=os.environ['NTL_STAGE06E_INPUT'], output_root=str(STAGE06E_ROOTS['style']))
    stage06e_pipeline_status('AUDIT_AND_CACHE_EXISTING_STATIC_MEMBERS')
    from stage06e_static_members import prepare_static_members
    prepare_static_members(STYLE_P, STYLE_CLASSES, STYLE_IDS)
    stage06e_pipeline_status('CAPTURE_ORIGINAL_CANONICAL_CPU_COMPUTATION')
    import stage06e_cache_checks as stage06e_cachechecks
    try:
        runpy.run_path(str(STAGE06E_ROOTS['sources'] / 'stage06e_capture_original.py'), run_name='__main__')
    except BaseException as stage06e_producer_error:
        stage06e_cachechecks.preserve_active_failure(stage06e_producer_error)
        raise
    stage06e_pipeline_status('CONSUME_PASSED_FROZEN_CACHES_WITHOUT_REFITTING')
    from stage06e_cached_ensemble import run_cached_ensemble
    STAGE06E_RESULT = run_cached_ensemble(
        canonical_root=STAGE06E_ROOTS['canonical'],
        static_root=STAGE06E_ROOTS['static'],
        expected_sources=STAGE06E_EXPECTED_SOURCES,
        output_root=STAGE06E_ROOTS['ensemble'])
    stage06e_pipeline_status('COMPLETE_CANONICAL_CPU_FIXED_AVAILABLE_COMPONENT_PROXY',
        valid_development_result=True, official_submission_performed=False,
        primary_raw_screen_pass=bool(STAGE06E_RESULT['primary_raw_screen_pass']))
    print('STAGE06E_PIPELINE_COMPLETE', flush=True)
except BaseException as stage06e_pipeline_error:
    stage06e_failure = dict(status='FAILED_PRESERVE_OUTPUTS', failed_phase=STAGE06E_PHASE,
        exception_type=type(stage06e_pipeline_error).__name__, exception_message=str(stage06e_pipeline_error),
        traceback=traceback.format_exc(), observed_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.monotonic() - STAGE06E_STARTED,
        valid_ensemble_result=False, official_submission_performed=False,
        expected_sources=STAGE06E_EXPECTED_SOURCES)
    if STAGE06E_PIPELINE_CREATED:
        stage06e_pipeline_json('failure.json', stage06e_failure)
        stage06e_pipeline_json('status.json', stage06e_failure)
    print('STAGE06E_PIPELINE_FAILED', json.dumps(stage06e_failure, allow_nan=False), flush=True)
    raise
'''
    return constants + pipeline, source_hashes


def main():
    sources = module_texts()
    source_hashes = {key: sha256(value.encode('utf-8')) for key, value in sources.items()}
    freeze = dict(
        schema_version='stage06e-execution-freeze-v1',
        status='FROZEN_SOURCE_CANDIDATE_NOT_EXECUTED',
        capture_source_sha256=source_hashes['producer'],
        cache_checks_source_sha256=source_hashes['capture_helper'],
        static_preparation_source_sha256=source_hashes['static_preparation'],
        consumer_source_sha256=source_hashes['consumer'],
        original_source_sha256=ORIGINAL_SHA256, plan_sha256=PLAN_SHA256,
        original_D_fusion_source_sha256=D_SHA256,
        input_version_ids=INPUT_VERSION_IDS, accelerator='none', internet_enabled=False,
        expected_runtime=EXPECTED_RUNTIME,
        expected_thread_limits={'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'},
        planned_head_fits=15, static_style_probability_fits=0, cache_consumer_fits=0,
        official_submission_performed=False,
        limitation='Preparation record only. Exact saved private Kaggle source, seven input versions and settings must be verified before execution.',
    )
    original_notebook = nbformat.read(ROOT / 'notebooks/Stage_06D_Fixed_Ensemble.ipynb', as_version=4)
    assert len(original_notebook.cells) == 5 and all(c.cell_type == 'code' for c in original_notebook.cells)
    cells = [c.source for c in original_notebook.cells]
    assert [sha256(s.encode()) for s in cells[:4]] == D_PAYLOAD_CELL_HASHES
    marker = '"""Fixed, CPU-only available-component ensemble comparison.'
    assert cells[4].count(marker) == 1
    transfer_prefix = cells[4][:cells[4].index(marker)]
    assert 'def transfer_style_reference(' in transfer_prefix
    assert 'run_fixed_ensemble(' not in transfer_prefix
    entrypoint, materialized_hashes = build_runner(sources, freeze)
    new_cells = cells[:4] + [transfer_prefix + entrypoint]
    assert len(new_cells) == 5 and all(len(s) < 190000 for s in new_cells)
    for i, source in enumerate(new_cells):
        ast.parse(source, filename=f'Stage_06E_cell_{i}')
        compile(source, f'Stage_06E_cell_{i}', 'exec')
    # Read generated literals back through the AST, without importing/executing
    # any producer, public-artifact audit, runner or consumer code.
    literals = {}
    for node in ast.parse(new_cells[4]).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ('STAGE06E_SOURCE_TEXT', 'STAGE06E_FILE_SHA256', 'STAGE06E_EXPECTED_SOURCES'):
                literals[name] = ast.literal_eval(node.value)
    assert len(literals) == 3
    assert literals['STAGE06E_FILE_SHA256'] == materialized_hashes
    assert set(literals['STAGE06E_SOURCE_TEXT']) == set(materialized_hashes)
    for name, text in literals['STAGE06E_SOURCE_TEXT'].items():
        assert sha256(text.encode('utf-8')) == materialized_hashes[name]
    assert literals['STAGE06E_EXPECTED_SOURCES'] == dict(
        source_hashes, freeze=sha256(json_bytes(freeze)))
    assert json.loads(literals['STAGE06E_SOURCE_TEXT']['EXECUTION_FREEZE.json']) == freeze
    source_metadata = copy.deepcopy(dict(original_notebook.metadata))
    source_metadata['kaggle'] = dict(accelerator='none', isGpuEnabled=False, isInternetEnabled=False,
        dataSources=[dict(sourceId=value, sourceType='competition' if i == 0 else 'kernelVersion')
                     for i, value in enumerate(INPUT_VERSION_IDS)])
    notebook = nbformat.v4.new_notebook(metadata=source_metadata)
    notebook.cells = [nbformat.v4.new_code_cell(source=source, id=f'stage06e-{i}') for i, source in enumerate(new_cells)]
    nbformat.validate(notebook)
    for cell in notebook.cells:
        assert cell.execution_count is None and cell.outputs == []
    result_dir = ROOT / 'results/stage_06e'
    result_dir.mkdir(parents=True, exist_ok=True)
    freeze_path = result_dir / 'EXECUTION_FREEZE.json'
    freeze_path.write_bytes(json_bytes(freeze))
    notebook_path = ROOT / 'notebooks/Stage_06E_Canonical_CPU_Evidence.ipynb'
    nbformat.write(notebook, notebook_path)
    runner_path = ROOT / 'scripts/kaggle_stage06e_multicell.py'
    runner_path.write_text(''.join(new_cells), encoding='utf-8')
    readback = nbformat.read(notebook_path, as_version=4)
    nbformat.validate(readback)
    assert [c.source for c in readback.cells] == new_cells
    assert runner_path.read_text() == ''.join(c.source for c in readback.cells)
    compile(runner_path.read_text(), str(runner_path), 'exec')
    assert materialized_hashes['EXECUTION_FREEZE.json'] == sha256(freeze_path.read_bytes())
    manifest = dict(
        status='PACKAGED_NOT_EXECUTED', package_schema='stage06e-five-cell-package-v1',
        builder_path='scripts/build_stage06e_package.py', builder_sha256=sha256(Path(__file__).read_bytes()),
        source_paths=MODULE_PATHS, source_sha256=source_hashes,
        execution_freeze_path='results/stage_06e/EXECUTION_FREEZE.json', execution_freeze_sha256=sha256(freeze_path.read_bytes()),
        notebook_path='notebooks/Stage_06E_Canonical_CPU_Evidence.ipynb', notebook_sha256=sha256(notebook_path.read_bytes()),
        linear_runner_path='scripts/kaggle_stage06e_multicell.py', linear_runner_sha256=sha256(runner_path.read_bytes()),
        style_transfer_prefix_sha256=sha256(transfer_prefix.encode()), style_transfer_prefix_characters=len(transfer_prefix),
        materialized_files=materialized_hashes,
        cells=[dict(index=i, characters=len(s), utf8_bytes=len(s.encode()), sha256=sha256(s.encode()),
                    exact_Stage06D_payload_cell=i < 4) for i, s in enumerate(new_cells)],
        validation=dict(nbformat_schema=True, all_five_cells_compile=True, all_module_sources_compile=True,
                        linear_source_equals_exact_cell_concatenation=True, first_four_cells_unchanged=True,
                        maximum_cell_characters=190000, empty_outputs_and_null_execution_counts=True,
                        exact_materialized_source_literal_hashes=True, no_self_referential_freeze_hash=True),
        prospective_kaggle_metadata=source_metadata['kaggle'], input_version_ids=INPUT_VERSION_IDS,
        accelerator='none', internet_enabled=False,
        stage06e_model_fits_during_packaging=0, stage06e_model_inference_calls_during_packaging=0,
        browser_actions_performed=False, official_submission_performed=False,
        limitations=['Source packaging is inert preparation; no valid development result exists yet.',
                     'Saved immutable Kaggle source/settings verification and independent review remain external pre-execution gates.',
                     'Regenerate after any reviewed-module change; only the final manifest hashes define launchable source.'])
    (result_dir / 'PACKAGING.json').write_bytes(json_bytes(manifest))
    print(json.dumps(dict(status=manifest['status'], source_sha256=source_hashes,
                         freeze_sha256=manifest['execution_freeze_sha256'],
                         notebook_sha256=manifest['notebook_sha256'], runner_sha256=manifest['linear_runner_sha256'],
                         cell_characters=[len(s) for s in new_cells], model_fits=0), indent=2))


if __name__ == '__main__':
    main()
