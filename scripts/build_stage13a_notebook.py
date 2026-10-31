"""Build the inert private Kaggle CPU admission notebook from reviewed sources."""
from pathlib import Path
import ast
import hashlib
import json


def digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def build(root):
    runner = (root / 'scripts/stage13a_cache_admission.py').read_text()
    ledger = (root / 'results/stage_13/S07_DEVELOPMENT_PROVENANCE.json').read_text()
    runner_sha, ledger_sha = digest(runner), digest(ledger)
    cells = []
    def markdown(cell_id, source):
        cells.append(dict(cell_type='markdown', id=cell_id, metadata={}, source=source.splitlines(keepends=True)))
    def code(cell_id, source):
        ast.parse(source)
        cells.append(dict(cell_type='code', id=cell_id, metadata={}, execution_count=None,
                          outputs=[], source=source.splitlines(keepends=True)))
    markdown('scope', '# Stage 13A — frozen-base cache admission\n\n'
              'This is a CPU input/provenance audit, with Internet off. It performs no model fitting, inference, '
              'balancing or accuracy evaluation. S07 remains the verified official public reference at 0.651.\n\n'
              '**A pass admits the GLM feature cache only. Held-out predictions and a clean development reference '
              'still require separate fold-local fits.**\n')
    markdown('inputs', '## Input contract\n\n'
              '| Input | Required immutable version |\n|---|---:|\n'
              '| Name That LLM competition | 170484 |\n'
              '| Own Stage06C feature-cache output | 359405443 |\n\n'
              'The audit checks exact train/cache digests, safe numeric/Unicode arrays, row and text alignment, '
              'finite features, all eighteen class labels and the canonical whole-prompt folds. '
              'It never imports attached model code or deserializes object arrays.\n')
    setup = ('from pathlib import Path\nimport hashlib\nimport json\nimport subprocess\nimport sys\n\n'
             + '_runner_source = ' + repr(runner) + '\n'
             + '_ledger_source = ' + repr(ledger) + '\n'
             + "assert hashlib.sha256(_runner_source.encode('utf-8')).hexdigest() == " + repr(runner_sha) + '\n'
             + "assert hashlib.sha256(_ledger_source.encode('utf-8')).hexdigest() == " + repr(ledger_sha) + '\n'
             + "_runner_path = Path('/kaggle/working/stage13a_cache_admission.py')\n"
             + "_runner_path.write_text(_runner_source, encoding='utf-8')\n"
             + "Path('/kaggle/working/stage13a_static_provenance.json').write_text(_ledger_source, encoding='utf-8')\n"
             + "print('Reviewed runner and static ledger hashes verified; no models loaded.')\n")
    code('verified-sources', setup)
    markdown('execute', '## Run the admission audit\n\n'
              'A failed contract stops this notebook. Output contains aggregate checks and hashes only. '
              'A report-writing failure is also a failure.\n')
    code('audit', "subprocess.run([sys.executable, str(_runner_path), '--input-root', '/kaggle/input', "
         "'--output-root', '/kaggle/working/stage13a_admission'], check=True)\n")
    markdown('interpret', '## Read the actual saved result\n\n'
              'The static ledger records source review before execution. The result below is read from the actual '
              'newly written audit report. Its feature-admission status is not an OOF result or an accuracy gain.\n')
    code('saved-result', "_report_path = Path('/kaggle/working/stage13a_admission/stage13a_input_admission.json')\n"
         "_actual = json.loads(_report_path.read_text(encoding='utf-8'))\n"
         "assert _actual['status'] == 'PASS_FROZEN_BASE_FEATURE_ADMISSION_ONLY'\n"
         "assert _actual['scope'] == 'frozen_base_feature_admission_only'\n"
         "assert _actual['contract_is_production'] is True and _actual['report_written'] is True\n"
         "for _flag in ['fit_performed','model_inference_performed','balancing_performed',\n"
         "              'held_out_predictions_produced','development_reference_admitted','full_s07_oof_admitted',\n"
         "              'stage13b_batch_coupled_comparison_admitted','official_score_estimated']:\n"
         "    assert _actual[_flag] is False\n"
         "print('STAGE13A_SAVED_REPORT_READBACK', _actual['status'])\n"
         "print(json.dumps({k: _actual[k] for k in ['rows','prompts','feature_shape','layers',"
         "'checks_passed_count','runtime_versions','eligible_for_future_fold_local_readout_fitting']}, sort_keys=True))\n")
    markdown('next', '## Next decision\n\n'
              'If the feature gate passes, freeze the exact common-prompt-fold readout protocol before fitting. '
              'Full-fit S07 checkpoints and incompatible source-fold predictions remain excluded from claims of '
              'clean batch-coupled validation. Stage13B and any official submission remain separate decisions.\n')
    nb = dict(nbformat=4, nbformat_minor=5, metadata={
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'},
        'stage13a': {'scope': 'frozen_base_feature_admission_only', 'runner_sha256': runner_sha,
                     'static_ledger_sha256': ledger_sha, 'intended_accelerator': 'None',
                     'intended_internet': False, 'expected_input_versions': [170484, 359405443],
                     'model_fitting_by_build': False, 'real_cache_execution_by_build': False}}, cells=cells)
    encoded = json.dumps(nb, indent=2, ensure_ascii=False) + '\n'
    reread = json.loads(encoded)
    assert reread['nbformat'] == 4 and len({c['id'] for c in cells}) == len(cells)
    assert sum(c['cell_type'] == 'code' for c in cells) == 3
    for cell in cells:
        assert isinstance(cell['source'], list)
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), '<notebook-cell>', 'exec')
            assert cell['execution_count'] is None and cell['outputs'] == []
    target = root / 'notebooks/Stage_13A_Cache_Admission.ipynb'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(encoded, encoding='utf-8')
    return {'notebook_sha256': digest(encoded), 'runner_sha256': runner_sha,
            'static_ledger_sha256': ledger_sha, 'code_cells': 3, 'total_cells': len(cells),
            'validation': 'JSON roundtrip, notebook structural invariants and compilation of all code cells; nbformat unavailable locally',
            'actual_cache_execution': False}


if __name__ == '__main__':
    print(json.dumps(build(Path(__file__).resolve().parents[1]), sort_keys=True))
