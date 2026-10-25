"""Read-only saved-output validation for Lightning V28 362014737.

Writes aggregate evidence only. Never imports models, upstream modules or unsafe
serializers, changes predictions, fits a model, or reports validation accuracy.
The exact-public-sample route is explicitly FORMAT_ONLY_PREVIEW.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

LABELS = [
    'gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',
    'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
    'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',
    'claude-fable-5', 'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3',
]
K = 18
PUBLIC_FIXTURE = [
    ('00353fb1c541', 321, '0c306cbd5d401399327e4cb46f766928'),
    ('005f5a9a6139', 764, 'c2853bf2b68bdd808a8a05cec2439b5d'),
    ('0062a53d5487', 1077, 'f8c583ac9cbc0806bee20f96c9124159'),
]
HELPER_MEMBERS = ['fourier', 'laplace', 'knn', 'rbf', 'centered', 'word_pca', 'char_pca', 'late']
DEFAULT_TEMP = '/kaggle/temp/ntl636-fulltext-qwen-query'
DEFAULT_METADATA = '/kaggle/input/ntl-olmo2-native24-final-inference-assets/ntl627-own-fulltext/ntl627-own-fulltext/final.json'
SOURCE_VERSION = 362014737
ATOL, RTOL = 2e-8, 3e-7


class OutputValidationError(ValueError):
    """Messages are aggregate check names, never test rows or text."""


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def softmax(value):
    value = np.asarray(value, dtype=np.float64)
    z = np.exp(value - value.max(1, keepdims=True))
    return z / z.sum(1, keepdims=True)


def source_sinkhorn(probabilities):
    """Exact arithmetic from V28 reference627_common.sinkhorn, no new settings."""
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != K or not np.isfinite(p).all() or (p < 0).any():
        raise OutputValidationError('sinkhorn:finite_nonnegative_matrix')
    p = np.maximum(p, 1e-300)
    if len(p) < 50:
        return p / p.sum(1, keepdims=True)
    for _ in range(300):
        p *= (len(p) / K) / p.sum(0, keepdims=True)
        p /= p.sum(1, keepdims=True)
    return p


def fixture_matches(frame):
    actual = [(r.id, len(r.text.encode('utf-8')),
               hashlib.blake2s(r.text.encode('utf-8'), digest_size=16).hexdigest())
              for r in frame.itertuples(index=False)]
    return actual == PUBLIC_FIXTURE


def validate_output(work_dir, test_path, *, temporary_dir=DEFAULT_TEMP,
                    namespace=None, final_metadata_path=DEFAULT_METADATA,
                    report_path=None, parquet_reader=None):
    """Validate on-disk artifacts, writing no original rows or predictions.

    Production uses pandas.read_parquet. The optional reader is exclusively a
    local synthetic-test boundary; the production notebook never supplies it.
    """
    work, temp, test_path = Path(work_dir), Path(temporary_dir), Path(test_path)
    report_path = Path(report_path) if report_path else work / 'stage11_output_validation.json'
    reader = pd.read_parquet if parquet_reader is None else parquet_reader
    report = {
        'stage': 11, 'source_version': SOURCE_VERSION, 'status': 'VALIDATION_IN_PROGRESS',
        'checks': [], 'files': {}, 'accuracy_validation': False,
        'original_ids_text_or_probability_rows_exported': False,
        'full_pre_ATLAS_internal_reconstruction_verified': False,
        'all_internal_activations_finite_verified': False,
        'parquet_readback': 'real pandas reader' if parquet_reader is None else 'synthetic reader boundary',
        'limitations': [
            'Saved SVC logits are unavailable, so the complete pre-ATLAS SVC/decoder/DeBERTa sum is not reconstructed.',
            'Finite saved artifacts do not prove every internal model activation is finite.',
            'These are execution/output checks, not honest OOF accuracy or private leaderboard evidence.',
            'The eight helper probability members are checked, but their fitted internal joblib weights are not deserialized.',
        ],
    }

    def check(name, condition):
        passed = bool(condition)
        report['checks'].append({'name': name, 'pass': passed})
        if not passed:
            raise OutputValidationError(name)

    def close(name, actual, expected, atol=ATOL, rtol=RTOL):
        check(name, np.shape(actual) == np.shape(expected) and
              np.allclose(actual, expected, atol=atol, rtol=rtol))

    def record(path):
        path = Path(path)
        check(path.name + ':exists', path.is_file())
        report['files'][path.name] = {'sha256': sha(path), 'bytes': path.stat().st_size}

    def read_json(path):
        path = Path(path)
        record(path)
        check(path.name + ':bounded_json', path.stat().st_size <= 4 * 1024 * 1024)
        value = json.loads(path.read_text(encoding='utf-8'))
        check(path.name + ':object', isinstance(value, dict))
        return value

    def strings(value, name):
        a = np.asarray(value)
        check(name + ':string_vector', a.ndim == 1 and a.dtype.kind in 'US')
        out = a.tolist()
        check(name + ':nonempty_strings', all(isinstance(x, str) and x for x in out))
        return out

    def axes(raw_ids, raw_labels, expected_ids, name, *, exact_ids=True):
        ids = strings(raw_ids, name + ':ids')
        labels = strings(raw_labels, name + ':labels')
        check(name + ':unique_complete_ids', len(ids) == len(expected_ids) and
              len(set(ids)) == len(ids) and set(ids) == set(expected_ids))
        if exact_ids:
            check(name + ':exact_id_order', ids == expected_ids)
        check(name + ':canonical_label_set', len(labels) == K and len(set(labels)) == K and set(labels) == set(LABELS))
        row_map = {x: i for i, x in enumerate(ids)}
        return np.asarray([row_map[x] for x in expected_ids]), [labels.index(x) for x in LABELS], labels

    def matrix(a, n, name, probability=False):
        a = np.asarray(a)
        check(name + ':real_N_by_18', a.shape == (n, K) and a.dtype.kind in 'fiu')
        check(name + ':finite', np.isfinite(a).all())
        if probability:
            check(name + ':bounded_probabilities', (a >= 0).all() and (a <= 1 + 1e-6).all())
            close(name + ':normalized_rows', a.sum(1), np.ones(n), atol=1e-6, rtol=0)
        return a

    def npz(path, expected_ids, fields, probabilities=(), *, exact_ids=True):
        path = Path(path)
        record(path)
        with np.load(path, allow_pickle=False) as z:
            check(path.name + ':required_keys', {'ids', 'labels', *fields}.issubset(z.files))
            rows, columns, labels = axes(z['ids'], z['labels'], expected_ids, path.name, exact_ids=exact_ids)
            values = {}
            for key in fields:
                raw = matrix(z[key], len(expected_ids), path.name + ':' + key, key in probabilities)
                values[key] = raw[rows][:, columns]
        return values, labels

    def frame(path, name, expected_columns):
        check(name + ':exists', Path(path).is_file())
        value = pd.read_csv(path, dtype=str, keep_default_na=False)
        check(name + ':columns', value.columns.tolist() == expected_columns)
        check(name + ':unique_nonempty_ids', value.id.is_unique and value.id.map(lambda x: isinstance(x, str) and bool(x)).all())
        return value

    try:
        test = frame(test_path, 'official_test', ['id', 'text'])
        ids = test.id.tolist()
        n = len(ids)
        check('official_test:nonempty', n > 0)
        check('preview_mode:known_boolean', isinstance(namespace, dict) and type(namespace.get('preview')) is bool)
        public_preview = namespace['preview']
        check('preview_mode:matches_exact_source_fixture', public_preview == fixture_matches(test))
        report['test_sha256'] = sha(test_path)
        report['rows'] = n
        report['source_preview_flag'] = public_preview
        record(work / 'submission.parquet')
        result = reader(work / 'submission.parquet')
        check('submission:dataframe', isinstance(result, pd.DataFrame))
        check('submission:columns', result.columns.tolist() == ['id', 'model'])
        check('submission:no_nulls', not result.isna().any().any())
        check('submission:string_ids', result.id.map(lambda x: isinstance(x, str)).all())
        check('submission:string_models', result.model.map(lambda x: isinstance(x, str)).all())
        check('submission:unique_complete_ids', result.id.is_unique and len(result) == n and set(result.id) == set(ids))
        check('submission:exact_id_order', result.id.tolist() == ids)
        check('submission:allowed_classes', result.model.isin(LABELS).all())

        if public_preview:
            check('format_preview:three_rows', n == 3)
            check('format_preview:published_constant_labels', result.model.tolist() == ['claude-fable-5'] * 3)
            mode = read_json(work / 'submission.mode.json')
            check('format_preview:mode', mode.get('mode') == 'exact_public_format_preview')
            check('format_preview:row_count', mode.get('rows') == 3)
            check('format_preview:no_models_loaded', mode.get('model_weights_loaded') is False)
            check('format_preview:no_hidden_runtime_verified', mode.get('hidden_runtime_verified') is False)
            check('format_preview:other_inputs_use_full_inference', mode.get('all_other_inputs') == 'full ensemble inference')
            report.update(status='FORMAT_ONLY_PREVIEW', full_saved_model_inference_evidence_verified=False,
                          hidden_runtime_verified=False, inference_verification='NOT_PERFORMED_PUBLIC_FIXTURE_STUB')
        else:
            # Version-specific flags are checked independently of saved mixture files.
            recipe = namespace.get('RECIPE')
            check('recipe:provided', isinstance(recipe, dict))
            for key, expected in {
                'status': 'FROZEN_ALLROWS_QWEN10_KIRILL10_CANDIDATE',
                'reference_family': 'part_two_complement_atlas_0636',
                'qwen_probability_share': .10, 'qwen_query_fraction': 1.,
                'helper_probability_share': .10, 'complement_share': .15,
                'ATLAS': True, 'ATLAS_share': .10, 'checkpoint_stage': 'final',
                'checkpoint_source_stage': 'official', 'checkpoint_source_update': 300,
                'official_fit_rows': 2699, 'official_holdout_rows': 0,
            }.items():
                check('recipe:' + key, key in recipe and type(recipe[key]) is type(expected) and recipe[key] == expected)
            # JSON numbers 1 and 1.0 are equivalent for fraction, but the actual source stores 1.0.
            record(temp / 'groups.npy')
            groups = np.load(temp / 'groups.npy', allow_pickle=False)
            check('groups:positive_integer_N_vector', groups.shape == (n,) and groups.dtype.kind in 'iu' and
                  (groups > 0).all() and (groups <= n).all())
            npz(temp / 'decoders.npz', ids, ['logits'])
            npz(temp / 'deberta.npz', ids, ['logits'])
            comp_meta = read_json(temp / 'comp.npy.json')
            cr, cc, _ = axes(np.asarray(comp_meta.get('ids')), np.asarray(comp_meta.get('labels')), ids, 'comp')
            check('comp:train_fits_zero', type(comp_meta.get('train_fits')) is int and comp_meta['train_fits'] == 0)
            record(temp / 'comp.npy')
            comp = matrix(np.load(temp / 'comp.npy', allow_pickle=False), n, 'comp_logits')[cr][:, cc]
            check('comp:aligned_finite', np.isfinite(comp).all())
            before, _ = npz(temp / 'before_atlas.npz', ids, ['probabilities'], ['probabilities'])
            before_atlas = np.asarray(before['probabilities'], dtype=np.float64)
            ordered = np.sort(before_atlas, axis=1)
            selected = np.flatnonzero(ordered[:, -1] - ordered[:, -2] < .3) if n >= 50 else np.arange(n)
            expected_atlas_ids = [ids[i] for i in selected]
            after_atlas = before_atlas.copy()
            if len(selected):
                aq = frame(temp / 'atlas-query.csv', 'atlas_query', ['id', 'text'])
                check('atlas_query:exact_selected_ids', aq.id.tolist() == expected_atlas_ids)
                check('atlas_query:text_axis', aq.text.tolist() == test.iloc[selected].text.tolist())
                atlas, _ = npz(temp / 'atlas.npz', expected_atlas_ids, ['probabilities'], ['probabilities'])
                after_atlas[selected] = .9 * after_atlas[selected] + .1 * atlas['probabilities']
            report['atlas_selected_rows'] = len(selected)
            helper, _ = npz(temp / 'night24-helper.npz', ids, ['probabilities', *HELPER_MEMBERS], ['probabilities', *HELPER_MEMBERS])
            evidence, _ = npz(temp / 'night24-blend-evidence.npz', ids,
                              ['reference_before_helper', 'helper', 'probabilities'],
                              ['reference_before_helper', 'helper', 'probabilities'])
            close('ATLAS:selected_90_10_reconstruction', evidence['reference_before_helper'], after_atlas)
            close('Night24:helper_evidence', evidence['helper'], helper['probabilities'])
            expected_helper_mix = .9 * after_atlas + .1 * helper['probabilities']
            close('Night24:90_10_reconstruction', evidence['probabilities'], expected_helper_mix)
            ref, _ = npz(temp / 'reference_logits.npz', ids, ['logits'])
            close('reference:log_helper_mix', ref['logits'], np.log(np.maximum(evidence['probabilities'], 1e-300)))
            reference = softmax(ref['logits'])
            # Both query CSVs are private, but their row axes are independently reconciled.
            for filename in ['query.csv', 'qwen-query.csv']:
                q = frame(temp / filename, filename, ['id', 'text'])
                check(filename + ':exact_ids', q.id.tolist() == ids)
                check(filename + ':exact_text_axis', q.text.tolist() == test.text.tolist())
            qwen, native_labels = npz(temp / 'qwen.npz', ids, ['logits'])
            metadata = read_json(temp / 'qwen.npz.json')
            check('Qwen:metadata_rows', metadata.get('rows') == n)
            check('Qwen:no_truncation', metadata.get('truncation') is False)
            check('Qwen:all_64_layers', metadata.get('decoder_layers') == 64)
            check('Qwen:last_eight_adapters', metadata.get('adapter_layers') == list(range(56, 64)))
            seconds = metadata.get('seconds')
            check('Qwen:positive_finite_seconds', type(seconds) in (int, float) and np.isfinite(seconds) and seconds > 0)
            peaks = metadata.get('peak_reserved_bytes')
            check('Qwen:two_positive_gpu_peaks', isinstance(peaks, list) and len(peaks) == 2 and
                  all(type(x) is int and x > 0 for x in peaks))
            final = read_json(final_metadata_path)
            for key, expected in {'stage': 'official', 'update': 300, 'official_fit_rows': 2699,
                                  'official_holdout_rows': 0, 'native_decoder_layers': 64,
                                  'executed_decoder_layers': 64, 'text_truncation': False,
                                  'inference_max_length': None}.items():
                check('checkpoint:' + key, key in final and type(final[key]) is type(expected) and final[key] == expected)
            check('checkpoint:native_labels_match_worker', final.get('labels') == native_labels)
            mix, _ = npz(temp / 'qwen_blend_probabilities.npz', ids, ['reference', 'probabilities'], ['reference', 'probabilities'])
            with np.load(temp / 'qwen_blend_probabilities.npz', allow_pickle=False) as z:
                check('Qwen:selection_key', 'selected_ids' in z.files)
                chosen = strings(z['selected_ids'], 'Qwen:selected_ids')
                check('Qwen:all_row_exact_selection', chosen == ids)
            close('Qwen:reference_evidence', mix['reference'], reference)
            expected_mixed = .9 * reference + .1 * softmax(qwen['logits'])
            close('Qwen:all_row_90_10_reconstruction', mix['probabilities'], expected_mixed)
            balanced = source_sinkhorn(mix['probabilities'])
            matrix(balanced, n, 'final_balanced', probability=True)
            expected_labels = np.asarray(LABELS)[balanced.argmax(1)].tolist()
            check('submission:reconstructed_argmax', result.model.tolist() == expected_labels)
            report['final_balanced_array_sha256'] = hashlib.sha256(balanced.astype('<f8',copy=False).tobytes()).hexdigest()
            report.update(status='PASS_SAVED_OUTPUT_CONTRACT', full_saved_model_inference_evidence_verified=True,
                          hidden_runtime_verified=False,
                          inference_verification='SAVED_FULL_INFERENCE_ARTIFACTS_RECONCILED',
                          final_balance_iterations=300 if n >= 50 else 0,
                          qwen_worker_runtime_seconds=float(seconds),
                          qwen_peak_reserved_bytes=peaks,
                          nominal_mixture={'reference_after_ATLAS': .81, 'Night24_helper': .09, 'Qwen': .10},
                          native_qwen_label_order_matches_final_checkpoint=True)
    except Exception as exc:
        report['status'] = 'FAILED_OUTPUT_VALIDATION'
        report['error_type'] = type(exc).__name__
        report['failed_check'] = str(exc) if isinstance(exc, OutputValidationError) else 'artifact_read_or_schema_error'
        raise
    finally:
        report['check_count'] = len(report['checks'])
        report['all_checks_pass'] = bool(report['checks']) and all(x['pass'] for x in report['checks']) and report['status'] != 'FAILED_OUTPUT_VALIDATION'
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        print('STAGE11_OUTPUT_VALIDATION_JSON_BEGIN')
        print(json.dumps(report, indent=2, allow_nan=False))
        print('STAGE11_OUTPUT_VALIDATION_JSON_END')
    return report


def self_test():
    """Synthetic artifact fixtures; no source/model execution or original passages."""
    global PUBLIC_FIXTURE
    import contextlib
    import io
    checks = {}
    original_fixture = list(PUBLIC_FIXTURE)

    def exercise(n=60, atlas='mixed', native_reverse=True, float32_helper=False):
        td = tempfile.TemporaryDirectory()
        root = Path(td.name); work = root / 'work'; tmp = root / 'temp'
        work.mkdir(); tmp.mkdir()
        ids = [f'synthetic-{i:04d}' for i in range(n)]
        texts = [f'Synthetic fixture passage {i}.' for i in range(n)]
        test = pd.DataFrame({'id': ids, 'text': texts}); test_path = root / 'test.csv'
        test.to_csv(test_path, index=False)
        rng = np.random.default_rng(381)
        before = np.tile(np.full(K, .02 / (K - 1)), (n, 1)); before[:, 0] = .98
        if atlas == 'all': before[:] = 1 / K
        elif atlas == 'mixed': before[::2] = 1 / K
        elif atlas != 'empty': raise ValueError(atlas)
        # Below50 source always selects every row, even with high confidence.
        rows = np.flatnonzero(np.sort(before, axis=1)[:, -1] - np.sort(before, axis=1)[:, -2] < .3) if n >= 50 else np.arange(n)
        def save(name, **values): np.savez_compressed(tmp / name, ids=np.asarray(ids), labels=np.asarray(LABELS), **values)
        np.save(tmp / 'groups.npy', np.ones(n, dtype=np.int64), allow_pickle=False)
        save('decoders.npz', logits=rng.normal(size=(n, K)))
        save('deberta.npz', logits=rng.normal(size=(n, K)))
        comp_labels = list(reversed(LABELS)); np.save(tmp / 'comp.npy', rng.normal(size=(n, K)), allow_pickle=False)
        (tmp / 'comp.npy.json').write_text(json.dumps({'ids': ids, 'labels': comp_labels, 'train_fits': 0}))
        save('before_atlas.npz', probabilities=before)
        after = before.copy()
        if len(rows):
            ap = softmax(rng.normal(size=(len(rows), K)))
            np.savez_compressed(tmp / 'atlas.npz', ids=np.asarray(ids)[rows], labels=np.asarray(LABELS), probabilities=ap)
            test.iloc[rows].to_csv(tmp / 'atlas-query.csv', index=False)
            after[rows] = .9 * after[rows] + .1 * ap
        helper = softmax(rng.normal(size=(n, K)))
        if float32_helper: helper = helper.astype(np.float32)
        save('night24-helper.npz', probabilities=helper, **{k: helper.copy() for k in HELPER_MEMBERS})
        hp = .9 * after + .1 * helper
        save('night24-blend-evidence.npz', reference_before_helper=after, helper=helper, probabilities=hp)
        ref = np.log(np.maximum(hp, 1e-300)); save('reference_logits.npz', logits=ref)
        native = list(reversed(LABELS)) if native_reverse else LABELS.copy()
        qlogits = rng.normal(size=(n, K))
        np.savez_compressed(tmp / 'qwen.npz', ids=np.asarray(ids), labels=np.asarray(native), logits=qlogits)
        (tmp / 'qwen.npz.json').write_text(json.dumps({'rows': n, 'seconds': 12.3, 'truncation': False, 'decoder_layers': 64, 'adapter_layers': list(range(56, 64)), 'peak_reserved_bytes': [1000, 2000]}))
        final = {'stage': 'official', 'update': 300, 'official_fit_rows': 2699, 'official_holdout_rows': 0,
                 'native_decoder_layers': 64, 'executed_decoder_layers': 64, 'text_truncation': False,
                 'inference_max_length': None, 'labels': native}
        metadata = root / 'final.json'; metadata.write_text(json.dumps(final))
        aligned_q = qlogits[:, [native.index(x) for x in LABELS]]
        reference = softmax(ref); mixed = .9 * reference + .1 * softmax(aligned_q)
        save('qwen_blend_probabilities.npz', reference=reference, probabilities=mixed, selected_ids=np.asarray(ids))
        for name in ['query.csv', 'qwen-query.csv']: test.to_csv(tmp / name, index=False)
        prediction = pd.DataFrame({'id': ids, 'model': np.asarray(LABELS)[source_sinkhorn(mixed).argmax(1)]})
        (work / 'submission.parquet').write_bytes(b'synthetic-reader-boundary')
        recipe = {'status': 'FROZEN_ALLROWS_QWEN10_KIRILL10_CANDIDATE', 'reference_family': 'part_two_complement_atlas_0636', 'qwen_probability_share': .1, 'qwen_query_fraction': 1., 'helper_probability_share': .1, 'complement_share': .15, 'ATLAS': True, 'ATLAS_share': .1, 'checkpoint_stage': 'final', 'checkpoint_source_stage': 'official', 'checkpoint_source_update': 300, 'official_fit_rows': 2699, 'official_holdout_rows': 0}
        kwargs = dict(work_dir=work, test_path=test_path, temporary_dir=tmp, namespace={'preview': False, 'RECIPE': recipe}, final_metadata_path=metadata, parquet_reader=lambda p: prediction.copy())
        return td, root, work, tmp, prediction, kwargs

    def run(name, mutate=None, *, reject=False, **fixture_args):
        td, root, work, tmp, prediction, kwargs = exercise(**fixture_args)
        try:
            if mutate: mutate(root, work, tmp, prediction, kwargs)
            with contextlib.redirect_stdout(io.StringIO()):
                try: result = validate_output(**kwargs)
                except (ValueError, KeyError, OSError, TypeError): checks[name] = bool(reject)
                else: checks[name] = not reject and result['all_checks_pass'] and result['status'] == 'PASS_SAVED_OUTPUT_CONTRACT'
        finally: td.cleanup()

    def change_npz(path, mutate):
        with np.load(path, allow_pickle=False) as z: data = {k: z[k].copy() for k in z.files}
        mutate(data); np.savez_compressed(path, **data)

    run('full_hidden_artifact_chain_and_native_permutation')
    run('empty_ATLAS_selection_valid', atlas='empty')
    run('all_ATLAS_selection_valid', atlas='all')
    run('below50_all_ATLAS_and_row_normalization', n=12, atlas='empty')
    run('float32_helper_arithmetic_valid', float32_helper=True)
    run('reject_missing_member_file', lambda r,w,t,p,k:(t/'deberta.npz').unlink(), reject=True)
    run('reject_wrong_submission_ids', lambda r,w,t,p,k:p.__setitem__('id',p.id.tolist()[::-1]), reject=True)
    run('reject_wrong_submission_predictions', lambda r,w,t,p,k:p.__setitem__('model',['glm-5.3']*len(p)), reject=True)
    run('reject_unknown_worker_label', lambda r,w,t,p,k:change_npz(t/'qwen.npz',lambda d:d.__setitem__('labels',np.asarray(['unknown']+LABELS[1:]))), reject=True)
    run('reject_duplicate_worker_ids', lambda r,w,t,p,k:change_npz(t/'qwen.npz',lambda d:d.__setitem__('ids',np.asarray(['same']*len(p)))), reject=True)
    run('reject_reordered_worker_ids', lambda r,w,t,p,k:change_npz(t/'qwen.npz',lambda d:d.__setitem__('ids',d['ids'][::-1])), reject=True)
    run('reject_nonfinite_member_logits', lambda r,w,t,p,k:change_npz(t/'decoders.npz',lambda d:d['logits'].__setitem__((0,0),np.nan)), reject=True)
    run('reject_wrong_probability_shape', lambda r,w,t,p,k:change_npz(t/'before_atlas.npz',lambda d:d.__setitem__('probabilities',d['probabilities'][:,:17])), reject=True)
    run('reject_wrong_ATLAS_selection', lambda r,w,t,p,k:pd.read_csv(t/'atlas-query.csv',dtype=str).iloc[::-1].to_csv(t/'atlas-query.csv',index=False), reject=True)
    run('reject_wrong_helper_mixture', lambda r,w,t,p,k:change_npz(t/'night24-blend-evidence.npz',lambda d:d.__setitem__('probabilities',np.roll(d['probabilities'],1,axis=1))), reject=True)
    run('reject_wrong_Qwen_mixture', lambda r,w,t,p,k:change_npz(t/'qwen_blend_probabilities.npz',lambda d:d.__setitem__('probabilities',d['reference'].copy())), reject=True)
    run('reject_partial_Qwen_coverage', lambda r,w,t,p,k:change_npz(t/'qwen_blend_probabilities.npz',lambda d:d.__setitem__('selected_ids',d['selected_ids'][:-1])), reject=True)
    run('reject_metadata_truncation', lambda r,w,t,p,k:(t/'qwen.npz.json').write_text(json.dumps({'rows':len(p),'seconds':1,'truncation':True,'decoder_layers':64,'adapter_layers':list(range(56,64)),'peak_reserved_bytes':[1,2]})), reject=True)
    run('reject_metadata_single_GPU', lambda r,w,t,p,k:(t/'qwen.npz.json').write_text(json.dumps({'rows':len(p),'seconds':1,'truncation':False,'decoder_layers':64,'adapter_layers':list(range(56,64)),'peak_reserved_bytes':[1]})), reject=True)
    run('reject_unknown_preview_flag', lambda r,w,t,p,k:k['namespace'].__setitem__('preview',None), reject=True)
    run('reject_false_preview_claim_on_full_rows', lambda r,w,t,p,k:k['namespace'].__setitem__('preview',True), reject=True)
    # A three-row synthetic fixture substitutes only the known hash fixture in this test.
    try:
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); ids=['synthetic-a','synthetic-b','synthetic-c']; texts=['short a','short b','short c']
            frame=pd.DataFrame({'id':ids,'text':texts});frame.to_csv(root/'test.csv',index=False)
            PUBLIC_FIXTURE=[(i,len(t.encode()),hashlib.blake2s(t.encode(),digest_size=16).hexdigest()) for i,t in zip(ids,texts)]
            (root/'submission.parquet').write_bytes(b'synthetic-reader-boundary')
            mode={'mode':'exact_public_format_preview','rows':3,'model_weights_loaded':False,'hidden_runtime_verified':False,'all_other_inputs':'full ensemble inference'}
            (root/'submission.mode.json').write_text(json.dumps(mode))
            pred=pd.DataFrame({'id':ids,'model':['claude-fable-5']*3})
            kwargs=dict(work_dir=root,test_path=root/'test.csv',namespace={'preview':True},parquet_reader=lambda p:pred.copy())
            with contextlib.redirect_stdout(io.StringIO()): report=validate_output(**kwargs)
            checks['format_only_preview_no_model_or_accuracy_claim']=report['status']=='FORMAT_ONLY_PREVIEW' and not report['full_saved_model_inference_evidence_verified'] and not report['accuracy_validation']
            mode['model_weights_loaded']=True;(root/'submission.mode.json').write_text(json.dumps(mode))
            with contextlib.redirect_stdout(io.StringIO()):
                try:validate_output(**kwargs)
                except OutputValidationError:checks['reject_preview_model_loaded_claim']=True
                else:checks['reject_preview_model_loaded_claim']=False
    finally: PUBLIC_FIXTURE=original_fixture
    report={'status':'PASS' if all(checks.values()) else 'FAIL','passed':sum(checks.values()),'total':len(checks),'checks':checks,'source_or_model_executed':False,'parquet_boundary_in_local_fixtures':'synthetic reader; production notebook always uses real pandas.read_parquet','synthetic_public_fixture_substitution_for_preview_test_only':True}
    if not all(checks.values()):raise AssertionError(json.dumps(report))
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--self-test',action='store_true');parser.add_argument('--work',default='/kaggle/working');parser.add_argument('--test');parser.add_argument('--temporary',default=DEFAULT_TEMP);parser.add_argument('--preview',action='store_true');args=parser.parse_args()
    if args.self_test:print(json.dumps(self_test(),indent=2))
    else:validate_output(args.work,args.test,temporary_dir=args.temporary,namespace={'preview':args.preview})
