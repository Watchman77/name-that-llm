"""No-fit synthetic checks for Stage06E cache consumption and frozen arithmetic."""
import ast
import importlib.util
import json
import tempfile
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rejected(function, label):
    try:
        function()
    except (AssertionError, ValueError, KeyError):
        return label
    raise AssertionError(('Expected fail closed', label))


def wrapped_reference(source, start, end, args):
    """Execute the archived D arithmetic statements, never its fitted runner."""
    block = source[source.index(start):source.index(end, source.index(start))]
    block = '\n'.join(line[8:] for line in block.splitlines())
    function_source = 'def reference(' + ', '.join(args) + '):\n' + '\n'.join('    ' + line for line in block.splitlines())
    function_source += '\n    return locals()\n'
    namespace = dict(np=np)
    exec(compile(function_source, '<archived-D-arithmetic>', 'exec'), namespace)
    return namespace['reference'], namespace


def main():
    consumer = load('stage06e_cached_fixture_subject', HERE / 'stage06e_cached_ensemble.py')
    old = load('stage06d_archived_fixture_subject', HERE / 'stage06d_fixed_ensemble.py')
    old_source = (HERE / 'stage06d_fixed_ensemble.py').read_text()
    assert consumer.digest(HERE / 'stage06d_fixed_ensemble.py') == consumer.D_SHA256
    results = []
    # Static preparation retains D audit arithmetic; consumer forbids model execution.
    for name in ('stage06e_cached_ensemble.py', 'stage06e_static_members.py'):
        tree = ast.parse((HERE / name).read_text())
        assert not [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute) and node.func.attr in ('fit', 'fit_transform')]
    consumer_tree = ast.parse((HERE / 'stage06e_cached_ensemble.py').read_text())
    forbidden = {'fit', 'fit_transform', 'predict_proba', 'decision_function', 'transform', 'split',
                 'matmul', 'dot', 'linkage', 'fcluster', 'partition_training', 'contexts', 'centered'}
    assert not [node for node in ast.walk(consumer_tree) if isinstance(node, ast.Call)
                and ((isinstance(node.func, ast.Attribute) and node.func.attr in forbidden)
                     or (isinstance(node.func, ast.Name) and node.func.id in forbidden))]
    assert not [node for node in ast.walk(consumer_tree) if isinstance(node, ast.MatMult)]
    results.append('AST: consumer has no fit, estimator inference, affine readout or group reconstruction; static audit has no fit')
    rng = np.random.default_rng(90210)
    for rows in (1, 2, 3, 18, 90):
        values = rng.normal(size=(rows, 18)) * 20
        np.testing.assert_array_equal(consumer.stage06e_balance(values), old.stage06d_balance(values))
    results.append('Exact D balancing for singleton, two-row and multirow evidence')
    n = 90
    y = np.tile(np.arange(18), 5)
    folds = np.repeat(np.arange(1, 6), 18)
    groups = np.repeat([f'p{i:03d}' for i in range(5)], 18)
    ids = np.asarray([f'i{i:04d}' for i in range(n)])
    def probability(shape):
        value = np.exp(rng.normal(size=shape))
        return value / value.sum(-1, keepdims=True)
    plain_probability = probability((3, n, 18))
    canonical = dict(ids=ids, y=y, fold=folds, groups=groups,
                     plain_probability=plain_probability,
                     plain_log_probability=np.log(np.clip(plain_probability, 1e-9, 1)),
                     eligible_inferred=rng.random((3, n)) > .3,
                     eligible_oracle=rng.random((3, n)) > .4,
                     global_batches=np.stack([folds * 1000000 + np.tile(np.arange(18) // size, 5)
                                              for size in (1, 3, 6)]))
    canonical['batches'] = canonical['global_batches'] - folds[None] * 1000000
    scores = {}
    for head in consumer.HEADS:
        active = canonical['eligible_oracle' if head.startswith('oracle_') else 'eligible_inferred']
        p = probability((3, n, 18)); p[~active] = 0
        logp = np.zeros_like(p); logp[active] = np.log(np.clip(p[active], 1e-9, 1))
        evidence = .05 * canonical['plain_log_probability'].copy()
        evidence[active] += .25 * logp[active]
        pred = evidence.argmax(2)
        canonical['centered_probability_' + head] = p
        canonical['centered_log_probability_' + head] = logp
        canonical['component_log_evidence_' + head] = evidence
        canonical['prediction_' + head] = pred
        scores[head] = [float(consumer.balanced_accuracy_score(y, row)) for row in pred]
    assert consumer._replay_components(canonical, scores) == scores
    corrupt = {**canonical, 'centered_log_probability_control': canonical['centered_log_probability_control'].copy()}
    inactive = np.argwhere(~canonical['eligible_inferred'])[0]
    corrupt['centered_log_probability_control'][*inactive, 0] = .1
    rejected(lambda: consumer._replay_components(corrupt, scores), 'inactive contribution corruption')
    corrupt = {**canonical, 'component_log_evidence_control': canonical['component_log_evidence_control'].copy()}
    corrupt['component_log_evidence_control'][0, 0, 0] += 1e-9
    rejected(lambda: consumer._replay_components(corrupt, scores), 'same-run component evidence corruption')
    results.append('All twelve saved component arrays replay exactly; inactive or evidence corruption rejected')
    static = {'probability_' + member: probability((n, 18)) for member in consumer.MEMBERS}
    static.update({'fold_' + member: folds.copy() for member in consumer.MEMBERS})
    # A differing member fold map makes the balanced endpoint diagnostic exposure explicit.
    static['fold_lora2'] = np.tile(np.arange(5), 18)
    fused = consumer._fixed_fusion(canonical, static)
    reference, namespace = wrapped_reference(old_source, '        static_log =', '        # 1-response grouping',
        ('component_probability', 'component_folds', 'weights', 'n', 'deb_log', 'eligible', 'batch_id', 'our_folds', 'scenarios', 'stage06d_balance'))
    namespace['stage06d_balance'] = old.stage06d_balance
    deb_log = {'plain': canonical['plain_log_probability'],
               'control': canonical['centered_log_probability_control'],
               'candidate': canonical['centered_log_probability_candidate']}
    actual = reference(fused['component_probability'], fused['component_folds'], consumer.WEIGHTS, n,
                       deb_log, canonical['eligible_inferred'], canonical['global_batches'], folds,
                       consumer.SCENARIOS, old.stage06d_balance)
    for name in ('evidence', 'raw_posterior', 'posterior', 'prediction'):
        for key, value in fused[name].items():
            np.testing.assert_array_equal(value, actual[name][key])
    assert fused['balancing_records'] == actual['balancing_records']
    assert sum(item['rows_with_cross_fold_peer'] for item in fused['balancing_records']) > 0
    results.append('Control/candidate fusion, raw/balanced arrays and contamination counts exactly match archived D statements')
    prediction = fused['prediction']
    score_summary = {key: [float(consumer.balanced_accuracy_score(y, part)) for part in value] for key, value in prediction.items()}
    reference, namespace = wrapped_reference(old_source, '        intervals = {}', '        # Persist all source-normalized',
                                              ('y', 'groups', 'prediction', 'score_summary'))
    ref = reference(y, groups, prediction, score_summary)['intervals']
    assert consumer._paired_intervals(y, groups, prediction, score_summary) == ref
    results.append('Exact archived D paired prompt/class bootstrap, 3000 draws and fixed seed')
    # Full official row/class/fold geometry with entirely synthetic predictions.
    # This verifies the production canonical gate without fitting or inference.
    train_path = ROOT.parent / 'name-that-llm/data/train.csv'
    assert consumer.digest(train_path) == consumer.TRAIN_SHA256
    train = consumer.pd.read_csv(train_path, dtype=str)
    large_ids = train.id.to_numpy(dtype=str)
    large_groups = train.prompt_id.to_numpy(dtype=str)
    large_classes = np.sort(train.model.unique()).astype(str)
    large_y = np.searchsorted(large_classes, train.model.to_numpy())
    fold_map = {'p000': 4, 'p001': 3, 'p002': 5}
    fold_map.update({f'p{i:03d}': 5 - i % 5 for i in range(3, 150)})
    large_fold = np.asarray([fold_map[group] for group in large_groups], dtype=np.int64)
    m = len(large_ids)
    large_p = probability((3, m, 18))
    full = dict(ids=large_ids, classes=large_classes, groups=large_groups, y=large_y, fold=large_fold,
                scenarios=np.asarray((1, 3, 6)), plain_probability=large_p,
                plain_log_probability=np.log(np.clip(large_p, 1e-9, 1)),
                plain_coverage=np.ones((3,m), dtype=np.int64), context_coverage=np.ones((3,m), dtype=np.int64),
                batches=np.tile(np.arange(m, dtype=np.int64), (3,1)))
    full['global_batches'] = full['batches'] + large_fold[None] * 1000000
    for mode in ('inferred', 'oracle'):
        full[mode] = full['batches'].copy()
        full['cluster_size_' + mode] = np.ones((3,m), dtype=np.int64)
        full['eligible_' + mode] = np.zeros((3,m), dtype=bool)
    full_scores = {}
    for head in consumer.HEADS:
        full['centered_probability_' + head] = np.zeros((3,m,18))
        full['centered_log_probability_' + head] = np.zeros((3,m,18))
        full['component_log_evidence_' + head] = .05 * full['plain_log_probability'].copy()
        full['prediction_' + head] = full['component_log_evidence_' + head].argmax(2)
        full['component_coverage_' + head] = np.ones((3,m), dtype=np.int64)
        full_scores[head] = [float(consumer.balanced_accuracy_score(large_y, row)) for row in full['prediction_' + head]]
    for number in range(5):
        for head in ('plain', 'control', 'candidate'):
            prefix = f'head_f{number}_{head}__'
            full.update({prefix+'scaler_mean': np.zeros(2048), prefix+'scaler_scale': np.ones(2048),
                         prefix+'scaler_var': np.ones(2048), prefix+'scaler_n_samples_seen': np.asarray(1),
                         prefix+'clf_coef': np.zeros((18,2048)), prefix+'clf_intercept': np.zeros(18),
                         prefix+'clf_classes': np.arange(18), prefix+'clf_n_iter': np.asarray([1])})
    source_bindings = dict(producer='a'*64, capture_helper='b'*64, plan='c'*64, freeze='d'*64,
                           original_C=consumer.ORIGINAL_C_SHA256)
    full_manifest = dict(complete=True, plan_sha256=source_bindings['plan'], freeze_sha256=source_bindings['freeze'],
                         input_version_ids=[170484,358745836,358831649,358911769,358928013,358948428,359405443],
                         original_source_sha256=consumer.ORIGINAL_C_SHA256,
                         source_sha256=source_bindings['producer'], capture_helper_sha256=source_bindings['capture_helper'],
                         input_hashes=dict(train_sha256=consumer.TRAIN_SHA256, deb_cache_sha256=consumer.DEB_SHA256,
                                           glm_cache_sha256=consumer.GLM_SHA256),
                         runtime=dict(python='3.13.15', packages=consumer.EXPECTED_PACKAGES,
                                      threadpools=[dict(num_threads=2, user_api='blas')],
                                      environment={key:'2' for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}), warning_records=[], component_scores=full_scores)
    full_gate = dict(fitted_heads=15, component_predictions_exact=True,
                     component_scores_atol_1e_12=True, warnings_non_deprecation=0)
    assert consumer._validate_canonical(full, full_manifest, full_gate, source_bindings) == full_scores
    damaged = {**full, 'context_coverage': full['context_coverage'].copy()}
    damaged['context_coverage'][0,0] = 0
    rejected(lambda: consumer._validate_canonical(damaged, full_manifest, full_gate, source_bindings), 'missing context row')
    damaged_manifest = {**full_manifest, 'plan_sha256': 'e'*64}
    rejected(lambda: consumer._validate_canonical(full, damaged_manifest, full_gate, source_bindings), 'wrong frozen plan')
    damaged_manifest = {**full_manifest, 'runtime': {**full_manifest['runtime'], 'python': '3.99.0'}}
    rejected(lambda: consumer._validate_canonical(full, damaged_manifest, full_gate, source_bindings), 'wrong prefit runtime')
    rejected(lambda: consumer._validate_canonical(full, full_manifest, {**full_gate, 'fitted_heads': 14}, source_bindings), 'missing head fit receipt')
    results.append('Canonical production gate accepts synthetic official geometry and rejects missing coverage/plan/runtime/head count')
    with tempfile.TemporaryDirectory(prefix='stage06e-nofit-fixtures-') as folder:
        root = Path(folder)
        value = dict(ids=ids, classes=np.asarray([f'c{i}' for i in range(18)]),
                     y=y, logs=canonical['plain_log_probability'])
        def save_cache(schema=consumer.SCHEMA):
            cache = root / 'canonical_cpu_evidence.npz'
            np.savez_compressed(cache, **value)
            gate = dict(status='PASS', canonical_cache_sha256=consumer.digest(cache), all_saved_arrays_exact_readback=True)
            consumer.write_json(root / 'cache_gate.json', gate)
            manifest = dict(schema_version=schema, status='PASS', cache_file=cache.name,
                            cache_sha256=consumer.digest(cache), internal_gate_file='cache_gate.json',
                            internal_gate_sha256=consumer.digest(root / 'cache_gate.json'),
                            arrays={key: dict(dtype=str(array.dtype), shape=list(array.shape), sha256=consumer.hashlib.sha256(array.tobytes(order='C')).hexdigest()) for key, array in value.items()})
            consumer.write_json(root / 'cache_manifest.json', manifest)
            return manifest
        manifest = save_cache()
        loaded, _, _, _ = consumer._load_bound_cache(root, 'cache_manifest.json', consumer.SCHEMA)
        for key, array in value.items(): np.testing.assert_array_equal(loaded[key], array)
        (root / 'canonical_cpu_evidence.npz').write_bytes((root / 'canonical_cpu_evidence.npz').read_bytes() + b'corrupt')
        rejected(lambda: consumer._load_bound_cache(root, 'cache_manifest.json', consumer.SCHEMA), 'changed cache SHA')
        manifest = save_cache(); (root / 'cache_gate.json').write_text('{}')
        rejected(lambda: consumer._load_bound_cache(root, 'cache_manifest.json', consumer.SCHEMA), 'changed gate SHA')
        manifest = save_cache(); manifest['arrays']['y']['shape'] = [n+1]
        consumer.write_json(root / 'cache_manifest.json', manifest)
        rejected(lambda: consumer._load_bound_cache(root, 'cache_manifest.json', consumer.SCHEMA), 'changed schema')
        manifest = save_cache(); manifest['arrays']['y']['sha256'] = '0' * 64
        consumer.write_json(root / 'cache_manifest.json', manifest)
        rejected(lambda: consumer._load_bound_cache(root, 'cache_manifest.json', consumer.SCHEMA), 'changed array bytes hash')
        manifest = save_cache(); manifest['cache_file'] = '../escape.npz'
        consumer.write_json(root / 'cache_manifest.json', manifest)
        rejected(lambda: consumer._load_bound_cache(root, 'cache_manifest.json', consumer.SCHEMA), 'path escape')
        value['classes'] = np.asarray(['unsupported-byte-strings'], dtype='S'); save_cache()
        rejected(lambda: consumer._load_bound_cache(root, 'cache_manifest.json', consumer.SCHEMA), 'bytes string array')
        value['classes'] = np.asarray(['unsafe'], dtype=object); save_cache()
        rejected(lambda: consumer._load_bound_cache(root, 'cache_manifest.json', consumer.SCHEMA), 'object array')
        results.append('Bound-cache exact roundtrip plus altered cache/gate/schema/array hash/path/byte strings/object rejection')
        output = root / 'failed-output'
        expected = dict(consumer=consumer.digest(HERE / 'stage06e_cached_ensemble.py'))
        rejected(lambda: consumer.run_cached_ensemble(root, root, expected, output), 'failed input gate')
        status = json.loads((output / 'status.json').read_text())
        assert status['status'] == 'FAILED_PRESERVE_OUTPUTS'
        assert not (output / 'ensemble_result.json').exists()
        rejected(lambda: consumer.run_cached_ensemble(root, root, expected, output), 'refuse existing trial output')
        results.append('Failed cache gate preserves failure and cannot emit result; existing trial output refused')
    report = dict(status='PASS_NOFIT_SYNTHETIC_FIXTURES', fixture_groups=len(results), checks=results,
                  model_fits=0, estimator_inference_calls=0, new_neural_inference=False,
                  actual_Kaggle_cache_evaluation=False,
                  source_sha256={name: consumer.digest(HERE / name) for name in ('stage06e_cached_ensemble.py', 'stage06e_static_members.py')},
                  archived_D_source_sha256=consumer.D_SHA256,
                  limitations=['Synthetic fixtures validate cached arithmetic and fail-closed behavior; no actual attached artifact or model fit was executed.'])
    output = ROOT / 'results/stage_06e/CONSUMER_NOFIT_FIXTURES.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    consumer.write_json(output, report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
