"""Stage 06E pure cache consumer for a fixed available-component comparison.

No fitting, estimator invocation, affine readout, feature extraction, random
batch construction, group recovery or probability inference occurs here.
Existing probabilities and clipped log-probabilities are read directly.
Fusion, endpoints and bootstrap follow frozen Stage 06D without weight search.
The new canonical CPU cache never passes the historical Stage 06D gate.
"""
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

SCHEMA = 'stage06e-canonical-cpu-v1'
STATIC_SCHEMA = 'stage06e-static-members-v1'
ORIGINAL_C_SHA256 = '5c27fcdffde2bcfd512d20fb4edec9bd7133430e79ccd71e31875151d235a23d'
D_SHA256 = 'c6005196a73100676141d18f67eb299c1bf686a547fbb077b91825d2bd2611b3'
TRAIN_SHA256 = '40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707'
DEB_SHA256 = '11eed58762ad128f97457dde9cc4ab03c7932e1b4b04845ec6664c834b1b1dda'
GLM_SHA256 = '2b41585a9482cef825962a8cc096697933c0b1237e8ede0712d1f8ab361d9414'
MEMBERS = ('stylegram', 'lora', 'lora2', 'tokattn', 'tokattn_deb')
HEADS = ('control', 'candidate', 'oracle_control', 'oracle_candidate')
WEIGHTS = {'stylegram': 1.516, 'lora': .177, 'lora2': .113, 'tokattn': .153,
           'tokattn_deb': .185, 'deb_plain': .05, 'deb_center': .25}
SCENARIOS = (1, 3, 6)
EXPECTED_PACKAGES = {'numpy': '2.1.3', 'pandas': '2.3.3', 'scipy': '1.16.3',
                     'scikit-learn': '1.6.1', 'threadpoolctl': '3.6.0'}


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def _load_bound_cache(root, manifest_name, schema):
    """Load one manifest-bound numeric/Unicode cache and its passed gate."""
    root = Path(root)
    manifest_path = root / manifest_name
    manifest = json.loads(manifest_path.read_text())
    assert manifest['schema_version'] == schema
    assert manifest.get('status', 'PASS') == 'PASS'
    # The producer can only nominate local fixed-name output files.
    expected_cache = 'canonical_cpu_evidence.npz' if schema == SCHEMA else 'static_member_probabilities.npz'
    expected_gate = 'cache_gate.json' if schema == SCHEMA else 'static_cache_gate.json'
    assert manifest['cache_file'] == expected_cache
    assert manifest['internal_gate_file'] == expected_gate
    cache_path = root / expected_cache
    gate_path = root / expected_gate
    assert digest(cache_path) == manifest['cache_sha256'], 'Cache file hash mismatch'
    assert digest(gate_path) == manifest['internal_gate_sha256'], 'Cache gate hash mismatch'
    gate = json.loads(gate_path.read_text())
    assert gate['status'] == 'PASS'
    assert gate.get('canonical_cache_sha256', gate.get('cache_sha256')) == manifest['cache_sha256']
    assert gate['all_saved_arrays_exact_readback'] is True
    arrays = {}
    with np.load(cache_path, allow_pickle=False) as saved:
        assert set(saved.files) == set(manifest['arrays']), 'Cache array inventory mismatch'
        for key in saved.files:
            value = saved[key].copy()
            spec = manifest['arrays'][key]
            assert not value.dtype.hasobject and value.dtype.kind in 'biufU', ('Unsafe array', key)
            assert list(value.shape) == spec['shape'] and value.dtype == np.dtype(spec['dtype']), ('Array schema', key)
            if value.dtype.kind in 'f':
                assert np.isfinite(value).all(), ('Nonfinite cache', key)
            assert hashlib.sha256(value.tobytes(order='C')).hexdigest() == spec['sha256'], ('Array byte hash', key)
            arrays[key] = value
    return arrays, manifest, gate, dict(manifest_path=str(manifest_path),
                                     manifest_sha256=digest(manifest_path), cache_path=str(cache_path),
                                     cache_sha256=manifest['cache_sha256'], gate_sha256=digest(gate_path))


def _check_probability_logs(probability, logs, active=None, fold=None, batches=None):
    """Check captured original probabilities; never replace the captured logs."""
    assert probability.shape == logs.shape and probability.ndim == 3
    assert probability.dtype.kind == logs.dtype.kind == 'f'
    assert np.isfinite(probability).all() and np.isfinite(logs).all()
    if active is None:
        active = np.ones(probability.shape[:2], dtype=bool)
    assert active.shape == probability.shape[:2] and active.dtype.kind == 'b'
    assert (probability[active] >= 0).all() and (probability[active] <= 1 + 2e-12).all()
    np.testing.assert_allclose(probability[active].sum(1), 1, rtol=0, atol=2e-12)
    assert np.all(probability[~active] == 0) and np.all(logs[~active] == 0)
    if fold is None:
        np.testing.assert_array_equal(logs[active], np.log(np.clip(probability[active], 1e-9, 1)))
    else:
        # Match C's original ufunc call shapes/order, never run an estimator.
        assert fold.shape == probability.shape[1:2]
        for scenario in range(probability.shape[0]):
            for number in np.unique(fold):
                selected = (fold == number) & active[scenario]
                if batches is None:
                    rows = np.flatnonzero(selected)
                    if len(rows):
                        np.testing.assert_array_equal(logs[scenario, rows], np.log(np.clip(probability[scenario, rows], 1e-9, 1)))
                else:
                    for batch in np.unique(batches[scenario, selected]):
                        rows = np.flatnonzero(selected & (batches[scenario] == batch))
                        np.testing.assert_array_equal(logs[scenario, rows], np.log(np.clip(probability[scenario, rows], 1e-9, 1)))


def _replay_components(arrays, component_scores):
    """Replay saved arithmetic only, preserving the original addition order."""
    y = arrays['y']
    plain = arrays['plain_log_probability']
    _check_probability_logs(arrays['plain_probability'], plain, fold=arrays['fold'], batches=arrays['batches'])
    replay_scores = {}
    for head in HEADS:
        mask = arrays['eligible_oracle' if head.startswith('oracle_') else 'eligible_inferred']
        centered = arrays['centered_log_probability_' + head]
        _check_probability_logs(arrays['centered_probability_' + head], centered, mask, fold=arrays['fold'])
        replay = .05 * plain.copy()
        replay[mask] += .25 * centered[mask]
        np.testing.assert_array_equal(replay, arrays['component_log_evidence_' + head])
        prediction = replay.argmax(2)
        np.testing.assert_array_equal(prediction, arrays['prediction_' + head])
        score = [float(balanced_accuracy_score(y, row)) for row in prediction]
        np.testing.assert_allclose(score, component_scores[head], rtol=0, atol=1e-12)
        # Independently reconcile the full confusion matrices and class recalls.
        for scenario, part in enumerate(prediction):
            matrix = confusion_matrix(y, part, labels=np.arange(plain.shape[2]))
            assert np.all(matrix.sum(1) > 0)
            np.testing.assert_allclose((matrix.diagonal() / matrix.sum(1)).mean(), score[scenario], rtol=0, atol=1e-15)
        replay_scores[head] = score
    return replay_scores


def _validate_canonical(arrays, manifest, gate, expected_sources):
    assert manifest['complete'] is True and gate['fitted_heads'] == 15
    assert manifest['plan_sha256'] == expected_sources['plan']
    assert manifest['freeze_sha256'] == expected_sources['freeze']
    assert manifest['input_version_ids'] == [170484, 358745836, 358831649, 358911769, 358928013, 358948428, 359405443]
    assert gate['component_predictions_exact'] is True
    assert gate['component_scores_atol_1e_12'] is True
    assert gate['warnings_non_deprecation'] == 0
    assert manifest['original_source_sha256'] == ORIGINAL_C_SHA256
    assert expected_sources['original_C'] == ORIGINAL_C_SHA256
    assert manifest['source_sha256'] == expected_sources['producer']
    assert manifest['capture_helper_sha256'] == expected_sources['capture_helper']
    assert manifest['input_hashes'] == dict(train_sha256=TRAIN_SHA256, deb_cache_sha256=DEB_SHA256,
                                            glm_cache_sha256=GLM_SHA256)
    runtime = manifest['runtime']
    assert runtime['python'] == '3.13.15' and runtime['packages'] == EXPECTED_PACKAGES
    assert runtime['threadpools'] and all(item['num_threads'] == 2 for item in runtime['threadpools'])
    assert any(item.get('user_api') == 'blas' for item in runtime['threadpools'])
    assert runtime['environment'] == {key: '2' for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}
    warning_records = manifest['warning_records']
    assert not [row for row in warning_records if row['category'] != 'DeprecationWarning']
    ids, classes, groups = arrays['ids'], arrays['classes'], arrays['groups']
    y, fold = arrays['y'], arrays['fold']
    n = len(ids)
    assert np.all(ids != '') and np.all(classes != '') and np.all(groups != '')
    assert n == 2699 and ids.dtype.kind == classes.dtype.kind == groups.dtype.kind == 'U'
    assert len(np.unique(ids)) == n and len(np.unique(groups)) == 150
    assert classes.shape == (18,) and len(np.unique(classes)) == 18 and np.array_equal(classes, np.sort(classes))
    assert y.shape == fold.shape == groups.shape == (n,)
    assert y.dtype.kind in 'iu' and fold.dtype.kind in 'iu'
    assert np.array_equal(np.unique(y), np.arange(18))
    assert sorted(np.bincount(y).tolist()) == [149] + [150] * 17
    assert np.array_equal(arrays['scenarios'], SCENARIOS)
    assert np.array_equal(np.unique(fold), np.arange(1, 6))
    expected_prompt_folds = {'p000': 4, 'p001': 3, 'p002': 5}
    expected_prompt_folds.update({f'p{i:03d}': 5 - i % 5 for i in range(3, 150)})
    assert all(expected_prompt_folds[g] == int(f) for g, f in zip(groups, fold))
    for number in range(1, 6):
        valid = fold == number
        assert len(np.unique(groups[valid])) == 30 and not set(groups[valid]) & set(groups[~valid])
    for key in ('plain_coverage', 'context_coverage'):
        assert arrays[key].shape == (3, n) and arrays[key].dtype.kind in 'iu' and np.all(arrays[key] == 1)
    for key in ('plain_probability', 'plain_log_probability'):
        assert arrays[key].shape == (3, n, 18)
    for key in ('batches', 'global_batches', 'inferred', 'oracle', 'cluster_size_inferred', 'cluster_size_oracle',
                'eligible_inferred', 'eligible_oracle'):
        assert arrays[key].shape == (3, n)
    assert arrays['batches'].dtype.kind in 'iu' and arrays['global_batches'].dtype.kind in 'iu'
    assert np.all(arrays['batches'] >= 0) and np.all(arrays['global_batches'] >= 0)
    assert np.array_equal(arrays['global_batches'], arrays['batches'] + fold[None] * 1000000)
    for mode in ('inferred', 'oracle'):
        labels = arrays[mode]
        sizes = arrays['cluster_size_' + mode]
        active = arrays['eligible_' + mode]
        assert labels.dtype.kind in 'iu' and (labels >= 0).all()
        assert sizes.dtype.kind in 'iu' and active.dtype.kind == 'b'
        assert np.array_equal(active, (sizes >= 2) & (sizes <= 18))
        for scenario in range(3):
            for number in range(1, 6):
                rows = np.flatnonzero(fold == number)
                local = labels[scenario, rows]
                expected_sizes = np.bincount(local)[local]
                assert np.array_equal(sizes[scenario, rows], expected_sizes)
                for group in np.unique(local):
                    members = rows[local == group]
                    assert len(np.unique(arrays['global_batches'][scenario, members])) == 1
    for head in HEADS:
        assert arrays['prediction_' + head].shape == (3, n) and arrays['prediction_' + head].dtype.kind in 'iu'
        coverage = arrays['component_coverage_' + head]
        assert coverage.shape == (3, n) and coverage.dtype.kind in 'iu' and np.all(coverage == 1)
        assert arrays['centered_probability_' + head].shape == (3, n, 18)
    # Model snapshots are archived audit evidence only, never invoked here.
    for number in range(5):
        for head in ('plain', 'control', 'candidate'):
            prefix = f'head_f{number}_{head}__'
            for suffix in ('scaler_mean', 'scaler_scale', 'scaler_var', 'scaler_n_samples_seen',
                           'clf_coef', 'clf_intercept', 'clf_classes', 'clf_n_iter'):
                assert prefix + suffix in arrays, ('Missing head snapshot', prefix, suffix)
            assert arrays[prefix + 'scaler_mean'].shape == arrays[prefix + 'scaler_scale'].shape == arrays[prefix + 'scaler_var'].shape == (2048,)
            assert arrays[prefix + 'clf_coef'].shape == (18, 2048)
            assert arrays[prefix + 'clf_intercept'].shape == (18,)
            np.testing.assert_array_equal(arrays[prefix + 'clf_classes'], np.arange(18))
    return _replay_components(arrays, manifest['component_scores'])


def _validate_static(arrays, manifest, gate, canonical, expected_sources):
    assert manifest['source_sha256'] == expected_sources['static_preparation']
    assert manifest['original_D_source_sha256'] == D_SHA256
    assert manifest['train_sha256'] == TRAIN_SHA256
    assert tuple(manifest['members']) == MEMBERS
    assert gate['model_fits'] == 0 and gate['new_neural_inference'] is False
    assert gate['existing_public_affine_readout_audit'] is True
    assert len(manifest['input_artifacts']) == 47 and len(manifest['fold_proofs']) == 25
    for key in ('ids', 'classes', 'groups', 'y', 'fold'):
        np.testing.assert_array_equal(arrays[key], canonical[key])
    assert arrays['train_sha256'].item() == TRAIN_SHA256
    n = len(arrays['ids'])
    for member in MEMBERS:
        probability = arrays['probability_' + member]
        member_fold = arrays['fold_' + member]
        assert probability.shape == (n, 18) and np.isfinite(probability).all()
        assert (probability >= 0).all() and (probability <= 1 + 2e-6).all()
        np.testing.assert_allclose(probability.sum(1), 1, rtol=0, atol=2e-6)
        assert member_fold.shape == (n,) and member_fold.dtype.kind in 'iu' and len(np.unique(member_fold)) == 5
        for number in np.unique(member_fold):
            valid = member_fold == number
            assert len(np.unique(arrays['groups'][valid])) == 30
            assert not set(arrays['groups'][valid]) & set(arrays['groups'][~valid])
        np.testing.assert_allclose(float(balanced_accuracy_score(arrays['y'], probability.argmax(1))),
                                   manifest['component_balanced_accuracy'][member], rtol=0, atol=1e-12)
    assert hashlib.sha256(np.ascontiguousarray(arrays['probability_stylegram'], dtype=np.float64).tobytes()).hexdigest() == '5e2e05577b7dc632f38f67d47958d7f3d069e8b758ead4664050f0d984803bad'


def stage06e_balance(log_evidence, alpha=0.5):
    """Exact frozen S02/D recipe, ONE captured batch per invocation."""
    value = np.asarray(log_evidence, dtype=np.float64)
    assert value.ndim == 2 and value.shape[0] and value.shape[1] > 1
    assert np.isfinite(value).all() and alpha == 0.5
    posterior = np.exp(value - value.max(1, keepdims=True))
    posterior /= posterior.sum(1, keepdims=True)
    if alpha > 0 and len(posterior) >= 2:
        target = (1 - alpha) * posterior.sum(0) + alpha * np.full(value.shape[1], len(posterior) / value.shape[1])
        for _ in range(50):
            posterior *= (target / (posterior.sum(0) + 1e-12))[None, :]
            posterior /= posterior.sum(1, keepdims=True) + 1e-12
    assert np.isfinite(posterior).all() and (posterior >= 0).all()
    np.testing.assert_allclose(posterior.sum(1), 1, rtol=0, atol=2e-12)
    return posterior


def _fixed_fusion(canonical, static):
    """Use captured logs directly, with exactly the frozen D arithmetic order."""
    n = len(canonical['ids'])
    component_probability = {key: static['probability_' + key] for key in MEMBERS}
    component_folds = {key: static['fold_' + key] for key in MEMBERS}
    component_folds.update(deb_plain=canonical['fold'], deb_center=canonical['fold'])
    static_log = {key: np.log(np.clip(value, 1e-9, 1)) for key, value in component_probability.items()}
    base_evidence = np.zeros((n, 18), dtype=np.float64)
    for component in ('stylegram', 'lora', 'lora2', 'tokattn', 'tokattn_deb'):
        base_evidence += WEIGHTS[component] * static_log[component]
    evidence, raw_posterior, posterior, prediction = {}, {}, {}, {}
    balancing_records = []
    eligible = canonical['eligible_inferred']
    batch_id = canonical['global_batches']
    our_folds = canonical['fold']
    for head in ('control', 'candidate'):
        evidence[head] = base_evidence[None] + WEIGHTS['deb_plain'] * canonical['plain_log_probability']
        evidence[head] = evidence[head].copy()
        evidence[head][eligible] += WEIGHTS['deb_center'] * canonical['centered_log_probability_' + head][eligible]
        raw_posterior[head] = np.exp(evidence[head] - evidence[head].max(2, keepdims=True))
        raw_posterior[head] /= raw_posterior[head].sum(2, keepdims=True)
        assert np.isfinite(raw_posterior[head]).all()
        np.testing.assert_allclose(raw_posterior[head].sum(2), 1, rtol=0, atol=1e-12)
        posterior[head] = np.empty((3, n, 18), dtype=np.float64)
        for scenario, size in enumerate(SCENARIOS):
            for batch in np.unique(batch_id[scenario]):
                rows = np.flatnonzero(batch_id[scenario] == batch)
                assert len(np.unique(our_folds[rows])) == 1
                posterior[head][scenario, rows] = stage06e_balance(evidence[head][scenario, rows])
                crossing = np.zeros(len(rows), dtype=bool)
                for member_folds in component_folds.values():
                    crossing |= np.any(member_folds[rows, None] != member_folds[rows][None, :], axis=1)
                balancing_records.append(dict(head=head, scenario=size, batch=int(batch), rows=len(rows),
                                              rows_with_cross_fold_peer=int(crossing.sum()),
                                              changed_predictions=int((posterior[head][scenario, rows].argmax(1) != evidence[head][scenario, rows].argmax(1)).sum())))
        prediction['raw_' + head] = evidence[head].argmax(2)
        prediction['balanced_' + head] = posterior[head].argmax(2)
    return dict(evidence=evidence, raw_posterior=raw_posterior, posterior=posterior,
                prediction=prediction, balancing_records=balancing_records,
                component_probability=component_probability, component_folds=component_folds)


def _paired_intervals(y, groups, prediction, score_summary):
    """Exact D bootstrap, seed/draw count and paired prompt/class aggregation."""
    intervals = {}
    for endpoint in ('raw', 'balanced'):
        control_correct = prediction[endpoint + '_control'] == y[None]
        candidate_correct = prediction[endpoint + '_candidate'] == y[None]
        difference = (candidate_correct.astype(float) - control_correct.astype(float)).mean(0)
        unique_prompts = np.unique(groups)
        numerator = np.stack([np.bincount(y[groups == prompt], weights=difference[groups == prompt], minlength=18) for prompt in unique_prompts])
        denominator = np.stack([np.bincount(y[groups == prompt], minlength=18) for prompt in unique_prompts])
        draws = np.random.default_rng(20261017).integers(len(unique_prompts), size=(3000, len(unique_prompts)))
        bootstrap = (numerator[draws].sum(1) / denominator[draws].sum(1).clip(1)).mean(1)
        gains = np.asarray(score_summary[endpoint + '_candidate']) - np.asarray(score_summary[endpoint + '_control'])
        np.testing.assert_allclose((numerator.sum(0) / denominator.sum(0)).mean(), gains.mean(), rtol=0, atol=1e-15)
        intervals[endpoint] = dict(scenario_gains=gains.tolist(), mean_gain=float(gains.mean()),
                                   conditional_prompt_95_interval=np.quantile(bootstrap, [.025, .975]).tolist(),
                                   bootstrap_draws=3000, bootstrap_seed=20261017,
                                   interpretation='Conditional on fixed fits, grouping, batch assignment and reused development prompts.' if endpoint == 'raw' else 'Potentially contaminated diagnostic: peer supervision can cross target prompt; not a clean validation interval.')
    return intervals


def run_cached_ensemble(canonical_root, static_root, expected_sources, output_root=None):
    """Require two frozen passed caches, then produce one fixed development result."""
    started = time.monotonic()
    output_root = Path(output_root or os.environ.get('NTL_STAGE06E_ENSEMBLE_OUTPUT', '/kaggle/working/stage06e_ensemble'))
    assert not output_root.exists() or not any(output_root.iterdir()), 'Refusing to overwrite an existing ensemble trial.'
    output_root.mkdir(parents=True, exist_ok=True)
    assert digest(Path(__file__)) == expected_sources['consumer']
    def status(phase, **extra):
        write_json(output_root / 'status.json', dict(status=phase, elapsed_seconds=time.monotonic() - started,
                  observed_utc=datetime.now(timezone.utc).isoformat(), **extra))
    status('VERIFYING_FROZEN_CACHES')
    try:
        canonical, canonical_manifest, canonical_gate, canonical_receipt = _load_bound_cache(canonical_root, 'cache_manifest.json', SCHEMA)
        static, static_manifest, static_gate, static_receipt = _load_bound_cache(static_root, 'static_cache_manifest.json', STATIC_SCHEMA)
        replay_scores = _validate_canonical(canonical, canonical_manifest, canonical_gate, expected_sources)
        _validate_static(static, static_manifest, static_gate, canonical, expected_sources)
        cache_receipt = dict(status='PASS', canonical=canonical_receipt, static=static_receipt,
                             canonical_component_scores=replay_scores, all_12_same_run_component_arrays_exact=True,
                             historical_Stage06D_reproduction_passed=False,
                             model_fits=0, model_inference_calls=0, group_recovery_calls=0)
        write_json(output_root / 'cache_consumption_gate.json', cache_receipt)
        print('STAGE06E_CACHE_CONSUMPTION_GATE_PASSED', json.dumps(cache_receipt), flush=True)
        status('FIXED_CACHED_FUSION')
        fused = _fixed_fusion(canonical, static)
        prediction = fused['prediction']
        y, groups, ids, classes, our_folds = (canonical[key] for key in ('y', 'groups', 'ids', 'classes', 'fold'))
        n = len(ids)
        score_summary = {key: [float(balanced_accuracy_score(y, part)) for part in value] for key, value in prediction.items()}
        per_class, per_fold, paired_counts, confusion_paths = [], [], [], []
        for key, value in prediction.items():
            for scenario, size in enumerate(SCENARIOS):
                matrix = confusion_matrix(y, value[scenario], labels=np.arange(18))
                support = matrix.sum(1); recall = matrix.diagonal() / support
                np.testing.assert_allclose(recall.mean(), score_summary[key][scenario], rtol=0, atol=1e-15)
                name = f'{key}_scenario{size}_confusion.csv'
                pd.DataFrame(matrix, index=classes, columns=classes).to_csv(output_root / name)
                confusion_paths.append(name)
                for number, label in enumerate(classes):
                    per_class.append(dict(endpoint=key, scenario=size, model=label, support=int(support[number]), recall=float(recall[number])))
                for fold in range(1, 6):
                    rows = np.flatnonzero(our_folds == fold)
                    per_fold.append(dict(endpoint=key, scenario=size, fold=fold, rows=len(rows),
                                         balanced_accuracy=float(balanced_accuracy_score(y[rows], value[scenario, rows]))))
        eligible = canonical['eligible_inferred']
        for endpoint in ('raw', 'balanced'):
            control_correct = prediction[endpoint + '_control'] == y[None]
            candidate_correct = prediction[endpoint + '_candidate'] == y[None]
            for scenario, size in enumerate(SCENARIOS):
                corrected = int((~control_correct[scenario] & candidate_correct[scenario]).sum())
                broken = int((control_correct[scenario] & ~candidate_correct[scenario]).sum())
                changed = int((prediction[endpoint + '_control'][scenario] != prediction[endpoint + '_candidate'][scenario]).sum())
                assert corrected - broken == int(candidate_correct[scenario].sum() - control_correct[scenario].sum())
                paired_counts.append(dict(endpoint=endpoint, scenario=size, changed=changed, corrected=corrected, broken=broken,
                                          changed_both_wrong=changed - corrected - broken,
                                          eligible_rows=int(eligible[scenario].sum())))
        intervals = _paired_intervals(y, groups, prediction, score_summary)
        arrays = dict(ids=ids, classes=classes, y=y, prompt_ids=groups, fold=our_folds,
                      scenarios=canonical['scenarios'], batch_id=canonical['global_batches'],
                      inferred_group_id=canonical['inferred'], oracle_group_id=canonical['oracle'],
                      eligible=eligible, oracle_eligible=canonical['eligible_oracle'], train_sha256=np.asarray(TRAIN_SHA256))
        arrays.update({'probability_' + key: value for key, value in fused['component_probability'].items()})
        arrays.update({'fold_' + key: value for key, value in fused['component_folds'].items()})
        arrays['deb_log_probability_plain'] = canonical['plain_log_probability']
        arrays.update({'deb_log_probability_' + key: canonical['centered_log_probability_' + key] for key in HEADS})
        arrays.update({'canonical_cpu_prediction_' + key: canonical['prediction_' + key] for key in HEADS})
        arrays.update({'log_evidence_' + key: value for key, value in fused['evidence'].items()})
        arrays.update({'raw_probability_' + key: value for key, value in fused['raw_posterior'].items()})
        arrays.update({'balanced_probability_' + key: value for key, value in fused['posterior'].items()})
        arrays.update({'prediction_' + key: value for key, value in prediction.items()})
        assert all(not np.asarray(value).dtype.hasobject for value in arrays.values())
        np.savez_compressed(output_root / 'ensemble_arrays.npz', **arrays)
        with np.load(output_root / 'ensemble_arrays.npz', allow_pickle=False) as saved:
            assert set(saved.files) == set(arrays)
            for key, value in arrays.items():
                assert np.array_equal(saved[key], value), ('Saved array mismatch', key)
        pd.DataFrame(per_class).to_csv(output_root / 'per_class_recall.csv', index=False)
        pd.DataFrame(per_fold).to_csv(output_root / 'fold_scores.csv', index=False)
        pd.DataFrame(paired_counts).to_csv(output_root / 'paired_changes.csv', index=False)
        pd.DataFrame(fused['balancing_records']).to_csv(output_root / 'balancing_batches.csv', index=False)
        warning_records = canonical_manifest['warning_records']
        write_json(output_root / 'warnings.json', warning_records)
        component_summary = {key: float(balanced_accuracy_score(y, value.argmax(1))) for key, value in fused['component_probability'].items()}
        limitations = [
            'Available-component proxy, not complete S02 and not an official public or private leaderboard score.',
            'New self-consistent canonical CPU development reference; historical C discrepancy remains separately reported and Stage06D remains failed.',
            'Plain LoRA1/LoRA2 and two attention members only: missing centered LoRA, other centered LLMs, LightGBM and fine-tuned encoders.',
            'S02 source weights were learned using development labels; reuse and repeated experiments make every reported score development evidence.',
            'Rowwise fixed fusion excludes each target prompt from every supervised member fit; member fold schemes differ.',
            'Batchwise balancing can use peer predictions trained on the target prompt under another member fold scheme; balanced scores and intervals are potentially contaminated diagnostics.',
            'Frozen-fit prompt bootstrap omits refitting, group recovery and transductive-batch uncertainty; not an untouched uncertainty estimate.',
            'Thirty-prompt fold batches differ from the full hidden-test batch; feature caches are quantized and package precision may differ.',
            'LoRA2 completion files contain no numeric file hashes; current hashes are recorded and cached fp16 features are checked against reported accuracy.',
            'Public neural provenance audited from saved artifacts, not independently retrained models; no paid services or submission.']
        result = dict(status='COMPLETE_CANONICAL_CPU_FIXED_AVAILABLE_COMPONENT_PROXY', rows=n, prompts=len(np.unique(groups)), classes=classes.tolist(),
                      train_sha256=TRAIN_SHA256, fixed_weights=WEIGHTS, weight_search_performed=False, component_selection_performed=False,
                      primary_endpoint='raw_fixed_rowwise_fusion_mean_balanced_accuracy',
                      balancing_endpoint='exact_S02_alpha0.5_per_batch_potentially_contaminated_diagnostic',
                      scenarios=list(SCENARIOS), scores=score_summary,
                      mean_scores={key: float(np.mean(value)) for key, value in score_summary.items()}, paired_comparison=intervals,
                      primary_raw_screen_pass=bool(intervals['raw']['mean_gain'] > 0
                                                   and intervals['raw']['conditional_prompt_95_interval'][0] > 0
                                                   and min(intervals['raw']['scenario_gains']) >= -.005
                                                   and not [w for w in warning_records if w['category'] != 'DeprecationWarning']),
                      primary_screen_rule='Raw mean gain > 0; conditional prompt CI lower > 0; every scenario gain >= -0.005; no non-deprecation fit warnings. Class harms reported, not a post-hoc veto.',
                      component_standalone_balanced_accuracy=component_summary,
                      canonical_cpu_cache=cache_receipt, historical_Stage06D_reproduction_passed=False,
                      all_warning_records=warning_records, paired_counts=paired_counts,
                      balancing_rows_with_cross_fold_peer=sum(row['rows_with_cross_fold_peer'] for row in fused['balancing_records'] if row['head'] == 'control'),
                      expected_sources=expected_sources,
                      seconds=time.monotonic() - started, limitations=limitations,
                      model_fits_in_consumer=0, model_inference_calls_in_consumer=0, group_recovery_calls_in_consumer=0,
                      official_submission_performed=False, new_neural_training=False, paid_services_used=False)
        write_json(output_root / 'ensemble_result.json', result)
        output_manifest = {path.name: dict(sha256=digest(path), bytes=path.stat().st_size) for path in sorted(output_root.iterdir())
                           if path.is_file() and path.name not in ('status.json', 'output_manifest.json')}
        write_json(output_root / 'output_manifest.json', output_manifest)
        status('COMPLETE_CANONICAL_CPU_FIXED_AVAILABLE_COMPONENT_PROXY', valid=True)
        print('ENSEMBLE_PER_CLASS_CSV', json.dumps(pd.DataFrame(per_class).to_csv(index=False)), flush=True)
        print('ENSEMBLE_FOLD_SCORES_CSV', json.dumps(pd.DataFrame(per_fold).to_csv(index=False)), flush=True)
        print('ENSEMBLE_PAIRED_CHANGES_CSV', json.dumps(pd.DataFrame(paired_counts).to_csv(index=False)), flush=True)
        print('ENSEMBLE_RESULT', json.dumps(result, allow_nan=False), flush=True)
        print('STAGE06E_ENSEMBLE_COMPLETE', flush=True)
        return result
    except BaseException as error:
        status('FAILED_PRESERVE_OUTPUTS', exception_type=type(error).__name__, exception_message=str(error))
        print('STAGE06E_CACHED_ENSEMBLE_FAILED', type(error).__name__, str(error), flush=True)
        raise
