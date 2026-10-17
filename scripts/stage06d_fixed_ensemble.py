"""Fixed, CPU-only available-component ensemble comparison.

Protocol: plans/STAGE_06D_FIXED_ENSEMBLE_PROXY.md. Callable entry point only;
this module never starts fitting merely on import. No network, model weights,
new neural training, weight search, submission or final-selection changes.

Attribution: frozen weights and exact 50-iteration marginal balancing are from
Lucas Boesen's public Apache-2.0 notebook V359126546. Numeric LoRA1 readout audit
follows lightningv08's Apache-2.0 public source V358831649. These supervised
artifacts are not independently retrained here.
"""


def stage06d_balance(log_evidence, alpha=0.5):
    """Exact S02 balancing recipe, applied by the caller to ONE test batch."""
    import numpy as np
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


def stage06d_self_check():
    """No model fits: stable balancing, singleton and paired-count fixtures."""
    import numpy as np
    values = np.asarray([[9., 0., -3.], [8., 1., -4.], [0., 1., 2.]])
    actual = stage06d_balance(values)
    expected = np.exp(values - values.max(1, keepdims=True))
    expected /= expected.sum(1, keepdims=True)
    target = .5 * expected.sum(0) + .5 * np.full(3, len(expected) / 3)
    for _ in range(50):
        expected *= (target / (expected.sum(0) + 1e-12))[None, :]
        expected /= expected.sum(1, keepdims=True) + 1e-12
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(stage06d_balance(values + 1000), actual, rtol=0, atol=1e-14)
    singleton = stage06d_balance(values[:1])
    np.testing.assert_allclose(singleton, [[1 / (1 + np.exp(-9) + np.exp(-12)),
                                          np.exp(-9) / (1 + np.exp(-9) + np.exp(-12)),
                                          np.exp(-12) / (1 + np.exp(-9) + np.exp(-12))]], rtol=0, atol=1e-15)
    y = np.asarray([0, 1, 2, 0]); control = np.asarray([1, 1, 0, 0]); candidate = np.asarray([0, 0, 2, 0])
    corrected = (~(control == y) & (candidate == y)).sum()
    broken = ((control == y) & ~(candidate == y)).sum()
    assert corrected == 2 and broken == 1
    return {'status': 'PASSED', 'model_fits': 0,
            'checks': ['source balancing exact order', 'row-constant stability', 'singleton no balance', 'corrected/broken arithmetic']}


def run_fixed_ensemble(style_probabilities, style_classes, style_ids):
    """Run the fixed paired comparison with the caller-provided frozen style OOF.

    Transferred frozen style artifacts must be under NTL_STAGE06D_STYLE_ROOT or uniquely
    discoverable as /kaggle/working/**/stylegram_oof.npz. The supplied arrays
    are checked against those artifacts and their whole-prompt exclusion proof.
    All other inputs are immutable Kaggle attachments, never executed as code.
    """
    import hashlib
    import io
    import json
    import os
    import pickletools
    import platform
    import time
    import warnings
    import zipfile
    from datetime import datetime, timezone
    from importlib.metadata import version
    from pathlib import Path
    import numpy as np
    import pandas as pd
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    from scipy.special import logsumexp
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits

    started = time.monotonic()
    threadpool_limits(2)
    input_root = Path(os.environ.get('NTL_STAGE06D_INPUT', '/kaggle/input'))
    output_root = Path(os.environ.get('NTL_STAGE06D_OUTPUT', '/kaggle/working/stage06d_ensemble'))
    assert not output_root.exists() or not any(output_root.iterdir()), 'Refusing to overwrite an existing ensemble trial.'
    output_root.mkdir(parents=True, exist_ok=True)
    train_sha = '40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707'
    expected_glm_sha = '2b41585a9482cef825962a8cc096697933c0b1237e8ede0712d1f8ab361d9414'
    expected_deb_sha = '11eed58762ad128f97457dde9cc4ab03c7932e1b4b04845ec6664c834b1b1dda'
    weights = {'stylegram': 1.516, 'lora': .177, 'lora2': .113, 'tokattn': .153,
               'tokattn_deb': .185, 'deb_plain': .05, 'deb_center': .25}
    scenarios = (1, 3, 6)
    artifact_records = []
    id_checks = []
    fold_proofs = []
    warning_records = []

    def digest(path):
        hasher = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                hasher.update(chunk)
        return hasher.hexdigest()

    def strings_digest(values):
        return hashlib.sha256(json.dumps(np.asarray(values).tolist(), ensure_ascii=False,
                                         separators=(',', ':')).encode('utf-8')).hexdigest()

    def one(pattern, root=None):
        matches = sorted((input_root if root is None else root).rglob(pattern))
        assert len(matches) == 1, (pattern, [str(p) for p in matches])
        return matches[0]

    def record(path, expected=None):
        actual = digest(path)
        assert expected is None or actual == expected, ('File hash mismatch', str(path), actual)
        artifact_records.append({'path': str(path), 'sha256': actual, 'bytes': path.stat().st_size})
        return actual

    def write_json(name, value):
        (output_root / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')

    def status(phase, **extra):
        write_json('status.json', dict(status=phase, elapsed_seconds=time.monotonic() - started,
                                      observed_utc=datetime.now(timezone.utc).isoformat(), **extra))

    status('VALIDATING_INPUTS')
    self_check = stage06d_self_check()
    try:
        train_path = one('train.csv')
        record(train_path, train_sha)
        train = pd.read_csv(train_path, dtype=str)
        assert len(train) == 2699 and train.id.is_unique and not train.isna().any().any()
        assert train.prompt_id.nunique() == 150 and not train.duplicated(['prompt_id', 'model']).any()
        ids = train.id.to_numpy(dtype=str)
        groups = train.prompt_id.to_numpy(dtype=str)
        texts = train.text.to_numpy()
        classes = np.sort(train.model.unique()).astype(str)
        assert len(classes) == 18
        y = np.searchsorted(classes, train.model.to_numpy())
        n = len(ids)
        id_set = set(ids)

        def safe_ids(path, expected):
            """No pickle execution: only unique official-ID literals are inspected."""
            try:
                with np.load(path, allow_pickle=False) as saved:
                    actual = saved['ids'].astype(str)
                mode = 'numpy_nonobject'
            except ValueError as error:
                assert 'Object arrays cannot be loaded' in str(error)
                with zipfile.ZipFile(path) as archive:
                    info = archive.getinfo('ids.npy')
                    assert info.file_size < 2_000_000
                    raw = io.BytesIO(archive.read(info))
                file_version = np.lib.format.read_magic(raw)
                assert file_version in ((1, 0), (2, 0))
                reader = np.lib.format.read_array_header_1_0 if file_version == (1, 0) else np.lib.format.read_array_header_2_0
                shape, fortran, dtype = reader(raw)
                assert shape == (len(expected),) and dtype.hasobject and not fortran
                literals = []
                for opcode, argument, position in pickletools.genops(raw.read()):
                    if isinstance(argument, bytes):
                        try:
                            argument = argument.decode('ascii')
                        except UnicodeDecodeError:
                            continue
                    if isinstance(argument, str) and argument in id_set:
                        literals.append(argument)
                actual = np.asarray(literals, dtype=str)
                mode = 'object_npy_static_unique_ID_literals_no_unpickling'
            assert np.array_equal(actual, expected), ('ID alignment mismatch', str(path))
            assert len(np.unique(actual)) == len(expected)
            id_checks.append(dict(path=str(path), mode=mode, rows=len(expected), ordered_ids_sha256=strings_digest(actual)))
            return actual

        def probabilities(value, name, atol=2e-6):
            value = np.asarray(value)
            assert value.shape == (n, 18) and np.isfinite(value).all(), name
            assert (value >= 0).all() and (value <= 1 + atol).all(), name
            np.testing.assert_allclose(value.sum(1), 1, rtol=0, atol=atol, err_msg=name)
            return value

        def prove_folds(component, fold, declared=None):
            assert fold.shape == (n,) and np.issubdtype(fold.dtype, np.integer)
            unique_folds = np.unique(fold)
            assert len(unique_folds) == 5
            coverage = np.zeros(n, dtype=np.int64)
            for number in unique_folds:
                valid = np.flatnonzero(fold == number)
                fit = np.flatnonzero(fold != number)
                assert len(valid) and len(fit) and not set(groups[fit]) & set(groups[valid])
                assert len(np.unique(groups[valid])) == 30
                assert np.array_equal(np.unique(y[fit]), np.arange(18))
                coverage[valid] += 1
                if declared is not None:
                    entry = declared[int(number)]
                    assert np.array_equal(np.sort(entry['fit']), fit) and np.array_equal(np.sort(entry['valid']), valid)
                fold_proofs.append(dict(component=component, fold=int(number), fit_rows=len(fit), valid_rows=len(valid),
                                       held_out_prompts=np.unique(groups[valid]).tolist(), prompt_overlap=0,
                                       fit_id_sha256=strings_digest(ids[fit]), valid_id_sha256=strings_digest(ids[valid])))
            assert np.all(coverage == 1)

        # Exact same GroupKFold split/order as 06C and the archived style prerequisite.
        splits = list(GroupKFold(5).split(texts, y, groups))
        our_folds = np.full(n, -1, dtype=np.int64)
        for number, (fit, valid) in enumerate(splits):
            our_folds[valid] = number + 1
        expected_prompt_folds = {'p000': 4, 'p001': 3, 'p002': 5}
        expected_prompt_folds.update({f'p{i:03d}': 5 - i % 5 for i in range(3, 150)})
        assert all(expected_prompt_folds[g] == int(f) for g, f in zip(groups, our_folds))
        prove_folds('frozen_DeBERTa_CPU_heads', our_folds)

        assert np.array_equal(np.asarray(style_classes, dtype=str), classes)
        assert np.array_equal(np.asarray(style_ids, dtype=str), ids)
        style_probability = probabilities(style_probabilities, 'stylegram')
        style_root_setting = os.environ.get('NTL_STAGE06D_STYLE_ROOT')
        style_path = Path(style_root_setting) / 'stylegram_oof.npz' if style_root_setting else one('stylegram_oof.npz', Path('/kaggle/working'))
        style_root = style_path.parent
        for name in ('stylegram_oof.npz', 'id_provenance.json', 'metrics.json', 'warnings.json', 'status.json'):
            record(style_root / name)
        transfer_verification_path = style_root / 'transfer_verification.json'
        if transfer_verification_path.exists():
            record(transfer_verification_path)
        style_provenance = json.loads((style_root / 'id_provenance.json').read_text())
        style_metrics = json.loads((style_root / 'metrics.json').read_text())
        style_status = json.loads((style_root / 'status.json').read_text())
        assert style_provenance['train_sha256'] == train_sha
        assert style_metrics['valid_for_proxy_fusion'] and style_status['valid_for_proxy_fusion']
        assert style_metrics['convergence_warning_count'] == 0
        assert style_provenance['validation_rows_seen_exactly_once']
        with np.load(style_path, allow_pickle=False) as saved:
            for key, value in [('ids', ids), ('classes', classes), ('prompt_ids', groups), ('fold', our_folds), ('y', y),
                               ('probabilities', style_probability)]:
                assert np.array_equal(saved[key], value), ('style', key)
            assert saved['train_sha256'].item() == train_sha
        for entry in style_provenance['folds']:
            number = entry['fold']
            assert number in range(1, 6) and entry['prompt_overlap'] == 0
            assert entry['training_prompts'] == np.unique(groups[our_folds != number]).tolist()
            assert entry['validation_prompts'] == np.unique(groups[our_folds == number]).tolist()
        assert len(style_provenance['folds']) == 5
        prove_folds('stylegram', our_folds)
        component_probability = {'stylegram': style_probability.astype(np.float64)}
        component_folds = {'stylegram': our_folds.copy(), 'deb_plain': our_folds.copy(), 'deb_center': our_folds.copy()}

        # Audit and reconstruct LoRA1 plain predictions ONLY; no cross-fit feature centering.
        lora_oof_path = one('lora24_oof.npz')
        lora_root = lora_oof_path.parent
        record(lora_oof_path)
        spec_path = lora_root / 'training_spec.json'
        comparison_path = lora_root / 'lora24_comparison.json'
        record(spec_path); record(comparison_path)
        spec = json.loads(spec_path.read_text())
        comparison = json.loads(comparison_path.read_text())
        plan = spec['plan']
        assert plan['train_sha256'] == train_sha and plan['labels'] == classes.tolist()
        assert comparison['train_sha256'] == train_sha and comparison['classes'] == classes.tolist()
        assert [job for job in spec['jobs'] if job != 'final'] == [f'outer{i}' for i in range(5)]
        with np.load(lora_oof_path, allow_pickle=False) as saved:
            for key, value in [('ids', ids), ('labels', classes), ('y', y), ('groups', groups)]:
                assert np.array_equal(saved[key], value), ('lora1', key)
            lora1_published = saved['plain'].copy()
        assert lora1_published.shape == (3, n, 18) and np.isfinite(lora1_published).all()
        lora1_log = np.empty((n, 18), dtype=np.float64)
        lora1_fold = np.full(n, -1, dtype=np.int64)
        coverage = np.zeros(n, dtype=np.int64)
        declared = {}
        for number in range(5):
            job = f'outer{number}'
            fit = np.asarray(plan['jobs'][job]['fit'], dtype=np.int64)
            valid = np.asarray(plan['jobs'][job]['valid'], dtype=np.int64)
            assert len(np.unique(fit)) == len(fit) and len(np.unique(valid)) == len(valid)
            assert not set(fit) & set(valid) and set(fit) | set(valid) == set(range(n))
            assert not set(groups[fit]) & set(groups[valid])
            declared[number] = dict(fit=fit, valid=valid)
            folder = lora_root / 'lora24' / job
            complete_path = folder / 'COMPLETE.json'
            record(complete_path)
            complete = json.loads(complete_path.read_text())
            assert complete['job'] == job and complete['train_sha256'] == train_sha
            for filename in ('validation_features.npz', 'readouts.npz', 'training_report.json'):
                record(folder / filename, complete['files'][filename])
            training_report = json.loads((folder / 'training_report.json').read_text())
            assert training_report['job'] == job and training_report['train_sha256'] == train_sha
            assert training_report['epochs'] == 3
            feature_path = folder / 'validation_features.npz'
            safe_ids(feature_path, ids[valid])
            with np.load(feature_path, allow_pickle=False) as saved:
                assert np.array_equal(saved['classes'], classes) and saved['train_sha256'].item() == train_sha
                features = saved['features'].astype(np.float64)
            with np.load(folder / 'readouts.npz', allow_pickle=False) as saved:
                assert np.array_equal(saved['classes'], classes) and saved['train_sha256'].item() == train_sha
                coef = saved['plain_coef'].astype(np.float64)
                intercept = saved['plain_intercept'].astype(np.float64)
            assert features.shape[0] == len(valid) and coef.shape == (18, features.shape[1])
            assert np.isfinite(features).all() and np.isfinite(coef).all() and np.isfinite(intercept).all()
            logits = features @ coef.T + intercept
            lora1_log[valid] = logits - logsumexp(logits, axis=1, keepdims=True)
            lora1_fold[valid] = number; coverage[valid] += 1
            del features, coef, intercept
        assert np.all(coverage == 1)
        prove_folds('lora', lora1_fold, declared)
        for scenario in range(3):
            np.testing.assert_allclose(lora1_log, lora1_published[scenario], rtol=1e-7, atol=2e-6)
            np.testing.assert_allclose(float(balanced_accuracy_score(y, lora1_log.argmax(1))),
                                       comparison['standalone']['plain'][scenario], rtol=0, atol=1e-12)
        lora1_reconstruction_error = float(np.max(np.abs(lora1_log[None] - lora1_published)))
        component_probability['lora'] = probabilities(np.exp(lora1_log), 'lora', atol=1e-12)
        component_folds['lora'] = lora1_fold

        source_prompts = np.sort(np.unique(groups))
        np.random.RandomState(0).shuffle(source_prompts)
        source_map = {prompt: i % 5 for i, prompt in enumerate(source_prompts)}
        source_folds = np.asarray([source_map[prompt] for prompt in groups], dtype=np.int64)
        # Same numeric fp32 affine arithmetic as the previous LoRA2 audit.
        first_lora2_readout = one('job_0/readouts.npz')
        lora2_root = first_lora2_readout.parent.parent
        assert lora2_root.name == 'lora_out'
        lora2_probability = np.empty((n, 18), dtype=np.float64)
        coverage = np.zeros(n, dtype=np.int64)
        lora2_fold_records = []
        for number in range(5):
            folder = lora2_root / f'job_{number}'
            valid = np.flatnonzero(source_folds == number)
            feature_path = folder / 'valid_features.npz'
            safe_ids(feature_path, ids[valid])
            for filename in ('valid_features.npz', 'readouts.npz', 'COMPLETE.json'):
                record(folder / filename)
            with np.load(feature_path, allow_pickle=False) as saved:
                assert saved['features'].dtype == np.float16
                features = saved['features'].astype(np.float32)
            with np.load(folder / 'readouts.npz', allow_pickle=False) as saved:
                assert np.array_equal(saved['classes'], classes)
                assert np.array_equal(saved['pool_layers'], [2, 4, 6, 8, 12, 16, 20, 24])
                coef = saved['plain_coef']; intercept = saved['plain_intercept']
                assert coef.dtype == np.float32 and intercept.dtype == np.float32
                assert features.shape == (len(valid), 32768) and coef.shape == (18, 32768)
                logits = features @ coef.T + intercept
            assert logits.dtype == np.float32 and np.isfinite(logits).all()
            complete = json.loads((folder / 'COMPLETE.json').read_text())
            assert str(complete['job']) == str(number) and complete['skipped'] == 0 and complete['steps'] == 360
            accuracy = float((logits.argmax(1) == y[valid]).mean())
            np.testing.assert_allclose(accuracy, complete['valid_acc'], rtol=0, atol=1e-12)
            logits -= logits.max(1, keepdims=True)
            probability = np.exp(logits)
            probability /= probability.sum(1, keepdims=True)
            lora2_probability[valid] = probability
            coverage[valid] += 1
            lora2_fold_records.append(dict(fold=number, reconstructed_accuracy=accuracy, reported_accuracy=complete['valid_acc'],
                                            steps=complete['steps'], skipped_updates=complete['skipped']))
            del features, coef, intercept
        assert np.all(coverage == 1)
        prove_folds('lora2', source_folds)
        component_probability['lora2'] = probabilities(lora2_probability, 'lora2')
        component_folds['lora2'] = source_folds.copy()
        np.testing.assert_allclose(float(balanced_accuracy_score(y, lora2_probability.argmax(1))), .4804424558786976, rtol=0, atol=1e-12)

        token_sources = [('tokattn', 'tokattn_oof.npz', 'bdb4b20de1236edb3618464f464157468a81179d407d577d766c0941c5735077', .39378076062639816),
                         ('tokattn_deb', 'tokattn_debl23_oof.npz', '69b97d2e2693c7b646dbdb868c8803d0e4048ff3f62c874cd87be39d70395532', .4108153119562516)]
        for component, filename, expected_sha, expected_score in token_sources:
            path = one(filename)
            record(path, expected_sha); safe_ids(path, ids)
            with np.load(path, allow_pickle=False) as saved:
                assert np.array_equal(saved['classes'], classes) and np.array_equal(saved['fold'], source_folds)
                probability = saved['probs'].copy()
            component_probability[component] = probabilities(probability, component).astype(np.float64)
            component_folds[component] = source_folds.copy()
            prove_folds(component, source_folds)
            np.testing.assert_allclose(float(balanced_accuracy_score(y, probability.argmax(1))), expected_score, rtol=0, atol=1e-12)
        print('STAGE06D_COMPONENTS_AUDITED', json.dumps({'members': list(component_probability), 'files': len(artifact_records)}), flush=True)

        # One immutable own-06C output directory binds the feature cache, report
        # and saved argmax tensors together. Refitting cannot select a new recipe.
        glm_path = one('glm_features.npz')
        own_root = glm_path.parent
        assert (own_root / 'paired_predictions.npz').is_file()
        for name in ('glm_features.npz', 'glm_manifest.json', 'paired_predictions.npz', 'centering_result.json'):
            record(own_root / name, expected_glm_sha if name == 'glm_features.npz' else None)
        glm_manifest = json.loads((own_root / 'glm_manifest.json').read_text())
        frozen_result = json.loads((own_root / 'centering_result.json').read_text())
        assert glm_manifest['status'] == 'COMPLETE' and glm_manifest['cache_sha256'] == expected_glm_sha
        assert glm_manifest['input_provenance']['train_sha256'] == train_sha and glm_manifest['layers'] == [8, 3, 24]
        assert frozen_result['train_sha256'] == train_sha and frozen_result['cache_sha256'] == expected_deb_sha
        assert frozen_result['glm_cache_sha256'] == expected_glm_sha and frozen_result['scenarios'] == list(scenarios)
        assert frozen_result['screen_pass'] is True, 'Candidate rejected by 06C; fusion is not authorized by this fixed protocol.'
        with np.load(glm_path, allow_pickle=False) as saved:
            assert np.array_equal(saved['ids'], ids) and np.array_equal(saved['layers'], [8, 3, 24])
            assert saved['train_sha256'].item() == train_sha
            expected_text_hash = np.asarray([hashlib.sha256(t.encode('utf-8')).hexdigest() for t in texts], dtype=str)
            assert np.array_equal(saved['text_sha256'], expected_text_hash)
            glm_features = saved['features'].copy()
        assert glm_features.shape == (n, 3, 4096) and glm_features.dtype == np.float32 and np.isfinite(glm_features).all()
        glm_group_features = glm_features[:, 2].copy()
        del glm_features
        with np.load(own_root / 'paired_predictions.npz', allow_pickle=False) as saved:
            for key, value in [('ids', ids), ('classes', classes), ('y', y), ('groups', groups), ('fold', our_folds)]:
                assert np.array_equal(saved[key], value), ('06C frozen', key)
            frozen_prediction = {key: saved[key].copy() for key in ('control', 'candidate', 'oracle_control', 'oracle_candidate')}
        assert all(part.shape == (3, n) for part in frozen_prediction.values())
        deb_cache = one('enc_debv3l2_train.npz')
        record(deb_cache, expected_deb_sha); safe_ids(deb_cache, ids)
        with np.load(deb_cache, allow_pickle=False) as saved:
            layers = saved['layers']; embeddings = saved['emb']
            assert embeddings.shape == (n, 9, 1024) and embeddings.dtype == np.float16
            features = np.concatenate([embeddings[:, np.flatnonzero(layers == layer).item()] for layer in (24, 6)], axis=1).astype(np.float32)
        assert features.shape == (n, 2048) and np.isfinite(features).all()
        del embeddings

        def centered(value, labels):
            result = value.copy()
            for label in np.unique(labels):
                positions = np.flatnonzero(labels == label)
                result[positions] -= value[positions].mean(0)
            return result

        def partition_training(rows, size, rng):
            assignments = np.full(len(rows), -1, dtype=int)
            exposure = np.zeros(len(rows), dtype=int)
            offset = 0
            for prompt in np.unique(groups[rows]):
                positions = rng.permutation(np.flatnonzero(groups[rows] == prompt))
                chunks = [positions[start:start + size] for start in range(0, len(positions), size)]
                if len(chunks[-1]) == 1 and len(chunks) > 1:
                    tail = chunks.pop(); chunks[-1] = np.concatenate([chunks[-1], tail])
                for chunk in chunks:
                    assignments[chunk] = offset; exposure[chunk] += 1; offset += 1
            assert np.all(assignments >= 0) and np.all(exposure == 1)
            return assignments

        def contexts(rows, size, number):
            rng = np.random.default_rng(20261017 + 1000 + number)
            batches = np.empty(len(rows), dtype=int)
            for prompt in np.unique(groups[rows]):
                members = rng.permutation(np.flatnonzero(groups[rows] == prompt))
                batches[members] = np.arange(len(members)) // size
            inferred = np.empty(len(rows), dtype=int)
            oracle = np.empty(len(rows), dtype=int)
            offset = 0; oracle_offset = 0
            for batch in np.unique(batches):
                positions = np.flatnonzero(batches == batch)
                if len(positions) > 1:
                    tf = TfidfVectorizer(stop_words='english', sublinear_tf=True).fit_transform(texts[rows[positions]])
                    lexical = np.clip(1 - (tf @ tf.T).toarray(), 0, None)
                    value = np.asarray(glm_group_features[rows[positions]], dtype=np.float64)
                    value = value - value.mean(0)
                    value /= np.linalg.norm(value, axis=1, keepdims=True) + 1e-12
                    distance = (lexical + np.clip(1 - value @ value.T, 0, None)) / 2
                    np.fill_diagonal(distance, 0)
                    recovered = fcluster(linkage(squareform(distance, checks=False), method='average'), .60, criterion='distance')
                else:
                    recovered = np.zeros(1, dtype=int)
                inferred[positions] = recovered + offset; offset += int(recovered.max()) + 1
                _, true = np.unique(groups[rows[positions]], return_inverse=True)
                oracle[positions] = true + oracle_offset; oracle_offset += int(true.max()) + 1
            return batches, inferred, oracle

        def fit_head(value, targets, fold, role, sample_weights=None):
            scaler = StandardScaler().fit(value, sample_weight=sample_weights)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                classifier = LogisticRegression(C=.005, max_iter=5000).fit(scaler.transform(value), targets, sample_weight=sample_weights)
            warning_records.extend(dict(fold=fold, head=role, category=w.category.__name__, message=str(w.message)) for w in caught)
            assert np.array_equal(classifier.classes_, np.arange(18))
            return scaler, classifier

        def log_probability(head, value):
            scaler, classifier = head
            return np.log(np.clip(classifier.predict_proba(scaler.transform(value)), 1e-9, 1))

        status('REFITTING_FIXED_DEBERTA_HEADS')
        deb_log = {key: np.full((3, n, 18), np.nan) for key in ('plain', 'control', 'candidate', 'oracle_control', 'oracle_candidate')}
        reproduction = {key: np.full((3, n), -1, dtype=np.int64) for key in frozen_prediction}
        batch_id = np.full((3, n), -1, dtype=np.int64)
        inferred_id = np.full((3, n), -1, dtype=np.int64)
        oracle_id = np.full((3, n), -1, dtype=np.int64)
        eligible = np.zeros((3, n), dtype=bool)
        oracle_eligible = np.zeros((3, n), dtype=bool)
        realised = []
        for number, (fit, valid) in enumerate(splits):
            assert not set(groups[fit]) & set(groups[valid])
            plain = fit_head(features[fit], y[fit], number, 'plain')
            control = fit_head(centered(features[fit], groups[fit]), y[fit], number, 'control')
            rng = np.random.default_rng(20261017 + number)
            pieces = []
            for size in (2, 3, 6):
                grouping = partition_training(fit, size, rng)
                pieces.append(centered(features[fit], grouping))
                sizes, counts = np.unique(np.bincount(grouping), return_counts=True)
                realised.append(dict(fold=number, target=size, sizes=sizes.tolist(), counts=counts.tolist()))
            candidate = fit_head(np.concatenate(pieces), np.tile(y[fit], 3), number, 'candidate', np.full(3 * len(fit), 1 / 3))
            del pieces
            for scenario, size in enumerate(scenarios):
                batches, inferred, oracle = contexts(valid, size, number)
                assert int(batches.max()) < 100 and int(inferred.max()) < 10000 and int(oracle.max()) < 10000
                # Different outer folds always remain distinct transductive batches.
                batch_id[scenario, valid] = number * 100 + batches
                inferred_id[scenario, valid] = number * 10000 + inferred
                oracle_id[scenario, valid] = number * 10000 + oracle
                plain_log = np.empty((len(valid), 18))
                for batch in np.unique(batches):
                    pos = np.flatnonzero(batches == batch)
                    value = features[valid[pos]]
                    standardized = (value - value.mean(0)) / (value.std(0) + 1e-6) if len(pos) >= 2 else plain[0].transform(value)
                    plain_log[pos] = np.log(np.clip(plain[1].predict_proba(standardized), 1e-9, 1))
                deb_log['plain'][scenario, valid] = plain_log
                for mode, labels in [('inferred', inferred), ('oracle', oracle)]:
                    sizes = np.bincount(labels)[labels]
                    active = (sizes >= 2) & (sizes <= 18)
                    (eligible if mode == 'inferred' else oracle_eligible)[scenario, valid] = active
                    residual = centered(features[valid], labels)
                    for label, head in [('control', control), ('candidate', candidate)]:
                        key = label if mode == 'inferred' else 'oracle_' + label
                        # Full class log-probabilities are retained even on inactive
                        # rows; exactly the original eligibility mask gates fusion.
                        full_log = log_probability(head, residual)
                        deb_log[key][scenario, valid] = full_log
                        evidence = .05 * plain_log.copy()
                        if active.any():
                            evidence[active] += .25 * full_log[active]
                        reproduction[key][scenario, valid] = evidence.argmax(1)
            print('STAGE06D_DEBERTA_FOLD_COMPLETE', number + 1, 'seconds', round(time.monotonic() - started, 1), flush=True)
            del plain, control, candidate
        assert (batch_id >= 0).all() and (inferred_id >= 0).all() and (oracle_id >= 0).all()
        assert all(np.isfinite(value).all() for value in deb_log.values())
        assert realised == frozen_result['realised_training_group_sizes']
        reproduction_scores = {}
        for key, value in reproduction.items():
            assert np.array_equal(value, frozen_prediction[key]), ('Frozen 06C argmax mismatch', key)
            score = [float(balanced_accuracy_score(y, row)) for row in value]
            np.testing.assert_allclose(score, frozen_result['scores'][key], rtol=0, atol=1e-12)
            reproduction_scores[key] = score
        unexpected_warnings = [w for w in warning_records if w['category'] != 'DeprecationWarning']
        assert not unexpected_warnings, ('Unreviewed head warnings', unexpected_warnings)
        print('STAGE06D_06C_REPRODUCTION_PASSED', json.dumps({'argmax_arrays_matched': 12, 'scores': reproduction_scores}), flush=True)

        status('FIXED_FUSION')
        static_log = {key: np.log(np.clip(value, 1e-9, 1)) for key, value in component_probability.items()}
        base_evidence = np.zeros((n, 18), dtype=np.float64)
        for component in ('stylegram', 'lora', 'lora2', 'tokattn', 'tokattn_deb'):
            base_evidence += weights[component] * static_log[component]
        evidence = {}
        posterior = {}
        raw_posterior = {}
        prediction = {}
        balancing_records = []
        for head in ('control', 'candidate'):
            evidence[head] = base_evidence[None] + weights['deb_plain'] * deb_log['plain']
            evidence[head] = evidence[head].copy()
            evidence[head][eligible] += weights['deb_center'] * deb_log[head][eligible]
            raw_posterior[head] = np.exp(evidence[head] - evidence[head].max(2, keepdims=True))
            raw_posterior[head] /= raw_posterior[head].sum(2, keepdims=True)
            assert np.isfinite(raw_posterior[head]).all()
            np.testing.assert_allclose(raw_posterior[head].sum(2), 1, rtol=0, atol=1e-12)
            posterior[head] = np.empty((3, n, 18), dtype=np.float64)
            for scenario, size in enumerate(scenarios):
                for batch in np.unique(batch_id[scenario]):
                    rows = np.flatnonzero(batch_id[scenario] == batch)
                    assert len(np.unique(our_folds[rows])) == 1
                    posterior[head][scenario, rows] = stage06d_balance(evidence[head][scenario, rows])
                    # Labels from the target prompt may reach a peer's supervised
                    # prediction under a different source fold map. This flags
                    # contamination exposure rather than calling balancing OOF.
                    crossing = np.zeros(len(rows), dtype=bool)
                    for member_folds in component_folds.values():
                        crossing |= np.any(member_folds[rows, None] != member_folds[rows][None, :], axis=1)
                    balancing_records.append(dict(head=head, scenario=size, batch=int(batch), rows=len(rows),
                                                  rows_with_cross_fold_peer=int(crossing.sum()),
                                                  changed_predictions=int((posterior[head][scenario, rows].argmax(1) != evidence[head][scenario, rows].argmax(1)).sum())))
            prediction['raw_' + head] = evidence[head].argmax(2)
            prediction['balanced_' + head] = posterior[head].argmax(2)
        # 1-response grouping can infer a few eligible false merges: do not
        # assume identical complete evidence beyond the verified saved 06C argmax.
        score_summary = {key: [float(balanced_accuracy_score(y, part)) for part in value] for key, value in prediction.items()}
        per_class = []
        per_fold = []
        paired_counts = []
        confusion_paths = []
        for key, value in prediction.items():
            for scenario, size in enumerate(scenarios):
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
        for endpoint in ('raw', 'balanced'):
            control_correct = prediction[endpoint + '_control'] == y[None]
            candidate_correct = prediction[endpoint + '_candidate'] == y[None]
            for scenario, size in enumerate(scenarios):
                corrected = int((~control_correct[scenario] & candidate_correct[scenario]).sum())
                broken = int((control_correct[scenario] & ~candidate_correct[scenario]).sum())
                changed = int((prediction[endpoint + '_control'][scenario] != prediction[endpoint + '_candidate'][scenario]).sum())
                assert corrected - broken == int(candidate_correct[scenario].sum() - control_correct[scenario].sum())
                paired_counts.append(dict(endpoint=endpoint, scenario=size, changed=changed, corrected=corrected, broken=broken,
                                          changed_both_wrong=changed - corrected - broken,
                                          eligible_rows=int(eligible[scenario].sum())))
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
        # Persist all source-normalized probabilities, fixed log evidence and
        # the exact labels/batches/eligibility needed to audit the calculation.
        arrays = dict(ids=ids, classes=classes, y=y, prompt_ids=groups, fold=our_folds,
                      scenarios=np.asarray(scenarios, dtype=np.int64), batch_id=batch_id,
                      inferred_group_id=inferred_id, oracle_group_id=oracle_id,
                      eligible=eligible, oracle_eligible=oracle_eligible, train_sha256=np.asarray(train_sha))
        arrays.update({'probability_' + key: value for key, value in component_probability.items()})
        arrays.update({'fold_' + key: value for key, value in component_folds.items()})
        arrays.update({'deb_log_probability_' + key: value for key, value in deb_log.items()})
        arrays.update({'frozen06c_prediction_' + key: value for key, value in reproduction.items()})
        arrays.update({'log_evidence_' + key: value for key, value in evidence.items()})
        arrays.update({'raw_probability_' + key: value for key, value in raw_posterior.items()})
        arrays.update({'balanced_probability_' + key: value for key, value in posterior.items()})
        arrays.update({'prediction_' + key: value for key, value in prediction.items()})
        assert all(not np.asarray(value).dtype.hasobject for value in arrays.values())
        np.savez_compressed(output_root / 'ensemble_arrays.npz', **arrays)
        with np.load(output_root / 'ensemble_arrays.npz', allow_pickle=False) as saved:
            for key, value in arrays.items():
                assert np.array_equal(saved[key], value), ('Saved array mismatch', key)
        pd.DataFrame(per_class).to_csv(output_root / 'per_class_recall.csv', index=False)
        pd.DataFrame(per_fold).to_csv(output_root / 'fold_scores.csv', index=False)
        pd.DataFrame(paired_counts).to_csv(output_root / 'paired_changes.csv', index=False)
        pd.DataFrame(balancing_records).to_csv(output_root / 'balancing_batches.csv', index=False)
        write_json('warnings.json', warning_records)
        write_json('id_checks.json', id_checks)
        write_json('fold_proofs.json', fold_proofs)
        write_json('input_artifacts.json', artifact_records)
        component_summary = {key: float(balanced_accuracy_score(y, value.argmax(1))) for key, value in component_probability.items()}
        limitations = [
            'Available-component proxy, not complete S02 and not an official public or private leaderboard score.',
            'Plain LoRA1/LoRA2 and two attention members only: missing centered LoRA, other centered LLMs, LightGBM and fine-tuned encoders.',
            'S02 source weights were learned using development labels; reuse and repeated experiments make every reported score development evidence.',
            'Rowwise fixed fusion excludes each target prompt from every supervised member fit; member fold schemes differ.',
            'Batchwise balancing can use peer predictions trained on the target prompt under another member fold scheme; balanced scores and intervals are potentially contaminated diagnostics.',
            'Frozen-fit prompt bootstrap omits refitting, group recovery and transductive-batch uncertainty; not an untouched uncertainty estimate.',
            'Thirty-prompt fold batches differ from the full hidden-test batch; feature caches are quantized and package precision may differ.',
            'LoRA2 completion files contain no numeric file hashes; current hashes are recorded and cached fp16 features are checked against reported accuracy.',
            'Public neural training and checkpoint provenance are audited artifacts, not independently retrained models; no paid services or submission.'
        ]
        result = dict(status='COMPLETE_FIXED_AVAILABLE_COMPONENT_PROXY', rows=n, prompts=len(np.unique(groups)), classes=classes.tolist(),
                      train_sha256=train_sha, fixed_weights=weights, weight_search_performed=False, component_selection_performed=False,
                      primary_endpoint='raw_fixed_rowwise_fusion_mean_balanced_accuracy',
                      balancing_endpoint='exact_S02_alpha0.5_per_batch_potentially_contaminated_diagnostic',
                      scenarios=list(scenarios), scores=score_summary,
                      mean_scores={key: float(np.mean(value)) for key, value in score_summary.items()}, paired_comparison=intervals,
                      primary_raw_screen_pass=bool(intervals['raw']['mean_gain'] > 0
                                                   and intervals['raw']['conditional_prompt_95_interval'][0] > 0
                                                   and min(intervals['raw']['scenario_gains']) >= -.005
                                                   and not unexpected_warnings),
                      primary_screen_rule='Raw mean gain > 0; conditional prompt CI lower > 0; every scenario gain >= -0.005; no non-deprecation fit warnings. Class harms reported, not a post-hoc veto.',
                      component_standalone_balanced_accuracy=component_summary,
                      frozen06c_reproduction=dict(all_12_argmax_arrays_exact_match=True, scores=reproduction_scores,
                                                 realised_training_group_sizes_exact_match=True),
                      lora1_maximum_log_probability_reconstruction_error=lora1_reconstruction_error,
                      lora2_folds=lora2_fold_records, style_convergence_warning_count=0,
                      deb_convergence_warning_count=len([w for w in warning_records if w['category'] == 'ConvergenceWarning']),
                      all_warning_records=warning_records, paired_counts=paired_counts,
                      balancing_rows_with_cross_fold_peer=sum(row['rows_with_cross_fold_peer'] for row in balancing_records if row['head'] == 'control'),
                      input_versions_expected=dict(own_stage06c=359405443, deb_cache=358745836, lora1=358831649, lora2=358911769,
                                                   tokattn=358928013, tokattn_deb=358948428),
                      input_version_note='Binding established by Kaggle save-version UI; numeric files are hashed here.',
                      provenance=dict(ordered_ids_sha256=strings_digest(ids), ordered_prompts_sha256=strings_digest(groups),
                                      ordered_labels_sha256=strings_digest(classes[y]), fold_sha256=strings_digest(our_folds)),
                      environment=dict(python=platform.python_version(), platform=platform.platform(),
                                       packages={name: version(name) for name in ('numpy', 'pandas', 'scipy', 'scikit-learn', 'threadpoolctl')}),
                      self_check=self_check, limitations=limitations, seconds=time.monotonic() - started,
                      official_submission_performed=False, new_neural_training=False, paid_services_used=False)
        write_json('ensemble_result.json', result)
        output_manifest = {path.name: dict(sha256=digest(path), bytes=path.stat().st_size) for path in sorted(output_root.iterdir())
                           if path.is_file() and path.name not in ('status.json', 'output_manifest.json')}
        write_json('output_manifest.json', output_manifest)
        status('COMPLETE_FIXED_AVAILABLE_COMPONENT_PROXY', valid=True)
        print('ENSEMBLE_PER_CLASS_CSV', json.dumps(pd.DataFrame(per_class).to_csv(index=False)), flush=True)
        print('ENSEMBLE_FOLD_SCORES_CSV', json.dumps(pd.DataFrame(per_fold).to_csv(index=False)), flush=True)
        print('ENSEMBLE_PAIRED_CHANGES_CSV', json.dumps(pd.DataFrame(paired_counts).to_csv(index=False)), flush=True)
        print('ENSEMBLE_RESULT', json.dumps(result, allow_nan=False), flush=True)
        print('STAGE06D_ENSEMBLE_COMPLETE', flush=True)
        return result
    except BaseException as error:
        write_json('warnings.json', warning_records)
        write_json('input_artifacts.json', artifact_records)
        status('FAILED_PRESERVE_OUTPUTS', exception_type=type(error).__name__, exception_message=str(error))
        print('STAGE06D_ENSEMBLE_FAILED', type(error).__name__, str(error), flush=True)
        raise
