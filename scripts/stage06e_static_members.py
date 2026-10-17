"""Stage 06E one-time preparation of five fixed public/static evidence members.

The audit/reconstruction statements are mechanically copied from Stage 06D
c6005196a73100676141d18f67eb299c1bf686a547fbb077b91825d2bd2611b3.
The unchanged LoRA affine readout audits are evaluated once against archived
completion evidence. This is not new neural extraction or training. A separate
consumer only reads the resulting probabilities; it executes no readout/head.

Attribution: Lucas Boesen Apache-2.0 public source V359126546 for fixed weights;
lightningv08 Apache-2.0 source V358831649 for LoRA1 public readout audit.
Callable entry point only. No fit, GPU, network, weights search or submission.
"""

D_STATIC_SOURCE_SHA256 = 'c6005196a73100676141d18f67eb299c1bf686a547fbb077b91825d2bd2611b3'
STATIC_SCHEMA = 'stage06e-static-members-v1'
MEMBERS = ('stylegram', 'lora', 'lora2', 'tokattn', 'tokattn_deb')

def prepare_static_members(style_probabilities, style_classes, style_ids):
    """Audit existing public evidence once and archive exact numeric probabilities.

    style arrays come from the unchanged frozen probability transfer. Other
    inputs are immutable Kaggle attachments and are never executed as code.
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
    from scipy.special import logsumexp
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix
    from sklearn.model_selection import GroupKFold
    from threadpoolctl import threadpool_limits

    started = time.monotonic()
    threadpool_limits(2)
    input_root = Path(os.environ.get('NTL_STAGE06E_INPUT', '/kaggle/input'))
    output_root = Path(os.environ.get('NTL_STAGE06E_STATIC_OUTPUT', '/kaggle/working/stage06e_static_members'))
    assert not output_root.exists() or not any(output_root.iterdir()), 'Refusing to overwrite an existing ensemble trial.'
    output_root.mkdir(parents=True, exist_ok=True)
    train_sha = '40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707'
    weights = {'stylegram': 1.516, 'lora': .177, 'lora2': .113, 'tokattn': .153,
               'tokattn_deb': .185, 'deb_plain': .05, 'deb_center': .25}
    scenarios = (1, 3, 6)
    artifact_records = []
    id_checks = []
    fold_proofs = []

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

        assert np.array_equal(np.asarray(style_classes, dtype=str), classes)
        assert np.array_equal(np.asarray(style_ids, dtype=str), ids)
        style_probability = probabilities(style_probabilities, 'stylegram')
        assert hashlib.sha256(np.ascontiguousarray(style_probability, dtype=np.float64).tobytes()).hexdigest() == '5e2e05577b7dc632f38f67d47958d7f3d069e8b758ead4664050f0d984803bad', 'Frozen style probability bytes changed'
        style_root_setting = os.environ.get('NTL_STAGE06E_STYLE_ROOT')
        style_path = Path(style_root_setting) / 'stylegram_oof.npz' if style_root_setting else one('stylegram_oof.npz', Path('/kaggle/working'))
        style_root = style_path.parent
        for name in ('stylegram_oof.npz', 'id_provenance.json', 'metrics.json', 'warnings.json', 'status.json'):
            record(style_root / name)
        transfer_verification_path = style_root / 'transfer_verification.json'
        assert transfer_verification_path.is_file(), 'Exact frozen style transfer receipt is required'
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
        component_folds = {'stylegram': our_folds.copy()}

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

        assert tuple(component_probability) == MEMBERS and len(artifact_records) == 47
        # Five whole-prompt exclusion proofs, each with five held-out folds.
        assert len(fold_proofs) == 25
        assert not [path for path in output_root.iterdir() if path.name != 'status.json']
        arrays = dict(ids=ids, classes=classes, y=y, groups=groups,
                      train_sha256=np.asarray(train_sha), fold=our_folds)
        arrays.update({'probability_' + key: value for key, value in component_probability.items()})
        arrays.update({'fold_' + key: value for key, value in component_folds.items()})
        assert all(not np.asarray(value).dtype.hasobject for value in arrays.values())
        cache_path = output_root / 'static_member_probabilities.npz'
        np.savez_compressed(cache_path, **arrays)
        with np.load(cache_path, allow_pickle=False) as saved:
            assert set(saved.files) == set(arrays)
            for key, value in arrays.items():
                assert np.array_equal(saved[key], value), ('Static cache readback', key)
        gate = dict(status='PASS', schema_version=STATIC_SCHEMA, cache_file=cache_path.name,
                    cache_sha256=digest(cache_path), all_saved_arrays_exact_readback=True,
                    model_fits=0, new_neural_inference=False,
                    existing_public_affine_readout_audit=True,
                    rows=n, classes=18, prompts=150, whole_prompt_fold_proofs=fold_proofs)
        write_json('static_cache_gate.json', gate)
        write_json('input_artifacts.json', artifact_records)
        write_json('id_checks.json', id_checks)
        write_json('fold_proofs.json', fold_proofs)
        manifest = dict(schema_version=STATIC_SCHEMA, status='PASS', cache_file=cache_path.name,
                        cache_sha256=digest(cache_path), internal_gate_file='static_cache_gate.json',
                        internal_gate_sha256=digest(output_root / 'static_cache_gate.json'),
                        source_sha256=digest(Path(__file__)), original_D_source_sha256=D_STATIC_SOURCE_SHA256,
                        train_sha256=train_sha, arrays={key: dict(dtype=str(value.dtype), shape=list(value.shape), sha256=hashlib.sha256(value.tobytes(order='C')).hexdigest())
                                                      for key, value in arrays.items()},
                        members=list(MEMBERS), input_artifacts=artifact_records,
                        id_checks=id_checks, fold_proofs=fold_proofs,
                        component_balanced_accuracy={key: float(balanced_accuracy_score(y, value.argmax(1)))
                                                     for key, value in component_probability.items()},
                        lora1_maximum_log_probability_reconstruction_error=lora1_reconstruction_error,
                        lora2_folds=lora2_fold_records,
                        environment=dict(python=platform.python_version(),
                                         packages={name: version(name) for name in ('numpy', 'pandas', 'scipy', 'scikit-learn', 'threadpoolctl')}),
                        seconds=time.monotonic() - started,
                        provenance_limits=['LoRA2 completion files supply no numeric file hashes; current hashes and published validation accuracy audited.',
                                           'Public supervised provenance audited from saved artifacts; models not independently retrained.'])
        write_json('static_cache_manifest.json', manifest)
        status('COMPLETE_FIXED_STATIC_MEMBER_CACHE', valid=True)
        print('STAGE06E_STATIC_MEMBERS_COMPLETE', json.dumps(dict(cache_sha256=digest(cache_path),
              members=list(MEMBERS), input_files_audited=len(artifact_records), model_fits=0)), flush=True)
        return manifest
    except BaseException as error:
        write_json('input_artifacts.json', artifact_records)
        status('FAILED_PRESERVE_OUTPUTS', exception_type=type(error).__name__, exception_message=str(error))
        print('STAGE06E_STATIC_MEMBERS_FAILED', type(error).__name__, str(error), flush=True)
        raise
