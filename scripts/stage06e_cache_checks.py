"""Output-only instrumentation and fail-closed gates for canonical Stage 06E.

No estimator is fitted or invoked here. Capture methods copy values produced by the
literal Stage 06C computation. All persisted NPZ arrays are numeric or Unicode.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
import warnings
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 'stage06e-canonical-cpu-v1'
ORIGINAL_SHA256 = '5c27fcdffde2bcfd512d20fb4edec9bd7133430e79ccd71e31875151d235a23d'
PLAN_SHA256 = '89dbf531820b965f09fab07157475b199f929f4d297bd268e0f141c765a80f32'
EXPECTED_RUNTIME = {'python': '3.13.15', 'numpy': '2.1.3', 'pandas': '2.3.3',
                    'scipy': '1.16.3', 'scikit-learn': '1.6.1', 'threadpoolctl': '3.6.0'}
EXPECTED_INPUT_VERSIONS = [170484, 358745836, 358831649, 358911769, 358928013, 358948428, 359405443]
KEYS = ('control', 'candidate', 'oracle_control', 'oracle_candidate')
ACTIVE_CAPTURE = None


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            value.update(chunk)
    return value.hexdigest()


def json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    return value


def write_json(path, value):
    path = Path(path)
    text = json.dumps(json_safe(value), indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(text)
    temporary.replace(path)


def exact_saved_arrays(path, captured):
    """Verify every stored array against its in-memory capture, with no pickle."""
    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) == set(captured), 'Serialized array keys differ'
        for key, expected in captured.items():
            actual = archive[key]
            assert actual.dtype == expected.dtype and actual.shape == expected.shape, (key, 'array schema')
            assert not actual.dtype.hasobject and actual.dtype.kind in 'biufcU', (key, 'unsafe array dtype')
            assert np.array_equal(actual, expected), (key, 'saved array differs from capture')
    return True


def reconcile_predictions(y, prediction, classes):
    """Balanced accuracy reconciled independently through class confusion counts."""
    y = np.asarray(y)
    prediction = np.asarray(prediction)
    count = len(classes)
    assert y.shape == prediction.shape and y.ndim == 1
    assert np.all((prediction >= 0) & (prediction < count))
    confusion = np.bincount(y * count + prediction, minlength=count * count).reshape(count, count)
    support = confusion.sum(1)
    assert np.all(support > 0)
    recalls = confusion.diagonal() / support
    return dict(balanced_accuracy=float(recalls.mean()), confusion=confusion.tolist(),
                support=support.tolist(), recall=recalls.tolist())


def validate_canonical_arrays(arrays, scores=None, reference_folds=None):
    """Internal cache checks only; this function never infers, regroups or fits."""
    for name,value in arrays.items():
        assert isinstance(value,np.ndarray) and not value.dtype.hasobject and value.dtype.kind in 'biufU', (name,'unsafe array type')
    ids, classes, y, groups, folds = (arrays[k] for k in ('ids', 'classes', 'y', 'groups', 'fold'))
    n, c = len(ids), len(classes)
    assert ids.dtype.kind == classes.dtype.kind == groups.dtype.kind == 'U'
    assert n > 0 and c > 1 and len(np.unique(ids)) == n and len(np.unique(classes)) == c
    assert np.all(ids != '') and np.all(classes != '') and np.all(groups != '')
    assert y.dtype.kind in 'iu' and folds.dtype.kind in 'iu'
    assert y.shape == groups.shape == folds.shape == ids.shape
    assert np.array_equal(arrays['scenarios'], [1, 3, 6])
    assert np.all((y >= 0) & (y < c))
    assert set(np.unique(folds)) == set(range(1, 6))
    assert all(len(np.unique(folds[groups==group])) == 1 for group in np.unique(groups)), 'Prompt spans folds'
    if reference_folds is not None:
        assert all(int(f) == int(reference_folds[g]) for g, f in zip(groups, folds))
    assert arrays['plain_coverage'].shape == (3, n)
    assert np.all(arrays['plain_coverage'] == 1), 'Missing or duplicated plain scored rows'
    assert arrays['context_coverage'].shape == (3, n) and np.all(arrays['context_coverage'] == 1)
    plain_p, plain_log = arrays['plain_probability'], arrays['plain_log_probability']
    assert plain_p.shape == plain_log.shape == (3, n, c)
    assert np.isfinite(plain_p).all() and np.isfinite(plain_log).all()
    assert np.all((plain_p >= 0) & (plain_p <= 1))
    assert np.allclose(plain_p.sum(-1), 1, rtol=0, atol=1e-12)
    # Preserve original elementwise log call boundaries as well as values.
    for scenario in range(3):
        for fold in range(1,6):
            fold_rows = folds == fold
            for batch in np.unique(arrays['batches'][scenario,fold_rows]):
                rows = fold_rows & (arrays['batches'][scenario] == batch)
                assert np.array_equal(plain_log[scenario,rows],np.log(np.clip(plain_p[scenario,rows],1e-9,1)))
    assert arrays['batches'].shape == arrays['global_batches'].shape == (3, n)
    assert arrays['batches'].dtype.kind in 'iu' and arrays['global_batches'].dtype.kind in 'iu'
    assert np.all(arrays['batches'] >= 0) and np.all(arrays['global_batches'] >= 0)
    assert np.array_equal(arrays['global_batches'], arrays['batches'] + folds[None, :] * 1000000)
    for mode in ('inferred', 'oracle'):
        labels = arrays[mode]
        cluster_size = arrays['cluster_size_' + mode]
        eligible = arrays['eligible_' + mode]
        assert labels.shape == cluster_size.shape == eligible.shape == (3, n)
        assert labels.dtype.kind in 'iu' and np.all(labels >= 0)
        assert eligible.dtype.kind == 'b'
        for scenario in range(3):
            for fold in range(1, 6):
                rows = folds == fold
                expected_size = np.bincount(labels[scenario, rows])[labels[scenario, rows]]
                assert np.array_equal(cluster_size[scenario, rows], expected_size)
                # Original C assigns a cluster wholly within one batch.
                for label in np.unique(labels[scenario, rows]):
                    selected = rows & (labels[scenario] == label)
                    assert len(np.unique(arrays['batches'][scenario, selected])) == 1
        assert np.array_equal(eligible, (cluster_size >= 2) & (cluster_size <= 18))
    reconciled = {}
    for key in KEYS:
        mode = 'oracle' if key.startswith('oracle_') else 'inferred'
        mask = arrays['eligible_' + mode]
        probability = arrays['centered_probability_' + key]
        centered_log = arrays['centered_log_probability_' + key]
        evidence = arrays['component_log_evidence_' + key]
        prediction = arrays['prediction_' + key]
        assert probability.shape == centered_log.shape == evidence.shape == (3, n, c)
        assert prediction.shape == (3, n) and prediction.dtype.kind in 'iu'
        assert arrays['component_coverage_'+key].shape == (3,n) and np.all(arrays['component_coverage_'+key] == 1)
        assert np.isfinite(probability).all() and np.isfinite(centered_log).all() and np.isfinite(evidence).all()
        assert np.all(probability[~mask] == 0) and np.all(centered_log[~mask] == 0)
        assert np.all((probability[mask] >= 0) & (probability[mask] <= 1))
        assert np.allclose(probability[mask].sum(-1), 1, rtol=0, atol=1e-12)
        for scenario in range(3):
            for fold in range(1,6):
                rows = (folds == fold) & mask[scenario]
                if rows.any():
                    assert np.array_equal(centered_log[scenario,rows],np.log(np.clip(probability[scenario,rows],1e-9,1)))
        replay = .05 * plain_log.copy()
        replay[mask] += .25 * centered_log[mask]
        assert np.array_equal(replay, evidence), (key, 'component evidence replay differs')
        assert np.array_equal(replay.argmax(-1), prediction), (key, 'component prediction replay differs')
        reconciled[key] = []
        for scenario in range(3):
            summary = reconcile_predictions(y, prediction[scenario], classes)
            if scores is not None:
                assert np.isclose(summary['balanced_accuracy'], scores[key][scenario], rtol=0, atol=1e-12)
            reconciled[key].append(summary)
    return reconciled


def preserve_active_failure(error):
    """Call from the notebook runpy wrapper before reraising a producer error."""
    if ACTIVE_CAPTURE is not None:
        ACTIVE_CAPTURE.failure(error)


class CanonicalCapture:
    def __init__(self, output, input_root, source_path):
        global ACTIVE_CAPTURE
        self.out, self.input_root, self.source_path = Path(output), Path(input_root), Path(source_path)
        assert not self.out.exists() or not any(self.out.iterdir()), 'Refusing to overwrite an existing Stage 06E trial'
        self.out.mkdir(parents=True, exist_ok=True)
        self.started = time.monotonic()
        self.arrays = {}
        self.inputs = []
        self.head_parameters = {}
        self.partition_records = []
        self.fold_records = []
        self.emitted_warnings = []
        self.fit_active = False
        self.fit_count = 0
        self.completed = False
        self.runtime = None
        self.freeze = None
        self._last_centered = None
        self._prior_showwarning = warnings.showwarning
        def capture_warning(message, category, filename, lineno, file=None, line=None):
            self.emitted_warnings.append(dict(category=category.__name__, message=str(message),
                                             filename=str(filename), line=int(lineno), during_fit=self.fit_active))
            return self._prior_showwarning(message, category, filename, lineno, file=file, line=line)
        warnings.showwarning = capture_warning
        ACTIVE_CAPTURE = self
        self.status('CREATED_NO_FITS')

    def status(self, status, **extra):
        write_json(self.out / 'status.json', dict(status=status, observed_utc=datetime.now(timezone.utc).isoformat(),
                   elapsed_seconds=time.monotonic() - self.started, fitted_heads=self.fit_count, **extra))

    def _source_preflight(self):
        freeze_path = Path(os.environ['NTL_STAGE06E_FREEZE'])
        self.freeze = json.loads(freeze_path.read_text())
        expected = {'capture_source_sha256': sha256(self.source_path),
                    'cache_checks_source_sha256': sha256(Path(__file__)),
                    'original_source_sha256': ORIGINAL_SHA256, 'plan_sha256': PLAN_SHA256}
        for key, value in expected.items():
            assert self.freeze[key] == value, ('Pre-fit frozen source discrepancy', key, self.freeze.get(key), value)
        original = self.source_path.with_name('stage06c_mixed_confirmation.py')
        assert sha256(original) == ORIGINAL_SHA256, 'Original C source not byte identical'
        assert self.freeze['input_version_ids'] == EXPECTED_INPUT_VERSIONS
        assert self.freeze['accelerator'] == 'none' and self.freeze['internet_enabled'] is False
        self.freeze_sha256 = sha256(freeze_path)

    def _runtime_preflight(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from threadpoolctl import threadpool_info
        actual = {'python': platform.python_version(), **{name: version(name) for name in EXPECTED_RUNTIME if name != 'python'}}
        pools = threadpool_info()
        parameters = dict(scaler=StandardScaler().get_params(deep=True),
                          classifier=LogisticRegression(C=.005,max_iter=5000).get_params(deep=True))
        self.runtime = dict(python=actual['python'], packages={k:v for k,v in actual.items() if k != 'python'}, platform=platform.platform(), python_executable=sys.executable,
                            threadpools=pools, resolved_estimator_parameters=parameters,
                            environment={k: os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')})
        write_json(self.out / 'runtime_preflight.json', self.runtime)
        assert actual == EXPECTED_RUNTIME, ('Pre-fit numeric environment mismatch', actual, EXPECTED_RUNTIME)
        assert pools and any(p.get('user_api') == 'blas' for p in pools), 'No observed BLAS threadpool'
        assert all(p.get('num_threads') == 2 for p in pools), ('Pre-fit threadpool mismatch', pools)
        assert all(self.runtime['environment'][k] == '2' for k in self.runtime['environment'])

    def initialize(self, ids, classes, y, groups, folds, splits, reference_folds, train, deb, glm):
        self._source_preflight()
        self._runtime_preflight()
        self.reference_folds = dict(reference_folds)
        self.splits = [(fit.copy(), valid.copy()) for fit, valid in splits]
        n, c = len(ids), len(classes)
        assert n == 2699 and c == 18 and len(np.unique(groups)) == 150 and len(splits) == 5
        self.arrays.update(ids=np.asarray(ids,dtype=str).copy(), classes=np.asarray(classes,dtype=str).copy(),
                           groups=np.asarray(groups,dtype=str).copy(), y=y.copy(), fold=folds.copy(), scenarios=np.array([1,3,6]))
        self.arrays['plain_probability'] = np.full((3,n,c), np.nan)
        self.arrays['plain_log_probability'] = np.full((3,n,c), np.nan)
        self.arrays['plain_coverage'] = np.zeros((3,n), dtype=np.int64)
        self.arrays['context_coverage'] = np.zeros((3,n), dtype=np.int64)
        for key in ('batches','global_batches','inferred','oracle','cluster_size_inferred','cluster_size_oracle'):
            self.arrays[key] = np.full((3,n), -1, dtype=np.int64)
        for key in ('eligible_inferred','eligible_oracle'):
            self.arrays[key] = np.zeros((3,n), dtype=bool)
        for key in KEYS:
            self.arrays['centered_probability_'+key] = np.zeros((3,n,c))
            self.arrays['centered_log_probability_'+key] = np.zeros((3,n,c))
            self.arrays['component_log_evidence_'+key] = np.full((3,n,c),np.nan)
            self.arrays['prediction_'+key] = np.full((3,n),-1,dtype=np.int64)
            self.arrays['component_coverage_'+key] = np.zeros((3,n),dtype=np.int64)
        for path, expected in ((train,'40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707'),
                               (deb,'11eed58762ad128f97457dde9cc4ab03c7932e1b4b04845ec6664c834b1b1dda'),
                               (glm,'2b41585a9482cef825962a8cc096697933c0b1237e8ede0712d1f8ab361d9414')):
            actual = sha256(path)
            assert actual == expected
            self.inputs.append(dict(path=str(path),sha256=actual,bytes=path.stat().st_size))
        self.historical_root = Path(glm).parent
        for filename in ('glm_manifest.json','paired_predictions.npz','centering_result.json'):
            path = self.historical_root / filename
            assert path.is_file(), ('Missing historical C artifact',str(path))
            self.inputs.append(dict(path=str(path),sha256=sha256(path),bytes=path.stat().st_size))
        manifest = json.loads((self.historical_root/'glm_manifest.json').read_text())
        result = json.loads((self.historical_root/'centering_result.json').read_text())
        assert manifest['status'] == 'COMPLETE' and manifest['cache_sha256'] == sha256(glm)
        assert manifest['input_provenance']['train_sha256'] == sha256(train) and manifest['layers'] == [8,3,24]
        assert result['train_sha256'] == sha256(train) and result['cache_sha256'] == sha256(deb)
        assert result['glm_cache_sha256'] == sha256(glm) and result['scenarios'] == [1,3,6]
        self.historical_result = result
        for number,(fit,valid) in enumerate(splits):
            assert not set(groups[fit]) & set(groups[valid])
            assert np.all(folds[valid] == number+1)
            self.fold_records.append(dict(fold=number+1,fit_rows=len(fit),valid_rows=len(valid),
                                          train_prompts=np.unique(groups[fit]).tolist(),
                                          valid_prompts=np.unique(groups[valid]).tolist()))
        write_json(self.out/'input_artifacts.json',self.inputs)
        write_json(self.out/'source_preflight.json',dict(freeze=self.freeze,freeze_sha256=self.freeze_sha256,
                   capture_source_sha256=sha256(self.source_path),cache_checks_source_sha256=sha256(Path(__file__)),
                   original_source_sha256=ORIGINAL_SHA256))
        self.status('PREFLIGHT_PASS_NO_FITS')

    def begin_fit(self):
        assert self.runtime is not None and self.freeze is not None
        assert not self.fit_active and self.fit_count < 15
        self.fit_active = True

    def end_fit(self):
        assert self.fit_active
        self.fit_active = False
        self.fit_count += 1

    def training_partition(self, number, size, fit, valid, grouping):
        assert not set(self.arrays['groups'][fit]) & set(self.arrays['groups'][valid])
        prefix = f'training_f{number}_size{size}__'
        self.arrays[prefix+'rows'] = fit.copy()
        self.arrays[prefix+'assignments'] = grouping.copy()
        for label in np.unique(grouping):
            assert len(np.unique(self.arrays['groups'][fit[grouping==label]])) == 1
        self.partition_records.append(dict(fold=number+1,target=size,rows=len(fit),groups=len(np.unique(grouping)),
                                          assignments_array=prefix+'assignments',rows_array=prefix+'rows'))

    def head(self, number, label, head, fit, valid):
        scaler, classifier = head
        assert label in ('plain','control','candidate')
        prefix = f'head_f{number}_{label}__'
        attributes = dict(scaler_mean=scaler.mean_,scaler_scale=scaler.scale_,scaler_var=scaler.var_,
                          scaler_n_samples_seen=np.asarray(scaler.n_samples_seen_),clf_coef=classifier.coef_,
                          clf_intercept=classifier.intercept_,clf_classes=classifier.classes_,clf_n_iter=classifier.n_iter_,
                          fit_rows=fit,valid_rows=valid)
        for key,value in attributes.items():
            self.arrays[prefix+key] = np.asarray(value).copy()
        self.head_parameters[prefix] = dict(scaler=scaler.get_params(deep=True),classifier=classifier.get_params(deep=True))
        assert np.array_equal(classifier.classes_,np.arange(18))
        assert not set(self.arrays['groups'][fit]) & set(self.arrays['groups'][valid])
        # Persist fitted state early so a later failure cannot erase the expensive fits.
        path = self.out / f'head_f{number}_{label}.npz'
        payload = {k:self.arrays[prefix+k] for k in attributes}
        np.savez_compressed(path,**payload)
        exact_saved_arrays(path,payload)
        write_json(self.out/'head_parameters.json',self.head_parameters)
        self.status('FITTED_HEADS_CAPTURED',last_head=f'{number+1}:{label}')

    def context(self, scenario, number, valid, batches, inferred, oracle):
        self.arrays['context_coverage'][scenario,valid] += 1
        self.arrays['batches'][scenario,valid] = batches
        self.arrays['global_batches'][scenario,valid] = batches+(number+1)*1000000
        for mode,labels in (('inferred',inferred),('oracle',oracle)):
            sizes = np.bincount(labels)[labels]
            self.arrays[mode][scenario,valid] = labels
            self.arrays['cluster_size_'+mode][scenario,valid] = sizes
            self.arrays['eligible_'+mode][scenario,valid] = (sizes>=2)&(sizes<=18)

    def plain(self, scenario, rows, probability, log_probability):
        self.arrays['plain_probability'][scenario,rows] = probability
        self.arrays['plain_log_probability'][scenario,rows] = log_probability
        self.arrays['plain_coverage'][scenario,rows] += 1

    def last_centered(self, probability, log_probability):
        self._last_centered = (probability.copy(),log_probability.copy())

    def component(self, key, scenario, valid, eligible, evidence, prediction):
        mode = 'oracle' if key.startswith('oracle_') else 'inferred'
        assert np.array_equal(eligible,self.arrays['eligible_'+mode][scenario,valid])
        if eligible.any():
            assert self._last_centered is not None
            probability,log_probability = self._last_centered
            assert probability.shape == log_probability.shape == (int(eligible.sum()),len(self.arrays['classes']))
            self.arrays['centered_probability_'+key][scenario,valid[eligible]] = probability
            self.arrays['centered_log_probability_'+key][scenario,valid[eligible]] = log_probability
        self.arrays['component_log_evidence_'+key][scenario,valid] = evidence
        self.arrays['prediction_'+key][scenario,valid] = prediction
        self.arrays['component_coverage_'+key][scenario,valid] += 1
        self._last_centered = None

    def _historical(self):
        result = []
        historical = self.historical_root/'paired_predictions.npz'
        affected = set()
        repeated = 0
        with np.load(historical,allow_pickle=False) as archive:
            for key in ('ids','y','groups','fold'):
                assert np.array_equal(archive[key],self.arrays[key]), ('Historical coordinate mismatch',key)
            # Historical classes are object dtype. Never deserialize them. Ordered labels
            # are bound by official train hash, original source hash and manifest provenance.
            for key in KEYS:
                old = archive[key]
                current = self.arrays['prediction_'+key]
                assert old.shape == current.shape
                for scenario,size in enumerate((1,3,6)):
                    mask = old[scenario] != current[scenario]
                    mismatch = int(mask.sum())
                    repeated += mismatch
                    affected.update(self.arrays['ids'][mask].tolist())
                    old_summary = reconcile_predictions(self.arrays['y'],old[scenario],self.arrays['classes'])
                    assert np.isclose(old_summary['balanced_accuracy'],self.historical_result['scores'][key][scenario],rtol=0,atol=1e-12)
                    result.append(dict(head=key,scenario=size,mismatches=mismatch,
                                       mismatch_ids=self.arrays['ids'][mask].tolist(),
                                       historical=old_summary,current=reconcile_predictions(self.arrays['y'],current[scenario],self.arrays['classes']),
                                       folds=[dict(fold=fold,mismatches=int((mask&(self.arrays['fold']==fold)).sum()),
                                                   historical=reconcile_predictions(self.arrays['y'][self.arrays['fold']==fold],old[scenario,self.arrays['fold']==fold],self.arrays['classes']),
                                                   current=reconcile_predictions(self.arrays['y'][self.arrays['fold']==fold],current[scenario,self.arrays['fold']==fold],self.arrays['classes'])) for fold in range(1,6)]))
        return dict(status='SEPARATE_HISTORICAL_COMPARISON_NOT_AN_INTERNAL_GATE',
                    historical_prediction_sha256=sha256(historical),repeated_comparison_cells=12*len(self.arrays['ids']),
                    repeated_mismatches=repeated,unique_affected_ids=len(affected),affected_ids=sorted(affected),comparisons=result,
                    interpretation='New canonical CPU development reference; any historical discrepancy remains unresolved. Stage06D retains its failed gate.')

    def finalize(self,predictions,scores,warning_records,convergence,id_checks,fold_scores,group_diagnostics):
        self.status('VALIDATING_CAPTURE')
        assert sha256(self.source_path) == self.freeze['capture_source_sha256']
        assert sha256(Path(__file__)) == self.freeze['cache_checks_source_sha256']
        assert sha256(self.source_path.with_name('stage06c_mixed_confirmation.py')) == ORIGINAL_SHA256
        assert all(sha256(item['path']) == item['sha256'] for item in self.inputs), 'Input changed during capture'
        assert self.fit_count == 15 and len(self.head_parameters) == 15 and len(self.partition_records) == 15
        assert not self.fit_active
        for key in KEYS:
            assert np.all(self.arrays['component_coverage_'+key] == 1)
            assert np.array_equal(self.arrays['prediction_'+key],predictions[key])
        for prefix in self.head_parameters:
            assert self.arrays[prefix+'clf_coef'].shape == (18,2048)
            assert self.arrays[prefix+'clf_intercept'].shape == (18,)
            assert self.arrays[prefix+'scaler_mean'].shape == self.arrays[prefix+'scaler_scale'].shape == self.arrays[prefix+'scaler_var'].shape == (2048,)
            assert np.isfinite(self.arrays[prefix+'clf_coef']).all()
            assert np.isfinite(self.arrays[prefix+'clf_intercept']).all()
            assert np.isfinite(self.arrays[prefix+'scaler_mean']).all() and np.isfinite(self.arrays[prefix+'scaler_scale']).all() and np.isfinite(self.arrays[prefix+'scaler_var']).all()
        reconciled = validate_canonical_arrays(self.arrays,scores,self.reference_folds)
        all_fit_warnings = list(warning_records)+[{k:v for k,v in w.items() if k!='during_fit'} for w in self.emitted_warnings if w['during_fit']]
        non_deprecation = [w for w in all_fit_warnings if w['category'] != 'DeprecationWarning']
        write_json(self.out/'warnings.json',dict(original_caught=warning_records,emitted=self.emitted_warnings,fit_warning_records=all_fit_warnings,non_deprecation=non_deprecation))
        write_json(self.out/'fold_and_training_proof.json',dict(folds=self.fold_records,partitions=self.partition_records))
        write_json(self.out/'same_run_reconciliation.json',dict(scores=scores,class_confusions=reconciled,fold_scores=fold_scores))
        cache_path=self.out/'canonical_cpu_evidence.npz'
        np.savez_compressed(cache_path,**self.arrays)
        exact_saved_arrays(cache_path,self.arrays)
        # Re-run solely array checks against the serialized values; never invoke a head.
        with np.load(cache_path,allow_pickle=False) as archive:
            readback={key:archive[key] for key in archive.files}
        validate_canonical_arrays(readback,scores,self.reference_folds)
        cache_sha=sha256(cache_path)
        historical=self._historical()
        write_json(self.out/'historical_comparison.json',historical)
        assert not convergence and not non_deprecation, ('Non-deprecation fit warnings',non_deprecation,convergence)
        gate=dict(status='PASS',canonical_cache_sha256=cache_sha,all_saved_arrays_exact_readback=True,
                  component_predictions_exact=True,component_scores_atol_1e_12=True,warnings_non_deprecation=0,
                  fitted_heads=15,internal_reference='same-run captured original-script evidence',
                  historical_comparison_is_separate=True,historical_repeated_mismatches=historical['repeated_mismatches'],
                  historical_unique_affected_ids=historical['unique_affected_ids'])
        write_json(self.out/'cache_gate.json',gate)
        arrays={key:dict(dtype=str(value.dtype),shape=list(value.shape),sha256=hashlib.sha256(value.tobytes(order='C')).hexdigest()) for key,value in self.arrays.items()}
        manifest=dict(schema_version=SCHEMA_VERSION,cache_file=cache_path.name,cache_sha256=cache_sha,
                      internal_gate_file='cache_gate.json',internal_gate_sha256=sha256(self.out/'cache_gate.json'),
                      source_sha256=sha256(self.source_path),cache_checks_source_sha256=sha256(Path(__file__)),capture_helper_sha256=sha256(Path(__file__)),
                      original_source_sha256=ORIGINAL_SHA256,plan_sha256=PLAN_SHA256,freeze_sha256=self.freeze_sha256,
                      input_hashes=dict(train_sha256=self.inputs[0]['sha256'],deb_cache_sha256=self.inputs[1]['sha256'],glm_cache_sha256=self.inputs[2]['sha256']),
                      input_artifacts=self.inputs,input_version_ids=EXPECTED_INPUT_VERSIONS,arrays=arrays,runtime=self.runtime,
                      fit_parameters=self.head_parameters,warning_records=all_fit_warnings,component_scores=scores,
                      inactive_centered_arrays='Zero storage is inactive log-evidence contribution; it is not a normalized probability.',
                      historical_comparison_is_separate=True,complete=True)
        write_json(self.out/'cache_manifest.json',manifest)
        self.completed=True
        warnings.showwarning=self._prior_showwarning
        self.status('COMPLETE_CANONICAL_CPU_EVIDENCE',cache_sha256=cache_sha,internal_gate='PASS',
                    historical_repeated_mismatches=historical['repeated_mismatches'])
        print('STAGE06E_CACHE_GATE',json.dumps(gate),flush=True)
        print('STAGE06E_HISTORICAL_COMPARISON',json.dumps({k:v for k,v in historical.items() if k not in ('comparisons','affected_ids')}),flush=True)
        print('STAGE06E_CACHE_MANIFEST',json.dumps(manifest),flush=True)

    def failure(self,error):
        if self.completed:
            return
        # Preserve partial numeric/string arrays separately. This is never a valid cache.
        if self.arrays:
            safe={k:v for k,v in self.arrays.items() if not v.dtype.hasobject and v.dtype.kind in 'biufcU'}
            np.savez_compressed(self.out/'FAILED_partial_capture.npz',**safe)
        write_json(self.out/'FAILED.json',dict(status='FAILED_CANONICAL_CPU_EVIDENCE',error_type=type(error).__name__,error=str(error),
                   fitted_heads=self.fit_count,source_sha256=sha256(self.source_path),runtime=self.runtime,
                   fit_parameters=self.head_parameters,emitted_warnings=self.emitted_warnings,
                   successful_ensemble_evaluation=False))
        self.status('FAILED_CANONICAL_CPU_EVIDENCE',error_type=type(error).__name__,error=str(error))
        warnings.showwarning=self._prior_showwarning
