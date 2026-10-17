#!/usr/bin/env python3
"""Prepare the fixed S02 stylegram member's prompt-held-out evidence.

This is an available-component development artifact, not a full S02 evaluation,
calibrated probability model, new candidate search, or official submission.
Use --self-check for transform/wiring checks only; it never fits a model.

Source attribution: Lucas Boesen, public "LoRA x adaptive centering" notebook,
version 359126546, Apache License 2.0, section 1 (style-aware n-grams).
https://www.kaggle.com/code/lucasboesen/lora-x-adaptive-centering?scriptVersionId=359126546
This adaptation changes evaluation to fold-local held-out predictions and adds
provenance/output checks; it preserves the fixed feature/classifier recipe.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import time
import warnings

import numpy as np
import pandas as pd
import scipy
from scipy.sparse import hstack
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.svm import LinearSVC


TRAIN_SHA256 = "40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707"
VENDOR_SHA256 = "261e27094fbefbfe8e6496feda4a274f8e9a8587a19b304da074a71f49cbc2f3"
LABELS = sorted([
    'gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',
    'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
    'claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5',
    'claude-sonnet-5', 'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3',
])
FUNCTION_WORDS = set("a an the and or but if while although because as so than that which who whom whose this these those it its they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from with without for of by over under between through into about is are was were be been being have has had do does did can could will would shall should may might must not no nor all any some each every both either neither other another such only also just very too more most less least here there when where why how what whether then now still yet even much many few little".split())
SPEC = [('char', 3, 5, False, 250000, 1.0), ('word', 1, 3, False, 180000, 0.7),
        ('shape', 2, 6, False, 60000, 0.5), ('function', 1, 3, False, 50000, 0.5)]
TOKEN_PATTERN = r"(?u)\b\w+\b|[^\w\s]"
# Exact resolved defaults of sklearn 1.8.0, with only the source's C/seed/limit changed.
SVC_PARAMS = dict(C=0.5, class_weight=None, dual='auto', fit_intercept=True,
                  intercept_scaling=1, loss='squared_hinge', max_iter=4000,
                  multi_class='ovr', penalty='l2', random_state=42, tol=0.0001, verbose=0)


def shape_text(text: str) -> str:
    return re.sub(r"[^\W\d_]+", lambda m: "X" if m[0].isupper() else "x", text)


def function_text(text: str) -> str:
    return " ".join(t.lower() if t.lower() in FUNCTION_WORDS or not t.isalnum() else "CONTENT"
                    for t in re.findall(r"\w+|[^\w\s]", text))


def vectorizer_spec(spec: tuple) -> tuple[TfidfVectorizer, object, float]:
    analyzer, lo, hi, lower, maxf, weight = spec
    transform = shape_text if analyzer == 'shape' else function_text if analyzer == 'function' else str
    actual = 'char' if analyzer == 'shape' else 'word' if analyzer == 'function' else analyzer
    options = dict(analyzer=actual, ngram_range=(lo, hi), lowercase=lower,
                   sublinear_tf=True, min_df=2, max_features=maxf, dtype=np.float64)
    if actual == 'word':
        options['token_pattern'] = TOKEN_PATTERN
    return TfidfVectorizer(**options), transform, weight


def self_check() -> dict:
    """Fixed transform/token wiring fixtures, no fit or vectorizer vocabulary learning."""
    text = "THE Cat, and 42 naïve NASA! foo_bar can't"
    assert shape_text(text) == "X x, x 42 x X! x_x x'x"
    assert function_text(text) == "the CONTENT , and CONTENT CONTENT CONTENT ! foo_bar can ' CONTENT"
    assert function_text("I, YOU; their 123 _ …") == "i , you ; their CONTENT _ …"
    defaults = LinearSVC(C=0.5, random_state=42, max_iter=4000).get_params()
    assert defaults == SVC_PARAMS, "Installed LinearSVC defaults differ from fixed sklearn 1.8.0 recipe"
    for spec in SPEC:
        vectorizer, transform, weight = vectorizer_spec(spec)
        params = vectorizer.get_params()
        assert params['lowercase'] is False and params['norm'] == 'l2'
        assert params['sublinear_tf'] and params['min_df'] == 2 and params['dtype'] is np.float64
        assert params['use_idf'] and params['smooth_idf'] and params['binary'] is False
        assert params['ngram_range'] == spec[1:3] and weight == spec[-1]
        if params['analyzer'] == 'word':
            assert vectorizer.build_tokenizer()("word, don't 123 _!") == ['word', ',', 'don', "'", 't', '123', '_', '!']
        assert isinstance(transform(text), str)
    return {'status': 'PASSED', 'checks': ['source transforms', 'punctuation-aware tokenization',
            'four-block vectorizer wiring', 'sklearn 1.8.0 classifier defaults'], 'model_fits': 0}


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def strings_digest(values: np.ndarray) -> str:
    return hashlib.sha256(json.dumps(values.tolist(), ensure_ascii=False,
                                     separators=(',', ':')).encode('utf-8')).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def validate_inputs(data_dir: Path, reference_dir: Path) -> tuple:
    train_path = data_dir / 'train.csv'
    assert digest(train_path) == TRAIN_SHA256, 'Official training file SHA mismatch'
    references = {name: reference_dir / name for name in [
        'baseline_oof.csv', 'baseline_folds.csv', 'baseline_metrics.json', 'data_audit.json']}
    baseline_metrics = json.loads(references['baseline_metrics.json'].read_text())
    audit = json.loads(references['data_audit.json'].read_text())
    assert baseline_metrics['source_sha256']['train.csv'] == TRAIN_SHA256
    assert audit['sha256']['train.csv'] == TRAIN_SHA256
    train = pd.read_csv(train_path, dtype=str)
    assert {'id', 'prompt_id', 'model', 'text'} <= set(train.columns)
    assert not train[['id', 'prompt_id', 'model', 'text']].isna().any().any()
    assert train['id'].is_unique and not train['text'].str.strip().eq('').any()
    assert len(train) == 2699 and train['prompt_id'].nunique() == 150
    assert sorted(train['model'].unique()) == LABELS
    assert not train.duplicated(['prompt_id', 'model']).any()
    assert train['model'].value_counts().sort_index().to_dict() == audit['audit']['class_counts']
    ids = train['id'].to_numpy(dtype=str)
    groups = train['prompt_id'].to_numpy(dtype=str)
    classes = np.asarray(LABELS, dtype=str)
    y = np.asarray([LABELS.index(label) for label in train['model']], dtype=np.int64)
    assert np.array_equal(classes[y], train['model'].to_numpy(dtype=str))
    previous = pd.read_csv(references['baseline_oof.csv'], dtype={'id': str, 'prompt_id': str, 'model': str})
    assert len(previous) == len(train) and previous['id'].is_unique
    assert set(previous['id']) == set(ids)
    previous = previous.set_index('id').loc[ids].reset_index()
    assert np.array_equal(previous['prompt_id'].to_numpy(dtype=str), groups)
    assert np.array_equal(previous['model'].to_numpy(dtype=str), classes[y])
    historical_folds = previous['fold'].to_numpy(dtype=np.int64)
    assert set(historical_folds) == set(range(1, 6))
    fold_summary = pd.read_csv(references['baseline_folds.csv']).set_index('fold')
    assert len(fold_summary) == 5 and set(fold_summary.index) == set(range(1, 6))
    splits = list(GroupKFold(n_splits=5).split(train['text'], y, groups))
    recomputed = np.full(len(train), -1, dtype=np.int64)
    coverage = np.zeros(len(train), dtype=np.int64)
    records = []
    for fold, (tr, va) in enumerate(splits, start=1):
        recomputed[va] = fold
        coverage[va] += 1
        assert set(groups[tr]).isdisjoint(set(groups[va])), 'Prompt leakage'
        assert np.array_equal(np.unique(y[tr]), np.arange(len(classes)))
        assert np.array_equal(np.unique(y[va]), np.arange(len(classes)))
        assert len(va) == int(fold_summary.loc[fold, 'validation_rows'])
        assert len(np.unique(groups[va])) == int(fold_summary.loc[fold, 'validation_prompts'])
        historical_result = baseline_metrics['fold_results'][fold - 1]
        assert historical_result['fold'] == fold and historical_result['validation_rows'] == len(va)
        assert historical_result['validation_prompts'] == len(np.unique(groups[va]))
        records.append({'fold': fold, 'training_rows': len(tr), 'validation_rows': len(va),
                        'training_prompts': sorted(set(groups[tr])),
                        'validation_prompts': sorted(set(groups[va])), 'prompt_overlap': 0})
    assert (coverage == 1).all() and np.array_equal(recomputed, historical_folds)
    provenance = {'train_sha256': TRAIN_SHA256, 'reference_sha256': {name: digest(path) for name, path in references.items()},
                  'rows': len(train), 'prompts': len(set(groups)), 'classes': classes.tolist(),
                  'id_order_sha256': strings_digest(ids), 'prompt_order_sha256': strings_digest(groups),
                  'label_order_sha256': strings_digest(classes[y]), 'fold_mapping_sha256': strings_digest(recomputed),
                  'mapping_authority': 'baseline_oof.csv, aligned by unique official id',
                  'fold_count_authority': 'baseline_folds.csv and baseline_metrics.json',
                  'recomputed_GroupKFold_mapping_exact_match': True, 'validation_rows_seen_exactly_once': True,
                  'folds': records}
    return train, ids, groups, classes, y, historical_folds, splits, provenance


def run(data_dir: Path, reference_dir: Path, output_dir: Path) -> dict:
    started = time.perf_counter()
    check = self_check()
    assert sklearn.__version__ == '1.8.0', 'Run the fixed artifact under sklearn 1.8.0'
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f'Refusing to overwrite nonempty output: {output_dir}')
    train, ids, groups, classes, y, folds, splits, provenance = validate_inputs(data_dir, reference_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / 'status.json', {'status': 'RUNNING', 'started_utc': datetime.now(timezone.utc).isoformat()})
    write_json(output_dir / 'id_provenance.json', provenance)
    probabilities = np.full((len(train), len(classes)), np.nan, dtype=np.float64)
    scores = np.full_like(probabilities, np.nan)
    coverage = np.zeros(len(train), dtype=np.int64)
    fold_records, warning_records = [], []
    texts = train['text'].tolist()
    try:
        for fold, (tr, va) in enumerate(splits, start=1):
            fold_started = time.perf_counter()
            training_blocks, validation_blocks, block_records = [], [], []
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                for spec in SPEC:
                    vectorizer, transform, weight = vectorizer_spec(spec)
                    training = vectorizer.fit_transform([transform(texts[i]) for i in tr]) * weight
                    validation = vectorizer.transform([transform(texts[i]) for i in va]) * weight
                    assert training.dtype == np.float64 and validation.dtype == np.float64
                    assert training.shape[1] == validation.shape[1] and training.shape[1] > 0
                    training_blocks.append(training)
                    validation_blocks.append(validation)
                    params = vectorizer.get_params()
                    params['dtype'] = 'float64'
                    params['ngram_range'] = list(params['ngram_range'])
                    block_records.append({'name': spec[0], 'weight': weight, 'features': training.shape[1],
                                          'parameters': params, 'vocabulary_and_idf_training_prompts_only': True})
                a = hstack(training_blocks, format='csr')
                b = hstack(validation_blocks, format='csr')
                classifier = LinearSVC(**SVC_PARAMS).fit(a, y[tr])
                assert np.array_equal(classifier.classes_, np.arange(len(classes))), 'Numeric classifier label order mismatch'
                decision = np.asarray(classifier.decision_function(b), dtype=np.float64)
                assert decision.shape == (len(va), len(classes)) and np.isfinite(decision).all()
                exponents = np.exp(decision - decision.max(1, keepdims=True))
                p = np.zeros_like(decision)
                p[:, classifier.classes_] = exponents / exponents.sum(1, keepdims=True)
            assert np.isfinite(p).all() and (p >= 0).all() and np.allclose(p.sum(1), 1, atol=1e-14, rtol=0)
            probabilities[va] = p
            scores[va] = decision
            coverage[va] += 1
            fold_warnings = [{'fold': fold, 'category': warning.category.__name__,
                              'message': str(warning.message), 'filename': str(warning.filename),
                              'lineno': int(warning.lineno),
                              'is_convergence_warning': issubclass(warning.category, ConvergenceWarning)} for warning in caught]
            warning_records.extend(fold_warnings)
            record = {'fold': fold, 'training_rows': len(tr), 'validation_rows': len(va),
                      'training_prompts': len(set(groups[tr])), 'validation_prompts': len(set(groups[va])),
                      'prompt_overlap': 0, 'balanced_accuracy': float(balanced_accuracy_score(y[va], p.argmax(1))),
                      'feature_count': a.shape[1], 'feature_blocks': block_records,
                      'classifier_parameters': classifier.get_params(),
                      'dual_auto_resolved': bool(a.shape[0] < a.shape[1]),
                      'dual_resolution_basis': 'sklearn 1.8.0 auto with l2/squared_hinge/ovr; both primal and dual supported',
                      'n_iter': int(classifier.n_iter_), 'warning_count': len(fold_warnings),
                      'seconds': time.perf_counter() - fold_started}
            fold_records.append(record)
            print('STYLEGRAM_FOLD ' + json.dumps({k: record[k] for k in ['fold', 'balanced_accuracy', 'seconds', 'feature_count', 'n_iter', 'warning_count']}), flush=True)
            del classifier, training_blocks, validation_blocks, a, b, vectorizer
        assert (coverage == 1).all() and np.isfinite(probabilities).all() and np.isfinite(scores).all()
        assert np.allclose(probabilities.sum(1), 1, atol=1e-14, rtol=0)
        # Source add() floors before log, without re-normalizing those floored values.
        logp = np.log(np.clip(probabilities, 1e-9, 1))
        prediction = probabilities.argmax(1).astype(np.int64)
        cm = confusion_matrix(y, prediction, labels=np.arange(len(classes)))
        support = cm.sum(1)
        recall = cm.diagonal() / support
        score = float(balanced_accuracy_score(y, prediction))
        assert cm.sum() == len(train) and np.isclose(score, recall.mean(), atol=1e-15, rtol=0)
        npz_path = output_dir / 'stylegram_oof.npz'
        np.savez_compressed(npz_path, ids=ids, classes=classes, prompt_ids=groups, fold=folds, y=y,
                            probabilities=probabilities, log_probabilities=logp,
                            decision_scores=scores, prediction=prediction, train_sha256=np.asarray(TRAIN_SHA256))
        with np.load(npz_path, allow_pickle=False) as saved:
            for key in saved.files:
                assert not saved[key].dtype.hasobject
            assert np.array_equal(saved['probabilities'], probabilities)
            assert np.array_equal(saved['log_probabilities'], logp)
            assert np.array_equal(saved['ids'], ids) and np.array_equal(saved['classes'], classes)
        table = pd.DataFrame({'id': ids, 'prompt_id': groups, 'model': classes[y], 'fold': folds,
                              'y': y, 'prediction': classes[prediction]})
        probability_columns = ['prob__' + label for label in classes]
        table[probability_columns] = probabilities
        table.to_csv(output_dir / 'stylegram_oof_probabilities.csv', index=False, float_format='%.17g')
        roundtrip = pd.read_csv(output_dir / 'stylegram_oof_probabilities.csv', dtype={'id': str}, float_precision='round_trip')
        assert np.array_equal(roundtrip['id'].to_numpy(dtype=str), ids)
        assert np.array_equal(roundtrip[probability_columns].to_numpy(dtype=np.float64), probabilities)
        pd.DataFrame(cm, index=classes, columns=classes).rename_axis('true_class').to_csv(output_dir / 'confusion_matrix.csv')
        pd.DataFrame({'class': classes, 'support': support, 'correct': cm.diagonal(), 'recall': recall}).to_csv(
            output_dir / 'per_class_recall.csv', index=False, float_format='%.17g')
        pd.DataFrame([{key: record[key] for key in ['fold', 'training_rows', 'validation_rows', 'training_prompts',
                     'validation_prompts', 'balanced_accuracy', 'feature_count', 'n_iter', 'warning_count', 'seconds']}
                     for record in fold_records]).to_csv(output_dir / 'fold_metrics.csv', index=False, float_format='%.17g')
        convergence_count = sum(record['is_convergence_warning'] for record in warning_records)
        write_json(output_dir / 'warnings.json', warning_records)
        result = {'status': 'COMPLETE_FIXED_STYLEGRAM_OOF', 'experiment': 'stage06d_fixed_s02_stylegram',
                  'balanced_accuracy': score, 'metric': 'development prompt-grouped balanced accuracy',
                  'rows': len(train), 'classes': classes.tolist(), 'train_sha256': TRAIN_SHA256,
                  'source': {'public_source_version': 359126546, 'source_archive_sha256': VENDOR_SHA256,
                             'script_sha256': digest(Path(__file__)), 'section': '1: style-aware n-grams + LinearSVC'},
                  'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
                                  'scipy': scipy.__version__, 'sklearn': sklearn.__version__},
                  'classifier_parameters': SVC_PARAMS, 'folds': fold_records, 'self_check': check,
                  'warnings': warning_records, 'convergence_warning_count': convergence_count,
                  'valid_for_proxy_fusion': convergence_count == 0,
                  'probability_note': 'Normalized softmax of uncalibrated LinearSVC scores; not calibrated probabilities.',
                  'log_probability_note': 'S02 add(): log(clip(probability, 1e-9, 1)); no re-normalization after clipping.',
                  'fold_note': 'Archived original 1-based GroupKFold5 folds; all held-out prompts excluded from fit/vocabulary/IDF.',
                  'transductive_operations': False, 'full_training_fit_performed': False,
                  'official_submission_performed': False, 'finished_utc': datetime.now(timezone.utc).isoformat(),
                  'seconds': time.perf_counter() - started,
                  'limitations': ['Available component only; not a full S02 held-out result.',
                                  'Development prompts have been reused; this is not an untouched final evaluation.']}
        write_json(output_dir / 'metrics.json', result)
        artifacts = {path.name: {'sha256': digest(path), 'bytes': path.stat().st_size}
                     for path in sorted(output_dir.iterdir()) if path.name != 'status.json'}
        write_json(output_dir / 'artifact_manifest.json', artifacts)
        write_json(output_dir / 'status.json', {'status': result['status'], 'valid_for_proxy_fusion': convergence_count == 0,
                                              'finished_utc': result['finished_utc'], 'seconds': result['seconds']})
        print('STYLEGRAM_COMPLETE ' + json.dumps({'balanced_accuracy': score, 'seconds': result['seconds'],
                                                 'convergence_warning_count': convergence_count}), flush=True)
        return result
    except Exception as error:
        write_json(output_dir / 'status.json', {'status': 'FAILED_PRESERVE_OUTPUTS', 'exception_type': type(error).__name__,
                                              'message': str(error), 'seconds': time.perf_counter() - started})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--reference-dir', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--self-check', action='store_true', help='Run transform/wiring fixtures only; no fitting')
    args = parser.parse_args()
    if args.self_check:
        print(json.dumps(self_check(), indent=2))
        return
    if any(value is None for value in [args.data_dir, args.reference_dir, args.output_dir]):
        parser.error('--data-dir, --reference-dir, and --output-dir are required for OOF fitting')
    run(args.data_dir, args.reference_dir, args.output_dir)


if __name__ == '__main__':
    main()
