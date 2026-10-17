"""Fixed Stage 06F pair diagnosis: cached probability arithmetic only.

No estimator, model, regrouping, new prediction, optimization or text output.
Production requires immutable E output and train bytes plus a source/plan freeze.
"""
import hashlib
import json
import os
import platform
import re
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PAIR = ('gemini-3.7-flash', 'gemini-3.8-flash')
SCENARIOS = (1, 3, 6)
MEMBERS = ('stylegram', 'lora', 'lora2', 'tokattn', 'tokattn_deb', 'deb_plain', 'deb_center')
WEIGHTS = dict(zip(MEMBERS, (1.516, .177, .113, .153, .185, .05, .25)))
EXPECTED_HASHES = {
    'train.csv': '40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707',
    'ensemble_arrays.npz': '9348eeed149ad48bdd828d8055e945f57e384d8917c654125c2ebbe8d4b99519',
    'ensemble_result.json': '9717bcb12f0c730b4af631067621969dcdbf239e6bed2f84304b470a38458018',
    'output_manifest.json': 'a5fcffaef27f4942af8a679370c30c247b8d872b4b6d711eccc0101ea7006dbd',
    'raw_control_scenario1_confusion.csv': 'a48b61f2a7ffb50764d44bb060adb925e13122abd63fed4730c9ba097685c595',
    'raw_control_scenario3_confusion.csv': 'fae2720b0f34b0bb701bfdffaace99051f0add84d52c91e6973aa6f124299463',
    'raw_control_scenario6_confusion.csv': '938f3bf5a1ea51fdda8b8a75512810c5c74002063e741feef0d8e5dd614c7513',
}
CANONICAL_SHA = '2c6aa02a0bafd9389f436a25de70868b581d829bd8e39835b8e2f6a8354a3b2b'
STATIC_SHA = '513a889477b8940d5f4a1df304b793361ff3860857e4e846e29e28429d9b3d08'
MARGIN_BINS = np.asarray([0, .25, .5, 1, 2, 4, np.inf])
CONFIDENCE_BINS = np.asarray([0, .5, .6, .7, .8, .9, 1.])


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def fixed_confusion(y, pred, nclasses):
    y, pred = np.asarray(y), np.asarray(pred)
    assert y.shape == pred.shape and y.ndim == 1
    assert np.all((y >= 0) & (y < nclasses)) and np.all((pred >= 0) & (pred < nclasses))
    return np.bincount(y.astype(int) * nclasses + pred.astype(int), minlength=nclasses * nclasses).reshape(nclasses, nclasses)


def rate_interval(values, groups, support, draws=3000, seed=20261018):
    """Jointly prompt-resample fixed context counts; no fitted uncertainty."""
    values, support = np.asarray(values), np.asarray(support)
    assert values.ndim == 2 and support.shape == values.shape[1:]
    unique, inverse = np.unique(groups, return_inverse=True)
    count = np.asarray([np.bincount(inverse, weights=row, minlength=len(unique)) for row in values])
    denom = np.bincount(inverse, weights=support, minlength=len(unique))
    samples = np.random.default_rng(seed).integers(0, len(unique), size=(draws, len(unique)))
    total = denom[samples].sum(1)
    assert np.all(total > 0)
    rates = count[:, samples].sum(2) / total[None]
    intervals = np.quantile(rates, [.025, .975], axis=1).T
    mean = np.quantile(rates.mean(0), [.025, .975])
    return dict(by_context=intervals.tolist(), mean_across_contexts=mean.tolist(),
                draws=draws, seed=seed, resampling_unit='prompt',
                interpretation='Conditional descriptive interval for fixed reused-development predictions; omits fitting and new-data uncertainty.')


def context_overlap(mask, ids):
    """Count repeated observations separately from the exact unique-row union."""
    mask, ids = np.asarray(mask, dtype=bool), np.asarray(ids)
    assert mask.ndim == 2 and mask.shape[1] == len(ids) and len(np.unique(ids)) == len(ids)
    count = mask.sum(0)
    patterns = {''.join(str(int(v)) for v in pattern): int(np.all(mask.T == pattern, axis=1).sum())
                for pattern in np.asarray(np.meshgrid(*([[False, True]] * mask.shape[0]))).T.reshape(-1, mask.shape[0])}
    return dict(repeated_context_observations=int(mask.sum()), unique_passages_union=int((count > 0).sum()),
                unique_passages_all_contexts=int((count == mask.shape[0]).sum()),
                unique_passages_by_number_of_contexts={str(i): int((count == i).sum()) for i in range(mask.shape[0] + 1)},
                context_pattern_counts=patterns,
                pairwise_intersections={f'{SCENARIOS[a]}_{SCENARIOS[b]}': int((mask[a] & mask[b]).sum())
                                        for a in range(mask.shape[0]) for b in range(a + 1, mask.shape[0])})


def row_metrics(evidence, probability, y, pair):
    """Descriptive scores only; no corrective gate or new argmax."""
    a, b = pair
    margin = evidence[..., a] - evidence[..., b]
    # Stable logistic of the ordered pair log-evidence margin.
    pair_a = np.exp(-np.logaddexp(0, -margin))
    true_pair = (y == a) | (y == b)
    orientation = np.where(y == a, 1., np.where(y == b, -1., 0.))
    ordered = np.argsort(-evidence, axis=2, kind='stable')
    top2pair = ((ordered[..., :2] == a).any(2) & (ordered[..., :2] == b).any(2))
    return dict(pair_margin=margin, true_pair=true_pair, orientation=orientation,
                truth_margin=margin * orientation[None], pair_a_mass=pair_a,
                pair_confidence=np.maximum(pair_a, 1 - pair_a),
                global_confidence=probability.max(2), pair_global_mass=probability[..., a] + probability[..., b],
                top2_is_pair=top2pair, ordered_top2=ordered[..., :2])


def pair_counts(y, pred, pair, top2=None):
    a, b = pair
    true_pair = (y == a) | (y == b)
    pred_pair = (pred == a) | (pred == b)
    cross = ((y == a) & (pred == b)) | ((y == b) & (pred == a))
    correct = true_pair & (pred == y)
    other = true_pair & ~pred_pair
    false_entry = ~true_pair & pred_pair
    result = dict(pair_support=int(true_pair.sum()), pair_correct=int(correct.sum()), pair_cross=int(cross.sum()),
                  pair_other_error=int(other.sum()), a_to_b=int(((y == a) & (pred == b)).sum()),
                  b_to_a=int(((y == b) & (pred == a)).sum()), false_entries=int(false_entry.sum()),
                  predicted_pair=int(pred_pair.sum()), global_errors=int((pred != y).sum()))
    assert result['pair_support'] == result['pair_correct'] + result['pair_cross'] + result['pair_other_error']
    if top2 is not None:
        result.update(top2_scope_rows=int(top2.sum()), top2_scope_true_pair=int((top2 & true_pair).sum()),
                      top2_scope_cross=int((top2 & cross).sum()), top2_scope_pair_correct=int((top2 & correct).sum()),
                      top2_scope_other_class=int((top2 & ~true_pair).sum()), top2_scope_false_entries=int((top2 & false_entry).sum()))
    return result


def text_features(texts):
    """Deterministic numeric passage shape; never return text/tokens."""
    names = ('characters', 'words', 'lines', 'mean_word_characters', 'uppercase_fraction', 'digit_fraction',
             'punctuation_fraction', 'blank_line_fraction', 'heading_line_fraction', 'bullet_line_fraction',
             'code_fences', 'questions')
    output = np.zeros((len(texts), len(names)), dtype=np.float64)
    for i, value in enumerate(texts):
        value = str(value)
        lines, words = value.splitlines(), value.split()
        n, nl = max(len(value), 1), max(len(lines), 1)
        output[i] = (len(value), len(words), len(lines), sum(map(len, words)) / max(len(words), 1),
                     sum(c.isupper() for c in value) / n, sum(c.isdigit() for c in value) / n,
                     sum(unicodedata.category(c).startswith('P') for c in value) / n,
                     sum(not line.strip() for line in lines) / nl,
                     sum(bool(re.match(r'^\s*#{1,6}\s', line)) for line in lines) / nl,
                     sum(bool(re.match(r'^\s*(?:[-*+]\s|\d+[.)]\s)', line)) for line in lines) / nl,
                     value.count('```'), value.count('?'))
    return np.asarray(names, dtype=str), output


def _locate(root, filename):
    paths = list(Path(root).rglob(filename))
    assert len(paths) == 1, ('Expected unique input file', filename, len(paths))
    return paths[0]


def _load_inputs(root):
    paths = {name: _locate(root, name) for name in EXPECTED_HASHES}
    for name, path in paths.items():
        assert digest(path) == EXPECTED_HASHES[name], ('Immutable input hash mismatch', name)
    result = json.loads(paths['ensemble_result.json'].read_text())
    manifest = json.loads(paths['output_manifest.json'].read_text())
    for name in EXPECTED_HASHES:
        if name not in ('train.csv', 'output_manifest.json'):
            assert manifest[name]['sha256'] == EXPECTED_HASHES[name]
            assert manifest[name]['bytes'] == paths[name].stat().st_size
    arrays = {}
    with np.load(paths['ensemble_arrays.npz'], allow_pickle=False) as saved:
        for key in saved.files:
            value = saved[key].copy()
            assert not value.dtype.hasobject and value.dtype.kind in 'biufU', ('Unsafe dtype', key)
            if value.dtype.kind == 'f':
                assert np.isfinite(value).all(), ('Nonfinite array', key)
            arrays[key] = value
    train = pd.read_csv(paths['train.csv'], dtype={'id': str, 'prompt_id': str, 'model': str}, keep_default_na=False)
    return arrays, train, result, paths


def _source_gate(arrays, train, result, paths):
    ids, classes, y, groups, fold = (arrays[k] for k in ('ids', 'classes', 'y', 'prompt_ids', 'fold'))
    n, c = len(ids), len(classes)
    assert n == 2699 and c == 18 and len(np.unique(ids)) == n and len(np.unique(groups)) == 150
    assert ids.dtype.kind == classes.dtype.kind == groups.dtype.kind == 'U'
    assert tuple(arrays['scenarios']) == SCENARIOS and tuple(result['scenarios']) == SCENARIOS
    assert arrays['train_sha256'].item() == EXPECTED_HASHES['train.csv'] == result['train_sha256']
    assert result['fixed_weights'] == WEIGHTS
    assert result['canonical_cpu_cache']['status'] == 'PASS'
    assert result['canonical_cpu_cache']['canonical']['cache_sha256'] == CANONICAL_SHA
    assert result['canonical_cpu_cache']['static']['cache_sha256'] == STATIC_SHA
    assert result['status'] == 'COMPLETE_CANONICAL_CPU_FIXED_AVAILABLE_COMPONENT_PROXY'
    assert result['classes'] == classes.tolist() and np.array_equal(classes, np.sort(train.model.unique()))
    for key, column in (('ids', 'id'), ('prompt_ids', 'prompt_id')):
        np.testing.assert_array_equal(arrays[key], train[column].to_numpy(dtype=str))
    np.testing.assert_array_equal(classes[y], train.model.to_numpy(dtype=str))
    expected = {'p000': 4, 'p001': 3, 'p002': 5}
    expected.update({f'p{i:03d}': 5 - i % 5 for i in range(3, 150)})
    assert all(expected[g] == int(f) for g, f in zip(groups, fold))
    for member in MEMBERS:
        member_fold = arrays['fold_' + member]
        assert member_fold.shape == (n,) and len(np.unique(member_fold)) == 5
        for f in np.unique(member_fold):
            assert len(np.unique(groups[member_fold == f])) == 30
            assert not set(groups[member_fold == f]) & set(groups[member_fold != f])
    eligible = arrays['eligible']
    assert eligible.shape == (3, n) and eligible.dtype.kind == 'b'
    base = np.zeros((n, c), dtype=np.float64)
    terms = []
    for member in MEMBERS[:5]:
        probability = arrays['probability_' + member]
        assert probability.shape == (n, c) and np.all(probability >= 0) and np.all(probability <= 1 + 2e-6)
        np.testing.assert_allclose(probability.sum(1), 1, rtol=0, atol=2e-6)
        logs = np.log(np.clip(probability, 1e-9, 1))
        base += WEIGHTS[member] * logs
        terms.append(np.broadcast_to(WEIGHTS[member] * logs, (3, n, c)))
    plain = arrays['deb_log_probability_plain']
    centered = arrays['deb_log_probability_control']
    assert plain.shape == centered.shape == (3, n, c)
    assert np.all(centered[~eligible] == 0)
    evidence = base[None] + .05 * plain
    evidence = evidence.copy()
    evidence[eligible] += .25 * centered[eligible]
    np.testing.assert_array_equal(evidence, arrays['log_evidence_control'])
    np.testing.assert_array_equal(evidence.argmax(2), arrays['prediction_raw_control'])
    terms.extend((.05 * plain, .25 * centered))
    probability = np.exp(evidence - evidence.max(2, keepdims=True))
    probability /= probability.sum(2, keepdims=True)
    np.testing.assert_allclose(probability, arrays['raw_probability_control'], rtol=0, atol=2e-15)
    scores, confusions = [], []
    for s, scenario in enumerate(SCENARIOS):
        cm = fixed_confusion(y, arrays['prediction_raw_control'][s], c)
        prior = pd.read_csv(paths[f'raw_control_scenario{scenario}_confusion.csv'], index_col=0)
        assert list(prior.index) == list(prior.columns) == classes.tolist()
        np.testing.assert_array_equal(cm, prior.to_numpy())
        score = float((cm.diagonal() / cm.sum(1)).mean())
        np.testing.assert_allclose(score, result['scores']['raw_control'][s], rtol=0, atol=1e-15)
        scores.append(score); confusions.append(cm)
    np.testing.assert_allclose(np.mean(scores), result['mean_scores']['raw_control'], rtol=0, atol=1e-15)
    pair = tuple(int(np.flatnonzero(classes == label)[0]) for label in PAIR)
    counts = [pair_counts(y, row, pair) for row in arrays['prediction_raw_control']]
    for key, expected_count in dict(a_to_b=[50,49,52], b_to_a=[44,38,42], pair_correct=[135,146,139],
                                    pair_other_error=[71,67,67], false_entries=[65,54,49], global_errors=[1278,1206,1174]).items():
        assert [row[key] for row in counts] == expected_count, ('Source count reconciliation', key)
    contribution = np.stack([term[..., pair[0]] - term[..., pair[1]] for term in terms], axis=2)
    np.testing.assert_allclose(contribution.sum(2), evidence[..., pair[0]] - evidence[..., pair[1]], rtol=0, atol=1e-12)
    return dict(status='PASS', immutable_input_hashes=EXPECTED_HASHES, source_script_version_id=359514801,
                source_canonical_cache_sha256=CANONICAL_SHA, source_static_cache_sha256=STATIC_SHA,
                ordered_ids_labels_prompts_and_folds_exact=True, exact_raw_evidence_and_predictions=True,
                source_confusions_exact=True, raw_control_scores=scores,
                source_raw_mean_balanced_accuracy=float(np.mean(scores)),
                weighted_pair_contribution_sum_atol=1e-12,
                raw_probability_replay_atol=2e-15), contribution


def _summaries(values):
    values = np.asarray(values)
    if not len(values):
        return dict(mean=None, median=None, q25=None, q75=None)
    return dict(mean=float(values.mean()), median=float(np.median(values)),
                q25=float(np.quantile(values, .25)), q75=float(np.quantile(values, .75)))


def _binned(values, mask, outcome, bins, metric, scenario):
    rows = []
    for i in range(len(bins) - 1):
        active = mask & (values >= bins[i]) & ((values <= bins[i + 1]) if i == len(bins) - 2 else (values < bins[i + 1]))
        rows.append(dict(scenario=scenario, metric=metric, lower=float(bins[i]),
                         upper=None if np.isinf(bins[i + 1]) else float(bins[i + 1]),
                         upper_inclusive=i == len(bins) - 2, rows=int(active.sum()),
                         correct=int((active & (outcome == 1)).sum()), cross=int((active & (outcome == 2)).sum()),
                         other_error=int((active & (outcome == 3)).sum())))
    assert sum(row['rows'] for row in rows) == int(mask.sum())
    return rows


def analyze(arrays, train, contribution):
    ids, classes, y, groups, fold = (arrays[k] for k in ('ids', 'classes', 'y', 'prompt_ids', 'fold'))
    pair = tuple(int(np.flatnonzero(classes == label)[0]) for label in PAIR)
    evidence, probability, pred = (arrays[k] for k in ('log_evidence_control', 'raw_probability_control', 'prediction_raw_control'))
    metrics = row_metrics(evidence, probability, y, pair)
    true_pair = metrics['true_pair']
    correct = true_pair[None] & (pred == y[None])
    cross = ((y[None] == pair[0]) & (pred == pair[1])) | ((y[None] == pair[1]) & (pred == pair[0]))
    pred_pair = (pred == pair[0]) | (pred == pair[1])
    false_entry = ~true_pair[None] & pred_pair
    outcome = np.where(correct, 1, np.where(cross, 2, np.where(true_pair[None], 3, np.where(false_entry, 4, 0)))).astype(np.int8)
    summaries = [dict(scenario=scenario, **pair_counts(y, pred[s], pair, metrics['top2_is_pair'][s])) for s, scenario in enumerate(SCENARIOS)]
    bins, member_rows, false_rows, fold_rows, prompt_rows, feature_rows, pair_rows = [], [], [], [], [], [], []
    feature_names, features = text_features(train.text.to_numpy())
    orientation = metrics['orientation']
    truth_contribution = contribution * orientation[None, :, None]
    votes = np.sign(contribution).astype(np.int8)
    centered_neutral = ~arrays['eligible']
    assert np.all(votes[..., -1][centered_neutral] == 0)
    positive, negative = (votes > 0).sum(2), (votes < 0).sum(2)
    disagreement = (positive > 0) & (negative > 0)
    for s, scenario in enumerate(SCENARIOS):
        for metric, values, edges in (('absolute_pair_log_margin', np.abs(metrics['pair_margin'][s]), MARGIN_BINS),
                                      ('global_softmax_max', metrics['global_confidence'][s], CONFIDENCE_BINS),
                                      ('conditional_pair_max', metrics['pair_confidence'][s], CONFIDENCE_BINS)):
            bins.extend(_binned(values, true_pair, outcome[s], edges, metric, scenario))
        for m, member in enumerate(MEMBERS):
            for code, label in ((1, 'pair_correct'), (2, 'pair_cross'), (3, 'pair_other_error')):
                selected = outcome[s] == code
                true_vote = votes[s, :, m] * orientation
                member_rows.append(dict(scenario=scenario, member=member, outcome=label, rows=int(selected.sum()),
                                        supports_true=int((selected & (true_vote > 0)).sum()),
                                        supports_other_pair=int((selected & (true_vote < 0)).sum()),
                                        neutral=int((selected & (true_vote == 0)).sum()),
                                        **{'truth_weighted_margin_' + k: v for k, v in _summaries(truth_contribution[s, selected, m]).items()}))
        for cls, label in enumerate(classes):
            if cls in pair:
                continue
            selected = y == cls
            false_rows.append(dict(scenario=scenario, true_model=str(label), support=int(selected.sum()),
                                   predicted_a=int((selected & (pred[s] == pair[0])).sum()),
                                   predicted_b=int((selected & (pred[s] == pair[1])).sum()),
                                   top2_pair_scope=int((selected & metrics['top2_is_pair'][s]).sum())))
        for f in range(1, 6):
            selected = fold == f
            fold_rows.append(dict(scenario=scenario, fold=f, **pair_counts(y[selected], pred[s, selected], pair, metrics['top2_is_pair'][s, selected])))
        for group in np.unique(groups):
            selected = groups == group
            prompt_rows.append(dict(scenario=scenario, prompt_id=str(group), fold=int(fold[selected][0]),
                                    **pair_counts(y[selected], pred[s, selected], pair, metrics['top2_is_pair'][s, selected])))
        for code, label in ((1, 'pair_correct'), (2, 'pair_cross'), (3, 'pair_other_error')):
            selected = outcome[s] == code
            for f, name in enumerate(feature_names):
                feature_rows.append(dict(scenario=scenario, outcome=label, rows=int(selected.sum()), feature=str(name), **_summaries(features[selected, f])))
        for i in np.flatnonzero(true_pair | false_entry[s]):
            pair_rows.append(dict(scenario=scenario, id=str(ids[i]), prompt_id=str(groups[i]), fold=int(fold[i]),
                                  true_model=str(classes[y[i]]), predicted_model=str(classes[pred[s, i]]), is_true_pair=bool(true_pair[i]), outcome=int(outcome[s, i]),
                                  pair_log_margin=float(metrics['pair_margin'][s, i]), truth_pair_log_margin=float(metrics['truth_margin'][s, i]),
                                  global_softmax_max=float(metrics['global_confidence'][s, i]), conditional_pair_a_mass=float(metrics['pair_a_mass'][s, i]),
                                  pair_global_softmax_mass=float(metrics['pair_global_mass'][s, i]), top2_is_pair=bool(metrics['top2_is_pair'][s, i]),
                                  member_disagreement=bool(disagreement[s, i]), members_supporting_true=int(((votes[s, i] * orientation[i]) > 0).sum()),
                                  active_centered=bool(arrays['eligible'][s, i])))
    overlap = {}
    for name, mask, population in (('pair_cross', cross, true_pair), ('pair_correct', correct, true_pair),
                                   ('pair_other_error', outcome == 3, true_pair), ('nonpair_false_entry', false_entry, ~true_pair)):
        overlap[name] = context_overlap(mask[:, population], ids[population])
        overlap[name]['population_passages'] = int(population.sum())
        overlap[name]['population'] = 'true_pair' if name != 'nonpair_false_entry' else 'true_nonpair'
    repeated = int(cross.sum()); union = overlap['pair_cross']['unique_passages_union']
    assert repeated >= union
    uncertainty = dict(pair_correct_rate=rate_interval(correct, groups, true_pair), pair_cross_rate=rate_interval(cross, groups, true_pair))
    disagreements = [dict(scenario=scenario, pair_rows_with_member_disagreement=int((true_pair & disagreement[s]).sum()),
                          cross_rows_with_member_disagreement=int((cross[s] & disagreement[s]).sum()),
                          cross_rows_with_any_member_supporting_true=int((cross[s] & ((votes[s] * orientation[:, None]) > 0).any(1)).sum()),
                          cross_rows_with_no_member_supporting_true=int((cross[s] & ~((votes[s] * orientation[:, None]) > 0).any(1)).sum()),
                          inactive_centered_rows=int((~arrays['eligible'][s]).sum())) for s, scenario in enumerate(SCENARIOS)]
    saved = dict(ids=ids, classes=classes, y=y, prompt_ids=groups, fold=fold, scenarios=np.asarray(SCENARIOS),
                 pair_class_indices=np.asarray(pair), source_prediction_raw_control=pred,
                 pair_true_mask=true_pair, pair_outcome=outcome, pair_margin=metrics['pair_margin'],
                 truth_pair_margin=metrics['truth_margin'], pair_a_mass=metrics['pair_a_mass'],
                 pair_global_mass=metrics['pair_global_mass'], global_softmax_max=metrics['global_confidence'],
                 top2_is_pair=metrics['top2_is_pair'], ordered_top2=metrics['ordered_top2'], members=np.asarray(MEMBERS),
                 weighted_pair_contribution=contribution, member_pair_preference=votes, member_disagreement=disagreement,
                 centered_active=arrays['eligible'], text_feature_names=feature_names, text_features=features)
    return dict(summary=summaries, overlap=overlap, uncertainty=uncertainty, disagreements=disagreements), saved, dict(
        confidence_bins=bins, component_contributions=member_rows, other_class_false_entries=false_rows, folds=fold_rows,
        prompts=prompt_rows, text_features=feature_rows, pair_rows=pair_rows)


def run_diagnostics(input_root, output_root, freeze_path=None):
    output_root = Path(output_root)
    assert not output_root.exists() or not any(output_root.iterdir()), 'Fresh diagnostic output directory required'
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    def status(value, **extra):
        write_json(output_root / 'status.json', dict(status=value, observed_utc=datetime.now(timezone.utc).isoformat(), **extra))
    status('VERIFYING_IMMUTABLE_SOURCE')
    try:
        freeze_receipt = {}
        if freeze_path is not None:
            freeze = json.loads(Path(freeze_path).read_text())
            assert digest(__file__) == freeze['diagnostic_source_sha256']
            plan_path = Path(freeze.get('plan_path', Path(__file__).with_name('STAGE_06F_PAIR_DIAGNOSTICS.md')))
            assert digest(plan_path) == freeze['plan_sha256']
            assert freeze['input_version_ids'] == [170484, 359514801]
            assert freeze['accelerator'] == 'none' and freeze['internet_enabled'] is False
            freeze_receipt = dict(freeze_sha256=digest(freeze_path), diagnostic_source_sha256=digest(__file__), plan_sha256=digest(plan_path), input_version_ids=freeze['input_version_ids'])
        arrays, train, source_result, paths = _load_inputs(input_root)
        gate, contribution = _source_gate(arrays, train, source_result, paths)
        gate['execution_freeze'] = freeze_receipt
        write_json(output_root / 'source_gate.json', gate)
        print('STAGE06F_SOURCE_GATE', json.dumps(gate, allow_nan=False), flush=True)
        status('DESCRIPTIVE_FIXED_PAIR_DIAGNOSIS')
        findings, saved, tables = analyze(arrays, train, contribution)
        assert all(not value.dtype.hasobject and value.dtype.kind in 'biufU' for value in saved.values())
        np.savez_compressed(output_root / 'pair_diagnostics.npz', **saved)
        with np.load(output_root / 'pair_diagnostics.npz', allow_pickle=False) as readback:
            assert set(readback.files) == set(saved)
            for key, value in saved.items():
                np.testing.assert_array_equal(readback[key], value)
        for name, rows in tables.items():
            pd.DataFrame(rows).to_csv(output_root / (name + '.csv'), index=False)
        result = dict(status='COMPLETE_FIXED_PAIR_DIAGNOSTICS', pair=list(PAIR), source_script_version_id=359514801,
                      primary_source='raw_control', contexts=list(SCENARIOS), rows=len(arrays['ids']),
                      distinct_true_pair_passages=int(saved['pair_true_mask'].sum()), context_rows_are_repeated_passages=True,
                      source_raw_scores=gate['raw_control_scores'], source_raw_mean=gate['source_raw_mean_balanced_accuracy'],
                      **findings, source_gate=gate, runtime=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__),
                      fits=0, model_inference_calls=0, regrouping_calls=0, optimization_or_threshold_search=False,
                      changed_predictions=0, corrective_candidate_evaluated=False, official_submission=False, paid_services=False,
                      seconds=time.monotonic()-started,
                      limitations=['Available-component development proxy, not full S02 or an official score.',
                                   'Pair selected after inspecting reused-development confusion counts; findings generate hypotheses only.',
                                   'Contexts reuse the same passages and trained heads; repeated counts are not independent examples.',
                                   'Softmax and conditional pair masses are descriptive uncalibrated scores.',
                                   'Prompt bootstrap fixes all learned models and predictions; it omits fitting, grouping and future-data uncertainty.',
                                   'Text-shape associations and component disagreement do not establish causal mechanisms or recoverable accuracy.',
                                   'Top-two scope is diagnostic only and never changes a prediction.'])
        write_json(output_root / 'pair_result.json', result)
        manifest = {p.name: dict(bytes=p.stat().st_size, sha256=digest(p)) for p in sorted(output_root.iterdir())
                    if p.is_file() and p.name not in ('status.json', 'output_manifest.json')}
        write_json(output_root / 'output_manifest.json', manifest)
        status('COMPLETE_FIXED_PAIR_DIAGNOSTICS', valid=True)
        for name in ('component_contributions', 'confidence_bins', 'other_class_false_entries', 'folds', 'text_features'):
            print('STAGE06F_' + name.upper() + '_CSV', json.dumps(pd.DataFrame(tables[name]).to_csv(index=False)), flush=True)
        print('STAGE06F_RESULT', json.dumps(result, allow_nan=False), flush=True)
        print('STAGE06F_OUTPUT_MANIFEST', json.dumps(manifest, allow_nan=False), flush=True)
        print('STAGE06F_COMPLETE', flush=True)
        return result
    except BaseException as error:
        status('FAILED_PRESERVE_OUTPUTS', exception_type=type(error).__name__, exception_message=str(error))
        print('STAGE06F_FAILED', type(error).__name__, str(error), flush=True)
        raise


if __name__ == '__main__':
    freeze_path = os.environ.get('NTL_STAGE06F_FREEZE')
    assert freeze_path, 'Production execution requires NTL_STAGE06F_FREEZE'
    run_diagnostics(os.environ.get('NTL_STAGE06F_INPUT', '/kaggle/input'),
                    os.environ.get('NTL_STAGE06F_OUTPUT', '/kaggle/working/stage06f_diagnostics'), freeze_path)
