"""One frozen, no-fit Stage 06G attention-pair development comparison.

Consumes immutable Stage 06E probabilities through the unchanged reviewed F
source gates. No estimator, inference, regrouping, tuning or batch balancing.
"""
import hashlib
import importlib.util
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PAIR = ('gemini-3.7-flash', 'gemini-3.8-flash')
SCENARIOS = (1, 3, 6)
SOURCE_GATE_SHA256 = 'a8bb68181087640fb563539870840117531a4d57e529cbeaf69e73d4c0a05be7'
PLAN_SHA256 = 'ece986eb6244bd322564cd7fd7a437239e5f74635a2cea2b9a39968bd891b927'
COMPLETE = 'COMPLETE_FIXED_ATTENTION_PAIR_COMPLEMENT'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_digest(value):
    """Bind dtype, shape and bytes; never serialize passage text."""
    value = np.asarray(value)
    header = json.dumps([value.dtype.str, list(value.shape)], separators=(',', ':')).encode()
    return hashlib.sha256(header + np.ascontiguousarray(value).tobytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def fixed_confusion(y, prediction, nclasses):
    y, prediction = np.asarray(y), np.asarray(prediction)
    assert y.ndim == 1 and prediction.shape == y.shape
    assert y.dtype.kind in 'iu' and prediction.dtype.kind in 'iu'
    assert np.all((y >= 0) & (y < nclasses))
    assert np.all((prediction >= 0) & (prediction < nclasses))
    return np.bincount(y.astype(int) * nclasses + prediction.astype(int),
                       minlength=nclasses * nclasses).reshape(nclasses, nclasses)


def derive_candidate(classes, evidence, source_prediction, qwen_probability, deb_probability):
    """Apply the single label-blind rule, preserving native log arithmetic.

    This pure helper accepts synthetic fixtures. Production first verifies the
    immutable F gates and later verifies all actual source scope counts.
    """
    classes, evidence, source_prediction = map(np.asarray, (classes, evidence, source_prediction))
    qwen_probability, deb_probability = map(np.asarray, (qwen_probability, deb_probability))
    assert classes.ndim == 1 and len(np.unique(classes)) == len(classes)
    pair = tuple(int(np.flatnonzero(classes == label).item()) for label in PAIR)
    assert evidence.ndim == 3 and evidence.shape[2] == len(classes)
    assert source_prediction.shape == evidence.shape[:2] and source_prediction.dtype.kind in 'iu'
    assert qwen_probability.shape == deb_probability.shape == evidence.shape[1:]
    assert np.isfinite(evidence).all()
    for probability in (qwen_probability, deb_probability):
        assert probability.dtype.kind == 'f' and np.isfinite(probability).all()
        assert np.all(probability >= 0) and np.all(probability <= 1 + 2e-6)
        np.testing.assert_allclose(probability.sum(1), 1, rtol=0, atol=2e-6)
    np.testing.assert_array_equal(source_prediction, evidence.argmax(2))
    # Calculate log(clip(P)) for each entire saved matrix, retaining its dtype.
    qwen_log = np.log(np.clip(qwen_probability, 1e-9, 1))
    deb_log = np.log(np.clip(deb_probability, 1e-9, 1))
    a, b = pair
    qwen_margin = qwen_log[:, a] - qwen_log[:, b]
    deb_margin = deb_log[:, a] - deb_log[:, b]
    d = qwen_margin + deb_margin
    assert np.isfinite(d).all()
    ordered_top2 = np.argsort(-evidence, axis=2, kind='stable')[..., :2]
    scope = ((ordered_top2 == a).any(2) & (ordered_top2 == b).any(2))
    candidate = source_prediction.copy()
    candidate[scope & (d[None] > 0)] = a
    candidate[scope & (d[None] < 0)] = b
    # d == 0 has no assignment: retain the exact original source prediction.
    return dict(candidate_prediction=candidate, scope=scope, ordered_top2=ordered_top2,
                qwen_pair_margin=qwen_margin, deb_pair_margin=deb_margin,
                combined_pair_margin=d, pair_class_indices=np.asarray(pair, dtype=np.int64))


def comparison_invariants(y, source_prediction, candidate_prediction, scope, pair, nclasses):
    """Reconcile every change, including already-wrong nonpair destinations."""
    y = np.asarray(y)
    source, candidate, scope = map(np.asarray, (source_prediction, candidate_prediction, scope))
    assert source.shape == candidate.shape == scope.shape and source.shape[1:] == y.shape
    assert scope.dtype.kind == 'b'
    a, b = pair
    true_pair = (y == a) | (y == b)
    source_entry = (source == a) | (source == b)
    candidate_entry = (candidate == a) | (candidate == b)
    changed = source != candidate
    old_correct, new_correct = source == y[None], candidate == y[None]
    corrected = ~old_correct & new_correct
    broken = old_correct & ~new_correct
    changed_both_wrong = changed & ~old_correct & ~new_correct
    assert not changed[~scope].any(), 'Outside-scope prediction changed'
    assert source_entry[scope].all(), 'Scoped original prediction was not in pair'
    assert source_entry[changed].all() and candidate_entry[changed].all()
    np.testing.assert_array_equal(source_entry, candidate_entry)
    np.testing.assert_array_equal(old_correct[:, ~true_pair], new_correct[:, ~true_pair])
    assert not changed[:, true_pair][~source_entry[:, true_pair]].any(), 'Other-top-one true pair changed'
    assert not corrected[:, ~true_pair].any() and not broken[:, ~true_pair].any()
    assert not changed_both_wrong[:, true_pair].any()
    np.testing.assert_array_equal(changed, corrected | broken | changed_both_wrong)
    control_cm, candidate_cm = [], []
    for s in range(len(source)):
        old_cm = fixed_confusion(y, source[s], nclasses)
        new_cm = fixed_confusion(y, candidate[s], nclasses)
        supports = old_cm.sum(1)
        assert (supports > 0).all()
        np.testing.assert_array_equal(supports, new_cm.sum(1))
        others = np.ones(nclasses, dtype=bool)
        others[[a, b]] = False
        np.testing.assert_array_equal(old_cm.diagonal()[others], new_cm.diagonal()[others])
        np.testing.assert_array_equal((old_cm.diagonal() / supports)[others],
                                      (new_cm.diagonal() / supports)[others])
        delta = int(corrected[s].sum()) - int(broken[s].sum())
        assert delta == int(new_correct[s].sum()) - int(old_correct[s].sum())
        assert delta == int((new_cm.diagonal() - old_cm.diagonal())[[a, b]].sum())
        assert int(changed[s].sum()) == int(corrected[s].sum() + broken[s].sum() + changed_both_wrong[s].sum())
        control_cm.append(old_cm)
        candidate_cm.append(new_cm)
    return dict(true_pair=true_pair, changed=changed, corrected=corrected, broken=broken,
                changed_both_wrong=changed_both_wrong, source_pair_entry=source_entry,
                candidate_pair_entry=candidate_entry, source_confusion=np.stack(control_cm),
                candidate_confusion=np.stack(candidate_cm))


def paired_prompt_interval(y, groups, source_prediction, candidate_prediction, context_gains,
                           draws=3000, seed=20261017):
    """Inherited D/E bootstrap: fixed context-mean changes by prompt and class.

    One joint prompt draw is shared by all contexts. No model, routing or
    prediction is refitted. Parameters permit independent synthetic fixtures;
    production always calls the exact frozen defaults.
    """
    y, groups = np.asarray(y), np.asarray(groups)
    source, candidate = np.asarray(source_prediction), np.asarray(candidate_prediction)
    assert source.shape == candidate.shape and source.shape[1:] == y.shape == groups.shape
    difference = ((candidate == y[None]).astype(float) - (source == y[None]).astype(float)).mean(0)
    unique = np.unique(groups)
    nclasses = int(y.max()) + 1
    numerator = np.stack([np.bincount(y[groups == prompt], weights=difference[groups == prompt],
                                      minlength=nclasses) for prompt in unique])
    denominator = np.stack([np.bincount(y[groups == prompt], minlength=nclasses) for prompt in unique])
    samples = np.random.default_rng(seed).integers(len(unique), size=(draws, len(unique)))
    sampled_denominator = denominator[samples].sum(1)
    assert (sampled_denominator > 0).all()
    bootstrap = (numerator[samples].sum(1) / sampled_denominator.clip(1)).mean(1)
    mean_gain = float(np.asarray(context_gains).mean())
    np.testing.assert_allclose((numerator.sum(0) / denominator.sum(0)).mean(), mean_gain,
                               rtol=0, atol=1e-15)
    return dict(mean_gain=mean_gain, conditional_prompt_95_interval=np.quantile(bootstrap, [.025, .975]).tolist(),
                bootstrap_draws=int(draws), bootstrap_seed=int(seed), prompt_count=int(len(unique)),
                resampling_unit='Original prompt, jointly across contexts',
                interpretation='Conditional on fixed reused-development fits, probabilities and routing. Omits refitting, scope selection and future-data uncertainty.')


def evaluate_candidate(arrays):
    """Score exactly one fixed candidate; no source predictions are replaced."""
    ids, classes, y, groups, fold = (arrays[k] for k in ('ids', 'classes', 'y', 'prompt_ids', 'fold'))
    source, evidence = arrays['prediction_raw_control'], arrays['log_evidence_control']
    qwen, deb = arrays['probability_tokattn'], arrays['probability_tokattn_deb']
    n, c = len(ids), len(classes)
    assert n == 2699 and c == 18 and source.shape == (3, n)
    assert tuple(arrays['scenarios']) == SCENARIOS and len(np.unique(groups)) == 150
    support = np.bincount(y, minlength=c)
    assert sorted(support.tolist()) == [149] + [150] * 17
    protected = {key: array_digest(arrays[key]) for key in
                 ('prediction_raw_control', 'log_evidence_control', 'raw_probability_control',
                  'probability_tokattn', 'probability_tokattn_deb')}
    rule = derive_candidate(classes, evidence, source, qwen, deb)
    pair = tuple(int(x) for x in rule['pair_class_indices'])
    a, b = pair
    np.testing.assert_array_equal(support[[a, b]], [150, 150])
    candidate, scope, d = (rule[k] for k in ('candidate_prediction', 'scope', 'combined_pair_margin'))
    scope_sha256 = array_digest(scope)
    invariants = comparison_invariants(y, source, candidate, scope, pair, c)
    true_pair = invariants['true_pair']
    cross = ((y[None] == a) & (source == b)) | ((y[None] == b) & (source == a))
    old_correct, new_correct = source == y[None], candidate == y[None]
    expected = dict(scope_rows=[193, 196, 190], scoped_cross=[68, 62, 71],
                    scoped_correct_pair=[101, 113, 101], scoped_nonpair=[24, 21, 18])
    observed = dict(scope_rows=scope.sum(1).astype(int).tolist(),
                    scoped_cross=(scope & cross).sum(1).astype(int).tolist(),
                    scoped_correct_pair=(scope & old_correct & true_pair[None]).sum(1).astype(int).tolist(),
                    scoped_nonpair=(scope & ~true_pair[None]).sum(1).astype(int).tolist())
    assert observed == expected, ('Original source scope counts mismatch', observed)
    assert np.array_equal(scope.sum(1), (scope & cross).sum(1) +
                          (scope & old_correct & true_pair[None]).sum(1) + (scope & ~true_pair[None]).sum(1))
    assert not invariants['changed'][scope & (d[None] == 0)].any()
    context_rows, class_rows, fold_rows, prompt_rows, false_rows, scope_rows = [], [], [], [], [], []
    tables = {}
    source_scores, candidate_scores = [], []
    for s, scenario in enumerate(SCENARIOS):
        old_cm, new_cm = invariants['source_confusion'][s], invariants['candidate_confusion'][s]
        old_recall, new_recall = old_cm.diagonal() / support, new_cm.diagonal() / support
        old_score, new_score = float(old_recall.mean()), float(new_recall.mean())
        source_scores.append(old_score)
        candidate_scores.append(new_score)
        corrected, broken, changed_wrong = (invariants[k][s] for k in ('corrected', 'broken', 'changed_both_wrong'))
        np.testing.assert_allclose(new_score-old_score,
                                   (int(corrected.sum())-int(broken.sum())) / 2700,
                                   rtol=0, atol=1e-15)
        context_rows.append(dict(scenario=scenario, rows=n, source_balanced_accuracy=old_score,
                                 candidate_balanced_accuracy=new_score, gain=new_score-old_score,
                                 source_correct=int(old_correct[s].sum()), candidate_correct=int(new_correct[s].sum()),
                                 corrected=int(corrected.sum()), broken=int(broken.sum()),
                                 changed_both_wrong=int(changed_wrong.sum()), changed=int(invariants['changed'][s].sum()),
                                 scope_rows=int(scope[s].sum()), scope_fraction=float(scope[s].mean()),
                                 scoped_true_pair=int((scope[s] & true_pair).sum()),
                                 scoped_nonpair=int((scope[s] & ~true_pair).sum()),
                                 exact_tie_scoped=int((scope[s] & (d == 0)).sum()),
                                 scoped_d_positive=int((scope[s] & (d > 0)).sum()),
                                 scoped_d_negative=int((scope[s] & (d < 0)).sum()),
                                 pair_corrected=int((corrected & true_pair).sum()),
                                 pair_broken=int((broken & true_pair).sum()),
                                 scoped_nonpair_destination_changes=int((invariants['changed'][s] & ~true_pair).sum()),
                                 source_pair_entries=int(invariants['source_pair_entry'][s].sum()),
                                 candidate_pair_entries=int(invariants['candidate_pair_entry'][s].sum()),
                                 true_pair_other_top_one_unchanged=int((true_pair & ~invariants['source_pair_entry'][s]).sum())))
        for cls, label in enumerate(classes):
            selected = y == cls
            class_rows.append(dict(scenario=scenario, model=str(label), is_target_pair=cls in pair,
                                   support=int(support[cls]), source_correct=int(old_cm[cls, cls]),
                                   candidate_correct=int(new_cm[cls, cls]), source_recall=float(old_recall[cls]),
                                   candidate_recall=float(new_recall[cls]), recall_gain=float(new_recall[cls]-old_recall[cls]),
                                   corrected=int((corrected & selected).sum()), broken=int((broken & selected).sum()),
                                   changed_both_wrong=int((changed_wrong & selected).sum())))
            if cls not in pair:
                false_rows.append(dict(scenario=scenario, true_model=str(label), support=int(support[cls]),
                                       source_to_a=int((selected & (source[s] == a)).sum()),
                                       source_to_b=int((selected & (source[s] == b)).sum()),
                                       candidate_to_a=int((selected & (candidate[s] == a)).sum()),
                                       candidate_to_b=int((selected & (candidate[s] == b)).sum()),
                                       scoped_rows=int((selected & scope[s]).sum()),
                                       destination_changes=int((selected & invariants['changed'][s]).sum()),
                                       source_correct=int(old_cm[cls, cls]), candidate_correct=int(new_cm[cls, cls])))
        for f in sorted(np.unique(fold)):
            selected = fold == f
            old_fold = fixed_confusion(y[selected], source[s, selected], c)
            new_fold = fixed_confusion(y[selected], candidate[s, selected], c)
            fold_support = old_fold.sum(1)
            assert (fold_support > 0).all()
            old_fold_score = float((old_fold.diagonal() / fold_support).mean())
            new_fold_score = float((new_fold.diagonal() / fold_support).mean())
            fold_rows.append(dict(scenario=scenario, fold=int(f), rows=int(selected.sum()),
                                  source_balanced_accuracy=old_fold_score, candidate_balanced_accuracy=new_fold_score,
                                  gain=new_fold_score-old_fold_score, corrected=int((corrected & selected).sum()),
                                  broken=int((broken & selected).sum()), changed_both_wrong=int((changed_wrong & selected).sum()),
                                  scope_rows=int((scope[s] & selected).sum())))
        for prompt in np.unique(groups):
            selected = groups == prompt
            assert len(np.unique(fold[selected])) == 1
            prompt_rows.append(dict(scenario=scenario, prompt_id=str(prompt), fold=int(fold[selected][0]),
                                    rows=int(selected.sum()), source_correct=int((old_correct[s] & selected).sum()),
                                    candidate_correct=int((new_correct[s] & selected).sum()),
                                    corrected=int((corrected & selected).sum()), broken=int((broken & selected).sum()),
                                    changed_both_wrong=int((changed_wrong & selected).sum()),
                                    scope_rows=int((scope[s] & selected).sum())))
        for population, mask in (('true_pair', true_pair), ('nonpair', ~true_pair)):
            for old_label, old_outcome in (('correct', old_correct[s]), ('wrong', ~old_correct[s])):
                active = mask & scope[s] & old_outcome
                scope_rows.append(dict(scenario=scenario, population=population, source_outcome=old_label,
                                       rows=int(active.sum()), corrected=int((active & corrected).sum()),
                                       broken=int((active & broken).sum()),
                                       changed_both_wrong=int((active & changed_wrong).sum()),
                                       unchanged=int((active & ~invariants['changed'][s]).sum()),
                                       exact_tie=int((active & (d == 0)).sum())))
        for label, cm in (('source', old_cm), ('candidate', new_cm)):
            frame = pd.DataFrame(cm, index=classes, columns=classes)
            frame.index.name = 'true_model'
            tables[f'{label}_scenario{scenario}_confusion'] = frame
    gains = np.asarray(candidate_scores) - np.asarray(source_scores)
    np.testing.assert_allclose(gains.mean(),
                               (int(invariants['corrected'].sum())-int(invariants['broken'].sum())) / 8100,
                               rtol=0, atol=1e-15)
    interval = paired_prompt_interval(y, groups, source, candidate, gains)
    assert interval['bootstrap_draws'] == 3000 and interval['bootstrap_seed'] == 20261017
    screen = bool(gains.mean() > 0 and interval['conditional_prompt_95_interval'][0] > 0 and gains.min() >= -.005)
    for key, prior in protected.items():
        assert array_digest(arrays[key]) == prior, ('Immutable source array mutated', key)
    assert array_digest(scope) == scope_sha256
    other_class_rows = [row for row in class_rows if not row['is_target_pair']]
    assert len(other_class_rows) == 48 and all(row['recall_gain'] == 0 for row in other_class_rows)
    pair_rows = [row for row in class_rows if row['is_target_pair']]
    saved = dict(ids=ids, classes=classes, y=y, prompt_ids=groups, fold=fold,
                 scenarios=np.asarray(SCENARIOS), pair_class_indices=rule['pair_class_indices'],
                 source_prediction_raw_control=source.copy(), candidate_prediction=candidate,
                 scope=scope, ordered_top2=rule['ordered_top2'],
                 qwen_pair_margin=rule['qwen_pair_margin'], deb_pair_margin=rule['deb_pair_margin'],
                 combined_pair_margin=d, true_pair_mask=true_pair,
                 source_pair_entry=invariants['source_pair_entry'], candidate_pair_entry=invariants['candidate_pair_entry'],
                 changed=invariants['changed'], corrected=invariants['corrected'], broken=invariants['broken'],
                 changed_both_wrong=invariants['changed_both_wrong'],
                 source_confusion=invariants['source_confusion'], candidate_confusion=invariants['candidate_confusion'])
    tables.update(context_summary=pd.DataFrame(context_rows), class_recall=pd.DataFrame(class_rows),
                  fold_scores=pd.DataFrame(fold_rows), prompt_counts=pd.DataFrame(prompt_rows),
                  other_class_false_entries=pd.DataFrame(false_rows), scope_outcomes=pd.DataFrame(scope_rows))
    findings = dict(source_raw_scores=source_scores, candidate_raw_scores=candidate_scores,
                    source_raw_mean=float(np.mean(source_scores)), candidate_raw_mean=float(np.mean(candidate_scores)),
                    scenario_gains=gains.tolist(), primary_mean_gain=float(gains.mean()),
                    paired_prompt_95_interval=interval['conditional_prompt_95_interval'], bootstrap=interval,
                    primary_raw_screen_pass=screen,
                    primary_screen_rule='Raw mean gain > 0; conditional prompt CI lower > 0; every context gain >= -0.005; all integrity and numerical gates passed. Class harms reported without a post-hoc veto.',
                    context_summary=context_rows, target_class_recalls=pair_rows,
                    source_scope_counts=observed, exact_tie_passages_all_rows=int((d == 0).sum()),
                    integrity_gates=dict(status='PASS', outside_scope_unchanged=True,
                                         changes_toggle_only_target_pair=True, pair_entry_masks_identical=True,
                                         all_16_nonpair_class_recalls_identical=True,
                                         true_pair_other_top_one_unchanged=True,
                                         counts_and_confusion_diagonals_reconciled=True,
                                         all_protected_source_arrays_unchanged=True, scope_unchanged=True,
                                         original_source_scope_counts_exact=True, exact_ties_keep_original=True),
                    protected_source_array_sha256=protected, scope_sha256=scope_sha256,
                    saved_array_sha256={key: array_digest(value) for key, value in saved.items()})
    return findings, saved, tables


def _verify_freeze(freeze_path):
    freeze_path = Path(freeze_path)
    freeze = json.loads(freeze_path.read_text())
    gate_path = Path(freeze.get('source_gate_path', Path(__file__).with_name('stage06f_pair_diagnostics.py')))
    plan_path = Path(freeze.get('plan_path', Path(__file__).with_name('STAGE_06G_PAIR_COMPLEMENT.md')))
    assert digest(__file__) == freeze['implementation_source_sha256']
    assert digest(gate_path) == freeze['source_gate_source_sha256'] == SOURCE_GATE_SHA256
    assert digest(plan_path) == freeze['plan_sha256'] == PLAN_SHA256
    assert freeze['input_version_ids'] == [170484, 359514801]
    assert freeze['accelerator'] == 'none' and freeze['internet_enabled'] is False
    spec = importlib.util.spec_from_file_location('stage06f_immutable_source_gate', gate_path)
    assert spec is not None and spec.loader is not None
    source_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source_module)
    assert tuple(source_module.PAIR) == PAIR and tuple(source_module.SCENARIOS) == SCENARIOS
    receipt = dict(freeze_sha256=digest(freeze_path), implementation_source_sha256=digest(__file__),
                   source_gate_source_sha256=digest(gate_path), plan_sha256=digest(plan_path),
                   input_version_ids=freeze['input_version_ids'], accelerator='none', internet_enabled=False)
    return source_module, receipt


def _save_chart(output_root, rows):
    """Small descriptive figure of the fixed endpoint and correction accounting."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    scenarios = [str(row['scenario']) for row in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), constrained_layout=True)
    gains = [100 * row['gain'] for row in rows]
    axes[0].bar(x, gains, color=['#237b55' if g > 0 else '#a73535' for g in gains])
    axes[0].axhline(0, color='#333333', linewidth=.8)
    axes[0].set(title='Fixed development gain', xlabel='Context size', ylabel='Balanced accuracy change (pp)', xticks=x, xticklabels=scenarios)
    axes[1].bar(x-.18, [row['corrected'] for row in rows], width=.36, label='Corrected', color='#237b55')
    axes[1].bar(x+.18, [row['broken'] for row in rows], width=.36, label='Broken', color='#a73535')
    axes[1].set(title='True-pair correction accounting', xlabel='Context size', ylabel='Passages per context', xticks=x, xticklabels=scenarios)
    axes[1].legend(frameon=False)
    fig.suptitle('Stage 06G: one fixed no-fit rule; reused development proxy')
    fig.savefig(output_root / 'context_gain.png', dpi=150)
    plt.close(fig)


def run_complement(input_root, output_root, freeze_path):
    """Production entry: strict freeze/source gates, one evaluation, exact readback."""
    assert freeze_path is not None, 'Production execution requires a source and plan freeze'
    output_root = Path(output_root)
    assert not output_root.exists() or not any(output_root.iterdir()), 'Fresh Stage 06G output directory required'
    input_root = Path(input_root)
    assert not output_root.resolve().is_relative_to(input_root.resolve()), 'Output cannot be inside immutable inputs'
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    def status(value, **extra):
        write_json(output_root / 'status.json', dict(status=value, observed_utc=datetime.now(timezone.utc).isoformat(), **extra))

    status('VERIFYING_FROZEN_SOURCE_AND_IMMUTABLE_INPUTS')
    try:
        source_module, freeze_receipt = _verify_freeze(freeze_path)
        arrays, train, source_result, paths = source_module._load_inputs(input_root)
        source_gate, unused_contribution = source_module._source_gate(arrays, train, source_result, paths)
        del train, unused_contribution
        source_gate['execution_freeze'] = freeze_receipt
        write_json(output_root / 'source_gate.json', source_gate)
        print('STAGE06G_SOURCE_GATE', json.dumps(source_gate, allow_nan=False), flush=True)
        status('EVALUATING_SINGLE_FIXED_ATTENTION_PAIR_RULE')
        findings, saved, tables = evaluate_candidate(arrays)
        np.testing.assert_allclose(findings['source_raw_scores'], source_gate['raw_control_scores'], rtol=0, atol=1e-15)
        assert all(not value.dtype.hasobject and value.dtype.kind in 'biufU' for value in saved.values())
        assert all(np.isfinite(value).all() for value in saved.values() if value.dtype.kind == 'f')
        np.savez_compressed(output_root / 'pair_complement_arrays.npz', **saved)
        with np.load(output_root / 'pair_complement_arrays.npz', allow_pickle=False) as readback:
            assert set(readback.files) == set(saved)
            for key, value in saved.items():
                assert readback[key].dtype == value.dtype
                np.testing.assert_array_equal(readback[key], value)
                assert array_digest(readback[key]) == findings['saved_array_sha256'][key]
        findings['integrity_gates']['numeric_unicode_npz_allow_pickle_false_exact_readback'] = True
        for name, frame in tables.items():
            frame.to_csv(output_root / (name + '.csv'), index=name.endswith('_confusion'))
        _save_chart(output_root, findings['context_summary'])
        result = dict(status=COMPLETE, pair=list(PAIR), contexts=list(SCENARIOS), rows=len(arrays['ids']),
                      classes=arrays['classes'].tolist(), class_supports=np.bincount(arrays['y'], minlength=18).tolist(),
                      primary_endpoint='Mean raw rowwise balanced accuracy across all 18 classes and three contexts',
                      source_script_version_id=359514801, diagnostic_selection_script_version_id=359521305,
                      fixed_rule=dict(members=['tokattn', 'tokattn_deb'], member_coefficients=[1.0, 1.0],
                                      probability_clip=[1e-9, 1], pair_margin_order='(qwen[A]-qwen[B])+(deb[A]-deb[B])',
                                      top2='Stable descending unchanged raw-control log evidence; class-index tie order',
                                      scope='Ordered top two labels exactly the fixed pair',
                                      positive_margin='A', negative_margin='B', exact_zero='Retain source prediction'),
                      **findings, source_gate=source_gate,
                      fits=0, model_inference_calls=0, regrouping_calls=0, optimization_or_threshold_search=False,
                      probability_calibration=False, batch_balancing=False, single_candidate_evaluated=True,
                      official_submission=False, final_selection_changed=False, paid_services=False,
                      official_S02_public_balanced_accuracy_separate=0.549,
                      runtime=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__),
                      seconds=time.monotonic()-started,
                      limitations=['Available-component development proxy, not full S02 or an official public score.',
                                   'Pair, members and scope selected after reused-development diagnostics; this is not untouched validation.',
                                   'Contexts reuse the same passages and fitted members; repeated context counts are not independent passages.',
                                   'Prompt interval conditions on fixed evidence and routing; omits refitting, scope-selection and future-data uncertainty.',
                                   'A positive development screen supports matched full-system investigation only; no official score change is established.',
                                   'Oracle correction of scoped mutual cross-errors is bounded by about 2.4815 proxy percentage points; this narrow rule cannot substantiate 70%.'])
        write_json(output_root / 'pair_complement_result.json', result)
        status(COMPLETE, valid=True, primary_raw_screen_pass=result['primary_raw_screen_pass'])
        manifest = {path.name: dict(bytes=path.stat().st_size, sha256=digest(path))
                    for path in sorted(output_root.iterdir()) if path.is_file() and path.name != 'output_manifest.json'}
        write_json(output_root / 'output_manifest.json', manifest)
        for name, frame in tables.items():
            print('STAGE06G_' + name.upper() + '_CSV',
                  json.dumps(frame.to_csv(index=name.endswith('_confusion'))), flush=True)
        print('STAGE06G_CONTEXT_SUMMARY', flush=True)
        print(tables['context_summary'][['scenario', 'source_balanced_accuracy', 'candidate_balanced_accuracy',
                                        'gain', 'corrected', 'broken', 'changed_both_wrong', 'scope_rows']].to_string(index=False), flush=True)
        print('STAGE06G_TARGET_CLASS_RECALLS', flush=True)
        print(tables['class_recall'][tables['class_recall'].is_target_pair][
            ['scenario', 'model', 'source_recall', 'candidate_recall', 'recall_gain']].to_string(index=False), flush=True)
        print('STAGE06G_RESULT', json.dumps(result, allow_nan=False), flush=True)
        print('STAGE06G_OUTPUT_MANIFEST', json.dumps(manifest, allow_nan=False), flush=True)
        print('STAGE06G_COMPLETE', flush=True)
        return result
    except BaseException as error:
        status('FAILED_PRESERVE_OUTPUTS', exception_type=type(error).__name__, exception_message=str(error))
        print('STAGE06G_FAILED', type(error).__name__, str(error), flush=True)
        raise


if __name__ == '__main__':
    freeze_path = os.environ.get('NTL_STAGE06G_FREEZE')
    assert freeze_path, 'Production execution requires NTL_STAGE06G_FREEZE'
    run_complement(os.environ.get('NTL_STAGE06G_INPUT', '/kaggle/input'),
                   os.environ.get('NTL_STAGE06G_OUTPUT', '/kaggle/working/stage06g_pair_complement'), freeze_path)
