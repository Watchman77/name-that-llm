"""Aggregate-only validation appended after unchanged Stage 08C inference.

No fitting, inference, network or model loading. Pass the live test/sub/P/LOGIT
objects and frozen LABELS to validate_output, then explicitly assert PASS.
Optional parquet_path reads the saved candidate back with pandas for parity.
No individual ID, passage, prediction, probability or logit is emitted.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

OFFICIAL_LABELS = ['gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra', 'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5', 'claude-fable-5', 'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3']


def validate_output(test, sub, P, LOGIT, LABELS, parquet_path=None):
    """Return bounded aggregate checks; callers must assert status == PASS.

    Optional readback parity compares the saved parquet to the passed candidate,
    including row order, IDs (the source writes string IDs) and model labels.
    """
    checks = []
    def check(name, passed, **counts):
        checks.append({'name': name, 'passed': bool(passed), **counts})
        return bool(passed)
    observed = {}
    try:
        labels = list(LABELS)
        check('labels:exact_official_order', labels == OFFICIAL_LABELS)
        test_ok = isinstance(test, pd.DataFrame) and 'id' in test.columns
        sub_ok = isinstance(sub, pd.DataFrame) and list(sub.columns) == ['id', 'model']
        check('test:id_column', test_ok)
        check('submission:exact_schema', sub_ok)
        n = len(test) if isinstance(test, pd.DataFrame) else 0
        observed['test_rows'] = n
        check('test:nonempty', n > 0)
        if test_ok:
            ids = test['id'].astype(str)
            check('test:unique_nonnull_nonempty_ids', test['id'].notna().all() and ids.str.strip().ne('').all() and ids.is_unique)
        if sub_ok:
            observed['submission_rows'] = len(sub)
            sids = sub['id'].astype(str)
            check('submission:row_count', len(sub) == n)
            check('submission:unique_nonnull_nonempty_ids', sub['id'].notna().all() and sids.str.strip().ne('').all() and sids.is_unique)
            check('submission:allowed_nonnull_labels', sub['model'].notna().all() and sub['model'].isin(OFFICIAL_LABELS).all())
            if test_ok:
                check('submission:exact_test_id_order', sids.tolist() == ids.tolist())
        arrays = {}
        for name, value in [('probabilities', P), ('logits', LOGIT)]:
            array = np.asarray(value)
            arrays[name] = array
            numeric = np.issubdtype(array.dtype, np.number) and not np.issubdtype(array.dtype, np.complexfloating)
            check(name + ':real_numeric', numeric)
            check(name + ':shape_N_by_18', array.shape == (n, 18))
            if numeric:
                check(name + ':finite', np.isfinite(array).all())
        probs = arrays['probabilities']
        numeric = np.issubdtype(probs.dtype, np.number) and not np.issubdtype(probs.dtype, np.complexfloating)
        if numeric:
            check('probabilities:bounds_0_to_1', ((probs >= 0) & (probs <= 1)).all())
            if probs.shape == (n, 18):
                sums = probs.sum(axis=1)
                check('probabilities:row_sums_one', np.allclose(sums, 1.0, rtol=1e-6, atol=1e-8))
                if sub_ok and len(sub) == n and labels == OFFICIAL_LABELS and np.isfinite(probs).all():
                    predicted = np.asarray(labels)[probs.argmax(axis=1)]
                    check('submission:matches_probability_argmax', np.array_equal(sub['model'].to_numpy(), predicted))
        if parquet_path is not None:
            try:
                saved = pd.read_parquet(Path(parquet_path))
                schema = list(saved.columns) == ['id', 'model']
                check('parquet:readable_exact_schema', schema)
                if schema and sub_ok:
                    check('parquet:row_count_parity', len(saved) == len(sub))
                    check('parquet:id_order_parity', saved['id'].astype(str).tolist() == sub['id'].astype(str).tolist())
                    check('parquet:model_label_parity', saved['model'].tolist() == sub['model'].tolist())
            except Exception as exc:
                check('parquet:readback', False, exception_type=type(exc).__name__)
        return {'stage': '08c_output_validation', 'status': 'PASS' if all(c['passed'] for c in checks) else 'FAIL',
                'checks': checks, 'observed': observed, 'parquet_readback_requested': parquet_path is not None,
                'individual_values_emitted': False, 'algorithm_modified': False}
    except Exception as exc:
        check('validation:no_exception', False, exception_type=type(exc).__name__)
        return {'stage': '08c_output_validation', 'status': 'FAIL', 'checks': checks,
                'observed': observed, 'raw_exception_message_omitted': True,
                'parquet_readback_requested': parquet_path is not None,
                'individual_values_emitted': False, 'algorithm_modified': False}


def self_test():
    from unittest.mock import patch
    tests = {}
    test = pd.DataFrame({'id': ['example-a', 'example-b'], 'text': ['synthetic', 'synthetic']})
    probs = np.full((2, 18), 0.5 / 17)
    probs[:, 0] = 0.5
    logits = np.log(probs)
    sub = pd.DataFrame({'id': test['id'], 'model': [OFFICIAL_LABELS[0]] * 2})
    # Keep the six checks in-memory; serialization/engine availability is checked
    # by the actual Kaggle readback, not claimed from this mocked boundary.
    path = 'synthetic-submission.parquet'
    with patch.object(pd, 'read_parquet', return_value=sub.copy()):
        tests['valid_probabilities_and_mocked_readback_parity'] = validate_output(test, sub, probs, logits, OFFICIAL_LABELS, path)['status'] == 'PASS'
        bad = probs.copy(); bad[0, 0] = np.nan
        bad_logits = logits.copy(); bad_logits[1, 0] = np.inf
        tests['nonfinite_probabilities_and_logits_rejected'] = validate_output(test, sub, bad, bad_logits, OFFICIAL_LABELS)['status'] == 'FAIL'
        wrong_ids = sub.iloc[::-1].reset_index(drop=True)
        duplicate_ids = sub.copy(); duplicate_ids.loc[1, 'id'] = duplicate_ids.loc[0, 'id']
        tests['wrong_order_and_duplicate_ids_rejected'] = all(validate_output(test, candidate, probs, logits, OFFICIAL_LABELS)['status'] == 'FAIL' for candidate in [wrong_ids, duplicate_ids])
        invalid = sub.copy(); invalid.loc[0, 'model'] = 'invalid-model'
        mismatched = sub.copy(); mismatched.loc[0, 'model'] = OFFICIAL_LABELS[1]
        tests['invalid_label_and_argmax_mismatch_rejected'] = all(validate_output(test, candidate, probs, logits, OFFICIAL_LABELS)['status'] == 'FAIL' for candidate in [invalid, mismatched])
        scaled = probs * 0.5
        negative = probs.copy(); negative[0, 1] = -0.01
        tests['shape_bounds_and_normalization_rejected'] = all(validate_output(test, sub, candidate, logits, OFFICIAL_LABELS)['status'] == 'FAIL' for candidate in [probs[:, :-1], scaled, negative])
    with patch.object(pd, 'read_parquet', return_value=mismatched):
        tests['changed_mocked_saved_parquet_rejected'] = validate_output(test, sub, probs, logits, OFFICIAL_LABELS, path)['status'] == 'FAIL'
    return {'stage': '08c_output_validation_synthetic_tests', 'status': 'PASS' if all(tests.values()) else 'FAIL',
            'tests': tests, 'real_inputs_read': False, 'model_code_executed': False,
            'parquet_serialization_tested': False, 'readback_boundary_mocked': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true', required=True)
    parser.parse_args()
    report = self_test()
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report['status'] == 'PASS' else 2


if __name__ == '__main__':
    raise SystemExit(main())
