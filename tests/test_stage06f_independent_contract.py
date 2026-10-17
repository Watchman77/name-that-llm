"""Independent no-fit checks of the fixed Stage 06F diagnostic contract.

Hand-counted synthetic data and archived Stage06E confusion matrices only.
No real model, probability cache, classifier or browser is invoked.
"""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'scripts/stage06f_pair_diagnostics.py'
SPEC = importlib.util.spec_from_file_location('stage06f_independent_subject', SOURCE)
SUBJECT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUBJECT)


class IndependentContract(unittest.TestCase):
    def test_source_has_no_estimator_or_search_invocations(self):
        tree = ast.parse(SOURCE.read_text())
        forbidden = {'fit', 'fit_transform', 'predict', 'predict_proba', 'decision_function',
                     'transform', 'partial_fit', 'minimize', 'grid_search', 'cluster',
                     'AgglomerativeClustering', 'LogisticRegression', 'load_model'}
        calls = [n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id
                 for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, (ast.Name, ast.Attribute))]
        self.assertFalse(set(calls) & forbidden)
        imports = [n.module or '' for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        imports += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        self.assertFalse(any(x.split('.')[0] in {'sklearn', 'torch', 'transformers', 'requests',
                                               'urllib', 'pickle'} for x in imports))

    def test_every_npz_load_explicitly_disables_pickle(self):
        loads = [n for n in ast.walk(ast.parse(SOURCE.read_text())) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and n.func.attr == 'load']
        self.assertEqual(len(loads), 2)
        for n in loads:
            self.assertTrue(any(k.arg == 'allow_pickle' and isinstance(k.value, ast.Constant)
                                and k.value.value is False for k in n.keywords))

    def test_immutable_hash_constants_match_archived_e_outputs(self):
        for name in ('ensemble_result.json',
                     'raw_control_scenario1_confusion.csv', 'raw_control_scenario3_confusion.csv',
                     'raw_control_scenario6_confusion.csv'):
            actual = hashlib.sha256((ROOT / 'results/stage_06e' / name).read_bytes()).hexdigest()
            self.assertEqual(actual, SUBJECT.EXPECTED_HASHES[name], name)
        manifest = json.loads((ROOT / 'results/stage_06e/output_manifest.json').read_text())
        self.assertEqual(manifest['ensemble_arrays.npz']['sha256'],
                         SUBJECT.EXPECTED_HASHES['ensemble_arrays.npz'])
        result = json.loads((ROOT / 'results/stage_06e/ensemble_result.json').read_text())
        self.assertEqual(result['fixed_weights'], SUBJECT.WEIGHTS)
        self.assertEqual(SUBJECT.PAIR, ('gemini-3.7-flash', 'gemini-3.8-flash'))

    def test_writer_reconstructed_manifest_candidate_is_semantically_identical(self):
        # The archive is a writer reconstruction, not a downloaded original.
        # Original Kaggle bytes will be verified by the unchanged strict runtime gate.
        prior = ROOT / 'results/stage_06e/output_manifest.json'
        candidate = ROOT / 'results/stage_06f/E_MANIFEST_WRITER_RECONSTRUCTED.json'
        self.assertEqual(hashlib.sha256(prior.read_bytes()).hexdigest(),
                         '626b428135f4ff377d593de29c716042a07593c74678893f874c91b7b71e2bbf')
        old, reconstructed = json.loads(prior.read_text()), json.loads(candidate.read_text())
        self.assertEqual(old, reconstructed)
        self.assertEqual(len(reconstructed), 20)
        expected = {key: dict(sha256=value['sha256'], bytes=value['bytes'])
                    for key, value in sorted(old.items())}
        exact = (json.dumps(expected, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()
        self.assertEqual(candidate.read_bytes(), exact)
        self.assertEqual(hashlib.sha256(exact).hexdigest(), SUBJECT.EXPECTED_HASHES['output_manifest.json'])
        self.assertTrue(all(list(value) == ['sha256', 'bytes'] for value in reconstructed.values()))

    def test_confusion_matches_hand_count(self):
        got = SUBJECT.fixed_confusion(np.array([0, 0, 1, 1, 2]),
                                      np.array([0, 1, 0, 1, 2]), 3)
        np.testing.assert_array_equal(got, [[1, 1, 0], [1, 1, 0], [0, 0, 1]])

    def test_confusion_rejects_shape_mismatch(self):
        with self.assertRaises(AssertionError):
            SUBJECT.fixed_confusion(np.array([0]), np.array([0, 1]), 3)

    def test_confusion_rejects_out_of_bounds(self):
        with self.assertRaises(AssertionError):
            SUBJECT.fixed_confusion(np.array([0]), np.array([3]), 3)

    def _metrics(self, evidence=None, pair=(0, 1)):
        if evidence is None:
            evidence = np.broadcast_to(np.array([[2., -2., 0.], [2., -2., 0.],
                                                [-1., -1., 2.], [0., 0., -1.]]), (3, 4, 3)).copy()
        probability = np.exp(evidence - evidence.max(2, keepdims=True))
        probability /= probability.sum(2, keepdims=True)
        return SUBJECT.row_metrics(evidence, probability, np.array([0, 1, 2, 0]), pair)

    def test_signed_and_truth_oriented_margins(self):
        got = self._metrics()
        np.testing.assert_array_equal(got['pair_margin'], np.tile([4., 4., 0., 0.], (3, 1)))
        np.testing.assert_array_equal(got['truth_margin'], np.tile([4., -4., 0., 0.], (3, 1)))
        np.testing.assert_array_equal(got['true_pair'], [True, True, False, True])
        np.testing.assert_array_equal(got['orientation'], [1., -1., 0., 1.])

    def test_pair_mass_is_logistic_and_not_global_mass(self):
        got = self._metrics()
        np.testing.assert_allclose(got['pair_a_mass'][:, :2], 1 / (1 + np.exp(-4)), atol=1e-15)
        self.assertLess(float(got['pair_global_mass'][0, 2]), .1)
        self.assertEqual(float(got['pair_a_mass'][0, 2]), .5)

    def test_extreme_margins_remain_finite(self):
        evidence = np.broadcast_to(np.array([[1000., 0., -1.], [0., 1000., -1.],
                                            [0., 0., 0.], [0., 0., 0.]]), (3, 4, 3)).copy()
        with np.errstate(over='raise', invalid='raise'):
            got = self._metrics(evidence)
        self.assertTrue(np.isfinite(got['pair_a_mass']).all())
        np.testing.assert_array_equal(got['pair_a_mass'][0], [1., 0., .5, .5])
        np.testing.assert_array_equal(got['pair_confidence'][0], [1., 1., .5, .5])

    def test_top_two_ties_have_stable_index_order(self):
        got = self._metrics(np.zeros((3, 4, 3)), pair=(1, 2))
        np.testing.assert_array_equal(got['ordered_top2'], np.tile([0, 1], (3, 4, 1)))
        self.assertFalse(got['top2_is_pair'].any())

    def test_pair_partition_and_routing_false_entries(self):
        y = np.array([0, 1, 0, 1, 0, 1, 2, 2]); p = np.array([0, 1, 1, 0, 2, 2, 0, 2])
        got = SUBJECT.pair_counts(y, p, (0, 1), np.array([1, 0, 1, 0, 0, 0, 1, 0], bool))
        expected = dict(pair_support=6, pair_correct=2, pair_cross=2, pair_other_error=2,
                        a_to_b=1, b_to_a=1, false_entries=1, predicted_pair=5, global_errors=5,
                        top2_scope_rows=3, top2_scope_true_pair=2, top2_scope_cross=1,
                        top2_scope_pair_correct=1, top2_scope_other_class=1, top2_scope_false_entries=1)
        self.assertEqual(got, expected)

    def test_context_overlap_counts_passages_once(self):
        mask = np.array([[1, 1, 1, 0, 0], [1, 1, 0, 1, 0], [1, 0, 0, 1, 0]], bool)
        got = SUBJECT.context_overlap(mask, np.array(list('abcde')))
        self.assertEqual(got['repeated_context_observations'], 8)
        self.assertEqual(got['unique_passages_union'], 4)
        self.assertEqual(got['unique_passages_all_contexts'], 1)
        self.assertEqual(got['unique_passages_by_number_of_contexts'], {'0': 1, '1': 1, '2': 2, '3': 1})
        self.assertEqual(got['pairwise_intersections'], {'1_3': 2, '1_6': 1, '3_6': 2})
        self.assertEqual(sum(got['context_pattern_counts'].values()), 5)

    def test_context_overlap_rejects_duplicate_ids(self):
        with self.assertRaises(AssertionError):
            SUBJECT.context_overlap(np.zeros((3, 2), bool), np.array(['same', 'same']))

    def test_margin_bins_cover_boundaries_once(self):
        rows = SUBJECT._binned(np.array([0., .25, .5, 1., 2., 4., 100.]),
                               np.ones(7, bool), np.ones(7), SUBJECT.MARGIN_BINS, 'test', 1)
        self.assertEqual([r['rows'] for r in rows], [1, 1, 1, 1, 1, 2])
        self.assertEqual(sum(r['correct'] for r in rows), 7)

    def test_confidence_final_bin_includes_one(self):
        rows = SUBJECT._binned(np.array([0., .5, .6, .7, .8, .9, 1.]), np.ones(7, bool),
                               np.full(7, 2), SUBJECT.CONFIDENCE_BINS, 'test', 1)
        self.assertEqual([r['rows'] for r in rows], [1, 1, 1, 1, 1, 2])
        self.assertEqual(sum(r['cross'] for r in rows), 7)

    def test_text_shape_hand_counts_without_text_output(self):
        names, values = SUBJECT.text_features(['A1?', ''])
        self.assertEqual(values.shape, (2, 12))
        self.assertEqual(names.dtype.kind, 'U')
        row = dict(zip(names.tolist(), values[0]))
        self.assertEqual(row['characters'], 3); self.assertEqual(row['words'], 1)
        self.assertEqual(row['questions'], 1); self.assertEqual(row['mean_word_characters'], 3)
        self.assertAlmostEqual(row['uppercase_fraction'], 1/3)
        self.assertAlmostEqual(row['digit_fraction'], 1/3)
        self.assertAlmostEqual(row['punctuation_fraction'], 1/3)
        np.testing.assert_array_equal(values[1], np.zeros(12))

    def test_constant_prompt_rates_have_exact_descriptive_intervals(self):
        values = np.array([[1, 0, 1, 0], [1, 1, 1, 1], [0, 0, 0, 0]])
        got = SUBJECT.rate_interval(values, np.array(['a', 'a', 'b', 'b']), np.ones(4, bool), draws=50)
        np.testing.assert_array_equal(got['by_context'], [[.5, .5], [1., 1.], [0., 0.]])
        np.testing.assert_array_equal(got['mean_across_contexts'], [.5, .5])

    def test_joint_bootstrap_keeps_complementary_context_mean_fixed(self):
        got = SUBJECT.rate_interval(np.array([[1, 1, 0, 0], [0, 0, 1, 1]]),
                                    np.array(['a', 'a', 'b', 'b']), np.ones(4, bool), draws=100)
        np.testing.assert_array_equal(got['mean_across_contexts'], [.5, .5])
        self.assertEqual(got['draws'], 100); self.assertEqual(got['seed'], 20261018)

    def test_archived_confusions_reconcile_fixed_pair_counts(self):
        expected = [(50, 44, 135, 71, 65, 1278), (49, 38, 146, 67, 54, 1206),
                    (52, 42, 139, 67, 49, 1174)]
        for scenario, counts in zip((1, 3, 6), expected):
            cm = pd.read_csv(ROOT / f'results/stage_06e/raw_control_scenario{scenario}_confusion.csv', index_col=0)
            n = len(cm); y, pred = [], []
            for i in range(n):
                for j in range(n):
                    y.extend([i] * int(cm.iloc[i, j])); pred.extend([j] * int(cm.iloc[i, j]))
            pair = tuple(cm.index.get_loc(x) for x in SUBJECT.PAIR)
            got = SUBJECT.pair_counts(np.array(y), np.array(pred), pair)
            self.assertEqual(tuple(got[k] for k in ('a_to_b', 'b_to_a', 'pair_correct',
                                                    'pair_other_error', 'false_entries', 'global_errors')), counts)
            self.assertEqual(got['pair_support'], 300)

    def test_complete_synthetic_analysis_preserves_rows_and_neutral_components(self):
        y = np.array([0, 1, 0, 1, 0, 1, 2, 2]); pred = np.array([0, 1, 1, 0, 2, 2, 0, 2])
        evidence = np.zeros((3, 8, 3))
        for s in range(3):
            evidence[s, np.arange(8), pred] = 2
        prob = np.exp(evidence - evidence.max(2, keepdims=True)); prob /= prob.sum(2, keepdims=True)
        arrays = dict(ids=np.array([f'id{i}' for i in range(8)]),
                      classes=np.array([*SUBJECT.PAIR, 'other']), y=y,
                      prompt_ids=np.full(8, 'p0'), fold=np.ones(8, dtype=int),
                      log_evidence_control=evidence, raw_probability_control=prob,
                      prediction_raw_control=np.tile(pred, (3, 1)), eligible=np.zeros((3, 8), bool))
        original = {key: value.copy() for key, value in arrays.items()}
        contribution = np.zeros((3, 8, 7))
        contribution[..., 0] = evidence[..., 0] - evidence[..., 1]
        # One wrong pair decision has an opposing member, while total evidence stays fixed.
        contribution[:, 2, 1] = 1
        contribution[:, 2, 0] -= 1
        findings, saved, tables = SUBJECT.analyze(arrays, pd.DataFrame({'text': ['A1?'] * 8}), contribution)
        for key, value in original.items():
            np.testing.assert_array_equal(arrays[key], value)
        np.testing.assert_array_equal(saved['source_prediction_raw_control'], original['prediction_raw_control'])
        self.assertTrue(all(not value.dtype.hasobject for value in saved.values()))
        self.assertFalse(saved['member_pair_preference'][..., -1].any())
        center = [row for row in tables['component_contributions'] if row['member'] == 'deb_center']
        self.assertEqual(len(center), 9)
        self.assertTrue(all(row['neutral'] == row['rows'] == 2 for row in center))
        self.assertTrue(all(row['supports_true'] == row['supports_other_pair'] == 0 for row in center))
        self.assertEqual(findings['overlap']['pair_cross']['unique_passages_union'], 2)
        self.assertEqual(findings['overlap']['pair_cross']['repeated_context_observations'], 6)
        self.assertEqual(findings['overlap']['pair_cross']['population_passages'], 6)
        self.assertEqual(findings['overlap']['nonpair_false_entry']['population_passages'], 2)
        self.assertEqual([r['cross_rows_with_any_member_supporting_true'] for r in findings['disagreements']], [1, 1, 1])
        self.assertEqual(len(tables['pair_rows']), 21)
        self.assertTrue(all('is_true_pair' in row for row in tables['pair_rows']))
        for rows in tables.values():
            for row in rows:
                self.assertFalse(set(row) & {'text', 'passage', 'tokens', 'snippet', 'transcript'})

    def test_nonempty_output_is_preserved_and_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'output'; output.mkdir(); marker = output / 'prior.txt'
            marker.write_text('prior trial')
            with self.assertRaises(AssertionError):
                SUBJECT.run_diagnostics(Path(tmp) / 'no-input', output)
            self.assertEqual(marker.read_text(), 'prior trial')
            self.assertEqual([p.name for p in output.iterdir()], ['prior.txt'])

    def test_corrupt_freeze_blocks_before_input_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); freeze = root / 'freeze.json'
            freeze.write_text(json.dumps({'diagnostic_source_sha256': '0' * 64}))
            with self.assertRaises(AssertionError):
                SUBJECT.run_diagnostics(root / 'no-input', root / 'output', freeze)
            status = json.loads((root / 'output/status.json').read_text())
            self.assertEqual(status['status'], 'FAILED_PRESERVE_OUTPUTS')
            self.assertFalse((root / 'output/pair_result.json').exists())


if __name__ == '__main__':
    unittest.main()
