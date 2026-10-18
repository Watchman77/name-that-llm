"""Independent, hand-expected Stage 06G contract fixtures.

Only generated synthetic arrays are evaluated here. No real E probability NPZ,
model, browser, official submission or candidate performance is invoked.
"""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'scripts/stage06g_pair_complement.py'
SPEC = importlib.util.spec_from_file_location('stage06g_independent_subject', SOURCE)
SUBJECT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUBJECT)
PAIR = ('gemini-3.7-flash', 'gemini-3.8-flash')


def fixture(dtype=np.float64):
    """Nine hand-labelled rows, three identical contexts, 18 classes."""
    classes = np.asarray([*PAIR, 'fable', *[f'other{i}' for i in range(15)]])
    # Rows0/1 correctable cross-errors; 2/3 breakable correct answers; 4/5
    # already-wrong nonpair entries; 6 a pair truth routed elsewhere; 7 has
    # a different runner-up; 8 has an exact native complement tie.
    truth = np.asarray([0, 1, 0, 1, 2, 2, 0, 0, 1])
    source = np.asarray([1, 0, 0, 1, 0, 1, 2, 0, 1])
    evidence = np.full((3, 9, 18), -10., dtype=dtype)
    evidence[..., 0] = 1; evidence[..., 1] = 1
    for i, top in enumerate(source):
        evidence[:, i, top] = 2
    evidence[:, 7, 2] = 1.5  # Pair top one, but outside exact top-two scope.
    qwen = np.full((9, 18), dtype(.5 / 16), dtype=dtype)
    deb = qwen.copy()
    decisions = [0, 1, 1, 0, 1, 0, 1, 1, None]
    for i, decision in enumerate(decisions):
        qwen[i, :2] = (.25, .25) if decision is None else ((.3, .2) if decision == 0 else (.2, .3))
        deb[i, :2] = qwen[i, :2]
    return classes, truth, evidence, np.tile(source, (3, 1)), qwen, deb


class IndependentContract(unittest.TestCase):
    def test_no_estimator_model_network_or_search_calls(self):
        tree = ast.parse(SOURCE.read_text())
        forbidden = {'fit', 'fit_transform', 'predict', 'predict_proba', 'decision_function',
                     'partial_fit', 'minimize', 'GridSearchCV', 'RandomizedSearchCV',
                     'LogisticRegression', 'load_model'}
        calls = [n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id
                 for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, (ast.Name, ast.Attribute))]
        self.assertFalse(set(calls) & forbidden)
        imports = [n.module or '' for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        imports += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        self.assertFalse(any(x.split('.')[0] in {'sklearn', 'torch', 'transformers', 'requests',
                                               'urllib', 'pickle'} for x in imports))

    def test_bound_source_gate_and_prospective_plan_unchanged(self):
        expected = {
            'scripts/stage06f_pair_diagnostics.py': 'a8bb68181087640fb563539870840117531a4d57e529cbeaf69e73d4c0a05be7',
            'plans/STAGE_06G_PAIR_COMPLEMENT.md': 'ece986eb6244bd322564cd7fd7a437239e5f74635a2cea2b9a39968bd891b927',
        }
        for name, digest in expected.items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest)

    def test_hand_expected_fixes_damage_scope_and_wrong_destination_changes(self):
        classes, truth, evidence, source, qwen, deb = fixture()
        got = SUBJECT.derive_candidate(classes, evidence, source, qwen, deb)
        expected = np.tile([0, 1, 1, 0, 1, 0, 2, 0, 1], (3, 1))
        np.testing.assert_array_equal(got['candidate_prediction'], expected)
        np.testing.assert_array_equal(got['scope'], np.tile([1, 1, 1, 1, 1, 1, 0, 0, 1], (3, 1)))
        # Two fixes, two breaks and two destination changes with unchanged error
        # status. The two truths in other classes remain incorrect throughout.
        corrected = (source != truth) & (expected == truth)
        broken = (source == truth) & (expected != truth)
        both_wrong = (source != expected) & (source != truth) & (expected != truth)
        np.testing.assert_array_equal(corrected.sum(1), [2, 2, 2])
        np.testing.assert_array_equal(broken.sum(1), [2, 2, 2])
        np.testing.assert_array_equal(both_wrong.sum(1), [2, 2, 2])
        np.testing.assert_array_equal((source != expected).sum(1), [6, 6, 6])
        np.testing.assert_array_equal(np.isin(source, [0, 1]), np.isin(expected, [0, 1]))
        np.testing.assert_array_equal((source == truth)[:, truth > 1], (expected == truth)[:, truth > 1])

    def test_all_source_arrays_remain_byte_identical(self):
        classes, _, evidence, source, qwen, deb = fixture(np.float32)
        values = [classes, evidence, source, qwen, deb]
        before = [(x.dtype.str, x.shape, x.tobytes()) for x in values]
        got = SUBJECT.derive_candidate(classes, evidence, source, qwen, deb)
        self.assertEqual(before, [(x.dtype.str, x.shape, x.tobytes()) for x in values])
        self.assertFalse(np.shares_memory(got['candidate_prediction'], source))
        self.assertFalse(np.shares_memory(got['scope'], evidence))

    def test_stable_top_two_uses_original_class_index_ties(self):
        classes, _, evidence, source, qwen, deb = fixture()
        evidence.fill(0); source.fill(0)
        got = SUBJECT.derive_candidate(classes, evidence, source, qwen, deb)
        np.testing.assert_array_equal(got['ordered_top2'], np.tile([0, 1], (3, 9, 1)))
        self.assertTrue(got['scope'].all())
        classes = classes.copy(); classes[[0, 2]] = classes[[2, 0]]
        got = SUBJECT.derive_candidate(classes, evidence, source, qwen, deb)
        self.assertFalse(got['scope'].any())
        np.testing.assert_array_equal(got['candidate_prediction'], source)

    def test_native_exact_zero_keeps_control_and_near_zero_sign_is_not_tied(self):
        classes, _, evidence, source, qwen, deb = fixture()
        qwen[:, :2] = .25; deb[:, :2] = .25
        tiny = np.nextafter(np.float64(.25), np.float64(1))
        deb[0, 0] = tiny; deb[1, 1] = tiny
        qwen[8, :2] = [.3, .2]; deb[8, :2] = [.2, .3]
        got = SUBJECT.derive_candidate(classes, evidence, source, qwen, deb)
        d = got['combined_pair_margin']
        self.assertGreater(d[0], 0); self.assertLess(d[1], 0)
        self.assertLess(abs(d[0]), 1e-14); self.assertLess(abs(d[1]), 1e-14)
        self.assertEqual(d[8], 0)
        np.testing.assert_array_equal(got['candidate_prediction'][:, :2], np.tile([0, 1], (3, 1)))
        np.testing.assert_array_equal(got['candidate_prediction'][:, 8], source[:, 8])

    def test_full_matrix_clipping_logs_retain_native_float32_arithmetic(self):
        classes, _, evidence, source, qwen, deb = fixture(np.float32)
        qwen[0, 0] = 0; qwen[0, 1] = np.float32(.5)
        got = SUBJECT.derive_candidate(classes, evidence, source, qwen, deb)
        q = np.log(np.clip(qwen, 1e-9, 1)); d = np.log(np.clip(deb, 1e-9, 1))
        expected = (q[:, 0] - q[:, 1]) + (d[:, 0] - d[:, 1])
        self.assertEqual(got['combined_pair_margin'].dtype, np.dtype('float32'))
        np.testing.assert_array_equal(got['combined_pair_margin'], expected)
        self.assertTrue(np.isfinite(expected).all())

    def test_malformed_context_probability_and_prediction_shapes_fail(self):
        classes, _, evidence, source, qwen, deb = fixture()
        cases = [(classes, evidence[:, :, :-1], source, qwen, deb),
                 (classes, evidence, source[:, :-1], qwen, deb),
                 (classes, evidence, source, qwen[:-1], deb)]
        for args in cases:
            with self.subTest(shapes=[x.shape for x in args]):
                with self.assertRaises((AssertionError, ValueError)):
                    SUBJECT.derive_candidate(*args)

    def test_full_18_class_confusions_keep_149_support_and_16_recalls_unchanged(self):
        classes, truth, evidence, source, qwen, deb = fixture()
        got = SUBJECT.derive_candidate(classes, evidence, source, qwen, deb)
        target_support = np.asarray([150, 150, 149, *([150] * 15)])
        extra = np.repeat(np.arange(18), target_support - np.bincount(truth, minlength=18))
        y = np.concatenate([truth, extra])
        original = np.concatenate([source, np.tile(extra, (3, 1))], axis=1)
        candidate = np.concatenate([got['candidate_prediction'], np.tile(extra, (3, 1))], axis=1)
        scope = np.concatenate([got['scope'], np.zeros((3, len(extra)), dtype=bool)], axis=1)
        candidate[:, 3] = source[:, 3]  # Retain one correct answer: net +1/context.
        audit = SUBJECT.comparison_invariants(y, original, candidate, scope, (0, 1), 18)
        self.assertEqual(len(y), 2699)
        np.testing.assert_array_equal(audit['source_confusion'].sum(2), np.tile(target_support, (3, 1)))
        np.testing.assert_array_equal(audit['corrected'].sum(1), [2, 2, 2])
        np.testing.assert_array_equal(audit['broken'].sum(1), [1, 1, 1])
        np.testing.assert_array_equal(audit['changed_both_wrong'].sum(1), [2, 2, 2])
        old = np.diagonal(audit['source_confusion'], axis1=1, axis2=2) / target_support
        new = np.diagonal(audit['candidate_confusion'], axis1=1, axis2=2) / target_support
        np.testing.assert_array_equal(new[:, 2:], old[:, 2:])
        np.testing.assert_allclose((new - old).mean(1), [1 / 2700] * 3, rtol=0, atol=1e-17)
        self.assertNotEqual(1 / 2700, 1 / 2699)
        # The two nonpair false-entry destinations toggle, but their error status
        # and class2 recall stay fixed.
        np.testing.assert_array_equal(audit['candidate_confusion'][:, 2, :2],
                                      audit['source_confusion'][:, 2, :2][:, ::-1])
        outside_change = candidate.copy(); outside_change[:, 6] = 0
        with self.assertRaises(AssertionError):
            SUBJECT.comparison_invariants(y, original, outside_change, scope, (0, 1), 18)
        nonpair_entry_change = candidate.copy(); nonpair_entry_change[:, 4] = 2
        with self.assertRaises(AssertionError):
            SUBJECT.comparison_invariants(y, original, nonpair_entry_change, scope, (0, 1), 18)

    def _full_prompt_panel(self):
        # One example/class/prompt, except the149th Fable example is absent.
        y = np.tile(np.arange(18), 150)
        groups = np.repeat(np.asarray([f'p{i:03d}' for i in range(150)]), 18)
        keep = ~((y == 2) & (groups == 'p149'))
        return y[keep], groups[keep]

    def test_joint_prompt_bootstrap_cancels_opposite_context_effects(self):
        y, groups = self._full_prompt_panel()
        source = np.tile(y, (3, 1)); candidate = source.copy()
        target = (y == 0) & (groups < 'p075')
        source[0, target] = 1  # +75 corrected in context0.
        candidate[1, target] = 1  # -75 broken in context1, same prompts.
        got = SUBJECT.paired_prompt_interval(y, groups, source, candidate,
                                             [75 / 2700, -75 / 2700, 0])
        self.assertEqual(got['bootstrap_draws'], 3000)
        self.assertEqual(got['bootstrap_seed'], 20261017)
        self.assertEqual(got['prompt_count'], 150)
        self.assertEqual(got['mean_gain'], 0)
        np.testing.assert_array_equal(got['conditional_prompt_95_interval'], [0, 0])

    def test_prompt_bootstrap_matches_independent_hand_count_oracle_with_149_support(self):
        y, groups = self._full_prompt_panel()
        source = np.tile(y, (3, 1)); candidate = source.copy()
        # A149-support class and a150-support class each lose a fixed row.
        # This synthetic interval helper check intentionally permits arbitrary
        # labels; production candidate/invariants separately restrict A/B toggles.
        candidate[:, (y == 2) & (groups == 'p000')] = 0
        candidate[:, (y == 3) & (groups == 'p001')] = 0
        gains = np.full(3, -(1 / 149 + 1 / 150) / 18)
        got = SUBJECT.paired_prompt_interval(y, groups, source, candidate, gains)
        draws = np.random.default_rng(20261017).integers(150, size=(3000, 150))
        # Class2 has support for every drawn prompt other than p149. Class3
        # has one row in all150 prompts. The two observed row losses occur
        # whenever p000/p001 is sampled, including repeated draws.
        loss149 = (draws == 0).sum(1) / (150 - (draws == 149).sum(1))
        loss150 = (draws == 1).sum(1) / 150
        oracle = -(loss149 + loss150) / 18
        np.testing.assert_allclose(got['conditional_prompt_95_interval'],
                                   np.quantile(oracle, [.025, .975]), rtol=0, atol=1e-18)
        self.assertAlmostEqual(got['mean_gain'], gains[0], places=16)

    def test_all_npz_readbacks_explicitly_disable_pickle(self):
        loads = [n for n in ast.walk(ast.parse(SOURCE.read_text())) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and n.func.attr == 'load']
        self.assertEqual(len(loads), 1)
        self.assertTrue(all(any(k.arg == 'allow_pickle' and isinstance(k.value, ast.Constant)
                               and k.value.value is False for k in n.keywords) for n in loads))

    def test_prior_output_is_preserved_and_refused_before_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); output = root / 'output'; output.mkdir()
            marker = output / 'prior.txt'; marker.write_text('prior trial')
            with self.assertRaises(AssertionError):
                SUBJECT.run_complement(root / 'no-input', output, root / 'no-freeze')
            self.assertEqual(marker.read_text(), 'prior trial')
            self.assertEqual([path.name for path in output.iterdir()], ['prior.txt'])

    def test_corrupt_implementation_freeze_fails_before_input_or_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); freeze = root / 'freeze.json'
            freeze.write_text(json.dumps({'implementation_source_sha256': '0' * 64}))
            with self.assertRaises(AssertionError):
                SUBJECT.run_complement(root / 'no-input', root / 'output', freeze)
            status = json.loads((root / 'output/status.json').read_text())
            self.assertEqual(status['status'], 'FAILED_PRESERVE_OUTPUTS')
            self.assertFalse((root / 'output/pair_complement_result.json').exists())


if __name__ == '__main__':
    unittest.main()
