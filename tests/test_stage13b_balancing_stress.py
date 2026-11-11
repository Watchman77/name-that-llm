"""Synthetic coverage for the balancing boundary, matched masks and count reconciliation."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import stage13b_balancing_stress as stress


class StressChecks(unittest.TestCase):
    def test_17_row_bypass_and_18_row_source_transform(self):
        p = np.tile(np.arange(1, 19, dtype=np.float64) / 171, (18, 1))
        old = p.copy()
        log17, residual17 = stress.evidence(p[:17], True)
        self.assertIsNone(residual17)
        np.testing.assert_array_equal(log17, np.log(p[:17] + 1e-12))
        log18, residual18 = stress.evidence(p, True)
        np.testing.assert_allclose(log18, np.log(1/18 + 1e-12), atol=1e-14, rtol=0)
        self.assertLess(residual18, 1e-12)
        np.testing.assert_array_equal(p, old)

    def test_reject_nonpositive_input(self):
        p = np.ones((18, 18)) / 18
        p[0, 0] = 0
        with self.assertRaisesRegex(stress.GateError, "invalid_probability_values"):
            stress.evidence(p, True)

    def test_paired_scenarios_reconcile_on_synthetic_axes(self):
        y = np.tile(np.arange(18), 150)[:-1]
        ids = np.asarray([f"fixture-{i}" for i in range(len(y))])
        groups = np.repeat([f"g{i:03d}" for i in range(150)], 18)[:-1]
        folds = np.repeat(np.arange(1, 6), 540)[:-1]
        classes = np.asarray([f"model-{i}" for i in range(18)])
        p = np.random.default_rng(9).uniform(0.01, 1.0, (len(y), 18))
        p[np.arange(len(y)), y] *= 1.15
        p /= p.sum(1, keepdims=True)
        cm = stress.confusion(y, p.argmax(1))
        anchor = float(stress.recalls(cm).mean())
        prior = {"confusion_counts": cm.tolist(), "pooled_balanced_accuracy": anchor}
        with patch.object(stress, "EXPECTED_RECALL", anchor):
            r, gates = stress.diagnostics(ids, groups, y, folds, classes, p, prior)
        self.assertEqual(set(r), set(stress.SCENARIOS))
        self.assertEqual(r["natural_full"]["off_ba"], anchor)
        self.assertEqual(r["matched_balanced"]["class_supports"], [100] * 18)
        np.testing.assert_array_equal(np.add(r["skew_a"]["class_supports"], r["skew_b"]["class_supports"]), [200] * 18)
        for b in r.values():
            self.assertEqual(len(b["batches"]), 5)
            self.assertEqual(np.trace(b["off_confusion_counts"]) - np.trace(b["uniform_confusion_counts"]),
                             b["corrected_by_off"] - b["broken_by_off"])
        self.assertEqual(len(gates), 4)

    def test_missing_cache_is_a_failure_without_scores(self):
        with tempfile.TemporaryDirectory() as temp:
            r = stress.run(temp, Path(temp) / "report")
        self.assertEqual(r["status"], "FAILED_GLM_SINKHORN_STRESS")
        self.assertFalse(r["fit_performed"])
        self.assertNotIn("scenarios", r)


if __name__ == "__main__":
    unittest.main()
