"""Exercise the admission guard and the one-rule paired contrast."""
from fractions import Fraction
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import stage13g_small_group_core as stage


class SmallGroupChecks(unittest.TestCase):
    def setUp(self):
        self.targets = np.tile(np.arange(stage.K), 4)
        self.folds = np.ones(len(self.targets), dtype=np.int64)
        self.groups = np.repeat(np.array(["p1", "p2", "p3", "p4"]), stage.K)
        self.clusters = np.arange(len(self.targets))
        self.clusters[:12] = 0
        self.clusters[18:28] = 18
        self.clusters[36:47] = 36
        self.clusters[54:63] = 54
        p = np.full((len(self.targets), stage.K), 0.001, dtype=np.float64)
        p[np.arange(len(self.targets)), self.targets] = 0.95
        p[[1, 19, 37, 55], :2] = [0.70, 0.28]
        p[20, [2, 3]] = [0.05, 0.90]
        p[21, [2, 3]] = [0.40, 0.55]
        p /= p.sum(axis=1, keepdims=True)
        self.evidence = np.log(p + 1e-12)

    def test_preserves_baseline_and_changes_only_the_fixed_small_sizes(self):
        g = stage.extension_geometry(self.clusters, self.folds, self.groups, self.targets)
        self.assertTrue(g["passes_extension_geometry_gate"])
        self.assertEqual(g["prior_untouched_rows"], 60)
        self.assertEqual(g["extension"]["rows"], 21)
        self.assertEqual(g["untouched_by_size"]["9"]["rows"], 9)
        self.assertEqual(g["untouched_by_size"]["10"]["rows"], 10)
        self.assertEqual(g["untouched_by_size"]["11"]["rows"], 11)
        old = stage.prior.compare(self.evidence, self.clusters, self.folds, self.groups, self.targets)
        snapshot = self.evidence.copy()
        result = stage.compare(self.evidence, self.clusters, self.folds, self.groups, self.targets,
                               expected_baseline_confusion_sha256=old["assignment_on_confusion_sha256"])
        self.assertEqual(result["baseline_ba"], old["assignment_on_ba"])
        self.assertEqual((result["corrected"], result["broken"], result["changed"]), (2, 1, 3))
        self.assertEqual(result["delta_pp_exact"], str(Fraction(25, 18)))
        self.assertEqual(result["class_delta_pp"][1], 50)
        self.assertEqual(result["class_delta_pp"][3], -25)
        self.assertTrue(result["outside_extension_unchanged"])
        np.testing.assert_array_equal(self.evidence, snapshot)

    def test_geometry_fails_closed_before_assigning_any_labels(self):
        mixed = self.groups.copy()
        mixed[18] = "p1"
        repeated = self.targets.copy()
        repeated[37] = repeated[36]
        summary = stage.extension_geometry(self.clusters, self.folds, mixed, repeated)
        self.assertFalse(summary["passes_extension_geometry_gate"])
        self.assertEqual(summary["extension"]["mixed_prompt_clusters"], 1)
        self.assertEqual(summary["extension"]["duplicate_true_label_clusters"], 1)
        with patch.object(stage, "_assigned") as assign:
            with self.assertRaisesRegex(stage.reference.GateError, "extension_geometry_gate"):
                stage.compare(self.evidence, self.clusters, self.folds, mixed, repeated)
            assign.assert_not_called()

    def test_replay_failure_blocks_extension_assignment(self):
        with patch.object(stage, "_assigned", wraps=stage._assigned) as assign:
            with self.assertRaisesRegex(stage.reference.GateError, "stage13f_on_confusion_replay"):
                stage.compare(self.evidence, self.clusters, self.folds, self.groups, self.targets,
                              expected_baseline_confusion_sha256="0" * 64)
            self.assertEqual(assign.call_count, 1)


if __name__ == "__main__":
    unittest.main()
