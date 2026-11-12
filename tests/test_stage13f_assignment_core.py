"""Synthetic checks for fold isolation and the paired assignment contrast."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import stage13f_assignment_core as stage


class AssignmentCoreChecks(unittest.TestCase):
    def test_geometry_uses_each_folds_text_and_never_truth(self):
        n = 250
        texts = np.asarray(["acoustic cello bridge strings"] * 12
                           + [f"qpz{i}" for i in range(12, n)]
                           + ["acoustic cello bridge strings"] * 12
                           + [f"qpz{i}" for i in range(n + 12, 2 * n)])
        folds = np.repeat([1, 2], n)
        groups = np.array([f"prompt{i}" for i in range(2 * n)], dtype="U16")
        groups[:6], groups[6:12], groups[n:n + 12] = "left", "right", "pure"
        targets = np.tile(np.arange(stage.K), 28)[:2 * n]
        targets[:12] = np.tile(np.arange(6), 2)
        targets[n:n + 12] = np.arange(12)

        report, inferred = stage.geometry(texts, folds, groups, targets)
        self.assertEqual(report["eligible_clusters"], 2)
        self.assertEqual(report["eligible_rows"], 24)
        self.assertEqual(report["eligible_mixed_prompt_clusters"], 1)
        self.assertEqual(report["eligible_duplicate_true_label_excess_rows"], 6)
        self.assertEqual(report["eligible_row_weighted_prompt_purity"], 0.75)
        self.assertTrue(np.intersect1d(inferred[:n], inferred[n:]).size == 0)
        _, alternative = stage.geometry(texts, folds, np.roll(groups, 20), targets[::-1])
        np.testing.assert_array_equal(inferred, alternative)

    def test_assignment_only_changes_dense_groups_and_reconciles_pairs(self):
        targets = np.tile(np.arange(stage.K), 2)
        probabilities = np.full((len(targets), stage.K), 0.001, dtype=np.float64)
        probabilities[np.arange(len(targets)), targets] = 0.95
        probabilities[1, [0, 1]] = [0.70, 0.28]
        probabilities[2, [2, 3]] = [0.05, 0.90]
        probabilities[3, [2, 3]] = [0.40, 0.55]
        probabilities[19, [0, 1]] = [0.70, 0.28]
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        evidence = np.log(probabilities + 1e-12)
        frozen = evidence.copy()
        cluster_ids = np.arange(len(targets))
        cluster_ids[:12] = 0
        # Smaller groups with label collisions are intentionally untouched.
        cluster_ids[12:23] = 12
        folds = np.ones(len(targets), dtype=np.int64)
        groups = np.repeat(np.array(["p1", "p2"]), stage.K)

        report = stage.compare(evidence, cluster_ids, folds, groups, targets)
        self.assertEqual((report["eligible_clusters"], report["eligible_rows"]), (1, 12))
        self.assertEqual((report["corrected"], report["broken"], report["changed"]), (1, 1, 2))
        self.assertEqual(report["delta_pp_exact"], "0")
        self.assertEqual(report["class_delta_pp"][1], 50)
        self.assertEqual(report["class_delta_pp"][3], -50)
        self.assertTrue(report["outside_eligible_clusters_unchanged"])
        np.testing.assert_array_equal(evidence, frozen)

    def test_source_tie_order_cap_and_empty_vocabulary_fallback(self):
        inferred = stage.capped_agglo(np.ones((20, 20)), stage.CAP, stage.THRESHOLD)
        np.testing.assert_array_equal(inferred, [0] * 18 + [18] * 2)
        groups, fallback = stage.infer_groups(np.array(["the", "and", "was"]))
        self.assertTrue(fallback)
        np.testing.assert_array_equal(groups, np.arange(3))


if __name__ == "__main__":
    unittest.main()
