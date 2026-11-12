"""Verify residual reconciliation, rank ties, and the frozen-baseline gate."""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import stage13h_residual_core as stage


class ResidualAuditChecks(unittest.TestCase):
    def setUp(self):
        self.targets = np.tile(np.arange(stage.K), 2)
        self.folds = np.repeat(np.array([1, 2]), stage.K)
        self.prompts = np.repeat(np.array(["opaque_test_prompt_A", "opaque_test_prompt_B"]), stage.K)
        self.clusters = np.arange(len(self.targets))
        self.clusters[:12] = 0
        self.clusters[18:29] = 18  # The prior 13G extension must not be applied.
        p = np.full((len(self.targets), stage.K), .001, dtype=np.float64)
        p[np.arange(len(self.targets)), self.targets] = .95
        p[1, [0, 1]] = [.55, .43]  # assignment corrects the collision with row 0
        p[19, [0, 1]] = [.45, .45]  # argmax tie follows original class order
        p[20, [2, 3, 4]] = [.20, .50, .28]
        p[21, [3, 5, 6, 7]] = [.10, .45, .25, .18]
        p[[13, 31], 13] = .17
        p[[13, 31], 17] = .80
        p /= p.sum(axis=1, keepdims=True)
        self.evidence = np.log(p + 1e-12)

    def test_exact_13f_partition_and_rank_diagnostics(self):
        snapshot = self.evidence.copy()
        base = stage.assignment.compare(self.evidence, self.clusters, self.folds, self.prompts, self.targets)
        report = stage.diagnose(self.evidence, self.clusters, self.folds, self.prompts, self.targets,
                                expected_baseline_confusion_sha256=base["assignment_on_confusion_sha256"])
        self.assertEqual(report["baseline_confusion_counts"], base["assignment_on_confusion_counts"])
        self.assertEqual(report["baseline_correct"], 31)
        assigned, left = report["assigned"], report["unassigned"]
        self.assertEqual((assigned["rows"], assigned["pre_assignment_correct"], assigned["correct"]), (12, 11, 12))
        self.assertEqual((left["rows"], left["correct"], left["errors"]), (24, 19, 5))
        self.assertEqual((left["errors_with_true_top2"], left["errors_with_true_top3"]), (3, 4))
        self.assertEqual(left["error_evidence_rank_counts"][:5], [0, 3, 1, 1, 0])
        self.assertEqual(left["evidence_margin_bins"][0]["errors"], 1)
        self.assertEqual(left["top_directed_error_pairs"][0], dict(true_class_index=13, predicted_class_index=17,
                         errors=2, true_class_support=2, true_class_error_rate=1.0, folds_with_error=2))
        self.assertEqual(report["by_fold"][1]["assigned"]["balanced_accuracy"], None)
        self.assertEqual(assigned["represented_classes"], 12)
        self.assertEqual(assigned["class_recall"][12:], [None] * 6)
        self.assertFalse(next(x for x in report["by_cluster_size"] if x["cluster_size"] == 11)["assignment_applied"])
        np.testing.assert_array_equal(self.evidence, snapshot)
        self.assertNotIn("opaque_test_prompt", json.dumps(report, allow_nan=False))
        self.assertFalse(report["row_level_outputs"])

    def test_mismatched_frozen_control_blocks_diagnostics(self):
        with patch.object(stage, "_stratum") as report:
            with self.assertRaisesRegex(stage.reference.GateError, "stage13f_on_confusion_replay"):
                stage.diagnose(self.evidence, self.clusters, self.folds, self.prompts, self.targets,
                               expected_baseline_confusion_sha256="0" * 64)
            report.assert_not_called()


if __name__ == "__main__":
    unittest.main()
