"""Verify fold-local specialist behavior and the exact replay stop."""

import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import stage13f_assignment_audit as baseline
import stage13i_gemini_core as stage


class GeminiSpecialistChecks(unittest.TestCase):
    def setUp(self):
        self.y = np.tile(np.arange(stage.K), 4)
        self.folds = np.repeat(np.array([1, 2]), 2 * stage.K)
        self.prompts = np.repeat(np.array(["opaque_a", "opaque_b", "opaque_c", "opaque_d"]), stage.K)
        self.clusters = np.arange(len(self.y))
        self.clusters[:stage.K] = 0
        self.features = np.zeros((len(self.y), 2048), dtype=np.float64)
        self.features[np.arange(len(self.y)), self.y] = 1
        self.features[23, 5] = 0
        self.features[23, 6] = 1  # A correct baseline decision that the specialist breaks.
        predicted = self.y.copy()
        predicted[[24, 43]] = [7, 8]  # Two correctable within-family errors.
        predicted[26] = 10  # The specialist cannot recover a Gemini predicted outside Gemini.
        predicted[28] = 5  # A non-Gemini false positive still goes through the prediction gate.
        p = np.full((len(self.y), stage.K), .001, dtype=np.float64)
        p[np.arange(len(self.y)), predicted] = .95
        p /= p.sum(axis=1, keepdims=True)
        self.evidence = np.log(p + 1e-12)
        self.digest = stage.reference.digest_json(stage.reference.confusion(self.y, predicted).tolist())

    def run_compare(self, digest):
        return stage.compare(self.features, self.evidence, self.clusters, self.folds, self.prompts, self.y,
                             baseline.CLASSES, expected_baseline_confusion_sha256=digest)

    def test_real_readout_fixes_and_breaks_only_predicted_family(self):
        report = self.run_compare(self.digest)
        self.assertEqual(report["baseline_confusion_sha256"], self.digest)
        self.assertEqual((report["eligible_rows"], report["eligible_true_gemini_rows"],
                          report["eligible_true_non_gemini_rows"]), (12, 11, 1))
        self.assertEqual((report["corrected"], report["broken"]), (2, 1))
        self.assertEqual((report["eligible_baseline_correct"], report["eligible_candidate_correct"]), (9, 10))
        self.assertAlmostEqual(report["delta_pp"], 100 / len(self.y))
        self.assertEqual(report["baseline_family_confusion_counts"], report["candidate_family_confusion_counts"])
        self.assertEqual(report["candidate_confusion_counts"][8][10], 1)
        self.assertTrue(all(f["fit_class_counts"] == [2] * 4 and f["fit_prompts"] == 2 and f["prompt_overlap"] == 0
                            for f in report["folds"]))
        self.assertNotIn("opaque_", json.dumps(report, allow_nan=False))
        self.assertFalse(report["row_level_outputs"])

    def test_bad_anchor_blocks_training(self):
        with patch.object(stage, "_fit_and_predict") as model:
            with self.assertRaisesRegex(stage.reference.GateError, "stage13f_on_confusion_replay"):
                self.run_compare("0" * 64)
            model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
