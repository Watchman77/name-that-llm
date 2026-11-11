"""Small synthetic checks for leakage, real saved bytes and fail-closed gates."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import stage13_glm_readout as readout


class GlmReadoutChecks(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(413)
        labels = np.tile(np.arange(3), 6)
        groups = np.repeat([f"synthetic-{i}" for i in range(6)], 3)
        folds = np.repeat([1, 2, 3, 1, 2, 3], 3)
        features = np.ascontiguousarray(rng.normal(size=(18, 8)))
        features[:, 0] += labels * 2
        return features, labels, groups, folds, ("a", "b", "c")

    def test_heldout_labels_and_peer_features_do_not_affect_other_predictions(self):
        x, y, groups, folds, classes = self.fixture()
        p, summaries, _, _ = readout.fit_grouped(x, y, groups, folds, classes, 3)
        mask = folds == 1
        x_changed, y_changed = x.copy(), y.copy()
        x_changed[0] += 40
        y_changed[mask] = (y_changed[mask] + 1) % 3
        changed, _, _, _ = readout.fit_grouped(x_changed, y_changed, groups, folds, classes, 3)
        unaffected = mask.copy()
        unaffected[0] = False
        np.testing.assert_allclose(p[unaffected], changed[unaffected], atol=1e-14, rtol=0)
        self.assertGreater(np.linalg.norm(p[0] - changed[0]), 1e-4)
        self.assertEqual(sum(f["valid_rows"] for f in summaries), len(y))
        self.assertTrue(all(f["prompt_overlap"] == 0 for f in summaries))

    def test_split_prompt_is_rejected(self):
        x, y, groups, folds, classes = self.fixture()
        folds[0] = 2
        with self.assertRaisesRegex(readout.admission.AdmissionFailure, "prompt_crosses_folds"):
            readout.fit_grouped(x, y, groups, folds, classes, 3)

    def test_cache_round_trip_and_count_based_balanced_accuracy(self):
        targets = np.asarray([0, 0, 0, 1, 2], dtype=np.int64)
        probabilities = np.full((5, 3), 0.1)
        probabilities[np.arange(5), [0, 1, 0, 1, 1]] = 0.8
        score, counts, supports, recalls = readout.metrics(targets, probabilities, 3)
        self.assertAlmostEqual(score, (2 / 3 + 1) / 3)
        np.testing.assert_array_equal(counts, [[2, 1, 0], [0, 1, 0], [0, 1, 0]])
        np.testing.assert_array_equal(supports, [3, 1, 1])
        np.testing.assert_allclose(recalls, [2 / 3, 1, 0])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "heldout_probabilities.npz"
            digest = readout.save_private_cache(path, ids=np.asarray(list("abcde")),
                                                groups=np.asarray(["p"] * 5), targets=targets,
                                                folds=np.ones(5, dtype=np.int64), classes=("a", "b", "c"),
                                                probabilities=probabilities)
            self.assertEqual(digest, readout.admission.sha256(path))
            with np.load(path, allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["probabilities"], probabilities)
                np.testing.assert_array_equal(saved["targets"], targets)

    def test_missing_production_mounts_never_fit(self):
        with tempfile.TemporaryDirectory() as temp:
            report = readout.run(Path(temp), Path(temp) / "out")
        self.assertEqual(report["failure_code"], "real_input_admission_failed")
        self.assertFalse(report["fit_attempted"])
        self.assertFalse(report["heldout_predictions_produced"])
        self.assertNotIn("pooled_balanced_accuracy", report)


if __name__ == "__main__":
    unittest.main()
