"""Test private-cache admission, held-out isolation, and the paired score rule."""
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import stage13c_deberta_complement as stage


def npy(array):
    f = io.BytesIO()
    np.save(f, array)
    return f.getvalue()


def fixture():
    prompts = np.repeat([f"prompt{i:03d}" for i in range(150)], 18)[:-1]
    y = np.tile(np.arange(18), 150)[:-1]
    folds = np.repeat(np.arange(1, 6), 540)[:-1]
    return prompts, y, folds


class ComplementChecks(unittest.TestCase):
    def test_bounded_static_ids_preserve_order_without_unpickling(self):
        ids = np.asarray(["id-a", "id-b", "id-c"])
        for dtype in (str, object):
            blob = npy(ids.astype(dtype))
            with patch("pickle.load", side_effect=AssertionError("unsafe")):
                with patch("pickle.loads", side_effect=AssertionError("unsafe")):
                    self.assertEqual(stage.checked_ids(blob, ids),
                                     "numpy_nonobject" if dtype is str else
                                     "object_npy_static_unique_ID_literals_no_unpickling")
            with self.assertRaisesRegex(stage.reference.GateError, "deberta_ids_order_mismatch"):
                stage.checked_ids(blob, ids[::-1])
        with self.assertRaisesRegex(stage.reference.GateError, "deberta_ids_pickle_trailing_data"):
            stage.checked_ids(npy(ids.astype(object)) + b"x", ids)

    def test_hash_gate_stops_before_archive_parsing(self):
        with tempfile.TemporaryDirectory() as temp:
            bad = Path(temp) / "input.npz"
            bad.write_bytes(b"not the pinned cache")
            with patch.object(stage, "DEBERTA_BYTES", bad.stat().st_size):
                with patch.object(stage.zipfile, "ZipFile", side_effect=AssertionError("read before hash")):
                    with self.assertRaisesRegex(stage.reference.GateError, "deberta_cache_hash_mismatch"):
                        stage.load_features(bad, np.asarray(["unused"]))

    def test_group_boundary_violations_are_rejected(self):
        prompts, y, folds = fixture()
        masks = stage.fold_masks(prompts, y, folds)
        self.assertEqual([int(m.sum()) for m in masks], [540, 540, 540, 540, 539])
        folds[[0, 540]] = folds[[540, 0]]
        with self.assertRaisesRegex(stage.reference.GateError, "complement_fold_exclusion"):
            stage.fold_masks(prompts, y, folds)

    def test_fold_fit_and_predict_use_only_their_allowed_rows(self):
        prompts, y, folds = fixture()
        masks = stage.fold_masks(prompts, y, folds)
        x = np.arange(stage.ROWS, dtype=np.float64).reshape(-1, 1)
        seen = []

        class Scaler:
            def fit(self, values):
                self.fold = len(seen)
                np.testing.assert_array_equal(values, x[~masks[self.fold]])
                seen.append(self.fold)
                return self

            def transform(self, values):
                return values

        class Head:
            def __init__(self, **params):
                self.params = params
                self.classes_ = np.arange(stage.K)
                self.n_iter_ = np.array([1])

            def fit(self, values, targets):
                self.fold = seen[-1]
                np.testing.assert_array_equal(values, x[~masks[self.fold]])
                np.testing.assert_array_equal(targets, y[~masks[self.fold]])
                return self

            def predict_proba(self, values):
                np.testing.assert_array_equal(values, x[masks[self.fold]])
                ids = values[:, 0].astype(int)
                p = np.full((len(ids), stage.K), 0.01)
                p[np.arange(len(ids)), (ids + self.fold) % stage.K] = 0.83
                return p

        with patch.object(stage, "StandardScaler", Scaler), patch.object(stage, "LogisticRegression", Head):
            p, details, pools = stage.fit_oof(x, y, masks)
        self.assertEqual(seen, list(range(5)))
        self.assertEqual(len(details), 5)
        self.assertTrue(all(item["threads"] <= 2 for item in pools))
        np.testing.assert_array_equal(p.argmax(axis=1), (np.arange(stage.ROWS) + folds - 1) % stage.K)

    def test_identical_arms_have_zero_delta_and_cannot_pass(self):
        prompts, y, folds = fixture()
        masks = stage.fold_masks(prompts, y, folds)
        p = np.random.default_rng(17).uniform(0.1, 1, (stage.ROWS, stage.K))
        p /= p.sum(axis=1, keepdims=True)
        pred = np.full(stage.ROWS, -1, dtype=int)
        residuals = []
        for val in masks:
            scored, residual = stage.reference.evidence(p[val], True)
            pred[val] = scored.argmax(axis=1)
            residuals.append(residual)
        cm = stage.reference.confusion(y, pred)
        out = stage.paired_scores(p, p, prompts, y, masks, pred, cm, residuals, [{} for _ in masks])
        self.assertEqual(out["delta_pp"], 0)
        self.assertEqual(out["conditional_delta_ci95_pp"], [0, 0])
        self.assertEqual(out["changed_by_candidate"], 0)
        self.assertFalse(out["passes_proxy_screen"])


if __name__ == "__main__":
    unittest.main()
