"""Targeted synthetic checks for the Stage 13D experiment contract."""
from contextlib import ExitStack
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import stage13d_broader_complement as stage


class BroaderComplementChecks(unittest.TestCase):
    def test_official_bytes_are_checked_before_csv_parsing(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "train.csv"
            path.write_bytes(b"wrong")
            with patch.object(stage.pd, "read_csv") as read:
                with self.assertRaisesRegex(stage.reference.GateError, "stage13d_official_train_hash"):
                    stage.load_texts(path, None, None, None, None)
            read.assert_not_called()

    def test_train_requires_exact_id_prompt_model_order(self):
        ids = np.array(["id0", "id1"])
        prompts = np.array(["groupA", "groupB"])
        models = np.array(["m0", "m1"])
        y = np.arange(2, dtype=np.int64)
        texts = np.array(["Case-sensitive text, first.", "Second text.\nNew line."])
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "train.csv"
            pd.DataFrame(dict(id=ids, text=texts, prompt_id=prompts, model=models)).to_csv(path, index=False)
            with ExitStack() as stack:
                stack.enter_context(patch.object(stage, "ROWS", 2))
                stack.enter_context(patch.object(stage, "TRAIN_SHA", hashlib.sha256(path.read_bytes()).hexdigest()))
                stack.enter_context(patch.object(stage, "TEXT_SHA", stage.reference.digest_json(
                    [hashlib.sha256(x.encode()).hexdigest() for x in texts])))
                np.testing.assert_array_equal(stage.load_texts(path, ids, prompts, y, models), texts)
                for axes, error in (([ids[::-1], prompts, y, models], "stage13d_train_id_axis"),
                                    ([ids, prompts[::-1], y, models], "stage13d_train_prompt_axis"),
                                    ([ids, prompts, y[::-1], models], "stage13d_train_model_axis")):
                    with self.subTest(error=error), self.assertRaisesRegex(stage.reference.GateError, error):
                        stage.load_texts(path, *axes)

    def test_char_vocabulary_and_idf_see_only_training_prompts(self):
        y = np.tile(np.arange(stage.K, dtype=np.int64), 4)
        texts = np.array([f"Class {'abcdefghjklmnopqrs'[k]} marker. Group {group}. Common clause."
                          for group in range(4) for k in range(stage.K)])
        val0 = np.arange(len(y)) < len(y) // 2
        masks = [val0, ~val0]
        recorded = []
        base = stage.TfidfVectorizer

        class RecordingVectorizer(base):
            def fit_transform(self, docs, y=None):
                recorded.append(("fit", docs.copy()))
                return super().fit_transform(docs, y)

            def transform(self, docs):
                recorded.append(("score", docs.copy()))
                return super().transform(docs)

        progress = []
        with patch.object(stage, "ROWS", len(y)), patch.object(stage, "FOLDS", 2), \
                patch.object(stage, "TfidfVectorizer", RecordingVectorizer):
            probabilities, details, _ = stage.fit_char_oof(texts, y, masks, on_fold=progress.append)
        self.assertEqual([k for k, _ in recorded], ["fit", "score", "fit", "score"])
        for j, mask in enumerate(masks):
            np.testing.assert_array_equal(recorded[2*j][1], texts[~mask])
            np.testing.assert_array_equal(recorded[2*j+1][1], texts[mask])
        self.assertEqual(len(details), 2)
        self.assertEqual([p["fold"] for p in progress], [1, 2])
        self.assertTrue(all(set(p) == {"fold", "feature_count", "seconds"} for p in progress))
        stage.reference.validate_p(probabilities)

    def test_exact_half_pp_and_paired_counts(self):
        y = np.tile(np.arange(stage.K, dtype=np.int64), 100)
        baseline = y.copy()
        bad = np.flatnonzero(y == 0)[:9]
        baseline[bad] = 1
        supports = np.full((5, stage.K), 100, dtype=np.int64)
        delta = np.zeros_like(supports)
        delta[:, 0] = 9
        result = stage.paired_summary(y, baseline, y, supports, delta)
        self.assertEqual(result["delta_pp_exact"], "1/2")
        self.assertEqual((result["corrected"], result["broken"], result["changed"]), (9, 0, 9))
        self.assertTrue(result["gain_at_least_half_pp"] and result["ci_lower_positive"])

    def test_identical_arms_are_not_admitted(self):
        prompts = np.repeat([f"p{k:03d}" for k in range(150)], stage.K)[1:]
        y = np.tile(np.arange(stage.K, dtype=np.int64), 150)[1:]
        fold = np.repeat([5] + [1, 2, 3, 4, 5] * 29 + [1, 2, 3, 4], stage.K)[1:]
        masks = stage.prior.fold_masks(prompts, y, fold)
        p = np.full((stage.ROWS, stage.K), 0.1 / (stage.K - 1))
        p[np.arange(len(y)), y] = 0.9
        glm_pred, residuals = stage.uniform_decisions(p, masks)
        result = stage.compare_arms(p, p, p, prompts, y, masks, glm_pred, residuals, [dict() for _ in masks])
        self.assertEqual(result["glm_control_ba"], 1.0)
        self.assertEqual(result["delta_pp"], 0.0)
        self.assertEqual(result["broader_delta_pp"], 0.0)
        self.assertFalse(result["broader_control_admitted"] or result["passes_proxy_screen"])


if __name__ == "__main__":
    unittest.main()
