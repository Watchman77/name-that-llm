"""Targeted synthetic checks for the fixed external-panel comparison."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import stage13e_external_score as stage


def panel():
    y = np.tile(np.arange(stage.K), 60)
    prompts = np.repeat([f"prompt-{i:02d}" for i in range(60)], stage.K)
    blocks = np.repeat([1, 2], 30 * stage.K)
    return y, prompts, blocks


class ExternalPanelChecks(unittest.TestCase):
    def test_reject_repeated_class_and_prompt_mixed_between_blocks(self):
        y, prompts, blocks = panel()
        bad_y = y.copy()
        bad_y[0] = bad_y[1]
        with self.assertRaisesRegex(stage.ref.GateError, "prompt_missing_or_repeated_class"):
            stage._holdout_layout(bad_y, prompts, blocks)
        crossed = blocks.copy()
        crossed[[0, 30 * stage.K]] = [2, 1]
        with self.assertRaisesRegex(stage.ref.GateError, "holdout_prompt_crosses_block"):
            stage._holdout_layout(y, prompts, crossed)

    def test_known_prompt_paired_gains_and_identical_control_fail(self):
        y, prompts, blocks = panel()
        y, masks = stage._holdout_layout(y, prompts, blocks)
        ids = np.arange(len(y))
        good = np.full((len(y), stage.K), 0.10 / (stage.K - 1))
        good[ids, y] = 0.90

        def with_swaps(p, model_pairs):
            out = p.copy()
            for first in model_pairs:
                for cls, swapped in [(first, first + 1), (first + 1, first)]:
                    mask = masks[0] & (y == cls)
                    out[mask] = 0.45 / (stage.K - 2)
                    out[mask, cls], out[mask, swapped] = 0.26, 0.29
            return out

        glm = with_swaps(good, [0, 2])
        deberta = with_swaps(good, [2])
        result = stage._compare(glm, deberta, good, y, prompts, masks)
        self.assertTrue(result["passes_external_screen"])
        self.assertAlmostEqual(result["glm_ba"], 16 / 18)
        self.assertAlmostEqual(result["pair_ba"], 17 / 18)
        self.assertEqual(result["triple_ba"], 1)
        for contrast in ("pair_vs_glm", "triple_vs_pair"):
            r = result[contrast]
            self.assertEqual((r["delta_pp_exact"], r["corrected"], r["broken"]), ("50/9", 60, 0))
            np.testing.assert_allclose(r["conditional_delta_ci95_pp"], [50 / 9, 50 / 9])
        weak_control = stage._compare(glm, glm, good, y, prompts, masks)
        self.assertFalse(weak_control["stronger_comparator_admitted"])
        self.assertTrue(weak_control["candidate_marginal_pass"])
        self.assertFalse(weak_control["passes_external_screen"])


if __name__ == "__main__":
    unittest.main()
