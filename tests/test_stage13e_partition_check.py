"""Check the new partition and the two paired decisions without production data."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import stage13e_partition_check as stage


def fixture():
    prompts = np.repeat([f"p{i:03d}" for i in range(stage.PROMPTS)], stage.K)
    labels = np.tile(np.arange(stage.K, dtype=np.int64), stage.PROMPTS)
    canonical = np.repeat(np.repeat(np.arange(1, 6), 30), stage.K)
    # Put the singleton missing row in alternate fold 2, independent of canonical placement.
    prompt_map = stage.rank_prompts(prompts)
    absent_prompt = next(key for key, fold in prompt_map.items() if fold == 2)
    present = (prompts != absent_prompt) | (labels != 0)
    return prompts[present], labels[present], canonical[present]


class AlternatePartitionChecks(unittest.TestCase):
    def test_fixed_partition_is_order_independent_and_allows_short_fold_anywhere(self):
        prompts, y, canonical = fixture()
        folds, masks, details, digest = stage.alternate_partition(prompts, y, canonical)
        self.assertEqual([d["valid_rows"] for d in details], [540, 539, 540, 540, 540])
        self.assertTrue(all(not set(prompts[m]).intersection(prompts[~m]) for m in masks))
        reversed_folds, _, _, reversed_digest = stage.alternate_partition(prompts[::-1], y[::-1], canonical[::-1])
        np.testing.assert_array_equal(folds, reversed_folds[::-1])
        self.assertEqual(digest, reversed_digest)
        with self.assertRaisesRegex(stage.reference.GateError, "partition_only_renumbered_canonical_folds"):
            stage.alternate_partition(prompts, y, 6 - folds)

    def test_paired_arms_share_control_and_require_both_gates(self):
        prompts, y, canonical = fixture()
        _, masks, details, _ = stage.alternate_partition(prompts, y, canonical)
        prompt_no = np.array([int(p[1:]) for p in prompts])
        first = (prompt_no % 5 == 0) & np.isin(y, [1, 2])
        second = (prompt_no % 5 == 1) & np.isin(y, [3, 4])

        def probabilities(wrong):
            scores = np.full((len(y), stage.K), 0.01 / (stage.K - 1))
            scores[np.arange(len(y)), y] = 0.99
            flipped = np.where(y < 3, 3 - y, 7 - y)
            scores[wrong] = 0.01 / (stage.K - 2)
            scores[np.flatnonzero(wrong), y[wrong]] = 0.48
            scores[np.flatnonzero(wrong), flipped[wrong]] = 0.51
            return scores

        g, deb, char = probabilities(first | second), probabilities(second), probabilities(np.zeros(len(y), dtype=bool))
        passed = stage.compare_arms(g, deb, char, prompts, y, masks, details)
        self.assertTrue(passed["comparator_admitted"] and passed["passes_split_stress_screen"])
        for key in ("C_minus_G", "D_minus_C"):
            self.assertEqual(passed[key]["delta_pp_exact"], "20/9")
            self.assertEqual((passed[key]["corrected"], passed[key]["broken"]), (60, 0))
        failed = stage.compare_arms(g, g, char, prompts, y, masks, details)
        self.assertFalse(failed["comparator_admitted"] or failed["passes_split_stress_screen"])
        self.assertEqual(failed["C_minus_G"]["delta_pp"], 0)
        self.assertEqual(failed["D_minus_C"]["corrected"], 120)


if __name__ == "__main__":
    unittest.main()
