"""Small synthetic contracts exercise admission failures without real private data."""
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd


MODULE = Path(__file__).resolve().parents[1] / "scripts" / "stage13a_cache_admission.py"
SPEC = importlib.util.spec_from_file_location("stage13a_cache_admission", MODULE)
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


class CacheAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        self.output = self.root / "output"
        rows = [{"id": f"private-row-{g}-{c}", "text": f"secret passage {g}/{c}",
                 "prompt_id": f"g{g}", "model": c}
                for g in range(5) for c in ("a", "b", "c")]
        self.train = pd.DataFrame(rows)
        self.train_path = self.inputs / "train.csv"
        self.train.to_csv(self.train_path, index=False)
        train_hash = audit.sha256(self.train_path)
        texts = np.asarray([hashlib.sha256(x.encode()).hexdigest() for x in self.train.text], dtype=str)
        self.arrays = {
            "features": np.arange(15 * 3 * 4, dtype=np.float32).reshape(15, 3, 4),
            "ids": self.train.id.to_numpy(dtype=str), "layers": np.array([8, 3, 24], dtype=np.int64),
            "text_sha256": texts, "token_lengths": np.full(15, 20, dtype=np.int64),
            "train_sha256": np.asarray(train_hash),
        }
        self.cache_path = self.inputs / "glm_features.npz"
        self.save_cache()
        # Equal three-row groups yield these explicit five GroupKFold partitions.
        expected_map = {f"g{i}": 5 - i for i in range(5)}
        self.contract = audit.Contract(
            train_hash, audit.sha256(self.cache_path), audit.json_hash(texts.tolist()),
            audit.json_hash(expected_map), rows=15, prompts=5, hidden=4, classes=("a", "b", "c"),
        )

    def save_cache(self):
        np.savez_compressed(self.cache_path, **self.arrays)

    def structural_contract(self):
        self.save_cache()
        return replace(self.contract, glm_sha256=audit.sha256(self.cache_path))

    def run_audit(self, contract=None):
        return audit.audit_inputs(self.inputs, self.output, contract=contract or self.contract)

    def assertFailure(self, report, expected):
        self.assertEqual(report["status"], "FAILED_INPUT_ADMISSION")
        self.assertEqual(report["failure_code"], expected)
        self.assertFalse(report["held_out_predictions_produced"])
        self.assertFalse(report["development_reference_admitted"])

    def test_valid_synthetic_gate_is_not_production_or_oof(self):
        report = self.run_audit()
        self.assertEqual(report["status"], "PASS_SYNTHETIC_FEATURE_ADMISSION_ONLY")
        self.assertFalse(report["contract_is_production"])
        self.assertFalse(report["fit_performed"])
        self.assertFalse(report["model_inference_performed"])
        self.assertFalse(report["full_s07_oof_admitted"])
        self.assertTrue(report["eligible_for_future_fold_local_readout_fitting"])
        self.assertEqual([x["valid_rows"] for x in report["fold_partitions"]], [3] * 5)
        self.assertEqual(json.loads((self.output / "stage13a_input_admission.json").read_text()), report)

    def test_report_exports_no_private_row_values(self):
        text = json.dumps(self.run_audit())
        self.assertNotIn("private-row-", text)
        self.assertNotIn("secret passage", text)
        self.assertNotIn(str(self.inputs), text)
        self.assertNotIn('"g0"', text)

    def test_runtime_versions_are_observed_aggregate_metadata(self):
        versions = self.run_audit()["runtime_versions"]
        self.assertEqual(set(versions), {"python", "numpy", "pandas", "scikit_learn"})
        self.assertTrue(all(isinstance(value, str) and value for value in versions.values()))

    def test_report_write_failure_is_sanitized_and_revokes_eligibility(self):
        with mock.patch.object(Path, "write_text", side_effect=OSError("private-path-canary")):
            report = self.run_audit()
        self.assertFailure(report, "report_write_failure")
        self.assertFalse(report["report_written"])
        self.assertFalse(report["eligible_for_future_fold_local_readout_fitting"])
        self.assertNotIn("private-path-canary", json.dumps(report))

    def test_cache_byte_corruption_fails_hash(self):
        with self.cache_path.open("ab") as stream:
            stream.write(b"corrupt")
        self.assertFailure(self.run_audit(), "glm_cache_sha256")

    def test_train_byte_change_fails_before_parsing(self):
        with self.train_path.open("a") as stream:
            stream.write("private malformed value\n")
        self.assertFailure(self.run_audit(), "official_train_sha256")

    def test_reordered_ids_are_rejected_even_with_matching_cache_hash(self):
        self.arrays["ids"] = self.arrays["ids"][::-1]
        self.assertFailure(self.run_audit(self.structural_contract()), "glm_exact_official_id_order")

    def test_reordered_text_digests_are_rejected(self):
        self.arrays["text_sha256"] = self.arrays["text_sha256"][::-1]
        self.assertFailure(self.run_audit(self.structural_contract()), "glm_exact_text_hash_order")

    def test_nonfinite_feature_is_rejected(self):
        self.arrays["features"][4, 2, 3] = np.nan
        self.assertFailure(self.run_audit(self.structural_contract()), "glm_features_finite")

    def test_wrong_feature_shape_rejected_from_header(self):
        self.arrays["features"] = np.zeros((15, 3, 5), dtype=np.float32)
        self.assertFailure(self.run_audit(self.structural_contract()), "glm_feature_shape_dtype")

    def test_object_ids_rejected_without_object_deserialization(self):
        self.arrays["ids"] = self.arrays["ids"].astype(object)
        report = self.run_audit(self.structural_contract())
        self.assertFailure(report, "npz_object_dtype_rejected")
        self.assertFalse(report["object_deserialization_performed"])

    def test_unknown_npz_member_rejected(self):
        self.arrays["unreviewed"] = np.zeros(2)
        self.assertFailure(self.run_audit(self.structural_contract()), "glm_cache_schema")

    def test_layer_reordering_rejected(self):
        self.arrays["layers"] = np.array([3, 8, 24], dtype=np.int64)
        self.assertFailure(self.run_audit(self.structural_contract()), "glm_exact_layer_order")

    def test_recorded_train_binding_rejected(self):
        self.arrays["train_sha256"] = np.asarray("0" * 64)
        self.assertFailure(self.run_audit(self.structural_contract()), "glm_bound_train_digest")

    def test_canonical_fold_hash_is_mandatory(self):
        bad = replace(self.contract, fold_map_sha256="0" * 64)
        self.assertFailure(self.run_audit(bad), "canonical_prompt_fold_mapping")

    def test_prompt_crossing_folds_is_rejected(self):
        groups = np.array(["private-prompt"] * 2 + ["x", "y", "z", "q"])
        folds = np.array([1, 2, 3, 4, 5, 5])
        with self.assertRaisesRegex(audit.AdmissionFailure, "prompt_crosses_folds"):
            audit.validate_fold_partition(groups, folds, 5)

    def test_duplicate_discovery_fails_closed(self):
        other = self.inputs / "other"
        other.mkdir()
        (other / "glm_features.npz").write_bytes(self.cache_path.read_bytes())
        self.assertFailure(self.run_audit(), "unique_glm_features_npz")

    def test_nonempty_output_is_preserved(self):
        self.output.mkdir()
        marker = self.output / "previous.json"
        marker.write_text("preserve")
        report = self.run_audit()
        self.assertFailure(report, "fresh_output_directory")
        self.assertFalse(report["report_written"])
        self.assertEqual(marker.read_text(), "preserve")
        self.assertEqual([x.name for x in self.output.iterdir()], ["previous.json"])


if __name__ == "__main__":
    unittest.main()
