"""Read-only output checks for the frozen public Stage 10 inference recipe.

Only the aggregate validation report is written. No model, prediction, submission
or parameter is changed. NumPy archives are always opened with allow_pickle=False.
An upstream exact-public-fixture stub is explicitly format-only, never proof that
weights loaded or full inference succeeded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

LABELS = [
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
    "claude-fable-5", "qwen3.8-max", "glm-5.3", "glm-5.3-flash", "kimi-k3",
]
COMP_LABELS = sorted(LABELS)
MAIN_REQUIRED = {"svc", "lora_qwen3_14b", "lora_qwen25_14b", "deberta"}
COMP_REQUIRED = {
    "stylegram", "stylometry_lgb", "debv3l", "electl", "desk", "lora24_public",
    "lora24_seed7", "lora24_claude", "lora20_q25_instyle", "tokattn_q35",
    "tokattn_deb", "ultrabert",
}
K = len(LABELS)
ATOL, RTOL = 1e-6, 1e-5


class OutputValidationError(ValueError):
    """An output contract failed; messages contain no row identifiers or passages."""


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def softmax(logits):
    logits = np.asarray(logits)
    values = np.exp(logits - logits.max(1, keepdims=True))
    return values / values.sum(1, keepdims=True)


def published_sinkhorn(probabilities):
    """The published 300-iteration balancing arithmetic, with no tuned settings."""
    probabilities = np.asarray(probabilities)
    scale = np.ones(K)
    target = np.full(K, len(probabilities) / K)
    with np.errstate(divide="raise", invalid="raise", over="raise"):
        for _ in range(300):
            rows = probabilities * scale
            rows /= rows.sum(1, keepdims=True)
            scale *= target / rows.sum(0)
        rows = probabilities * scale
        return rows / rows.sum(1, keepdims=True)


def validate_output(work_dir, test_path, namespace=None, public_preview=None,
                    parquet_reader=None, report_path=None):
    """Validate actual saved output artifacts and return/write aggregate evidence.

    `namespace` supplies only the upstream PUBLIC_PREVIEW mode flag. No model
    arrays or success flags are accepted from globals. Production callers leave
    parquet_reader unset so saved Parquet bytes are read through pandas.
    """
    work, test_path = Path(work_dir), Path(test_path)
    report_path = Path(report_path) if report_path else work / "stage10_output_validation.json"
    report = {"stage": "10", "status": "VALIDATING", "checks": [],
              "full_actual_model_inference_verified": False,
              "hidden_runtime_verified": False, "accuracy_validation": False,
              "original_ids_text_or_probability_rows_exported": False,
              "parquet_readback": "real pandas reader" if parquet_reader is None else "injected reader; not a real parquet-engine check"}
    checked = report["checks"]

    def require(value, name):
        ok = bool(value)
        checked.append({"name": name, "pass": ok})
        if not ok:
            raise OutputValidationError(name)

    def save_report():
        report["check_count"] = len(checked)
        report["all_checks_pass"] = all(item["pass"] for item in checked)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    def load_json(name):
        path = work / name
        require(path.is_file(), name + ":exists")
        try:
            value = json.loads(path.read_text())
        except Exception as error:
            raise OutputValidationError(name + ":valid_json") from error
        require(isinstance(value, dict), name + ":object")
        return value

    def read_npz(name, keys):
        path = work / name
        require(path.is_file(), name + ":exists")
        try:
            with np.load(path, allow_pickle=False) as archive:
                require(set(keys).issubset(archive.files), name + ":required_keys")
                # Only declared arrays are accessed. Extra upstream metadata may
                # contain an object array; it is never deserialized or required.
                return {key: np.array(archive[key], copy=True) for key in keys}
        except OutputValidationError:
            raise
        except Exception as error:
            raise OutputValidationError(name + ":safe_numpy_read") from error

    def real_matrix(value, name, rows=None):
        array = np.asarray(value)
        require(array.shape == (n if rows is None else rows, K), name + ":shape")
        require(np.issubdtype(array.dtype, np.number) and not np.iscomplexobj(array), name + ":real_numeric")
        require(np.isfinite(array).all(), name + ":finite")
        return array

    def probability(value, name, rows=None):
        array = real_matrix(value, name, rows)
        require((array >= 0).all() and (array <= 1).all(), name + ":bounded")
        require(np.allclose(array.sum(1), 1, atol=ATOL, rtol=RTOL), name + ":row_sums")
        return array

    def labels_order(value, expected, name):
        array = np.asarray(value)
        require(array.ndim == 1 and array.dtype.kind in "US", name + ":safe_string_columns")
        require(array.tolist() == expected, name + ":canonical_column_order")

    def identities(value, name):
        array = np.asarray(value)
        require(array.ndim == 1 and array.dtype.kind in "US", name + ":safe_string_ids")
        values = array.tolist()
        require(len(values) == n and len(set(values)) == n and all(values), name + ":unique_complete_ids")
        require(values == expected_ids, name + ":exact_id_order")

    def close(actual, expected, name):
        require(np.shape(actual) == np.shape(expected), name + ":same_shape")
        require(np.allclose(actual, expected, atol=ATOL, rtol=RTOL), name + ":numerical_parity")

    def member_archive(name, expected_order, required):
        path = work / name
        require(path.is_file(), name + ":exists")
        try:
            with np.load(path, allow_pickle=False) as archive:
                keys = set(archive.files)
                require({"ids", "labels"}.issubset(keys), name + ":identity_keys")
                identities(archive["ids"], name)
                labels_order(archive["labels"], expected_order, name)
                members = keys - {"ids", "labels"}
                require(required.issubset(members), name + ":mandatory_members")
                total = np.zeros((n, K), dtype=np.float64)
                for key in sorted(members):
                    total += real_matrix(archive[key], name + ":member:" + key)
                return members, total
        except OutputValidationError:
            raise
        except Exception as error:
            raise OutputValidationError(name + ":safe_numpy_read") from error

    try:
        require(test_path.is_file(), "official_test:exists")
        test = pd.read_csv(test_path, dtype=str, keep_default_na=False)
        require(list(test.columns) == ["id", "text"], "official_test:columns")
        expected_ids = test.id.tolist()
        n = len(expected_ids)
        require(n > 0 and test.id.is_unique and all(expected_ids), "official_test:unique_nonempty_ids")
        report["rows"] = n
        flag = None if namespace is None else namespace.get("PUBLIC_PREVIEW")
        if public_preview is not None:
            require(isinstance(public_preview, bool), "preview_mode:explicit_boolean")
            require(flag is None or flag is public_preview, "preview_mode:no_conflict")
            flag = public_preview
        require(isinstance(flag, bool), "preview_mode:known_boolean")
        report["source_public_preview_flag"] = flag
        submission_path = work / "submission.parquet"
        require(submission_path.is_file(), "submission:exists")
        try:
            submission = (pd.read_parquet if parquet_reader is None else parquet_reader)(submission_path)
        except Exception as error:
            raise OutputValidationError("submission:actual_readback") from error
        require(isinstance(submission, pd.DataFrame), "submission:dataframe")
        require(list(submission.columns) == ["id", "model"], "submission:columns")
        require(not submission.isna().any().any(), "submission:no_nulls")
        require(submission.id.map(lambda x: isinstance(x, str)).all(), "submission:string_ids")
        require(submission.model.map(lambda x: isinstance(x, str)).all(), "submission:string_labels")
        identities(submission.id.to_numpy(dtype=str), "submission")
        require(submission.model.isin(LABELS).all(), "submission:allowed_labels")
        report["submission_sha256"] = sha(submission_path)
        report["test_sha256"] = sha(test_path)

        if flag:
            require(n == 3, "format_only_preview:three_rows")
            require(submission.model.eq("claude-fable-5").all(), "format_only_preview:published_stub_labels")
            mode = load_json("submission.mode.json")
            require(mode.get("mode") == "exact_public_format_preview", "format_only_preview:mode_receipt")
            require(mode.get("model_weights_loaded") is False and mode.get("hidden_runtime_verified") is False,
                    "format_only_preview:no_model_inference_claim")
            report.update(status="FORMAT_ONLY_PREVIEW", inference_verification="NOT_PERFORMED_PUBLIC_FIXTURE_STUB")
            save_report()
            return report

        enhanced = read_npz("enhanced_probabilities.npz", ["ids", "labels", "main_raw", "complement", "before_atlas", "atlas", "atlas_rows", "before_balancing", "probabilities"])
        identities(enhanced["ids"], "enhanced")
        labels_order(enhanced["labels"], LABELS, "enhanced")
        main_raw = probability(enhanced["main_raw"], "enhanced:main_raw")
        comp_probability = probability(enhanced["complement"], "enhanced:complement")
        before_atlas = probability(enhanced["before_atlas"], "enhanced:before_atlas")
        before_balancing = probability(enhanced["before_balancing"], "enhanced:before_balancing")
        final = probability(enhanced["probabilities"], "enhanced:probabilities")
        atlas = real_matrix(enhanced["atlas"], "enhanced:atlas")
        rows = enhanced["atlas_rows"]
        require(rows.ndim == 1 and rows.dtype.kind in "iu", "atlas_rows:integer_vector")
        require(len(np.unique(rows)) == len(rows) and ((rows >= 0) & (rows < n)).all(), "atlas_rows:unique_in_range")
        margins = np.sort(before_atlas, axis=1)
        expected_rows = np.flatnonzero(margins[:, -1] - margins[:, -2] < .3) if n >= 50 else np.arange(n)
        require(np.array_equal(rows, expected_rows), "atlas_rows:published_margin_selector")
        probability(atlas[rows], "enhanced:atlas_selected", len(rows))
        outside = np.ones(n, dtype=bool)
        outside[rows] = False
        require(np.allclose(atlas[outside], 0, atol=ATOL, rtol=RTOL), "atlas_unselected:zeros")
        close(before_atlas, .85 * main_raw + .15 * comp_probability, "blend:main_complement")

        comp_meta = load_json("comp_meta.json")
        require(comp_meta.get("ids") == expected_ids, "comp_meta:exact_id_order")
        require(comp_meta.get("labels") == COMP_LABELS, "comp_meta:canonical_column_order")
        comp_path = work / "comp_logit.npy"
        require(comp_path.is_file(), "comp_logit:exists")
        try:
            comp_logits = real_matrix(np.load(comp_path, allow_pickle=False), "comp_logit")
        except OutputValidationError:
            raise
        except Exception as error:
            raise OutputValidationError("comp_logit:safe_numpy_read") from error
        comp_members, comp_sum = member_archive("comp_members.npz", COMP_LABELS, COMP_REQUIRED)
        close(comp_sum, comp_logits, "comp_members:sum_logits")
        permutation = [COMP_LABELS.index(label) for label in LABELS]
        close(comp_probability, softmax(comp_logits[:, permutation]), "complement:softmax_and_columns")

        main = read_npz("part_two_probabilities.npz", ["ids", "labels", "logits", "probabilities", "groups"])
        identities(main["ids"], "part_two")
        labels_order(main["labels"], LABELS, "part_two")
        main_logits = real_matrix(main["logits"], "part_two:logits")
        main_final = probability(main["probabilities"], "part_two:probabilities")
        require(main["groups"].shape == (n,) and main["groups"].dtype.kind in "iu", "part_two:group_vector")
        main_members, main_sum = member_archive("part_two_members.npz", LABELS, MAIN_REQUIRED)
        close(main_sum, main_logits, "part_two_members:sum_logits")
        close(main_raw, softmax(main_logits), "main_raw:softmax_logits")
        close(main_final, published_sinkhorn(main_raw) if n >= 50 else main_raw, "part_two:published_balancing")

        helper = read_npz("night24_helper_probabilities.npz", ["ids", "labels", "probabilities"])
        identities(helper["ids"], "night24_helper")
        labels_order(helper["labels"], COMP_LABELS, "night24_helper")
        helper_probability = probability(helper["probabilities"], "night24_helper:probabilities")[:, permutation]
        bank = read_npz("night24_blend_evidence.npz", ["ids", "labels", "unchanged_top1", "helper", "before_balancing"])
        identities(bank["ids"], "night24_blend")
        labels_order(bank["labels"], LABELS, "night24_blend")
        unchanged = probability(bank["unchanged_top1"], "night24_blend:unchanged_top1")
        bank_helper = probability(bank["helper"], "night24_blend:helper")
        bank_before = probability(bank["before_balancing"], "night24_blend:before_balancing")
        after_atlas = before_atlas.copy()
        after_atlas[rows] = .9 * after_atlas[rows] + .1 * atlas[rows]
        close(unchanged, after_atlas, "blend:selective_atlas")
        close(bank_helper, helper_probability, "night24_helper:canonical_reordering")
        reconstructed_before = .90 * after_atlas + .10 * helper_probability
        close(bank_before, reconstructed_before, "blend:night24")
        close(before_balancing, reconstructed_before, "enhanced:before_balancing")
        reconstructed_final = published_sinkhorn(reconstructed_before) if n >= 50 else reconstructed_before
        close(final, reconstructed_final, "blend:published_final_balancing")
        require(submission.model.tolist() == np.asarray(LABELS)[final.argmax(1)].tolist(), "submission:saved_probability_argmax")

        main_proof = load_json("part_two_verification.json")
        enhanced_proof = load_json("enhanced_verification.json")
        for name, proof in [("part_two_verification", main_proof), ("enhanced_verification", enhanced_proof)]:
            require(proof.get("rows") == n, name + ":rows")
            require(proof.get("full_actual_model_inference") is True, name + ":full_inference")
            require(proof.get("accuracy_validation") is False, name + ":no_accuracy_claim")
            require(proof.get("format_only") is (n == 3), name + ":published_format_flag")
        require(set(main_proof.get("members", [])) == main_members, "part_two_verification:actual_members")
        require(main_proof.get("test_sha256") == sha(test_path), "part_two_verification:actual_test_hash")
        train_path = test_path.parent / "train.csv"
        require(train_path.is_file(), "official_train:exists")
        require(main_proof.get("train_sha256") == sha(train_path), "part_two_verification:actual_train_hash")
        token_audit = main_proof.get("training_token_audit")
        require(isinstance(token_audit, list) and len(token_audit) == 2, "training_tokens:two_lora_receipts")
        require(all(isinstance(item, dict) and item.get("training_eos_id") == 151643 for item in token_audit), "training_tokens:published_eos_151643")
        require(set(enhanced_proof.get("required_complement_members", [])) == COMP_REQUIRED, "enhanced_verification:required_members")
        require(set(enhanced_proof.get("actual_complement_members", [])) == comp_members, "enhanced_verification:actual_members")
        require(enhanced_proof.get("all_required_components_verified") is True, "enhanced_verification:complete_flag")
        require(enhanced_proof.get("atlas_rows") == len(rows), "enhanced_verification:atlas_rows")
        require(enhanced_proof.get("night24_condition") == "synthetic4404", "enhanced_verification:helper_condition")
        for key, value in [("comp_weight", .15), ("atlas_weight", .10), ("night24_helper_weight", .10)]:
            require(enhanced_proof.get(key) == value, "enhanced_verification:" + key)
        report.update(status="FULL_MODEL_OUTPUT_VALIDATED", full_actual_model_inference_verified=True,
                      inference_verification="ACTUAL_SAVED_ARTIFACTS_AND_BLEND_PARITY", main_members=sorted(main_members),
                      complement_members=sorted(comp_members), atlas_rows=len(rows),
                      final_balancing_applied=n >= 50,
                      maximum_blend_reconstruction_error=float(np.abs(final - reconstructed_final).max()),
                      limitations=["This checks saved artifacts, not every internal activation or the safety of deserialized upstream code/weights.",
                                   "No development accuracy or hidden-test score follows from this validation.",
                                   "Input asset provenance, model hashes and runtime must be checked separately."])
        save_report()
        return report
    except Exception as error:
        report.update(status="FAIL", full_actual_model_inference_verified=False,
                      error_type=type(error).__name__,
                      error=str(error) if isinstance(error, OutputValidationError) else "Validation did not complete")
        save_report()
        if isinstance(error, OutputValidationError):
            raise
        raise OutputValidationError("Output validation did not complete") from error


def synthetic_tests():
    """Bounded artifact mutation checks; the Parquet engine boundary is mocked."""
    passed = []

    def fixture(root, n=3, atlas_selection="all", float32_helper=False):
        root.mkdir(parents=True, exist_ok=True)
        ids = np.asarray([f"synthetic_{i}" for i in range(n)])
        pd.DataFrame({"id": ids, "text": ["Synthetic fixture only."] * n}).to_csv(root / "test.csv", index=False)
        (root / "train.csv").write_text("id,prompt_id,text,model\nfixture,p0,Synthetic fixture only.,gpt-5.5\n")
        (root / "submission.parquet").write_bytes(b"MOCK_PARQUET_READBACK_BOUNDARY")
        rng = np.random.default_rng(42 + n)
        main_logits = rng.normal(0, .4, (n, K))
        comp_logits = rng.normal(0, .4, (n, K))
        if atlas_selection in {"none", "subset"}:
            if n < 50:
                raise ValueError("The published small-input path always selects every ATLAS row")
            # High-confidence rows must be excluded by the source's margin rule.
            # Remaining random diffuse rows form the selected subset.
            confident = np.arange(n) if atlas_selection == "none" else np.arange(n // 2)
            main_logits[confident] = -5
            main_logits[confident, 0] = 5
            comp_logits[confident] = -5
            comp_logits[confident, COMP_LABELS.index(LABELS[0])] = 5
        main_raw = softmax(main_logits)
        comp_p = softmax(comp_logits[:, [COMP_LABELS.index(label) for label in LABELS]])
        before_atlas = .85 * main_raw + .15 * comp_p
        margins = np.sort(before_atlas, axis=1)
        atlas_rows = np.flatnonzero(margins[:, -1] - margins[:, -2] < .3) if n >= 50 else np.arange(n)
        atlas = np.zeros((n, K), dtype=np.float32)
        atlas[atlas_rows] = softmax(rng.normal(0, .4, (len(atlas_rows), K)))
        after_atlas = before_atlas.copy()
        after_atlas[atlas_rows] = .9 * after_atlas[atlas_rows] + .1 * atlas[atlas_rows]
        helper_logits = rng.normal(0, .4, (n, K))
        if float32_helper:
            helper_logits = helper_logits.astype(np.float32)
        helper_comp = softmax(helper_logits)
        helper = helper_comp[:, [COMP_LABELS.index(label) for label in LABELS]]
        before = .90 * after_atlas + .10 * helper
        final = published_sinkhorn(before) if n >= 50 else before
        main_final = published_sinkhorn(main_raw) if n >= 50 else main_raw
        submission = pd.DataFrame({"id": ids, "model": np.asarray(LABELS)[final.argmax(1)]})
        np.savez_compressed(root / "enhanced_probabilities.npz", ids=ids, labels=np.asarray(LABELS), main_raw=main_raw,
                            complement=comp_p, before_atlas=before_atlas, atlas=atlas, atlas_rows=atlas_rows,
                            before_balancing=before, probabilities=final)
        np.savez_compressed(root / "part_two_probabilities.npz", ids=ids, labels=np.asarray(LABELS),
                            logits=main_logits, probabilities=main_final, groups=np.ones(n, dtype=int))
        np.savez_compressed(root / "part_two_members.npz", ids=ids, labels=np.asarray(LABELS),
                            **{key: main_logits / len(MAIN_REQUIRED) for key in MAIN_REQUIRED})
        np.savez_compressed(root / "comp_members.npz", ids=ids, labels=np.asarray(COMP_LABELS),
                            **{key: comp_logits / len(COMP_REQUIRED) for key in COMP_REQUIRED})
        np.save(root / "comp_logit.npy", comp_logits)
        (root / "comp_meta.json").write_text(json.dumps({"ids": ids.tolist(), "labels": COMP_LABELS}))
        np.savez_compressed(root / "night24_helper_probabilities.npz", ids=ids, labels=np.asarray(COMP_LABELS), probabilities=helper_comp)
        np.savez_compressed(root / "night24_blend_evidence.npz", ids=ids, labels=np.asarray(LABELS),
                            unchanged_top1=after_atlas, helper=helper, before_balancing=before)
        (root / "part_two_verification.json").write_text(json.dumps({"rows": n, "format_only": n == 3,
            "accuracy_validation": False, "full_actual_model_inference": True,
            "members": sorted(MAIN_REQUIRED), "test_sha256": sha(root / "test.csv"), "train_sha256": sha(root / "train.csv"),
            "training_token_audit": [{"training_eos_id": 151643}] * 2}))
        (root / "enhanced_verification.json").write_text(json.dumps({"rows": n, "format_only": n == 3,
            "accuracy_validation": False, "full_actual_model_inference": True, "required_complement_members": sorted(COMP_REQUIRED),
            "actual_complement_members": sorted(COMP_REQUIRED), "all_required_components_verified": True,
            "atlas_rows": len(atlas_rows), "night24_condition": "synthetic4404", "comp_weight": .15,
            "atlas_weight": .1, "night24_helper_weight": .1}))
        return submission

    def mutate_npz(path, key, fn):
        with np.load(path, allow_pickle=False) as data:
            values = {name: np.array(data[name], copy=True) for name in data.files}
        values[key] = fn(values[key])
        np.savez_compressed(path, **values)

    def remove_member(path, key):
        with np.load(path, allow_pickle=False) as data:
            values = {name: np.array(data[name], copy=True) for name in data.files if name != key}
        np.savez_compressed(path, **values)

    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        for name, n in [("valid_three_row_actual_inference", 3), ("valid_large_balanced_inference", 54)]:
            root = base / name
            sub = fixture(root, n)
            result = validate_output(root, root / "test.csv", public_preview=False, parquet_reader=lambda p, sub=sub: sub.copy())
            if result["status"] != "FULL_MODEL_OUTPUT_VALIDATED":
                raise AssertionError(name)
            passed.append(name)

        for name, selection, float32_helper in [
            ("valid_zero_atlas_rows", "none", False),
            ("valid_subset_atlas_rows", "subset", False),
            ("valid_subset_atlas_with_float32_helper", "subset", True),
        ]:
            root = base / name
            sub = fixture(root, 54, atlas_selection=selection, float32_helper=float32_helper)
            result = validate_output(root, root / "test.csv", public_preview=False,
                                     parquet_reader=lambda p, sub=sub: sub.copy())
            if result["status"] != "FULL_MODEL_OUTPUT_VALIDATED":
                raise AssertionError(name)
            if selection == "none" and result["atlas_rows"] != 0:
                raise AssertionError(name + ":fixture_did_not_exercise_zero_rows")
            if selection == "subset" and not 0 < result["atlas_rows"] < 54:
                raise AssertionError(name + ":fixture_did_not_exercise_subset_rows")
            if float32_helper:
                with np.load(root / "night24_helper_probabilities.npz", allow_pickle=False) as archive:
                    if archive["probabilities"].dtype != np.float32:
                        raise AssertionError(name + ":fixture_did_not_keep_float32")
            passed.append(name)

        cases = [
            ("wrong_probability_dimensions_rejected", lambda root, sub: mutate_npz(root / "enhanced_probabilities.npz", "probabilities", lambda x: x[:, :-1])),
            ("reordered_ids_rejected", lambda root, sub: mutate_npz(root / "enhanced_probabilities.npz", "ids", lambda x: x[::-1])),
            ("nonfinite_probability_rejected", lambda root, sub: mutate_npz(root / "enhanced_probabilities.npz", "probabilities", lambda x: np.full_like(x, np.nan))),
            ("missing_mandatory_member_rejected", lambda root, sub: remove_member(root / "comp_members.npz", "ultrabert")),
            ("bad_saved_parquet_label_rejected", lambda root, sub: sub.__setitem__("model", ["claude-fable-5"] * len(sub))),
            ("helper_label_order_rejected", lambda root, sub: mutate_npz(root / "night24_helper_probabilities.npz", "labels", lambda x: x[::-1])),
            ("wrong_blend_reconstruction_rejected", lambda root, sub: mutate_npz(root / "enhanced_probabilities.npz", "before_atlas", lambda x: x[:, ::-1])),
            ("nonfinite_component_logits_rejected", lambda root, sub: np.save(root / "comp_logit.npy", np.full((3, K), np.inf))),
            ("reordered_parquet_ids_rejected", lambda root, sub: sub.__setitem__("id", sub.id.iloc[::-1].to_numpy())),
        ]
        for name, mutate in cases:
            root = base / name
            sub = fixture(root)
            mutate(root, sub)
            try:
                validate_output(root, root / "test.csv", public_preview=False, parquet_reader=lambda p, sub=sub: sub.copy())
            except OutputValidationError:
                failed = json.loads((root / "stage10_output_validation.json").read_text())
                if failed["status"] != "FAIL":
                    raise AssertionError(name + ":not_fail_closed")
                passed.append(name)
            else:
                raise AssertionError(name + ":invalid_fixture_accepted")

        root = base / "format_only_preview"
        root.mkdir()
        ids = ["public_fixture_0", "public_fixture_1", "public_fixture_2"]
        pd.DataFrame({"id": ids, "text": ["Synthetic stub fixture."] * 3}).to_csv(root / "test.csv", index=False)
        (root / "submission.parquet").write_bytes(b"MOCK_PARQUET_READBACK_BOUNDARY")
        sub = pd.DataFrame({"id": ids, "model": ["claude-fable-5"] * 3})
        (root / "submission.mode.json").write_text(json.dumps({"mode": "exact_public_format_preview", "model_weights_loaded": False, "hidden_runtime_verified": False}))
        result = validate_output(root, root / "test.csv", public_preview=True, parquet_reader=lambda p: sub.copy())
        if result["status"] != "FORMAT_ONLY_PREVIEW" or result["full_actual_model_inference_verified"]:
            raise AssertionError("preview_stub_misrepresented")
        passed.append("stub_is_format_only_not_inference_verified")

    return {"status": "PASS", "checks": len(passed), "passed": passed,
            "actual_model_inference_executed": False, "actual_parquet_engine_executed": False,
            "parquet_boundary": "Explicit injected DataFrame reader; real parquet readback remains mandatory on Kaggle.",
            "limitations": ["Synthetic artifact contracts and published-mixture arithmetic only; no model, custom code or pickle execution.",
                            "The upstream exact-public-fixture fast path remains format-only."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path("/kaggle/working"))
    parser.add_argument("--test", type=Path)
    parser.add_argument("--format-only-preview", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(synthetic_tests(), indent=2))
    else:
        if args.test is None:
            parser.error("--test is required outside --self-test")
        print(json.dumps(validate_output(args.work_dir, args.test, public_preview=args.format_only_preview), indent=2))
