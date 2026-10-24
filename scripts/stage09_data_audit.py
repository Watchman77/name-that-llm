"""Stage 09: fixed, CPU-only admission audit for the existing direct-OpenAI V4 pilot.

This helper never trains, calls an API, serializes original rows, or certifies
semantic prompt independence. Its returned data frames are private in-memory
inputs for the separate training runner. Public artifacts contain aggregates,
external-data identifiers and hashes only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import tempfile
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors

TRAIN_SHA = "40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707"
SOURCE_HASHES = {
    "passages_clean.csv": "23f90b6ba4cc83137ce8d48b3b3787439b44867532d63e2fe52f8a5ea49cd1e6",
    "prompts.csv": "7cc4651b362ceadff4a1fd88dd1f82bad0c84d10b621a236be72c27f1b8f3a52",
    "provenance.jsonl": "ddc69f2c221d921f928bf7785b5971d9763eaa136815c5968a83760994e405f3",
}
VALIDATION_IDS_SHA = "4a39f1ed94d312fc815fe69ee0d9a467fe8578628886433a92a4f72256bf4a0a"
KNOWN_PROMPT_SOURCE = "own (written for this dataset, host-style protocol)"
MODELS = {
    "gpt-5.5": "gpt-5.5-2026-04-23",
    "gpt-5.6-luna": "gpt-5.6-luna",
    "gpt-5.6-sol": "gpt-5.6-sol",
    "gpt-5.6-terra": "gpt-5.6-terra",
    "gpt-6-astra": "gpt-6-astra",
}
DERIVATIVE = re.compile(
    r"train[_\s-]*regen|reconstruct(?:ed|ion)?|official[_\s-]*prompt|"
    r"competition[_\s-]*prompt|derived[_\s-]*from[_\s-]*(?:official|train|competition)",
    re.I,
)


class AuditError(ValueError):
    """An input, source association, or frozen pilot contract failed."""


def require(condition, message):
    if not bool(condition):
        raise AuditError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha(value):
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def check_hash(path, expected):
    require(Path(path).is_file(), "Required frozen input is unavailable: " + Path(path).name)
    require(sha(path) == expected, "Frozen input checksum mismatch: " + Path(path).name)


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", str(text)).casefold().split())


def punctuation_normalize(text):
    return " ".join(re.findall(r"\w+", normalize(text)))


def grams(text):
    words = punctuation_normalize(text).split()
    return {" ".join(words[i:i + 8]) for i in range(len(words) - 7)}


def exclusion_blocks(prompt_rows):
    """Conservative metadata admission; one bad record excludes its whole block."""
    reasons = {}
    for row in prompt_rows.itertuples():
        bad = []
        source = str(row.prompt_source).strip()
        licence = str(row.prompt_licence).strip()
        if source != KNOWN_PROMPT_SOURCE:
            bad.append("unknown_or_unadmitted_prompt_origin")
        if licence != "CC BY 4.0":
            bad.append("unadmitted_prompt_licence")
        if str(row.split) != "new":
            bad.append("not_new_prompt_split")
        origin_fields = " ".join(str(getattr(row, k, "")) for k in
                                 ["prompt_source", "source_prompt_id", "prompt_id"])
        if DERIVATIVE.search(origin_fields):
            bad.append("explicit_official_reconstruction_or_derivative")
        if not str(row.prompt_body).strip():
            bad.append("empty_prompt_body")
        if bad:
            reasons.setdefault(str(row.prompt_id), set()).update(bad)
    return {key: sorted(value) for key, value in sorted(reasons.items())}


def verify_envelope(row, envelope):
    """Match the selected public CSV cell to its published first-party call envelope."""
    body = envelope["request"]["body"]
    response = envelope["response"]["json"]
    require(envelope["response"]["http_status"] == 200, "Selected call HTTP status failed")
    require("choices" in response and len(response["choices"]) > 0, "Wrong selected response schema")
    answer = response["choices"][0]["message"]["content"]
    request_prompt = body["messages"][0]["content"]
    checks = {
        "response": " ".join(answer.split()) == " ".join(row.text.split()),
        "requested_model": body["model"] == row.requested_model_id,
        "served_model": response["model"] == row.served_model_id,
        "actual_prompt": request_prompt == row.prompt_sent,
        "envelope_prompt": envelope["prompt_sent"] == row.prompt_sent,
        "envelope_prompt_id": str(envelope["prompt_id"]) == str(row.prompt_id),
        "label": envelope["label"] == row.model,
        "route": envelope["route"] == "openai",
        "split": envelope["split"] == "new",
        "call_id": envelope["call_id"] == row.call_id,
    }
    require(all(checks.values()), "Published call evidence mismatch: " + ",".join(k for k, v in checks.items() if not v))
    return answer == row.text


def frozen_split(train):
    """Return the predeclared fold 1, checked against actual saved Stage 07A membership."""
    require(len(train) == 2699 and train.prompt_id.nunique() == 150 and train.model.nunique() == 18,
            "Official training dimensions changed")
    require(train.id.is_unique, "Official row identity is not unique")
    tr, va = next(GroupKFold(5).split(train.text, train.model, train.prompt_id))
    val_sha = text_sha("\n".join(train.id.iloc[va]) + "\n")
    require(val_sha == VALIDATION_IDS_SHA, "Fold 1 no longer matches the frozen Stage 07A membership")
    require(set(train.prompt_id.iloc[tr]).isdisjoint(train.prompt_id.iloc[va]), "Official prompt leakage")
    labels = sorted(train.model.unique())
    require(set(train.model.iloc[tr]) == set(labels) == set(train.model.iloc[va]), "A pilot split lacks official labels")
    receipt = {
        "fold": 1,
        "algorithm": "sklearn.model_selection.GroupKFold(n_splits=5, shuffle=False)",
        "official_train_sha256": TRAIN_SHA,
        "validation_id_order_sha256": val_sha,
        "training_id_order_sha256": text_sha("\n".join(train.id.iloc[tr]) + "\n"),
        "training_rows": len(tr), "validation_rows": len(va),
        "training_prompts": int(train.prompt_id.iloc[tr].nunique()),
        "validation_prompts": int(train.prompt_id.iloc[va].nunique()),
        "labels": labels,
        "training_class_support": train.model.iloc[tr].value_counts().sort_index().to_dict(),
        "validation_class_support": train.model.iloc[va].value_counts().sort_index().to_dict(),
        "group_disjoint": True,
        "frozen_stage07a_validation_membership_matches": True,
        "evaluation_role": "reused development fold; no untouched-test claim",
        "original_ids_or_prompt_ids_published": False,
    }
    return tr, va, receipt


def scan_overlap(train, extra):
    overlap = {}
    normalized_official = [normalize(x) for x in train.text]
    normalized_extra = [normalize(x) for x in extra.text]
    suspect = np.zeros(len(extra), dtype=bool)
    for name, fn in [("exact", lambda x: x), ("unicode_case_whitespace", normalize),
                     ("punctuation", punctuation_normalize)]:
        mask = extra.text.map(fn).isin(set(train.text.map(fn))).to_numpy()
        overlap[name] = int(mask.sum())
        suspect |= mask
    official_grams = set().union(*(grams(t) for t in train.text))
    shared = np.array([len(grams(t) & official_grams) for t in extra.text])
    substring = np.array([any(a in b or b in a for a in normalized_official) for b in normalized_extra])
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2,
                               max_features=80000, sublinear_tf=True, dtype=np.float32)
    official_x = vectorizer.fit_transform(train.text)
    external_x = vectorizer.transform(extra.text)
    distance, _ = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute", n_jobs=2).fit(
        official_x).kneighbors(external_x)
    cosine = 1 - distance[:, 0]
    suspect |= substring | (shared >= 3) | (cosine >= .85)
    reasons = {str(prompt): ["response_overlap_or_similarity_threshold"] for prompt in
               sorted(extra.loc[suspect, "prompt_id"].unique())}
    result = {
        "comparison_scope": "all 2699 official responses, labels ignored, before either arm trains",
        "full_response_overlap_rows": overlap,
        "substring_overlap_rows": int(substring.sum()),
        "rows_with_at_least_three_distinct_shared_word_8grams": int((shared >= 3).sum()),
        "rows_with_character_cosine_at_least_0_85": int((cosine >= .85).sum()),
        "maximum_character_cosine": float(cosine.max()),
        "maximum_shared_word_8grams": int(shared.max()),
        "flagged_rows": int(suspect.sum()),
        "excluded_external_prompt_blocks": sorted(reasons),
        "screen_features_used_for_training": False,
        "semantic_prompt_independence_certified": False,
    }
    return reasons, result, shared, cosine, substring


def self_tests(train=None):
    """Small contract fixtures test rejection paths without fitting a classifier."""
    passed = []

    def rejected(name, fn):
        try:
            fn()
        except (AuditError, KeyError):
            passed.append(name)
        else:
            raise AssertionError("Failure fixture was accepted: " + name)

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "frozen.csv"
        path.write_text("unchanged", encoding="utf-8")
        expected = sha(path)
        check_hash(path, expected)
        path.write_text("tampered", encoding="utf-8")
        rejected("tampered_immutable_input_rejected", lambda: check_hash(path, expected))
        rejected("missing_immutable_input_rejected", lambda: check_hash(Path(td) / "missing", expected))
    fixture = pd.DataFrame([{"prompt_id": "external_fixture", "prompt_source": KNOWN_PROMPT_SOURCE,
                             "prompt_licence": "CC BY 4.0", "split": "new", "prompt_body": "Fixture topic"}])
    require(not exclusion_blocks(fixture), "Known-origin fixture failed")
    passed.append("known_origin_admitted")
    for name, column, value in [
        ("unknown_origin_whole_block_excluded", "prompt_source", ""),
        ("reconstructed_origin_whole_block_excluded", "prompt_source", "official prompt reconstruction"),
        ("train_regen_id_whole_block_excluded", "prompt_id", "train_regen_fixture"),
        ("unknown_licence_whole_block_excluded", "prompt_licence", "unknown"),
        ("non_new_prompt_whole_block_excluded", "split", "train_regen"),
        ("empty_prompt_whole_block_excluded", "prompt_body", ""),
    ]:
        altered = fixture.copy()
        altered.loc[0, column] = value
        require(len(exclusion_blocks(altered)) == 1, "Origin exclusion fixture failed: " + name)
        passed.append(name)
    row = pd.Series({"text": "Test answer", "requested_model_id": "test-model", "served_model_id": "test-model",
                     "prompt_sent": "Test prompt", "prompt_id": "external_fixture", "model": "test-label", "call_id": "test-call"})
    envelope = {"request": {"body": {"model": "test-model", "messages": [{"content": "Test prompt"}]}},
                "response": {"http_status": 200, "json": {"model": "test-model", "choices": [{"message": {"content": "Test answer"}}]}},
                "prompt_sent": "Test prompt", "prompt_id": "external_fixture", "label": "test-label", "route": "openai", "split": "new", "call_id": "test-call"}
    verify_envelope(row, envelope)
    passed.append("matching_call_envelope_admitted")
    changed = json.loads(json.dumps(envelope))
    changed["response"]["json"]["model"] = "wrong-model"
    rejected("wrong_served_identity_rejected", lambda: verify_envelope(row, changed))
    changed = json.loads(json.dumps(envelope))
    changed["request"]["body"]["messages"][0]["content"] = "Different prompt"
    rejected("wrong_actual_prompt_rejected", lambda: verify_envelope(row, changed))
    if train is not None:
        altered = train.copy()
        _, validation, _ = frozen_split(train)
        altered.loc[validation[0], "id"] = "stage09_split_tamper_fixture"
        rejected("changed_actual_validation_identity_rejected", lambda: frozen_split(altered))
    return {"status": "PASS", "checks": len(passed), "passed": passed,
            "classifier_fitted": False, "provider_api_called": False}


def audit(train_path, source, out, reference_oof=None, persist=True):
    """Return (private official frame, private admitted external frame, aggregate receipt)."""
    train_path, source, out = Path(train_path), Path(source), Path(out)
    check_hash(train_path, TRAIN_SHA)
    for name, expected in SOURCE_HASHES.items():
        check_hash(source / name, expected)
    train = pd.read_csv(train_path, keep_default_na=False, dtype=str)
    require(not train[["id", "prompt_id", "model", "text"]].eq("").any().any(), "Empty official input field")
    tr, va, split = frozen_split(train)
    reference_checked = False
    if reference_oof is not None:
        old = pd.read_csv(reference_oof, keep_default_na=False, dtype={"id": str, "prompt_id": str, "model": str})
        require(train[["id", "prompt_id", "model"]].equals(old[["id", "prompt_id", "model"]]), "Saved Stage 07A identity/order mismatch")
        for fold, (_, ix) in enumerate(GroupKFold(5).split(train.text, train.model, train.prompt_id), 1):
            require(old.iloc[ix].fold.eq(fold).all(), "Saved Stage 07A folds differ")
        reference_checked = True
    table = pd.read_csv(source / "passages_clean.csv", keep_default_na=False)
    bank = pd.read_csv(source / "prompts.csv", keep_default_na=False)
    selected = table[table.split.eq("new") & table.provider_route.eq("openai")].copy()
    require(len(selected) == 750 and selected.prompt_id.nunique() == 150, "Frozen external selection dimensions differ")
    require(selected.id.is_unique and selected.call_id.is_unique, "External cell/call identity duplicate")
    require(set(selected.model) == set(MODELS), "Selected external labels differ")
    require(selected.groupby("model").size().eq(150).all(), "Selected external class support differs")
    require(selected.groupby("prompt_id").model.nunique().eq(5).all(), "Incomplete five-label prompt block")
    require(not selected.duplicated(["prompt_id", "model"]).any(), "Repeated prompt/label cell")
    require(not selected.text.map(normalize).duplicated().any(), "Duplicated normalized external response")
    require(not train.text.map(normalize).duplicated().any(), "Duplicated normalized official response")
    require(selected.text.str.split().str.len().between(40, 250).all(), "External word bounds failed")
    require(selected.finish_reason_norm.eq("stop").all(), "External completion was not a clean stop")
    for flag in ["is_refusal", "is_truncated", "is_empty"]:
        require(not selected[flag].any(), "Selected external quality flag: " + flag)
    require(selected.passes_funnel.all() and selected.in_word_range.all(), "Selected external quality funnel failed")
    for model, identifier in MODELS.items():
        rows = selected[selected.model.eq(model)]
        require(rows.requested_model_id.eq(identifier).all() and rows.served_model_id.eq(identifier).all(), "Requested/served identity changed")
    require(bank.prompt_id.is_unique, "Duplicate prompt bank association")
    missing = set(selected.prompt_id) - set(bank.prompt_id)
    require(not missing, "Missing external prompt bank association")
    prompt_rows = selected[["prompt_id", "source_prompt_id"]].drop_duplicates().merge(
        bank[["prompt_id", "split", "prompt_source", "prompt_licence", "prompt_body"]],
        on="prompt_id", validate="many_to_one")
    origin_reasons = exclusion_blocks(prompt_rows)
    joined = selected.merge(bank[["prompt_id", "prompt_body", "prompt_source", "prompt_licence"]],
                            on="prompt_id", validate="many_to_one")
    body_match = np.array([bool(body.strip()) and body in sent for body, sent in zip(joined.prompt_body, joined.prompt_sent)])
    for prompt in joined.loc[~body_match, "prompt_id"].unique():
        origin_reasons.setdefault(str(prompt), []).append("actual_request_prompt_not_associated_with_prompt_bank_body")
    evidence = {}
    calls = set(selected.call_id)
    with (source / "provenance.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            envelope = json.loads(line)
            if envelope.get("call_id") in calls:
                require(envelope["call_id"] not in evidence, "Duplicated selected raw call evidence")
                evidence[envelope["call_id"]] = envelope
    require(set(evidence) == calls, "Selected raw call evidence is incomplete")
    exact_raw = 0
    for row in selected.itertuples():
        exact_raw += int(verify_envelope(row, evidence[row.call_id]))
    overlap_reasons, overlap, shared, cosine, substring = scan_overlap(train, joined)
    excluded = set(origin_reasons) | set(overlap_reasons)
    joined["excluded_prompt_block"] = joined.prompt_id.isin(excluded)
    manifest = joined[["id", "call_id", "prompt_id", "model", "requested_model_id", "served_model_id",
                       "prompt_source", "prompt_licence", "source_prompt_id", "excluded_prompt_block"]].copy()
    manifest["response_sha256"] = joined.text.map(text_sha)
    manifest["actual_request_prompt_sha256"] = joined.prompt_sent.map(text_sha)
    manifest["prompt_body_sha256"] = joined.prompt_body.map(text_sha)
    manifest["shared_official_word_8grams"] = shared
    manifest["nearest_official_character_cosine"] = cosine
    manifest["substring_overlap"] = substring
    manifest = manifest.sort_values(["prompt_id", "model", "id"]).reset_index(drop=True)
    manifest_bytes = manifest.to_csv(index=False, lineterminator="\n", float_format="%.9g").encode("utf-8")
    admitted = joined[~joined.excluded_prompt_block].sort_values(["prompt_id", "model"]).reset_index(drop=True)
    require(len(admitted) > 0, "No external prompt blocks were admitted")
    admitted_ids_sha = text_sha("\n".join(admitted.id) + "\n")
    fingerprint_inputs = {
        "official_train_sha256": TRAIN_SHA,
        "source_files_sha256": SOURCE_HASHES,
        "audit_source_sha256": sha(Path(__file__)),
        "validation_id_order_sha256": VALIDATION_IDS_SHA,
        "admitted_external_id_order_sha256": admitted_ids_sha,
        "admitted_rows": len(admitted),
        "admitted_prompt_blocks": int(admitted.prompt_id.nunique()),
    }
    admission_fingerprint = text_sha(json.dumps(fingerprint_inputs, sort_keys=True, separators=(",", ":")))
    # The initial fixed pilot is exactly the previously audited 750 rows. A new
    # exclusion changes the experimental input and must not silently launch it.
    pass_frozen = len(admitted) == 750 and not excluded
    receipt = {
        "stage": "09", "status": "PASS_FROZEN_PILOT" if pass_frozen else "BLOCKED_FROZEN_SELECTION_CHANGED",
        "dataset": "prvsiyan/prvsiyans-extended-name-that-llm-dataset", "dataset_version": 4,
        "selection": "passages_clean.csv; split=new; provider_route=openai; five fixed GPT labels",
        "official_train_sha256": TRAIN_SHA, "source_files_sha256": SOURCE_HASHES,
        "audit_source_sha256": sha(Path(__file__)),
        "admission_fingerprint": admission_fingerprint,
        "admission_fingerprint_inputs": fingerprint_inputs,
        "admitted_external_id_order_sha256": admitted_ids_sha,
        "external_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "selected_rows": len(selected), "selected_prompt_blocks": int(selected.prompt_id.nunique()),
        "admitted_rows": len(admitted), "admitted_prompt_blocks": int(admitted.prompt_id.nunique()),
        "external_class_support": admitted.model.value_counts().sort_index().to_dict(),
        "official_class_count": int(train.model.nunique()),
        "external_class_count": int(admitted.model.nunique()),
        "prompt_origin_declarations": prompt_rows.groupby(["prompt_source", "prompt_licence", "split"]).size().reset_index(name="prompt_blocks").to_dict("records"),
        "blank_source_prompt_id_rows": int(selected.source_prompt_id.eq("").sum()),
        "missing_prompt_bank_associations": len(missing),
        "request_prompts_containing_mapped_prompt_body": int(body_match.sum()),
        "selected_raw_call_envelopes_matched": len(evidence),
        "raw_response_text_exact_rows": exact_raw,
        "raw_response_comparison_policy": "whitespace normalization only; no text alteration for training",
        "origin_excluded_blocks": origin_reasons, "overlap": overlap,
        "exclusion_policy": "exclude every response in an external prompt block when any response or source association fails",
        "split": split, "saved_stage07a_all_five_folds_compared_in_this_check": reference_checked,
        "self_tests": self_tests(train),
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__},
        "limitations": [
            "Official prompt wording is not supplied. Guessed/reconstructed official topics and response overlap cannot certify semantic prompt independence.",
            "The actual mapped prompt bank declares own-written host-style prompts and CC BY 4.0; origin and call envelopes are publisher-supplied evidence, not independent provider attestation.",
            "source_prompt_id is blank for all selected rows; association is corroborated through prompt_id, mapped prompt_body and the actual request envelope.",
            "Admission is for this bounded private undistributed discriminative-classifier pilot; no generative fine-tuning, public trained weights or official deployment admission is inferred.",
            "Only five external GPT labels are covered. All eighteen official labels remain in training and development evaluation.",
            "The fixed official fold has been reused in earlier development work and is not a new untouched test.",
        ],
        "training_started": False, "provider_api_called": False, "paid_services_used": False,
        "original_text_ids_prompt_ids_or_predictions_published": False,
    }
    if persist:
        out.mkdir(parents=True, exist_ok=True)
        (out / "external_manifest.csv").write_bytes(manifest_bytes)
        (out / "split_receipt.json").write_text(json.dumps(split, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        (out / "DATA_AUDIT.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        lines = ["# Stage 09 data admission audit", "", "Status: **" + receipt["status"] + "**.", "",
                 f"The immutable V4 selection contains {len(selected):,} direct OpenAI rows in {selected.prompt_id.nunique()} five-label prompt blocks. "
                 f"{len(admitted):,} rows / {admitted.prompt_id.nunique()} blocks remain after whole-block source and response screens.", "",
                 f"All {len(evidence)} selected published call envelopes match response, requested/served identity, actual request prompt, route and label. "
                 f"All {int(body_match.sum())} actual prompts contain the mapped prompt-bank body. "
                 "All selected prompt-bank records declare `own (written for this dataset, host-style protocol)` and `CC BY 4.0`.", "",
                 f"Response screening found {overlap['flagged_rows']} thresholded rows and excluded {len(excluded)} prompt blocks. "
                 f"Maximum character cosine to any official response: {overlap['maximum_character_cosine']:.6f}. "
                 "Checks include exact/normalized/punctuation response equality, full-passage substring, at least three shared distinct word 8-grams and character cosine at least 0.85.", "",
                 "Fold 1 is frozen at 2,159 training responses/120 prompts and 540 validation responses/30 prompts, with all 18 labels in each. "
                 "Its ordered validation-ID hash matches the actual earlier Stage 07A fold. "
                 + ("All five memberships were additionally compared to the saved Stage 07A OOF file in this check." if reference_checked else "Membership is enforced by the predeclared hash without requiring a row-level OOF attachment."), "",
                 "**Admission is not semantic-independence certification.** Official prompt wording is unavailable. The own-written origin declaration and published call envelopes are evidence from the dataset publisher. "
                 "The blank source_prompt_id field is documented; actual prompt-body/request association is checked. Unknown origins and explicit official reconstructions/train_regen derivatives are rejected as whole external blocks. "
                 "The original fold remains reused development data.", "",
                 "This audit does not train a classifier or use an inference API. It admits a bounded private discriminative-classifier experiment; it grants no public trained-weight or generative-model-use approval. "
                 "Public artifacts contain aggregate results and external identifiers/hashes only.", "",
                 f"Self-test fixtures: {receipt['self_tests']['checks']}/{receipt['self_tests']['checks']} passed. "
                 "They exercise immutable-input tampering/missing files, unknown and reconstructed origins, train_regen, unknown licensing, invalid splits/empty prompts, and actual-call identity/prompt mismatches.", "",
                 "Corpus, source, manifest and split hashes are in DATA_AUDIT.json and split_receipt.json.", ""]
        (out / "DATA_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    return train, admitted, receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--reference-oof", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_tests(), indent=2))
    else:
        if not all([args.train, args.source, args.out]):
            parser.error("--train, --source and --out are required")
        _, _, result = audit(args.train, args.source, args.out, args.reference_oof)
        print(json.dumps({key: result[key] for key in ["status", "selected_rows", "admitted_rows", "admitted_prompt_blocks",
                         "selected_raw_call_envelopes_matched", "external_manifest_sha256", "self_tests"]}, indent=2))
        require(result["status"] == "PASS_FROZEN_PILOT", "Frozen experiment blocked; retain audit and amend before training")
