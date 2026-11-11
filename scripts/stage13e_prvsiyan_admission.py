"""Stage 13E outcome-blind check of a pinned public external corpus.

Read the exact official training file, never Kaggle test data. No scoring,
external calls, or raw content/row identifiers are written. A passing panel is
a public external proxy for the frozen Stage 13C/13D readouts only, not S07.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits

TRAIN_SHA256 = "40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707"
DATASET_SLUG = "prvsiyans-extended-name-that-llm-dataset"
SPLITS = ("new", "new2")
CLASSES = (
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5",
    "claude-sonnet-5", "gemini-3.1-pro-preview", "gemini-3.6-flash",
    "gemini-3.7-flash", "gemini-3.8-flash", "glm-5.3", "glm-5.3-flash",
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra",
    "gpt-6-astra", "kimi-k3", "qwen3.8-max",
)
FILES = ("passages_clean.csv", "prompts.csv", "provenance.jsonl")
# Unique prompt IDs from results/stage_07a/admitted_ids.csv at
# e6f83acbdc72746ffbdfeb931f91f095b65c928d. Stage 09 reused the same 750 calls.
PRIOR_PROMPTS = frozenset("""
n001 n002 n003 n004 n005 n007 n009 n011 n012 n015 n016 n018 n019 n020 n022 n023 n024 n025 n026 n028
n029 n030 n031 n032 n033 n034 n035 n038 n039 n040 n041 n043 n044 n045 n048 n049 n050 n051 n052 n053
n055 n056 n057 n058 n059 n060 n061 n062 n063 n065 n066 n067 n068 n069 n070 n072 n075 n076 n077 n078
n079 n080 n081 n082 n083 n084 n085 n088 n090 n092 n093 n096 n097 n098 n099 n102 n103 n104 n105 n106
n107 n108 n110 n111 n112 n113 n114 n115 n116 n117 n118 n120 n122 n123 n124 n125 n126 n128 n129 n130
n131 n132 n133 n134 n137 n138 n139 n140 n142 n143 n144 n146 n147 n148 n150 n151 n152 n153 n154 n155
n158 n159 n161 n162 n163 n164 n165 n166 n168 n170 n171 n172 n173 n175 n176 n177 n178 n179 n180 n181
n182 n183 n184 n187 n188 n189 n190 n191 n194 n195
""".split())
PRIOR_NEAR_PROMPT_EXCLUSIONS = frozenset()  # V4 manual top 30: no confirmed overlaps.
RANK_SALT = "Stage13E-prvsiyan-panel-v1-20261111"
BLOCK_PROMPTS = 30
MAX_PROMPTS = 90
DERIVATIVE = re.compile(r"reconstruct|train_regen|pjmathematician", re.I)
FINISH = {"stop", "end_turn", "stop_sequence"}
OPENROUTER = {
    "glm-5.3": ("z-ai/glm-5.3", {"zai", "zhipuai", "zaius", "zaieu"}),
    "glm-5.3-flash": ("z-ai/glm-5.3-flash", {"zai", "zhipuai", "zaius", "zaieu"}),
    "kimi-k3": ("moonshotai/kimi-k3", {"moonshotai", "moonshot"}),
    "qwen3.8-max": ("qwen/qwen3.8-max-0902", {"alibaba", "alibabacloud", "qwen"}),
}
CELL_COLUMNS = {
    "id", "split", "prompt_id", "model", "text", "call_id", "provider_route",
    "requested_model_id", "served_model_id", "prompt_sent", "finish_reason_norm",
}
PROMPT_COLUMNS = {"prompt_id", "split", "prompt_body", "prompt_source", "prompt_licence"}


class AuditError(ValueError):
    """Known source/structure failure; messages are fixed non-data codes."""


def need(condition, code):
    if not condition:
        raise AuditError(code)


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def text_hash(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def words(text):
    return re.findall(r"\w+", normalize(text))


def punctuation_normalize(text):
    return " ".join(words(text))


def grams(text):
    tokens = words(text)
    return {" ".join(tokens[i:i + 8]) for i in range(len(tokens) - 7)}


def provider_key(value):
    return re.sub("[^a-z0-9]", "", value.casefold()) if isinstance(value, str) else ""


def expected_identity(label):
    if label.startswith("claude-"):
        return "anthropic", label
    if label.startswith("gemini-"):
        return "google", label
    if label.startswith("gpt-"):
        return "openai", ("gpt-5.5-2026-04-23" if label == "gpt-5.5" else label)
    return "openrouter", OPENROUTER[label][0]


def decode_envelope(row, envelope):
    """Read actual request/response fields rather than trusting derived CSV labels."""
    route, model_id = expected_identity(row.model)
    if envelope["response"]["http_status"] != 200:
        return None
    raw = envelope["response"]["json"]
    request = envelope["request"]["body"]
    if not all((
        envelope["call_id"] == row.call_id,
        envelope["prompt_id"] == row.prompt_id,
        envelope["prompt_sent"] == row.prompt_sent,
        envelope["label"] == row.model,
        envelope["route"] == route,
        envelope["split"] == row.split,
        request["model"] == model_id,
    )):
        return None
    if route == "google":
        candidates = raw["candidates"]
        if len(candidates) != 1:
            return None
        reply = candidates[0]
        answer = "".join(
            part.get("text", "") for part in reply["content"]["parts"] if not part.get("thought", False)
        )
        served = raw.get("model_version", raw.get("modelVersion"))
        contents = request["contents"]
        prompt = contents if isinstance(contents, str) else "".join(
            part.get("text", "") for item in contents for part in item.get("parts", [])
        )
        stop = reply.get("finish_reason", reply.get("finishReason"))
    else:
        messages = request["messages"]
        if len(messages) != 1 or messages[0]["role"] != "user" or request.get("system"):
            return None
        prompt = messages[0]["content"]
        served = raw["model"]
        if route == "anthropic":
            answer = "".join(p["text"] for p in raw["content"] if p.get("type") == "text")
            stop = raw["stop_reason"]
        else:
            if len(raw["choices"]) != 1:
                return None
            answer = raw["choices"][0]["message"]["content"]
            stop = raw["choices"][0]["finish_reason"]
            if route == "openrouter":
                providers = OPENROUTER[row.model][1]
                routing = request["provider"]
                only = routing.get("only")
                order = routing.get("order")
                if not (
                    provider_key(raw["provider"]) in providers
                    and routing.get("allow_fallbacks") is False
                    and isinstance(only, list) and len(only) == 1 and provider_key(only[0]) in providers
                    and isinstance(order, list) and len(order) == 1 and provider_key(order[0]) in providers
                ):
                    return None
    if (
        answer != row.text or served != model_id or prompt != row.prompt_sent
        or not isinstance(stop, str) or stop.casefold() not in FINISH
    ):
        return None
    return answer


def complete_groups(table):
    expected = set(CLASSES)
    return {prompt for prompt, group in table.groupby("prompt_id", sort=False)
            if len(group) == len(CLASSES) and set(group.model) == expected
            and group.split.nunique() == 1}


def response_overlap(official, extra):
    if extra.empty:
        return set(), {
            "evaluated_rows": 0, "exact_rows": 0, "normalized_rows": 0, "punctuation_rows": 0,
            "substring_rows": 0, "shared_8gram_rows": 0, "character_cosine_rows": 0,
            "maximum_character_cosine": None,
        }
    overlap = {"evaluated_rows": len(extra)}
    suspect = np.zeros(len(extra), dtype=bool)
    for name, norm in (("exact", lambda t: t), ("normalized", normalize), ("punctuation", punctuation_normalize)):
        exact = extra.text.map(norm).isin(set(official.text.map(norm))).to_numpy()
        overlap[name + "_rows"] = int(exact.sum())
        suspect |= exact
    original = list(map(normalize, official.text))
    candidates = list(map(normalize, extra.text))
    substring = np.array([any(a in b or b in a for a in original) for b in candidates], dtype=bool)
    official_grams = set().union(*(grams(t) for t in official.text))
    shared = np.array([len(grams(t) & official_grams) >= 3 for t in extra.text], dtype=bool)
    # This is the previous Stage 07A/09 audit's similarity measurement; no classifier is fit.
    vectorizer = TfidfVectorizer(
        analyzer="char", ngram_range=(3, 5), min_df=2, max_features=80000,
        sublinear_tf=True, dtype=np.float32,
    )
    with threadpool_limits(limits=2):
        original_x = vectorizer.fit_transform(official.text)
        candidate_x = vectorizer.transform(extra.text)
        neighbors = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute", n_jobs=2)
        distance, _ = neighbors.fit(original_x).kneighbors(candidate_x)
    cosine = 1 - distance[:, 0]
    need(bool(np.isfinite(cosine).all()), "nonfinite_overlap_distance")
    high = cosine >= .85
    suspect |= substring | shared | high
    overlap.update(substring_rows=int(substring.sum()), shared_8gram_rows=int(shared.sum()),
                   character_cosine_rows=int(high.sum()), maximum_character_cosine=float(max(cosine)))
    return set(extra.loc[suspect, "prompt_id"]), overlap


def audit(source, official_path, dataset_version):
    need(isinstance(dataset_version, int) and dataset_version > 0, "dataset_version_pin_required")
    need(len(PRIOR_PROMPTS) == 150, "earlier_prompt_exclusions_changed")
    input_hashes = {n: file_hash(source / n) for n in FILES}
    official_hash = file_hash(official_path)
    need(official_hash == TRAIN_SHA256, "wrong_official_train_hash")
    official = pd.read_csv(official_path, dtype=str, keep_default_na=False)
    need(len(official) == 2699 and official.prompt_id.nunique() == 150
         and official.id.is_unique and set(official.model) == set(CLASSES)
         and official.text.str.strip().ne("").all(), "wrong_official_train_structure")
    table = pd.read_csv(source / "passages_clean.csv", dtype=str, keep_default_na=False)
    bank = pd.read_csv(source / "prompts.csv", dtype=str, keep_default_na=False)
    need(CELL_COLUMNS <= set(table) and PROMPT_COLUMNS <= set(bank), "unrecognized_source_csv_schema")
    need(bank.prompt_id.is_unique and bank.prompt_id.str.strip().ne("").all(), "ambiguous_prompt_bank")
    need(table.id.is_unique and table.call_id.is_unique and table.call_id.str.strip().ne("").all(),
         "ambiguous_cell_or_call_id")
    eligible = table[table.split.isin(SPLITS)].copy()
    need(not eligible.empty and set(eligible.model) == set(CLASSES), "no_canonical_external_classes")
    need(eligible.prompt_id.isin(bank.prompt_id).all(), "missing_prompt_bank_ids")
    eligible["_normalized_text"] = eligible.text.map(punctuation_normalize)
    by_id = bank.set_index("prompt_id", verify_integrity=True)
    splits = dict(zip(eligible.prompt_id, eligible.split, strict=True))
    metrics = {}

    def tally(stage, groups):
        metrics[stage] = {split: sum(splits[g] == split for g in groups) for split in SPLITS}

    tally("input_distinct_prompt_ids", set(eligible.prompt_id))
    current = complete_groups(eligible)
    tally("complete_18_class_prompt_groups", current)
    current -= PRIOR_PROMPTS | PRIOR_NEAR_PROMPT_EXCLUSIONS
    tally("after_prior_exposure_exclusions", current)
    origin_text = bank.prompt_body.map(punctuation_normalize)
    prior_norms = set(origin_text[bank.prompt_id.isin(PRIOR_PROMPTS)])
    reconstruction_norms = set(origin_text[bank.split.eq("train_regen")])
    bank_group_counts = Counter(origin_text[bank.split.isin(SPLITS)])
    normalized_by_prompt = dict(zip(bank.prompt_id, origin_text, strict=True))

    def independent_origin(g):
        b = by_id.loc[g]
        normalized = normalized_by_prompt[g]
        fields = " ".join((b.prompt_source, b.get("source_prompt_id", ""), g))
        return (b.split == splits[g] and b.prompt_licence.strip().casefold() == "cc by 4.0"
                and bool(b.prompt_source.strip()) and bool(normalized)
                and not DERIVATIVE.search(fields) and b.get("source_prompt_id", "") not in set(official.prompt_id)
                and normalized not in prior_norms and normalized not in reconstruction_norms
                and bank_group_counts[normalized] == 1)

    current = {g for g in current if independent_origin(g)}
    tally("after_prompt_origin_license_duplicate_screen", current)
    flagged_duplicate_rows = eligible["_normalized_text"].duplicated(keep=False) | eligible["_normalized_text"].eq("")
    current -= set(eligible.loc[flagged_duplicate_rows, "prompt_id"])
    tally("after_external_text_duplicate_screen", current)

    def acceptable_cell(row):
        route, model_id = expected_identity(row.model)
        return (row.provider_route == route and row.requested_model_id == row.served_model_id == model_id
                and row.finish_reason_norm.casefold() in FINISH
                and 40 <= len(row.text.split()) <= 250
                and by_id.at[row.prompt_id, "prompt_body"].strip() in row.prompt_sent)

    route_rows = eligible[eligible.prompt_id.isin(current)]
    bad_routes = {r.prompt_id for r in route_rows.itertuples(index=False) if not acceptable_cell(r)}
    current -= bad_routes
    tally("after_clean_provider_identity_quality_screen", current)
    checked = eligible[eligible.prompt_id.isin(current)]
    needed_calls = set(checked.call_id)
    evidence = {}
    duplicate_call_ids = set()
    if needed_calls:
        with (source / "provenance.jsonl").open(encoding="utf-8") as fh:
            for line in fh:
                entry = json.loads(line)
                key = entry.get("call_id")
                if key in needed_calls:
                    if key in evidence:
                        duplicate_call_ids.add(key)
                    else:
                        evidence[key] = entry
    reasons = Counter()
    bad_envelopes = set()
    for row in checked.itertuples(index=False):
        key = row.call_id
        reason = None
        if key in duplicate_call_ids:
            reason = "duplicate_call"
        elif key not in evidence:
            reason = "missing_call"
        else:
            try:
                if decode_envelope(row, evidence[key]) is None:
                    reason = "mismatched_call"
            except (KeyError, IndexError, TypeError, AttributeError, ValueError):
                reason = "unrecognized_call_schema"
        if reason:
            bad_envelopes.add(row.prompt_id)
            reasons[reason] += 1
    current -= bad_envelopes
    tally("after_exact_publisher_raw_envelope_checks", current)

    overlap_groups, overlap = response_overlap(official, eligible[eligible.prompt_id.isin(current)])
    current -= overlap_groups
    tally("after_official_response_overlap_checks", current)
    ranked = sorted(current, key=lambda g: (text_hash(RANK_SALT + "\0" + normalized_by_prompt[g]), g))
    quota = min(MAX_PROMPTS, len(ranked) // BLOCK_PROMPTS * BLOCK_PROMPTS)
    chosen = ranked[:quota]
    tally("selected_prompt_groups", chosen)
    private_manifest = []
    for i, prompt in enumerate(chosen):
        block = i // BLOCK_PROMPTS + 1
        group = eligible[eligible.prompt_id.eq(prompt)].set_index("model", verify_integrity=True)
        for label in CLASSES:
            row = group.loc[label]
            private_manifest.append([block, prompt, label, row.id, row.call_id, text_hash(row.text)])
    digest = text_hash(json.dumps(private_manifest, separators=(",", ":"), ensure_ascii=False)) if chosen else None
    return {
        "status": "PANEL_DATA_CHECKS_PASS" if chosen else "BLOCKED_INSUFFICIENT_FRESH_COMPLETE_PROMPT_GROUPS",
        "schema_version": 1, "audit_scope": "external_proxy_stage13c_vs_stage13d",
        "dataset": "prvsiyan/" + DATASET_SLUG, "dataset_version": dataset_version,
        "input_sha256": {**input_hashes, "official_train.csv": official_hash},
        "prior_prompt_exclusions_count": len(PRIOR_PROMPTS),
        "source_declared_public_snapshot_only": True,
        "stage_counts_by_split": metrics, "envelope_failure_rows": dict(sorted(reasons.items())),
        "official_response_overlap": overlap,
        "eligible_prompt_groups": len(ranked), "selected_prompt_groups": quota,
        "selected_rows": quota * len(CLASSES), "selected_blocks": quota // BLOCK_PROMPTS,
        "panel_manifest_sha256": digest,
        "ranking_salt": RANK_SALT,
        "selection_rule": "maximum up to 90 in blocks of 30 by fixed salted normalized-prompt hash",
        "limits": [
            "Host prompt wording is unavailable; metadata, exact deduplication and response similarity cannot prove semantic independence.",
            "Publisher API records are evidence, not independent provider attestations.",
            "The source declarations and envelope checks do not resolve every upstream use condition.",
            "Older versions contributed data to S07; this audit cannot establish an unseen panel for S07.",
        ],
        "trained_or_scored": False, "official_test_read": False, "raw_rows_or_identifiers_exported": False,
    }


def discover(root, name, required_parts):
    paths = [
        p for p in root.rglob(name) if p.is_file() and all(s in p.parts for s in required_parts)
    ]
    need(len(paths) == 1, "ambiguous_or_missing_input_" + name.replace(".", "_"))
    return paths[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    ap.add_argument("--source", type=Path)
    ap.add_argument("--official", type=Path)
    ap.add_argument("--dataset-version", type=int, required=True, help="Explicit mounted Kaggle version")
    ap.add_argument("--out", type=Path, default=Path("stage13e_prvsiyan_admission.json"))
    args = ap.parse_args()
    try:
        source = args.source or discover(args.input_root, "passages_clean.csv", (DATASET_SLUG,)).parent
        official = args.official or discover(args.input_root, "train.csv", ("name-that-llm",))
        result = audit(source, official, args.dataset_version)
    except AuditError as exc:
        print("STAGE13E_DATA_AUDIT_ERROR", str(exc), flush=True)
        raise SystemExit(2) from None
    except Exception as exc:
        print("STAGE13E_DATA_AUDIT_ERROR unexpected_" + type(exc).__name__, flush=True)
        raise SystemExit(2) from None
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    args.out.write_text(encoded, encoding="utf-8")
    need(json.loads(args.out.read_text(encoding="utf-8")) == result, "saved_audit_readback_failed")
    print("STAGE13E_PRVSIYAN_ADMISSION", json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
