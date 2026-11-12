"""Read-only CPU feasibility audit for Oleg's exact Kaggle short-view V1.

The inherited Stage 07 first-500-character refusal screen gives an upper
bound: full-context, source, prior-use and overlap checks can only shrink it.
This runner never admits a panel, fits a model, or prints/saves passage rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import pandas as pd

DATASET = 'olegzholobov/name-that-llm-response-corpus'
SLUG = DATASET.split('/')[1]
MIN_PROMPTS = 30
LABELS = frozenset('''claude-fable-5 claude-opus-4-6 claude-opus-4-8 claude-opus-5 claude-sonnet-5
gemini-3.1-pro-preview gemini-3.6-flash gemini-3.7-flash gemini-3.8-flash glm-5.3 glm-5.3-flash
gpt-5.5 gpt-5.6-luna gpt-5.6-sol gpt-5.6-terra gpt-6-astra kimi-k3 qwen3.8-max'''.split())
FOCUS = 'gemini-3.8-flash'
WIDE = 'external_short_wide.parquet'
CELLS = 'external_cell_metadata.parquet'
ROWS = 'external_row_metadata.parquet'
PROVENANCE = 'external_cell_provenance.parquet'
HASHES = {
    'manifest.json': '276d8f9442de9da2d4c55f8ba80f52ff59ae1a82dabdf942ae015139ee80276a',
    'data_dictionary.json': '2faffbc7f2485fa20a3dcce6ae52ff9c12d1c3d99df644cf196c79dab8ddd60d',
    'validation.json': 'fd0fbb3307a12fe47852e84086275f74a82744776bcc5912f0c219b1820da17e',
    'source_catalog.csv': 'c1033245fa80f8c0ca01104c8aeb2a76be0b823aec111f0c77d04345faec53d3',
    WIDE: '15a67d022315b2508af8edd749a7c041d42cc37e266106028c73e405659c400f',
    CELLS: '397af75e5f2809107d0758a147646a68a0132479a6ea77432d86243e92ad9079',
    PROVENANCE: '906345a4d1f418da63267bf686e1e6847c09f347ecf9eb939860d9aae32abdd6',
    ROWS: 'e2ee04939c33afecb9a5d8cbf25daca4e9bed9df354873c6ad054760cc6ce5d6',
}
# Exact Stage 07 expression from scripts/stage07_speechmap_review.py.
REFUSAL = re.compile(r"\b(?:i (?:can[’']t|cannot|won[’']t|am unable to)|i[’']m (?:sorry|unable)|i must (?:decline|refuse)|i (?:can not|will not) (?:help|assist|provide|write|create))\b", re.I)
PASS_FLAGS = ('prompt_exact', 'text_only_suitable', 'mechanically_eligible')
STOP_FLAGS = ('license_unknown', 'output_terms_review', 'no_train', 'possible_truncation', 'possible_reasoning_in_answer')


class AuditError(ValueError):
    pass


def need(ok, code):
    if not ok:
        raise AuditError(code)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def count_complete(table, key):
    labels = table.groupby(key, sort=False).model.agg(frozenset)
    return int(labels.eq(LABELS).sum())


def per_label_cells(table):
    counts = table.groupby('model').cell_id.nunique()
    return {model: int(counts.get(model, 0)) for model in sorted(LABELS)}


def analyze(wide, meta, rows, provenance):
    need(set(wide) == LABELS | {'id', 'prompt'}, 'wide_label_schema')
    need({'view', 'id', 'model', 'cell_id', 'response_id', 'prompt_group_id', 'prompt_exact', 'word_count'} <= set(meta), 'cell_schema')
    need({'view', 'id', 'prompt_group_id'} <= set(rows), 'row_schema')
    need({'cell_id_short', 'occurrence_id', 'source', 'model', 'response_id', 'prompt_group_id', *PASS_FLAGS, *STOP_FLAGS} <= set(provenance), 'provenance_schema')
    need(wide.id.notna().all() and wide.id.is_unique, 'wide_row_identity')
    long = wide.melt(id_vars=['id', 'prompt'], var_name='model', value_name='text').dropna(subset=['text'])
    need(long.text.map(lambda x: isinstance(x, str) and bool(x.strip())).all(), 'invalid_passage')
    meta = meta.loc[meta.view.eq('short'), ['id', 'model', 'cell_id', 'response_id', 'prompt_group_id', 'prompt_exact', 'word_count']]
    need(not meta.duplicated(['id', 'model']).any() and meta.cell_id.is_unique, 'duplicate_cell_identity')
    need(meta[['id', 'model', 'cell_id', 'response_id', 'prompt_group_id']].notna().all().all(), 'missing_cell_identity')
    data = long.merge(meta, on=['id', 'model'], how='outer', validate='one_to_one', indicator=True)
    need(data._merge.eq('both').all(), 'wide_cell_link')
    data = data.drop(columns='_merge')
    need(set(data.model) == LABELS, 'missing_label')
    words = data.text.str.split().str.len()
    need(words.between(40, 250).all() and words.eq(data.word_count).all(), 'short_word_bound')

    rows = rows.loc[rows.view.eq('short'), ['id', 'prompt_group_id']]
    need(rows.id.is_unique and set(rows.id) == set(wide.id), 'row_coverage')
    check = data.merge(rows, on='id', validate='many_to_one', suffixes=('', '_row'))
    need(check.prompt_group_id.eq(check.prompt_group_id_row).all(), 'row_group_link')
    short = provenance.loc[provenance.cell_id_short.notna()].copy()
    need(short.occurrence_id.notna().all() and short.occurrence_id.is_unique, 'occurrence_identity')
    need(short.source.map(lambda x: isinstance(x, str) and bool(x.strip())).all(), 'source_identity')
    need(set(data.cell_id) == set(short.cell_id_short), 'provenance_cell_coverage')
    links = short.merge(data[['cell_id', 'response_id', 'model', 'prompt_group_id']],
                        left_on='cell_id_short', right_on='cell_id', validate='many_to_one', suffixes=('', '_cell'))
    for name in ('response_id', 'model', 'prompt_group_id'):
        need(links[name].eq(links[f'{name}_cell']).all(), f'provenance_{name}_link')

    # Unknown sidecar flags cannot qualify. This remains a diagnostic only.
    pass_flag = links.source.notna()
    for col in PASS_FLAGS:
        pass_flag &= links[col].eq(True).fillna(False)
    for col in STOP_FLAGS:
        pass_flag &= links[col].eq(False).fillna(False)
    flag_cells = set(links.loc[pass_flag, 'cell_id_short'])
    data['refusal_marker'] = data.text.map(lambda x: bool(REFUSAL.search(x[:500])))
    nonrefusal = data.loc[~data.refusal_marker]
    exact = data.loc[data.prompt_exact.eq(True).fillna(False) & data.prompt.map(lambda x: isinstance(x, str) and bool(x.strip()))]
    exact_after = exact.loc[~exact.refusal_marker]
    exact_after_flags = exact_after.loc[exact_after.cell_id.isin(flag_cells)]
    by_label = per_label_cells(nonrefusal)
    exact_after_complete = count_complete(exact_after, 'prompt')
    # A strict 18-model group requires at least one surviving cell from every
    # model, so this bound is independent of publisher group and context IDs.
    cap = min(by_label.values())
    complete_cap = min(cap, exact_after_complete)
    source_focus = links.loc[links.model.eq(FOCUS)]
    refusals = set(data.loc[data.refusal_marker, 'cell_id'])
    focus_summary = {}
    for name, part in source_focus.groupby('source', sort=True):
        ids = set(part.cell_id_short)
        focus_summary[name] = {
            'unique_cells': len(ids), 'opening_refusal_marker_cells': len(ids & refusals),
            'cells_after_refusal_marker_screen': len(ids - refusals),
            'cells_with_unknown_license_flag': int(part.loc[part.license_unknown.eq(True), 'cell_id_short'].nunique()),
            'cells_with_output_terms_review_flag': int(part.loc[part.output_terms_review.eq(True), 'cell_id_short'].nunique()),
        }
    return {
        'decision': 'BLOCKED_UPPER_BOUND_BELOW_30' if complete_cap < MIN_PROMPTS else 'UPPER_BOUND_NOT_DECISIVE_NO_PANEL_ADMITTED',
        'minimum_required_prompt_groups': MIN_PROMPTS,
        'wide_rows': len(wide), 'answer_cells': len(data),
        'unique_response_ids': int(data.response_id.nunique()),
        'source_occurrences': len(links), 'publisher_prompt_groups': int(data.prompt_group_id.nunique()),
        'complete_publisher_prompt_groups_before_screen': count_complete(data, 'prompt_group_id'),
        'complete_exact_user_text_groups_before_screen': count_complete(exact, 'prompt'),
        'complete_exact_user_text_groups_after_refusal_marker_screen': exact_after_complete,
        'complete_exact_user_text_groups_after_refusal_and_source_flags': count_complete(exact_after_flags, 'prompt'),
        'cells_by_label_before_screen': per_label_cells(data),
        'cells_by_label_after_refusal_marker_screen': by_label,
        'minimum_per_label_cell_upper_bound_after_refusal_marker_screen': cap,
        'max_possible_groups_after_refusal_and_exact_user_text': complete_cap,
        'opening_refusal_marker_cells': int(data.refusal_marker.sum()),
        'gemini_3_8_flash_source_counts': focus_summary,
        'limitations': [
            'Exact user text is only a loose upper bound and does not prove matching full request context.',
            'The fixed first-500-character refusal regex is a screening rule, not a complete semantic refusal annotation.',
            'Source sidecar flags are not an independent licence or identity determination.',
            'Earlier prompt use and official overlap are not checked in this feasibility audit; those gates can only reduce the bound.',
        ],
        'selected_prompt_groups': 0, 'panel_admitted': False, 'models_trained_or_scored': False,
        'official_submission_created': False, 'raw_rows_or_ids_exported': False,
    }


def audit(corpus):
    observed = {name: file_hash(corpus / name) for name in HASHES}
    for name, expected in HASHES.items():
        need(observed[name] == expected, f'wrong_v1_sha256_{name}')
    manifest = json.loads((corpus / 'manifest.json').read_text())
    for name in set(HASHES) - {'manifest.json'}:
        need(manifest['files'][name]['sha256'] == observed[name], 'publisher_manifest_mismatch')
    result = analyze(*(pd.read_parquet(corpus / name) for name in (WIDE, CELLS, ROWS, PROVENANCE)))
    need((result['wide_rows'], result['answer_cells'], result['unique_response_ids'], result['source_occurrences'], result['publisher_prompt_groups'])
         == (13642, 24025, 24015, 24110, 12032), 'prior_verified_v1_structure_mismatch')
    return {'dataset': DATASET, 'dataset_version': 1, 'input_sha256': observed, **result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path)
    parser.add_argument('--input-root', type=Path, default=Path('/kaggle/input'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    try:
        corpus = args.corpus
        if corpus is None:
            matches = [p.parent for p in args.input_root.rglob(WIDE) if SLUG in p.parts]
            need(len(matches) == 1, 'dataset_mount_not_unique')
            corpus = matches[0]
        report = audit(corpus)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, sort_keys=True, indent=2) + '\n')
        need(json.loads(args.out.read_text()) == report, 'saved_report_mismatch')
    except AuditError as error:
        print(f'STAGE13E_OLEG_AUDIT_ERROR {error}', flush=True)
        raise SystemExit(2) from None
    except Exception as error:
        print(f'STAGE13E_OLEG_AUDIT_ERROR {type(error).__name__}', flush=True)
        raise SystemExit(2) from None
    print('STAGE13E_OLEG_UPPER_BOUND ' + json.dumps(report, sort_keys=True, separators=(',', ':')), flush=True)


if __name__ == '__main__':
    main()
