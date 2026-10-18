"""Read-only, reproducible audit; never fits a model or exports passage text.

Usage: python scripts/stage07_corpus_audit.py --corpus DIR --train train.csv --out DIR
Fetch V1 of olegzholobov/name-that-llm-response-corpus through Kaggle first.
Required assets are listed in FILES. No credentials or paid APIs are used.
"""
import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

FILES = ['data_dictionary.json', 'validation.json', 'source_catalog.csv',
         'external_short_wide.parquet', 'external_cell_metadata.parquet',
         'external_cell_provenance.parquet', 'external_row_metadata.parquet']
TRAIN_SHA = '40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707'


def normalize(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def punctuation_normalize(text):
    return ' '.join(re.findall(r'\w+', normalize(text)))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(corpus, train_path, out):
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((corpus / 'manifest.json').read_text())
    hashes = {n: sha(corpus / n) for n in FILES}
    for n, h in hashes.items():
        if h != manifest['files'][n]['sha256']:
            raise ValueError('Corpus checksum mismatch: ' + n)
    if sha(train_path) != TRAIN_SHA:
        raise ValueError('Official training snapshot mismatch')
    train = pd.read_csv(train_path, keep_default_na=False, dtype=str)
    assert len(train) == 2699 and train.prompt_id.nunique() == 150
    wide = pd.read_parquet(corpus / 'external_short_wide.parquet')
    labels = sorted(set(train.model))
    assert sorted(set(wide.columns) - {'id', 'prompt'}) == labels
    assert wide.id.is_unique
    long = wide.melt(id_vars=['id', 'prompt'], var_name='model', value_name='text').dropna(subset=['text'])
    assert long.text.map(lambda x: isinstance(x, str) and bool(x.strip())).all()
    meta = pd.read_parquet(corpus / 'external_cell_metadata.parquet')
    assert set(meta.view) == {'all', 'short'}, set(meta.view)
    meta = meta[meta.view.eq('short')]
    assert not meta.duplicated(['id', 'model']).any()
    data = long.merge(meta, on=['id', 'model'], how='outer', validate='one_to_one', indicator=True)
    assert data._merge.eq('both').all()
    data = data.drop(columns='_merge')
    words = data.text.str.split().str.len()
    assert words.between(40, 250).all()
    assert np.array_equal(words.to_numpy(), data.word_count.to_numpy())
    rows = pd.read_parquet(corpus / 'external_row_metadata.parquet')
    rows = rows[rows.view.eq('short')]
    assert rows.id.is_unique
    check = data.merge(rows[['id', 'prompt_group_id']], on='id', validate='many_to_one', suffixes=('', '_row'))
    assert check.prompt_group_id.eq(check.prompt_group_id_row).all()
    provenance = pd.read_parquet(corpus / 'external_cell_provenance.parquet')
    short_prov = provenance[provenance.cell_id_short.notna()].copy()
    assert short_prov.occurrence_id.is_unique
    assert set(data.cell_id) == set(short_prov.cell_id_short)
    links = short_prov.merge(data[['cell_id', 'response_id', 'model', 'prompt_group_id']],
                            left_on='cell_id_short', right_on='cell_id', validate='many_to_one', suffixes=('', '_cell'))
    for key in ['response_id', 'model', 'prompt_group_id']:
        assert links[key].eq(links[key + '_cell']).all()
    counts = data.groupby('model').agg(cells=('cell_id', 'size'), unique_responses=('response_id', 'nunique'),
                                      prompt_groups=('prompt_group_id', 'nunique')).reindex(labels)
    counts.to_csv(out / 'class_coverage.csv')
    source_counts = short_prov.groupby(['source', 'license_unknown', 'output_terms_review', 'model_evidence_class']).agg(
        occurrences=('occurrence_id', 'size'), cells=('cell_id_short', 'nunique'),
        unique_responses=('response_id', 'nunique'), models=('model', 'nunique'))
    source_counts.to_csv(out / 'source_review.csv')
    source_models = short_prov.groupby(['source', 'model']).cell_id_short.nunique().unstack(fill_value=0)
    source_models.reindex(columns=labels, fill_value=0).to_csv(out / 'source_model_counts.csv')
    # Candidate policy inspection only. Passing these flags is NOT source approval.
    permitted_flags = ((~short_prov.license_unknown) & (~short_prov.output_terms_review)
                       & (~short_prov.no_train) & short_prov.prompt_exact
                       & short_prov.text_only_suitable & short_prov.mechanically_eligible
                       & (~short_prov.possible_truncation) & (~short_prov.possible_reasoning_in_answer))
    flag_pass = short_prov[permitted_flags]
    flag_counts = flag_pass.groupby('model').response_id.nunique().reindex(labels, fill_value=0)
    flag_counts.rename('unique_responses_passing_flags_not_approved').to_csv(out / 'flag_filter_coverage.csv')
    # Independent official/external full-response checks, regardless of external label.
    unique = data.drop_duplicates('response_id').copy()
    overlap = {}
    for name, fn in [('exact', lambda x: x), ('unicode_case_whitespace', normalize),
                     ('punctuation', punctuation_normalize)]:
        original_set = set(train.text.map(fn))
        overlap[name] = int(unique.text.map(fn).isin(original_set).sum())
    unique['normalized'] = unique.text.map(normalize)
    cross_labels = unique.groupby('normalized').model.nunique().gt(1).sum()
    # Retrieval diagnostic, NOT an exhaustive semantic/prompt-independence certificate.
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), min_df=2, max_features=80000,
                                 sublinear_tf=True, dtype=np.float32)
    official_x = vectorizer.fit_transform(train.text)
    external_x = vectorizer.transform(unique.text)
    nearest = NearestNeighbors(n_neighbors=1, metric='cosine', algorithm='brute', n_jobs=2).fit(official_x)
    distances, indices = nearest.kneighbors(external_x)
    unique['nearest_official_id'] = train.id.iloc[indices[:, 0]].to_numpy()
    unique['nearest_official_prompt_id'] = train.prompt_id.iloc[indices[:, 0]].to_numpy()
    unique['character_cosine_similarity'] = 1 - distances[:, 0]
    unique[['response_id', 'model', 'nearest_official_id', 'nearest_official_prompt_id',
            'character_cosine_similarity']].sort_values('character_cosine_similarity', ascending=False).head(100).to_csv(
                out / 'nearest_official_responses.csv', index=False)
    speech = short_prov[short_prov.source.eq('speechmap')]
    speech.groupby('model').agg(cells=('cell_id_short', 'nunique'), unique_responses=('response_id', 'nunique')).reindex(labels).to_csv(out / 'speechmap_coverage.csv')
    report = {
        'stage': '07_external_corpus_audit', 'dataset': 'olegzholobov/name-that-llm-response-corpus',
        'dataset_version': 1, 'snapshot_date': '2026-10-15', 'sha256': hashes, 'official_train_sha256': TRAIN_SHA,
        'wide_rows': len(wide), 'answer_cells': len(data), 'unique_response_ids': int(data.response_id.nunique()),
        'prompt_groups': int(data.prompt_group_id.nunique()), 'nonexact_prompt_cells': int((~data.prompt_exact).sum()),
        'response_ids_in_multiple_prompt_groups': int(data.groupby('response_id').prompt_group_id.nunique().gt(1).sum()),
        'provenance_occurrences': len(short_prov), 'observed_word_range': [int(words.min()), int(words.max())],
        'official_full_response_overlap': overlap, 'normalized_texts_with_multiple_labels': int(cross_labels),
        'char_cosine_nearest_max': float(unique.character_cosine_similarity.max()),
        'char_cosine_nearest_ge_0_9': int(unique.character_cosine_similarity.ge(.9).sum()),
        'flag_pass_occurrences_NOT_approved': len(flag_pass),
        'flag_pass_unique_responses_NOT_approved': int(flag_pass.response_id.nunique()),
        'flag_pass_model_count_NOT_approved': int(flag_pass.model.nunique()),
        'speechmap_cells': int(speech.cell_id_short.nunique()),
        'speechmap_unique_responses': int(speech.response_id.nunique()),
        'speechmap_output_terms_review_occurrences': int(speech.output_terms_review.sum()),
        'license_unknown_short_occurrences': int(short_prov.license_unknown.sum()),
        'output_terms_review_short_occurrences': int(short_prov.output_terms_review.sum()),
        'training_started': False, 'official_submission_created': False,
        'decision': 'HOLD_FOR_SOURCE_AND_VALIDATION_REVIEW',
        'limitations': [
            'Only short-view text and its sidecars independently audited; full corpus text not downloaded.',
            'Publisher overlap checks are separate claims; our nearest-neighbor check is not semantic independence.',
            'Official train lacks prompt wording. Reconstructed prompts are guesses, not exact overlap certification.',
            'Repository Apache license verified for SpeechMap; provider-output terms flag remains unresolved.',
            'Flag-pass candidates include weaker identity evidence and uneven class/source coverage; not approved data.',
            'All/short views overlap and must never be concatenated as independent observations.',
            'No Luna quarantine downloaded or admitted; no hidden-test data or labels used.'
        ]}
    (out / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'sha256'}, indent=2), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', type=Path, required=True)
    ap.add_argument('--train', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    audit(args.corpus, args.train, args.out)
