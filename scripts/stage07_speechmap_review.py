"""Verify every short SpeechMap cell against pinned primary JSONL files.

No generation, training, submission, network access or passage exports.
Usage: python scripts/stage07_speechmap_review.py --corpus DIR --raw DIR --train CSV --out DIR
"""
import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

COMMIT = '476ea8f0adf85b737bbb6e38e838448a8d4f2a9b'
REFUSAL = re.compile(r"\b(?:i (?:can[’']t|cannot|won[’']t|am unable to)|i[’']m (?:sorry|unable)|i must (?:decline|refuse)|i (?:can not|will not) (?:help|assist|provide|write|create))\b", re.I)


def extract(response):
    if not isinstance(response, dict):
        raise ValueError('response is not a JSON object')
    if 'choices' in response:
        choice = response['choices'][0]
        content = choice['message']['content']
        finish = choice.get('finish_reason')
    elif 'content' in response:
        content = response['content']
        finish = response.get('stop_reason')
    else:
        raise ValueError('unsupported response envelope')
    if isinstance(content, list):
        # Do not include thinking/tool blocks in the assistant answer.
        blocks = [x['text'] for x in content if x.get('type') == 'text']
        content = ''.join(blocks)
    if not isinstance(content, str):
        raise ValueError('assistant text is not a string')
    return content, response.get('model'), finish


def review(corpus, raw, train_path, out):
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((corpus / 'manifest.json').read_text())['files']
    for name in ['external_cell_provenance.parquet', 'external_short_wide.parquet']:
        assert hashlib.sha256((corpus / name).read_bytes()).hexdigest() == manifest[name]['sha256']
    assert hashlib.sha256(train_path.read_bytes()).hexdigest() == '40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707'
    q = pd.read_parquet(corpus / 'external_cell_provenance.parquet')
    q = q[q.cell_id_short.notna() & q.source.eq('speechmap')].copy()
    wide = pd.read_parquet(corpus / 'external_short_wide.parquet').set_index('id')
    records, source_files, failures = [], [], []
    for url, group in q.groupby('source_url', sort=True):
        prefix = f'https://github.com/xlr8harder/speechmap-data/blob/{COMMIT}/responses/'
        assert url.startswith(prefix)
        name = url[len(prefix):]
        assert '/' not in name
        b = (raw / name).read_bytes()
        assert not b.startswith(b'version https://git-lfs.github.com/spec/v1'), 'Resolve Git LFS object: ' + name
        lines = b.decode().splitlines()
        source_files.append({'file': name, 'source_url': url, 'bytes': len(b),
                             'sha256': hashlib.sha256(b).hexdigest(), 'lines': len(lines), 'selected_cells': len(group)})
        for row in group.itertuples():
            file_name, line = row.source_row_id.rsplit(':', 1)
            assert file_name == name
            original = json.loads(lines[int(line) - 1])
            answer, returned_model, finish = extract(original['response'])
            checks = {'answer': answer == wide.loc[row.row_id_short, row.model],
                      'prompt': original.get('question') == row.prompt,
                      'returned_model': returned_model == row.returned_model,
                      'requested_model': original.get('model') == row.raw_model}
            bad = [k for k, passed in checks.items() if not passed]
            if bad:
                failures.append({'cell_id': row.cell_id_short, 'source_row_id': row.source_row_id, 'mismatches': bad})
            records.append({'cell_id': row.cell_id_short, 'response_id': row.response_id,
                            'prompt_group_id': row.prompt_group_id, 'target_model': row.model,
                            'requested_model': original.get('model'), 'returned_model': returned_model,
                            'api_provider': original.get('api_provider'),
                            'inference_provider': original['response'].get('provider'),
                            'finish_reason': finish, 'timestamp': original.get('timestamp'),
                            'response_status': original.get('response_status'), 'domain': original.get('domain'),
                            'category': original.get('category'), 'source_file': name,
                            'opening_refusal_marker': bool(REFUSAL.search(answer[:500])),
                            'primary_source_all_fields_match': not bad})
    verified = pd.DataFrame(records)
    verified.to_csv(out / 'speechmap_verified_cells.csv', index=False)
    mapping = verified.groupby(['target_model', 'requested_model', 'returned_model'], dropna=False).agg(
        cells=('cell_id', 'size'), primary_matched=('primary_source_all_fields_match', 'sum')).reset_index()
    mapping.to_csv(out / 'speechmap_model_id_mapping.csv', index=False)
    per_model = verified.groupby('target_model').agg(cells=('cell_id', 'size'),
        prompt_groups=('prompt_group_id', 'nunique'), opening_refusal_markers=('opening_refusal_marker', 'sum'))
    per_model['opening_refusal_marker_fraction'] = per_model.opening_refusal_markers / per_model.cells
    per_model.to_csv(out / 'speechmap_model_review.csv')
    train = pd.read_csv(train_path, keep_default_na=False)
    official_marker = train.text.map(lambda t: bool(REFUSAL.search(t[:500])))
    # Screen public prompts against reconstructed instructions. This is retrieval,
    # not semantic adjudication, and the reconstruction is not ground truth.
    inferred = pd.read_csv(corpus / 'train_prompts_reconstructed.csv', keep_default_na=False)
    assert set(inferred.prompt_id) == set(train.prompt_id) and inferred.prompt_id.is_unique
    prompts = q[['prompt_group_id', 'prompt']].drop_duplicates()
    assert prompts.prompt_group_id.is_unique and prompts.prompt.ne('').all()
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words='english', dtype=np.float32)
    x = vec.fit_transform(pd.concat([inferred.inferred_prompt, prompts.prompt], ignore_index=True))
    nn = NearestNeighbors(n_neighbors=1, metric='cosine', algorithm='brute').fit(x[:len(inferred)])
    distance, ix = nn.kneighbors(x[len(inferred):])
    prompts['nearest_official_prompt_id'] = inferred.prompt_id.iloc[ix[:, 0]].to_numpy()
    prompts['word_cosine_similarity'] = 1 - distance[:, 0]
    prompts[['prompt_group_id', 'nearest_official_prompt_id', 'word_cosine_similarity']].sort_values(
        'word_cosine_similarity', ascending=False).to_csv(out / 'speechmap_prompt_screen.csv', index=False)
    # Diagnostic snippets are private scratch only, outside the repository output.
    diagnostics = prompts.sort_values('word_cosine_similarity', ascending=False).head(40).copy()
    diagnostics['inferred_official_instruction'] = diagnostics.nearest_official_prompt_id.map(
        inferred.set_index('prompt_id').inferred_prompt)
    diagnostics.to_json(raw.parent / 'prompt_screen_private.json', orient='records', indent=2)
    report = {'source_commit': COMMIT, 'files_checked': len(source_files), 'cells_checked': len(verified),
              'target_models': int(verified.target_model.nunique()),
              'cells_matching_all_four_fields': int(verified.primary_source_all_fields_match.sum()),
              'mismatch_count': len(failures), 'mismatches': failures,
              'primary_file_receipts': source_files,
              'requested_returned_target_mappings': len(mapping),
              'finish_reasons': verified.finish_reason.fillna('<missing>').value_counts().to_dict(),
              'response_status': verified.response_status.fillna('<missing>').value_counts().to_dict(),
              'api_providers': verified.api_provider.fillna('<missing>').value_counts().to_dict(),
              'inference_providers': verified.inference_provider.fillna('<missing>').value_counts().to_dict(),
              'domain_counts': verified.domain.fillna('<missing>').value_counts().to_dict(),
              'source_timestamp_min': verified.timestamp.min(), 'source_timestamp_max': verified.timestamp.max(),
              'opening_refusal_marker_cells': int(verified.opening_refusal_marker.sum()),
              'opening_refusal_marker_fraction': float(verified.opening_refusal_marker.mean()),
              'official_opening_refusal_marker_rows': int(official_marker.sum()),
              'official_opening_refusal_marker_fraction': float(official_marker.mean()),
              'refusal_marker_caveat': 'Regex in first 500 characters; neither exhaustive refusal labels nor a quality verdict.',
              'distinct_speechmap_prompt_groups': len(prompts),
              'exact_prompt_matches_to_reconstructions': int(prompts.prompt.isin(set(inferred.inferred_prompt)).sum()),
              'nearest_reconstructed_prompt_cosine_max': float(prompts.word_cosine_similarity.max()),
              'prompt_screen_caveat': 'Lexical retrieval against guessed instructions; not proof of prompt independence.',
              'training_performed': False, 'all_source_terms_approved': False}
    (out / 'speechmap_primary_review.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'primary_file_receipts'}, indent=2))
    if failures:
        raise ValueError('Primary-source mismatches found; do not admit these cells')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    for name in ['corpus', 'raw', 'train', 'out']:
        ap.add_argument('--' + name, type=Path, required=True)
    args = ap.parse_args()
    review(args.corpus, args.raw, args.train, args.out)
