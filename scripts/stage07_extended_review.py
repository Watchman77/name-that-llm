"""Read-only review of prvsiyan extended corpus V4, using published call envelopes.

No model training or external API calls. Publisher envelopes are evidence, not
independent provider attestations. Passage text is not exported.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path
import pandas as pd
from stage07_corpus_audit import normalize, punctuation_normalize
from stage07_speechmap_review import REFUSAL


def raw_fields(envelope):
    response = envelope['response']['json']
    body = envelope['request']['body']
    if 'candidates' in response:
        text = ''.join(x.get('text', '') for x in response['candidates'][0]['content']['parts'] if not x.get('thought', False))
        model = response.get('model_version', response.get('modelVersion'))
        content = body['contents']
        if isinstance(content, str):
            prompt = content
        else:
            prompt = ''.join(x.get('text', '') for c in content for x in c.get('parts', []))
    elif 'choices' in response:
        text = response['choices'][0]['message']['content']
        model = response['model']
        prompt = body['messages'][0]['content']
    elif 'content' in response:
        text = ''.join(x['text'] for x in response['content'] if x.get('type') == 'text')
        model = response['model']
        prompt = body['messages'][0]['content']
    else:
        raise ValueError('Unknown response schema')
    return text, model, body['model'], prompt


def review(source, official_path, out):
    out.mkdir(parents=True, exist_ok=True)
    assert hashlib.sha256(official_path.read_bytes()).hexdigest() == '40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707'
    original = pd.read_csv(official_path, keep_default_na=False)
    table = pd.read_csv(source / 'passages_clean.csv', keep_default_na=False)
    data = table[table.split.eq('new')].copy()
    assert len(data) == 4068 and data.id.is_unique and data.call_id.is_unique
    assert set(data.model) == set(original.model)
    assert data.groupby('prompt_id').model.nunique().eq(18).all()
    assert not data.duplicated(['prompt_id', 'model']).any()
    assert data.groupby('model').size().eq(226).all()
    assert data.text.str.split().str.len().between(40, 250).all()
    selected = set(data.call_id)
    evidence = {}
    with (source / 'provenance.jsonl').open() as f:
        for line in f:
            record = json.loads(line)
            if record.get('call_id') in selected:
                assert record['call_id'] not in evidence
                evidence[record['call_id']] = record
    assert set(evidence) == selected
    mismatches, comparisons = [], []
    for row in data.itertuples():
        env = evidence[row.call_id]
        text, served, requested, prompt = raw_fields(env)
        checks = {'response_normalized_whitespace_only': ' '.join(text.split()) == ' '.join(row.text.split()),
                  'served_id': served == row.served_model_id, 'requested_id': requested == row.requested_model_id,
                  'prompt': prompt == row.prompt_sent, 'envelope_label': env['label'] == row.model,
                  'envelope_prompt_id': env['prompt_id'] == row.prompt_id}
        bad = [k for k, ok in checks.items() if not ok]
        if bad:
            mismatches.append({'id': row.id, 'call_id': row.call_id, 'mismatches': bad})
        comparisons.append({'id': row.id, 'call_id': row.call_id, 'model': row.model, 'route': row.provider_route,
                            'all_checks_passed': not bad, 'raw_text_exact': text == row.text})
    comparison = pd.DataFrame(comparisons)
    marker = data.text.map(lambda x: bool(REFUSAL.search(x[:500])))
    data['opening_refusal_marker'] = marker
    data['em_dash'] = data.text.str.contains('—', regex=False)
    original['em_dash'] = original.text.str.contains('—', regex=False)
    profile = data.groupby('model').agg(rows=('id', 'size'), opening_refusal_markers=('opening_refusal_marker', 'sum'),
                                       external_em_dash_fraction=('em_dash', 'mean'))
    profile['official_em_dash_fraction'] = original.groupby('model').em_dash.mean()
    profile.to_csv(out / 'extended_v4_model_profile.csv')
    overlap = {}
    for key, fn in [('exact', lambda x:x), ('unicode_case_whitespace', normalize), ('punctuation', punctuation_normalize)]:
        overlap[key] = int(data.text.map(fn).isin(set(original.text.map(fn))).sum())
    report = {'dataset':'prvsiyan/prvsiyans-extended-name-that-llm-dataset', 'version':4,
              'scope':'new split in passages_clean.csv only; proxy_v1 and all other splits excluded',
              'rows':len(data), 'prompt_groups':int(data.prompt_id.nunique()), 'models':int(data.model.nunique()),
              'per_class_rows':226, 'all_rows_have_published_call_envelopes':True,
              'raw_fields_matching_rows':int(comparison.all_checks_passed.sum()), 'mismatch_count':len(mismatches),
              'mismatches':mismatches, 'raw_text_exact_rows':int(comparison.raw_text_exact.sum()),
              'raw_text_comparison_policy':'Whitespace normalization only; no spelling, punctuation or content edits.',
              'raw_call_evidence_limit':'Publisher-supplied logs; no independent provider attestation.',
              'exact_duplicate_passages':int(data.text.duplicated().sum()),
              'normalized_duplicate_passages':int(data.text.map(normalize).duplicated().sum()),
              'official_full_response_overlap':overlap,
              'opening_refusal_markers':int(marker.sum()),
              'routes':data.provider_route.value_counts().to_dict(),
              'gemini_37_routes':data[data.model.eq('gemini-3.7-flash')].provider_route.value_counts().to_dict(),
              'gemini_37_raw_served_versions':data[data.model.eq('gemini-3.7-flash')].served_model_id.value_counts().to_dict(),
              'files':{p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
                       for p in sorted(source.iterdir()) if p.is_file()},
              'decision':'BETTER_DOMAIN_CANDIDATE_REQUIRES_PROMPT_AND_ROUTE_REVIEW',
              'training_performed':False, 'accuracy_improvement_demonstrated':False,
              'limitations':['Prompt independence cannot be certified from guessed official instructions; separate lexical/manual review is recorded alongside this artifact.',
                             'Source CC BY 4.0 declaration does not resolve every underlying route/provider agreement.',
                             'No per-label typography normalization or automatic parameter tuning authorized by this review.',
                             'Measured Gemini em-dash distribution differs from official train; labels alone do not ensure matching generation settings.']}
    (out / 'extended_v4_review.json').write_text(json.dumps(report,indent=2)+'\n')
    data[['id','call_id','prompt_id','model','provider_route','requested_model_id','served_model_id']].to_csv(
        out / 'extended_v4_candidate_ids.csv', index=False)
    print(json.dumps({k:v for k,v in report.items() if k!='files'},indent=2))
    if mismatches:
        raise ValueError('Raw evidence mismatch; inspect report before data admission')


if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    for name in ['source','official','out']:
        ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args()
    review(args.source,args.official,args.out)
