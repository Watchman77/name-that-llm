"""Stage 04: fixed, offline MPNet representation comparison (no tuning)."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
import platform
import time
import warnings

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import numpy as np
import pandas as pd
import scipy
from scipy.sparse import csr_matrix, hstack
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.svm import LinearSVC
import torch
import transformers
from transformers import AutoModel, AutoTokenizer
from style_comparison import build_model

SEED = 20261016
SOURCE_SHA = '40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707'
PLAN_COMMIT = '9d34e75b0c2b2faff0f995bd441a8a3297bd77c4'
MODEL_ID = 'sentence-transformers/all-mpnet-base-v2'
REVISION = 'e8c3b32edf5434bc2275fc9bab85f82640a19130'
MAX_LENGTH = 384
BATCH_SIZE = 8
THREADS = 6
VARIANTS = ['mpnet_only', 'character_style_mpnet']
REFERENCE = 'character_plus_style'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encode(texts, model_dir, cache_dir, budget_started):
    """Fixed label-free transform; an exact-hash cache permits identical reruns."""
    files = sorted(p for p in model_dir.rglob('*') if p.is_file() and '.cache' not in p.parts)
    manifest = {str(p.relative_to(model_dir)): sha(p) for p in files}
    for required in ['config.json', 'model.safetensors', 'tokenizer.json']:
        assert required in manifest, f'Missing offline checkpoint file: {required}'
    cache_key = hashlib.sha256(json.dumps({
        'source': SOURCE_SHA, 'model': MODEL_ID, 'revision': REVISION,
        'weights': manifest, 'max_length': MAX_LENGTH, 'batch_size': BATCH_SIZE,
        'threads': THREADS, 'torch': torch.__version__, 'transformers': transformers.__version__,
    }, sort_keys=True).encode()).hexdigest()
    cache_dir.mkdir(parents=True, exist_ok=True)
    vector_path = cache_dir / f'{cache_key}.npy'
    metadata_path = cache_dir / f'{cache_key}.json'
    if vector_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        assert sha(vector_path) == metadata['embeddings_sha256']
        vectors = np.load(vector_path, allow_pickle=False)
        assert vectors.shape == (len(texts), 768)
        return vectors, metadata
    torch.set_num_threads(THREADS)
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, use_fast=True)
    assert tokenizer.is_fast
    lengths = np.array([len(ids) for ids in tokenizer(texts.tolist(), truncation=False,
                                                    padding=False)['input_ids']])
    encoder = AutoModel.from_pretrained(model_dir, local_files_only=True).float().eval()
    vectors = np.empty((len(texts), 768), dtype=np.float32)
    started = time.perf_counter()
    for offset in range(0, len(texts), BATCH_SIZE):
        if time.perf_counter() - budget_started > 3600:
            raise TimeoutError('Predeclared 60-minute modelling budget exceeded')
        batch = tokenizer(texts.iloc[offset:offset+BATCH_SIZE].tolist(), padding=True,
                          truncation=True, max_length=MAX_LENGTH, return_tensors='pt')
        with torch.inference_mode():
            hidden = encoder(**batch).last_hidden_state
            mask = batch['attention_mask'].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        vectors[offset:offset+len(pooled)] = pooled.cpu().numpy()
        if offset % 128 == 0:
            print(f'Encoded {min(offset+BATCH_SIZE,len(texts))}/{len(texts)}; '
                  f'{time.perf_counter()-started:.1f}s', flush=True)
    assert np.isfinite(vectors).all()
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1., atol=1e-5)
    np.save(vector_path, vectors, allow_pickle=False)
    metadata = {
        'model_id': MODEL_ID, 'revision': REVISION, 'licence': 'Apache-2.0',
        'source_sha256': SOURCE_SHA, 'checkpoint_sha256': manifest,
        'embeddings_sha256': sha(vector_path), 'cache_key': cache_key,
        'rows': len(texts), 'dimensions': 768, 'dtype': 'float32',
        'pooling': 'attention-mask mean, then L2 normalize',
        'max_tokens_including_special': MAX_LENGTH, 'batch_size': BATCH_SIZE,
        'truncated_rows': int((lengths > MAX_LENGTH).sum()),
        'full_token_length_quantiles': {str(q):float(np.quantile(lengths,q)) for q in [0,.5,.9,.95,1]},
        'encoding_seconds': time.perf_counter()-started,
        'executed_utc': datetime.now(timezone.utc).isoformat(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return vectors, metadata


def paired_intervals(train, output, labels):
    prompts = sorted(train.prompt_id.unique())
    gi = train.prompt_id.map({x:i for i,x in enumerate(prompts)}).to_numpy()
    yi = train.model.map({x:i for i,x in enumerate(labels)}).to_numpy()
    totals = np.zeros((len(prompts),len(labels)), dtype=np.int64)
    np.add.at(totals,(gi,yi),1)
    weights = np.random.default_rng(SEED).multinomial(len(prompts),
        np.full(len(prompts),1/len(prompts)),size=2000)
    denominator = weights @ totals
    assert (denominator > 0).all()
    bootstrap = {}
    scores = {}
    for kind in VARIANTS + [REFERENCE]:
        correct = np.zeros_like(totals)
        np.add.at(correct,(gi,yi),(output[kind].to_numpy()==train.model.to_numpy()).astype(int))
        bootstrap[kind] = ((weights @ correct)/denominator).mean(axis=1)
        scores[kind] = float(balanced_accuracy_score(train.model,output[kind]))
    rows = []
    for kind in VARIANTS:
        low,high = np.quantile(bootstrap[kind]-bootstrap[REFERENCE],[.025,.975])
        rows.append({'candidate':kind,'reference':REFERENCE,
            'delta_balanced_accuracy':scores[kind]-scores[REFERENCE],
            'lower_95':float(low),'upper_95':float(high)})
    return rows


def run(data_dir, previous_dir, out_dir, model_dir, cache_dir):
    assert not out_dir.exists(), f'Output already exists; preserve it before an intentional rerun: {out_dir}'
    out_dir.mkdir(parents=True)
    started = time.perf_counter()
    run_record = {'stage':'04_frozen_encoder', 'status':'running', 'plan_commit':PLAN_COMMIT,
                  'started_utc':datetime.now(timezone.utc).isoformat()}
    (out_dir/'run_status.json').write_text(json.dumps(run_record,indent=2))
    try:
        assert sha(data_dir/'train.csv') == SOURCE_SHA
        train = pd.read_csv(data_dir/'train.csv',dtype=str)
        old = pd.read_csv(previous_dir/'oof_predictions.csv',dtype={'id':str,'prompt_id':str,'model':str})
        assert train.id.is_unique and old.id.is_unique
        assert train[['id','prompt_id','model']].equals(old[['id','prompt_id','model']])
        assert len(train)==2699 and train.prompt_id.nunique()==150
        assert not train[['id','prompt_id','text','model']].isna().any().any()
        assert set(old.fold)=={1,2,3,4,5} and old.groupby('prompt_id').fold.nunique().eq(1).all()
        labels = sorted(train.model.unique())
        assert len(labels)==18
        assert np.isclose(balanced_accuracy_score(train.model,old[REFERENCE]),.35413124533929907)
        vectors,encoder_metadata = encode(train.text,model_dir,cache_dir,started)
        (out_dir/'encoder_provenance.json').write_text(json.dumps(encoder_metadata,indent=2))
        output = old[['id','prompt_id','model','fold',REFERENCE]].copy()
        rows,fold_rows,warning_records = [],[],[]
        for kind in VARIANTS:
            tick = time.perf_counter()
            predictions = np.empty(len(train),dtype=object)
            for fold in range(1,6):
                fold_started = time.perf_counter()
                if time.perf_counter()-started>3600:
                    raise TimeoutError('Predeclared 60-minute modelling budget exceeded')
                valid = old.fold.eq(fold).to_numpy()
                fit = ~valid
                assert set(train.loc[fit,'prompt_id']).isdisjoint(train.loc[valid,'prompt_id'])
                assert set(train.loc[fit,'model'])==set(labels)==set(train.loc[valid,'model'])
                x_fit,x_valid = vectors[fit],vectors[valid]
                if kind=='character_style_mpnet':
                    features = build_model('character_plus_style').named_steps['features']
                    x_fit = hstack([features.fit_transform(train.loc[fit,'text']),csr_matrix(x_fit)],format='csr')
                    x_valid = hstack([features.transform(train.loc[valid,'text']),csr_matrix(x_valid)],format='csr')
                classifier = LinearSVC(C=1.,class_weight='balanced',random_state=SEED,max_iter=10000,dual=True)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    classifier.fit(x_fit,train.loc[fit,'model'])
                warning_records.extend({'variant':kind,'fold':fold,'category':w.category.__name__,
                                        'message':str(w.message)} for w in caught)
                predictions[valid] = classifier.predict(x_valid)
                score = float(balanced_accuracy_score(train.loc[valid,'model'],predictions[valid]))
                fold_rows.append({'variant':kind,'fold':fold,'balanced_accuracy':score,
                    'rows':int(valid.sum()),'prompts':int(train.loc[valid,'prompt_id'].nunique()),
                    'seconds':time.perf_counter()-fold_started})
                print(f'{kind}, fold {fold}: {score:.6f}',flush=True)
            assert all(isinstance(x,str) for x in predictions)
            output[kind] = predictions
            rows.append({'variant':kind,'balanced_accuracy':float(balanced_accuracy_score(train.model,predictions)),
                         'seconds':time.perf_counter()-tick,'reused_predictions':False})
            output.to_csv(out_dir/'oof_predictions.csv',index=False)
        rows.append({'variant':REFERENCE,'balanced_accuracy':float(balanced_accuracy_score(train.model,old[REFERENCE])),
                     'seconds':None,'reused_predictions':True})
        for fold in range(1,6):
            selected = old.fold.eq(fold)
            fold_rows.append({'variant':REFERENCE,'fold':fold,
                'balanced_accuracy':float(balanced_accuracy_score(train.loc[selected,'model'],old.loc[selected,REFERENCE])),
                'rows':int(selected.sum()),'prompts':int(train.loc[selected,'prompt_id'].nunique()),'seconds':None})
        recalls,length_rows = [],[]
        bins = pd.cut(train.text.str.split().str.len(),[-np.inf,80,150,np.inf],
                      labels=['<=80 words','81-150 words','>150 words'])
        for kind in VARIANTS+[REFERENCE]:
            cm = confusion_matrix(train.model,output[kind],labels=labels)
            per_class = cm.diagonal()/cm.sum(axis=1)
            assert np.isclose(per_class.mean(),next(r['balanced_accuracy'] for r in rows if r['variant']==kind),atol=1e-12)
            recalls.extend({'variant':kind,'model':label,'recall':float(recall),'support':int(support)}
                           for label,recall,support in zip(labels,per_class,cm.sum(axis=1)))
            for name in bins.cat.categories:
                selected = bins.eq(name)
                length_rows.append({'variant':kind,'word_count_bin':name,'rows':int(selected.sum()),
                    'classes_present':int(train.loc[selected,'model'].nunique()),
                    'balanced_accuracy_present_classes':float(balanced_accuracy_score(train.loc[selected,'model'],output.loc[selected,kind]))})
        intervals = paired_intervals(train,output,labels)
        summary = pd.DataFrame(rows).sort_values('balanced_accuracy',ascending=False)
        summary.to_csv(out_dir/'comparison.csv',index=False)
        pd.DataFrame(fold_rows).to_csv(out_dir/'fold_scores.csv',index=False)
        pd.DataFrame(recalls).to_csv(out_dir/'per_model_recall.csv',index=False)
        pd.DataFrame(length_rows).to_csv(out_dir/'length_diagnostics.csv',index=False)
        pd.DataFrame(intervals).to_csv(out_dir/'paired_differences.csv',index=False)
        record = {**run_record,'status':'complete','source_sha256':SOURCE_SHA,
            'previous_predictions_sha256':sha(previous_dir/'oof_predictions.csv'),
            'code_sha256':{p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('style_comparison.py'),
                                                 Path(__file__).with_name('style_features.py')]},
            'comparisons':rows,'intervals':intervals,'warnings':warning_records,
            'total_seconds':time.perf_counter()-started,
            'environment':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
                'scipy':scipy.__version__,'sklearn':sklearn.__version__,'torch':torch.__version__,
                'transformers':transformers.__version__},
            'bootstrap':{'unit':'prompt_id','resamples':2000,'seed':SEED,
                'limitation':'Conditional development predictions; no refitting, selection or multiplicity correction.'},
            'finished_utc':datetime.now(timezone.utc).isoformat()}
        (out_dir/'metrics.json').write_text(json.dumps(record,indent=2))
        (out_dir/'run_status.json').write_text(json.dumps({**run_record,'status':'complete',
            'finished_utc':record['finished_utc']},indent=2))
        print(summary.to_string(index=False),flush=True)
        print('Paired differences:',json.dumps(intervals),flush=True)
        print('Warnings:',len(warning_records),flush=True)
    except Exception as exc:
        (out_dir/'run_status.json').write_text(json.dumps({**run_record,'status':'incomplete',
            'error_type':type(exc).__name__,'error':str(exc),'failed_utc':datetime.now(timezone.utc).isoformat()},indent=2))
        raise


if __name__=='__main__':
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir',type=Path,default=root/'data')
    parser.add_argument('--previous-dir',type=Path,default=root/'results/stage_03')
    parser.add_argument('--out-dir',type=Path,default=root/'results/stage_04')
    parser.add_argument('--model-dir',type=Path,required=True)
    parser.add_argument('--cache-dir',type=Path,required=True)
    args = parser.parse_args()
    run(args.data_dir,args.previous_dir,args.out_dir,args.model_dir,args.cache_dir)
