"""Stage 07A: one frozen, private character/style data-augmentation pilot.

No network calls, model serialization, predictions on test, or submission.
Only the first-party OpenAI route from the reviewed V4 new/clean corpus.
"""
import argparse
import hashlib
import json
import platform
import re
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.sparse import vstack
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import LinearSVC

from stage07_corpus_audit import normalize, punctuation_normalize
from style_features import SurfaceStyleFeatures

SEED = 20261018
TRAIN_SHA = '40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707'
CLEAN_SHA = '23f90b6ba4cc83137ce8d48b3b3787439b44867532d63e2fe52f8a5ea49cd1e6'
PROMPTS_SHA = '7cc4651b362ceadff4a1fd88dd1f82bad0c84d10b621a236be72c27f1b8f3a52'
STYLE_SHA = '8229a8fb4c0839cae8dec2ee50065b4e7863192b329faf0a463016a1055fccd4'
MODELS = {'gpt-5.5': 'gpt-5.5-2026-04-23', 'gpt-5.6-luna': 'gpt-5.6-luna',
          'gpt-5.6-sol': 'gpt-5.6-sol', 'gpt-5.6-terra': 'gpt-5.6-terra', 'gpt-6-astra': 'gpt-6-astra'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def grams(text):
    words = punctuation_normalize(text).split()
    return {' '.join(words[i:i+8]) for i in range(len(words)-7)}


def preflight(train_path, source, out):
    out.mkdir(parents=True, exist_ok=True)
    for path, expected in [(train_path, TRAIN_SHA), (source/'passages_clean.csv', CLEAN_SHA),
                           (source/'prompts.csv', PROMPTS_SHA), (Path(__file__).with_name('style_features.py'), STYLE_SHA)]:
        if sha(path) != expected:
            raise ValueError('Frozen input checksum mismatch: ' + str(path))
    train = pd.read_csv(train_path, keep_default_na=False, dtype=str)
    data = pd.read_csv(source/'passages_clean.csv', keep_default_na=False)
    extra = data[data.split.eq('new') & data.provider_route.eq('openai')].copy()
    assert len(train)==2699 and train.prompt_id.nunique()==150 and train.model.nunique()==18
    assert train.id.is_unique and not train.text.map(normalize).duplicated().any()
    assert len(extra)==750 and extra.id.is_unique and extra.call_id.is_unique
    assert set(extra.model)==set(MODELS)
    assert extra.groupby('model').size().eq(150).all()
    assert extra.groupby('prompt_id').model.nunique().eq(5).all()
    assert extra.prompt_id.nunique()==150 and not extra.duplicated(['prompt_id','model']).any()
    assert extra.text.str.split().str.len().between(40,250).all()
    for model, identifier in MODELS.items():
        rows=extra[extra.model.eq(model)]
        assert rows.requested_model_id.eq(identifier).all() and rows.served_model_id.eq(identifier).all()
    assert extra.finish_reason_norm.eq('stop').all()
    assert not extra.is_refusal.any() and not extra.is_truncated.any() and not extra.is_empty.any()
    assert extra.passes_funnel.all() and extra.in_word_range.all()
    assert not extra.text.map(normalize).duplicated().any()
    # Labels are not used in this conservative all-official-text exclusion screen.
    # These diagnostic features are discarded and never used by the CV classifiers.
    normalized_official=[normalize(x) for x in train.text]
    normalized_extra=[normalize(x) for x in extra.text]
    overlap=[]
    for name, fn in [('exact',lambda x:x),('unicode',normalize),('punctuation',punctuation_normalize)]:
        original=set(train.text.map(fn))
        overlap.append((name, extra.text.map(fn).isin(original).to_numpy()))
    official_grams=set().union(*(grams(t) for t in train.text))
    shared=np.array([len(grams(t)&official_grams) for t in extra.text])
    substring=np.array([any(a in b or b in a for a in normalized_official) for b in normalized_extra])
    vectorizer=TfidfVectorizer(analyzer='char',ngram_range=(3,5),min_df=2,max_features=80000,
                               sublinear_tf=True,dtype=np.float32)
    x=vectorizer.fit_transform(train.text)
    ext=vectorizer.transform(extra.text)
    dist, ix=NearestNeighbors(n_neighbors=1,metric='cosine',algorithm='brute',n_jobs=2).fit(x).kneighbors(ext)
    similarities=1-dist[:,0]
    suspicious=substring|(shared>=3)|(similarities>=.85)
    for _,mask in overlap:
        suspicious |= mask
    # Exclude entire external prompt blocks globally, before fitting either arm.
    excluded=set(extra.loc[suspicious,'prompt_id'])
    manifest=extra[['id','call_id','prompt_id','model','requested_model_id','served_model_id']].copy()
    manifest['shared_official_8grams']=shared
    manifest['nearest_official_id']=train.id.iloc[ix[:,0]].to_numpy()
    manifest['nearest_official_character_cosine']=similarities
    manifest['substring_overlap']=substring
    manifest['excluded_prompt_block']=manifest.prompt_id.isin(excluded)
    manifest.to_csv(out/'candidate_manifest.csv',index=False)
    extra=extra[~extra.prompt_id.isin(excluded)].sort_values(['prompt_id','model']).reset_index(drop=True)
    assert len(extra)>0 and extra.groupby('prompt_id').model.nunique().eq(5).all()
    # Every admitted response has exactly one prompt association in this split.
    assert not extra.text.map(normalize).duplicated().any()
    receipt={'stage':'07A_private_openai_route_pilot','seed':SEED,'official_train_sha256':TRAIN_SHA,
             'source_clean_sha256':CLEAN_SHA,'source_prompts_sha256':PROMPTS_SHA,
             'style_code_sha256':STYLE_SHA,'candidate_rows_before_exclusions':750,
             'candidate_rows_after_exclusions':len(extra),'candidate_prompts_after_exclusions':int(extra.prompt_id.nunique()),
             'excluded_external_prompt_ids':sorted(excluded),
             'full_response_overlap_rows':{name:int(mask.sum()) for name,mask in overlap},
             'substring_overlap_rows':int(substring.sum()),'shared_8gram_threshold_rows':int((shared>=3).sum()),
             'max_nearest_official_character_cosine':float(similarities.max()),
             'character_similarity_threshold_rows':int((similarities>=.85).sum()),
             'candidate_manifest_sha256':sha(out/'candidate_manifest.csv'),
             'source':'prvsiyan/prvsiyans-extended-name-that-llm-dataset V4; new clean; direct openai only',
             'use_scope':'Private undistributed SVM classifier; no generative fine-tune, model serialization or test inference.',
             'official_validation':'GroupKFold(5) on original prompt_id, all 18 classes',
             'prompt_review':'150 selected named external topics inspected against 150 reconstructed official topics; no same named topic identified. Top-30 lexical pairs from full candidate corpus separately reviewed. Reconstructions are guesses; semantic independence not certified.',
             'status':'PREFLIGHT_COMPLETE_NO_FITTING',
             'environment':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
                            'scipy':scipy.__version__,'sklearn':sklearn.__version__}}
    (out/'preflight.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return train, extra, receipt


def features():
    return FeatureUnion([('character',TfidfVectorizer(analyzer='char',ngram_range=(3,5),min_df=3,
                                  max_features=120000,sublinear_tf=True)),
                         ('style',Pipeline([('extract',SurfaceStyleFeatures()),('scale',StandardScaler()),
                                             ('normalize',Normalizer())]))])


def weights(y, external_y, labels):
    counts=pd.Series(y).value_counts()
    base=np.array([len(y)/(len(labels)*counts[v]) for v in y])
    ext_counts=pd.Series(external_y).value_counts()
    ext=np.array([.5*len(y)/(len(labels)*ext_counts[v]) for v in external_y])
    return base,ext


def bootstrap(table, labels, draws=10000):
    # Prompt x class sufficient counts; preserves whole-group dependence and
    # recomputes class recall on each draw, including the one missing Fable row.
    groups=sorted(table.prompt_id.unique())
    cube=np.zeros((len(groups),len(labels),3),dtype=np.int64)
    gi={g:i for i,g in enumerate(groups)};li={label:i for i,label in enumerate(labels)}
    for r in table.itertuples():
        i,j=gi[r.prompt_id],li[r.model]
        cube[i,j]+=[1,int(r.control==r.model),int(r.candidate==r.model)]
    rng=np.random.default_rng(SEED);delta=[]
    for _ in range(draws):
        s=cube[rng.integers(0,len(groups),len(groups))].sum(axis=0)
        assert (s[:,0]>0).all()
        delta.append(float(np.mean(s[:,2]/s[:,0])-np.mean(s[:,1]/s[:,0])))
    return np.quantile(delta,[.025,.975]).tolist()


def run(train_path, source, out, freeze_receipt):
    if (out/'metrics.json').exists():
        raise ValueError('A terminal result already exists; no automatic rerun')
    if not freeze_receipt.is_file():
        raise ValueError('Missing pre-execution source freeze receipt')
    frozen=json.loads(freeze_receipt.read_text())
    if frozen.get('runner_sha256')!=sha(Path(__file__)) or frozen.get('style_sha256')!=STYLE_SHA:
        raise ValueError('Source differs from pre-execution freeze')
    started=time.perf_counter()
    train,extra,receipt=preflight(train_path,source,out)
    if frozen.get('candidate_manifest_sha256')!=receipt['candidate_manifest_sha256']:
        raise ValueError('Admitted row manifest differs from freeze')
    labels=sorted(train.model.unique());n=len(train)
    control=np.empty(n,dtype=object);candidate=np.empty(n,dtype=object)
    folds=np.full(n,-1);margins={a:np.empty((n,len(labels))) for a in ['control','candidate']}
    fold_results=[];all_warnings=[]
    for fold,(tr,va) in enumerate(GroupKFold(5).split(train.text,train.model,train.prompt_id),1):
        assert set(train.prompt_id.iloc[tr]).isdisjoint(train.prompt_id.iloc[va])
        assert set(train.model.iloc[tr])==set(labels)==set(train.model.iloc[va])
        tick=time.perf_counter();mapping=features()
        x=mapping.fit_transform(train.text.iloc[tr],train.model.iloc[tr])
        v=mapping.transform(train.text.iloc[va]);e=mapping.transform(extra.text)
        assert mapping.transformer_list[1][1].named_steps['extract'].get_feature_names_out().size==111
        base_weight,extra_weight=weights(train.model.iloc[tr].to_numpy(),extra.model.to_numpy(),labels)
        result={'fold':fold,'official_training_rows':len(tr),'validation_rows':len(va),
                'external_rows':len(extra),'validation_prompts':int(train.prompt_id.iloc[va].nunique()),
                'feature_seconds':time.perf_counter()-tick}
        for arm in ['control','candidate']:
            estimator=LinearSVC(C=1.,class_weight=None,random_state=SEED,dual=True,max_iter=10000,tol=.0001)
            arm_x=x if arm=='control' else vstack([x,e],format='csr')
            arm_y=train.model.iloc[tr].to_numpy() if arm=='control' else np.concatenate([train.model.iloc[tr],extra.model])
            arm_w=base_weight if arm=='control' else np.concatenate([base_weight,extra_weight])
            tick=time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always',ConvergenceWarning)
                estimator.fit(arm_x,arm_y,sample_weight=arm_w)
            all_warnings += [{'fold':fold,'arm':arm,'message':str(w.message)} for w in caught]
            assert list(estimator.classes_)==labels
            predictions=estimator.predict(v)
            (control if arm=='control' else candidate)[va]=predictions
            margins[arm][va]=estimator.decision_function(v)
            result[arm+'_balanced_accuracy']=float(balanced_accuracy_score(train.model.iloc[va],predictions))
            result[arm+'_fit_seconds']=time.perf_counter()-tick
            result[arm+'_max_iterations']=int(estimator.n_iter_)
        folds[va]=fold;fold_results.append(result)
        pd.DataFrame(fold_results).to_csv(out/'folds.csv',index=False)
        print(json.dumps(result),flush=True)
    assert (folds>0).all() and np.isfinite(margins['control']).all() and np.isfinite(margins['candidate']).all()
    table=train[['id','prompt_id','model']].assign(fold=folds,control=control,candidate=candidate)
    table.to_csv(out/'oof_predictions.csv',index=False)
    np.savez_compressed(out/'development_margins.npz',control=margins['control'],candidate=margins['candidate'],
                        ids=train.id.to_numpy(dtype=str),labels=np.array(labels))
    recalls={}
    scores={}
    for arm in ['control','candidate']:
        cm=confusion_matrix(table.model,table[arm],labels=labels)
        assert cm.sum()==n
        scores[arm]=float(balanced_accuracy_score(table.model,table[arm]))
        recalls[arm]=cm.diagonal()/cm.sum(axis=1)
        assert np.isclose(scores[arm],recalls[arm].mean())
        pd.DataFrame(cm,index=labels,columns=labels).to_csv(out/(arm+'_confusion.csv'))
    per_class=pd.DataFrame({'model':labels,'control_recall':recalls['control'],'candidate_recall':recalls['candidate']})
    per_class['delta_percentage_points']=100*(per_class.candidate_recall-per_class.control_recall)
    per_class.to_csv(out/'per_class_recall.csv',index=False)
    ci=bootstrap(table,labels)
    corrected=int(((table.control!=table.model)&(table.candidate==table.model)).sum())
    broken=int(((table.control==table.model)&(table.candidate!=table.model)).sum())
    passed=scores['candidate']>scores['control'] and ci[0]>0 and not all_warnings
    metrics={'stage':'07A_private_openai_route_pilot','completed_utc':datetime.now(timezone.utc).isoformat(),
             'metric':'official-training development balanced accuracy; not official public score',
             'scores':scores,'delta_percentage_points':100*(scores['candidate']-scores['control']),
             'paired_prompt_bootstrap_95_delta_percentage_points':[100*x for x in ci],
             'bootstrap_draws':10000,'bootstrap_seed':SEED,'corrected':corrected,'broken':broken,
             'changed':int(table.control.ne(table.candidate).sum()),'folds':fold_results,
             'convergence_warnings':all_warnings,'fixed_improvement_screen_passed':passed,
             'decision':'POSITIVE_COMPONENT_SIGNAL_REQUIRES_INDEPENDENCE_AND_ENSEMBLE_REVIEW' if passed else 'RETAIN_CONTROL_NO_AUTOMATIC_VARIANT',
             'full_ensemble_tested':False,'official_submission_created':False,'models_serialized':False,
             'total_seconds':time.perf_counter()-started,'preflight':receipt,'freeze':frozen,
             'artifacts_sha256':{p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name not in ['metrics.json','RUN.log']},
             'limits':['Conditional prompt bootstrap does not include all model-training uncertainty.',
                       'Official prompt wording is unavailable; reconstructed-topic screening cannot certify semantic independence.',
                       'Five GPT labels receive external data; all 18 classes are evaluated.',
                       'Private classifier use scope does not authorize distributing weights or a generative fine-tune.']}
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print(json.dumps({k:v for k,v in metrics.items() if k not in ['preflight','freeze','folds','artifacts_sha256']},indent=2),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for name in ['train','source','out']:
        ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--preflight-only',action='store_true')
    ap.add_argument('--freeze-receipt',type=Path)
    args=ap.parse_args()
    if args.preflight_only:
        _,_,r=preflight(args.train,args.source,args.out);print(json.dumps(r,indent=2),flush=True)
    else:
        if args.freeze_receipt is None:ap.error('--freeze-receipt is required for fitting')
        run(args.train,args.source,args.out,args.freeze_receipt)
