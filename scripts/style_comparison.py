"""Stage 03: two fixed, prospectively recorded style-feature candidates."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import platform
import time
import warnings
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from style_features import SurfaceStyleFeatures

SEED=20261016
SOURCE_SHA='40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707'
PLAN_COMMIT='09e2fb41f37fd420caf3184d4d2512b864c0e3e7'

def build_model(kind):
    style=Pipeline([('extract',SurfaceStyleFeatures()),('scale',StandardScaler()),('normalize',Normalizer())])
    if kind=='style_only':
        features=style
    elif kind=='character_plus_style':
        features=FeatureUnion([
            ('character',TfidfVectorizer(analyzer='char',ngram_range=(3,5),min_df=3,
                                        sublinear_tf=True,max_features=120000)),
            ('style',style)])
    else:
        raise ValueError(kind)
    return Pipeline([('features',features),('classifier',LinearSVC(C=1.0,class_weight='balanced',
                        random_state=SEED,max_iter=10000,dual=True))])

def run(data_dir,previous_dir,out_dir):
    out_dir.mkdir(parents=True,exist_ok=True)
    train_path=data_dir/'train.csv'
    assert hashlib.sha256(train_path.read_bytes()).hexdigest()==SOURCE_SHA
    train=pd.read_csv(train_path,dtype=str)
    old=pd.read_csv(previous_dir/'oof_predictions.csv',dtype={'id':str,'prompt_id':str,'model':str})
    assert train.id.is_unique and old.id.is_unique
    assert train[['id','prompt_id','model']].equals(old[['id','prompt_id','model']])
    assert set(old.fold)=={1,2,3,4,5}
    assert old.groupby('prompt_id').fold.nunique().eq(1).all()
    labels=sorted(train.model.unique())
    assert len(labels)==18
    assert not train[['id','prompt_id','text','model']].isna().any().any()
    output=old[['id','prompt_id','model','fold','character_only','combined']].copy()
    names=SurfaceStyleFeatures().get_feature_names_out()
    assert len(names)==111
    (out_dir/'feature_names.json').write_text(json.dumps(names.tolist(),indent=2))
    rows=[]
    fold_rows=[]
    warning_records=[]
    for kind in ['style_only','character_plus_style']:
        started_run=time.perf_counter()
        predictions=np.empty(len(train),dtype=object)
        warn_count=0
        for fold in range(1,6):
            valid=old.fold.eq(fold).to_numpy()
            fit=~valid
            assert set(train.loc[fit,'prompt_id']).isdisjoint(set(train.loc[valid,'prompt_id']))
            assert set(train.loc[fit,'model'])==set(labels) and set(train.loc[valid,'model'])==set(labels)
            model=build_model(kind)
            tick=time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always',ConvergenceWarning)
                model.fit(train.loc[fit,'text'],train.loc[fit,'model'])
            cw=[str(w.message) for w in caught if issubclass(w.category,ConvergenceWarning)]
            warn_count+=len(cw)
            warning_records.extend({'variant':kind,'fold':fold,'message':w} for w in cw)
            predictions[valid]=model.predict(train.loc[valid,'text'])
            score=float(balanced_accuracy_score(train.loc[valid,'model'],predictions[valid]))
            fold_rows.append({'variant':kind,'fold':fold,'balanced_accuracy':score,'rows':int(valid.sum()),
                              'prompts':int(train.loc[valid,'prompt_id'].nunique()),'seconds':time.perf_counter()-tick})
            print(f'{kind}, fold {fold}: {score:.6f}',flush=True)
        assert all(isinstance(x,str) for x in predictions)
        output[kind]=predictions
        score=float(balanced_accuracy_score(train.model,predictions))
        rows.append({'variant':kind,'balanced_accuracy':score,'seconds':time.perf_counter()-started_run,
                     'convergence_warnings':warn_count})
        print(f'{kind}, pooled: {score:.6f}',flush=True)
    for kind in ['character_only','combined']:
        assert set(old[kind])<=set(labels)
        rows.append({'variant':kind,'balanced_accuracy':float(balanced_accuracy_score(train.model,old[kind])),
                     'seconds':None,'convergence_warnings':0})
        for fold in range(1,6):
            valid=old.fold.eq(fold)
            fold_rows.append({'variant':kind,'fold':fold,
                'balanced_accuracy':float(balanced_accuracy_score(train.loc[valid,'model'],old.loc[valid,kind])),
                'rows':int(valid.sum()),'prompts':int(train.loc[valid,'prompt_id'].nunique()),'seconds':None})
    variants=['style_only','character_plus_style','character_only','combined']
    per_class=[]
    pairs=[]
    length_rows=[]
    length_bin=pd.cut(train.text.str.split().str.len(),bins=[-np.inf,80,150,np.inf],
                      labels=['<=80 words','81-150 words','>150 words'])
    for kind in variants:
        cm=confusion_matrix(train.model,output[kind],labels=labels)
        recalls=cm.diagonal()/cm.sum(axis=1)
        assert np.isclose(recalls.mean(),next(r['balanced_accuracy'] for r in rows if r['variant']==kind),atol=1e-12)
        for label,recall,support in zip(labels,recalls,cm.sum(axis=1)):
            per_class.append({'variant':kind,'model':label,'recall':float(recall),'support':int(support)})
        errors=output[output[kind]!=output.model].groupby(['model',kind]).size().reset_index(name='errors')
        errors.columns=['true_model','predicted_model','errors']
        errors['variant']=kind
        pairs.append(errors.sort_values('errors',ascending=False).head(10))
        for bin_name in length_bin.cat.categories:
            selected=length_bin.eq(bin_name)
            present=int(train.loc[selected,'model'].nunique())
            length_rows.append({'variant':kind,'word_count_bin':bin_name,'rows':int(selected.sum()),
                'classes_present':present,'balanced_accuracy_present_classes':float(balanced_accuracy_score(
                    train.loc[selected,'model'],output.loc[selected,kind])) if selected.any() else None})
    prompts=sorted(train.prompt_id.unique())
    gi=train.prompt_id.map({x:i for i,x in enumerate(prompts)}).to_numpy()
    yi=train.model.map({x:i for i,x in enumerate(labels)}).to_numpy()
    totals=np.zeros((len(prompts),len(labels)),dtype=np.int64)
    np.add.at(totals,(gi,yi),1)
    weights=np.random.default_rng(SEED).multinomial(len(prompts),np.full(len(prompts),1/len(prompts)),size=2000)
    denom=weights@totals
    assert (denom>0).all()
    bootstrap={}
    for kind in variants:
        correct=np.zeros_like(totals)
        np.add.at(correct,(gi,yi),(output[kind].to_numpy()==train.model.to_numpy()).astype(int))
        bootstrap[kind]=((weights@correct)/denom).mean(axis=1)
    intervals=[]
    for kind in ['style_only','character_plus_style']:
        for reference in ['character_only','combined']:
            delta=bootstrap[kind]-bootstrap[reference]
            low,high=np.quantile(delta,[.025,.975])
            intervals.append({'candidate':kind,'reference':reference,
                'delta_balanced_accuracy':next(r['balanced_accuracy'] for r in rows if r['variant']==kind)
                    -next(r['balanced_accuracy'] for r in rows if r['variant']==reference),
                'lower_95':float(low),'upper_95':float(high)})
    summary=pd.DataFrame(rows).sort_values('balanced_accuracy',ascending=False)
    summary.to_csv(out_dir/'comparison.csv',index=False)
    pd.DataFrame(fold_rows).to_csv(out_dir/'fold_scores.csv',index=False)
    pd.DataFrame(per_class).to_csv(out_dir/'per_model_recall.csv',index=False)
    pd.concat(pairs,ignore_index=True).to_csv(out_dir/'confusion_pairs.csv',index=False)
    pd.DataFrame(length_rows).to_csv(out_dir/'length_diagnostics.csv',index=False)
    output.to_csv(out_dir/'oof_predictions.csv',index=False)
    pd.DataFrame(intervals).to_csv(out_dir/'paired_differences.csv',index=False)
    result={'stage':'03_style_features','plan_commit':PLAN_COMMIT,'source_sha256':SOURCE_SHA,
        'previous_predictions_sha256':hashlib.sha256((previous_dir/'oof_predictions.csv').read_bytes()).hexdigest(),
        'code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in [Path(__file__),Path(__file__).parent/'style_features.py']},
        'executed_utc':datetime.now(timezone.utc).isoformat(),'comparisons':rows,'intervals':intervals,
        'feature_count':len(names),'warnings':warning_records,
        'environment':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
                       'scipy':scipy.__version__,'sklearn':sklearn.__version__},
        'bootstrap':{'resamples':2000,'seed':SEED,'unit':'prompt_id',
                     'limitation':'Conditional on fitted OOF predictions; no refitting/selection uncertainty or multiplicity correction.'},
        'status':'Development only; no Kaggle submission, paid APIs or new data.'}
    (out_dir/'metrics.json').write_text(json.dumps(result,indent=2))
    print(summary.to_string(index=False),flush=True)
    print('Paired differences:',json.dumps(intervals),flush=True)
    assert not warning_records, 'Convergence issue: retain and report rather than changing settings.'

if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-dir',type=Path,default=root/'data')
    parser.add_argument('--previous-dir',type=Path,default=root/'results/stage_02')
    parser.add_argument('--out-dir',type=Path,default=root/'results/stage_03')
    args=parser.parse_args()
    run(args.data_dir,args.previous_dir,args.out_dir)
