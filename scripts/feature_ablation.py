"""Stage 02: two predeclared feature variants on the baseline's saved folds."""
from pathlib import Path
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
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

SEED = 20261016
SOURCE_SHA256 = '40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707'

def run(data_dir, baseline_dir, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    source = data_dir / 'train.csv'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == SOURCE_SHA256
    train = pd.read_csv(source, dtype=str)
    old = pd.read_csv(baseline_dir/'baseline_oof.csv', dtype={'id':str,'prompt_id':str,'model':str,'prediction':str})
    assert train.id.is_unique and old.id.is_unique
    assert train.id.tolist() == old.id.tolist()
    assert train.prompt_id.tolist() == old.prompt_id.tolist()
    assert train.model.tolist() == old.model.tolist()
    assert set(old.fold) == {1,2,3,4,5}
    assert old.groupby('prompt_id').fold.nunique().eq(1).all()
    assert not train[['id','prompt_id','model','text']].isna().any().any()
    labels = sorted(train.model.unique())
    assert len(labels) == 18
    output = old[['id','prompt_id','model','fold']].copy()
    output['combined'] = old.prediction
    reference = balanced_accuracy_score(train.model, old.prediction)
    meta = json.loads((baseline_dir/'baseline_metrics.json').read_text())
    assert np.isclose(reference, meta['balanced_accuracy'], atol=1e-12)
    assert set(old.prediction) <= set(labels)
    specs = {
        'word_only': {'ngram_range':(1,2),'min_df':2,'sublinear_tf':True,'max_features':80000},
        'character_only': {'analyzer':'char','ngram_range':(3,5),'min_df':3,'sublinear_tf':True,'max_features':120000},
    }
    rows=[]
    fold_rows=[]
    errors=[]
    for name, spec in specs.items():
        predictions=np.empty(len(train), dtype=object)
        run_start=time.perf_counter()
        convergence=[]
        for fold in range(1,6):
            valid=old.fold.eq(fold).to_numpy()
            fit=~valid
            assert set(train.loc[fit,'prompt_id']).isdisjoint(set(train.loc[valid,'prompt_id']))
            assert set(train.loc[fit,'model']) == set(labels)
            assert set(train.loc[valid,'model']) == set(labels)
            model=Pipeline([
                ('features', TfidfVectorizer(**spec)),
                ('classifier', LinearSVC(C=1.0,class_weight='balanced',random_state=SEED,
                                          max_iter=10000,dual=True))])
            started=time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always', ConvergenceWarning)
                model.fit(train.loc[fit,'text'],train.loc[fit,'model'])
            convergence += [str(w.message) for w in caught if issubclass(w.category,ConvergenceWarning)]
            predictions[valid]=model.predict(train.loc[valid,'text'])
            score=float(balanced_accuracy_score(train.loc[valid,'model'],predictions[valid]))
            fold_rows.append({'variant':name,'fold':fold,'balanced_accuracy':score,
                              'rows':int(valid.sum()),'prompts':int(train.loc[valid,'prompt_id'].nunique()),
                              'seconds':time.perf_counter()-started})
            print(f'{name}, fold {fold}: {score:.6f}',flush=True)
        assert all(isinstance(x,str) for x in predictions)
        output[name]=predictions
        pooled=float(balanced_accuracy_score(train.model,predictions))
        rows.append({'variant':name,'balanced_accuracy':pooled,'seconds':time.perf_counter()-run_start,
                     'convergence_warnings':len(convergence)})
        errors += [{'variant':name,'message':w} for w in convergence]
        print(f'{name}, pooled balanced accuracy: {pooled:.6f}',flush=True)
    rows.append({'variant':'combined','balanced_accuracy':float(reference),'seconds':None,
                 'convergence_warnings':len(meta['convergence_warnings'])})
    for fold in range(1,6):
        valid=old.fold.eq(fold)
        fold_rows.append({'variant':'combined','fold':fold,
                          'balanced_accuracy':float(balanced_accuracy_score(train.loc[valid,'model'],old.loc[valid,'prediction'])),
                          'rows':int(valid.sum()),'prompts':int(train.loc[valid,'prompt_id'].nunique()),'seconds':None})
    per_class=[]
    for name in ['word_only','character_only','combined']:
        cm=confusion_matrix(train.model,output[name],labels=labels)
        recalls=cm.diagonal()/cm.sum(axis=1)
        measured=next(r['balanced_accuracy'] for r in rows if r['variant']==name)
        assert np.isclose(recalls.mean(),measured,atol=1e-12)
        for label, recall, support in zip(labels,recalls,cm.sum(axis=1)):
            per_class.append({'variant':name,'model':label,'recall':float(recall),'support':int(support)})
    # Paired prompt-cluster bootstrap, conditional on these already-fitted OOF predictions.
    prompts=sorted(train.prompt_id.unique())
    group_map={x:i for i,x in enumerate(prompts)}
    label_map={x:i for i,x in enumerate(labels)}
    gi=train.prompt_id.map(group_map).to_numpy()
    yi=train.model.map(label_map).to_numpy()
    totals=np.zeros((len(prompts),len(labels)),dtype=np.int64)
    np.add.at(totals,(gi,yi),1)
    rng=np.random.default_rng(SEED)
    weights=rng.multinomial(len(prompts),np.full(len(prompts),1/len(prompts)),size=2000)
    denominators=weights@totals
    assert (denominators>0).all()
    bootstrap={}
    for name in ['word_only','character_only','combined']:
        correct=np.zeros_like(totals)
        np.add.at(correct,(gi,yi),(output[name].to_numpy()==train.model.to_numpy()).astype(int))
        bootstrap[name]=((weights@correct)/denominators).mean(axis=1)
    intervals={}
    for name in ['word_only','character_only']:
        delta=bootstrap[name]-bootstrap['combined']
        intervals[name]={'delta_vs_combined':next(r['balanced_accuracy'] for r in rows if r['variant']==name)-float(reference),
                         'paired_prompt_bootstrap_95_percentile':np.quantile(delta,[.025,.975]).tolist()}
    summary=pd.DataFrame(rows).sort_values('balanced_accuracy',ascending=False)
    summary['delta_vs_combined']=summary.balanced_accuracy-reference
    summary.to_csv(out_dir/'comparison.csv',index=False)
    pd.DataFrame(fold_rows).to_csv(out_dir/'fold_scores.csv',index=False)
    pd.DataFrame(per_class).to_csv(out_dir/'per_model_recall.csv',index=False)
    output.to_csv(out_dir/'oof_predictions.csv',index=False)
    result={'stage':'02_feature_comparison','source_sha256':SOURCE_SHA256,
        'baseline_predictions_sha256':hashlib.sha256((baseline_dir/'baseline_oof.csv').read_bytes()).hexdigest(),
        'plan_commit':'aafa46b8ab9fdef40bb4f88062414524f27a5f8b',
        'comparisons':rows,'paired_intervals':intervals,'warnings':errors,
        'bootstrap':{'resamples':2000,'unit':'prompt_id','seed':SEED,
                     'limitation':'Conditional on fixed OOF predictions; does not measure refitting or model-selection uncertainty.'},
        'environment':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
                       'scipy':scipy.__version__,'sklearn':sklearn.__version__},
        'status':'Development comparisons only. No Kaggle submission or paid API usage.'}
    (out_dir/'metrics.json').write_text(json.dumps(result,indent=2))
    print('\nAll variants:',flush=True)
    print(summary.to_string(index=False),flush=True)
    print('Paired intervals:',json.dumps(intervals),flush=True)
    assert not errors, 'Convergence issue: report this stage as incomplete, without changing parameters.'

if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-dir',type=Path,default=root/'data')
    parser.add_argument('--baseline-dir',type=Path,default=root/'results/baseline_01')
    parser.add_argument('--out-dir',type=Path,default=root/'results/stage_02')
    args=parser.parse_args()
    run(args.data_dir,args.baseline_dir,args.out_dir)
