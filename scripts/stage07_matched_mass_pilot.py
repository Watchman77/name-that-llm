"""07B: class-mass-matched follow-up. No network, model export or test inference."""
import argparse
import json
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import vstack
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.svm import LinearSVC
import stage07_private_pilot as original

SEED=20261018
DEPENDENCIES=['stage07_private_pilot.py','stage07_corpus_audit.py','style_features.py']


def matched_weights(y,external_y,labels):
    base,extra=original.weights(y,external_y,labels)
    augmented=set(external_y)
    candidate_official=base.copy()
    candidate_official[np.isin(y,list(augmented))]/=1.5
    extra/=1.5
    for label in labels:
        assert np.isclose(base[y==label].sum(),candidate_official[y==label].sum()+extra[external_y==label].sum(),atol=1e-10)
    return base,candidate_official,extra


def preflight(train_path,source,out):
    train,extra,r=original.preflight(train_path,source,out)
    labels=sorted(train.model.unique());records=[]
    for fold,(tr,va) in enumerate(GroupKFold(5).split(train.text,train.model,train.prompt_id),1):
        y=train.model.iloc[tr].to_numpy();ey=extra.model.to_numpy()
        base,official,ext=matched_weights(y,ey,labels)
        for label in labels:
            records.append({'fold':fold,'model':label,'control_official_mass':float(base[y==label].sum()),
                            'candidate_official_mass':float(official[y==label].sum()),
                            'candidate_external_mass':float(ext[ey==label].sum()),
                            'candidate_total_mass':float(official[y==label].sum()+ext[ey==label].sum())})
    pd.DataFrame(records).to_csv(out/'weight_mass_audit.csv',index=False)
    r.update(stage='07B_matched_class_mass_pilot',weight_policy='Per-class objective mass matched. Augmented classes: two-thirds official, one-third external. Other classes unchanged.',
             earlier_validation_reused=True,interpretation='Follow-up diagnostic after observing 07A; not fresh independent validation.')
    (out/'preflight.json').write_text(json.dumps(r,indent=2)+'\n')
    return train,extra,r


def run(train_path,source,out,freeze_path):
    if (out/'metrics.json').exists():raise ValueError('Terminal result exists; no automatic rerun')
    frozen=json.loads(freeze_path.read_text())
    if frozen['runner_sha256']!=original.sha(Path(__file__)):raise ValueError('Runner differs from freeze')
    for name,h in frozen['dependencies_sha256'].items():
        if original.sha(Path(__file__).with_name(name))!=h:raise ValueError('Dependency differs from freeze: '+name)
    started=time.perf_counter();train,extra,receipt=preflight(train_path,source,out)
    if receipt['candidate_manifest_sha256']!=frozen['candidate_manifest_sha256']:raise ValueError('Admission manifest differs')
    labels=sorted(train.model.unique());n=len(train);folds=np.full(n,-1)
    pred={arm:np.empty(n,dtype=object) for arm in ['control','candidate']}
    margins={arm:np.empty((n,len(labels))) for arm in pred}
    all_warnings=[];fold_results=[]
    for fold,(tr,va) in enumerate(GroupKFold(5).split(train.text,train.model,train.prompt_id),1):
        assert set(train.prompt_id.iloc[tr]).isdisjoint(train.prompt_id.iloc[va])
        assert set(train.model.iloc[tr])==set(train.model.iloc[va])==set(labels)
        tick=time.perf_counter();mapping=original.features()
        x=mapping.fit_transform(train.text.iloc[tr],train.model.iloc[tr]);v=mapping.transform(train.text.iloc[va]);e=mapping.transform(extra.text)
        y=train.model.iloc[tr].to_numpy();ey=extra.model.to_numpy()
        control_w,official_w,external_w=matched_weights(y,ey,labels)
        result={'fold':fold,'official_training_rows':len(tr),'validation_rows':len(va),'external_rows':len(extra),
                'validation_prompts':int(train.prompt_id.iloc[va].nunique()),'feature_seconds':time.perf_counter()-tick}
        for arm in pred:
            estimator=LinearSVC(C=1.,class_weight=None,random_state=SEED,dual=True,max_iter=10000,tol=.0001)
            arm_x=x if arm=='control' else vstack([x,e],format='csr')
            arm_y=y if arm=='control' else np.concatenate([y,ey])
            arm_w=control_w if arm=='control' else np.concatenate([official_w,external_w])
            tick=time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always',ConvergenceWarning);estimator.fit(arm_x,arm_y,sample_weight=arm_w)
            all_warnings += [{'fold':fold,'arm':arm,'message':str(w.message)} for w in caught]
            assert list(estimator.classes_)==labels
            pred[arm][va]=estimator.predict(v);margins[arm][va]=estimator.decision_function(v)
            result[arm+'_balanced_accuracy']=float(balanced_accuracy_score(train.model.iloc[va],pred[arm][va]))
            result[arm+'_fit_seconds']=time.perf_counter()-tick;result[arm+'_max_iterations']=int(estimator.n_iter_)
        folds[va]=fold;fold_results.append(result);pd.DataFrame(fold_results).to_csv(out/'folds.csv',index=False)
        print(json.dumps(result),flush=True)
    assert (folds>0).all() and all(np.isfinite(m).all() for m in margins.values())
    table=train[['id','prompt_id','model']].assign(fold=folds,control=pred['control'],candidate=pred['candidate'])
    table.to_csv(out/'oof_predictions.csv',index=False)
    np.savez_compressed(out/'development_margins.npz',**margins,ids=train.id.to_numpy(dtype=str),labels=np.array(labels))
    scores={};recalls={}
    for arm in pred:
        cm=confusion_matrix(table.model,table[arm],labels=labels);assert cm.sum()==n
        scores[arm]=float(balanced_accuracy_score(table.model,table[arm]));recalls[arm]=cm.diagonal()/cm.sum(axis=1)
        assert np.isclose(scores[arm],recalls[arm].mean())
        pd.DataFrame(cm,index=labels,columns=labels).to_csv(out/(arm+'_confusion.csv'))
    per_class=pd.DataFrame({'model':labels,'control_recall':recalls['control'],'candidate_recall':recalls['candidate']})
    per_class['delta_percentage_points']=100*(per_class.candidate_recall-per_class.control_recall)
    per_class.to_csv(out/'per_class_recall.csv',index=False)
    ci=original.bootstrap(table,labels)
    passed=scores['candidate']>scores['control'] and ci[0]>0 and not all_warnings
    metrics={'stage':'07B_matched_class_mass_pilot','completed_utc':datetime.now(timezone.utc).isoformat(),
             'metric':'original-training held-out-prompt development balanced accuracy; not official public score',
             'scores':scores,'delta_percentage_points':100*(scores['candidate']-scores['control']),
             'paired_prompt_bootstrap_95_delta_percentage_points':[100*x for x in ci],
             'bootstrap_draws':10000,'bootstrap_seed':SEED,
             'corrected':int((table.control.ne(table.model)&table.candidate.eq(table.model)).sum()),
             'broken':int((table.control.eq(table.model)&table.candidate.ne(table.model)).sum()),
             'changed':int(table.control.ne(table.candidate).sum()),'folds':fold_results,
             'convergence_warnings':all_warnings,'fixed_improvement_screen_passed':passed,
             'decision':'PROVISIONAL_MATCHED_MASS_SIGNAL_REQUIRES_FRESH_VALIDATION' if passed else 'RETAIN_CONTROL_NO_AUTOMATIC_VARIANT',
             'full_ensemble_tested':False,'official_submission_created':False,'models_serialized':False,
             'total_seconds':time.perf_counter()-started,'preflight':receipt,'freeze':frozen,
             'artifacts_sha256':{p.name:original.sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name not in ['metrics.json','RUN.log']},
             'limits':['Designed after observing 07A using the same official folds; not an independent confirmation.',
                       'Prompt bootstrap is conditional on fitted folds, not all training uncertainty.',
                       'Official prompt wording is unavailable; reconstruction screening cannot certify semantic independence.',
                       'Only the five GPT labels receive external data, with class mass now matched.',
                       'Undistributed-classifier use assessment does not authorize publishing weights or a generative fine-tune.']}
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print(json.dumps({k:v for k,v in metrics.items() if k not in ['preflight','freeze','folds','artifacts_sha256']},indent=2),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for name in ['train','source','out']:ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--preflight-only',action='store_true');ap.add_argument('--freeze-receipt',type=Path)
    args=ap.parse_args()
    if args.preflight_only:
        _,_,r=preflight(args.train,args.source,args.out);print(json.dumps(r,indent=2))
    else:
        if args.freeze_receipt is None:ap.error('--freeze-receipt required')
        run(args.train,args.source,args.out,args.freeze_receipt)
