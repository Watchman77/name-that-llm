"""07C pure consumer: one fixed hard-vote addition to the recorded 06E proxy.

No fitting, calibration, inference, regrouping, weight search or submission.
SVM native margins are never interpreted as probabilities.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

CACHE_SHA='9348eeed149ad48bdd828d8055e945f57e384d8917c654125c2ebbe8d4b99519'
OOF_SHA='72d7d43119367ef100511d0f8174b9483ec86bf28bfc7b7e61d8fb599ee1209d'
TRAIN_SHA='40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707'
BASE_SCORES=np.array([.526380810340542,.553067362664678,.5649192145165299])
WEIGHT=.05


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def vote_mix(probability,prediction):
    assert probability.ndim==3 and prediction.shape==(probability.shape[1],)
    assert np.isfinite(probability).all() and (probability>=0).all()
    assert np.allclose(probability.sum(2),1.,atol=1e-10)
    assert prediction.dtype.kind in 'iu' and prediction.min()>=0 and prediction.max()<probability.shape[2]
    votes=np.eye(probability.shape[2])[prediction]
    return (1-WEIGHT)*probability+WEIGHT*votes[None]


def interval(y,groups,first,second,classes,draws=10000):
    # Mean of context-specific macro recalls, jointly paired by original prompt.
    difference=(second==y[None]).astype(float)-(first==y[None]).astype(float)
    by_group=[];denominator=[]
    for group in np.unique(groups):
        mask=groups==group
        by_group.append(np.bincount(y[mask],weights=difference[:,mask].mean(0),minlength=classes))
        denominator.append(np.bincount(y[mask],minlength=classes))
    numerator=np.asarray(by_group);denominator=np.asarray(denominator)
    rng=np.random.default_rng(20261018);values=[]
    for _ in range(draws):
        chosen=rng.integers(0,len(numerator),len(numerator))
        n=numerator[chosen].sum(0);d=denominator[chosen].sum(0)
        assert (d>0).all();values.append(float(np.mean(n/d)))
    return np.quantile(values,[.025,.975]).tolist()


def run(cache,oof,official,out,freeze_path):
    frozen=json.loads(freeze_path.read_text())
    if frozen['consumer_sha256']!=sha(Path(__file__)):raise ValueError('Consumer differs from freeze')
    if (out/'result.json').exists():raise ValueError('Terminal result exists; no automatic variant')
    for path,expected in [(cache,CACHE_SHA),(oof,OOF_SHA),(official,TRAIN_SHA)]:
        if sha(path)!=expected:raise ValueError('Input hash mismatch: '+path.name)
    train=pd.read_csv(official,keep_default_na=False,dtype=str)
    component=pd.read_csv(oof,keep_default_na=False)
    pd.testing.assert_frame_equal(component[['id','prompt_id','model']],train[['id','prompt_id','model']])
    with np.load(cache,allow_pickle=False) as z:
        classes=z['classes'];ids=z['ids'];y=z['y'];groups=z['prompt_ids'];fold=z['fold'];scenarios=z['scenarios']
        p=z['raw_probability_control'];cached_prediction=z['prediction_raw_control']
        assert z['train_sha256'].item()==TRAIN_SHA
        assert np.array_equal(ids,train.id) and np.array_equal(groups,train.prompt_id)
        assert np.array_equal(classes,np.sort(train.model.unique())) and len(classes)==18
        assert np.array_equal(y,np.searchsorted(classes,train.model.to_numpy()))
        assert np.array_equal(fold,component.fold) and np.array_equal(scenarios,[1,3,6])
        assert p.shape==(3,2699,18) and np.array_equal(cached_prediction,p.argmax(2))
        for name in ['stylegram','lora','lora2','tokattn','tokattn_deb']:
            member=z['fold_'+name]
            assert member.shape==(2699,)
            assert all(len(np.unique(member[groups==g]))==1 for g in np.unique(groups))
    assert len(component)==2699 and component.id.is_unique
    assert set(component.control)<=set(classes) and set(component.candidate)<=set(classes)
    votes={arm:np.searchsorted(classes,component[arm].to_numpy()) for arm in ['control','candidate']}
    predictions={'base':p.argmax(2)}
    predictions.update({arm:vote_mix(p,votes[arm]).argmax(2) for arm in votes})
    scores={arm:[float(balanced_accuracy_score(y,row)) for row in value] for arm,value in predictions.items()}
    assert np.allclose(scores['base'],BASE_SCORES,atol=1e-12)
    changes={};intervals={}
    for baseline in ['control','base']:
        delta=np.array(scores['candidate'])-scores[baseline]
        changes[baseline]={'scenario_delta_pp':(100*delta).tolist(),'mean_delta_pp':float(100*delta.mean())}
        intervals[baseline]=[100*x for x in interval(y,groups,predictions[baseline],predictions['candidate'],18)]
    passed=(all(changes[k]['mean_delta_pp']>0 and intervals[k][0]>0 for k in changes)
            and min(changes['base']['scenario_delta_pp'])>=-.5)
    out.mkdir(parents=True,exist_ok=True)
    per_class=[];pairs=[]
    for scenario_index,size in enumerate(scenarios):
        for label_index,label in enumerate(classes):
            mask=y==label_index
            per_class.append({'scenario':int(size),'model':label,**{k:float((v[scenario_index,mask]==y[mask]).mean()) for k,v in predictions.items()}})
        for baseline in ['base','control']:
            a=predictions[baseline][scenario_index];b=predictions['candidate'][scenario_index]
            pairs.append({'scenario':int(size),'baseline':baseline,'corrected':int(((a!=y)&(b==y)).sum()),
                          'broken':int(((a==y)&(b!=y)).sum()),'changed':int((a!=b).sum())})
    pd.DataFrame(per_class).to_csv(out/'per_class_recall.csv',index=False)
    pd.DataFrame(pairs).to_csv(out/'paired_counts.csv',index=False)
    result={'stage':'07C_fixed_vote_proxy_marginal_test','cache_sha256':CACHE_SHA,'oof_sha256':OOF_SHA,
            'proxy':'seven-member Stage 06E V2 original-control raw posterior; NOT full S02',
            'fixed_vote_weight':WEIGHT,'scenarios':[1,3,6],'scores':scores,'comparisons':changes,
            'conditional_prompt_95_delta_pp':intervals,'bootstrap_draws':10000,'bootstrap_seed':20261018,
            'fixed_screen_passed':passed,'decision':'PROVISIONAL_PROXY_SIGNAL_REQUIRES_FULL_REFERENCE_AND_FRESH_VALIDATION' if passed else 'RETAIN_BASE_NO_AUTOMATIC_VARIANT',
            'new_fitting':False,'new_calibration':False,'official_submission':False,'full_S02_tested':False,
            'original_row_predictions_published':False,'freeze':frozen,
            'limits':['Diagnostic hard-vote mixture, not probability calibration or a tuned soft ensemble.',
                      'Development prompts and existing proxy source weights have been reused.',
                      'Contexts repeat the same 2699 passages and are resampled jointly by prompt.',
                      'The source proxy omits active full-S02 representations; no official score claim.']}
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for name in ['cache','oof','official','out','freeze']:ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();run(a.cache,a.oof,a.official,a.out,a.freeze)
