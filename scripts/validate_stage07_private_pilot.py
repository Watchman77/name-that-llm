"""Recalculate Stage 07A metrics from private row-level predictions.

Uses an independent numpy/count formulation, does not refit models, and exports
aggregate checks only. Do not publish the row-level official prediction table.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def validate(out, official_path):
    metrics=json.loads((out/'metrics.json').read_text())
    table=pd.read_csv(out/'oof_predictions.csv',keep_default_na=False)
    official=pd.read_csv(official_path,keep_default_na=False)
    assert hashlib.sha256(official_path.read_bytes()).hexdigest()==metrics['preflight']['official_train_sha256']
    pd.testing.assert_frame_equal(table[['id','prompt_id','model']],official[['id','prompt_id','model']])
    assert table.id.is_unique and len(table)==2699 and table.prompt_id.nunique()==150
    labels=sorted(official.model.unique());groups=sorted(official.prompt_id.unique())
    expected=np.full(len(table),-1)
    for fold,(tr,va) in enumerate(GroupKFold(5).split(official.text,official.model,official.prompt_id),1):
        assert set(official.prompt_id.iloc[tr]).isdisjoint(official.prompt_id.iloc[va])
        expected[va]=fold
    assert np.array_equal(table.fold.to_numpy(),expected)
    scores={arm:float(np.mean([table.loc[table.model.eq(c),arm].eq(c).mean() for c in labels]))
            for arm in ['control','candidate']}
    for arm in scores:
        assert abs(scores[arm]-metrics['scores'][arm])<1e-12
    by_prompt=[]
    for g in groups:
        s=table[table.prompt_id.eq(g)]
        row=[]
        for label in labels:
            t=s[s.model.eq(label)]
            row.extend([len(t),int(t.control.eq(label).sum()),int(t.candidate.eq(label).sum())])
        by_prompt.append(row)
    sufficient=np.asarray(by_prompt).reshape(len(groups),len(labels),3)
    rng=np.random.default_rng(20261018);deltas=[]
    for _ in range(10000):
        multiplicity=np.bincount(rng.integers(0,len(groups),size=len(groups)),minlength=len(groups))
        counts=np.tensordot(multiplicity,sufficient,axes=(0,0))
        deltas.append(float(np.mean((counts[:,2]-counts[:,1])/counts[:,0])))
    interval=np.percentile(deltas,[2.5,97.5])*100
    assert np.allclose(interval,metrics['paired_prompt_bootstrap_95_delta_percentage_points'],atol=1e-10)
    corrected=int((table.control.ne(table.model)&table.candidate.eq(table.model)).sum())
    broken=int((table.control.eq(table.model)&table.candidate.ne(table.model)).sum())
    assert corrected==metrics['corrected'] and broken==metrics['broken']
    assert int(table.control.ne(table.candidate).sum())==metrics['changed']
    for record in metrics['folds']:
        s=table[table.fold.eq(record['fold'])]
        for arm in scores:
            value=float(np.mean([s.loc[s.model.eq(c),arm].eq(c).mean() for c in labels]))
            assert abs(value-record[arm+'_balanced_accuracy'])<1e-12
    for name,digest in metrics['artifacts_sha256'].items():
        assert hashlib.sha256((out/name).read_bytes()).hexdigest()==digest,name
    assert not metrics['models_serialized'] and not metrics['official_submission_created']
    assert metrics['fixed_improvement_screen_passed']==bool(scores['candidate']>scores['control'] and interval[0]>0 and not metrics['convergence_warnings'])
    # Aggregate sufficient counts are inspectable without exporting official row labels.
    pd.DataFrame(sufficient.sum(axis=0),index=labels,columns=['validation_support','control_correct','candidate_correct']).to_csv(out/'aggregate_validation_counts.csv')
    receipt={'validation':'PASS','method':'separate count-based metric and bootstrap implementation; no refit',
             'independent_agent_review':False,'official_rows_aligned':2699,'official_prompt_groups':150,
             'fold_assignments_reconciled':True,'all_artifact_hashes_match':True,
             'scores_recalculated':scores,'delta_percentage_points':100*(scores['candidate']-scores['control']),
             'paired_prompt_bootstrap_95_delta_percentage_points':interval.tolist(),
             'corrected':corrected,'broken':broken,'trained_models_distributed':False,
             'official_row_level_prediction_artifacts_published':False,
             'limits':'Validates saved output arithmetic and source fold assignment; does not independently reproduce training or certify semantic prompt independence.'}
    (out/'TERMINAL_VALIDATION.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--official',type=Path,required=True)
    args=ap.parse_args();validate(args.out,args.official)
