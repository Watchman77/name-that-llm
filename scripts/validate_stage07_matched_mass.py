"""Independent aggregate verification of 07B; never publishes row predictions."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from validate_stage07_private_pilot import validate as validate_arithmetic


def validate(out,official,previous):
    validate_arithmetic(out,official)
    frozen=json.loads((out/'SOURCE_FREEZE.json').read_text())
    prior_path=previous/'oof_predictions.csv'
    assert hashlib.sha256(prior_path.read_bytes()).hexdigest()==frozen['prior_oof_sha256']
    a=pd.read_csv(prior_path,keep_default_na=False);b=pd.read_csv(out/'oof_predictions.csv',keep_default_na=False)
    pd.testing.assert_frame_equal(a[['id','prompt_id','model','fold','control']],b[['id','prompt_id','model','fold','control']])
    metrics=json.loads((out/'metrics.json').read_text());mass=pd.read_csv(out/'weight_mass_audit.csv')
    assert len(mass)==90 and not mass.duplicated(['fold','model']).any()
    for r in metrics['folds']:
        group=mass[mass.fold.eq(r['fold'])];target=r['official_training_rows']/18
        assert len(group)==18
        assert np.allclose(group.control_official_mass,target,atol=1e-10)
        assert np.allclose(group.candidate_total_mass,target,atol=1e-10)
        assert np.allclose(group.candidate_official_mass+group.candidate_external_mass,target,atol=1e-10)
        is_gpt=group.model.str.startswith('gpt-')
        assert np.allclose(group.loc[is_gpt,'candidate_external_mass'],target/3,atol=1e-10)
        assert np.allclose(group.loc[is_gpt,'candidate_official_mass'],2*target/3,atol=1e-10)
        assert np.allclose(group.loc[~is_gpt,'candidate_external_mass'],0,atol=1e-10)
        assert np.allclose(group.loc[~is_gpt,'candidate_official_mass'],target,atol=1e-10)
    recall=pd.read_csv(out/'per_class_recall.csv');groups=[]
    for name,mask in [('five_augmented_GPT_labels',recall.model.str.startswith('gpt-')),('other_13_labels',~recall.model.str.startswith('gpt-'))]:
        s=recall[mask];part=b[b.model.isin(set(s.model))]
        groups.append({'group':name,'classes':len(s),'control_macro_recall':float(s.control_recall.mean()),
                       'candidate_macro_recall':float(s.candidate_recall.mean()),'delta_percentage_points':float(s.delta_percentage_points.mean()),
                       'corrected':int((part.control.ne(part.model)&part.candidate.eq(part.model)).sum()),
                       'broken':int((part.control.eq(part.model)&part.candidate.ne(part.model)).sum())})
    pd.DataFrame(groups).to_csv(out/'group_diagnostics.csv',index=False)
    pd.DataFrame({'control_prediction_count':b.control.value_counts(),'candidate_prediction_count':b.candidate.value_counts()}).fillna(0).astype(int).to_csv(out/'prediction_distribution.csv')
    receipt=json.loads((out/'TERMINAL_VALIDATION.json').read_text())
    receipt.update(stage='07B',class_mass_audit_all_90_rows='PASS',control_identical_to_07A_all_2699_predictions=True,
                   prior_oof_hash_matches=True,GPT_prediction_count_control=int(b.control.str.startswith('gpt-').sum()),
                   GPT_prediction_count_candidate=int(b.candidate.str.startswith('gpt-').sum()),earlier_validation_reused=True)
    (out/'TERMINAL_VALIDATION.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2));print(json.dumps(groups,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['out','official','previous']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();validate(a.out,a.official,a.previous)
