"""Independently reconcile printed Stage06E terminal evidence without model calls.

This validates source-bound log receipts and aggregates. It cannot replace an
independent raw prediction/cache/paired-bootstrap audit when NPZ is unavailable.
"""
import hashlib
import io
import itertools
import json
from pathlib import Path
import re
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
TRAIN=ROOT.parent/'name-that-llm/data/train.csv'
TRAIN_SHA='40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707'
SCENARIOS=(1,3,6)
ENDPOINTS=('raw_control','raw_candidate','balanced_control','balanced_candidate')
WEIGHTS={'stylegram':1.516,'lora':.177,'lora2':.113,'tokattn':.153,'tokattn_deb':.185,'deb_plain':.05,'deb_center':.25}

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def one_marker(lines,name):
    values=[]
    for line in lines:
        # Kaggle log row prefix, followed by exactly one emitted marker.
        match=re.match(r'^\s*\d+(?:\.\d+)?s\s+\d+\s+'+re.escape(name)+r'(?:\s+(.*))?$',line)
        if match:values.append(match.group(1))
    assert len(values)==1,(name,'expected exactly one marker',len(values))
    return json.loads(values[0]) if values[0] else None

def assert_close(a,b,atol=1e-12):np.testing.assert_allclose(a,b,rtol=0,atol=atol)
def fold_correct_options(score,support):
    """Integer count feasibility, not reconstruction of per-fold predictions."""
    support=np.asarray(support,int)
    if np.all(support==support[0]):
        total=score*len(support)*support[0]
        assert abs(total-round(total))<1e-8,('fold balanced accuracy not integer-feasible',score)
        return [dict(total_correct=int(round(total)),minority_correct=None)]
    unique=np.unique(support)
    assert np.array_equal(unique,[29,30]) and np.sum(support==29)==1
    possibilities=[]
    for minority in range(30):
        majority=(score*len(support)-minority/29)*30
        if abs(majority-round(majority))<1e-8 and 0<=round(majority)<=17*30:
            possibilities.append(dict(total_correct=int(round(majority))+minority,minority_correct=minority))
    assert possibilities,('fold balanced accuracy not integer-feasible',score)
    return possibilities


def validate(log_path):
    text=Path(log_path).read_text();lines=text.splitlines()
    assert digest(TRAIN)==TRAIN_SHA
    train=pd.read_csv(TRAIN,dtype=str)
    classes=np.sort(train.model.unique());supports=train.groupby('model').size().reindex(classes)
    foldmap={'p000':4,'p001':3,'p002':5};foldmap.update({f'p{i:03d}':5-i%5 for i in range(3,150)})
    folds=train.prompt_id.map(foldmap).to_numpy()
    fold_support={f:pd.crosstab(pd.Series(folds,name='fold'),train.model).loc[f].reindex(classes).to_numpy() for f in range(1,6)}
    assert sorted(supports.tolist())==[149]+[150]*17
    assert not re.search(r'\bSTAGE06E_(?:PIPELINE_FAILED|CACHED_ENSEMBLE_FAILED)\b',text)
    style=one_marker(lines,'STYLE_TRANSFER_COMPLETE')
    static=one_marker(lines,'STAGE06E_STATIC_MEMBERS_COMPLETE')
    gate=one_marker(lines,'STAGE06E_CACHE_GATE')
    manifest=one_marker(lines,'STAGE06E_CACHE_MANIFEST')
    historical=one_marker(lines,'STAGE06E_HISTORICAL_COMPARISON')
    consume=one_marker(lines,'STAGE06E_CACHE_CONSUMPTION_GATE_PASSED')
    result=one_marker(lines,'ENSEMBLE_RESULT')
    perclass=pd.read_csv(io.StringIO(one_marker(lines,'ENSEMBLE_PER_CLASS_CSV')))
    foldcsv=pd.read_csv(io.StringIO(one_marker(lines,'ENSEMBLE_FOLD_SCORES_CSV')))
    paired=pd.read_csv(io.StringIO(one_marker(lines,'ENSEMBLE_PAIRED_CHANGES_CSV')))
    one_marker(lines,'STAGE06E_PIPELINE_COMPLETE')
    assert result['status']=='COMPLETE_CANONICAL_CPU_FIXED_AVAILABLE_COMPONENT_PROXY'
    assert result['primary_endpoint']=='raw_fixed_rowwise_fusion_mean_balanced_accuracy'
    assert result['fixed_weights']==WEIGHTS and not result['weight_search_performed']
    assert not result['component_selection_performed'] and not result['official_submission_performed']
    assert not result['paid_services_used'] and not result['new_neural_training']
    assert result['model_fits_in_consumer']==result['model_inference_calls_in_consumer']==result['group_recovery_calls_in_consumer']==0
    assert result['rows']==2699 and result['prompts']==150 and result['classes']==classes.tolist()
    assert result['historical_Stage06D_reproduction_passed'] is False
    assert consume['historical_Stage06D_reproduction_passed'] is False
    assert static['model_fits']==0 and static['input_files_audited']==47
    assert static['members']==['stylegram','lora','lora2','tokattn','tokattn_deb']
    assert style['raw_probability_sha256']=='5e2e05577b7dc632f38f67d47958d7f3d069e8b758ead4664050f0d984803bad'
    assert style['entire_probability_array_bitwise_preserved'] and style['new_model_fits']==0
    assert gate['status']=='PASS' and gate['fitted_heads']==15
    assert gate['all_saved_arrays_exact_readback'] and gate['component_predictions_exact'] and gate['component_scores_atol_1e_12']
    assert gate['warnings_non_deprecation']==0 and gate['historical_comparison_is_separate']
    assert manifest['complete'] and manifest['historical_comparison_is_separate']
    assert manifest['cache_sha256']==gate['canonical_cache_sha256']==consume['canonical']['cache_sha256']
    assert static['cache_sha256']==consume['static']['cache_sha256']
    assert consume['all_12_same_run_component_arrays_exact'] and consume['model_fits']==consume['model_inference_calls']==consume['group_recovery_calls']==0
    assert result['canonical_cpu_cache']==consume
    assert len(manifest['fit_parameters'])==15
    assert set(manifest['fit_parameters'])=={f'head_f{f}_{head}__' for f in range(5) for head in ('plain','control','candidate')}
    for p in manifest['fit_parameters'].values():
        assert p['classifier']['C']==.005 and p['classifier']['max_iter']==5000
    completed=[]
    for line in lines:
        m=re.match(r'^\s*\d+(?:\.\d+)?s\s+\d+\s+CENTERING_FOLD_COMPLETE\s+(\d+)\s+seconds\s+([0-9.]+)$',line)
        if m:completed.append((int(m.group(1)),float(m.group(2))))
    assert [v[0] for v in completed]==[1,2,3,4,5]
    runtime=manifest['runtime']
    assert runtime['python']=='3.13.15'
    assert runtime['packages']=={'numpy':'2.1.3','pandas':'2.3.3','scipy':'1.16.3','scikit-learn':'1.6.1','threadpoolctl':'3.6.0'}
    assert runtime['threadpools'] and all(p['num_threads']==2 for p in runtime['threadpools'])
    assert all(runtime['environment'][k]=='2' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'))
    assert not [w for w in manifest['warning_records'] if w['category']!='DeprecationWarning']
    assert result['all_warning_records']==manifest['warning_records']
    assert manifest['input_hashes']==dict(train_sha256=TRAIN_SHA,deb_cache_sha256='11eed58762ad128f97457dde9cc4ab03c7932e1b4b04845ec6664c834b1b1dda',glm_cache_sha256='2b41585a9482cef825962a8cc096697933c0b1237e8ede0712d1f8ab361d9414')
    packaging=json.loads((ROOT/'results/stage_06e/PACKAGING.json').read_text())
    expected=packaging['source_sha256'].copy();expected['freeze']=packaging['execution_freeze_sha256']
    assert result['expected_sources']==expected
    assert manifest['source_sha256']==expected['producer'] and manifest['capture_helper_sha256']==expected['capture_helper']
    assert manifest['original_source_sha256']==expected['original_C'] and manifest['plan_sha256']==expected['plan']
    assert manifest['freeze_sha256']==expected['freeze']
    assert manifest['input_version_ids']==packaging['input_version_ids']
    assert len(perclass)==216 and len(foldcsv)==60 and len(paired)==6
    correct_totals={};class_rows=[];summaries={}
    for endpoint in ENDPOINTS:
        scores=[];correct_totals[endpoint]=[]
        for idx,size in enumerate(SCENARIOS):
            frame=perclass[(perclass.endpoint==endpoint)&(perclass.scenario==size)].set_index('model').reindex(classes)
            assert len(frame)==18 and frame.notna().all().all()
            assert np.array_equal(frame.support.to_numpy(),supports.to_numpy())
            integer=frame.recall.to_numpy()*supports.to_numpy()
            assert np.max(np.abs(integer-np.rint(integer)))<1e-10
            integer=np.rint(integer).astype(int);assert np.all((integer>=0)&(integer<=supports.to_numpy()))
            score=float(np.mean(integer/supports.to_numpy()))
            assert_close(score,result['scores'][endpoint][idx],1e-15)
            scores.append(score);correct_totals[endpoint].append(int(integer.sum()))
            class_rows.extend(dict(endpoint=endpoint,scenario=size,model=cl,support=int(sup),correct=int(cor),recall=float(cor/sup)) for cl,sup,cor in zip(classes,supports,integer))
        assert_close(np.mean(scores),result['mean_scores'][endpoint],1e-15)
        summaries[endpoint]=dict(scenario_scores=scores,mean_score=float(np.mean(scores)),correct_totals=correct_totals[endpoint])
    fold_checks=[]
    for endpoint in ENDPOINTS:
        for idx,size in enumerate(SCENARIOS):
            frame=foldcsv[(foldcsv.endpoint==endpoint)&(foldcsv.scenario==size)].sort_values('fold')
            assert frame.fold.tolist()==[1,2,3,4,5]
            options=[]
            for row in frame.itertuples():
                assert row.rows==int((folds==row.fold).sum())
                options.append(fold_correct_options(row.balanced_accuracy,fold_support[row.fold]))
            compatible=[comb for comb in itertools.product(*options) if sum(v['total_correct'] for v in comb)==correct_totals[endpoint][idx]]
            assert compatible,('Fold score integer totals incompatible with class recalls',endpoint,size)
            fold_checks.append(dict(endpoint=endpoint,scenario=size,integer_count_feasible=True,compatible_fold_total_options=len(compatible)))
    paired_checks=[]
    for endpoint in ('raw','balanced'):
        diff=np.asarray(summaries[endpoint+'_candidate']['scenario_scores'])-np.asarray(summaries[endpoint+'_control']['scenario_scores'])
        interval=result['paired_comparison'][endpoint]
        assert_close(interval['scenario_gains'],diff,1e-15)
        assert_close(interval['mean_gain'],diff.mean(),1e-15)
        assert interval['bootstrap_draws']==3000 and interval['bootstrap_seed']==20261017
        assert len(interval['conditional_prompt_95_interval'])==2
        assert interval['conditional_prompt_95_interval'][0]<=interval['conditional_prompt_95_interval'][1]
        for idx,size in enumerate(SCENARIOS):
            row=paired[(paired.endpoint==endpoint)&(paired.scenario==size)]
            assert len(row)==1;row=row.iloc[0]
            for col in ('changed','corrected','broken','changed_both_wrong','eligible_rows'):
                assert int(row[col])==row[col] and 0<=row[col]<=2699
            delta=correct_totals[endpoint+'_candidate'][idx]-correct_totals[endpoint+'_control'][idx]
            assert int(row.corrected)-int(row.broken)==delta
            assert int(row.changed_both_wrong)==int(row.changed)-int(row.corrected)-int(row.broken)
            recorded=next(r for r in result['paired_counts'] if r['endpoint']==endpoint and r['scenario']==size)
            assert all(recorded[k]==row[k] for k in recorded)
            paired_checks.append(dict(endpoint=endpoint,scenario=size,corrected=int(row.corrected),broken=int(row.broken),net_correct_change=delta))
    # Independently reconcile the three actual aggregate confusion previews.
    output_manifest=json.loads((ROOT/'results/stage_06e/output_manifest.json').read_text())
    preview=json.loads((ROOT/'results/stage_06e/CONFUSION_PREVIEW_PROVENANCE.json').read_text())
    manifest_files={}
    for name in preview['verified_file_hashes']:
        path=ROOT/'results/stage_06e'/name
        spec=output_manifest[name]
        assert path.stat().st_size==spec['bytes'] and digest(path)==spec['sha256'],('Runtime output manifest binding',name)
        manifest_files[name]=dict(bytes=spec['bytes'],sha256=spec['sha256'])
    confusion_checks=[]
    for idx,size in enumerate(SCENARIOS):
        name=f'raw_control_scenario{size}_confusion.csv'
        path=ROOT/'results/stage_06e'/name
        matrix=pd.read_csv(path,index_col=0)
        assert matrix.shape==(18,18) and matrix.index.tolist()==classes.tolist() and matrix.columns.tolist()==classes.tolist()
        values=matrix.to_numpy()
        assert np.issubdtype(values.dtype,np.integer) and np.all(values>=0) and values.sum()==2699
        assert np.array_equal(values.sum(1),supports.to_numpy())
        diagonal=values.diagonal()
        rows=[r for r in class_rows if r['endpoint']=='raw_control' and r['scenario']==size]
        assert np.array_equal(diagonal,np.asarray([r['correct'] for r in rows]))
        score=float(np.mean(diagonal/supports.to_numpy()))
        assert_close(score,result['scores']['raw_control'][idx],1e-15)
        assert int(diagonal.sum())==correct_totals['raw_control'][idx]
        proof=next(row for row in preview['matrices'] if row['file']==name)
        assert proof['sha256']==digest(path) and proof['bytes']==path.stat().st_size
        confusion_checks.append(dict(file=name,rows=18,columns=18,total=int(values.sum()),correct=int(diagonal.sum()),
                                     errors=int(values.sum()-diagonal.sum()),balanced_accuracy=score,sha256=digest(path),
                                     supports_and_diagonal_exactly_reconciled=True,runtime_manifest_hash_exact=True))
    raw=result['paired_comparison']['raw']
    screen=bool(raw['mean_gain']>0 and raw['conditional_prompt_95_interval'][0]>0 and min(raw['scenario_gains'])>=-.005 and not [w for w in manifest['warning_records'] if w['category']!='DeprecationWarning'])
    assert result['primary_raw_screen_pass']==screen
    assert gate['historical_repeated_mismatches']==historical['repeated_mismatches']
    assert gate['historical_unique_affected_ids']==historical['unique_affected_ids']
    assert historical['repeated_comparison_cells']==12*2699
    assert historical['repeated_mismatches']>=historical['unique_affected_ids']>=0
    return dict(status='PASS_INDEPENDENT_PRINTED_TERMINAL_EVIDENCE_RECONCILIATION',
        log_path=str(log_path.relative_to(ROOT)),log_sha256=digest(log_path),train_sha256=TRAIN_SHA,
        run_script_version_id=359514801,development_only=True,official_submission_performed=False,
        fit_completion=dict(ordered_fold_markers=completed,reported_completed_heads=15,archived_head_parameter_entries=15),
        canonical_cache_sha256=manifest['cache_sha256'],static_cache_sha256=static['cache_sha256'],
        source_hashes=result['expected_sources'],runtime=runtime,
        independently_reconciled_scores=summaries,paired_integer_count_checks=paired_checks,
        fold_score_checks=fold_checks,primary_raw_screen_pass=screen,
        reported_conditional_intervals=result['paired_comparison'],
        historical_comparison_summary=historical,
        recovered_confusion_checks=confusion_checks,runtime_output_manifest_file_hashes=manifest_files,
        warning_categories=pd.Series([w['category'] for w in manifest['warning_records']]).value_counts().to_dict(),
        validations=['216 printed per-class recalls reconcile all12 ensemble scores/4means with official149/150supports and integerclass-correctcounts.',
         'Six paired corrected-minus-broken totals reconcile independently recovered global correct-count changes.',
         '60 printed fold scores are integer-count feasible under archived prompt folds and compatible with pooled class-correct totals; per-fold class recalls unavailable.',
         'Five ordered completion markers, fifteen archived parameter entries, passed same-run and cache-consumption gates, source/freeze/plan/runtime/input hashes match printed receipts.',
         'Three complete recovered raw-control confusion matrices independently reconcile class supports, diagonal correct counts and all context balanced accuracies; eight recovered aggregate files match preserved runtime output_manifest hashes and byte sizes.',
         'Fixed raw screen formula recomputed from reconciled gains and reported conditional confidence bounds; balanced results excluded from selection.'],
        actual_raw_npz_available_locally=False,raw_predictions_independently_recomputed=False,bootstrap_draws_independently_recomputed=False,
        limits=['Validation uses complete authoritative Downloadlogs preserved by root; runtime cache-hash and exact-array claims are source-bound run receipts.',
         'Raw canonical/static/ensemble NPZ arrays are not locally available to reviewer; no independent row-level predictions/paired membership, captured log arrays or bootstrapdraw audit is claimed. Aggregate confusion counts were independently reconciled from the three complete recovered previews.',
         'Printed confidence intervals checked for bookkeeping/formula use only; confidence bounds themselves were not independently recomputed.',
         'HistoricalC discrepancy remains unresolved and Stage06D remains failed; this is a new CPU development proxy, not full S02 or an official score.'])

if __name__=='__main__':
    log=ROOT/'results/stage_06e/KAGGLE_V2.log'
    if len(sys.argv)>1:log=Path(sys.argv[1]).resolve()
    output=ROOT/'results/stage_06e/INDEPENDENT_TERMINAL_VALIDATION.json'
    try:result=validate(log)
    except BaseException as error:
        result=dict(status='FAIL_INDEPENDENT_TERMINAL_RECONCILIATION',error_type=type(error).__name__,error=str(error),
                    log_exists=log.exists(),model_fits=0,model_inference_calls=0)
        output.write_text(json.dumps(result,indent=2)+'\n');raise
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status','independently_reconciled_scores','primary_raw_screen_pass')},indent=2))
