#!/usr/bin/env python3
"""Validate the one fixed Stage 12B SVC-topic intervention; never load models.

Only safe numeric arrays, JSON, CSV and static Python source are read. Reports
contain aggregate checks and hashes, never original test IDs, text or logits.
"""
from __future__ import annotations
import argparse
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import tempfile
import zipfile
import numpy as np
from scipy.special import softmax

LABELS = ["gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra", "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5", "claude-fable-5", "qwen3.8-max", "glm-5.3", "glm-5.3-flash", "kimi-k3"]
PUBLIC_FIXTURE = [('00353fb1c541',321,'0c306cbd5d401399327e4cb46f766928'),('005f5a9a6139',764,'c2853bf2b68bdd808a8a05cec2439b5d'),('0062a53d5487',1077,'f8c583ac9cbc0806bee20f96c9124159')]
DEFAULT_TEMP = '/kaggle/temp/ntl636-fulltext-qwen-query'
DEFAULT_RUNTIME = '/kaggle/temp/ntl636-fulltext-qwen-external3200-runtime'
EXPECTED_ORCHESTRATOR_SHA256 = '6c5cc9d153763729a716fb503726d99cc76ae31db3acc42a84d2991d119a096d'
EXPECTED_COMMON_SHA256 = '7775a24e23b80fbe549848cc029f9919d0baca9ae7b98d4f759db75b7d991721'
ATOL, RTOL = 2e-11, 2e-12


class InterventionValidationError(ValueError):
    """Aggregate check names only; no row contents in error messages."""


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def canonical_python_sha(text):
    return hashlib.sha256((text.rstrip()+'\n').encode()).hexdigest()


def clipped_logp(logits):
    # Match the released reference's finite probability floor exactly.
    return np.log(np.clip(softmax(np.asarray(logits,dtype=np.float64),axis=1),1e-9,1))


def adaptive_svc_topic_correction(total,svc_scores,groups):
    """Replay the unchanged source mathematical function exactly once."""
    scores=np.asarray(svc_scores,dtype=np.float64)
    centered=scores.copy()
    sizes=np.bincount(groups)[groups]
    valid=(sizes>=8)&(sizes<=18)
    for group in np.unique(groups[valid]):
        rows=groups==group
        centered[rows]-=scores[rows].mean(0,keepdims=True)
    delta=clipped_logp(centered)-clipped_logp(scores)
    delta[~valid]=0
    ordered=np.sort(total,axis=1)
    gate=np.clip((2.-(ordered[:,-1]-ordered[:,-2]))/1.6,.15,1.)
    gate[~valid]=0
    return total+.62*delta*gate[:,None]


def function_ast(text,name):
    node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==name)
    if node.body and isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str):node.body=node.body[1:]
    return ast.dump(node,include_attributes=False)


def source_contract(orchestrator,common):
    tree=ast.parse(orchestrator)
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='adaptive_svc_topic_correction']
    if len(calls)!=1:raise InterventionValidationError('source:exactly_one_correction_call')
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    def condition(node,value):
        return isinstance(node,ast.If) and isinstance(node.test,ast.Compare) and isinstance(node.test.left,ast.Name) and node.test.left.id=='family' and len(node.test.comparators)==1 and isinstance(node.test.comparators[0],ast.Constant) and node.test.comparators[0].value==value
    branch=next(i for i,n in enumerate(main.body) if condition(n,'encoder_0627'))
    comp=next(i for i,n in enumerate(main.body) if condition(n,'part_two_complement_atlas_0636'))
    index=next(i for i,n in enumerate(main.body) if any(x is calls[0] for x in ast.walk(n)))
    copy=main.body[branch+1]
    if not (index==branch+2 and index<comp and isinstance(copy,ast.Assign) and isinstance(copy.targets[0],ast.Name) and copy.targets[0].id=='svc_topic_before_total'):
        raise InterventionValidationError('source:after_complete_deberta_before_complement')
    call=calls[0]
    if [ast.dump(x,include_attributes=False) for x in call.args]!=[ast.dump(ast.Name(id=name,ctx=ast.Load()),include_attributes=False) for name in ['total','svc_scores','groups']] or call.keywords:
        raise InterventionValidationError('source:fixed_original_arguments')
    if function_ast(common,'adaptive_svc_topic_correction')!=function_ast(Path(__file__).read_text(),'adaptive_svc_topic_correction'):
        raise InterventionValidationError('source:replay_function_AST_matches')
    if function_ast(common,'clipped_logp')!=function_ast(Path(__file__).read_text(),'clipped_logp'):
        raise InterventionValidationError('source:log_probability_floor_matches')
    return {'exactly_one_correction_call':True,'after_complete_deberta_before_complement':True,'fixed_original_arguments':True,'replay_function_AST_matches':True,'log_probability_floor_matches':True}


def safe_npz(path):
    path=Path(path)
    if not path.is_file():raise InterventionValidationError('array_file:exists')
    with zipfile.ZipFile(path) as archive:
        if sum(x.file_size for x in archive.infolist())>1024**3:raise InterventionValidationError('array_file:bounded_uncompressed_bytes')
    with np.load(path,allow_pickle=False) as loaded:
        return {key:loaded[key].copy() for key in loaded.files}


def validate_intervention(work_dir='/kaggle/working',test_path=None,*,temporary_dir=DEFAULT_TEMP,runtime_dir=DEFAULT_RUNTIME,namespace=None,emit=True):
    namespace={} if namespace is None else namespace
    checks=[]; files={}
    def check(name,condition):
        okay=bool(condition);checks.append({'name':name,'pass':okay})
        if not okay:raise InterventionValidationError(name)
    work=Path(work_dir); temp=Path(temporary_dir); runtime=Path(runtime_dir)
    test_path=Path(test_path or '/kaggle/input/competitions/name-that-llm/test.csv')
    check('test:exists',test_path.is_file())
    with test_path.open(newline='',encoding='utf-8') as f:
        reader=csv.DictReader(f);check('test:columns',reader.fieldnames==['id','text']);rows=list(reader)
    ids=[r['id'] for r in rows];n=len(ids)
    check('test:unique_nonempty_ids',n>0 and all(ids) and len(set(ids))==n)
    preview=namespace.get('preview')
    check('preview:known_boolean',isinstance(preview,(bool,np.bool_)))
    fingerprint=[(r['id'],len(r['text'].encode()),hashlib.blake2s(r['text'].encode(),digest_size=16).hexdigest()) for r in rows]
    check('preview:exact_source_fixture',bool(preview)==(fingerprint==PUBLIC_FIXTURE))
    orchestrator_path=runtime/'run_fulltext_selected_ensemble.py';common_path=runtime/'reference627_common.py'
    check('source:orchestrator_exists',orchestrator_path.is_file());check('source:common_exists',common_path.is_file())
    orchestrator=orchestrator_path.read_text();common=common_path.read_text()
    check('source:fixed_orchestrator_hash',canonical_python_sha(orchestrator)==EXPECTED_ORCHESTRATOR_SHA256)
    check('source:unchanged_common_hash',canonical_python_sha(common)==EXPECTED_COMMON_SHA256)
    for name,value in source_contract(orchestrator,common).items():check('source:'+name,value)
    files['orchestrator']={'sha256':sha(orchestrator_path),'canonical_sha256':EXPECTED_ORCHESTRATOR_SHA256}
    evidence=temp/'svc-topic-correction-evidence.npz'
    if preview:
        previous=namespace.get('validation_report',{})
        check('preview:original_output_guard_passed',previous.get('status')=='FORMAT_ONLY_PREVIEW' and previous.get('all_checks_pass') is True and previous.get('rows')==3)
        check('preview:intervention_evidence_absent',not evidence.exists())
        result={'stage':'12B','status':'FORMAT_ONLY_PREVIEW_INTERVENTION_NOT_EXECUTED','rows':n,'intervention_executed':False,'intervention_activation_verified':False,'eligible_rows':None,'hidden_runtime_verified':False,'official_public_score':None,'checks':checks,'files':files,'source_version':362014737,'parent_own_official_score':0.642,'original_ids_text_or_row_logits_exported':False}
    else:
        values=safe_npz(evidence)
        check('evidence:exact_keys',set(values)=={'ids','labels','groups','svc_scores','before_total','after_total'})
        check('evidence:string_axes',values['ids'].dtype.kind in 'US' and values['labels'].dtype.kind in 'US')
        check('evidence:exact_id_axis',values['ids'].shape==(n,) and values['ids'].astype(str).tolist()==ids)
        check('evidence:exact_class_axis',values['labels'].shape==(18,) and values['labels'].astype(str).tolist()==LABELS)
        groups=values['groups']
        check('evidence:bounded_integer_groups',groups.shape==(n,) and groups.dtype.kind in 'iu' and int(groups.min())>=1 and int(groups.max())<=n)
        svc,before,after=[values[k] for k in ['svc_scores','before_total','after_total']]
        for key,array in [('svc_scores',svc),('before_total',before),('after_total',after)]:
            check('evidence:'+key+':shape_finite',array.shape==(n,18) and array.dtype.kind=='f' and np.isfinite(array).all())
        def member_logits(name):
            v=safe_npz(temp/name)
            check(name+':axes',set(['ids','labels','logits'])<=set(v) and v['ids'].shape==(n,) and v['ids'].astype(str).tolist()==ids and v['labels'].shape==(18,) and len(set(v['labels'].astype(str)))==18 and set(v['labels'].astype(str))==set(LABELS))
            logits=v['logits'][:,[v['labels'].astype(str).tolist().index(label) for label in LABELS]]
            check(name+':finite',logits.shape==(n,18) and logits.dtype.kind=='f' and np.isfinite(logits).all())
            return logits
        expected_before=clipped_logp(svc)+member_logits('decoders.npz')
        expected_before+=.25*clipped_logp(member_logits('deberta.npz'))
        check('intervention:before_matches_complete_reference_sum',np.allclose(before,expected_before,atol=ATOL,rtol=RTOL))
        replay=adaptive_svc_topic_correction(before,svc,groups)
        check('intervention:single_application_replay_matches',np.allclose(after,replay,atol=ATOL,rtol=RTOL))
        sizes=np.bincount(groups)[groups];eligible=(sizes>=8)&(sizes<=18)
        check('intervention:ineligible_rows_exactly_unchanged',np.array_equal(after[~eligible],before[~eligible]))
        prior=safe_npz(temp/'before_atlas.npz')
        check('mixture:prior_axes',prior['ids'].shape==(n,) and prior['ids'].astype(str).tolist()==ids and prior['labels'].astype(str).tolist()==LABELS)
        metadata=json.loads((temp/'comp.npy.json').read_text())
        check('mixture:complement_axes',metadata['ids']==ids and len(metadata['labels'])==18 and set(metadata['labels'])==set(LABELS))
        comp=np.load(temp/'comp.npy',allow_pickle=False)
        check('mixture:complement_shape_finite',comp.shape==(n,18) and comp.dtype.kind=='f' and np.isfinite(comp).all())
        comp=comp[:,[metadata['labels'].index(label) for label in LABELS]]
        expected_prior=.85*softmax(after,axis=1)+.15*softmax(comp,axis=1)
        check('mixture:corrected_total_feeds_actual_pre_ATLAS_prior',prior['probabilities'].shape==(n,18) and np.isfinite(prior['probabilities']).all() and np.allclose(prior['probabilities'],expected_prior,atol=ATOL,rtol=RTOL))
        files[evidence.name]={'bytes':evidence.stat().st_size,'sha256':sha(evidence)}
        changed=np.max(np.abs(after-before),axis=1)>ATOL
        result={'stage':'12B','status':'PASS_FIXED_SVC_INTERVENTION_REPLAY','rows':n,'intervention_executed':True,'intervention_activation_verified':True,'eligible_rows':int(eligible.sum()),'eligible_fraction':float(eligible.mean()),'eligible_clusters':int(sum(8<=count<=18 for count in np.bincount(groups)[1:])),'rows_with_nontrivial_score_change':int(changed.sum()),'maximum_absolute_score_change':float(np.max(np.abs(after-before))),'source_version':362014737,'parent_own_official_score':0.642,'coefficient':.62,'topic_threshold':.85,'checks':checks,'files':files,'official_public_score':None,'accuracy_evaluated':False,'original_ids_text_or_row_logits_exported':False}
    result['check_count']=len(checks);result['all_checks_pass']=all(x['pass'] for x in checks)
    result['limitations']=['Replay and saved-output checks do not establish honest full-S05 OOF accuracy or private leaderboard performance.','The original Stage11 output guard remains separate and unchanged.','The CPU component study gated using SVC scores; this production intervention gates on complete reference-ensemble scores, so a component gain does not predict its official gain.']
    work.mkdir(parents=True,exist_ok=True)
    (work/'stage12b_intervention_validation.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    if emit:
        print('STAGE12B_INTERVENTION_JSON_BEGIN');print(json.dumps(result,indent=2,allow_nan=False));print('STAGE12B_INTERVENTION_JSON_END')
    return result


def self_test():
    from copy import deepcopy
    project=Path(__file__).resolve().parents[1]
    nb=json.loads((project/'notebooks/Stage_12B_SVC_Topic_Correction.ipynb').read_text())
    codes=[''.join(c['source']) if isinstance(c['source'],list) else c['source'] for c in nb['cells'] if c['cell_type']=='code']
    orchestrator=codes[29].split('\n',1)[1];common=codes[4].split('\n',1)[1]
    checks={};checks['source_AST_one_call_and_exact_placement']=all(source_contract(orchestrator,common).values())
    bad=orchestrator.replace('    svc_topic_before_total = total.copy()','    total = adaptive_svc_topic_correction(total, svc_scores, groups)\n    svc_topic_before_total = total.copy()',1)
    try:source_contract(bad,common)
    except InterventionValidationError:checks['reject_source_double_application']=True
    else:checks['reject_source_double_application']=False
    # Analytic golden case: the gate must use complete ensemble margins (.4),
    # not SVC margins (2). Thus corrected class0-minus-class1 is -.84.
    gold_svc=np.zeros((8,18));gold_svc[:,0]=2
    gold_before=np.zeros((8,18));gold_before[:,0]=.4
    gold=adaptive_svc_topic_correction(gold_before,gold_svc,np.ones(8,dtype=int))
    delta=math.log(math.exp(2)+17)-math.log(18)
    checks['analytic_golden_ensemble_gate_and_strength']=np.allclose(gold[:,0],.4+.62*(delta-2),atol=1e-14) and np.allclose(gold[:,1:],.62*delta,atol=1e-14) and np.allclose(gold[:,0]-gold[:,1],-.84,atol=1e-14)
    checks['analytic_golden_changes_prediction']=np.all(gold_before.argmax(1)==0) and np.all(gold.argmax(1)==1)
    checks['golden_second_application_is_distinct']=not np.allclose(adaptive_svc_topic_correction(gold,gold_svc,np.ones(8,dtype=int)),gold)
    def fixture():
        td=tempfile.TemporaryDirectory();root=Path(td.name);work=root/'working';temp=root/'temporary';runtime=root/'runtime'
        for p in [work,temp,runtime]:p.mkdir()
        (runtime/'run_fulltext_selected_ensemble.py').write_text(orchestrator);(runtime/'reference627_common.py').write_text(common)
        groups=np.array([1]*7+[2]*8+[3]*18+[4]*19,dtype=np.int64);n=len(groups);ids=np.asarray([f'synthetic-{i}' for i in range(n)])
        rng=np.random.default_rng(12025);svc=rng.normal(size=(n,18));svc[:,0]+=1.7
        dec=rng.normal(scale=.3,size=(n,18));deb=rng.normal(scale=.4,size=(n,18));comp=rng.normal(scale=.2,size=(n,18))
        before=clipped_logp(svc)+dec;before+=.25*clipped_logp(deb);after=adaptive_svc_topic_correction(before,svc,groups)
        evidence=dict(ids=ids,labels=np.asarray(LABELS),groups=groups,svc_scores=svc,before_total=before,after_total=after)
        np.savez_compressed(temp/'svc-topic-correction-evidence.npz',**evidence)
        for name,logits in [('decoders.npz',dec),('deberta.npz',deb)]:np.savez_compressed(temp/name,ids=ids,labels=np.asarray(LABELS),logits=logits)
        np.save(temp/'comp.npy',comp,allow_pickle=False);(temp/'comp.npy.json').write_text(json.dumps({'ids':ids.tolist(),'labels':LABELS}))
        np.savez_compressed(temp/'before_atlas.npz',ids=ids,labels=np.asarray(LABELS),probabilities=.85*softmax(after,axis=1)+.15*softmax(comp,axis=1))
        with (root/'test.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['id','text']);w.writerows((i,'synthetic fixture passage') for i in ids)
        kwargs=dict(work_dir=work,test_path=root/'test.csv',temporary_dir=temp,runtime_dir=runtime,namespace={'preview':False},emit=False)
        return td,root,evidence,kwargs
    def mutate_npz(path,fun):
        values=safe_npz(path);fun(values);np.savez_compressed(path,**values)
    def exercise(name,mutation=None,reject=False):
        td,root,evidence,kwargs=fixture()
        try:
            if mutation:mutation(root,evidence,kwargs)
            try:r=validate_intervention(**kwargs)
            except (InterventionValidationError,ValueError,KeyError):checks[name]=reject
            else:checks[name]=not reject and r['status']=='PASS_FIXED_SVC_INTERVENTION_REPLAY' and r['eligible_rows']==26
        finally:td.cleanup()
    exercise('real_NPZ_readback_and_full_chain_replay')
    exercise('reject_second_application_arrays',lambda r,e,k:mutate_npz(r/'temporary/svc-topic-correction-evidence.npz',lambda v:v.__setitem__('after_total',adaptive_svc_topic_correction(v['after_total'],v['svc_scores'],v['groups']))),True)
    exercise('reject_wrong_ID_axis',lambda r,e,k:mutate_npz(r/'temporary/svc-topic-correction-evidence.npz',lambda v:v.__setitem__('ids',v['ids'][::-1])),True)
    exercise('reject_wrong_class_axis',lambda r,e,k:mutate_npz(r/'temporary/svc-topic-correction-evidence.npz',lambda v:v.__setitem__('labels',v['labels'][::-1])),True)
    exercise('reject_NaN_evidence',lambda r,e,k:mutate_npz(r/'temporary/svc-topic-correction-evidence.npz',lambda v:v['after_total'].__setitem__((0,0),np.nan)),True)
    exercise('reject_unbounded_group_index',lambda r,e,k:mutate_npz(r/'temporary/svc-topic-correction-evidence.npz',lambda v:v['groups'].__setitem__(0,10**8)),True)
    exercise('reject_modified_ineligible_row',lambda r,e,k:mutate_npz(r/'temporary/svc-topic-correction-evidence.npz',lambda v:v['after_total'].__setitem__((0,0),v['after_total'][0,0]+.01)),True)
    exercise('reject_wrong_before_ensemble_sum',lambda r,e,k:mutate_npz(r/'temporary/svc-topic-correction-evidence.npz',lambda v:v['before_total'].__setitem__((8,0),v['before_total'][8,0]+.01)),True)
    exercise('reject_intervention_not_used_by_prior',lambda r,e,k:mutate_npz(r/'temporary/before_atlas.npz',lambda v:v.__setitem__('probabilities',np.full_like(v['probabilities'],1/18))),True)
    exercise('reject_unknown_preview_flag',lambda r,e,k:k.__setitem__('namespace',{}),True)
    # Synthetic fixture substitution only; production retains the exact public
    # three-row fingerprint. No original competition passages are bundled.
    global PUBLIC_FIXTURE
    saved=deepcopy(PUBLIC_FIXTURE)
    try:
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);runtime=root/'runtime';runtime.mkdir();(runtime/'run_fulltext_selected_ensemble.py').write_text(orchestrator);(runtime/'reference627_common.py').write_text(common)
            records=[('synthetic-a','fixture a'),('synthetic-b','fixture b'),('synthetic-c','fixture c')]
            PUBLIC_FIXTURE=[(i,len(t.encode()),hashlib.blake2s(t.encode(),digest_size=16).hexdigest()) for i,t in records]
            with (root/'test.csv').open('w',newline='') as f:w=csv.writer(f);w.writerow(['id','text']);w.writerows(records)
            r=validate_intervention(root,root/'test.csv',temporary_dir=root/'temp',runtime_dir=runtime,namespace={'preview':True,'validation_report':{'status':'FORMAT_ONLY_PREVIEW','all_checks_pass':True,'rows':3}},emit=False)
            checks['preview_declares_intervention_not_executed']=r['status']=='FORMAT_ONLY_PREVIEW_INTERVENTION_NOT_EXECUTED' and r['intervention_executed'] is False and r['intervention_activation_verified'] is False
    finally:PUBLIC_FIXTURE=saved
    result={'stage':'12B','status':'PASS' if all(checks.values()) else 'FAIL','passed':sum(bool(v) for v in checks.values()),'total':len(checks),'checks':{k:bool(v) for k,v in checks.items()},'upstream_models_executed':False,'classifier_fitted':False,'actual_hidden_evaluation_run':False,'synthetic_public_fixture_substitution_for_test_only':True}
    if not all(checks.values()):raise AssertionError(json.dumps(result))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--self-test',action='store_true');p.add_argument('--work',default='/kaggle/working');p.add_argument('--test');p.add_argument('--temporary',default=DEFAULT_TEMP);p.add_argument('--runtime',default=DEFAULT_RUNTIME);p.add_argument('--preview',action='store_true');p.add_argument('--receipt');a=p.parse_args()
    r=self_test() if a.self_test else validate_intervention(a.work,a.test,temporary_dir=a.temporary,runtime_dir=a.runtime,namespace={'preview':a.preview})
    if a.receipt:Path(a.receipt).write_text(json.dumps(r,indent=2)+'\n')
    if a.self_test:print(json.dumps(r,indent=2))
