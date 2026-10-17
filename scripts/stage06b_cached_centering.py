"""CPU-only public artifact audit and fixed DeBERTa centering comparison.
Protocol: plans/STAGE_06B_CACHED_CENTERING.md. No competition submission.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
import hashlib, io, json, pickletools, time, warnings, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
threadpool_limits(2)
START=time.monotonic()
INPUT=Path('/kaggle/input'); OUT=Path('/kaggle/working/stage06b'); OUT.mkdir(exist_ok=True)
TRAIN_SHA='40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707'

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()

def one(pattern):
    paths=list(INPUT.rglob(pattern)); assert len(paths)==1,(pattern,[str(p) for p in paths]); return paths[0]

def write_json(name,value): (OUT/name).write_text(json.dumps(value,indent=2))
train_path=one('train.csv'); assert digest(train_path)==TRAIN_SHA
train=pd.read_csv(train_path,dtype=str); assert len(train)==2699 and train.id.is_unique
ids=train.id.to_numpy(dtype=str); groups=train.prompt_id.to_numpy(dtype=str)
classes=np.sort(train.model.unique()); assert len(classes)==18
label_index={value:i for i,value in enumerate(classes)}
y=np.asarray([label_index[v] for v in train.model]); texts=train.text.to_numpy()
ID_SET=set(ids); id_checks=[]

def safe_ids(path,expected):
    """Read numeric/string arrays normally; inspect object ID bytecode, NEVER unpickle."""
    try:
        with np.load(path,allow_pickle=False) as z: actual=z['ids'].astype(str)
        mode='numpy_nonobject'
    except ValueError as error:
        assert 'Object arrays cannot be loaded' in str(error)
        with zipfile.ZipFile(path) as archive:
            info=archive.getinfo('ids.npy'); assert info.file_size<2_000_000
            raw=io.BytesIO(archive.read(info))
        v=np.lib.format.read_magic(raw)
        assert v in ((1,0),(2,0)),v
        shape,fortran,dtype=(np.lib.format.read_array_header_1_0(raw) if v==(1,0) else np.lib.format.read_array_header_2_0(raw))
        assert shape==(len(expected),) and dtype.hasobject and not fortran
        literals=[]
        for opcode,arg,pos in pickletools.genops(raw.read()):
            if isinstance(arg,bytes):
                try:arg=arg.decode('ascii')
                except UnicodeDecodeError:continue
            if isinstance(arg,str) and arg in ID_SET:literals.append(arg)
        actual=np.asarray(literals,dtype=str)
        mode='object_npy_static_unique_ID_literals_no_unpickling'
    assert np.array_equal(actual,expected),(str(path),'ID order mismatch',len(actual),len(expected))
    assert len(np.unique(actual))==len(expected)
    id_checks.append(dict(path=str(path.relative_to(INPUT)),mode=mode,rows=len(expected)))
    return actual

# Numeric schema inventory avoids loading model weights or object arrays.
inventory=[]
for p in sorted(INPUT.rglob('*.npz')):
    entries={}
    with zipfile.ZipFile(p) as archive:
        for member in archive.infolist():
            if not member.filename.endswith('.npy'):continue
            with archive.open(member) as stream:
                v=np.lib.format.read_magic(stream)
                shape,order,dtype=(np.lib.format.read_array_header_1_0(stream) if v==(1,0) else np.lib.format.read_array_header_2_0(stream))
                entries[member.filename[:-4]]=dict(shape=list(shape),dtype=str(dtype),object=bool(dtype.hasobject))
    inventory.append(dict(path=str(p.relative_to(INPUT)),bytes=p.stat().st_size,arrays=entries))
write_json('numeric_inventory.json',inventory)

# Validate probability outputs against the source's exact seed-0 prompt folds.
source_prompts=np.sort(np.unique(groups)); np.random.RandomState(0).shuffle(source_prompts)
source_fold_map={p:i%5 for i,p in enumerate(source_prompts)}
source_folds=np.asarray([source_fold_map[g] for g in groups])
component_audits=[]
for filename in ('tokattn_oof.npz','tokattn_debl23_oof.npz','tokattn_q8l22_oof.npz'):
    p=one(filename);safe_ids(p,ids)
    with np.load(p,allow_pickle=False) as z:
        assert np.array_equal(z['classes'],classes) and np.array_equal(z['fold'],source_folds)
        prob=z['probs'].copy()
    assert prob.shape==(2699,18) and np.isfinite(prob).all() and (prob>=0).all()
    np.testing.assert_allclose(prob.sum(1),1,rtol=0,atol=2e-6)
    score=float(balanced_accuracy_score(y,prob.argmax(1)))
    component_audits.append(dict(component=filename,balanced_accuracy=score,sha256=digest(p),fold_provenance='source seed0 prompt-disjoint; numeric OOF verified, training not rerun'))

lora_root=one('worker.py').parent/'lora_out'
lora_prob=np.empty((len(train),18));coverage=np.zeros(len(train),int);lora_folds=[]
for number in range(5):
    folder=lora_root/f'job_{number}'; valid=np.flatnonzero(source_folds==number)
    assert not set(groups[valid])&set(groups[source_folds!=number])
    safe_ids(folder/'valid_features.npz',ids[valid])
    with np.load(folder/'valid_features.npz',allow_pickle=False) as z: features=z['features'].astype(np.float32)
    with np.load(folder/'readouts.npz',allow_pickle=False) as z:
        assert np.array_equal(z['classes'],classes) and np.array_equal(z['pool_layers'],[2,4,6,8,12,16,20,24])
        logits=features@z['plain_coef'].T+z['plain_intercept']
    assert logits.shape==(len(valid),18) and np.isfinite(logits).all()
    completion=json.loads((folder/'COMPLETE.json').read_text()); assert str(completion['job'])==str(number)
    score=float(balanced_accuracy_score(y[valid],logits.argmax(1)))
    raw_accuracy=float((logits.argmax(1)==y[valid]).mean())
    lora_folds.append(dict(fold=number,balanced_accuracy=score,reconstructed_accuracy=raw_accuracy,source_fp32_accuracy=completion['valid_acc'],difference_after_fp16_cache=raw_accuracy-completion['valid_acc'],skipped_updates=completion['skipped'],steps=completion['steps']))
    lora_prob[valid]=logits;coverage[valid]+=1
assert np.all(coverage==1)
component_audits.append(dict(component='lora24_seed7_plain',balanced_accuracy=float(balanced_accuracy_score(y,lora_prob.argmax(1))),folds=lora_folds,limitation='Saved features are float16; source completion accuracies used pre-serialization float32. COMPLETE has no file hashes.'))
write_json('component_audits.json',component_audits)
print('COMPONENT_AUDIT_RESULT',json.dumps(component_audits),flush=True)

# Preselected S02 frozen DeBERTa large representation, no layer/model search.
cache=one('enc_debv3l2_train.npz'); safe_ids(cache,ids)
with np.load(cache,allow_pickle=False) as z:
    layers=z['layers'];emb=z['emb']
    assert emb.shape==(len(train),9,1024) and emb.dtype==np.float16
    X=np.concatenate([emb[:,np.flatnonzero(layers==layer).item()] for layer in (24,6)],axis=1).astype(np.float32)
assert X.shape==(2699,2048) and np.isfinite(X).all()
del emb
# REFERENCE_PROMPT_FOLDS is inserted from our archived baseline before launch.
REFERENCE_PROMPT_FOLDS = {'p000': 4, 'p001': 3, 'p002': 5, 'p003': 2, 'p004': 1, 'p005': 5, 'p006': 4, 'p007': 3, 'p008': 2, 'p009': 1, 'p010': 5, 'p011': 4, 'p012': 3, 'p013': 2, 'p014': 1, 'p015': 5, 'p016': 4, 'p017': 3, 'p018': 2, 'p019': 1, 'p020': 5, 'p021': 4, 'p022': 3, 'p023': 2, 'p024': 1, 'p025': 5, 'p026': 4, 'p027': 3, 'p028': 2, 'p029': 1, 'p030': 5, 'p031': 4, 'p032': 3, 'p033': 2, 'p034': 1, 'p035': 5, 'p036': 4, 'p037': 3, 'p038': 2, 'p039': 1, 'p040': 5, 'p041': 4, 'p042': 3, 'p043': 2, 'p044': 1, 'p045': 5, 'p046': 4, 'p047': 3, 'p048': 2, 'p049': 1, 'p050': 5, 'p051': 4, 'p052': 3, 'p053': 2, 'p054': 1, 'p055': 5, 'p056': 4, 'p057': 3, 'p058': 2, 'p059': 1, 'p060': 5, 'p061': 4, 'p062': 3, 'p063': 2, 'p064': 1, 'p065': 5, 'p066': 4, 'p067': 3, 'p068': 2, 'p069': 1, 'p070': 5, 'p071': 4, 'p072': 3, 'p073': 2, 'p074': 1, 'p075': 5, 'p076': 4, 'p077': 3, 'p078': 2, 'p079': 1, 'p080': 5, 'p081': 4, 'p082': 3, 'p083': 2, 'p084': 1, 'p085': 5, 'p086': 4, 'p087': 3, 'p088': 2, 'p089': 1, 'p090': 5, 'p091': 4, 'p092': 3, 'p093': 2, 'p094': 1, 'p095': 5, 'p096': 4, 'p097': 3, 'p098': 2, 'p099': 1, 'p100': 5, 'p101': 4, 'p102': 3, 'p103': 2, 'p104': 1, 'p105': 5, 'p106': 4, 'p107': 3, 'p108': 2, 'p109': 1, 'p110': 5, 'p111': 4, 'p112': 3, 'p113': 2, 'p114': 1, 'p115': 5, 'p116': 4, 'p117': 3, 'p118': 2, 'p119': 1, 'p120': 5, 'p121': 4, 'p122': 3, 'p123': 2, 'p124': 1, 'p125': 5, 'p126': 4, 'p127': 3, 'p128': 2, 'p129': 1, 'p130': 5, 'p131': 4, 'p132': 3, 'p133': 2, 'p134': 1, 'p135': 5, 'p136': 4, 'p137': 3, 'p138': 2, 'p139': 1, 'p140': 5, 'p141': 4, 'p142': 3, 'p143': 2, 'p144': 1, 'p145': 5, 'p146': 4, 'p147': 3, 'p148': 2, 'p149': 1}
our_folds=np.full(len(train),-1,int)
splits=list(GroupKFold(5).split(X,y,groups))
for number,(_,valid) in enumerate(splits):our_folds[valid]=number+1
assert all(REFERENCE_PROMPT_FOLDS[g]==int(k) for g,k in zip(groups,our_folds))

def centered(features,labels):
    result=features.copy()
    for group in np.unique(labels):
        pos=np.flatnonzero(labels==group); result[pos]-=features[pos].mean(0)
    return result

def partition_training(rows,size,rng):
    assignments=np.full(len(rows),-1,int);exposures=np.zeros(len(rows),int);offset=0
    for prompt in np.unique(groups[rows]):
        positions=rng.permutation(np.flatnonzero(groups[rows]==prompt))
        chunks=[positions[start:start+size] for start in range(0,len(positions),size)]
        if len(chunks[-1])==1 and len(chunks)>1:
            tail=chunks.pop();chunks[-1]=np.concatenate([chunks[-1],tail])
        for chunk in chunks:
            assignments[chunk]=offset;exposures[chunk]+=1;offset+=1
    assert np.all(assignments>=0) and np.all(exposures==1)
    return assignments

def contexts(rows,size,number):
    rng=np.random.default_rng(20261017+1000+number)
    batches=np.empty(len(rows),int)
    for prompt in np.unique(groups[rows]):
        members=rng.permutation(np.flatnonzero(groups[rows]==prompt)); batches[members]=np.arange(len(members))//size
    inferred=np.empty(len(rows),int);oracle=np.empty(len(rows),int);offset=0;oracle_offset=0
    for batch in np.unique(batches):
        positions=np.flatnonzero(batches==batch)
        tf=TfidfVectorizer(stop_words='english',sublinear_tf=True).fit_transform(texts[rows[positions]])
        distance=np.clip(1-(tf@tf.T).toarray(),0,2);np.fill_diagonal(distance,0)
        recovered=fcluster(linkage(squareform(distance,checks=False),method='average'),.85,criterion='distance') if len(positions)>1 else np.zeros(1,int)
        inferred[positions]=recovered+offset;offset+=int(recovered.max())+1
        _,true=np.unique(groups[rows[positions]],return_inverse=True)
        oracle[positions]=true+oracle_offset;oracle_offset+=int(true.max())+1
    return batches,inferred,oracle

convergence=[]
def fit_head(features,targets,weights=None):
    scaler=StandardScaler().fit(features,sample_weight=weights)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always');clf=LogisticRegression(C=.005,max_iter=5000).fit(scaler.transform(features),targets,sample_weight=weights)
    convergence.extend(str(w.message) for w in caught)
    assert np.array_equal(clf.classes_,np.arange(18))
    return scaler,clf

def logprob(head,features):
    scaler,clf=head;return np.log(np.clip(clf.predict_proba(scaler.transform(features)),1e-9,1))

predictions={key:np.empty((3,len(train)),int) for key in ('control','candidate','oracle_control','oracle_candidate')}
fold_scores=[]; realised=[]
for number,(fit,valid) in enumerate(splits):
    assert not set(groups[fit])&set(groups[valid])
    plain=fit_head(X[fit],y[fit])
    control=fit_head(centered(X[fit],groups[fit]),y[fit])
    rng=np.random.default_rng(20261017+number)
    pieces=[]
    for size in (2,3,6):
        grouping=partition_training(fit,size,rng);pieces.append(centered(X[fit],grouping))
        counts=np.bincount(grouping);realised.append(dict(fold=number,target=size,sizes=np.unique(counts,return_counts=True)[0].tolist(),counts=np.unique(counts,return_counts=True)[1].tolist()))
    candidate=fit_head(np.concatenate(pieces),np.tile(y[fit],3),np.full(3*len(fit),1/3))
    del pieces
    for scenario,size in enumerate((1,3,6)):
        batches,inferred,oracle=contexts(valid,size,number)
        plain_log=np.empty((len(valid),18))
        _,plain_clf=plain
        for batch in np.unique(batches):
            pos=np.flatnonzero(batches==batch);features=X[valid[pos]]
            standardized=(features-features.mean(0))/(features.std(0)+1e-6) if len(pos)>=2 else plain[0].transform(features)
            plain_log[pos]=np.log(np.clip(plain_clf.predict_proba(standardized),1e-9,1))
        for mode,labels in [('inferred',inferred),('oracle',oracle)]:
            sizes=np.bincount(labels)[labels];eligible=(sizes>=2)&(sizes<=18)
            residual=centered(X[valid],labels)
            for label,head in [('control',control),('candidate',candidate)]:
                evidence=.05*plain_log.copy()
                if eligible.any():evidence[eligible]+=.25*logprob(head,residual[eligible])
                key=label if mode=='inferred' else 'oracle_'+label
                predictions[key][scenario,valid]=evidence.argmax(1)
                fold_scores.append(dict(fold=number,scenario=size,grouping=mode,head=label,balanced_accuracy=float(balanced_accuracy_score(y[valid],evidence.argmax(1))),eligible_rows=int(eligible.sum())))
    print('CENTERING_FOLD_COMPLETE',number+1,'seconds',round(time.monotonic()-START,1),flush=True)
    del plain,control,candidate

scores={key:[float(balanced_accuracy_score(y,part)) for part in value] for key,value in predictions.items()}
gains=np.asarray(scores['candidate'])-np.asarray(scores['control'])
diff=((predictions['candidate']==y[None]).astype(float)-(predictions['control']==y[None]).astype(float)).mean(0)
unique=np.unique(groups)
num=np.stack([np.bincount(y[groups==g],weights=diff[groups==g],minlength=18) for g in unique])
den=np.stack([np.bincount(y[groups==g],minlength=18) for g in unique])
draws=np.random.default_rng(20261017).integers(len(unique),size=(3000,len(unique)))
bootstrap=(num[draws].sum(1)/den[draws].sum(1).clip(1)).mean(1)
interval=np.quantile(bootstrap,[.025,.975]).tolist()
mean_gain=float(gains.mean());screen=bool(mean_gain>0 and interval[0]>0 and min(gains)>=-.005 and not convergence)
recalls=[]
for key,pred in predictions.items():
    for scenario,size in enumerate((1,3,6)):
        for i,label in enumerate(classes):recalls.append(dict(head=key,scenario=size,model=label,recall=float((pred[scenario,y==i]==i).mean())))
pd.DataFrame(recalls).to_csv(OUT/'per_class_recall.csv',index=False)
pd.DataFrame(fold_scores).to_csv(OUT/'fold_scores.csv',index=False)
np.savez_compressed(OUT/'paired_predictions.npz',ids=ids,classes=classes,y=y,groups=groups,fold=our_folds,**predictions)
result=dict(status='COMPLETE_CPU_COMPONENT_TEST',representation='frozen DeBERTa-v3-large mean L24+L6; fp16 public cache',train_sha256=TRAIN_SHA,cache_sha256=digest(cache),scenarios=[1,3,6],scores=scores,primary_mean_gain=mean_gain,scenario_gains=gains.tolist(),paired_prompt_95_interval=interval,screen_pass=screen,convergence_warnings=convergence,realised_training_group_sizes=realised,seconds=time.monotonic()-START,limitations=['Only one of five frozen centered twins tested; not a complete S02 ensemble.','TF-IDF-only clustering approximates mixed GLM/lexical S02 clustering.','Quantized cached features; not bitwise reproduction of fp32 S02 extraction.','Public OOF components use source seed0 folds; centering test uses our archived GroupKFold folds. They are not blended.','Historical development reuse remains; no official submission or paid/GPU run.'])
write_json('centering_result.json',result);write_json('id_checks.json',id_checks)
print('CENTERING_RESULT',json.dumps(result),flush=True)
print('NUMERIC_INVENTORY',json.dumps(inventory),flush=True)
print('STAGE06B_COMPLETE',flush=True)
