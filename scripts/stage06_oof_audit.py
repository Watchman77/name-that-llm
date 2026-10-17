"""CPU audit of public LoRA24 V1; no training, blending search or submission.
Source: lightningv08/ntl-lora24-fivefold-training, version 358831649 (Apache 2.0).
We independently reconstruct numeric predictions; never execute attached source.
"""
import hashlib
import json
import os
import time
from pathlib import Path
from importlib.metadata import version
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.special import logsumexp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

START = time.monotonic()
INPUT = Path(os.environ.get('NTL_AUDIT_INPUT', '/kaggle/input'))
OUTPUT = Path(os.environ.get('NTL_AUDIT_OUTPUT', '/kaggle/working/stage06_audit'))
OUTPUT.mkdir(parents=True, exist_ok=True)
TRAIN_SHA = '40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707'

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def one(paths, description):
    paths = list(paths)
    assert len(paths) == 1, (description, [str(p) for p in paths])
    return paths[0]

def record_json(name, value):
    (OUTPUT/name).write_text(json.dumps(value, indent=2))

train_path = one((p for p in INPUT.rglob('train.csv') if (p.parent/'test.csv').exists()), 'competition train')
assert digest(train_path) == TRAIN_SHA
train = pd.read_csv(train_path, dtype=str)
assert len(train) == 2699 and train.id.is_unique and not train.isna().any().any()
public_oof_path = one(INPUT.rglob('lora24_oof.npz'), 'public LoRA OOF')
root = public_oof_path.parent
spec = json.loads((root/'training_spec.json').read_text())
plan = spec['plan']
assert plan['train_sha256'] == TRAIN_SHA
classes = np.asarray(plan['labels'], dtype=str)
assert len(classes) == 18 and np.array_equal(classes, np.sort(train.model.unique()))
y = np.searchsorted(classes, train.model.to_numpy())
ids = train.id.to_numpy(dtype=str)
groups = train.prompt_id.to_numpy(dtype=str)
texts = train.text.to_numpy(dtype=object)
with np.load(public_oof_path, allow_pickle=False) as z:
    for key, expected in [('ids', ids), ('labels', classes), ('y', y), ('groups', groups)]:
        assert np.array_equal(z[key], expected), key
    published = {key: z[key].copy() for key in ('plain','center','half')}
assert all(x.shape == (3,len(train),18) and np.isfinite(x).all() for x in published.values())
report = json.loads((root/'lora24_comparison.json').read_text())
assert report['train_sha256'] == TRAIN_SHA and report['classes'] == classes.tolist()

# Reproduce the published deterministic sparse contexts. Prompt IDs are used
# only to construct the evaluation batches; clustering sees the mixed texts.
def recover_context(rows, size, job):
    seed = 20261015 + int(hashlib.sha256(job.encode()).hexdigest()[:6], 16)
    rng = np.random.default_rng(seed)
    batches = np.empty(len(rows), int)
    for prompt in np.unique(groups[rows]):
        positions = rng.permutation(np.flatnonzero(groups[rows] == prompt))
        batches[positions] = np.arange(len(positions)) // size
    recovered = np.empty(len(rows), int)
    offset = 0
    for batch in np.unique(batches):
        positions = np.flatnonzero(batches == batch)
        if len(positions) < 2:
            clusters = np.zeros(len(positions), int)
        else:
            tfidf = TfidfVectorizer(stop_words='english', sublinear_tf=True).fit_transform(texts[rows[positions]])
            distance = np.clip(1-(tfidf@tfidf.T).toarray(),0,2)
            np.fill_diagonal(distance,0)
            clusters = fcluster(linkage(squareform(distance,checks=False),method='average'), .9, criterion='distance')
        recovered[positions] = clusters + offset
        offset += int(clusters.max()) + 1
    return recovered

def affine(features, coef, intercept):
    logits = np.asarray(features,dtype=np.float64) @ np.asarray(coef,dtype=np.float64).T + np.asarray(intercept,dtype=np.float64)
    return logits-logsumexp(logits,axis=-1,keepdims=True)

reconstructed = {key: np.empty_like(published[key]) for key in ('plain','center')}
coverage = np.zeros(len(train), int)
fold = np.full(len(train), -1, int)
verified_files = []
fold_records = []
cluster_records = []
outer_jobs = [job for job in spec['jobs'] if job != 'final']
assert outer_jobs == [f'outer{i}' for i in range(5)], outer_jobs
for number, job in enumerate(outer_jobs):
    fit = np.asarray(plan['jobs'][job]['fit'],int)
    valid = np.asarray(plan['jobs'][job]['valid'],int)
    assert len(fit) == len(np.unique(fit)) and len(valid) == len(np.unique(valid))
    assert not set(fit)&set(valid) and set(fit)|set(valid) == set(range(len(train)))
    assert not set(groups[fit])&set(groups[valid])
    assert len(np.unique(groups[valid])) == 30
    coverage[valid] += 1
    fold[valid] = number
    checkpoint = root/'lora24'/job
    complete = json.loads((checkpoint/'COMPLETE.json').read_text())
    assert complete['job'] == job and complete['train_sha256'] == TRAIN_SHA
    # Verify the numeric inference artifacts used here. Large adapter weights
    # are not loaded or rehashed: this is not an independent retraining audit.
    for filename in ('validation_features.npz','readouts.npz','training_report.json'):
        path = checkpoint/filename
        actual = digest(path)
        assert actual == complete['files'][filename], (job,filename)
        verified_files.append(dict(job=job,file=filename,sha256=actual))
    training = json.loads((checkpoint/'training_report.json').read_text())
    assert training['job'] == job and training['train_sha256'] == TRAIN_SHA
    assert training['epochs'] == 3
    with np.load(checkpoint/'validation_features.npz',allow_pickle=False) as z:
        assert np.array_equal(z['ids'],ids[valid]) and np.array_equal(z['classes'],classes)
        assert z['train_sha256'].item() == TRAIN_SHA
        features = z['features'].copy()
    with np.load(checkpoint/'readouts.npz',allow_pickle=False) as z:
        assert np.array_equal(z['classes'],classes) and z['train_sha256'].item() == TRAIN_SHA
        heads = {key:z[key].copy() for key in ('plain_coef','plain_intercept','center_coef','center_intercept')}
    assert features.shape[0] == len(valid) and np.isfinite(features).all()
    plain = affine(features,heads['plain_coef'],heads['plain_intercept'])
    for scenario_index, size in enumerate((1,3,6)):
        inferred = recover_context(valid,size,job)
        sizes = np.bincount(inferred)[inferred]
        residual = features.copy()
        for cluster in np.unique(inferred):
            positions = np.flatnonzero(inferred == cluster)
            residual[positions] -= features[positions].mean(0)
        center = affine(residual,heads['center_coef'],heads['center_intercept'])
        center[sizes<2] = plain[sizes<2]
        reconstructed['plain'][scenario_index,valid] = plain
        reconstructed['center'][scenario_index,valid] = center
        for name, value in [('plain',plain),('center',center),('half',.5*plain+.5*center)]:
            fold_records.append(dict(component=name,scenario=size,fold=number,rows=len(valid),balanced_accuracy=float(balanced_accuracy_score(y[valid],value.argmax(1)))))
        a = inferred[:,None] == inferred[None,:]
        b = groups[valid,None] == groups[valid][None,:]
        off = ~np.eye(len(valid),dtype=bool)
        tp = int((a&b&off).sum())
        inferred_pairs = int((a&off).sum())
        true_pairs = int((b&off).sum())
        cluster_records.append(dict(scenario=size,fold=number,singleton_rows=int((sizes<2).sum()),rows=len(valid),pair_precision=tp/max(inferred_pairs,1),pair_recall=tp/max(true_pairs,1),pair_recall_note='Denominator includes true pairs separated by simulated batches.'))
    print('AUDITED',job,len(valid),'held-out rows',flush=True)
assert np.all(coverage==1) and np.all(fold>=0)
reconstructed['half'] = .5*reconstructed['plain'] + .5*reconstructed['center']
checks = {}
for name,value in reconstructed.items():
    assert np.isfinite(value).all()
    np.testing.assert_allclose(value,published[name],rtol=1e-7,atol=2e-6)
    scores = [float(balanced_accuracy_score(y,part.argmax(1))) for part in value]
    np.testing.assert_allclose(scores,report['standalone'][name],rtol=0,atol=1e-12)
    checks[name] = dict(scenario_balanced_accuracy=dict(zip(('1','3','6'),scores)),maximum_absolute_log_probability_error=float(np.max(np.abs(value-published[name]))))

recalls = []
pairs = []
for name,values in reconstructed.items():
    for scenario_index,size in enumerate((1,3,6)):
        predicted = values[scenario_index].argmax(1)
        counts = confusion_matrix(y,predicted,labels=np.arange(18))
        rates = counts/counts.sum(1,keepdims=True)
        pd.DataFrame(counts,index=classes,columns=classes).to_csv(OUTPUT/f'{name}_scenario{size}_confusion_counts.csv')
        pd.DataFrame(rates,index=classes,columns=classes).to_csv(OUTPUT/f'{name}_scenario{size}_confusion_rates.csv')
        for i,label in enumerate(classes):
            recalls.append(dict(component=name,scenario=size,model=label,support=int(counts[i].sum()),recall=float(rates[i,i])))
            for j,other in enumerate(classes):
                if i!=j:
                    pairs.append(dict(component=name,scenario=size,actual=label,predicted=other,count=int(counts[i,j]),rate=float(rates[i,j])))
pd.DataFrame(recalls).to_csv(OUTPUT/'per_class_recall.csv',index=False)
pd.DataFrame(pairs).sort_values('count',ascending=False).to_csv(OUTPUT/'confusion_pairs.csv',index=False)
pd.DataFrame(fold_records).to_csv(OUTPUT/'fold_scores.csv',index=False)
pd.DataFrame(cluster_records).to_csv(OUTPUT/'cluster_diagnostics.csv',index=False)
pd.DataFrame(dict(id=ids,prompt_id=groups,fold=fold)).to_csv(OUTPUT/'fold_assignments.csv',index=False)
inventory = [dict(path=str(p.relative_to(INPUT)),bytes=p.stat().st_size) for pattern in ('*.npz','*oof*.csv','*report*.json','COMPLETE.json') for p in INPUT.rglob(pattern)]
record_json('artifact_inventory.json',inventory)
summary = dict(status='VERIFIED_PUBLIC_LORA_COMPONENT_ONLY',source_version=358831649,train_sha256=TRAIN_SHA,rows=len(train),prompts=len(np.unique(groups)),classes=classes.tolist(),checks=checks,numeric_files_verified=verified_files,artifact_hashes={name:digest(root/name) for name in ('lora24_oof.npz','training_spec.json','lora24_comparison.json')},packages={name:version(name) for name in ('numpy','pandas','scipy','scikit-learn')},seconds=time.monotonic()-START,limitations=['These are public LoRA component development results, NOT S02 full-ensemble validation or an official public score.','No adapter retraining or hashing of adapter weight binaries; provenance relies on published source and checkpoint manifests.','The private Gemma reference is not needed, loaded or validated; its mixture scores are excluded.','Historical development reuse remains; no fresh untouched validation set has been created.','Centering experiment not run; frozen feature cache availability remains to be inventoried.'])
record_json('provenance.json',summary)
print('STAGE06_AUDIT_RESULT',json.dumps(summary),flush=True)
print('HARDEST_HALF_SCENARIO3',pd.DataFrame(recalls).query("component=='half' and scenario==3").sort_values('recall').head(6).to_json(orient='records'),flush=True)
print('TOP_HALF_SCENARIO3_CONFUSIONS',pd.DataFrame(pairs).query("component=='half' and scenario==3").nlargest(8,'count').to_json(orient='records'),flush=True)
print('AUDIT_COMPLETE',flush=True)
