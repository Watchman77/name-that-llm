"""Stage 04B: one fixed offline DeBERTa trial, with fresh weights per prompt fold."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import gc
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
import platform
import random
import time
import traceback
import warnings

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['USE_TF'] = '0'
os.environ['USE_FLAX'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import get_linear_schedule_with_warmup

SEED = 20261016
SOURCE_SHA = '40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707'
REFERENCE_SHA = '15ca5d9d95fead2690a6c81559a3862f12143f83ad65684d1c584a2827e92ab3'
MODEL_ID = 'microsoft/deberta-v3-base'
REVISION = '8ccc9b6f36199bec6961081d44eb72fb3f7353f3'
PLAN_COMMIT = '8ab67d337b139059669455d605cf802eda9d4301'
VARIANT = 'deberta_v3_base_final_epoch'
REFERENCE = 'character_plus_style'
MAX_LENGTH, BATCH_SIZE, ACCUMULATION, EPOCHS = 384, 4, 4, 3
MAX_SECONDS = 3 * 60 * 60


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False))


def reset_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def check_budget(started):
    # Reserve the last minute for saving status and any partial checkpoint.
    if time.perf_counter() - started >= MAX_SECONDS - 60:
        raise TimeoutError('Three-hour GPU budget reached; reserving time to preserve evidence')


class TokenRows(Dataset):
    def __init__(self, encoded, labels, indices):
        self.encoded, self.labels = encoded, labels
        self.indices = np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        index = int(self.indices[position])
        row = {key:value[index] for key,value in self.encoded.items()}
        row['labels'] = int(self.labels[index])
        return row


def make_collator(tokenizer):
    def collate(rows):
        labels = torch.tensor([r['labels'] for r in rows], dtype=torch.long)
        features = [{k:v for k,v in row.items() if k != 'labels'} for row in rows]
        batch = tokenizer.pad(features, padding=True, pad_to_multiple_of=8, return_tensors='pt')
        batch['labels'] = labels
        return batch
    return collate


def weighted_loss_sum(logits, labels, weights):
    # Explicit sum allows exact per-example normalization over partial accumulation groups.
    return F.cross_entropy(logits.float(), labels, weight=weights, reduction='sum')


def parameter_groups(model):
    decay, no_decay = [], []
    for name, value in model.named_parameters():
        if not value.requires_grad:
            continue
        if name.endswith('bias') or 'LayerNorm' in name or 'layer_norm' in name:
            no_decay.append(value)
        else:
            decay.append(value)
    assert len(decay) + len(no_decay) == sum(p.requires_grad for p in model.parameters())
    return [{'params':decay, 'weight_decay':0.01}, {'params':no_decay, 'weight_decay':0.0}]


def environment_record():
    packages = ['torch','transformers','tokenizers','sentencepiece','protobuf',
                'numpy','pandas','scikit-learn','huggingface-hub','safetensors']
    versions = {p:importlib.metadata.version(p) for p in packages}
    return {'python':platform.python_version(), 'packages':versions,
            'cuda':torch.version.cuda, 'gpu':torch.cuda.get_device_name(0),
            'visible_gpu_count':torch.cuda.device_count(),
            'determinism':'Seeded, deterministic algorithms warn-only; GPU nondeterminism remains possible.'}


def make_model(model_dir, labels):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, local_files_only=True, num_labels=len(labels),
        id2label={i:label for i,label in enumerate(labels)},
        label2id={label:i for i,label in enumerate(labels)})
    return model.float().to('cuda:0')


def paired_interval(oof, labels):
    prompts = sorted(oof.prompt_id.unique())
    gi = oof.prompt_id.map({x:i for i,x in enumerate(prompts)}).to_numpy()
    yi = oof.model.map({x:i for i,x in enumerate(labels)}).to_numpy()
    total = np.zeros((len(prompts),len(labels)),dtype=np.int64)
    np.add.at(total,(gi,yi),1)
    weights = np.random.default_rng(SEED).multinomial(len(prompts),
        np.full(len(prompts),1/len(prompts)),size=2000)
    denominator = weights @ total
    assert (denominator > 0).all()
    samples = {}
    for variant in [VARIANT,REFERENCE]:
        correct = np.zeros_like(total)
        np.add.at(correct,(gi,yi),(oof[variant].to_numpy()==oof.model.to_numpy()).astype(int))
        samples[variant] = ((weights @ correct)/denominator).mean(axis=1)
    low,high = np.quantile(samples[VARIANT]-samples[REFERENCE],[.025,.975])
    return {'candidate':VARIANT,'reference':REFERENCE,
        'delta_balanced_accuracy':float(balanced_accuracy_score(oof.model,oof[VARIANT])-
                                        balanced_accuracy_score(oof.model,oof[REFERENCE])),
        'lower_95':float(low),'upper_95':float(high),'unit':'prompt_id','resamples':2000,'seed':SEED,
        'limitation':'Conditional development predictions; no refitting/selection uncertainty or multiplicity correction.'}


def run(data_dir, resource_dir, out_dir, budget_started=None):
    started = budget_started if budget_started is not None else time.perf_counter()
    if out_dir.exists():
        raise FileExistsError(f'Preserve the previous attempt before rerunning: {out_dir}')
    out_dir.mkdir(parents=True)
    state = {'stage':'04B_supervised_encoder','status':'preflight','plan_commit':PLAN_COMMIT,
             'started_utc':utc_now(),'completed_folds':[], 'gpu_budget_seconds':MAX_SECONDS}
    status_path = out_dir/'run_status.json'
    save_json(status_path,state)
    training_log, fold_rows, warning_log = [], [], []
    prior_showwarning = warnings.showwarning
    def record_warning(message, category, filename, lineno, file=None, line=None):
        warning_log.append({'category':category.__name__,'message':str(message),
                            'filename':filename,'line':lineno})
        save_json(out_dir/'warnings.json',warning_log)
        prior_showwarning(message,category,filename,lineno,file=file,line=line)
    warnings.showwarning = record_warning
    model, optimizer, scaler, scheduler = None, None, None, None
    current_fold = None
    try:
        assert torch.cuda.is_available(), 'This recipe requires the declared included T4 GPU; no CPU fallback'
        torch.cuda.set_device(0)
        assert 'T4' in torch.cuda.get_device_name(0), 'Record an amendment before using different hardware'
        env = environment_record()
        assert env['packages']['transformers']=='4.56.2'
        assert env['packages']['sentencepiece']=='0.2.1'
        assert tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2]) >= (2,6)
        save_json(out_dir/'environment.json',env)
        assert file_sha(data_dir/'train.csv')==SOURCE_SHA
        assert file_sha(resource_dir/'stage03_oof_predictions.csv')==REFERENCE_SHA
        resource_manifest = json.loads((resource_dir/'resources_manifest.json').read_text())
        assert resource_manifest['model_id']==MODEL_ID and resource_manifest['revision']==REVISION
        model_dir = resource_dir/'deberta-v3-base'
        for name, expected in resource_manifest['model_files_sha256'].items():
            assert file_sha(model_dir/name)==expected, f'Checkpoint resource changed: {name}'
        save_json(out_dir/'resources_manifest.json',resource_manifest)
        train = pd.read_csv(data_dir/'train.csv',dtype=str)
        old = pd.read_csv(resource_dir/'stage03_oof_predictions.csv',
                          dtype={'id':str,'prompt_id':str,'model':str})
        assert train[['id','prompt_id','model']].equals(old[['id','prompt_id','model']])
        assert len(train)==2699 and train.id.is_unique and train.prompt_id.nunique()==150
        assert not train[['id','prompt_id','model','text']].isna().any().any()
        assert set(old.fold)=={1,2,3,4,5} and old.groupby('prompt_id').fold.nunique().eq(1).all()
        labels = sorted(train.model.unique())
        assert len(labels)==18
        y = train.model.map({label:i for i,label in enumerate(labels)}).to_numpy()
        assert np.isclose(balanced_accuracy_score(train.model,old[REFERENCE]),.35413124533929907)
        save_json(out_dir/'label_order.json',labels)
        tokenizer = AutoTokenizer.from_pretrained(model_dir,local_files_only=True,use_fast=False)
        tokenizer.truncation_side = 'right'
        full_lengths = np.array([len(x) for x in tokenizer(train.text.tolist(),truncation=False,
                                                         padding=False)['input_ids']])
        encoded = tokenizer(train.text.tolist(),truncation=True,max_length=MAX_LENGTH,padding=False)
        assert max(len(x) for x in encoded['input_ids'])<=MAX_LENGTH
        save_json(out_dir/'tokenization.json',{'max_length':MAX_LENGTH,'use_fast':False,
            'pad_to_multiple_of':8,'truncated_rows':int((full_lengths>MAX_LENGTH).sum()),
            'full_token_length_quantiles':{str(q):float(np.quantile(full_lengths,q)) for q in [0,.5,.9,.95,1]}})
        # Validate the real offline graph without fitting or evaluating held-out labels.
        reset_seed()
        model = make_model(model_dir,labels).eval()
        with torch.inference_mode(),torch.autocast(device_type='cuda',dtype=torch.float16):
            probe = tokenizer(['Offline model load check.','A second short input.'],padding=True,
                               pad_to_multiple_of=8,return_tensors='pt')
            logits = model(**{k:v.to('cuda:0') for k,v in probe.items()}).logits
        assert logits.shape==(2,18) and torch.isfinite(logits).all()
        del model, logits, probe
        model = None
        gc.collect()
        torch.cuda.empty_cache()
        state['status']='running'
        state['preflight_passed_utc']=utc_now()
        save_json(status_path,state)
        oof = old[['id','prompt_id','model','fold',REFERENCE]].copy()
        oof[VARIANT] = None
        all_logits = np.full((len(train),18),np.nan,dtype=np.float32)
        print('PREFLIGHT PASSED:',json.dumps(env),flush=True)
        for fold in range(1,6):
            current_fold = fold
            check_budget(started)
            reset_seed()
            model = make_model(model_dir,labels)
            valid = old.fold.eq(fold).to_numpy()
            fit = ~valid
            assert set(train.loc[fit,'prompt_id']).isdisjoint(train.loc[valid,'prompt_id'])
            assert set(train.loc[fit,'model'])==set(labels)==set(train.loc[valid,'model'])
            support = np.bincount(y[fit],minlength=18)
            class_weights = 1.0/support
            class_weights /= class_weights.mean()
            loss_weights = torch.tensor(class_weights,dtype=torch.float32,device='cuda:0')
            generator = torch.Generator().manual_seed(SEED)
            loader = DataLoader(TokenRows(encoded,y,np.flatnonzero(fit)),batch_size=BATCH_SIZE,
                shuffle=True,generator=generator,num_workers=0,collate_fn=make_collator(tokenizer))
            total_steps = EPOCHS * math.ceil(len(loader)/ACCUMULATION)
            warmup_steps = math.ceil(total_steps*.10)
            optimizer = torch.optim.AdamW(parameter_groups(model),lr=2e-5,betas=(.9,.999),eps=1e-8)
            scheduler = get_linear_schedule_with_warmup(optimizer,warmup_steps,total_steps)
            scaler = torch.amp.GradScaler('cuda',init_scale=1024.0,growth_interval=2000)
            fold_started = time.perf_counter()
            update_count, overflow_count = 0, 0
            for epoch in range(1,EPOCHS+1):
                model.train()
                epoch_loss_sum, epoch_rows = 0.0, 0
                epoch_started = time.perf_counter()
                iterator = iter(loader)
                while True:
                    group = list(itertools.islice(iterator,ACCUMULATION))
                    if not group:
                        break
                    check_budget(started)
                    group_rows = sum(len(batch['labels']) for batch in group)
                    optimizer.zero_grad(set_to_none=True)
                    for batch in group:
                        targets = batch.pop('labels').to('cuda:0')
                        inputs = {k:v.to('cuda:0') for k,v in batch.items()}
                        with torch.autocast(device_type='cuda',dtype=torch.float16):
                            logits = model(**inputs).logits
                            loss_sum = weighted_loss_sum(logits,targets,loss_weights)
                            loss = loss_sum/group_rows
                        if not torch.isfinite(loss):
                            raise FloatingPointError('Nonfinite training loss')
                        scaler.scale(loss).backward()
                        epoch_loss_sum += float(loss_sum.detach().cpu())
                        epoch_rows += len(targets)
                    scaler.unscale_(optimizer)
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                    previous_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    if scaler.get_scale()<previous_scale or not torch.isfinite(norm):
                        overflow_count += 1
                        save_json(out_dir/f'fold_{fold}_overflow.json',{'fold':fold,'epoch':epoch,
                            'completed_optimizer_steps':update_count,'overflow_count':overflow_count,
                            'old_scale':previous_scale,'new_scale':scaler.get_scale()})
                        raise FloatingPointError('Nonfinite gradients/AMP overflow; trial retained as incomplete')
                    scheduler.step()
                    update_count += 1
                    if update_count%25==0:
                        print(f'fold={fold} epoch={epoch} update={update_count}/{total_steps} '
                              f'elapsed={time.perf_counter()-started:.1f}s',flush=True)
                assert epoch_rows==int(fit.sum())
                record = {'fold':fold,'epoch':epoch,'rows':epoch_rows,
                    'mean_weighted_training_loss':epoch_loss_sum/epoch_rows,
                    'cumulative_optimizer_steps':update_count,'amp_overflow_steps':overflow_count,
                    'seconds':time.perf_counter()-epoch_started,'validation_performed':False}
                training_log.append(record)
                save_json(out_dir/'training_log.json',training_log)
                print('EPOCH:',json.dumps(record),flush=True)
            assert update_count==total_steps
            for parameter in model.parameters():
                if not torch.isfinite(parameter).all():
                    raise FloatingPointError('Nonfinite final model parameter')
            model.eval()
            valid_loader = DataLoader(TokenRows(encoded,y,np.flatnonzero(valid)),batch_size=BATCH_SIZE,
                shuffle=False,num_workers=0,collate_fn=make_collator(tokenizer))
            scores = []
            for batch in valid_loader:
                check_budget(started)
                batch.pop('labels')
                with torch.inference_mode(),torch.autocast(device_type='cuda',dtype=torch.float16):
                    logits = model(**{k:v.to('cuda:0') for k,v in batch.items()}).logits
                scores.append(logits.float().cpu().numpy())
            scores = np.concatenate(scores)
            assert scores.shape==(int(valid.sum()),18) and np.isfinite(scores).all()
            all_logits[valid] = scores
            predictions = np.asarray(labels)[scores.argmax(axis=1)]
            oof.loc[valid,VARIANT] = predictions
            score = float(balanced_accuracy_score(train.loc[valid,'model'],predictions))
            checkpoint_dir = out_dir/f'fold_{fold}_checkpoint'
            model.save_pretrained(checkpoint_dir,safe_serialization=True)
            tokenizer.save_pretrained(checkpoint_dir)
            checkpoint_hashes = {p.name:file_sha(p) for p in sorted(checkpoint_dir.glob('*')) if p.is_file()}
            fold_rows.append({'variant':VARIANT,'fold':fold,'balanced_accuracy':score,
                'rows':int(valid.sum()),'prompts':int(train.loc[valid,'prompt_id'].nunique()),
                'seconds':time.perf_counter()-fold_started,'epochs':EPOCHS,'optimizer_steps':update_count,
                'amp_overflow_steps':overflow_count,'warmup_steps':warmup_steps,
                'training_class_support':support.tolist(),'training_class_weights':class_weights.tolist(),
                'checkpoint_sha256':checkpoint_hashes})
            save_json(out_dir/'fold_records.json',fold_rows)
            oof.to_csv(out_dir/'oof_predictions.csv',index=False)
            np.save(out_dir/'oof_logits.npy',all_logits,allow_pickle=False)
            state['completed_folds'].append(fold)
            save_json(status_path,state)
            print(f'FINAL_EPOCH_FOLD_SCORE fold={fold} balanced_accuracy={score:.9f}',flush=True)
            loss,loss_sum,inputs,targets,norm,parameter = None,None,None,None,None,None
            del model,optimizer,scheduler,scaler,loss_weights,loader,valid_loader
            model,optimizer,scheduler,scaler = None,None,None,None
            gc.collect()
            torch.cuda.empty_cache()
        assert oof[VARIANT].notna().all() and np.isfinite(all_logits).all()
        variants = [VARIANT,REFERENCE]
        comparisons,recall_rows,length_rows,fold_table = [],[],[],[]
        bins = pd.cut(train.text.str.split().str.len(),[-np.inf,80,150,np.inf],
                      labels=['<=80 words','81-150 words','>150 words'])
        for variant in variants:
            score = float(balanced_accuracy_score(oof.model,oof[variant]))
            cm = confusion_matrix(oof.model,oof[variant],labels=labels)
            recalls = cm.diagonal()/cm.sum(axis=1)
            assert np.isclose(score,recalls.mean(),atol=1e-12)
            comparisons.append({'variant':variant,'balanced_accuracy':score})
            recall_rows.extend({'variant':variant,'model':label,'recall':float(value),'support':int(support)}
                               for label,value,support in zip(labels,recalls,cm.sum(axis=1)))
            for fold in range(1,6):
                selected = oof.fold.eq(fold)
                fold_table.append({'variant':variant,'fold':fold,
                    'balanced_accuracy':float(balanced_accuracy_score(oof.loc[selected,'model'],oof.loc[selected,variant]))})
            for name in bins.cat.categories:
                selected = bins.eq(name)
                length_rows.append({'variant':variant,'word_count_bin':name,'rows':int(selected.sum()),
                    'classes_present':int(train.loc[selected,'model'].nunique()),
                    'balanced_accuracy_present_classes':float(balanced_accuracy_score(oof.loc[selected,'model'],oof.loc[selected,variant]))})
        interval = paired_interval(oof,labels)
        pd.DataFrame(comparisons).to_csv(out_dir/'comparison.csv',index=False)
        pd.DataFrame(fold_table).to_csv(out_dir/'fold_scores.csv',index=False)
        pd.DataFrame(recall_rows).to_csv(out_dir/'per_model_recall.csv',index=False)
        pd.DataFrame(length_rows).to_csv(out_dir/'length_diagnostics.csv',index=False)
        for i,label in enumerate(labels):
            oof[f'logit_{i:02d}'] = all_logits[:,i]
        oof.to_csv(out_dir/'oof_predictions.csv',index=False,float_format='%.9g')
        result = {**state,'status':'complete','finished_utc':utc_now(),
            'source_sha256':SOURCE_SHA,'reference_sha256':REFERENCE_SHA,
            'model_id':MODEL_ID,'revision':REVISION,'comparisons':comparisons,
            'paired_difference':interval,'fold_records':fold_rows,'warnings':warning_log,
            'environment':env,'total_seconds':time.perf_counter()-started,
            'selection':'Final epoch 3, fixed prospectively; no validation-driven epoch or parameter selection.',
            'official_submission_made':False,'private_test_result':None}
        save_json(out_dir/'metrics.json',result)
        state.update(status='complete',finished_utc=result['finished_utc'])
        save_json(status_path,state)
        print('TRIAL_COMPLETE_JSON_BEGIN',flush=True)
        print(json.dumps(result,indent=2),flush=True)
        print('TRIAL_COMPLETE_JSON_END',flush=True)
        print(pd.DataFrame(recall_rows).to_string(index=False),flush=True)
        return result
    except Exception as error:
        state.update(status='incomplete',failed_utc=utc_now(),error_type=type(error).__name__,
                     error=str(error),current_fold=current_fold)
        save_json(status_path,state)
        (out_dir/'failure_traceback.txt').write_text(traceback.format_exc())
        save_json(out_dir/'training_log.json',training_log)
        if model is not None:
            try:
                model.save_pretrained(out_dir/f'incomplete_fold_{current_fold}',safe_serialization=True)
            except Exception as checkpoint_error:
                (out_dir/'checkpoint_save_error.txt').write_text(str(checkpoint_error))
        print('TRIAL_INCOMPLETE:',json.dumps(state),flush=True)
        raise
    finally:
        warnings.showwarning = prior_showwarning


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--resource-dir',type=Path,required=True)
    parser.add_argument('--out-dir',type=Path,required=True)
    args = parser.parse_args()
    run(args.data_dir,args.resource_dir,args.out_dir)
