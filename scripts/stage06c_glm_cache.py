"""Frozen S02 GLM cache only; no classifier fitting or competition submission.

Run on Kaggle with internet off and exact GLM input V358683059 attached.
The input-version binding must be verified in Kaggle's Input UI; filesystem
names alone cannot prove it. This runner records every local model file hash.
"""
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

TRAIN_SHA = '40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707'
LAYERS = [8, 3, 24]
SOURCE_VERSION = 358683059
SOURCE_SLUG = 'name-that-llm-glm4-9b-trunc25'
RECIPE_SOURCE_SHA = 'e3d97306f3185ea5f73d7904d81331b27803e3ceb45970596590ec306969762e'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    temporary.replace(path)


def extract_glm_cache(input_root='/kaggle/input', output_root='/kaggle/working/stage06c'):
    """Extract (2699,3,4096) float32 features, or verify a completed cache.

    Matches the S02 length ordering, BOS insertion, tokenizer settings, layer
    indexing and masked mean pooling. A nonfinite fp16 batch triggers the
    source's one-way switch to bf16, with an additional strict finite check.
    """
    for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE'):
        os.environ[key] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    import numpy as np
    import pandas as pd

    started = time.monotonic()
    input_root, output_root = Path(input_root), Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    cache_path = output_root / 'glm_features.npz'
    manifest_path = output_root / 'glm_manifest.json'
    status_path = output_root / 'glm_status.json'
    state = {'status': 'PREFLIGHT', 'source_version_expected': SOURCE_VERSION,
             'source_version_filesystem_verified': False, 'layers': LAYERS,
             'source_recipe_sha256': RECIPE_SOURCE_SHA, 'nonfinite_events': []}

    def checkpoint(status):
        state['status'] = status
        state['seconds'] = time.monotonic() - started
        _atomic_json(status_path, state)

    try:
        # Refuse to replace any prior result with a failed or partial rerun.
        existing_complete = manifest_path.exists() and json.loads(manifest_path.read_text()).get('status') == 'COMPLETE'
        if not existing_complete:
            assert not cache_path.exists(), 'Unverified cache exists; preserve and investigate it before running.'
            assert not manifest_path.exists(), 'Existing non-complete manifest must be investigated, not overwritten.'
        train_paths = [p for p in input_root.rglob('train.csv') if sha256(p) == TRAIN_SHA]
        assert len(train_paths) == 1, ('Expected one exact official train.csv', len(train_paths))
        train_path = train_paths[0]
        train = pd.read_csv(train_path, dtype=str)
        assert len(train) == 2699 and train.id.is_unique and train.text.notna().all()
        ids = train.id.to_numpy(dtype=str)
        texts = train.text.to_numpy(dtype=str)
        text_hashes = np.asarray([hashlib.sha256(t.encode('utf-8')).hexdigest() for t in texts], dtype=str)
        ordered_text_hash = _json_hash(text_hashes.tolist())
        configs = []
        for path in input_root.rglob('config.json'):
            if SOURCE_SLUG not in path.parts:
                continue
            cfg = json.loads(path.read_text())
            if cfg.get('model_type') == 'glm4' and cfg.get('hidden_size') == 4096:
                configs.append((path, cfg))
        assert len(configs) == 1, ('Expected one GLM model under pinned source slug', [str(p) for p, _ in configs])
        config_path, raw_config = configs[0]
        assert raw_config['num_hidden_layers'] == 25, 'Expected the exact 25-layer truncated source.'
        model_root = config_path.parent
        print('GLM_INPUT_HASHING', str(model_root), flush=True)
        file_records = [{'path': str(p.relative_to(model_root)), 'bytes': p.stat().st_size, 'sha256': sha256(p)}
                        for p in sorted(model_root.rglob('*')) if p.is_file()]
        weights = [r for r in file_records if r['path'].endswith(('.safetensors', '.bin'))]
        tokenizers = [r for r in file_records if any(s in r['path'].lower() for s in ('token', 'vocab', 'merges', 'special'))
                      and r not in weights]
        assert weights and tokenizers
        input_provenance = dict(train_sha256=TRAIN_SHA, ordered_text_sha256=ordered_text_hash,
                                model_files_sha256=_json_hash(weights), tokenizer_files_sha256=_json_hash(tokenizers),
                                config_sha256=sha256(config_path), all_model_files_sha256=_json_hash(file_records))
        state.update(input_provenance)
        state.update(model_root=str(model_root), train_path=str(train_path), rows=len(train), hidden_size=4096)

        if existing_complete:
            old = json.loads(manifest_path.read_text())
            assert old['input_provenance'] == input_provenance
            assert old['layers'] == LAYERS and old['cache_sha256'] == sha256(cache_path)
            with np.load(cache_path, allow_pickle=False) as z:
                assert np.array_equal(z['ids'], ids) and np.array_equal(z['text_sha256'], text_hashes)
                assert np.array_equal(z['layers'], LAYERS) and z['train_sha256'].item() == TRAIN_SHA
                assert z['features'].shape == (2699, 3, 4096) and z['features'].dtype == np.float32
                assert np.isfinite(z['features']).all()
            print('GLM_EXISTING_COMPLETE_CACHE_VERIFIED', old['cache_sha256'], flush=True)
            return cache_path, old

        checkpoint('CUDA_PREFLIGHT')
        import torch
        import transformers
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        assert torch.cuda.is_available() and torch.cuda.device_count() == 2, 'Requires the selected 2-GPU Kaggle runtime.'
        gpu = [dict(index=i, name=torch.cuda.get_device_name(i),
                    total_memory=torch.cuda.get_device_properties(i).total_memory,
                    free_memory=torch.cuda.mem_get_info(i)[0]) for i in range(2)]
        assert all('T4' in device['name'] for device in gpu), gpu
        print('GLM_CUDA_PREFLIGHT', json.dumps(gpu), flush=True)
        state['environment'] = dict(python=sys.version, platform=platform.platform(), torch=torch.__version__,
                                    transformers=transformers.__version__, numpy=np.__version__, pandas=pd.__version__,
                                    cuda=torch.version.cuda, gpu=gpu)
        tokenizer = AutoTokenizer.from_pretrained(model_root, local_files_only=True)
        tokenizer.padding_side = 'right'
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        assert tokenizer.pad_token_id is not None
        batches = [(tokenizer.bos_token + text) if tokenizer.bos_token else text for text in texts]
        lengths = np.asarray([len(tokenizer(text, add_special_tokens=False, truncation=False)['input_ids'])
                              for text in batches], dtype=np.int64)
        state['tokenization'] = dict(max_length=512, batch_size=8, add_special_tokens=False, padding_side='right',
                                     bos_prepend=tokenizer.bos_token, pad_token=tokenizer.pad_token,
                                     truncated_rows=int((lengths > 512).sum()), max_original_tokens=int(lengths.max()),
                                     min_original_tokens=int(lengths.min()), includes_bos_in_mean=True)
        config = AutoConfig.from_pretrained(model_root, local_files_only=True)
        config.num_hidden_layers = 25
        if getattr(config, 'layer_types', None):
            config.layer_types = config.layer_types[:25]
        checkpoint('MODEL_LOADING')
        model = AutoModel.from_pretrained(model_root, config=config, dtype=torch.float16,
                                         device_map='auto', local_files_only=True).eval()
        mapping = {str(k): str(v) for k, v in getattr(model, 'hf_device_map', {}).items()}
        assert mapping and all(v not in ('cpu', 'disk') for v in mapping.values()), mapping
        assert all(p.device.type == 'cuda' for p in model.parameters()), 'CPU or disk model parameter found.'
        state['device_map'] = mapping
        state['model_parameter_dtypes_initial'] = sorted({str(p.dtype) for p in model.parameters()})
        assert state['model_parameter_dtypes_initial'] == ['torch.float16']
        device = next(model.parameters()).device
        # np.argsort without stable=True intentionally matches the published source.
        order = np.argsort([len(text) for text in texts])
        features = np.empty((len(texts), len(LAYERS), 4096), dtype=np.float32)
        coverage = np.zeros(len(texts), dtype=np.int8)
        checkpoint('EXTRACTING')
        with torch.no_grad():
            for b0 in range(0, len(texts), 8):
                idx = order[b0:b0 + 8]
                encoded = tokenizer([batches[i] for i in idx], return_tensors='pt', padding=True,
                                    truncation=True, max_length=512, add_special_tokens=False).to(device)
                hidden = model(**encoded, output_hidden_states=True).hidden_states
                bad_layers = [layer for layer in LAYERS if not torch.isfinite(hidden[layer]).all().item()]
                if bad_layers:
                    state['nonfinite_events'].append(dict(batch_start=b0, row_indices=idx.tolist(),
                                                         layers=bad_layers, dtype=str(next(model.parameters()).dtype)))
                    checkpoint('NONFINITE_RETRY_BFLOAT16')
                    del hidden
                    model.to(torch.bfloat16)
                    hidden = model(**encoded, output_hidden_states=True).hidden_states
                    assert all(torch.isfinite(hidden[layer]).all().item() for layer in LAYERS), 'Nonfinite hidden state after bf16 retry.'
                for li, layer in enumerate(LAYERS):
                    h = hidden[layer]
                    mask = encoded['attention_mask'].float().unsqueeze(-1).to(h.device)
                    pooled = ((h.float() * mask).sum(1) / mask.sum(1).clamp(min=1)).cpu().numpy()
                    assert pooled.shape == (len(idx), 4096) and np.isfinite(pooled).all()
                    features[idx, li] = pooled
                coverage[idx] += 1
                del hidden, encoded
                if b0 % 80 == 0 or b0 + 8 >= len(texts):
                    state.update(completed_rows=int(coverage.sum()), completed_batches=b0 // 8 + 1)
                    checkpoint('EXTRACTING')
                    print('GLM_EXTRACTION_BATCH', state['completed_batches'], state['completed_rows'],
                          'seconds', round(state['seconds'], 1), flush=True)
        assert np.all(coverage == 1) and np.isfinite(features).all()
        state['model_parameter_dtypes_final'] = sorted({str(p.dtype) for p in model.parameters()})
        del model
        torch.cuda.empty_cache()
        temporary = output_root / 'glm_features.tmp.npz'
        np.savez_compressed(temporary, features=features, ids=ids, layers=np.asarray(LAYERS, dtype=np.int64),
                            text_sha256=text_hashes, token_lengths=lengths, train_sha256=np.asarray(TRAIN_SHA))
        assert not cache_path.exists(), 'Output appeared during extraction; refusing replacement.'
        temporary.replace(cache_path)
        with np.load(cache_path, allow_pickle=False) as check:
            assert np.array_equal(check['features'], features) and np.array_equal(check['ids'], ids)
        state['cache_sha256'] = sha256(cache_path)
        manifest = dict(state, status='COMPLETE', seconds=time.monotonic() - started, layers=LAYERS,
                        input_provenance=input_provenance, model_files=file_records,
                        shape=list(features.shape), dtype=str(features.dtype),
                        limitations=['Kaggle Input version binding requires UI confirmation.',
                                     'Frozen inference cache only; no classification or official score.'])
        _atomic_json(manifest_path, manifest)
        checkpoint('COMPLETE')
        print('GLM_CACHE_COMPLETE', json.dumps({k: manifest[k] for k in ('cache_sha256', 'shape', 'seconds')}), flush=True)
        return cache_path, manifest
    except BaseException as error:
        state['error'] = {'type': type(error).__name__, 'message': str(error)}
        # Completed artifacts remain untouched, including status, on failed reuse verification.
        if not locals().get('existing_complete', False):
            checkpoint('FAILED')
        print('GLM_CACHE_FAILED', json.dumps(state.get('error')), flush=True)
        raise


if __name__ == '__main__':
    extract_glm_cache()
