"""CPU-only input audit for Kirill source V1 361634630; never run its source.

Uses stdlib, NumPy and pandas. No Torch, Transformers, joblib, pickle, model
imports, custom code, fitting, inference or network calls are permitted here.
Custom Python/config text is copied only into a separate PRIVATE review export.
"""
from __future__ import annotations
import argparse
import ast
import glob
import hashlib
import json
import re
import struct
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE_VERSION = 361634630
UPSTREAM_SHA = "1d302075c9b5f620db265a147cc11138e786a11ea754c262a33b9223879905de"
COMP_STACK_SHA = "c09e4dcb31fd6cc4c34578c053b22dc425eb8a3a341a1db579e1fd733598c792"
TRAIN_SHA = "40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707"
LABELS = ['gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',
          'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
          'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5', 'claude-fable-5',
          'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3']
SORTED_LABELS = sorted(LABELS)
EXTRA_SOURCES = [('**/ntl_extra_passages_claude.csv', 'model', None), ('**/gen_passages.csv', 'model', None),
                 ('**/ntl_generations.csv', 'label', 'finish_reason'), ('**/ntl_proxy_generations.csv', 'label', None),
                 ('**/ntl_generated*.csv', 'model', 'finish'), ('**/passages_clean.csv', 'model', None)]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    path = Path(path)
    if path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError('JSON metadata exceeds bounded reader: ' + path.name)
    return json.loads(path.read_text(encoding='utf-8'))


def safe_relative(name):
    path = Path(str(name))
    if path.is_absolute() or '..' in path.parts or not path.parts or '\x00' in str(name):
        raise ValueError('Unsafe relative asset path')
    return path


def checked_labels(value, ordered=None):
    if isinstance(value, dict):
        value = list(value.values())
    if not isinstance(value, list) or len(value) != 18 or set(value) != set(LABELS):
        raise ValueError('Checkpoint label set differs from the 18 official labels')
    if ordered is not None and value != ordered:
        raise ValueError('Checkpoint label order differs from its frozen source head')
    return value


def safetensor_header(path):
    path = Path(path)
    with path.open('rb') as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError('Truncated safetensors length prefix')
        size = struct.unpack('<Q', raw)[0]
        if size < 2 or size > 16 * 1024 * 1024 or size + 8 > path.stat().st_size:
            raise ValueError('Invalid safetensors header length')
        header = json.loads(f.read(size))
    result = {}
    for key, value in header.items():
        if key == '__metadata__':
            continue
        shape, offsets = value['shape'], value['data_offsets']
        if (not isinstance(shape, list) or not all(isinstance(x, int) and x >= 0 for x in shape)
                or not isinstance(offsets, list) or len(offsets) != 2 or offsets[0] < 0
                or offsets[1] < offsets[0] or offsets[1] > path.stat().st_size - size - 8):
            raise ValueError('Invalid safetensors tensor extent')
        result[key] = {'shape': shape, 'dtype': value['dtype'], 'data_offsets': offsets}
    if not result:
        raise ValueError('Empty safetensors tensor inventory')
    return result


def npz_inventory(path, expected_classes=None, expected_layers=None, hidden=None):
    result = {}
    with np.load(path, allow_pickle=False) as data:
        for key in data.files:
            try:
                value = data[key]
            except ValueError as error:
                raise ValueError('Object/pickle NPZ metadata is not admitted by this CPU audit: ' + key) from error
            item = {'shape': list(value.shape), 'dtype': str(value.dtype)}
            if value.dtype.kind in 'fciu':
                if not np.isfinite(value).all():
                    raise ValueError('Nonfinite numeric NPZ member: ' + key)
                item['finite'] = True
            result[key] = item
        if expected_classes is not None:
            checked_labels(data['classes'].tolist(), expected_classes)
        if expected_layers is not None and data['pool_layers'].tolist() != expected_layers:
            raise ValueError('LoRA pooled layers differ from frozen source')
        if hidden is not None and expected_layers is not None:
            dimension = hidden * len(expected_layers)
            for key, shape in [('plain_coef', (18, dimension)), ('plain_intercept', (18,)),
                               ('center_coef', (18, dimension)), ('center_intercept', (18,))]:
                if key not in data or data[key].shape != shape:
                    raise ValueError('Required LoRA readout missing or dimension mismatch: ' + key)
            if 'claude_coef' in data:
                if data['claude_coef'].shape != (5, dimension) or data['claude_intercept'].shape != (5,):
                    raise ValueError('Optional Claude gate dimensions mismatch')
    return result


def validate_recovery(manifest_path):
    manifest_path = Path(manifest_path)
    doc = read_json(manifest_path)
    if doc.get('independently_verified_published_migration_manifest') is not True:
        raise ValueError('Recovery manifest does not declare published migration verification')
    records = doc.get('files')
    if not isinstance(records, dict) or not records:
        raise ValueError('Recovery archive has no asset records')
    verified = {}
    for relative, record in records.items():
        safe_relative(relative)
        flat = safe_relative(record['flat_file'])
        origin = manifest_path.parent / flat
        if not origin.resolve().is_relative_to(manifest_path.parent.resolve()) or not origin.is_file():
            raise ValueError('Missing/escaping recovery archive asset')
        expected = record['sha256']
        if not isinstance(expected, str) or not re.fullmatch('[0-9a-f]{64}', expected) or sha(origin) != expected:
            raise ValueError('Recovery archive asset SHA mismatch: ' + str(relative))
        verified[str(relative)] = {'flat_file': str(flat), 'bytes': origin.stat().st_size, 'sha256': expected}
    required = {'COMPLETE.json', 'readouts.npz', 'heads.safetensors',
                'adapter/adapter_config.json', 'adapter/adapter_model.safetensors', 'tokenizer/tokenizer_config.json'}
    if not required.issubset(verified):
        raise ValueError('Recovery archive omits mandatory lora24/final assets: ' + ','.join(sorted(required - set(verified))))
    readout_path = manifest_path.parent / verified['readouts.npz']['flat_file']
    readout = npz_inventory(readout_path, SORTED_LABELS, [2, 4, 6, 8, 12, 16, 20, 24], hidden=4096)
    head_path = manifest_path.parent / verified['heads.safetensors']['flat_file']
    heads = safetensor_header(head_path)
    return {'manifest_sha256': sha(manifest_path), 'verified_files': verified,
            'readouts': readout, 'head_tensor_headers': heads,
            'provenance_scope': 'Published manifest declaration and actual byte parity; independent original training provenance is not authenticated here.'}


class Audit:
    def __init__(self, root, out):
        self.root, self.out = Path(root), Path(out)
        self.checks, self.warnings, self.details, self.code = [], [], {}, []

    def check(self, name, fn):
        try:
            value = fn()
            self.details[name] = value
            self.checks.append({'name': name, 'status': 'PASS'})
            print('PREFLIGHT_PASS', name, flush=True)
            return value
        except Exception as error:
            self.checks.append({'name': name, 'status': 'FAIL', 'error_type': type(error).__name__, 'error': str(error)[:1200]})
            print('PREFLIGHT_FAIL', name, type(error).__name__, str(error)[:400], flush=True)
            return None

    def matches(self, pattern):
        return [Path(p) for p in sorted(glob.glob(str(self.root / pattern), recursive=True))]

    def unique(self, pattern):
        found = self.matches(pattern)
        if len(found) != 1:
            raise ValueError(f'Required asset {pattern}: expected 1, found {len(found)}')
        return found[0]

    def small_code(self, path, role):
        path = Path(path)
        if path.stat().st_size > 256 * 1024:
            raise ValueError('Custom code/config exceeds private text export bound: ' + path.name)
        content = path.read_text(encoding='utf-8')
        if path.suffix == '.py':
            ast.parse(content)
        digest = sha(path)
        record = {'path': str(path.relative_to(self.root)), 'role': role, 'bytes': path.stat().st_size, 'sha256': digest}
        identical = next((item for item in self.code if item['sha256'] == digest and 'text' in item), None)
        if identical is None:
            record['text'] = content
        else:
            record['same_text_as'] = identical['path']
        self.code.append(record)
        return {'path': str(path.relative_to(self.root)), 'bytes': path.stat().st_size, 'sha256': digest, 'executed': False}

    def tokenizer_inventory(self, directory):
        directory = Path(directory)
        assets = [p for p in directory.iterdir() if p.is_file() and p.name in
                  ['tokenizer.json', 'tokenizer.model', 'spm.model', 'vocab.txt', 'vocab.json', 'merges.txt',
                   'tokenizer_config.json', 'special_tokens_map.json', 'added_tokens.json']]
        if not any(p.name in ['tokenizer.json', 'tokenizer.model', 'spm.model', 'vocab.txt', 'vocab.json'] and p.stat().st_size > 0 for p in assets):
            raise ValueError('Required local tokenizer vocabulary/model is missing')
        return [{'name': p.name, 'bytes': p.stat().st_size} for p in sorted(assets)]

    def model_inventory(self, directory):
        directory = Path(directory)
        configs = read_json(directory / 'config.json')
        files = sorted(p for p in directory.rglob('*') if p.is_file() and '.cache' not in p.parts)
        weights = [p for p in files if p.name.endswith(('.safetensors', '.bin'))]
        if not weights:
            raise ValueError('Base/fold checkpoint has no attached weight files')
        for index in [p for p in files if p.name.endswith('.index.json')]:
            doc = read_json(index)
            for relative in set(doc.get('weight_map', {}).values()):
                candidate = directory / safe_relative(relative)
                if not candidate.is_file():
                    raise ValueError('Missing model index shard')
        return {'path': str(directory.relative_to(self.root)), 'model_type': configs.get('model_type'),
                'hidden_size': configs.get('hidden_size'), 'num_hidden_layers': configs.get('num_hidden_layers'),
                'config_sha256': sha(directory / 'config.json'),
                'weight_inventory': [{'name': str(p.relative_to(directory)), 'bytes': p.stat().st_size} for p in weights],
                'weight_values_or_custom_code_executed': False,
                'all_large_weight_bytes_hashed': False}

    def base(self, name, spec=None, directory_name=None, path_parts=None):
        matches = []
        for path in self.matches('**/config.json'):
            if directory_name is not None and path.parent.name != directory_name:
                continue
            if path_parts is not None and not all(part in str(path) for part in path_parts):
                continue
            try:
                cfg = read_json(path)
            except Exception:
                continue
            if spec is None or all(cfg.get(k) == v for k, v in spec.items()):
                matches.append(path)
        resolution = {'selector': {'spec': spec, 'directory_name': directory_name, 'path_parts': path_parts},
                      'matching_paths_sorted': [str(p.relative_to(self.root)) for p in matches],
                      'source_resolution': 'first sorted match (ATLAS explicit selector requires exactly one)',
                      'source_would_choose': str(matches[0].relative_to(self.root)) if matches else None}
        self.details[name + '_resolution'] = resolution
        if len(matches) != 1:
            raise ValueError(f'{name}: missing/ambiguous base selector; {len(matches)} candidates; source first-match resolution recorded for review')
        result = self.model_inventory(matches[0].parent)
        result['tokenizer_assets'] = self.tokenizer_inventory(matches[0].parent)
        return result

    def main_lora(self, name, hidden, keep):
        meta_path = self.unique('**/' + name + '/meta.json')
        directory, meta = meta_path.parent, read_json(meta_path)
        checked_labels(meta['labels'], LABELS)
        cfg = meta['cfg']; layers = cfg['layers']
        if cfg['keep'] != keep or cfg.get('eos_id') != 151643 or not (1 <= cfg['max_len'] <= 4096):
            raise ValueError('Main LoRA depth/EOS/token contract differs')
        if not layers or not all(isinstance(v, int) and 0 < v <= keep for v in layers):
            raise ValueError('Main LoRA pooling layers invalid')
        adapter = directory / 'adapter/adapter_model.safetensors'
        acfg = read_json(directory / 'adapter/adapter_config.json')
        if acfg.get('r', 0) <= 0 or acfg.get('lora_alpha', 0) <= 0:
            raise ValueError('Main adapter rank/scale invalid')
        heads = safetensor_header(directory / 'heads.safetensors')
        for prefix in ['plain', 'centered']:
            if heads[prefix + '.weight']['shape'] != [18, hidden * len(layers)] or heads[prefix + '.bias']['shape'] != [18]:
                raise ValueError('Main LoRA readout tensor shape differs')
        return {'path': str(directory.relative_to(self.root)), 'meta_sha256': sha(meta_path), 'cfg': cfg,
                'adapter_sha256': sha(adapter), 'adapter_headers': safetensor_header(adapter), 'head_headers': heads}

    def component_lora(self, pattern, layers):
        marker = self.unique(pattern)
        directory = marker.parent
        cfg = read_json(directory / 'adapter/adapter_config.json')
        header = safetensor_header(directory / 'adapter/adapter_model.safetensors')
        norm = directory / ('norms.safetensors' if (directory / 'norms.safetensors').exists() else 'heads.safetensors')
        if not (directory / 'tokenizer/tokenizer_config.json').is_file():
            raise ValueError('Component adapter tokenizer missing')
        hidden = 3584 if layers[-1] == 20 else 4096
        return {'path': str(directory.relative_to(self.root)), 'complete_sha256': sha(marker),
                'adapter_sha256': sha(directory / 'adapter/adapter_model.safetensors'), 'adapter_cfg': cfg,
                'adapter_header_tensor_count': len(header), 'norm_headers': safetensor_header(norm),
                'tokenizer_assets': self.tokenizer_inventory(directory / 'tokenizer'),
                'readouts': npz_inventory(directory / 'readouts.npz', SORTED_LABELS, layers, hidden=hidden)}

    def bank(self):
        bank = self.unique('**/synthetic4404.joblib')
        receipt = read_json(bank.parent / 'COMPLETE.json')
        if (not receipt.get('full_fit') or receipt.get('fold_adapter') or receipt.get('adapter_training')
                or receipt.get('raw_synthetic_data_in_export')):
            raise ValueError('Night24 bank declared fit/export contract differs')
        rec = receipt['banks']['synthetic4404']
        bank_sha = sha(bank)
        if rec['official_rows'] != 2699 or rec['synthetic_rows'] != 4404 or bank_sha != rec['readout_sha256']:
            raise ValueError('Night24 bank row count or actual SHA differs')
        implementation_sha = sha(bank.parent / 'readouts.py')
        declared_code_hashes = {k: v for k, v in receipt.items() if isinstance(v, str) and
                                'readout' in k.lower() and ('py' in k.lower() or 'code' in k.lower()) and 'sha' in k.lower()}
        for expected in declared_code_hashes.values():
            if expected != implementation_sha:
                raise ValueError('Night24 readouts.py differs from its declared implementation SHA')
        if not declared_code_hashes:
            self.warnings.append('Night24 COMPLETE does not directly declare a readouts.py implementation SHA; observed bytes are exported and hash-pinned for separate static review.')
        return {'path': str(bank.relative_to(self.root)), 'bytes': bank.stat().st_size, 'sha256': bank_sha,
                'complete_sha256': sha(bank.parent / 'COMPLETE.json'), 'official_rows': 2699, 'synthetic_rows': 4404,
                'readouts_python': self.small_code(bank.parent / 'readouts.py', 'Night24 custom readout implementation'),
                'declared_readouts_code_hashes': declared_code_hashes,
                'joblib_deserialized': False,
                'bank_internal_adapter_sha_alignment': 'Not inspected without deserializing; original runtime must verify adapter hash, label order and pool layers.'}

    def pools(self):
        pool_path = self.unique('**/pool_v4.parquet')
        pool = pd.read_parquet(pool_path)
        if not {'text', 'model'}.issubset(pool.columns) or len(pool) == 0:
            raise ValueError('Main SVC pool missing schema/rows')
        if pool.text.isna().any() or not pool.text.map(lambda x: isinstance(x, str) and bool(x.strip())).all():
            raise ValueError('Main SVC pool has empty/non-string text')
        if not pool.model.isin(LABELS).all() or set(pool.model) != set(LABELS):
            raise ValueError('Main SVC pool label set differs')
        parts, records = [], []
        for pattern, label_column, finish_column in EXTRA_SOURCES:
            for path in self.matches(pattern):
                frame = pd.read_csv(path)
                if not {'prompt_id', label_column, 'text'}.issubset(frame.columns):
                    raise ValueError('Required extra SVC CSV schema mismatch: ' + str(path.relative_to(self.root)))
                rows = len(frame)
                if finish_column and finish_column in frame:
                    frame = frame[frame[finish_column].eq('stop')]
                frame = frame.rename(columns={label_column: 'model'})[['prompt_id', 'model', 'text']].copy()
                frame['prompt_id'] = ('pv_' if path.name == 'passages_clean.csv' else '') + frame.prompt_id.astype(str)
                parts.append(frame)
                records.append({'path': str(path.relative_to(self.root)), 'sha256': sha(path), 'rows_before_source_finish_filter': rows,
                                'rows_after_source_finish_filter': len(frame), 'source_finish_column': finish_column})
        if not parts:
            raise ValueError('No external SVC passage sources resolved; source would silently train a different no-extra variant')
        extra = pd.concat(parts, ignore_index=True)
        extra = extra[extra.model.isin(LABELS)].copy()
        extra['text'] = extra.text.astype(str).map(lambda t: re.sub(r'\s+', ' ', re.sub(r'^\*\*(.*)\*\*$', r'\1', t.strip())).strip())
        extra = extra[extra.text.str.split().str.len().between(30, 300)].drop_duplicates(['model', 'text'])
        extra = extra.drop_duplicates(['prompt_id', 'model'])
        extra = extra[extra.groupby('prompt_id').model.transform('nunique') >= 12]
        if len(extra) == 0 or extra.text.str.strip().eq('').any():
            raise ValueError('Filtered extra SVC pool is empty')
        quality = []
        for path in self.matches('**/quality_pool_public.parquet'):
            frame = pd.read_parquet(path)
            valid_schema = {'text', 'model'}.issubset(frame.columns)
            quality.append({'path': str(path.relative_to(self.root)), 'sha256': sha(path), 'rows': len(frame),
                            'columns': list(frame.columns), 'nonempty': len(frame) > 0,
                            'text_model_schema': valid_schema,
                            'all_labels_allowed': bool(frame.model.isin(LABELS).all()) if valid_schema else False,
                            'all_text_nonempty_strings': bool(frame.text.map(lambda x: isinstance(x, str) and bool(x.strip())).all()) if valid_schema else False,
                            'direct_outer_or_comp_stack_reference': False,
                            'use_scope': 'Attachment inventory only; no direct reference in frozen outer/comp source. Separate readouts.py review must determine any indirect helper use.'})
        return {'main_pool': {'path': str(pool_path.relative_to(self.root)), 'sha256': sha(pool_path), 'rows': len(pool),
                              'columns': list(pool.columns), 'label_counts': pool.model.value_counts().sort_index().to_dict()},
                'extra_csv_sources': records, 'source_exact_filtered_extra_rows': len(extra),
                'source_exact_filtered_extra_prompts': int(extra.prompt_id.nunique()),
                'source_exact_filtered_extra_label_counts': extra.model.value_counts().sort_index().to_dict(),
                'quality_pool_optional_attachment_inventory': quality,
                'quality_named_assets': [str(p.relative_to(self.root)) for p in self.matches('**/*')
                                         if p.is_file() and 'quality' in str(p).lower() and p.suffix in ['.csv', '.parquet', '.json']],
                'provenance_limit': 'Source-defined final-inference pool reproduced structurally; no blanket provider or prompt-independence certification.'}

    def ultrabert(self):
        candidates = []
        for root in self.matches('**/ultrabert-380m-ntl'):
            folds = [p for p in sorted(root.glob('fold*')) if (p / 'config.json').is_file()
                     and read_json(p / 'config.json').get('model_type') == 'ultrabert']
            if len(folds) == 5:
                candidates.append((root, folds))
        if len(candidates) != 1:
            raise ValueError('Missing/ambiguous Ultrabert root containing exactly five valid folds')
        root, folds = candidates[0]
        result = []
        for fold in folds:
            cfg = read_json(fold / 'config.json')
            checked_labels(cfg['id2label'])
            exported = [self.small_code(fold / 'config.json', 'Ultrabert fold config')]
            for name, mapping in cfg.get('auto_map', {}).items():
                for target in mapping if isinstance(mapping, list) else [mapping]:
                    if target is None:
                        continue
                    if '--' in target:
                        raise ValueError('Remote-repository Ultrabert auto_map requires explicit static resolution')
                    module = target.rsplit('.', 1)[0].replace('.', '/') + '.py'
                    if not (fold / safe_relative(module)).is_file():
                        raise ValueError('Ultrabert auto_map local code file missing')
            for path in sorted(fold.rglob('*.py')):
                exported.append(self.small_code(path, 'Ultrabert custom code; never imported by preflight'))
            tokenizer = fold / 'tokenizer_config.json'
            if tokenizer.is_file():
                exported.append(self.small_code(tokenizer, 'Ultrabert tokenizer config'))
            result.append({'inventory': self.model_inventory(fold), 'private_review_exports': exported})
        if not any((folds[0] / name).is_file() for name in ['tokenizer.json', 'tokenizer.model', 'spm.model', 'vocab.txt', 'vocab.json']):
            raise ValueError('First Ultrabert fold has no local tokenizer asset')
        return {'root': str(root.relative_to(self.root)), 'folds': result, 'trust_remote_code_executed': False}

    def atlas(self):
        path = self.unique('**/atlas_bundle.json'); directory = path.parent
        cfg = read_json(path)
        if not isinstance(cfg.get('training_config'), dict) or cfg['training_config'].get('max_length', 0) <= 0:
            raise ValueError('ATLAS training/token-length config missing')
        required = ['ntl_atlas/model.py', 'ntl_rlcd/data.py']
        exports = [self.small_code(path, 'ATLAS bundle config')]
        for relative in required:
            if not (directory / relative).is_file():
                raise ValueError('ATLAS local custom package asset missing: ' + relative)
        for package in ['ntl_atlas', 'ntl_rlcd']:
            for code in sorted((directory / package).rglob('*.py')):
                exports.append(self.small_code(code, 'ATLAS local code; never imported by preflight'))
        return {'path': str(directory.relative_to(self.root)), 'config_sha256': sha(path),
                'adapter_sha256': sha(directory / 'adapters.safetensors'),
                'adapter_headers': safetensor_header(directory / 'adapters.safetensors'),
                'private_review_exports': exports}

    def execute(self):
        official_candidates = [self.root / 'competitions/name-that-llm', self.root / 'name-that-llm']
        official = next((p for p in official_candidates if (p / 'test.csv').is_file()), None)
        def official_contract():
            if official is None or not (official / 'train.csv').is_file():
                raise ValueError('Official competition train/test path unavailable')
            if sha(official / 'train.csv') != TRAIN_SHA:
                raise ValueError('Official training hash changed')
            frame = pd.read_csv(official / 'train.csv', dtype=str, keep_default_na=False)
            if len(frame) != 2699 or frame.prompt_id.nunique() != 150 or set(frame.model) != set(LABELS):
                raise ValueError('Official dimensions/labels changed')
            return {'train_sha256': TRAIN_SHA, 'rows': len(frame), 'prompts': 150, 'classes': 18,
                    'test_file_opened_or_hashed': False,
                    'preview_warning': 'Frozen exact-three-row preview bypasses ALL model/input paths and emits constant labels; CPU pass is not GPU inference evidence.'}
        self.check('official_train', official_contract)
        self.check('legacy_recovery_archive', lambda: validate_recovery(self.unique('**/legacy_recovery_manifest.json')))
        self.check('night24_synthetic4404_bank', self.bank)
        self.check('main_and_quality_selected_extra_svc_pools', self.pools)
        for name, spec in [
            ('comp_glm9b', {'model_type': 'glm4', 'hidden_size': 4096}),
            ('comp_qwen3_8b', {'model_type': 'qwen3', 'hidden_size': 4096}),
            ('comp_qwen25_7b', {'model_type': 'qwen2', 'hidden_size': 3584}),
            ('comp_qwen3_4b', {'model_type': 'qwen3', 'hidden_size': 2560}),
            ('comp_token_attention_qwen35', {'model_type': 'qwen3_5'}),
            ('main_qwen3_14b', {'model_type': 'qwen3', 'hidden_size': 5120, 'num_hidden_layers': 40}),
            ('main_qwen25_14b', {'model_type': 'qwen2', 'hidden_size': 5120, 'num_hidden_layers': 48}),
        ]:
            self.check(name, lambda name=name, spec=spec: self.base(name, spec))
        for directory in ['deberta-v3-large', 'electra-large-discriminator', 'desklib-detector']:
            self.check('comp_encoder_' + directory, lambda directory=directory: self.base('comp_encoder_' + directory, directory_name=directory))
        self.check('atlas_qwen3_8b_base', lambda: self.base('atlas_qwen3_8b_base', path_parts=['/qwen-3/', '/8b-base/']))
        self.check('main_lora_qwen3_14b', lambda: self.main_lora('lora_qwen3_14b', 5120, 27))
        self.check('main_lora_qwen25_14b', lambda: self.main_lora('lora_qwen25_14b', 5120, 32))
        for name, pattern, layers in [
            ('comp_lora24_seed7', '**/name-that-llm-train-lora24-s7/lora_out/job_final/COMPLETE.json', [2,4,6,8,12,16,20,24]),
            ('comp_lora24_claude', '**/ntl-qwen3-8b-lora-claude-head/**/final/COMPLETE.json', [2,4,6,8,12,16,20,24]),
            ('comp_lora20_q25', '**/ntl-train-lora-qwen25-instyle/lora_out/job_final/COMPLETE.json', [2,4,6,8,10,12,16,20]),
        ]:
            self.check(name, lambda pattern=pattern, layers=layers: self.component_lora(pattern, layers))
        def main_deberta():
            path = self.unique('**/deberta_v3_large/meta.json'); meta = read_json(path)
            checked_labels(meta['labels'], LABELS)
            if not 1 <= meta.get('max_len', 0) <= 4096:
                raise ValueError('Main DeBERTa token length metadata invalid')
            inventory = self.model_inventory(path.parent / 'encoder')
            heads = safetensor_header(path.parent / 'head.safetensors')
            if heads['weight']['shape'] != [18, inventory['hidden_size']] or heads['bias']['shape'] != [18]:
                raise ValueError('Main DeBERTa head dimensions differ')
            return {'meta_sha256': sha(path), 'meta_max_len': meta['max_len'],
                    'encoder': inventory,
                    'tokenizer_assets': self.tokenizer_inventory(path.parent / 'encoder'),
                    'head_headers': heads}
        self.check('main_adapted_deberta', main_deberta)
        for filename in ['tokattn_final.pt', 'tokattn_debl23_final.pt']:
            def pt_asset(filename=filename):
                path = self.unique('**/' + filename)
                if path.stat().st_size == 0:
                    raise ValueError('Empty Torch head file')
                return {'path': str(path.relative_to(self.root)), 'bytes': path.stat().st_size,
                        'sha256': sha(path), 'torch_or_pickle_deserialized': False}
            self.check(filename, pt_asset)
        self.check('ultrabert_five_folds', self.ultrabert)
        self.check('atlas_bundle_and_custom_packages', self.atlas)
        failed = [c for c in self.checks if c['status'] != 'PASS']
        self.out.mkdir(parents=True, exist_ok=True)
        code_export = {'scope': 'PRIVATE static text for review; code has not been executed', 'files': self.code}
        (self.out / 'PRIVATE_CODE_EXPORT.json').write_text(json.dumps(code_export, indent=2) + '\n', encoding='utf-8')
        report = {'stage': '10', 'source_version': SOURCE_VERSION, 'upstream_source_sha256': UPSTREAM_SHA,
                  'embedded_complement_source_sha256': COMP_STACK_SHA,
                  'status': 'PASS_CPU_INPUT_CONTRACT_ONLY' if not failed else 'BLOCKED_INPUT_CONTRACT',
                  'passed': len(self.checks) - len(failed), 'total': len(self.checks), 'checks': self.checks,
                  'details': self.details, 'warnings': self.warnings, 'audit_script_sha256': sha(Path(__file__)),
                  'private_code_export_sha256': sha(self.out / 'PRIVATE_CODE_EXPORT.json'),
                  'private_code_export_files': len(self.code), 'model_or_custom_code_executed': False,
                  'joblib_pickle_torch_checkpoint_deserialized': False, 'test_labels_or_hidden_text_accessed': False,
                  'training_or_inference_started': False,
                  'limitations': ['CPU inventory/metadata/hash checks cannot establish GPU model parity, finite model outputs or runtime budget.',
                                 'Safetensors headers prove tensor layout only, not finite tensor values.',
                                 'The Night24 joblib internals are not deserialized; adapter/label/layer compatibility must pass the original runtime assertions.',
                                 'The source normal preview emits constants without loading weights; never report it as a real inference preview.',
                                 'Multiple source-first base matches are recorded and blocked for explicit resolution, rather than silently choosing different bytes.',
                                 'The original recipe fits final heads on official rows; its in-sample checks and fitted bank are not honest OOF accuracy.']}
        (self.out / 'INPUT_AUDIT.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        print('STAGE10_CPU_AUDIT_SUMMARY', json.dumps({k: report[k] for k in ['status','passed','total','private_code_export_files','model_or_custom_code_executed']}), flush=True)
        return report


def self_test():
    checks = {}
    def rejects(name, fn):
        try:
            fn()
        except Exception:
            checks[name] = True
        else:
            checks[name] = False
    for value in ['/outside', '../escape', 'folder/../escape']:
        rejects('path_traversal_' + str(len(checks)), lambda value=value: safe_relative(value))
    rejects('label_set_mismatch', lambda: checked_labels(['wrong'] * 18))
    rejects('label_order_mismatch', lambda: checked_labels(SORTED_LABELS, LABELS))
    with tempfile.TemporaryDirectory() as td:
        root = Path(td); a = Audit(root, root / 'out')
        rejects('missing_recovery_archive', lambda: a.unique('**/legacy_recovery_manifest.json'))
        (root / 'one').mkdir(); (root / 'two').mkdir()
        (root / 'one/legacy_recovery_manifest.json').write_text('{}')
        (root / 'two/legacy_recovery_manifest.json').write_text('{}')
        rejects('ambiguous_recovery_archive', lambda: a.unique('**/legacy_recovery_manifest.json'))
        payload = root / 'payload'; payload.write_bytes(b'fixture')
        m = root / 'manifest.json'
        m.write_text(json.dumps({'independently_verified_published_migration_manifest': True,
                                 'files': {'COMPLETE.json': {'flat_file':'payload','sha256':'0'*64}}}))
        rejects('recovery_actual_hash_mismatch', lambda: validate_recovery(m))
        m.write_text(json.dumps({'independently_verified_published_migration_manifest': True,
                                 'files': {'../escape': {'flat_file':'payload','sha256':sha(payload)}}}))
        rejects('recovery_relative_path_escape', lambda: validate_recovery(m))
        invalid = root / 'bad.safetensors'; invalid.write_bytes(struct.pack('<Q', 1000) + b'{}')
        rejects('truncated_safetensors_header', lambda: safetensor_header(invalid))
        np.savez(root / 'object.npz', classes=np.array(['bad'], dtype=object))
        rejects('object_npz_pickle_rejected', lambda: npz_inventory(root / 'object.npz'))
        np.savez(root / 'nonfinite.npz', weight=np.array([np.nan]))
        rejects('nonfinite_numeric_npz_rejected', lambda: npz_inventory(root / 'nonfinite.npz'))
        np.savez(root / 'missing_readout.npz', classes=np.array(SORTED_LABELS), pool_layers=np.array([2]))
        rejects('missing_required_readout_key_rejected', lambda: npz_inventory(root / 'missing_readout.npz', SORTED_LABELS, [2], hidden=1))
    report = {'status': 'PASS' if all(checks.values()) else 'FAIL', 'checks': checks,
              'passed': sum(checks.values()), 'total': len(checks), 'source_or_model_executed': False}
    if not all(checks.values()):
        raise ValueError(json.dumps(report))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/kaggle/input'))
    parser.add_argument('--out', type=Path, default=Path('/kaggle/working/stage10_preflight'))
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        result = self_test()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result, indent=2))
    else:
        result = Audit(args.root, args.out).execute()
        if result['status'] != 'PASS_CPU_INPUT_CONTRACT_ONLY':
            raise SystemExit('Stage10 input contract blocked; inspect INPUT_AUDIT.json before GPU execution.')
