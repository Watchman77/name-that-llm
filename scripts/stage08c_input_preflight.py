"""Standalone read-only CPU contracts for public source V1 361311282.

No Torch, Transformers, pickle, model/custom-code execution, fitting, inference,
network, or input writes. Emits aggregate JSON, never passages or individual IDs.
"""
from __future__ import annotations
import argparse
import glob
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import unicodedata
import numpy as np
import pandas as pd

SOURCE_VERSION = 361311282
TRAIN_SHA = "40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707"
LABELS = ['gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra', 'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5', 'claude-fable-5', 'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3']
SPECS = {
    'lora_qwen3_14b': {'model_type': 'qwen3', 'hidden_size': 5120, 'num_hidden_layers': 40, 'keep': 27},
    'lora_qwen25_14b': {'model_type': 'qwen2', 'hidden_size': 5120, 'num_hidden_layers': 48, 'keep': 32},
}
Q25_LAYERS = [3, 5, 8, 11, 16, 21, 27, 32]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def norm(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


class Audit:
    def __init__(self, root):
        self.root, self.checks, self.inventory = root, [], {}

    def check(self, name, passed, **counts):
        self.checks.append({'name': name, 'passed': bool(passed), **counts})
        return bool(passed)

    def receipt(self, path, hashed=True):
        value = {'path': str(path.relative_to(self.root)), 'bytes': path.stat().st_size}
        if hashed:
            value['sha256'] = digest(path)
        return value

    def unique(self, name, pattern):
        found = sorted({Path(p) for p in glob.glob(str(self.root / pattern), recursive=True) if Path(p).is_file()})
        self.inventory[name + ':matches'] = [self.receipt(p, False) for p in found]
        return found[0] if self.check(name + ':one_file', len(found) == 1, count=len(found)) else None

    def file(self, name, path, hashed=True):
        if not self.check(name + ':nonempty_file', path.is_file() and path.stat().st_size > 0):
            return False
        self.inventory[name] = self.receipt(path, hashed)
        return True

    def json(self, name, path):
        if not self.file(name, path):
            return None
        try:
            data = json.loads(path.read_text())
            if not isinstance(data, dict):
                raise TypeError()
            return data
        except Exception as exc:
            self.check(name + ':json_object', False, exception_type=type(exc).__name__)
            return None


def tensor_header(path):
    """Read only bounded safetensors JSON headers; validate payload sizes/ranges."""
    size = path.stat().st_size
    with path.open('rb') as f:
        prefix = f.read(8)
        if len(prefix) != 8:
            raise ValueError('short_header')
        length = struct.unpack('<Q', prefix)[0]
        if length > min(32 * 1024 * 1024, size - 8):
            raise ValueError('invalid_header_length')
        data = json.loads(f.read(length))
    if not isinstance(data, dict):
        raise TypeError('header_not_object')
    tensors = {k: v for k, v in data.items() if k != '__metadata__'}
    if not tensors:
        raise ValueError('empty_tensors')
    widths = {'BOOL': 1, 'U8': 1, 'I8': 1, 'I16': 2, 'U16': 2, 'F16': 2, 'BF16': 2,
              'I32': 4, 'U32': 4, 'F32': 4, 'I64': 8, 'U64': 8, 'F64': 8}
    ranges = []
    for value in tensors.values():
        shape, offsets, dtype = value['shape'], value['data_offsets'], value['dtype']
        if not isinstance(shape, list) or not all(type(x) is int and x >= 0 for x in shape):
            raise ValueError('invalid_shape')
        if not isinstance(offsets, list) or len(offsets) != 2 or not all(type(x) is int for x in offsets):
            raise ValueError('invalid_offsets')
        lo, hi = offsets
        if lo < 0 or hi < lo or hi > size - 8 - length or dtype not in widths:
            raise ValueError('invalid_bounds_or_dtype')
        if hi - lo != math.prod(shape) * widths[dtype]:
            raise ValueError('shape_size_mismatch')
        ranges.append((lo, hi))
    ranges.sort()
    if any(a[1] > b[0] for a, b in zip(ranges, ranges[1:])):
        raise ValueError('overlapping_tensors')
    return tensors


def safe_header(a, name, path):
    if not a.file(name, path):
        return None
    try:
        data = tensor_header(path)
        a.check(name + ':safe_header', True, tensor_count=len(data))
        return data
    except Exception as exc:
        a.check(name + ':safe_header', False, exception_type=type(exc).__name__)
        return None


def model_files(a, name, directory):
    """Verify local weight shards and tokenizer assets without deserializing weights."""
    a.file(name + ':tokenizer_config', directory / 'tokenizer_config.json')
    tokenizers = [directory / f for f in ['tokenizer.json', 'vocab.json', 'vocab.txt', 'spm.model', 'sentencepiece.bpe.model']]
    a.check(name + ':tokenizer_assets', any(p.is_file() and p.stat().st_size > 0 for p in tokenizers))
    configs = [directory / f for f in ['config.json', 'tokenizer_config.json', 'special_tokens_map.json', 'generation_config.json']]
    a.inventory[name + ':small_configs'] = [a.receipt(p) for p in configs if p.is_file()]
    shards = sorted(set(directory.glob('*.safetensors')) | set(directory.glob('pytorch_model*.bin')))
    indices = sorted(set(directory.glob('*.safetensors.index.json')) | set(directory.glob('pytorch_model*.bin.index.json')))
    for path in indices:
        index = a.json(name + ':index:' + path.name, path)
        if index is None:
            continue
        wm = index.get('weight_map')
        if a.check(name + ':index_weight_map', isinstance(wm, dict) and bool(wm)):
            values = list(wm.values())
            safe = all(isinstance(v, str) and Path(v).name == v for v in values)
            a.check(name + ':safe_shard_names', safe)
            if safe:
                required = {directory / value for value in values}
                a.check(name + ':all_index_shards_present', all(p.is_file() and p.stat().st_size > 0 for p in required), required=len(required))
                shards = sorted(set(shards) | {p for p in required if p.is_file()})
    a.check(name + ':weights_present', bool(shards) and all(p.stat().st_size > 0 for p in shards), shards=len(shards))
    a.check(name + ':multi_shard_index_present', len(shards) <= 1 or bool(indices))
    a.inventory[name + ':weight_shards'] = [a.receipt(p, False) for p in shards]


def pool_contract(a, pool, train, expected_rows=75783):
    a.check('pool:row_count', len(pool) == expected_rows, observed=len(pool), expected=expected_rows)
    if not a.check('pool:required_columns', {'text', 'model'} <= set(pool.columns)):
        return
    a.check('pool:nonnull_required_values', not pool[['text', 'model']].isna().any().any())
    text_ok = all(isinstance(t, str) and bool(t.strip()) for t in pool.text)
    a.check('pool:nonempty_string_text', text_ok)
    a.check('pool:exact_18_labels', set(pool.model.dropna()) == set(LABELS))
    a.inventory['pool_aggregates'] = {'rows': len(pool), 'columns': [str(c) for c in pool.columns],
        'class_counts': {label: int(pool.model.eq(label).sum()) for label in LABELS}}
    if text_ok and train is not None:
        original = set(train.text.astype(str))
        original_n = {norm(t) for t in original}
        texts, texts_n = pool.text.tolist(), [norm(t) for t in pool.text]
        exact, normalized = sum(t in original for t in texts), sum(t in original_n for t in texts_n)
        a.inventory['pool_overlap_admission'] = {
            'exact_official_training_response_rows': exact, 'normalized_official_training_response_rows': normalized,
            'duplicate_exact_text_rows_after_first': len(texts) - len(set(texts)),
            'duplicate_normalized_text_rows_after_first': len(texts_n) - len(set(texts_n)),
            'status': 'REVIEW_TRAIN_RESPONSE_OVERLAP' if exact or normalized else 'NO_FULL_RESPONSE_OVERLAP_FOUND',
            'rows_silently_removed': 0, 'overlap_is_not_an_inference_contract_failure': True,
            'overlap_alone_does_not_establish_hidden_test_leakage': True}
    else:
        a.inventory['pool_overlap_admission'] = {'status': 'UNAVAILABLE', 'rows_silently_removed': 0}


def meta_contract(a, name, meta, spec):
    a.check(name + ':label_order', meta.get('labels') == LABELS)
    cfg = meta.get('cfg')
    if not a.check(name + ':cfg_object', isinstance(cfg, dict)):
        return None
    layers, keep = cfg.get('layers'), cfg.get('keep')
    a.check(name + ':author_kept_depth', type(keep) is int and keep == spec['keep'])
    layer_ok = isinstance(layers, list) and bool(layers) and type(keep) is int and all(type(x) is int and 0 < x <= keep for x in layers) and layers == sorted(set(layers))
    a.check(name + ':pooled_layers_valid', layer_ok)
    if name == 'lora_qwen25_14b':
        a.check(name + ':author_q25_pool_depths', layers == Q25_LAYERS)
    a.check(name + ':max_len_positive', type(cfg.get('max_len')) is int and cfg['max_len'] > 0)
    a.check(name + ':training_eos_151643', type(cfg.get('eos_id')) is int and cfg['eos_id'] == 151643)
    a.inventory[name + ':cfg_values'] = {k: cfg.get(k) for k in ['keep', 'layers', 'max_len', 'eos_id']}
    return cfg if layer_ok else None


def adapter_contract(a, name, config, tensors, base, keep):
    rank, alpha = config.get('r'), config.get('lora_alpha')
    a.check(name + ':rank_scale', type(rank) is int and rank > 0 and type(alpha) in (int, float) and math.isfinite(alpha))
    a.check(name + ':manual_merge_compatible', config.get('peft_type') == 'LORA' and not config.get('use_dora', False)
            and not config.get('use_rslora', False) and not config.get('rank_pattern') and not config.get('alpha_pattern')
            and not config.get('modules_to_save') and config.get('bias', 'none') == 'none')
    a.inventory[name + ':adapter_values'] = {k: config.get(k) for k in ['r', 'lora_alpha', 'lora_dropout', 'target_modules', 'bias', 'peft_type']}
    if tensors is None or type(rank) is not int or base is None:
        return
    try:
        H, I = base['hidden_size'], base['intermediate_size']
        hd = base.get('head_dim', H // base['num_attention_heads'])
        q, kv = base['num_attention_heads'] * hd, base['num_key_value_heads'] * hd
        shapes = {'q_proj': (q, H), 'k_proj': (kv, H), 'v_proj': (kv, H), 'o_proj': (H, q),
                  'gate_proj': (I, H), 'up_proj': (I, H), 'down_proj': (H, I)}
        pairs = [(key, key.replace('.lora_A.', '.lora_B.')) for key in tensors if '.lora_A.' in key]
        all_used, good = set(), bool(pairs)
        for key, mate in pairs:
            match = re.search(r'layers\.(\d+)\.(self_attn|mlp)\.(\w+)\.lora_A\.', key)
            good = good and bool(match) and mate in tensors
            if not match or mate not in tensors:
                continue
            layer, projection = int(match.group(1)), match.group(3)
            if projection not in shapes:
                good = False
                continue
            out_width, in_width = shapes[projection]
            good = good and layer < keep and tensors[key]['shape'] == [rank, in_width] and tensors[mate]['shape'] == [out_width, rank]
            all_used.update([key, mate])
        a.check(name + ':adapter_layer_pair_dimensions', good and all_used == set(tensors), pairs=len(pairs))
    except Exception as exc:
        a.check(name + ':adapter_layer_pair_dimensions', False, exception_type=type(exc).__name__)


def run(root):
    root = root.resolve()
    a = Audit(root)
    train = None
    train_path = a.unique('official_train', '**/train.csv')
    test_path = a.unique('official_test', '**/test.csv')
    if train_path is not None:
        a.file('official_train_receipt', train_path)
        a.check('official:known_train_hash', digest(train_path) == TRAIN_SHA)
        train = pd.read_csv(train_path, keep_default_na=False)
        schema = {'id', 'text', 'model', 'prompt_id'} <= set(train.columns)
        a.check('official:train_schema', schema)
        if schema:
            a.check('official:2699_rows_150_prompts_18_labels', len(train) == 2699 and train.prompt_id.nunique() == 150 and set(train.model) == set(LABELS))
            a.check('official:unique_train_ids', train.id.is_unique)
        else:
            train = None
    if test_path is not None:
        a.file('official_test_receipt', test_path)
        test = pd.read_csv(test_path, keep_default_na=False)
        a.check('official:test_schema', {'id', 'text'} <= set(test.columns))
        if {'id', 'text'} <= set(test.columns):
            a.check('official:test_ids_unique_nonempty', test.id.is_unique and test.id.astype(str).str.strip().ne('').all())
            a.inventory['official_test_aggregates'] = {'rows': len(test), 'columns': [str(c) for c in test.columns]}
    pool_path = a.unique('pool', '**/pool_v4.parquet')
    if pool_path is not None:
        a.file('pool_receipt', pool_path)
        pool_contract(a, pd.read_parquet(pool_path), train)
    configs = []
    for path in sorted(root.rglob('config.json')):
        try:
            c = json.loads(path.read_text())
            if isinstance(c, dict):
                configs.append((path, c))
        except Exception:
            pass
    for name, spec in SPECS.items():
        matches = [(p, c) for p, c in configs if all(c.get(k) == v for k, v in spec.items() if k != 'keep')]
        a.inventory[name + ':base_config_matches'] = [a.receipt(p) for p, _ in matches]
        base = None
        if a.check(name + ':one_base_model', len(matches) == 1, count=len(matches)):
            path, base = matches[0]
            model_files(a, name + ':base', path.parent)
        marker = a.unique(name + ':meta', '**/' + name + '/meta.json')
        if marker is None:
            continue
        meta = a.json(name + ':meta_receipt', marker)
        cfg = meta_contract(a, name, meta, spec) if meta is not None else None
        heads = safe_header(a, name + ':heads', marker.parent / 'heads.safetensors')
        acfg = a.json(name + ':adapter_config', marker.parent / 'adapter/adapter_config.json')
        atensors = safe_header(a, name + ':adapter_weights', marker.parent / 'adapter/adapter_model.safetensors')
        if acfg is not None:
            adapter_contract(a, name, acfg, atensors, base, cfg['keep'] if cfg else spec['keep'])
        if cfg is not None and heads is not None:
            width, H = len(cfg['layers']) * spec['hidden_size'], spec['hidden_size']
            expected = {'plain.weight': [18, width], 'plain.bias': [18], 'centered.weight': [18, width], 'centered.bias': [18]}
            for i in range(len(cfg['layers'])):
                expected.update({f'norms.{i}.weight': [H], f'norms.{i}.bias': [H]})
            a.check(name + ':all_head_shapes', all(k in heads and heads[k]['shape'] == shape for k, shape in expected.items()))
        if cfg is not None and base is not None:
            a.check(name + ':eos_in_base_vocab', type(base.get('vocab_size')) is int and 0 <= cfg.get('eos_id', -1) < base['vocab_size'])
            a.inventory[name + ':end_token_note'] = {'training_eos_id': cfg.get('eos_id'), 'base_config_eos_token_id': base.get('eos_token_id'),
                'source_uses_training_eos_override': True, 'tokenizer_base_eos_mismatch_is_expected_for_qwen3': True}
    marker = a.unique('deberta_meta', '**/deberta_v3_large/meta.json')
    if marker is not None:
        meta = a.json('deberta_meta_receipt', marker)
        config = a.json('deberta_encoder_config', marker.parent / 'encoder/config.json')
        model_files(a, 'deberta_encoder', marker.parent / 'encoder')
        heads = safe_header(a, 'deberta_head', marker.parent / 'head.safetensors')
        if meta is not None:
            a.check('deberta:label_order', meta.get('labels') == LABELS)
            a.check('deberta:author_max_len_384', meta.get('max_len') == 384)
            a.inventory['deberta_meta_values'] = {'max_len': meta.get('max_len')}
        if config is not None:
            a.check('deberta:large_encoder_identity', config.get('model_type') == 'deberta-v2' and config.get('hidden_size') == 1024 and config.get('num_hidden_layers') == 24)
            if heads is not None:
                a.check('deberta:head_shapes', heads.get('weight', {}).get('shape') == [18, 1024] and heads.get('bias', {}).get('shape') == [18])
    return {'stage': '08c_input_preflight', 'public_source_version': SOURCE_VERSION,
            'status': 'PASS_INPUT_CONTRACT' if all(c['passed'] for c in a.checks) else 'FAIL_INPUT_CONTRACT',
            'read_only': True, 'new_fitting': False, 'new_inference': False, 'network_used': False,
            'pickle_used': False, 'public_source_executed': False, 'checks': a.checks, 'inventory': a.inventory,
            'limits': ['A pass verifies present contracts and records hashes; exact upstream attachment/version identity still requires matching immutable version receipts.',
                       'Large base/encoder weights are inventoried by name/size and index completeness, not hashed, deserialized, or executed.',
                       'Train-response overlap is separately reported for admission review; no rows are filtered and overlap alone does not establish hidden-test leakage.',
                       'The 75783-row shipped SVC pool is distinct from the author-described 83827-row adapter-training pool; checkpoint training provenance/CV is not reproduced.',
                       'No GPU compatibility, numerical feature parity, independent development accuracy, preview completion or official public score is established.']}


def self_test():
    import tempfile
    tests = {}
    with tempfile.TemporaryDirectory(prefix='stage08c_test_') as temp:
        root = Path(temp)
        a = Audit(root)
        tests['missing_input_rejected'] = a.unique('x', '**/x.json') is None and not a.checks[-1]['passed']
        for sub in ['a', 'b']:
            (root / sub).mkdir()
            (root / sub / 'x.json').write_text('{}')
        a = Audit(root)
        tests['ambiguous_input_rejected'] = a.unique('x', '**/x.json') is None and not a.checks[-1]['passed']
        train = pd.DataFrame({'text': ['Official response']})
        pool = pd.DataFrame({'model': LABELS, 'text': [f'unique text {i}' for i in range(18)]})
        def inspect(frame, expected=18):
            audit = Audit(root); pool_contract(audit, frame, train, expected); return audit
        tests['valid_18_class_pool'] = all(c['passed'] for c in inspect(pool).checks)
        tests['wrong_pool_count_rejected'] = not all(c['passed'] for c in inspect(pool, 19).checks)
        bad = pool.copy(); bad.loc[0, 'model'] = 'unknown'
        tests['unknown_class_rejected'] = not all(c['passed'] for c in inspect(bad).checks)
        bad = pool.copy(); bad.loc[0, 'text'] = None
        tests['null_text_rejected'] = not all(c['passed'] for c in inspect(bad).checks)
        bad = pool.copy(); bad.loc[0, 'text'] = '  OFFICIAL   response '
        audit = inspect(bad)
        tests['normalized_train_overlap_reported_separately'] = all(c['passed'] for c in audit.checks) and audit.inventory['pool_overlap_admission']['normalized_official_training_response_rows'] == 1 and audit.inventory['pool_overlap_admission']['rows_silently_removed'] == 0
        bad = pool.copy(); bad.loc[0, 'text'] = 'Official response'
        tests['exact_train_overlap_reported'] = inspect(bad).inventory['pool_overlap_admission']['exact_official_training_response_rows'] == 1
        valid = {'labels': LABELS, 'cfg': {'keep': 32, 'layers': Q25_LAYERS, 'max_len': 447, 'eos_id': 151643}}
        audit = Audit(root); meta_contract(audit, 'lora_qwen25_14b', valid, SPECS['lora_qwen25_14b'])
        tests['valid_metadata'] = all(c['passed'] for c in audit.checks)
        bad = json.loads(json.dumps(valid)); bad['labels'] = LABELS[::-1]
        audit = Audit(root); meta_contract(audit, 'lora_qwen25_14b', bad, SPECS['lora_qwen25_14b'])
        tests['reversed_head_labels_rejected'] = not all(c['passed'] for c in audit.checks)
        bad = json.loads(json.dumps(valid)); bad['cfg']['eos_id'] = 151645
        audit = Audit(root); meta_contract(audit, 'lora_qwen25_14b', bad, SPECS['lora_qwen25_14b'])
        tests['wrong_chat_eos_rejected'] = not all(c['passed'] for c in audit.checks)
        bad = json.loads(json.dumps(valid)); bad['cfg']['layers'][-1] = 33
        audit = Audit(root); meta_contract(audit, 'lora_qwen25_14b', bad, SPECS['lora_qwen25_14b'])
        tests['pool_depth_beyond_trunk_rejected'] = not all(c['passed'] for c in audit.checks)
        path = root / 'valid.safetensors'
        header = json.dumps({'weight': {'dtype': 'F32', 'shape': [2], 'data_offsets': [0, 8]}}).encode()
        path.write_bytes(struct.pack('<Q', len(header)) + header + bytes(8))
        tests['valid_safe_header'] = tensor_header(path)['weight']['shape'] == [2]
        path.write_bytes(struct.pack('<Q', 999999))
        try:
            tensor_header(path); tests['truncated_safe_header_rejected'] = False
        except ValueError:
            tests['truncated_safe_header_rejected'] = True
        base = {'hidden_size': 8, 'intermediate_size': 16, 'num_attention_heads': 2, 'num_key_value_heads': 1}
        config = {'r': 2, 'lora_alpha': 4, 'peft_type': 'LORA', 'bias': 'none'}
        ts = {'base_model.model.layers.0.self_attn.q_proj.lora_A.weight': {'shape': [2, 8]},
              'base_model.model.layers.0.self_attn.q_proj.lora_B.weight': {'shape': [8, 2]}}
        audit = Audit(root); adapter_contract(audit, 'x', config, ts, base, 2)
        tests['valid_adapter_pair'] = all(c['passed'] for c in audit.checks)
        bad = json.loads(json.dumps(ts)); next(v for k, v in bad.items() if '.lora_B.' in k)['shape'] = [7, 2]
        audit = Audit(root); adapter_contract(audit, 'x', config, bad, base, 2)
        tests['wrong_adapter_width_rejected'] = not all(c['passed'] for c in audit.checks)
        model = root / 'model'; model.mkdir()
        for filename, content in [('tokenizer_config.json', '{}'), ('tokenizer.json', '{}'), ('model.safetensors.index.json', '{"weight_map":{"x":"missing.safetensors"}}')]:
            (model / filename).write_text(content)
        audit = Audit(root); model_files(audit, 'x', model)
        tests['missing_index_shard_rejected'] = not all(c['passed'] for c in audit.checks)
    return {'stage': '08c_input_preflight_synthetic_tests', 'status': 'PASS' if all(tests.values()) else 'FAIL',
            'tests': tests, 'real_inputs_read': False, 'model_code_executed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, default=Path('/kaggle/input'))
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    try:
        result = self_test() if args.self_test else run(args.input_root)
    except Exception as exc:
        result = {'stage': '08c_input_preflight', 'status': 'FAIL_PREFLIGHT_EXCEPTION',
                  'exception_type': type(exc).__name__, 'raw_exception_message_omitted': True,
                  'public_source_executed': False}
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result['status'].startswith('PASS') else 2


if __name__ == '__main__':
    raise SystemExit(main())
