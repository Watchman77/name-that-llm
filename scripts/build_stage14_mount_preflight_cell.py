"""Render a first-party CPU byte audit of the exact Stage 14C V4 lock.

Extract only the 108 relative paths, byte sizes and SHA256s from the generated
private V4 cell. The generated audit contains no third-party code or data rows.
"""

import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import zlib


SOURCE_SHA = '2a2a69c7ec8d890d3f84d56fc241562e8c79218683d120a4022d4daf353aa9bf'
LOCK_SHA = '0d19b86dea044dc4df10e9cccfbc70b7a33767c638d43d10297e01b823bbc298'


def extract_locks(v4):
    raw = v4.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA:
        raise ValueError('Unexpected private V4 cell fingerprint')
    nodes = [x for x in ast.parse(raw).body if isinstance(x, ast.Assign)
             and len(x.targets) == 1 and isinstance(x.targets[0], ast.Tuple)
             and [getattr(t, 'id', None) for t in x.targets[0].elts] == ['cells', 'helpers', 'locked']]
    if len(nodes) != 1:
        raise ValueError('Missing canonical lock assignment')
    arg = nodes[0].value
    for func, attr in [('json', 'loads'), ('zlib', 'decompress'), ('base64', 'b64decode')]:
        if (not isinstance(arg, ast.Call) or len(arg.args) != 1 or not isinstance(arg.func, ast.Attribute)
                or getattr(arg.func.value, 'id', None) != func or arg.func.attr != attr):
            raise ValueError('Unexpected canonical lock envelope')
        arg = arg.args[0]
    _, _, locks = json.loads(zlib.decompress(base64.b64decode(ast.literal_eval(arg), validate=True)))
    if not isinstance(locks, dict) or len(locks) != 108:
        raise ValueError('Unexpected lock count')
    for name, pair in locks.items():
        path = PurePosixPath(name)
        if (path.is_absolute() or path.as_posix() != name or '..' in path.parts
                or not isinstance(pair, list) or len(pair) != 2
                or not (pair[0] is None or type(pair[0]) is int and pair[0] > 0)
                or not isinstance(pair[1], str) or not re.fullmatch(r'[0-9a-f]{64}', pair[1])):
            raise ValueError('Invalid lock path or identity')
    if hashlib.sha256(json.dumps(locks, sort_keys=True, separators=(',', ':')).encode()).hexdigest() != LOCK_SHA:
        raise ValueError('Unexpected canonical lock fingerprint')
    return dict(sorted(locks.items()))


def build(v4):
    locks = extract_locks(v4)
    spec = json.dumps(locks, sort_keys=True, separators=(',', ':'))
    cell = f'''import hashlib, json, os, stat, time
from collections import Counter
from pathlib import Path

LOCK_JSON = {spec!r}
LOCK_SHA = {LOCK_SHA!r}
V4_CELL_SHA = {SOURCE_SHA!r}
OPEN_POOL = 'name-that-llm-open-pool/ntl_open_pool.parquet'


def candidate_paths(root, relative):
    parts = Path(relative).parts
    namespace = parts[0]
    found = {{relative}}
    # Do not search by basename: retain the dataset/model slug and subpath.
    if namespace in ('datasets', 'models', 'notebooks'):
        found.add(Path(*parts[2:]).as_posix())
    elif namespace == 'competitions':
        found.add(Path(*parts[1:]).as_posix())
    else:
        for scope in ('datasets', 'notebooks', 'models'):
            for path in root.glob(f'{{scope}}/*/{{relative}}'):
                found.add(path.relative_to(root).as_posix())
    if relative == OPEN_POOL:
        found.add('datasets/crtproduction/name-that-llm-open-pool/ntl_open_pool.parquet')
    return [relative] + sorted(found - {{relative}})


def byte_record(path, expected_size, expected_hash, cache):
    try:
        before = path.stat()
        if not stat.S_ISREG(before.st_mode):
            return {{'status': 'not_regular_file'}}
        if expected_size is not None and before.st_size != expected_size:
            return {{'status': 'size_mismatch', 'bytes': before.st_size}}
        key = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        digest = cache.get(key)
        if digest is None:
            with path.open('rb') as stream:
                opened = os.fstat(stream.fileno())
                if not os.path.samestat(before, opened):
                    return {{'status': 'changed_during_check'}}
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
                after = os.fstat(stream.fileno())
                if (after.st_size, after.st_mtime_ns, after.st_ctime_ns) != key[2:]:
                    return {{'status': 'changed_during_check'}}
            cache[key] = digest
        return {{'status': 'match' if digest == expected_hash else 'sha256_mismatch',
                'bytes': before.st_size, 'sha256': digest}}
    except FileNotFoundError:
        return {{'status': 'missing'}}
    except OSError as exc:
        return {{'status': 'os_error', 'error_type': type(exc).__name__}}


def audit(root):
    if os.getenv('KAGGLE_IS_COMPETITION_RERUN') is not None:
        raise RuntimeError('Private audit cannot run as a competition submission')
    if not root.is_dir() or hashlib.sha256(LOCK_JSON.encode()).hexdigest() != LOCK_SHA:
        raise RuntimeError('Invalid mounted root or lock fingerprint')
    locks = json.loads(LOCK_JSON)
    if len(locks) != 108:
        raise RuntimeError('Unexpected lock count')
    cache, records, counts = {{}}, {{}}, Counter()
    started = time.monotonic()
    for relative, (size, digest) in locks.items():
        candidates = {{name: byte_record(root / name, size, digest, cache)
                      for name in candidate_paths(root, relative)}}
        exact = candidates[relative]['status'] == 'match'
        alias_matches = [name for name, record in candidates.items()
                         if name != relative and record['status'] == 'match']
        other_candidates = {{name: r for name, r in candidates.items()
                            if name != relative and r['status'] != 'missing'}}
        status = 'exact_match' if exact else 'alias_match_only' if alias_matches else 'no_identity_match'
        counts[status] += 1
        records[relative] = {{'status': status, 'expected_bytes': size, 'expected_sha256': digest,
                              'as_locked': candidates[relative], 'alias_matches': alias_matches,
                              'other_candidates': other_candidates}}
    return {{'status': 'COMPLETE_STAGE14C_MOUNT_PREFLIGHT', 'locks_checked': len(records),
             'lock_sha256': LOCK_SHA, 'private_v4_cell_sha256': V4_CELL_SHA,
             'counts': {{key: counts[key] for key in ('exact_match', 'alias_match_only', 'no_identity_match')}},
             'all_v4_locks_pass_as_written': counts['exact_match'] == len(locks),
             'all_lock_identities_found_somewhere': counts['no_identity_match'] == 0,
             'locked_inputs': records, 'seconds': round(time.monotonic() - started, 3),
             'inference_run': False, 'rows_read_or_exported': False,
             'third_party_code_imported': False, 'official_score_measured': False}}


if __name__ == '__main__':
    result = audit(Path('/kaggle/input'))
    out = Path('/kaggle/working/stage14c_mount_preflight.json')
    raw = json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + '\\n'
    out.write_text(raw, encoding='utf-8')
    if out.read_text(encoding='utf-8') != raw:
        raise RuntimeError('Audit report readback mismatch')
    summary = {{k: v for k, v in result.items() if k != 'locked_inputs'}}
    summary['open_pool'] = result['locked_inputs'][OPEN_POOL]
    summary['nonexact_locks'] = {{path: record for path, record in result['locked_inputs'].items()
                                 if record['status'] != 'exact_match'}}
    print('STAGE14C_MOUNT_PREFLIGHT ' + json.dumps(summary, sort_keys=True, allow_nan=False), flush=True)
'''
    compile(cell, 'stage14c_mount_preflight_live.py', 'exec')
    return cell


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--v4-cell', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.resolve().is_relative_to(Path(__file__).resolve().parent.parent):
        parser.error('write the transient Kaggle cell outside the repository')
    args.out.write_bytes(build(args.v4_cell).encode())
