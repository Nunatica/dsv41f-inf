#!/usr/bin/env python3
"""Read GGUF headers without mapping weights or requiring NumPy."""
import json
import os
from pathlib import Path
import re
import struct
import sys


def header(path):
    with open(path, 'rb') as f:
        def read(fmt):
            size = struct.calcsize('<' + fmt)
            b = f.read(size)
            if len(b) != size:
                raise ValueError(f'Truncated GGUF: {path}')
            return struct.unpack('<' + fmt, b)[0]
        def string():
            return f.read(read('Q')).decode('utf-8')
        def value(t, keep=True):
            if t == 8:
                n = read('Q')
                if keep:
                    return f.read(n).decode('utf-8')
                f.seek(n, 1)
                return None
            if t == 9:
                et, n = read('I'), read('Q')
                if not keep and et != 8:
                    size = struct.calcsize('<' + formats[et])
                    f.seek(n*size, 1)
                    return None
                v = [value(et, keep) for _ in range(n)]
                return v if keep else None
            return read(formats[t])
        formats = {0:'B', 1:'b', 2:'H', 3:'h', 4:'I', 5:'i', 6:'f', 7:'?', 10:'Q', 11:'q', 12:'d'}
        if f.read(4) != b'GGUF' or read('I') not in (2, 3):
            raise ValueError(f'Unsupported GGUF: {path}')
        nt, nk = read('Q'), read('Q')
        meta = {}
        for _ in range(nk):
            k = string()
            v = value(read('I'), k.startswith(('general.', 'deepseek4.', 'deepseek41.', 'split.')))
            if v is not None:
                meta[k] = v
        tensors = []
        for _ in range(nt):
            name = string()
            shape = [read('Q') for _ in range(read('I'))]
            typ, off = read('I'), read('Q')
            tensors.append((name, shape, typ, off))
        align = meta.get('general.alignment', 32)
        data_start = (f.tell() + align - 1)//align*align
    return meta, sorted(tensors, key=lambda t: t[3]), data_start


def inspect(path):
    path = Path(path).resolve()
    meta, _, _ = header(path)
    arch = meta.get('general.architecture')
    if arch not in ('deepseek4', 'deepseek41'):
        raise ValueError('This engine requires a deepseek4 or deepseek41 GGUF model')
    count = meta.get('split.count', 1)
    if count > 1:
        m = re.fullmatch(r'(.*)-\d{5}-of-\d{5}\.gguf', str(path))
        if not m:
            raise ValueError('Split GGUF must have a numbered filename')
        paths = [Path(f'{m[1]}-{i:05d}-of-{count:05d}.gguf') for i in range(1, count+1)]
        if path != paths[0]:
            raise ValueError('MODEL must point to the first GGUF shard')
    else:
        paths = [path]
    total = expert = 0
    for i, shard in enumerate(paths):
        sm, tensors, start = header(shard)
        if sm.get('split.no', 0) != i or sm.get('split.count', 1) != count:
            raise ValueError(f'Inconsistent shard metadata: {shard}')
        size = shard.stat().st_size
        total += size
        for j, t in enumerate(tensors):
            if re.search(r'\.ffn_(gate|up|down)_exps(?:\.[^.]+)?\.weight$', t[0]):
                end = tensors[j+1][3] if j+1 < len(tensors) else size-start
                expert += end-t[3]
    return {'model': str(path), 'name': meta.get('general.name'), 'shards': [str(p) for p in paths],
            'model_bytes': total, 'expert_bytes': expert, 'layers': meta.get('deepseek41.num_hidden_layers') if arch == 'deepseek41' else meta.get('deepseek4.block_count'),
            'context_limit': meta.get('deepseek41.max_position_embeddings') if arch == 'deepseek41' else meta.get('deepseek4.context_length')}


def inspect_mtp(path, manifest=None):
    """Inspect a DSpark raw binary + JSON index without reading weight data."""
    manifest = Path(manifest) if manifest else Path(path).with_suffix('.json')
    path = Path(path).resolve()
    index = json.loads(manifest.read_text())
    size = path.stat().st_size
    geometry = {'F32': (1, 4), 'BF16': (1, 2), 'Q8_0': (32, 34), 'IQ2_XXS': (256, 66)}
    names, spans, total = set(), [], 0
    for name, typ, offset, length, shape in index['tensors']:
        if not name.startswith('mtp.') or name.endswith('.bias_vl'):
            continue
        if name in names:
            raise ValueError('Duplicate MTP tensor: ' + name)
        names.add(name)
        if typ not in geometry or not 1 <= len(shape) <= 4 or any(type(d) is not int or d <= 0 for d in shape):
            raise ValueError('Invalid MTP type/shape: ' + name)
        block, byte_count = geometry[typ]
        elements = 1
        for d in shape:
            elements *= d
        if shape[-1] % block or length != elements // block * byte_count:
            raise ValueError('MTP byte size mismatch: ' + name)
        if type(offset) is not int or type(length) is not int or offset < 0 or length <= 0 or offset + length > size:
            raise ValueError('Truncated or invalid MTP tensor: ' + name)
        spans.append((offset, offset+length))
        total += length * (2 if typ == 'BF16' else 1)
    spans.sort()
    if any(left[1] > right[0] for left, right in zip(spans, spans[1:])):
        raise ValueError('Overlapping MTP tensors')
    for name in ['mtp.0.main_proj.weight', 'mtp.0.main_norm.weight',
                 'mtp.2.norm.weight', 'mtp.2.markov_head.embed.weight',
                 'mtp.2.markov_head.head.weight', 'mtp.2.confidence_head.proj.weight']:
        if name not in names:
            raise ValueError('MTP sidecar missing ' + name)
    return {'model': str(path), 'manifest': str(manifest.resolve()), 'model_bytes': total,
            'mtp_tensors': len(names), 'vision_tensors_loaded': 0}


def swap_protected():
    if len(Path('/proc/swaps').read_text().splitlines()) <= 1:
        return True
    for line in Path('/proc/self/cgroup').read_text().splitlines():
        if line.startswith('0::'):
            root = Path('/sys/fs/cgroup')
            path = root / line[3:].lstrip('/')
            while True:
                try:
                    if (path/'memory.swap.max').read_text().strip() == '0':
                        return True
                except FileNotFoundError:
                    pass
                if path == root:
                    break
                path = path.parent
    return False


if __name__ == '__main__':
    if sys.argv[1:] == ['--swap-check']:
        sys.exit(0 if swap_protected() else 1)
    try:
        print(json.dumps(inspect_mtp(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None) if sys.argv[1] == "--mtp" else inspect(sys.argv[1]), indent=2))
    except (OSError, ValueError, KeyError, TypeError) as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
