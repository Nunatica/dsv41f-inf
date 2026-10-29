#!/usr/bin/env python3
"""Read GGUF headers without mapping weights or requiring NumPy."""
import bisect
import json
import os
from pathlib import Path
import re
import struct
import sys

DSX_FILENAME = 'DeepSeek-V4.1-Flash-1017-dsx-2.6bit.gguf'
UD_FILENAME = 'DeepSeek-V4.1-Flash-1017-UD-Q4_K_XL-00001-of-00009.gguf'


def model_format(path):
    name = Path(path).name
    if name == DSX_FILENAME:
        return 'dsx-2.6bit'
    if name == UD_FILENAME:
        return 'ud-q4-k-xl'
    raise ValueError('Unsupported model filename; expected exactly '+DSX_FILENAME+' or '+UD_FILENAME)


def original_engram(meta, directory):
    layers = meta['deepseek41.ple.layers']
    per = (meta['deepseek41.ple.ngram_size']-1)*meta['deepseek41.ple.heads_per_ngram']
    sizes = meta['deepseek41.ple.head_vocab_sizes']
    width = meta['deepseek41.embedding_length_per_layer_input']
    if per <= 0 or len(sizes) != len(layers)*per or width <= 0 or width % 32:
        raise ValueError('Invalid native Engram geometry')
    required = {}
    for slot, layer in enumerate(layers):
        rows = sum(sizes[slot*per:(slot+1)*per])
        for suffix, dtype, columns in [('weight', 'F8_E4M3', width), ('scale', 'F8_E8M0', width//32)]:
            required[f'layers.{layer}.engram.embed.{suffix}'] = (dtype, [rows, columns])
    found = {}
    for path in sorted(Path(directory).glob('*.safetensors')):
        with path.open('rb') as f:
            raw = f.read(8)
            if len(raw) != 8:
                raise ValueError('Truncated safetensors header: '+str(path))
            n = struct.unpack('<Q', raw)[0]
            if n > 64*1024*1024:
                raise ValueError('Invalid safetensors header: '+str(path))
            entries = json.loads(f.read(n))
        for key, (dtype, shape) in required.items():
            if key not in entries:
                continue
            t = entries[key]
            off = t['data_offsets']
            if key in found:
                raise ValueError('Duplicate original Engram: '+key)
            if (t['dtype'] != dtype or t['shape'] != shape or len(off) != 2 or off[0] < 0 or
                    off[1]-off[0] != shape[0]*shape[1] or off[1] > path.stat().st_size-8-n):
                raise ValueError('Original Engram dtype/shape/extent mismatch: '+key)
            found[key] = str(path.resolve())
    if set(found) != set(required):
        raise ValueError('Missing original FP8 Engram: '+', '.join(sorted(set(required)-set(found))))
    return {'directory': str(Path(directory).resolve()), 'tensors': found}


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
    requested = Path(path)
    path = Path(path).resolve()
    meta, _, _ = header(path)
    arch = meta.get('general.architecture')
    if arch not in ('deepseek4', 'deepseek41'):
        raise ValueError('This engine requires a deepseek4 or deepseek41 GGUF model')
    variant = model_format(requested) if arch == 'deepseek41' else None
    native = variant == 'ud-q4-k-xl'
    if native and meta.get('split.count') != 9:
        raise ValueError('UD-Q4_K_XL requires exactly nine declared shards')
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
    total = expert = tensor_count = 0
    available, skipped = [], []
    external = original_engram(meta, os.environ.get('DSV4F_ENGRAM_DIR', requested.parent)) if native else None
    for i, shard in enumerate(paths):
        if native and i == 7 and not shard.exists():
            skipped.append(str(shard))
            continue
        sm, tensors, start = header(shard)
        if sm.get('split.no', 0) != i or sm.get('split.count', 1) != count:
            raise ValueError(f'Inconsistent shard metadata: {shard}')
        size = shard.stat().st_size
        available.append(str(shard))
        tensor_count += len(tensors)
        total += size
        for j, t in enumerate(tensors):
            if re.search(r'\.ffn_(gate|up|down)_exps(?:\.[^.]+)?\.weight$', t[0]):
                end = tensors[j+1][3] if j+1 < len(tensors) else size-start
                expert += end-t[3]
    if native and tensor_count+len(skipped) != meta['split.tensors.count']:
        raise ValueError('Native GGUF tensor count mismatch beyond the external Engram embedding')
    return {'model': str(path), 'name': meta.get('general.name'), 'shards': available,
            'format': variant, 'skipped_engram_shards': skipped, 'original_engram': external,
            'model_bytes': total, 'expert_bytes': expert, 'layers': meta.get('deepseek41.num_hidden_layers', meta.get('deepseek41.block_count')) if arch == 'deepseek41' else meta.get('deepseek4.block_count'),
            'context_limit': meta.get('deepseek41.max_position_embeddings', meta.get('deepseek41.context_length')) if arch == 'deepseek41' else meta.get('deepseek4.context_length')}


def select_pruning(layers, percentage):
    """Greedy normalized historical call-share cost per byte, matching native code.

    Each layer supplies counts, bytes, layer and top_k. Whole experts are
    removed until ceil(total expert bytes * percentage / 100) is reached.
    """
    if type(percentage) is not int or not 0 <= percentage <= 50:
        raise ValueError('DSV4F_ADAPTIVE_PRUNING must be in 0..50')
    if not percentage:
        return [set() for _ in layers]
    candidates, removed, active, seen = [], [], [], set()
    total_bytes = 0
    for slot, layer in enumerate(layers):
        counts, weights = layer['counts'], layer['bytes']
        ne, nk = len(counts), layer['top_k']
        if (not 1 <= ne <= 512 or len(weights) != ne or not 1 <= nk <= ne
                or layer['layer'] < 0 or layer['layer'] in seen
                or any(type(c) is not int or not 0 <= c <= sys.maxsize for c in counts)
                or any(type(w) is not int or w <= 0 for w in weights)):
            raise ValueError('Invalid pruning layer/counts/bytes')
        seen.add(layer['layer'])
        calls = 0.0
        for count in counts:
            calls += float(count)
        if not calls:
            raise ValueError(f"Empty routing history for layer {layer['layer']}")
        removed.append(set())
        active.append(ne)
        total_bytes += sum(weights)
        for expert, (count, weight) in enumerate(zip(counts, weights)):
            candidates.append((float(count) / calls / float(weight), layer['layer'], expert, slot, weight))
    if not 0 < total_bytes <= 2**64-1:
        raise ValueError('Invalid total expert byte size')
    target = (total_bytes*percentage + 99)//100
    choices = [sorted((w, i) for i, w in enumerate(l['bytes'])) for l in layers]
    min_keep = [sum(w for w, _ in options[:l['top_k']]) for l, options in zip(layers, choices)]
    retained_floor = sum(min_keep)
    if total_bytes-retained_floor < target:
        raise ValueError('Pruning cannot reach memory target while retaining top-k in every layer')
    saved = 0
    for _, _, expert, slot, weight in sorted(candidates):
        if saved >= target:
            break
        if active[slot] <= layers[slot]['top_k']:
            continue
        choices[slot].remove((weight, expert))
        next_keep = sum(w for w, _ in choices[slot][:layers[slot]['top_k']])
        next_floor = retained_floor-min_keep[slot]+next_keep
        if total_bytes-next_floor < target:
            bisect.insort(choices[slot], (weight, expert))
            continue
        retained_floor, min_keep[slot] = next_floor, next_keep
        removed[slot].add(expert)
        active[slot] -= 1
        saved += weight
    if saved < target:
        raise ValueError('Pruning cannot reach memory target while retaining top-k in every layer')
    return removed


def resident_pruning_plan(info, percentage, routing_dir='models'):
    """Read headers, format maps and history; plan exact routed-weight savings."""
    if not percentage:
        return {'original_bytes': 0, 'skipped_bytes': 0, 'layers': []}
    stem = re.sub(r'-[0-9]{5}-of-([0-9]{5})$', r'-00001-of-\1', Path(info['model']).stem)
    table = Path(routing_dir) / (stem + '.routing.json')
    if table.stat().st_size > 64*1024*1024:
        raise ValueError('Routing history exceeds 64 MiB')
    data = json.loads(table.read_text())
    if data['version'] != 1 or data['model'] != stem:
        raise ValueError('Pruning history schema/model mismatch')
    sources, layers = {}, {}
    geometry = {0: (1, 4), 1: (1, 2), 8: (32, 34), 10: (256, 84), 11: (256, 110),
                12: (256, 144), 13: (256, 176), 14: (256, 210),
                39: (32, 17), 42: (64, 18), 143: (128, 28)}
    top_k = None
    for path in info['shards']:
        meta, tensors, start = header(path)
        if top_k is None:
            top_k = meta.get('deepseek41.num_experts_per_tok', meta.get('deepseek41.expert_used_count'))
        for name, shape, typ, offset in tensors:
            sources[name] = (path, start+offset, shape, typ)
            match = re.fullmatch(r'blk\.(\d+)\.ffn_gate_exps(?:\.[^.]+)?\.weight', name)
            if match:
                layers.setdefault(int(match[1]), []).append(name)
    if type(top_k) is not int:
        raise ValueError('Missing expert top-k metadata for pruning')
    plan = []
    for layer, gates in sorted(layers.items()):
        map_name = f'blk.{layer}.ffn_exps_gu_type'
        if map_name in sources:
            path, offset, shape, _ = sources[map_name]
            with open(path, 'rb') as f:
                f.seek(offset)
                formats = struct.unpack('<' + 'i'*shape[0], f.read(4*shape[0]))
            mappings = {gate: [i for i, typ in enumerate(formats) if typ == sources[gate][3]] for gate in gates}
            count = len(formats)
        else:
            if len(gates) != 1:
                raise ValueError('Missing expert format map')
            count = sources[gates[0]][2][2]
            mappings = {gates[0]: list(range(count))}
        counters, weights, seen = [0]*count, [0]*count, set()
        for gate, global_ids in mappings.items():
            if len(global_ids) != sources[gate][2][2] or seen.intersection(global_ids):
                raise ValueError('Expert format map/shape mismatch')
            seen.update(global_ids)
            row = data['experts'].get(gate.replace('ffn_gate_exps', 'ffn_exps'), {})
            if any(not k.isdecimal() or str(int(k)) != k or not 0 <= int(k) < len(global_ids)
                   or type(v) is not int or not 0 <= v <= sys.maxsize for k, v in row.items()):
                raise ValueError('Invalid routing counter')
            plane = 0
            for role in ('gate', 'up', 'down'):
                _, _, shape, typ = sources[gate.replace('ffn_gate_exps', f'ffn_{role}_exps')]
                if typ not in geometry or shape[0] % geometry[typ][0]:
                    raise ValueError(f'Unsupported pruning tensor geometry: {typ}')
                block, size = geometry[typ]
                plane += shape[0]//block*size*shape[1]
            for local, global_id in enumerate(global_ids):
                counters[global_id] = row.get(str(local), 0)
                weights[global_id] = plane
        if len(seen) != count:
            raise ValueError('Incomplete expert format map')
        plan.append({'layer': layer, 'counts': counters, 'bytes': weights, 'top_k': top_k})
    removed = select_pruning(plan, percentage)
    for layer, excluded in zip(plan, removed):
        layer['removed'] = sorted(excluded)
        layer['skipped_bytes'] = sum(layer['bytes'][i] for i in excluded)
        layer['excluded_call_share'] = sum(layer['counts'][i] for i in excluded)/sum(layer['counts'])
    return {'original_bytes': sum(sum(l['bytes']) for l in plan),
            'skipped_bytes': sum(l['skipped_bytes'] for l in plan), 'layers': plan}


def resident_pruned_bytes(info, percentage, routing_dir='models'):
    return resident_pruning_plan(info, percentage, routing_dir)['skipped_bytes']


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
