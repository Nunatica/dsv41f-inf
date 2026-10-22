#!/usr/bin/env python3
"""Independent stdlib check of the rows dumped by dsv4f-v41-model-test MODEL DIR."""
import array
import json
import math
from pathlib import Path
import struct
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from model_info import header

model, directory = sys.argv[1:3]
meta, _, _ = header(model)
prefix = 'deepseek41.engram.'
cfg = json.loads(meta['deepseek41.config'])
cfg = cfg.get('text_config', cfg)
ngram, heads, dim = (cfg[k] for k in ('engram_max_ngram_size', 'engram_n_heads', 'engram_head_dim'))
layers, token_map, primes, multipliers = (meta[prefix + k] for k in ('layer_ids', 'token_map', 'primes', 'multipliers'))
shards = {}
for path in Path(directory).glob('*.safetensors'):
    with path.open('rb') as f:
        length, = struct.unpack('<Q', f.read(8))
        for key, value in json.loads(f.read(length)).items():
            if key != '__metadata__':
                shards[key] = (path, 8 + length + value['data_offsets'][0])
expected = []
for slot, layer in enumerate(layers):
    wp, wo = shards[f'layers.{layer}.engram.embed.weight']
    sp, so = shards[f'layers.{layer}.engram.embed.scale']
    with wp.open('rb') as weights, sp.open('rb') as scales:
        for t in range(4):
            rolling, offset, blocked = 0, 0, False
            for n in range(ngram):
                tid = -1 if t == 0 else 0 if t == 1 else 100 + t * 7 - n
                blocked |= tid < 0 or token_map[tid] == 0xffffffff
                token = meta[prefix + 'pad_id'] if blocked else token_map[tid]
                rolling ^= (token * multipliers[slot * ngram + n]) & ((1 << 64) - 1)
                if n == 0:
                    continue
                for h in range(heads):
                    prime = primes[slot * (ngram - 1) * heads + (n - 1) * heads + h]
                    signed = rolling if rolling < 1 << 63 else rolling - (1 << 64)
                    row = offset + signed % prime
                    offset += prime
                    weights.seek(wo + row * dim)
                    scales.seek(so + row * (dim // 32))
                    packed, exponents = weights.read(dim), scales.read(dim // 32)
                    assert len(packed) == dim and len(exponents) == dim // 32
                    for j, byte in enumerate(packed):
                        e, mantissa = (byte >> 3) & 15, byte & 7
                        value = math.ldexp(1 + mantissa / 8, e - 7) if e else math.ldexp(mantissa, -9)
                        if byte & 128:
                            value = -value
                        expected.append(value * math.ldexp(1, exponents[j // 32] - 127))
actual = array.array('f')
actual.frombytes(Path('/tmp/dsv4f-v41-engram.f32').read_bytes())
assert len(actual) == len(expected)
assert all(math.isfinite(a) and a == b for a, b in zip(actual, expected))
print(f'[dsv4f-v41-test] independent Engram hash + FP8 rows: {len(actual)} values PASS')
