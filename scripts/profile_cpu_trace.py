#!/usr/bin/env python3
"""Summarize an opt-in Q3_K or PTQ1_0 CPU trace, including per-request decode intervals."""
import argparse
import collections
import json
from pathlib import Path
import re
import struct

HEADER = struct.Struct('<4IQfI192s5d')
PHASES = ('prepare', 'gate_up', 'quantize', 'down', 'reduce')


def records(path):
    with Path(path).open('rb') as f:
        while True:
            raw = f.read(HEADER.size)
            if not raw:
                return
            if len(raw) != HEADER.size:
                raise ValueError('Truncated trace header')
            magic, ne, nf, nk, hits, limit, _, names, *times = HEADER.unpack(raw)
            if magic not in (0x51335431, 0x50315431) or not 0 < nk <= 64 or not 0 < ne <= 16384:
                raise ValueError('Invalid trace header')
            name = names[:64].split(b'\0')[0].decode()
            layer = int(re.fullmatch(r'blk\.(\d+)\.ffn_gate_exps(?:\.(?:q3_k|ptq1_0))?\.weight', name)[1])
            payload = f.read(nk*8+ne*4)
            if len(payload) != nk*8+ne*4:
                raise ValueError('Truncated trace payload')
            ids = struct.unpack_from(f'<{nk}i', payload)
            prob = struct.unpack_from(f'<{nk}f', payload, nk*4)
            misses = sum(p != 0 and not (hits >> i & 1) for i, p in enumerate(prob))
            yield dict(format='q3_k' if magic == 0x51335431 else 'ptq1_0', layer=layer, ne=ne, nf=nf, nk=nk, hits=hits,
                       misses=misses, ids=ids, phase_us=dict(zip(PHASES, times)))


def summarize(rows, token_count):
    phases = {phase: sum(r['phase_us'][phase] for r in rows)/1000/token_count for phase in PHASES}
    # Original block sizes: Q3_K=110/256, PTQ1_0=28/128 bytes per weight.
    weight_bytes = sum(r['misses']*3*r['ne']*r['nf']*(110/256 if r['format']=='q3_k' else 28/128) for r in rows)
    return dict(tokens=token_count, cpu_calls=len(rows), mean_misses_per_cpu_call=sum(r['misses'] for r in rows)/len(rows),
                misses_histogram=dict(sorted(collections.Counter(r['misses'] for r in rows).items())),
                phase_ms_per_token=phases, total_ms_per_token=sum(phases.values()),
                nominal_weight_mib_per_token=weight_bytes/1048576/token_count)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('trace', type=Path)
    p.add_argument('--benchmark', type=Path)
    p.add_argument('--aggregate-only', action='store_true', help='Use known generation count; do not infer request boundaries from sparse miss calls')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    tokens = []
    previous = 10**9
    for row in records(args.trace):
        if row['layer'] < previous:
            tokens.append([])
        tokens[-1].append(row)
        previous = row['layer']
    if not tokens:
        raise ValueError('Empty trace')
    result = {'trace': str(args.trace), 'inferred_tokens': len(tokens),
              'all': summarize([r for t in tokens for r in t], len(tokens))}
    if args.benchmark:
        benchmark = json.loads(args.benchmark.read_text())
        # Multi-token prompt prefill produces the first sample; only n-1
        # generated-token evaluations enter the single-token CPU service.
        expected = max(0, benchmark['warmup_tokens']-1)+sum(r['timings']['predicted_n']-1 for r in benchmark['runs'])
        if args.aggregate_only:
            # Initial native batch warmup evaluates multiple tokens per layer.
            # Sparse miss-only traces need not expose every token wrap later.
            initial = 0
            while initial < len(tokens) and len({r['layer'] for r in tokens[initial]}) != len(tokens[initial]):
                initial += 1
            if initial == len(tokens):
                raise ValueError('Cannot identify end of native batch warmup')
            result['initial_batch_warmup_groups'] = initial
            result['initial_batch_warmup_records'] = sum(len(t) for t in tokens[:initial])
            result['inferred_layer_wraps'] = result.pop('inferred_tokens')
            result.pop('all')
            result['generation_evaluations'] = expected
            result['aggregate'] = summarize([r for t in tokens[initial:] for r in t], expected)
            result['request_intervals_inferred'] = False
            args.output.write_text(json.dumps(result, indent=2)+'\n')
            print(json.dumps(result, indent=2))
            return
        initial = len(tokens)-expected
        if initial < 0 or any(len({r['layer'] for r in t}) != len(t) for t in tokens[initial:]) or any(len({r['layer'] for r in t}) == len(t) for t in tokens[:initial]):
            raise ValueError(f'Token count {len(tokens)} differs from expected {expected}; cannot separate native batch warmup')
        result['initial_batch_warmup_groups'] = initial
        tokens = tokens[initial:]
        start = max(0, benchmark['warmup_tokens']-1)
        result['runs'] = []
        for run in benchmark['runs']:
            n = run['timings']['predicted_n']
            rows = [r for t in tokens[start:start+n-1] for r in t]
            summary = summarize(rows, n-1)
            summary['native_ms_per_token'] = 1000/run['timings']['predicted_per_second']
            summary[('q3' if rows[0]['format']=='q3_k' else 'ptq')+'_cpu_fraction_of_native'] = summary['total_ms_per_token']/summary['native_ms_per_token']
            result['runs'].append(summary)
            start += n-1
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
