#!/usr/bin/env python3
"""Validate global expert traces, export frequencies and replay RAM caches.

Q4_K/Q4_0 estimates assume all routed gate/up/down tensors use 144/256
bytes per parameter. Non-expert weights, KV, staging and Engram are additional.
Belady is an offline lower bound, not an implementable predictor.
"""
import argparse
from collections import Counter, OrderedDict, defaultdict
import csv
import heapq
import json
from pathlib import Path
import statistics
from model_info import header

GiB = 2**30
MiB = 2**20


def load_routes(path, start, end, layers, experts, topk):
    """Transpose per-layer ubatches into token-major global (layer,expert) keys."""
    chunks = []
    current = None
    with Path(path).open('rb') as f:
        f.seek(start)
        while f.tell() < end:
            line = f.readline()
            fields = line.split()
            if not fields or fields[0] != b'R':
                raise ValueError('Invalid route record')
            layer, nt, nk = map(int, fields[1:4])
            ids = list(map(int, fields[4:]))
            if nk != topk or len(ids) != nt*nk or not 0 <= layer < layers:
                raise ValueError('Invalid route geometry')
            if any(not 0 <= eid < experts for eid in ids):
                raise ValueError('Invalid global expert ID')
            if layer == 0:
                if current is not None and len(current) != layers:
                    raise ValueError('Incomplete preceding ubatch')
                current = []
                chunks.append(current)
            if current is None or layer != len(current):
                raise ValueError('Missing, repeated or unordered layer')
            if current and nt != len(current[0]):
                raise ValueError('Inconsistent layer token counts')
            rows = [ids[t*nk:(t+1)*nk] for t in range(nt)]
            if any(len(set(row)) != nk for row in rows):
                raise ValueError('Duplicate expert in top-k')
            current.append(rows)
        if f.tell() != end or not current or len(current) != layers:
            raise ValueError('Incomplete trace or byte interval')
    tokens = []
    for chunk in chunks:
        for t in range(len(chunk[0])):
            tokens.append([l*experts+eid for l in range(layers) for eid in chunk[l][t]])
    return tokens, [len(c[0]) for c in chunks]


def lru_replay(sequence, capacity, cache=None):
    cache = OrderedDict() if cache is None else OrderedDict(cache)
    misses = []
    for key in sequence:
        miss = key not in cache
        misses.append(int(miss))
        if not miss:
            cache.move_to_end(key)
        elif capacity:
            if len(cache) == capacity:
                cache.popitem(last=False)
            cache[key] = None
    return misses, cache


def lru_misses(sequence, capacity):
    return lru_replay(sequence, capacity)[0]


def belady_misses(sequence, capacity):
    """Farthest-future replacement, cold start; exact whole-expert lower bound."""
    if not capacity:
        return [1]*len(sequence)
    future = [len(sequence)]*len(sequence)
    last = {}
    for i in range(len(sequence)-1, -1, -1):
        key = sequence[i]
        future[i] = last.get(key, len(sequence))
        last[key] = i
    cache = {}
    heap = []
    misses = []
    for i, key in enumerate(sequence):
        miss = key not in cache
        misses.append(int(miss))
        if miss and len(cache) == capacity:
            while heap:
                neg, victim = heapq.heappop(heap)
                if cache.get(victim) == -neg:
                    del cache[victim]
                    break
        cache[key] = future[i]
        heapq.heappush(heap, (-future[i], key))
        # Bound stale heap storage on traces with frequent hits.
        if len(heap) > 4*capacity+1024:
            heap = [(-value, key) for key, value in cache.items()]
            heapq.heapify(heap)
    return misses


def percentile(values, p):
    return sorted(values)[min(len(values)-1, int((len(values)-1)*p))]


def phase_summary(tokens, total, experts):
    counts = Counter(k for row in tokens for k in row)
    sorted_counts = sorted(counts.values(), reverse=True)
    route_count = sum(counts.values())
    concentration = {str(p): sum(sorted_counts[:int(total*p)])/route_count
                     for p in (.1, .25, .5, .6, .75, .9)}
    layers = defaultdict(set)
    for key in counts:
        layers[key//experts].add(key%experts)
    reuse = []
    last = {}
    for i, row in enumerate(tokens):
        for key in row:
            if key in last:
                reuse.append(i-last[key])
            last[key] = i
    windows = {}
    for size in (1, 8, 32, 128, 256):
        if len(tokens) < size:
            continue
        counter = Counter(k for row in tokens[:size] for k in row)
        unions = [len(counter)]
        for i in range(size, len(tokens)):
            counter.subtract(tokens[i-size])
            for key in tokens[i-size]:
                if counter[key] == 0:
                    del counter[key]
            counter.update(tokens[i])
            unions.append(len(counter))
        windows[str(size)] = dict(mean_unique=statistics.mean(unions), p95_unique=percentile(unions, .95),
                                  max_unique=max(unions), mean_fraction=statistics.mean(unions)/total)
    return dict(tokens=len(tokens), selections=route_count, unique=len(counts), total_experts=total,
                used_fraction=len(counts)/total, unused=total-len(counts),
                layer_unique_min=min(map(len,layers.values())), layer_unique_max=max(map(len,layers.values())),
                top_expert_fraction_to_selection_coverage=concentration,
                reuse_gap_tokens=dict(median=statistics.median(reuse), p90=percentile(reuse,.9),
                                      p99=percentile(reuse,.99), within_1=sum(g<=1 for g in reuse)/len(reuse),
                                      within_8=sum(g<=8 for g in reuse)/len(reuse)) if reuse else {},
                windows=windows)


def analyze(folder, output, capacities, total_model_gib=289.0):
    report = json.loads((folder/'report.json').read_text())
    meta, _, _ = header(report['state']['model'])
    layers = meta['deepseek41.num_hidden_layers']
    experts = meta['deepseek41.n_routed_experts']
    topk = meta['deepseek41.num_experts_per_tok']
    total = layers*experts
    reference_expert_bytes = 3*meta['deepseek41.hidden_size']*meta['deepseek41.moe_intermediate_size']*144//256
    # Preserve the routed/non-routed split derived from the model geometry when
    # the caller supplies a corrected aggregate weight size.
    reference_model_gib = 289.0
    expert_bytes = round(reference_expert_bytes*(total_model_gib/reference_model_gib))
    output.mkdir(parents=True, exist_ok=True)
    summary = dict(model=report['state']['model'], layers=layers, experts_per_layer=experts, topk=topk,
                   total_routed_experts=total, selected_per_token=layers*topk,
                   original_q4_model_gib=total_model_gib,
                   q4_assumption='Routed/non-routed split scaled from the 289 GiB Q4 geometry estimate',
                   q4_expert_bytes=expert_bytes, q4_all_routed_gib=total*expert_bytes/GiB,
                   q4_non_routed_weights_gib=total_model_gib-total*expert_bytes/GiB,
                   q4_no_cache_gib_per_token=layers*topk*expert_bytes/GiB,
                   caveats=['Routes measured on the available dsx 2.6bit model; original Q4 routes may differ.',
                            'Shared experts and non-expert weights are excluded from expert counts and cache capacity.',
                            'Trace timings include scheduler synchronization and readback overhead.',
                            'LRU/Belady replay is CPU RAM/SSD cache simulation; no disk streaming or Q4 compute was executed.',
                            'Prefill cache simulation uses logical token-major order; actual batched execution can reuse experts across the ubatch.',
                            'SSD bandwidth scenarios are assumptions, not measured random-read throughput.'], runs=[])
    frequency_rows = []
    cache_rows = []
    cumulative_rows = []
    for record in report['records']:
        label = record['label']
        tokens, chunks = load_routes(folder/'routes.trace', record['trace_start'], record['trace_end'], layers, experts, topk)
        prefill_n = record['timings']['prompt_n']
        decode_n = record['timings']['predicted_n']-1
        if len(tokens) != prefill_n+decode_n:
            raise ValueError(f'{label}: traced {len(tokens)} != prefill {prefill_n} + decode {decode_n}')
        phases = dict(prefill=tokens[:prefill_n], decode=tokens[prefill_n:], all=tokens)
        run = dict(label=label, timings_instrumented=record['timings'], chunks=chunks,
                   phases={name:phase_summary(rows,total,experts) for name,rows in phases.items()})
        for phase, rows in phases.items():
            counts = Counter(k for row in rows for k in row)
            ranks = Counter((key,rank%topk) for row in rows for rank,key in enumerate(row))
            layer_counts = Counter(k//experts for k in counts)
            for key in range(total):
                frequency_rows.append([label,phase,key//experts,key%experts,counts[key],len(rows),
                                       counts[key]/len(rows),counts[key]/(len(rows)*layers*topk),
                                       *[ranks[key,rank] for rank in range(topk)]])
            seen = set()
            for i,row in enumerate(rows):
                seen.update(row)
                if i<32 or (i+1)%32==0 or i+1==len(rows):
                    cumulative_rows.append([label,phase,i+1,len(seen),len(seen)/total,len(seen)*expert_bytes/GiB])
        # Prefill precedes decode: cache is warmed only by that request's own prefill.
        sequence = [k for row in tokens for k in row]
        boundary = prefill_n*layers*topk
        run['cache'] = []
        for gib in capacities:
            capacity = min(total,int(gib*GiB//expert_bytes))
            for policy, method in [('lru',lru_misses),('belady_offline',belady_misses)]:
                if policy == 'lru':
                    misses, warmed_cache = lru_replay(sequence, capacity)
                else:
                    misses = method(sequence,capacity)
                dm = misses[boundary:]
                per_token = [sum(dm[i:i+layers*topk]) for i in range(0,len(dm),layers*topk)]
                entry = dict(capacity_gib=gib,capacity_experts=capacity,capacity_fraction=capacity/total,policy=policy,
                             prefill_misses=sum(misses[:boundary]),decode_misses=sum(dm),
                             decode_hit_fraction=1-sum(dm)/len(dm),
                             decode_misses_per_token=statistics.mean(per_token),
                             decode_misses_p95=percentile(per_token,.95),
                             decode_mib_per_token=statistics.mean(per_token)*expert_bytes/MiB)
                entry['io_only_tps_at_gib_s'] = {str(bw): bw*1024/entry['decode_mib_per_token']
                                               if entry['decode_mib_per_token'] else None for bw in (1,2,3,5)}
                entry['decode_io_gib'] = sum(dm)*expert_bytes/GiB
                if policy == 'lru':
                    warm_misses, _ = lru_replay(sequence, capacity, warmed_cache)
                    warm_dm = warm_misses[boundary:]
                    entry['warm_replay_decode_hit_fraction'] = 1-sum(warm_dm)/len(warm_dm)
                    entry['warm_replay_decode_misses_per_token'] = sum(warm_dm)/len(phases['decode'])
                    entry['warm_replay_decode_mib_per_token'] = entry['warm_replay_decode_misses_per_token']*expert_bytes/MiB
                    hottest = {key for key, _ in Counter(sequence).most_common(capacity)}
                    static_misses = sum(key not in hottest for key in sequence[boundary:])
                    entry['oracle_static_hot_decode_hit_fraction'] = 1-static_misses/len(dm)
                    entry['oracle_static_hot_decode_misses_per_token'] = static_misses/len(phases['decode'])
                run['cache'].append(entry)
                cache_rows.append([label,gib,capacity,capacity/total,policy,entry['decode_hit_fraction'],
                                   entry['decode_misses_per_token'],entry['decode_misses_p95'],entry['decode_mib_per_token'],
                                   entry.get('warm_replay_decode_hit_fraction'),
                                   entry.get('warm_replay_decode_misses_per_token'),
                                   entry.get('warm_replay_decode_mib_per_token'),
                                   entry.get('oracle_static_hot_decode_hit_fraction'),
                                   entry.get('oracle_static_hot_decode_misses_per_token'),
                                   *[entry['io_only_tps_at_gib_s'][str(bw)] for bw in (1,2,3,5)]])
        summary['runs'].append(run)
        print(label, {p:(s['tokens'],s['unique']) for p,s in run['phases'].items()},flush=True)
    for name, headings, rows in [
        ('expert_frequency.csv',['prompt','phase','layer','expert','selections','phase_tokens','token_selection_probability','share_of_all_selections']+[f'rank_{i+1}' for i in range(topk)],frequency_rows),
        ('cache_replay.csv',['prompt','expert_cache_gib','capacity_experts','capacity_fraction','policy','decode_hit_fraction','misses_per_token','misses_p95','q4_mib_per_token','warm_replay_hit_fraction','warm_replay_misses_per_token','warm_replay_q4_mib_per_token','oracle_static_hot_hit_fraction','oracle_static_hot_misses_per_token','io_only_tps_1gibs','io_only_tps_2gibs','io_only_tps_3gibs','io_only_tps_5gibs'],cache_rows),
        ('cumulative_usage.csv',['prompt','phase','tokens','unique_experts','used_fraction','q4_working_set_gib'],cumulative_rows)]:
        with (output/name).open('w') as f:
            writer = csv.writer(f)
            writer.writerow(headings)
            writer.writerows(rows)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--capacities-gib',type=float,nargs='+',default=[96,128,144,152,160,168,176])
    p.add_argument('--total-weights-gib',type=float,default=289.0)
    args=p.parse_args()
    analyze(args.folder,args.output or args.folder.parent/'analysis',args.capacities_gib,args.total_weights_gib)


if __name__=='__main__':
    main()
