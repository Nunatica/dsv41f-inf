#!/usr/bin/env python3
"""Partition CPU expert/GPU overlap in native decode ROCProfiler CSV traces."""
import argparse
import bisect
import collections
import csv
import json
from pathlib import Path
import statistics


def load(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def interval(row):
    return int(row['Start_Timestamp']), int(row['End_Timestamp'])


def union(spans):
    merged = []
    for a, b in sorted(spans):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return merged


def category(name):
    n = name.lower()
    if any(s in n for s in ('ptr_product', 'ptr_gu', 'ptr_down', 'mxfp4_gu', 'mxfp4_down', 'ptr_mmv')):
        return 'routed_experts'
    if any(s in n for s in ('flash_attn', 'fattn', 'sparse_gather', 'sparse_ids')):
        return 'attention'
    if any(s in n for s in ('lightning', 'argsort', 'top_k', 'topk')):
        return 'indexer_sort'
    if any(s in n for s in ('mul_mat', 'gemv', 'gemm', 'dsv4f_q8_tile')):
        return 'matrix_products'
    if any(s in n for s in ('dequant', 'concat', 'cpy_', 'copy', 'cont_', 'set_rows', 'archive_rows')):
        return 'layout_kv_copy'
    if any(s in n for s in ('route(', 'route_plan_kernel', 'cpu_join', 'release_', 'reserve_slot', 'publish_slot')):
        return 'service_control'
    return 'other'


def partition(groups, start, end):
    events = collections.defaultdict(lambda: collections.Counter())
    for bit, spans in groups.items():
        for a, b in union(spans):
            a, b = max(a, start), min(b, end)
            if b > a:
                events[a][bit] += 1
                events[b][bit] -= 1
    active, durations, previous = collections.Counter(), collections.Counter(), start
    for t in sorted(events):
        mask = sum(bit for bit, count in active.items() if count)
        durations[mask] += t-previous
        active.update(events[t])
        previous = t
    durations[sum(bit for bit, count in active.items() if count)] += end-previous
    assert sum(durations.values()) == end-start
    return durations


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    p.add_argument('--labels', nargs='+', default=['short', 'long'],
                   help='One label per collection window (default: short long)')
    args = p.parse_args()
    trace = args.directory/'trace'
    kernels = load(next(trace.rglob('*kernel_trace.csv')))
    markers = load(next(trace.rglob('*marker_api_trace.csv')))
    apis = load(next(trace.rglob('*hip_api_trace.csv')))
    copy_paths = list(trace.rglob('*memory_copy_trace.csv'))
    copies = load(copy_paths[0]) if copy_paths else []
    # ROCTx range rows contain the user-provided label in Function.
    for r in markers:
        r['label'] = r.get('Message', r.get('Function', ''))
    inputs = sorted(interval(r) for r in markers if r['label'] == 'v41 input/decode')
    if not inputs:
        raise ValueError(f'No input ranges; marker columns: {list(markers[0])}')
    resumes = sorted(interval(r)[0] for r in markers if r['label'] == 'roctxProfilerResume')
    pauses = sorted(interval(r)[0] for r in markers if r['label'] == 'roctxProfilerPause')
    clusters = []
    for resume in resumes:
        pause = next(t for t in pauses if t > resume)
        clusters.append([span for span in inputs if resume < span[0] < pause])
    if len(clusters) != len(args.labels):
        raise ValueError(f'Expected {len(args.labels)} clusters, got {[len(c) for c in clusters]}')
    reports = []
    for label, cluster in zip(args.labels, clusters):
        # Start after the first single-token input and stop at the final input
        # start: only complete, warmed inter-token intervals, on one SDK clock.
        start, end = cluster[1][0], cluster[-1][0]
        tokens = len(cluster)-2
        scale = 1e-6/tokens
        selected = [r for r in kernels if interval(r)[0] >= start and interval(r)[1] <= end]
        selected_markers = [r for r in markers if interval(r)[0] >= start and interval(r)[1] <= end]
        experts = [r for r in selected_markers if r['label'].startswith('experts/')]
        callbacks = [r for r in selected_markers if r['label'].startswith('callback/')]
        input_rows = [r for r in selected_markers if r['label'].startswith('v41 input/')]
        joins = [r for r in selected if 'cpu_join(' in r['Kernel_Name']]
        stream = collections.Counter(r['Stream_Id'] for r in joins).most_common(1)[0][0]
        main_kernels = sorted((r for r in selected if r['Stream_Id'] == stream), key=lambda r: interval(r)[0])
        within_graph_gaps, between_submission_gaps = [], []
        for left, right in zip(main_kernels, main_kernels[1:]):
            gap = interval(left)[1], interval(right)[0]
            if gap[1] > gap[0]:
                (within_graph_gaps if left['Correlation_Id'] == right['Correlation_Id'] else between_submission_gaps).append(gap)
        d2h = [interval(r) for r in copies if r['Direction'] == 'MEMORY_COPY_DEVICE_TO_HOST']
        durations = partition({1: [interval(r) for r in kernels], 2: [interval(r) for r in experts],
                               4: [interval(r) for r in callbacks], 8: [interval(r) for r in input_rows],
                               16: [interval(r) for r in copies], 32: d2h,
                               64: within_graph_gaps, 128: between_submission_gaps}, start, end)
        amounts = collections.defaultdict(float)
        for mask, ns in durations.items():
            amounts['gpu_and_cpu' if mask & 3 == 3 else 'gpu_only' if mask & 1 else
                    'cpu_only' if mask & 2 else 'neither_gpu_nor_experts'] += ns*scale
            for bit, name in ((1, 'gpu_active'), (2, 'cpu_experts_active'), (4, 'callback_active'),
                              (8, 'input_active'), (16, 'copy_active')):
                if mask & bit:
                    amounts[name] += ns*scale
            if not mask & 3:
                amounts['neither_inside_callback' if mask & 4 else 'neither_outside_callback'] += ns*scale
                key = ('uncovered_d2h' if mask & 32 else 'uncovered_callback' if mask & 4 else
                       'uncovered_input' if mask & 8 else
                       'uncovered_within_graph_gap' if mask & 64 else 'uncovered_between_submissions' if mask & 128 else
                       'uncovered_other')
                amounts[key] += ns*scale
        sums, cats = collections.defaultdict(lambda: [0, 0]), collections.Counter()
        for r in selected:
            a, b = interval(r)
            sums[r['Kernel_Name']][0] += 1
            sums[r['Kernel_Name']][1] += b-a
            cats[category(r['Kernel_Name'])] += b-a
        api_sums = collections.defaultdict(lambda: [0, 0])
        for r in apis:
            a, b = interval(r)
            if a >= start and b <= end:
                api_sums[r['Function']][0] += 1
                api_sums[r['Function']][1] += b-a
        marker_sums = collections.defaultdict(lambda: [0, 0])
        for r in selected_markers:
            a, b = interval(r)
            marker_sums[r['label']][0] += 1
            marker_sums[r['label']][1] += b-a
        # Host callbacks stop the main compute stream. Measure the adjacent
        # GPU-to-callback and callback-to-GPU handoff gaps on that stream.
        gpu = sorted(interval(r) for r in kernels if r['Stream_Id'] == stream)
        starts, ends = sorted(a for a, b in gpu), sorted(b for a, b in gpu)
        handoffs = collections.Counter()
        for r in callbacks:
            a, b = interval(r)
            before, after = bisect.bisect_right(ends, a)-1, bisect.bisect_left(starts, b)
            if before >= 0 and after < len(starts):
                handoffs['gpu_to_callback_ms_per_token'] += (a-ends[before])*scale
                handoffs['callback_to_gpu_ms_per_token'] += (starts[after]-b)*scale
        token_times = [(right[0]-left[0])/1e6 for left, right in zip(cluster[1:], cluster[2:])]
        median = statistics.median(token_times)
        reports.append(dict(label=label, tokens=tokens, window_ns=[start, end],
                            token_ms_median=median, token_ms_p95=sorted(token_times)[int(.95*(len(token_times)-1))],
                            token_ms_max=max(token_times), intervals_above_3x_median=sum(t>3*median for t in token_times),
                            mean_ms_excluding_3x_outliers=statistics.mean(t for t in token_times if t<=3*median),
                            wall_ms_per_token=(end-start)*scale, partition_ms_per_token=dict(amounts),
                            callback_handoffs=dict(handoffs), kernel_categories_ms_per_token=dict(cats),
                            kernels=[dict(name=n, calls_per_token=v[0]/tokens, ms_per_token=v[1]*scale)
                                     for n, v in sorted(sums.items(), key=lambda x: -x[1][1])],
                            apis=[dict(name=n, calls_per_token=v[0]/tokens, ms_per_token=v[1]*scale)
                                  for n, v in sorted(api_sums.items(), key=lambda x: -x[1][1])],
                            markers=[dict(name=n, calls_per_token=v[0]/tokens, ms_per_token=v[1]*scale)
                                     for n, v in sorted(marker_sums.items(), key=lambda x: -x[1][1])]))
        reports[-1]['kernel_categories_ms_per_token'] = {k: v*scale for k, v in cats.items()}
    (args.directory/'timeline-summary.json').write_text(json.dumps(reports, indent=2))
    for row in reports:
        print(json.dumps({k: v for k, v in row.items() if k not in ('kernels', 'apis')}, indent=2))


if __name__ == '__main__':
    main()
