#!/usr/bin/env python3
"""Attribute decode inter-kernel gaps after subtracting measured useful work.

Run summarize_decode_timeline.py first. GPU, CPU expert, input, callback and
copy spans are unioned before subtraction, so overlapping spans are not added.
HIP API overlaps are diagnostic observations, not additional elapsed costs.
"""
import argparse
import bisect
import collections
import csv
import json
from pathlib import Path
import statistics

from summarize_decode_timeline import category, interval, union


class SpanIndex:
    def __init__(self, spans):
        self.spans = union(spans)
        self.starts = [a for a, b in self.spans]
        self.ends = [b for a, b in self.spans]

    def overlap(self, a, b):
        result = 0
        i = bisect.bisect_right(self.ends, a)
        while i < len(self.spans) and self.starts[i] < b:
            lo, hi = self.spans[i]
            result += max(0, min(b, hi)-max(a, lo))
            i += 1
        return result

    def complement(self, a, b):
        result = []
        for lo, hi in self.spans:
            if hi <= a:
                continue
            if lo >= b:
                break
            if lo > a:
                result.append((a, lo))
            a = max(a, hi)
        if a < b:
            result.append((a, b))
        return result


def read_window(path, start, end):
    with path.open() as f:
        return [r for r in csv.DictReader(f)
                if int(r['End_Timestamp']) > start and int(r['Start_Timestamp']) < end]


def name(kernel):
    kernel = kernel.replace('(anonymous namespace)::', '').replace('(ggml_type)', '')
    return kernel.split('>(')[0]+'>' if '>(' in kernel else kernel.split('(')[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    summaries = json.loads((args.directory/'timeline-summary.json').read_text())
    assert len(summaries) == 1, 'One collection window is required'
    summary = summaries[0]
    start, end = summary['window_ns']
    tokens = summary['tokens']
    scale = 1e-6/tokens
    trace = args.directory/'trace'
    kernels = read_window(next(trace.glob('*kernel_trace.csv')), start, end)
    markers = read_window(next(trace.glob('*marker_api_trace.csv')), start, end)
    apis = read_window(next(trace.glob('*hip_api_trace.csv')), start, end)
    copy_paths = list(trace.glob('*memory_copy_trace.csv'))
    copies = read_window(copy_paths[0], start, end) if copy_paths else []
    for r in markers:
        r['label'] = r.get('Message', r.get('Function', ''))
    experts = [r for r in markers if r['label'].startswith('experts/')]
    inputs = [r for r in markers if r['label'].startswith('v41 input/')]
    callbacks = [r for r in markers if r['label'].startswith('callback/')]
    stream = collections.Counter(r['Stream_Id'] for r in kernels
                                 if 'cpu_join(' in r['Kernel_Name']).most_common(1)[0][0]
    main_kernels = sorted((r for r in kernels if r['Stream_Id'] == stream), key=interval)
    covered = SpanIndex(interval(r) for r in kernels+experts+inputs+callbacks+copies)
    residual_spans = covered.complement(start, end)
    residual = SpanIndex(residual_spans)
    gpu_idle = SpanIndex(interval(r) for r in kernels).complement(start, end)
    pairs = collections.defaultdict(lambda: dict(count=0, raw_ns=0, uncovered_ns=0, samples=[]))
    categories = collections.Counter()
    histogram = collections.defaultdict(lambda: dict(count=0, raw_ns=0, uncovered_ns=0))
    largest = []
    within = collections.Counter()
    for left, right in zip(main_kernels, main_kernels[1:]):
        a, b = max(start, interval(left)[1]), min(end, interval(right)[0])
        if b <= a:
            continue
        uncovered = residual.overlap(a, b)
        kind = 'same_correlation' if left['Correlation_Id'] == right['Correlation_Id'] else 'different_correlation'
        key = (name(left['Kernel_Name']), name(right['Kernel_Name']), kind)
        row = pairs[key]
        row['count'] += 1
        row['raw_ns'] += b-a
        row['uncovered_ns'] += uncovered
        row['samples'].append((b-a)/1000)
        categories[category(left['Kernel_Name'])+' -> '+category(right['Kernel_Name'])] += uncovered
        within[kind] += uncovered
        us = (b-a)/1000
        bucket = '<=1us' if us <= 1 else '1-3us' if us <= 3 else '3-10us' if us <= 10 else '10-100us' if us <= 100 else '100-1000us' if us <= 1000 else '>1000us'
        h = histogram[bucket]
        h['count'] += 1
        h['raw_ns'] += b-a
        h['uncovered_ns'] += uncovered
        largest.append(dict(start_ns=a, end_ns=b, gap_us=us, uncovered_us=uncovered/1000,
                            previous=key[0], following=key[1], correlation=kind))
    # API time often includes waiting for GPU execution. Its intersection with
    # an unexplained GPU gap locates the wait; it does not establish its cause.
    graph_threads = collections.Counter(r['Thread_Id'] for r in apis if r['Function'] == 'hipGraphLaunch')
    host_thread = graph_threads.most_common(1)[0][0]
    host_api = collections.defaultdict(list)
    host_counts = collections.Counter()
    for r in apis:
        if r['Thread_Id'] == host_thread:
            host_api[r['Function']].append(interval(r))
            host_counts[r['Function']] += 1
    api_overlap = []
    for function, spans in host_api.items():
        idx = SpanIndex(spans)
        overlap = sum(idx.overlap(a, b) for a, b in residual_spans)
        api_overlap.append(dict(name=function, calls_per_token=host_counts[function]/tokens,
                                api_ms_per_token=sum(b-a for a, b in spans)*scale,
                                residual_overlap_ms_per_token=overlap*scale))
    duration_histogram = collections.defaultdict(lambda: dict(count=0, ns=0))
    for r in kernels:
        a, b = interval(r)
        ns = min(end, b)-max(start, a)
        bucket = '<=2us' if ns <= 2000 else '2-5us' if ns <= 5000 else '5-10us' if ns <= 10000 else '>10us'
        duration_histogram[bucket]['count'] += 1
        duration_histogram[bucket]['ns'] += ns
    ranked = []
    for (left, right, kind), r in sorted(pairs.items(), key=lambda item: -item[1]['uncovered_ns']):
        samples = sorted(r['samples'])
        ranked.append(dict(previous=left, following=right, correlation=kind,
                           calls_per_token=r['count']/tokens,
                           raw_ms_per_token=r['raw_ns']*scale,
                           uncovered_ms_per_token=r['uncovered_ns']*scale,
                           gap_median_us=statistics.median(samples),
                           gap_p95_us=samples[int(.95*(len(samples)-1))]))
    assert sum(within.values()) <= sum(b-a for a, b in residual_spans)
    report = dict(tokens=tokens, window_ns=[start, end], main_stream=stream,
                  host_graph_launch_thread=host_thread,
                  kernels_per_token=len(kernels)/tokens,
                  main_kernels_per_token=len(main_kernels)/tokens,
                  gpu_idle_ms_per_token=sum(b-a for a, b in gpu_idle)*scale,
                  residual_ms_per_token=sum(b-a for a, b in residual_spans)*scale,
                  residual_inter_kernel_ms_per_token={k:v*scale for k, v in within.items()},
                  residual_by_adjacent_categories_ms_per_token={k:v*scale for k, v in categories.most_common()},
                  gap_histogram={k:dict(calls_per_token=v['count']/tokens,
                                        raw_ms_per_token=v['raw_ns']*scale,
                                        uncovered_ms_per_token=v['uncovered_ns']*scale)
                                 for k, v in histogram.items()},
                  kernel_duration_histogram={k:dict(calls_per_token=v['count']/tokens,
                                                    kernel_ms_per_token=v['ns']*scale)
                                             for k, v in duration_histogram.items()},
                  host_apis=sorted(api_overlap, key=lambda r:-r['residual_overlap_ms_per_token']),
                  adjacent_pairs=ranked[:50],
                  largest_gaps=sorted(largest, key=lambda r:-r['uncovered_us'])[:20])
    (args.directory/'gap-analysis.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('adjacent_pairs','largest_gaps')},indent=2))


if __name__ == '__main__':
    main()
