#!/usr/bin/env python3
"""Summarize a HIP/ROCTx-only decode control, without inventing GPU timings."""
import argparse
import collections
import json
from pathlib import Path
import statistics

from summarize_decode_timeline import interval, load, partition, union


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    trace = args.directory/'trace'
    markers = load(next(trace.glob('*marker_api_trace.csv')))
    apis = load(next(trace.glob('*hip_api_trace.csv')))
    for r in markers:
        r['label'] = r.get('Message', r.get('Function', ''))
    resumes = [interval(r)[0] for r in markers if r['label'] == 'roctxProfilerResume']
    pauses = [interval(r)[0] for r in markers if r['label'] == 'roctxProfilerPause']
    assert len(resumes) == len(pauses) == 1
    cluster = sorted(interval(r) for r in markers if r['label'] == 'v41 input/decode'
                     and resumes[0] < interval(r)[0] < pauses[0])
    start, end = cluster[1][0], cluster[-1][0]
    tokens = len(cluster)-2
    scale = 1e-6/tokens
    selected = [r for r in markers if interval(r)[0] >= start and interval(r)[1] <= end]
    apis = [r for r in apis if interval(r)[0] >= start and interval(r)[1] <= end]
    thread = collections.Counter(r['Thread_Id'] for r in apis if r['Function'] == 'hipGraphLaunch').most_common(1)[0][0]
    host_apis = [r for r in apis if r['Thread_Id'] == thread]
    experts = [interval(r) for r in selected if r['label'].startswith('experts/')]
    inputs = [interval(r) for r in selected if r['label'].startswith('v41 input/')]
    callbacks = [interval(r) for r in selected if r['label'].startswith('callback/')]
    sync = [interval(r) for r in host_apis if r['Function'] in
            ('hipStreamSynchronize', 'hipEventSynchronize', 'hipDeviceSynchronize')]
    launches = [interval(r) for r in host_apis if r['Function'] == 'hipGraphLaunch']
    durations = partition({1:experts, 2:inputs, 4:callbacks, 8:sync, 16:launches}, start, end)
    disjoint = collections.Counter()
    for mask, ns in durations.items():
        key = ('cpu_experts' if mask & 1 else 'input' if mask & 2 else
               'callback_without_experts_or_input' if mask & 4 else
               'graph_sync_without_experts_or_input' if mask & 8 else
               'graph_launch_without_experts_or_input' if mask & 16 else 'other_host_time')
        disjoint[key] += ns*scale
    assert abs(sum(disjoint.values())-(end-start)*scale) < 1e-6
    marker_sums = collections.defaultdict(lambda:[0,0])
    for r in selected:
        a,b=interval(r);marker_sums[r['label']][0]+=1;marker_sums[r['label']][1]+=b-a
    api_sums = collections.defaultdict(lambda:[0,0])
    for r in host_apis:
        a,b=interval(r);api_sums[r['Function']][0]+=1;api_sums[r['Function']][1]+=b-a
    token_ms = [(b[0]-a[0])/1e6 for a,b in zip(cluster[1:],cluster[2:])]
    report = dict(tokens=tokens, window_ns=[start,end], wall_ms_per_token=(end-start)*scale,
                  token_ms_median=statistics.median(token_ms),
                  token_ms_p95=sorted(token_ms)[int(.95*(len(token_ms)-1))],
                  disjoint_host_partition_ms_per_token=dict(disjoint),
                  cpu_experts_union_ms_per_token=sum(b-a for a,b in union(experts))*scale,
                  input_ms_per_token=sum(b-a for a,b in union(inputs))*scale,
                  note='No GPU kernel trace: graph-sync time contains GPU execution and waits, not an independent synchronization penalty.',
                  markers=[dict(name=k,calls_per_token=v[0]/tokens,ms_per_token=v[1]*scale)
                           for k,v in sorted(marker_sums.items(),key=lambda item:-item[1][1])],
                  host_apis=[dict(name=k,calls_per_token=v[0]/tokens,ms_per_token=v[1]*scale)
                             for k,v in sorted(api_sums.items(),key=lambda item:-item[1][1])])
    (args.directory/'host-summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
