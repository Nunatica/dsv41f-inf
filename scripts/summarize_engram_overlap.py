#!/usr/bin/env python3
"""Measure Engram worker overlap and consumption waits in a decode GPU trace."""
import argparse
import json
from pathlib import Path
import statistics
from summarize_decode_timeline import load, interval, partition, union


def restrict(spans, windows):
    result=[]
    i=0
    for a,b in union(spans):
        while i<len(windows) and windows[i][1]<=a:
            i+=1
        j=i
        while j<len(windows) and windows[j][0]<b:
            x,y=max(a,windows[j][0]),min(b,windows[j][1])
            if y>x:
                result.append((x,y))
            j+=1
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    root = args.directory
    summaries = json.loads((root/'timeline-summary.json').read_text())
    markers = load(next((root/'trace').rglob('*marker_api_trace.csv')))
    kernels = load(next((root/'trace').rglob('*kernel_trace.csv')))
    for row in markers:
        row['label'] = row.get('Message', row.get('Function', ''))
    reports = []
    for summary in summaries:
        start, end = summary['window_ns']
        tokens = summary['tokens']
        scale = 1e-6/tokens
        gpu = [interval(r) for r in kernels]
        experts = [interval(r) for r in markers if r['label'].startswith('experts/')]
        report = dict(label=summary['label'], tokens=tokens,
                      wall_ms_per_token=summary['wall_ms_per_token'], tables=[])
        for suffix in ('slot0', 'later'):
            read_label = 'v41 engram/read '+suffix
            wait_label = 'v41 engram/wait '+suffix
            reads = [interval(r) for r in markers if r['label'] == read_label]
            waits = [interval(r) for r in markers if r['label'] == wait_label]
            durations = partition({1: reads, 2: gpu, 4: experts}, start, end)
            sums = dict(read_ms=0., read_with_gpu_ms=0., read_with_experts_ms=0.,
                        read_with_either_ms=0., read_without_either_ms=0.)
            for mask, ns in durations.items():
                if mask & 1:
                    sums['read_ms'] += ns*scale
                    sums['read_with_gpu_ms'] += ns*scale if mask & 2 else 0.
                    sums['read_with_experts_ms'] += ns*scale if mask & 4 else 0.
                    sums['read_with_either_ms'] += ns*scale if mask & 6 else 0.
                    sums['read_without_either_ms'] += ns*scale if not mask & 6 else 0.
            sums['consume_wait_ms'] = sum(max(0, min(b,end)-max(a,start)) for a,b in waits)*scale
            report['tables'].append(dict(table=suffix, **sums))
        report['note'] = ('Worker read ranges include FP8 decode and pinned-buffer copy. '
                          'Consumption-wait ranges include callback bookkeeping. '
                          'Overlap is temporal intersection, not independently measured latency saved.')
        inputs=sorted(interval(r)[0] for r in markers if r['label']=='v41 input/decode' and start<=interval(r)[0]<=end)
        windows=list(zip(inputs,inputs[1:]))
        assert len(windows)==tokens
        median=statistics.median(b-a for a,b in windows)
        kept=[(a,b) for a,b in windows if b-a<=3*median]
        normal_scale=1e-6/len(kept)
        filtered=dict(tokens=len(kept), excluded_intervals=len(windows)-len(kept),
                      wall_ms_per_token=sum(b-a for a,b in kept)*normal_scale,tables=[])
        gpu_kept=restrict(gpu,kept);experts_kept=restrict(experts,kept)
        for suffix in ('slot0','later'):
            reads=restrict([interval(r) for r in markers if r['label']=='v41 engram/read '+suffix],kept)
            waits=restrict([interval(r) for r in markers if r['label']=='v41 engram/wait '+suffix],kept)
            durations=partition({1:reads,2:gpu_kept,4:experts_kept},start,end)
            read=sum(ns for mask,ns in durations.items() if mask&1)*normal_scale
            overlap=sum(ns for mask,ns in durations.items() if mask&1 and mask&6)*normal_scale
            filtered['tables'].append(dict(table=suffix,read_ms=read,read_with_either_ms=overlap,
                read_without_either_ms=read-overlap,consume_wait_ms=sum(b-a for a,b in waits)*normal_scale))
        report['excluding_3x_intervals']=filtered
        reports.append(report)
    (root/'engram-overlap-summary.json').write_text(json.dumps(reports, indent=2)+'\n')
    print(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()
