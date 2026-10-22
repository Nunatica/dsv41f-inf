#!/usr/bin/env python3
"""Compare three Engram modes; retain raw timings and a declared stall filter."""
import argparse
import json
from pathlib import Path
import statistics


def stable_metrics(request):
    points = request['points']
    steps = []
    for left, right in zip(points, points[1:]):
        n = right.get('predicted_n', 0)
        dn = n-left.get('predicted_n', 0)
        dt = right.get('predicted_ms', 0)-left.get('predicted_ms', 0)
        if n > 128 and dn > 0 and dt > 0:
            steps.append((dn, dt, n))
    median = statistics.median(dt/dn for dn,dt,n in steps)
    kept = [(dn,dt,n) for dn,dt,n in steps if dt/dn <= 3*median]
    return dict(start_after_token=128, interval_filter='exclude per-token intervals above 3x median',
                tokens=sum(dn for dn,dt,n in kept),
                tokens_per_second=1000*sum(dn for dn,dt,n in kept)/sum(dt for dn,dt,n in kept),
                median_ms=median,
                excluded=[dict(end_n=n, tokens=dn, ms=dt) for dn,dt,n in steps if dt/dn > 3*median])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    args=p.parse_args()
    reports={name:json.loads((args.root/path/'report.json').read_text()) for name,path in
             [('sync','sync-run'),('all_async','overlap-run'),('later_async','later-run')]}
    result=[]
    for index in range(len(reports['sync']['requests'])):
        row={}
        for name,report in reports.items():
            request=report['requests'][index]
            row[name]=dict(seed=request['request']['seed'],timings=request['timings'],
                           stable=stable_metrics(request))
        baseline=row['sync']['stable']['tokens_per_second']
        for name in ('all_async','later_async'):
            row[name]['stable_gain_percent']=100*(row[name]['stable']['tokens_per_second']/baseline-1)
        result.append(row)
    output=dict(note='Unprofiled natural outputs; text/routes can differ. Raw and filtered metrics are both retained. '
                     'The fixed warm-up and 3x filter apply identically to every mode, per the request to ignore intermittent SSD stalls.',
                requests=result)
    (args.root/'comparison.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))


if __name__=='__main__':
    main()
