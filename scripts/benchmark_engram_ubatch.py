#!/usr/bin/env python3
"""Compare bounded Engram prefill reads and 1024/2048/4096 tiered ubatches."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    cases=[('baseline','0','1024'),('engram','1','1024'),('ubatch2048','0','2048'),
           ('ubatch4096','0','4096'),('combined4096','1','4096')]
    records=[]
    for index,(name,engram,ubatch) in enumerate(cases):
        directory=out/f'{index+1:02d}-{name}'
        command=[sys.executable,str(ROOT/'scripts/benchmark_prefill_io.py'),str(directory),
                 '--batch-io','1','--ram-budget-mib','131072','--preload','1',
                 '--io-sort','1','--io-pipeline','1','--engram-prefill-io',engram,
                 '--prefill-ubatch',ubatch,'--prompt-file',str(ROOT/'PROMPT_TEST2.md'),
                 '--runs','1','--block-device','nvme0n1','--engram-timers']
        print(f'Case {index+1}/{len(cases)}: {name}',flush=True)
        with (out/f'{index+1:02d}-{name}.log').open('w') as log:
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        report=json.loads((directory/'report.json').read_text())
        records.append(dict(case=name,directory=str(directory),report=report))
        (out/'reports.json').write_text(json.dumps(records,indent=2)+'\n')
        print(name,report['summary'],flush=True)
    assert len({r['report']['prompt_sha256'] for r in records})==1
    assert len({json.dumps(r['report']['binary_sha256'],sort_keys=True) for r in records})==1
    assert len({x['timings']['prompt_n'] for r in records for x in r['report']['runs']})==1
    summary={}
    for name,_,ubatch in cases:
        reports=[r['report'] for r in records if r['case']==name];row=dict(ubatch=int(ubatch))
        for phase,runs in [('first',[r['runs'][0] for r in reports])]:
            ms=statistics.mean(x['timings']['prompt_ms'] for x in runs)
            row[phase]=dict(samples=len(runs),mean_seconds=ms/1000,tps=runs[0]['timings']['prompt_n']*1000/ms)
            for key in ['expert_read_bytes','device_read_bytes','io_requests','engram_read_ms','engram_wait_ms','input_ms']:
                row[phase]['mean_'+key]=statistics.mean(x[key] for x in runs)
        row['peak_memory_gib']=max(x['after']['cgroup']['memory.peak']/2**30 for r in reports for x in r['runs'])
        row['preload_resident_gib']=[r['preload_snapshot']['weights']['resident_bytes']/2**30 for r in reports]
        row['workspace_log']=reports[0]['runs'][-1]['workspace_log']
        summary[name]=row
    for row in summary.values():
        for phase in ['first']:
            row[phase]['throughput_change_pct']=(row[phase]['tps']/summary['baseline'][phase]['tps']-1)*100
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
