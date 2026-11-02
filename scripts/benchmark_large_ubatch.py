#!/usr/bin/env python3
"""One first prefill per fresh 128-GiB server, using PROMPT_TEST.md."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    records=[];capacity_failure=None
    for ubatch in [4096,6144,8192,10240]:
        if capacity_failure:
            records.append(dict(ubatch=ubatch,status='skipped',reason=capacity_failure))
            (out/'summary.json').write_text(json.dumps(records,indent=2)+'\n')
            continue
        directory=out/str(ubatch)
        command=[sys.executable,str(ROOT/'scripts/benchmark_prefill_io.py'),str(directory),
                 '--batch-io','1','--ram-budget-mib','131072','--preload','1',
                 '--io-sort','1','--io-pipeline','1','--engram-prefill-io','0',
                 '--prefill-ubatch',str(ubatch),'--prompt-file',str(ROOT/'PROMPT_TEST.md'),
                 '--runs','1','--block-device','nvme0n1','--engram-timers','--sample-vram']
        print('Fresh server, first prefill only, ubatch',ubatch,flush=True)
        with (out/f'{ubatch}.log').open('w') as log:
            completed=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        report_path=directory/'report.json'
        report=json.loads(report_path.read_text()) if report_path.exists() else {}
        if completed.returncode:
            text='\n'.join(p.read_text(errors='replace') for p in [directory/'launcher.log',out/f'{ubatch}.log'] if p.exists())
            vram_failure=bool(re.search(r'insufficient VRAM|hipErrorOutOfMemory|hipMalloc.*failed|failed to allocate.*ROCm|cudaMalloc.*failed',text,re.I))
            row=dict(ubatch=ubatch,status='failed',returncode=completed.returncode,error=report.get('error'),vram_failure=vram_failure)
            records.append(row);(out/'summary.json').write_text(json.dumps(records,indent=2)+'\n')
            if not vram_failure:
                raise RuntimeError(f'ubatch {ubatch} failed without a confirmed VRAM capacity error; inspect logs')
            capacity_failure=f'VRAM capacity exhausted at ubatch {ubatch}'
            print(capacity_failure,flush=True)
            continue
        assert len(report['runs'])==1
        run=report['runs'][0]
        row=dict(ubatch=ubatch,status='passed',tps=run['timings']['prompt_per_second'],
                 seconds=run['timings']['prompt_ms']/1000,prompt_n=run['timings']['prompt_n'],
                 expert_read_gib=run['expert_read_bytes']/2**30,device_read_gib=run['device_read_bytes']/2**30,
                 ram_peak_gib=run['after']['cgroup']['memory.peak']/2**30,
                 vram_peak_gib=run['vram_peak_bytes']/2**30,vram_min_free_mib=run['vram_min_free_bytes']/2**20,
                 engram_read_ms=run['engram_read_ms'],engram_wait_ms=run['engram_wait_ms'],
                 observed_ubatches=run['observed_prefill_ubatches'],workspace_log=run['workspace_log'],
                 prompt_sha256=report['prompt_sha256'],binary_sha256=report['binary_sha256'])
        if records:
            baseline=records[0]
            assert row['prompt_n']==baseline['prompt_n'] and row['prompt_sha256']==baseline['prompt_sha256']
            assert row['binary_sha256']==baseline['binary_sha256']
            row['throughput_change_pct']=(row['tps']/baseline['tps']-1)*100
        else: row['throughput_change_pct']=0.0
        records.append(row);(out/'summary.json').write_text(json.dumps(records,indent=2)+'\n')
        print(json.dumps(row),flush=True)


if __name__=='__main__':
    main()
