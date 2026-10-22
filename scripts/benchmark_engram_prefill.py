#!/usr/bin/env python3
"""Test Engram prefill overlap/lookahead with the exact AGENTS.md long prompt.

Decode overlap stays enabled, MTP and prompt reuse are disabled. Only an owned
server is started/stopped. Native prompt timing is primary; lightweight batch
input/async timers explain waits without a GPU profiler or per-row timers.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time
from urllib.request import urlopen

from benchmark import generate
from diagnose_engram_io import snapshot

ROOT=Path(__file__).resolve().parents[1]


def batch_timers(text):
    inputs=[];async_rows=[];prefetch=[];prefetch_reads=[]
    for line in text.splitlines():
        if line.startswith(('[dsv4f-input]','[dsv4f-engram-async]')):
            row={k:float(v) for k,v in re.findall(r'(\w+)=([\d.]+)',line)}
            (inputs if line.startswith('[dsv4f-input]') else async_rows).append(row)
        elif line.startswith(('[dsv4f-engram-prefetch]','[dsv4f-engram-prefetch-read]')):
            row={k:float(v) for k,v in re.findall(r'(\w+)=([\d.]+)',line)}
            (prefetch if line.startswith('[dsv4f-engram-prefetch]') else prefetch_reads).append(row)
    return dict(inputs=inputs,async_rows=async_rows,prefetch=prefetch,prefetch_reads=prefetch_reads,
                input_ms=sum(r['total_us'] for r in inputs)/1000,
                async_wait_ms=sum(r['wait_us'] for r in async_rows)/1000,
                async_read_ms=sum(r['read_us'] for r in async_rows)/1000)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prefill-overlap',choices=('0','1','2'),required=True)
    p.add_argument('--prefetch',choices=('0','1'),default='0')
    p.add_argument('--runs',type=int,default=3)
    p.add_argument('--build-dir',type=Path,default=ROOT/'build')
    args=p.parse_args()
    if args.runs<1:p.error('--runs must be positive')
    if args.prefetch=='1' and args.prefill_overlap!='1':p.error('--prefetch 1 requires --prefill-overlap 1')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    build=args.build_dir.resolve()
    manifest=ROOT/'state/server.json'
    if manifest.exists():raise RuntimeError('Existing server; refusing to replace it')
    env=dict(os.environ)
    for k in list(env):
        if k.startswith(('ROCPROF','ROCTRACER')) or k=='LD_PRELOAD' or (
                k.startswith('DSV4F_') and ('PROFILE' in k or 'TIMELINE' in k)):
            env.pop(k)
    env.update(HOST='127.0.0.1',PORT='18092',BUILD_DIR=str(build),DSV4F_MTP='0',
               DSV4F_ENGRAM_OVERLAP='1',DSV4F_ENGRAM_PREFILL_OVERLAP=args.prefill_overlap,
               DSV4F_ENGRAM_PREFETCH=args.prefetch,
               DSV4F_INPUT_PROFILE='1',DSV4F_ENGRAM_ASYNC_PROFILE='1',
               LD_LIBRARY_PATH=str(build/'bin')+':'+env.get('LD_LIBRARY_PATH',''),
               GGML_BACKEND_PATH=str(build/'bin'))
    prompt=(ROOT/'PROMPT_TEST.md').read_text(encoding='utf-8')
    report=dict(prompt_file='PROMPT_TEST.md',prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                prefill_overlap=args.prefill_overlap,prefetch=args.prefetch,decode_overlap='1',mtp=False,cache_prompt=False,
                output_tokens=1,lightweight_batch_timers=True,gpu_profiler=False,requests=[])
    state=None
    launch=['systemd-run','--user','--scope','--quiet','-p','MemorySwapMax=0','-p','MemoryMax=184G','./run-server.sh']
    with (out/'launcher.log').open('w') as log:
        proc=subprocess.Popen(launch,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            print('Loading prefill mode',args.prefill_overlap,flush=True)
            for _ in range(480):
                if proc.poll() is not None:raise RuntimeError('Server exited; inspect launcher.log')
                try:
                    candidate=json.loads(manifest.read_text())
                    assert candidate['url']=='http://127.0.0.1:18092'
                    assert candidate['engine_options']['DSV4F_MTP']=='0'
                    assert candidate['engine_options']['DSV4F_ENGRAM_PREFILL_OVERLAP']==args.prefill_overlap
                    assert candidate['engine_options']['DSV4F_ENGRAM_PREFETCH']==args.prefetch
                    assert candidate['engine_options']['DSV4F_ENGRAM_OVERLAP']=='1'
                    assert Path(f'/proc/{candidate["pid"]}/exe').resolve()==build/'bin/dsv4f-server'
                    with urlopen(candidate['url']+'/health',timeout=1) as response:
                        assert json.load(response)['status']=='ok'
                    state=candidate;break
                except (OSError,ValueError,AssertionError):time.sleep(1)
            else:raise RuntimeError('Load timeout')
            report['state']=state
            maps=Path(f'/proc/{state["pid"]}/maps').read_text()
            assert 'librocprofiler-sdk' not in maps
            for line in maps.splitlines():
                if any(n in line for n in ('/libllama','/libggml')):assert str(build/'bin') in line
            (out/'process-maps.txt').write_text(maps)
            print('Ready; MTP off; full prompt recomputed each request',flush=True)
            for i in range(args.runs):
                offset=Path(state['log']).stat().st_size
                before=snapshot(state['pid'])
                print('Request',i+1,flush=True)
                result=generate(state['url'],1,prompt,cache_prompt=False)
                result['before']=before;result['after']=snapshot(state['pid'])
                assert result['after']['memory']['VmSwap']==0
                assert result['timings']['cache_n']==0
                with Path(state['log']).open('rb') as f:f.seek(offset);chunk=f.read()
                (out/f'request-{i+1}.log').write_bytes(chunk)
                result['batch_timers']=batch_timers(chunk.decode(errors='replace'))
                assert sum(r['tokens'] for r in result['batch_timers']['inputs'])==result['timings']['prompt_n']
                report['requests'].append(result)
                (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
                print('Completed',i+1,result['timings'],'input_ms',round(result['batch_timers']['input_ms'],3),
                      'async_wait_ms',round(result['batch_timers']['async_wait_ms'],3),flush=True)
        finally:
            if state:
                try:
                    if Path(f'/proc/{state["pid"]}/exe').resolve()==build/'bin/dsv4f-server':os.kill(state['pid'],signal.SIGINT)
                except OSError:pass
            elif proc.poll() is None:os.killpg(proc.pid,signal.SIGINT)
            try:proc.wait(timeout=45)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGINT)
                try:proc.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            if state:shutil.copyfile(state['log'],out/'server.log')
    print('Owned server stopped',flush=True)


if __name__=='__main__':
    main()
