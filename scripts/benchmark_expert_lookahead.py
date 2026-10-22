#!/usr/bin/env python3
"""Fixed teacher-token replay for temporal/early-router lookahead, MTP off."""
import argparse, hashlib, json, os, shlex, shutil, signal, statistics, subprocess, time
from pathlib import Path
from urllib.request import Request, urlopen

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('--mode',choices=['0','1','2','3','4','5'],required=True);p.add_argument('--score',choices=['raw','native'],default='raw');p.add_argument('--depth',type=int,choices=range(1,17),default=8);p.add_argument('--read',type=int,choices=range(1,17));p.add_argument('--profile',action='store_true');p.add_argument('--long',action='store_true');p.add_argument('--teacher',type=Path,default=ROOT/'logs/shared-routed-timing-20261021/teacher-tokens.json');args=p.parse_args()

args.read=args.depth if args.read is None else args.read
if args.read>args.depth:p.error('--read cannot exceed --depth')

out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
teacher_path=args.teacher.resolve()
teacher=json.loads(teacher_path.read_text());assert teacher['prompt']=='Explain story of the romance of three kingdoms.'
manifest=ROOT/'state/server.json';assert not manifest.exists(), 'An owned server is already running'
build=ROOT/'build'
env=dict(os.environ,HOST='127.0.0.1',PORT='18091',BUILD_DIR=str(build),DSV4F_MTP='0',DSV4F_CPU_THREADS='12',DSV4F_EXPERT_LOOKAHEAD=args.mode,DSV4F_EXPERT_LOOKAHEAD_DEPTH=str(args.depth),DSV4F_EXPERT_LOOKAHEAD_READ=str(args.read),DSV4F_EXPERT_LOOKAHEAD_SCORE=args.score,DSV4F_V41_DIRECT_KV='1',DSV4F_V41_HC_POST_PRE_FUSED='1',DSV4F_V41_ROUTE_FIRST='1',DSV4F_V41_ROUTER_FUSED='1',DSV4F_V41_SHARED_FUSED='1',DSV4F_ENGRAM_OVERLAP='1',DSV4F_SCHED_SUMMARY='1',LD_LIBRARY_PATH=str(build/'bin')+':'+os.environ.get('LD_LIBRARY_PATH',''),GGML_BACKEND_PATH=str(build/'bin'))
for key in list(env):
    if key.startswith(('ROCPROF','ROCTRACER')) or key=='LD_PRELOAD' or key.startswith('DSV4F_') and ('PROFILE' in key or 'TIMELINE' in key):env.pop(key)
headers={'Content-Type':'application/json','Authorization':'Bearer '+env.get('API_KEY','dummykey')}
measured=8 if args.profile else 256
report={'mode':args.mode,'feature':'current-hidden-next-router' if int(args.mode)>=3 else 'q3-temporal-lookahead','score':args.score,'depth':args.depth,'read':args.read,'teacher_sha256':hashlib.sha256(teacher_path.read_bytes()).hexdigest(),'prompt':teacher['prompt'],'warmup':128,'measured':measured,'profile':args.profile,'records':[]}
control=None
if args.profile:
    control=out/'collect';control.write_text('0')
    wrapper=out/'wrapper/bin/dsv4f-server';wrapper.parent.mkdir(parents=True)
    cmd=['/opt/rocm/bin/rocprofv3','--selected-regions','--hip-runtime-trace','--marker-trace','--kernel-trace','--output-format','csv','--output-directory',str(out/'trace'),'--output-file','decode','--preload','/opt/rocm/lib/librocprofiler-sdk-roctx.so','--',str(build/'bin/dsv4f-server')]
    wrapper.write_text('#!/bin/sh\nexec '+shlex.join(cmd)+' "$@"\n');wrapper.chmod(0o755)
    env.update(BUILD_DIR=str(out/'wrapper'),DSV4F_TIMELINE='1',DSV4F_TIMELINE_CONTROL=str(control))
    report['trace_command']=cmd
state=None
with (out/'launcher.log').open('w') as log:
    proc=subprocess.Popen(['systemd-run','--user','--scope','--quiet','-p','MemorySwapMax=0','-p','MemoryMax=184G','./run-server.sh'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    try:
        print('Loading mode='+args.mode,flush=True)
        for _ in range(480):
            if proc.poll() is not None:raise RuntimeError('Server exited during load')
            try:
                candidate=json.loads(manifest.read_text())
                assert candidate['url']=='http://127.0.0.1:18091'
                assert candidate['engine_options']['DSV4F_MTP']=='0'
                key='DSV4F_EXPERT_LOOKAHEAD'
                assert candidate['engine_options'][key]==args.mode
                assert candidate['engine_options']['DSV4F_EXPERT_LOOKAHEAD_DEPTH']==str(args.depth)
                assert candidate['engine_options']['DSV4F_EXPERT_LOOKAHEAD_READ']==str(args.read)
                assert candidate['engine_options']['DSV4F_EXPERT_LOOKAHEAD_SCORE']==args.score
                assert candidate['engine_options']['DSV4F_V41_HC_POST_PRE_FUSED']=='1'
                assert Path(f'/proc/{candidate["pid"]}/exe').resolve()==build/'bin/dsv4f-server'
                with urlopen(candidate['url']+'/health',timeout=1) as r:assert json.load(r)['status']=='ok'
                state=candidate;break
            except (OSError,ValueError,AssertionError):time.sleep(1)
        else:raise RuntimeError('Load timeout')
        maps=Path(f'/proc/{state["pid"]}/maps').read_text();(out/'process-maps.txt').write_text(maps)
        for line in maps.splitlines():
            if '/libllama' in line or '/libggml' in line:assert str(build/'bin') in line,line
        report['state']=state
        def memory():
            return {x.split(':')[0]:int(x.split()[1])*1024 for x in Path(f'/proc/{state["pid"]}/status').read_text().splitlines() if x.startswith(('VmRSS:','VmSwap:'))}
        report['memory_before']=memory();assert report['memory_before']['VmSwap']==0
        print(f'Ready; replaying 128 warmup + {measured} measured tokens',flush=True)
        for step in range(128+measured):
            if control and step==128:control.write_text('1')
            body={'prompt':teacher['prefix']+teacher['continuation'][:step],'n_predict':1,'cache_prompt':True,'temperature':0,'seed':42,'ignore_eos':True,'return_tokens':True}
            start=time.perf_counter()
            with urlopen(Request(state['url']+'/completion',data=json.dumps(body).encode(),headers=headers),timeout=180) as response:result=json.load(response)
            record={'step':step,'measured':step>=128,'wall_ms':(time.perf_counter()-start)*1000,'timings':result['timings'],'tokens':result.get('tokens')}
            if step>=128:assert result['timings']['prompt_n']==1,result['timings']
            report['records'].append(record)
            if step%64==63:(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
            if step%64==63:print('step',step,'eval_ms',round(result['timings']['prompt_ms']+result['timings']['predicted_ms'],3),flush=True)
        report['memory_after']=memory();assert report['memory_after']['VmSwap']==0
        if control:control.write_text('0')
        values=[r['timings']['prompt_ms']+r['timings']['predicted_ms'] for r in report['records'] if r['measured']]
        report['summary']={'eval_mean_ms':statistics.mean(values),'eval_median_ms':statistics.median(values),'tokens_per_second':1000/statistics.mean(values),'eval_max_ms':max(values)}
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(report['summary'],flush=True)
        if args.long:
            from benchmark import generate
            prompt=(ROOT/'PROMPT_TEST.md').read_text()
            print('Long prompt: full prefill + 128 decode, cache_prompt=False',flush=True)
            report['long']=generate(state['url'],128,prompt,cache_prompt=False)
            report['long']['prompt_sha256']=hashlib.sha256(prompt.encode()).hexdigest()
            report['long']['memory_after']=memory();assert report['long']['memory_after']['VmSwap']==0
            (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
            print('Long timings',report['long']['timings'],flush=True)
    finally:
        if state:
            try:
                if Path(f'/proc/{state["pid"]}/exe').resolve()==build/'bin/dsv4f-server':os.kill(state['pid'],signal.SIGINT)
            except OSError:pass
        elif proc.poll() is None:os.killpg(proc.pid,signal.SIGINT)
        try:proc.wait(timeout=45)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=10)
        if state:shutil.copyfile(state['log'],out/'server.log')
print('Owned server stopped',flush=True)
