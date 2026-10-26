#!/usr/bin/env python3
"""Compare cold/warm PROMPT_TEST2 prefill in a fresh 160-GiB server, MTP off."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import time
from urllib.request import urlopen

from benchmark import generate

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--batch-io', choices=['0', '1'], required=True)
    parser.add_argument('--runs', type=int, default=3)
    args = parser.parse_args()
    if args.runs < 2:
        parser.error('--runs must be at least 2 (cold and warm)')
    manifest = ROOT/'state/server.json'
    if manifest.exists():
        raise RuntimeError('An owned server already exists')
    prompt = (ROOT/'PROMPT_TEST2.md').read_text(encoding='utf-8')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    build = ROOT/'build'
    env = dict(os.environ, HOST='127.0.0.1', PORT='18094', BUILD_DIR=str(build),
               DSV4F_RAM_BUDGET_MIB='163840', DSV4F_MTP='0', DSV4F_EXPERT_LOOKAHEAD='0',
               DSV4F_RAM_LOOKAHEAD='0', DSV4F_PREFILL_GROUP_LEASE='1',
               DSV4F_PREFILL_BATCH_IO=args.batch_io, DSV4F_CPU_EXPERT_LEASE='1',
               DSV4F_RAM_UNLOCK_IO='1', DSV4F_ENGRAM_BATCH_IO='1',
               LD_LIBRARY_PATH=str(build/'bin')+':'+os.environ.get('LD_LIBRARY_PATH', ''),
               GGML_BACKEND_PATH=str(build/'bin'))
    for key in list(env):
        if key.startswith(('ROCPROF', 'ROCTRACER')) or key == 'LD_PRELOAD' or (
                key.startswith('DSV4F_') and any(x in key for x in ('PROFILE', 'TIMELINE', 'TRACE'))):
            env.pop(key)
    hashes = {name: hashlib.sha256((build/'bin'/name).read_bytes()).hexdigest()
              for name in ('dsv4f-server', 'libggml-hip.so', 'libllama.so')}
    report = dict(date_utc=datetime.now(timezone.utc).isoformat(), prompt_file='PROMPT_TEST2.md',
                  prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(), batch_io=args.batch_io,
                  ram_budget_mib=163840, mtp=0, binary_sha256=hashes, runs=[])
    state = None
    with (out/'launcher.log').open('w') as log:
        proc = subprocess.Popen(['systemd-run', '--user', '--scope', '--quiet',
                                 '-p', 'MemorySwapMax=0', '-p', 'MemoryMax=160G', './run-server.sh'],
                                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            print('Loading batch_io='+args.batch_io, flush=True)
            for _ in range(600):
                if proc.poll() is not None:
                    raise RuntimeError('Server exited during load; inspect launcher.log')
                try:
                    candidate = json.loads(manifest.read_text())
                    assert candidate['url'] == 'http://127.0.0.1:18094'
                    options = candidate['engine_options']
                    for key, value in [('DSV4F_RAM_BUDGET_MIB', '163840'), ('DSV4F_MTP', '0'),
                                       ('DSV4F_PREFILL_GROUP_LEASE', '1'), ('DSV4F_PREFILL_BATCH_IO', args.batch_io)]:
                        assert options[key] == value
                    assert Path(f'/proc/{candidate["pid"]}/exe').resolve() == build/'bin/dsv4f-server'
                    with urlopen(candidate['url']+'/health', timeout=1) as response:
                        assert json.load(response)['status'] == 'ok'
                    state = candidate
                    break
                except (OSError, ValueError, AssertionError):
                    time.sleep(1)
            else:
                raise RuntimeError('Server load timeout')
            report['state'] = {k: v for k, v in state.items() if k != 'command'}

            def snapshot():
                memory = {line.split(':')[0]: int(line.split()[1])*1024
                          for line in Path(f'/proc/{state["pid"]}/status').read_text().splitlines()
                          if line.startswith(('VmRSS:', 'VmSwap:', 'VmHWM:'))}
                io = {line.split(':')[0]: int(line.split()[1])
                      for line in Path(f'/proc/{state["pid"]}/io').read_text().splitlines()}
                assert memory['VmSwap'] == 0
                return {'memory': memory, 'io': io}

            for index in range(args.runs):
                before = snapshot()
                result = generate(state['url'], 1, prompt, cache_prompt=False)
                after = snapshot()
                assert result['timings']['cache_n'] == 0
                record = dict(index=index+1, timings=result['timings'], wall_seconds=result['wall_seconds'],
                              usage=result['usage'], content=result['content'], before=before, after=after,
                              read_bytes=after['io']['read_bytes']-before['io']['read_bytes'])
                report['runs'].append(record)
                (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
                print('run', index+1, 'tokens', result['timings']['prompt_n'],
                      'prefill_tps', round(result['timings']['prompt_per_second'], 3),
                      'read_gib', round(record['read_bytes']/2**30, 3),
                      'rss_gib', round(after['memory']['VmRSS']/2**30, 3), flush=True)
            values = [r['timings']['prompt_per_second'] for r in report['runs']]
            report['summary'] = dict(cold_prefill_tps=values[0],
                                     warm_median_prefill_tps=statistics.median(values[1:]))
            (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
            print(report['summary'], flush=True)
        finally:
            if state:
                try:
                    if Path(f'/proc/{state["pid"]}/exe').resolve() == build/'bin/dsv4f-server':
                        os.kill(state['pid'], signal.SIGINT)
                except OSError:
                    pass
            elif proc.poll() is None:
                os.killpg(proc.pid, signal.SIGINT)
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=10)
            if state:
                shutil.copyfile(state['log'], out/'server.log')
    print('Owned server stopped', flush=True)


if __name__ == '__main__':
    main()
