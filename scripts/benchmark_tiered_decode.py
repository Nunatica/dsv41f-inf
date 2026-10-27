#!/usr/bin/env python3
"""Measure decode and physical reads with the 2.6bpw model, 128 GiB, MTP off."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import statistics
import subprocess
import time
from urllib.request import Request, urlopen

from benchmark import PROMPT, api_headers, generate

ROOT = Path(__file__).resolve().parents[1]


def snapshot(state):
    pid = state['pid']
    memory = {line.split(':')[0]: int(line.split()[1])*1024
              for line in Path(f'/proc/{pid}/status').read_text().splitlines()
              if line.startswith(('VmRSS:', 'VmSwap:', 'VmHWM:'))}
    io = {k: int(v) for k, v in
          (line.split(':') for line in Path(f'/proc/{pid}/io').read_text().splitlines())}
    group = Path(state['cgroup']['path'])
    cgroup = {k: int((group/k).read_text()) for k in
              ('memory.max', 'memory.current', 'memory.peak', 'memory.swap.current')}
    events = {k: int(v) for k, v in
              (line.split() for line in (group/'memory.events').read_text().splitlines())}
    assert memory['VmSwap'] == cgroup['memory.swap.current'] == events['oom_kill'] == 0
    assert cgroup['memory.max'] == 128*2**30
    assert cgroup['memory.peak'] <= cgroup['memory.max']
    # FUSE/remote mounts may charge physical reads to their daemon, leaving
    # /proc/<engine>/io read_bytes at zero despite real O_DIRECT payload reads.
    # Engine samples are cumulative lower bounds, logged every 4096 reads.
    sampled_weights = None
    for line in reversed(Path(state['log']).read_text().splitlines()):
        if line.startswith('[dsv4f-ram] resident='):
            sampled_weights = {k: float(v) for k, v in re.findall(r'(\w+)=([\d.]+)', line)}
            break
    return dict(memory=memory, io=io, cgroup=cgroup, events=events,
                sampled_weights=sampled_weights)


def decode(state, tokens):
    body = dict(model='dsv4f', messages=[dict(role='user', content=PROMPT)],
                max_tokens=tokens, temperature=0, seed=42, cache_prompt=False,
                stream=True, stream_options=dict(include_usage=True), timings_per_token=True,
                chat_template_kwargs=dict(enable_thinking=False), reasoning_budget=0)
    result = dict(content='', reasoning_content='', before=snapshot(state))
    start = time.monotonic()
    with urlopen(Request(state['url']+'/v1/chat/completions', headers=api_headers(),
                         data=json.dumps(body).encode()), timeout=1800) as response:
        for line in response:
            if not line.startswith(b'data:'):
                continue
            raw = line[5:].strip()
            if raw == b'[DONE]':
                break
            item = json.loads(raw)
            if item.get('error'):
                raise RuntimeError(item['error'])
            if item.get('timings'):
                result['timings'] = item['timings']
            for choice in item.get('choices', []):
                delta = choice.get('delta', {})
                for key in ('content', 'reasoning_content'):
                    result[key] += delta.get(key) or ''
                if delta.get('content') and 'decode_start' not in result:
                    result['decode_start'] = snapshot(state)
            if item.get('usage'):
                result['usage'] = item['usage']
    result.update(after=snapshot(state), wall_seconds=time.monotonic()-start)
    assert not result['reasoning_content']
    assert result['timings']['cache_n'] == 0
    result['decode_read_bytes'] = result['after']['io']['read_bytes']-result['decode_start']['io']['read_bytes']
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--tokens', type=int, default=512)
    parser.add_argument('--long-prefill', action='store_true',
                        help='populate caches using all of PROMPT_TEST.md before decode')
    args = parser.parse_args()
    if args.runs < 1 or args.tokens < 2:
        parser.error('runs >= 1 and tokens >= 2 required')
    manifest = ROOT/'state/server.json'
    if manifest.exists():
        raise RuntimeError('An owned server already exists')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    build = ROOT/'build'
    env = dict(os.environ, MODEL=str(ROOT/'models/DeepSeek-V4.1-Flash-1017-dsx-2.6bit.gguf'),
               HOST='127.0.0.1', PORT='18095', BUILD_DIR=str(build),
               DSV4F_RAM_BUDGET_MIB='131072', DSV4F_MTP='0', DSV4F_ADAPTIVE_ROUTING='0',
               DSV4F_RAM_PRELOAD='0', DSV4F_EXPERT_LOOKAHEAD='0', DSV4F_RAM_LOOKAHEAD='0',
               LD_LIBRARY_PATH=str(build/'bin')+':'+os.environ.get('LD_LIBRARY_PATH', ''),
               GGML_BACKEND_PATH=str(build/'bin'))
    for key in list(env):
        if key.startswith(('ROCPROF', 'ROCTRACER')) or key == 'LD_PRELOAD' or (
                key.startswith('DSV4F_') and any(x in key for x in ('PROFILE', 'TIMELINE', 'TRACE'))):
            env.pop(key)
    report = dict(mtp=0, ram_budget_mib=131072, prompt=PROMPT, tokens=args.tokens,
                  binary_sha256={name: hashlib.sha256((build/'bin'/name).read_bytes()).hexdigest()
                                 for name in ('dsv4f-server', 'libggml-hip.so', 'libllama.so')}, runs=[])
    state = None
    with (out/'launcher.log').open('w') as log:
        proc = subprocess.Popen(['./run-server.sh'], cwd=ROOT, env=env,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            print('Loading 2.6bpw / 128 GiB / MTP=0', flush=True)
            for _ in range(600):
                if proc.poll() is not None:
                    raise RuntimeError('Server exited; inspect launcher.log')
                try:
                    candidate = json.loads(manifest.read_text())
                    assert candidate['url'] == 'http://127.0.0.1:18095'
                    assert candidate['engine_options']['DSV4F_RAM_BUDGET_MIB'] == '131072'
                    assert candidate['engine_options']['DSV4F_MTP'] == '0'
                    with urlopen(candidate['url']+'/health', timeout=1) as response:
                        assert json.load(response)['status'] == 'ok'
                    state = candidate
                    break
                except (OSError, ValueError, AssertionError):
                    time.sleep(1)
            else:
                raise RuntimeError('Server load timeout')
            report['state'] = {k: v for k, v in state.items() if k != 'command'}
            if args.long_prefill:
                print('Long-prompt prefill to populate both memory tiers', flush=True)
                report['long_prefill'] = generate(state['url'], 1, (ROOT/'PROMPT_TEST.md').read_text())
            for index in range(args.runs):
                result = decode(state, args.tokens)
                report['runs'].append(result)
                (out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
                print('run', index+1, 'decode_tps', round(result['timings']['predicted_per_second'], 3),
                      'decode_read_gib', round(result['decode_read_bytes']/2**30, 3),
                      'peak_gib', round(result['after']['cgroup']['memory.peak']/2**30, 3), flush=True)
            report['median_decode_tps'] = statistics.median(r['timings']['predicted_per_second'] for r in report['runs'])
            report['physical_reads_charged_to_engine'] = any(r['decode_read_bytes'] for r in report['runs'])
            (out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
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
                proc.wait(timeout=45)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=15)
            if state:
                shutil.copyfile(state['log'], out/'server.log')
    print('Owned server stopped', flush=True)


if __name__ == '__main__':
    main()
