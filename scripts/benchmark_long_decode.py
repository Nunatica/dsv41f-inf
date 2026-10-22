#!/usr/bin/env python3
"""Compare sustained short-prompt decode, with optional middle-of-output tracing.

Uses the UI sampling/reasoning settings, native SSE timings, natural EOS, and
the AGENTS.md prompt. Each request starts without prompt reuse; OS caches are
left intact. The driver stops only its own server.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import time
from urllib.request import Request, urlopen

from benchmark import PROMPT, api_headers
from diagnose_engram_io import snapshot, summarize

ROOT = Path(__file__).resolve().parents[1]


def generate(state, seed, control=None, *, ignore_eos=False):
    body = dict(model='dsv4f', messages=[dict(role='user', content=PROMPT)],
                stream=True, stream_options=dict(include_usage=True),
                timings_per_token=True, temperature=.6, seed=seed, top_p=1,
                top_k=40, min_p=.05, max_tokens=1200, cache_prompt=False,
                reasoning_effort='none', reasoning_format='deepseek',
                reasoning_budget_tokens=0,
                chat_template_kwargs=dict(enable_thinking=False, thinking=False,
                                          reasoning_effort='none'))
    if ignore_eos:
        body['ignore_eos'] = True
    start = time.monotonic()
    result = dict(request=body, points=[], content='', reasoning_content='',
                  trace_switches=[], before=snapshot(state['pid']))
    collecting = False
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
            for choice in item.get('choices', []):
                delta = choice.get('delta', {})
                for key in ('content', 'reasoning_content'):
                    result[key] += delta.get(key) or ''
                if choice.get('finish_reason'):
                    result['finish_reason'] = choice['finish_reason']
            if item.get('usage'):
                result['usage'] = item['usage']
            if item.get('timings'):
                timing = item['timings']
                result['timings'] = timing
                result['points'].append(dict(wall_seconds=time.monotonic()-start, **timing))
                if control:
                    n = timing.get('predicted_n', 0)
                    wanted = 256 <= n < 512
                    if wanted != collecting:
                        control.write_text('1' if wanted else '0')
                        result['trace_switches'].append(dict(predicted_n=n, collecting=wanted))
                        collecting = wanted
    if control:
        control.write_text('0')
    result.update(wall_seconds=time.monotonic()-start, after=snapshot(state['pid']))
    assert result['after']['memory']['VmSwap'] == 0
    assert not result['reasoning_content'], 'Unexpected reasoning'
    points = result['points']
    result['segments'] = []
    for lo, hi in ((1, 128), (128, 256), (256, 512), (512, 768), (768, 1200)):
        left, right = [min(points, key=lambda p: abs(p.get('predicted_n', 0)-n)) for n in (lo, hi)]
        dn = right['predicted_n']-left['predicted_n']
        dt = right['predicted_ms']-left['predicted_ms']
        if dn > 0 and dt > 0:
            result['segments'].append(dict(start_n=left['predicted_n'], end_n=right['predicted_n'],
                                           tokens_per_second=1000*dn/dt))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--build-dir', type=Path, default=ROOT/'build')
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--trace-mode', choices=('full', 'host'), default='full',
                        help='With --profile: full GPU/API traces, or lighter HIP/ROCTx-only overhead control')
    parser.add_argument('--pipeline', choices=('0', '1'), default='1')
    parser.add_argument('--ptq-dot', choices=('0', '1'), default='1')
    parser.add_argument('--cache-limits', type=int, nargs=3, metavar=('Q3_MIB', 'Q2_MIB', 'PTQ_MIB'),
                        help='Override launcher Q3/Q2/PTQ ceilings; default is an 8:1:5 split of the aggregate budget')
    parser.add_argument('--q8-tile', choices=('0','1'), default='1')
    parser.add_argument('--shared-fused', choices=('0','1'), default='1')
    parser.add_argument('--route-fused', choices=('0','1'), default='1')
    parser.add_argument('--engram-overlap', choices=('0','1','2'), default='0',
                        help='0 synchronous; 1 both tables async; 2 first synchronous, later async')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123])
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    build = args.build_dir.resolve()
    manifest = ROOT/'state/server.json'
    if manifest.exists():
        raise RuntimeError('Existing server; refusing to replace it')
    env = dict(os.environ)
    cache_keys = ('DSV4F_V41_CACHE_Q3_MIB', 'DSV4F_V41_CACHE_Q2_MIB', 'DSV4F_V41_CACHE_PTQ_MIB')
    for key in cache_keys:
        env.pop(key, None)
    for key in list(env):
        if key.startswith(('ROCPROF', 'ROCTRACER')) or key == 'LD_PRELOAD' or (
                key.startswith('DSV4F_') and ('PROFILE' in key or 'TIMELINE' in key)):
            env.pop(key)
    flag = '0' if args.baseline else '1'
    env.update(HOST='127.0.0.1', PORT='18091', BUILD_DIR=str(build), DSV4F_MTP='0',
               DSV4F_V41_HISTORY_GPU=flag, DSV4F_V41_COMPACT_KV=flag,
               DSV4F_V41_MOE_OVERLAP=flag, DSV4F_SCHED_SUMMARY='1',
               DSV4F_V41_MOE_PIPELINE=args.pipeline, DSV4F_V41_PTQ_DOT=args.ptq_dot,
               DSV4F_V41_SHARED_FUSED=args.shared_fused, GGML_HIP_Q8_TILE=args.q8_tile,
               DSV4F_V41_ROUTE_FUSED=args.route_fused, DSV4F_ENGRAM_OVERLAP=args.engram_overlap,
               LD_LIBRARY_PATH=str(build/'bin')+':'+os.environ.get('LD_LIBRARY_PATH', ''),
               GGML_BACKEND_PATH=str(build/'bin'))
    if args.cache_limits:
        if any(n<0 for n in args.cache_limits):
            raise ValueError('Cache limits must be non-negative')
        env.update(zip(cache_keys, map(str, args.cache_limits)))
    report, state, control = dict(requests=[]), None, None
    if args.profile:
        assert not args.baseline
        control = out/'collect'
        control.write_text('0')
        wrapper = out/'wrapper/bin/dsv4f-server'
        wrapper.parent.mkdir(parents=True)
        trace_flags = ['--hip-runtime-trace', '--marker-trace']
        if args.trace_mode == 'full':
            trace_flags += ['--kernel-trace', '--memory-copy-trace']
        cmd = ['/opt/rocm/bin/rocprofv3', '--selected-regions', *trace_flags,
               '--output-format', 'csv', '--output-directory', str(out/'trace'),
               '--output-file', 'decode', '--preload', '/opt/rocm/lib/librocprofiler-sdk-roctx.so',
               '--', str(build/'bin/dsv4f-server')]
        wrapper.write_text('#!/bin/sh\nexec '+shlex.join(cmd)+' "$@"\n')
        wrapper.chmod(0o755)
        env.update(BUILD_DIR=str(out/'wrapper'), DSV4F_TIMELINE='1',
                   DSV4F_TIMELINE_CONTROL=str(control), DSV4F_INPUT_PROFILE='1',
                   DSV4F_ENGRAM_PROFILE='1', DSV4F_SERVICE_PROFILE='1')
        report['trace_command'] = cmd
        report['trace_mode'] = args.trace_mode
    launch = ['systemd-run', '--user', '--scope', '--quiet', '-p', 'MemorySwapMax=0',
              '-p', 'MemoryMax=184G', './run-server.sh']
    with (out/'launcher.log').open('w') as log:
        proc = subprocess.Popen(launch, cwd=ROOT, env=env, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
        try:
            print('Loading', build, 'profile=', args.profile, flush=True)
            for _ in range(480):
                if proc.poll() is not None:
                    raise RuntimeError(f'Server exited {proc.returncode}; see launcher.log')
                try:
                    candidate = json.loads(manifest.read_text())
                    assert candidate['url'] == 'http://127.0.0.1:18091'
                    assert candidate['engine_options']['DSV4F_MTP'] == '0'
                    assert Path(f'/proc/{candidate["pid"]}/exe').resolve() == build/'bin/dsv4f-server'
                    with urlopen(candidate['url']+'/health', timeout=1) as response:
                        assert json.load(response)['status'] == 'ok'
                    state = candidate
                    break
                except (OSError, ValueError, AssertionError):
                    time.sleep(1)
            else:
                raise RuntimeError('Load timeout')
            maps = Path(f'/proc/{state["pid"]}/maps').read_text()
            (out/'process-maps.txt').write_text(maps)
            for line in maps.splitlines():
                if any(name in line for name in ('/libllama', '/libggml')):
                    assert str(build/'bin') in line, line
            report['state'] = state
            print('Ready; libraries verified', flush=True)
            for seed in args.seeds:
                label = f'ui-seed-{seed}'
                print(label, flush=True)
                offset = Path(state['log']).stat().st_size
                result = generate(state, seed, control)
                with Path(state['log']).open('rb') as f:
                    f.seek(offset)
                    chunk = f.read()
                (out/f'{label}.log').write_bytes(chunk)
                result['label'] = label
                report['requests'].append(result)
                (out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
                if args.profile:
                    result['engram_summary'] = summarize(chunk.decode(errors='replace'), result['timings'])
                    (out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
                print(label, result['timings'], result['segments'], flush=True)
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
                os.killpg(proc.pid, signal.SIGINT)
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
            if state:
                shutil.copyfile(state['log'], out/'server.log')
    print('Owned server stopped', flush=True)


if __name__ == '__main__':
    main()
