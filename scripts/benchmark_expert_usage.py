#!/usr/bin/env python3
"""Capture global expert routes for both AGENTS.md prompts (MTP off).

Use --no-trace in a second run for native timings without scheduler readbacks.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
from urllib.request import urlopen
from benchmark import generate

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output', type=Path)
    p.add_argument('--tokens', type=int, default=1024)
    p.add_argument('--no-trace', action='store_true')
    args = p.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    manifest = ROOT/'state/server.json'
    if manifest.exists():
        raise RuntimeError('An owned server already exists')
    env = dict(os.environ, HOST='127.0.0.1', PORT='18091', DSV4F_MTP='0',
               DSV4F_EXPERT_LOOKAHEAD='0', LD_LIBRARY_PATH=str(ROOT/'build/bin'),
               GGML_BACKEND_PATH=str(ROOT/'build/bin'))
    for key in list(env):
        if key.startswith(('ROCPROF', 'ROCTRACER')) or key == 'LD_PRELOAD' or (
                key.startswith('DSV4F_') and any(x in key for x in ('TRACE', 'PROFILE', 'TIMELINE'))):
            env.pop(key)
    trace = out/'routes.trace'
    if not args.no_trace:
        env['DSV4F_EXPERT_TRACE'] = str(trace)
    report = {'mtp': False, 'instrumented': not args.no_trace, 'max_tokens': args.tokens,
              'records': [], 'trace_format': 'R layer nt nk [token-major global top-k IDs]'}
    state = None
    with (out/'launcher.log').open('w') as log:
        proc = subprocess.Popen(['systemd-run', '--user', '--scope', '--quiet',
                                 '-p', 'MemorySwapMax=0', '-p', 'MemoryMax=184G',
                                 './run-server.sh'], cwd=ROOT, env=env, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
        try:
            print('Loading server', flush=True)
            for _ in range(900):
                if proc.poll() is not None:
                    raise RuntimeError('Server exited; inspect launcher.log')
                try:
                    candidate = json.loads(manifest.read_text())
                    assert candidate['url'] == 'http://127.0.0.1:18091'
                    assert candidate['engine_options']['DSV4F_MTP'] == '0'
                    assert Path(f'/proc/{candidate["pid"]}/exe').resolve() == ROOT/'build/bin/dsv4f-server'
                    with urlopen(candidate['url']+'/health', timeout=1) as response:
                        assert json.load(response)['status'] == 'ok'
                    state = candidate
                    break
                except (OSError, ValueError, AssertionError):
                    time.sleep(1)
            else:
                raise RuntimeError('Server load timeout')
            # Do not copy API credentials from the command into the report.
            report['state'] = {k: v for k, v in state.items() if k != 'command'}
            print('Ready; warmup 32 tokens', flush=True)
            generate(state['url'], 32)
            for label, prompt in [('short', 'Explain story of the romance of three kingdoms.'),
                                  ('long', (ROOT/'PROMPT_TEST.md').read_text())]:
                offset = trace.stat().st_size if trace.exists() else 0
                print(f'{label}: prefill + up to {args.tokens} decode tokens', flush=True)
                result = generate(state['url'], args.tokens, prompt, cache_prompt=False)
                end = trace.stat().st_size if trace.exists() else 0
                memory = {line.split(':')[0]: int(line.split()[1])*1024
                          for line in Path(f'/proc/{state["pid"]}/status').read_text().splitlines()
                          if line.startswith(('VmRSS:', 'VmSwap:'))}
                assert memory['VmSwap'] == 0
                record = dict(label=label, prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                              trace_start=offset, trace_end=end, memory=memory, **result)
                report['records'].append(record)
                (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
                print(label, result['timings'], flush=True)
        finally:
            if state:
                try:
                    if Path(f'/proc/{state["pid"]}/exe').resolve() == ROOT/'build/bin/dsv4f-server':
                        os.kill(state['pid'], signal.SIGINT)
                except OSError:
                    pass
            elif proc.poll() is None:
                os.killpg(proc.pid, signal.SIGINT)
            try:
                proc.wait(timeout=45)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=30)
            if state:
                shutil.copyfile(state['log'], out/'server.log')
    print('Owned server stopped', flush=True)


if __name__ == '__main__':
    main()
