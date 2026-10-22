#!/usr/bin/env python3
"""Unprofiled fresh-server comparison for V4.1 decode optimizations."""
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
from benchmark import PROMPT, generate

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--build-dir', type=Path, default=ROOT/'build')
    p.add_argument('--history', choices=('0', '1'), default='1')
    p.add_argument('--compact', choices=('0', '1'), default='1')
    p.add_argument('--overlap', choices=('0', '1'), default='1')
    args = p.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    build = args.build_dir.resolve()
    manifest = ROOT/'state/server.json'
    if manifest.exists():
        raise RuntimeError('Existing owned server; refusing to replace it')
    env = dict(os.environ, HOST='127.0.0.1', PORT='18091', BUILD_DIR=str(build),
               DSV4F_MTP='0', DSV4F_V41_HISTORY_GPU=args.history,
               DSV4F_V41_COMPACT_KV=args.compact, DSV4F_V41_MOE_OVERLAP=args.overlap,
               LD_LIBRARY_PATH=str(build/'bin')+':'+os.environ.get('LD_LIBRARY_PATH', ''),
               GGML_BACKEND_PATH=str(build/'bin'), DSV4F_SCHED_SUMMARY='1')
    for key in list(env):
        if (key.startswith(('ROCPROF', 'ROCTRACER')) or key == 'LD_PRELOAD' or
                (key.startswith('DSV4F_') and ('PROFILE' in key or 'TIMELINE' in key))):
            env.pop(key)
    launch = ['systemd-run', '--user', '--scope', '--quiet', '-p', 'MemorySwapMax=0',
              '-p', 'MemoryMax=184G', './run-server.sh']
    report, state = {'requests': []}, None
    with (out/'launcher.log').open('w') as log:
        proc = subprocess.Popen(launch, cwd=ROOT, env=env, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
        try:
            print('Loading', build, 'MTP=0; no tracing', flush=True)
            for _ in range(480):
                if proc.poll() is not None:
                    raise RuntimeError(f'Server exited {proc.returncode}; see launcher.log')
                try:
                    s = json.loads(manifest.read_text())
                    assert s['url'] == 'http://127.0.0.1:18091'
                    assert s['engine_options']['DSV4F_MTP'] == '0'
                    assert Path(f'/proc/{s["pid"]}/exe').resolve() == build/'bin/dsv4f-server'
                    with urlopen(s['url']+'/health', timeout=1) as response:
                        assert json.load(response)['status'] == 'ok'
                    state = s
                    break
                except (OSError, ValueError, AssertionError):
                    time.sleep(1)
            else:
                raise RuntimeError('Load timeout')
            report['state'] = state
            maps = Path(f'/proc/{state["pid"]}/maps').read_text()
            (out/'process-maps.txt').write_text(maps)
            for line in maps.splitlines():
                if any(name in line for name in ('/libllama', '/libggml')):
                    assert str(build/'bin') in line, line
            print('Ready; loaded libraries verified', flush=True)
            long_prompt = (ROOT/'PROMPT_TEST.md').read_text()
            cases = [('short-warmup', 128, PROMPT, False)]
            cases += [(f'short-{i}', 128, PROMPT, True) for i in range(1, 4)]
            cases += [('long-warmup', 32, long_prompt, False)]
            cases += [(f'long-{i}', 128, long_prompt, True) for i in range(1, 4)]
            for label, tokens, prompt, cached in cases:
                print(label, flush=True)
                result = generate(state['url'], tokens, prompt, cache_prompt=cached)
                result.update(label=label, cache_prompt=cached,
                              prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
                result['memory'] = {line.split(':')[0]: int(line.split()[1])*1024
                                    for line in Path(f'/proc/{state["pid"]}/status').read_text().splitlines()
                                    if line.startswith(('VmRSS:', 'VmSwap:'))}
                assert result['memory']['VmSwap'] == 0
                report['requests'].append(result)
                (out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
                print(label, result['timings'], flush=True)
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
                if state and Path(f'/proc/{state["pid"]}').exists():
                    os.kill(state['pid'], signal.SIGKILL)
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            if state:
                shutil.copyfile(state['log'], out/'server.log')
    print('Owned server stopped', flush=True)


if __name__ == '__main__':
    main()
