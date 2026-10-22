#!/usr/bin/env python3
"""Profile warmed short/long decode with CPU ROCTx ranges and GPU/HIP traces."""
import argparse
import json
import os
from pathlib import Path
import shlex
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
    args = p.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    manifest = ROOT/'state/server.json'
    if manifest.exists():
        raise RuntimeError('Existing owned server; refusing to replace it')
    control = out/'collect'
    control.write_text('0')
    wrapper = out/'wrapper/bin/dsv4f-server'
    wrapper.parent.mkdir(parents=True)
    cmd = ['/opt/rocm/bin/rocprofv3', '--selected-regions', '--hip-runtime-trace',
           '--kernel-trace', '--memory-copy-trace', '--marker-trace',
           '--output-format', 'csv', '--output-directory', str(out/'trace'),
           '--output-file', 'decode', '--preload', '/opt/rocm/lib/librocprofiler-sdk-roctx.so',
           '--', str(ROOT/'build/bin/dsv4f-server')]
    wrapper.write_text('#!/bin/sh\nexec '+shlex.join(cmd)+' "$@"\n')
    wrapper.chmod(0o755)
    env = dict(os.environ, HOST='127.0.0.1', PORT='18090', DSV4F_MTP='0',
               BUILD_DIR=str(out/'wrapper'), DSV4F_TIMELINE='1', DSV4F_TIMELINE_CONTROL=str(control),
               DSV4F_INPUT_PROFILE='1', DSV4F_ENGRAM_PROFILE='1', DSV4F_SERVICE_PROFILE='1')
    launch = ['systemd-run', '--user', '--scope', '--quiet', '-p', 'MemorySwapMax=0',
              '-p', 'MemoryMax=184G', './run-server.sh']
    report, state = {'requests': [], 'trace_command': cmd}, None
    with (out/'launcher.log').open('w') as log:
        proc = subprocess.Popen(launch, cwd=ROOT, env=env, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
        try:
            print('Loading owned MTP-off server; trace collection paused', flush=True)
            for _ in range(480):
                if proc.poll() is not None:
                    raise RuntimeError(f'Server exited {proc.returncode}; see launcher.log')
                try:
                    s = json.loads(manifest.read_text())
                    assert s['url'] == 'http://127.0.0.1:18090'
                    assert s['engine_options']['DSV4F_MTP'] == '0'
                    assert Path(f'/proc/{s["pid"]}/exe').resolve() == ROOT/'build/bin/dsv4f-server'
                    with urlopen(s['url']+'/health', timeout=1) as response:
                        assert json.load(response)['status'] == 'ok'
                    state = s
                    break
                except (OSError, ValueError, AssertionError):
                    time.sleep(1)
            else:
                raise RuntimeError('Load timeout')
            report['state'] = state
            print('Server ready', flush=True)
            long_prompt = (ROOT/'PROMPT_TEST.md').read_text()
            cases = [('short-warmup', 128, PROMPT, False, False),
                     ('short-before', 128, PROMPT, True, False),
                     ('short-profile', 128, PROMPT, True, True),
                     ('short-after', 128, PROMPT, True, False),
                     ('long-warmup', 32, long_prompt, False, False),
                     ('long-before', 128, long_prompt, True, False),
                     ('long-profile', 128, long_prompt, True, True),
                     ('long-after', 128, long_prompt, True, False)]
            for label, tokens, prompt, cached, profiled in cases:
                print(label, 'collecting' if profiled else 'collection paused', flush=True)
                control.write_text('1' if profiled else '0')
                offset = Path(state['log']).stat().st_size
                result = generate(state['url'], tokens, prompt, cache_prompt=cached)
                with Path(state['log']).open('rb') as f:
                    f.seek(offset)
                    (out/f'{label}.log').write_bytes(f.read())
                result.update(label=label, cache_prompt=cached, profiled=profiled)
                memory = {line.split(':')[0]: int(line.split()[1])*1024
                          for line in Path(f'/proc/{state["pid"]}/status').read_text().splitlines()
                          if line.startswith(('VmRSS:', 'VmSwap:'))}
                assert memory['VmSwap'] == 0
                result['memory'] = memory
                report['requests'].append(result)
                (out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
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
                os.killpg(proc.pid, signal.SIGINT)
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
            if state:
                shutil.copyfile(state['log'], out/'server.log')
    print('Owned server stopped; profiler output flushed', flush=True)


if __name__ == '__main__':
    main()
