"""Exercise prefill batching, checkpoint reuse and GPU workspace resizing."""
import contextlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import tempfile
import time
from urllib.request import Request, urlopen

server, model = (Path(arg).resolve() for arg in sys.argv[1:3])
gpu = '--gpu' in sys.argv[3:]
environment = {key: value for key, value in os.environ.items()
               if not key.startswith(('DSV4F_', 'GGML_', 'MTP_'))}
environment.update(DSV4F_MTP='0', DSV4F_RAM_BUDGET_MIB='0', DSV4F_KV_RESIDENT='0',
                   DSV4F_ADAPTIVE_ROUTING='0', DSV4F_CPU_THREADS='4',
                   DSV4F_PHASE_WORKSPACE='1' if gpu else '0',
                   DSV4F_GPU_SERVICE='1' if gpu else '0', DSV4F_WHOLE_EXPERT='1',
                   DSV4F_GPU_WAIT='callback', DSV4F_PREFILL_STREAM='1',
                   DSV4F_PREFILL_GROUP='2', DSV4F_EXPERT_CACHE_MIB='4',
                   GGML_CUDA_MOE_CACHE='0', DSV4F_VRAM_HEADROOM_MIB='64',
                   DSV4F_VRAM_GUARD_MIB='64', DSV4F_ENGRAM_DIR=str(model.with_suffix('.engram')))
if gpu:
    devices = subprocess.run([str(server), '--list-devices'], env=environment,
                             capture_output=True, timeout=20)
    if not re.search(rb'(CUDA|ROCm)\d+:', devices.stdout + devices.stderr):
        print('SKIP: HIP/CUDA device unavailable')
        sys.exit(77)

with tempfile.TemporaryDirectory(prefix='dsv4f-workspace-server-') as temporary:
    directory = Path(temporary)
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    url = f'http://127.0.0.1:{port}'
    log_path = directory / 'server.log'
    with log_path.open('w') as log:
        process = subprocess.Popen([
            str(server), '--model', str(model), '--ctx-size', '4096' if gpu else '256',
            '--override-kv', 'deepseek41.max_position_embeddings=int:' + ('4096' if gpu else '256'),
            '--parallel', '1', '--n-gpu-layers', '999' if gpu else '0', '--fit', 'off',
            '--load-mode', 'none', '--cpu-moe', '--no-repack', '--threads', '4', '--threads-batch', '4',
            '--batch-size', '2048' if gpu else '64', '--ubatch-size', '2048' if gpu else '64',
            '--flash-attn', 'on', '--cache-type-k', 'q8_0', '--cache-type-v', 'q8_0',
            '--ctx-checkpoints', '32', '--checkpoint-min-step', '0', '--cache-ram', '0',
            '--host', '127.0.0.1', '--port', str(port), '-lv', '5'],
            cwd=directory, env=environment, stdout=log, stderr=log)
        try:
            for _ in range(400):
                if process.poll() is not None:
                    raise RuntimeError('server exited during startup')
                try:
                    with urlopen(url + '/health', timeout=.2) as response:
                        if response.status == 200:
                            break
                except OSError:
                    time.sleep(.05)
            else:
                raise TimeoutError('server startup')

            def complete(tokens, cache=False):
                body = dict(prompt=tokens, n_predict=2, ignore_eos=True, temperature=0,
                            cache_prompt=cache)
                with urlopen(Request(url + '/completion', json.dumps(body).encode(),
                                     {'Content-Type': 'application/json'}), timeout=60) as response:
                    result = json.load(response)
                assert result['tokens_predicted'] == 2, result
                return result

            if gpu:
                for length in (65, 1505, 65):
                    complete([2 + i % 8 for i in range(length)])
                phases = []
                for line in log_path.read_text(errors='replace').splitlines():
                    if '[dsv4f-workspace] phase=prefill' in line:
                        phases.append(int(re.search(r'reserve_tokens=(\d+)', line)[1]))
                assert phases == [1024, 2048, 1024], phases
            else:
                complete([2 + i % 8 for i in range(100)])
                checkpoints = [int(value) for value in re.findall(
                    r'created context checkpoint[^\n]*n_tokens = (\d+)',
                    log_path.read_text(errors='replace'))]
                if checkpoints:
                    assert 64 in checkpoints and 96 in checkpoints, checkpoints
                else:
                    # V4.1 can rewind its KV history directly and needs no snapshots.
                    text = log_path.read_text(errors='replace')
                    assert re.search(r'cached n_tokens = 64,', text), 'missing natural batch boundary'
                assert 36 not in checkpoints, 'prompt was split to leave one ubatch'

            length = 65 if gpu else 100
            followup = complete([2 + i % 8 for i in range(length)] + [12, 13], True)
            assert followup['timings']['cache_n'] > 0, 'checkpoint/prompt cache was not reused'
            print('PASS: ' + ('GPU 1024 -> 2048 -> 1024 workspaces' if gpu else
                             'natural batch boundary and final-four-token checkpoints') + ', cached follow-up')
        except Exception:
            print('\n'.join(log_path.read_text(errors='replace').splitlines()[-120:]), file=sys.stderr)
            raise
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
