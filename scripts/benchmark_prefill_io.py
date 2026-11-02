#!/usr/bin/env python3
"""Compare cold/warm long-prompt prefill under a fixed RAM budget, MTP off."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import threading
import time
from urllib.request import urlopen

from benchmark import generate

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--batch-io', choices=['0', '1'], required=True)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--ram-budget-mib', type=int, default=163840)
    parser.add_argument('--scan-cache', choices=['0', '1'], default='0')
    parser.add_argument('--prompt-file', type=Path, default=ROOT/'PROMPT_TEST.md')
    parser.add_argument('--validation-tokens', type=int, default=0,
                        help='generate a short-prompt sample after all timed runs')
    parser.add_argument('--preload', choices=['0', '1'], default='0')
    parser.add_argument('--io-sort', choices=['0', '1'], default='0')
    parser.add_argument('--io-pipeline', choices=['0', '1'], default='0')
    parser.add_argument('--model', type=Path, default=ROOT/'models/DeepSeek-V4.1-Flash-1017-dsx-2.6bit.gguf')
    parser.add_argument('--block-device', help='optional whole-device counters, e.g. nvme0n1')
    parser.add_argument('--engram-prefill-io', choices=['0', '1'], default='0')
    parser.add_argument('--prefill-ubatch', choices=['1024', '2048', '4096', '6144', '8192', '10240', '12288'], default='1024')
    parser.add_argument('--engram-timers', action='store_true', help='lightweight per-batch read/wait timers')
    parser.add_argument('--sample-vram', action='store_true', help='sample AMD VRAM usage every 100 ms')
    args = parser.parse_args()
    if args.runs < 1:
        parser.error('--runs must be positive')
    if args.ram_budget_mib <= 4096:
        parser.error('--ram-budget-mib must exceed the 4096-MiB runtime reserve')
    manifest = ROOT/'state/server.json'
    if manifest.exists():
        raise RuntimeError('An owned server already exists')
    prompt = args.prompt_file.read_text(encoding='utf-8')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    build = ROOT/'build'
    env = dict(os.environ, HOST='127.0.0.1', PORT='18094', BUILD_DIR=str(build),
               MODEL=str(args.model.resolve()), DSV4F_RAM_PRELOAD=args.preload,
               DSV4F_RAM_RESERVE_MIB='1024', DSV4F_LOAD_PIPELINE_MIB='1024',
               DSV4F_ADAPTIVE_ROUTING='0', DSV4F_ADAPTIVE_PRUNING='0',
               DSV4F_PREFILL_IO_SORT=args.io_sort, DSV4F_PREFILL_IO_PIPELINE=args.io_pipeline,
               DSV4F_PREFILL_GROUP='16',
               DSV4F_ENGRAM_PREFILL_IO=args.engram_prefill_io, DSV4F_RAM_PREFILL_UBATCH=args.prefill_ubatch,
               DSV4F_RAM_BUDGET_MIB=str(args.ram_budget_mib), DSV4F_MTP='0', DSV4F_EXPERT_LOOKAHEAD='0',
               DSV4F_RAM_LOOKAHEAD='0', DSV4F_PREFILL_GROUP_LEASE='1',
               DSV4F_PREFILL_BATCH_IO=args.batch_io, DSV4F_CPU_EXPERT_LEASE='1',
               DSV4F_PREFILL_SCAN_CACHE=args.scan_cache, DSV4F_PREFILL_RAM_STATS='1',
               DSV4F_RAM_UNLOCK_IO='1', DSV4F_ENGRAM_BATCH_IO='1',
               LD_LIBRARY_PATH=str(build/'bin')+':'+os.environ.get('LD_LIBRARY_PATH', ''),
               GGML_BACKEND_PATH=str(build/'bin'))
    env.pop('DSV4F_RAM_CACHE_MIB', None)
    for key in list(env):
        if key.startswith(('ROCPROF', 'ROCTRACER')) or key == 'LD_PRELOAD' or (
                key.startswith('DSV4F_') and any(x in key for x in ('PROFILE', 'TIMELINE', 'TRACE'))):
            env.pop(key)
    if args.engram_timers:
        env.update(DSV4F_ENGRAM_ASYNC_PROFILE='1', DSV4F_INPUT_PROFILE='1')
    hashes = {name: hashlib.sha256((build/'bin'/name).read_bytes()).hexdigest()
              for name in ('dsv4f-server', 'libggml-hip.so', 'libllama.so')}
    report = dict(date_utc=datetime.now(timezone.utc).isoformat(), prompt_file=str(args.prompt_file),
                  prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(), batch_io=args.batch_io,
                  ram_budget_mib=args.ram_budget_mib, scan_cache=args.scan_cache,
                  preload=args.preload, io_sort=args.io_sort, io_pipeline=args.io_pipeline,
                  model=str(args.model.resolve()), model_bytes=args.model.stat().st_size,
                  block_device=args.block_device,
                  engram_prefill_io=args.engram_prefill_io, prefill_ubatch=int(args.prefill_ubatch),
                  engram_timers=args.engram_timers,
                  mtp=0, binary_sha256=hashes, runs=[])
    state = None
    with (out/'launcher.log').open('w') as log:
        proc = subprocess.Popen(['systemd-run', '--user', '--scope', '--quiet',
                                 '-p', 'MemorySwapMax=0', '-p', f'MemoryMax={args.ram_budget_mib*2**20}', './run-server.sh'],
                                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            print(f'Loading batch_io={args.batch_io} scan_cache={args.scan_cache} ram_mib={args.ram_budget_mib}', flush=True)
            for _ in range(600):
                if proc.poll() is not None:
                    raise RuntimeError('Server exited during load; inspect launcher.log')
                try:
                    candidate = json.loads(manifest.read_text())
                    assert candidate['url'] == 'http://127.0.0.1:18094'
                    options = candidate['engine_options']
                    for key, value in [('DSV4F_RAM_BUDGET_MIB', str(args.ram_budget_mib)), ('DSV4F_MTP', '0'),
                                       ('DSV4F_PREFILL_SCAN_CACHE', args.scan_cache),
                                       ('DSV4F_RAM_PRELOAD', args.preload), ('DSV4F_ADAPTIVE_ROUTING', '0'),
                                       ('DSV4F_ADAPTIVE_PRUNING', '0'),
                                       ('DSV4F_PREFILL_IO_SORT', args.io_sort),
                                       ('DSV4F_PREFILL_IO_PIPELINE', args.io_pipeline),
                                       ('DSV4F_ENGRAM_PREFILL_IO', args.engram_prefill_io),
                                       ('DSV4F_RAM_PREFILL_UBATCH', args.prefill_ubatch),
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
            assert state['format'] == 'dsx-2.6bit' if 'dsx-2.6bit' in str(args.model) else True
            if args.preload == '1':
                log_text = Path(state['log']).read_text()
                assert 'order=shard,offset O_DIRECT' in log_text
                completed = [line for line in log_text.splitlines() if line.startswith('[dsv4f-ram-preload] complete:')]
                assert len(completed) == 1 and 'stop=budget' in completed[0], 'preload did not fill cache budget'
                report['preload_log'] = completed[0]

            def snapshot():
                memory = {line.split(':')[0]: int(line.split()[1])*1024
                          for line in Path(f'/proc/{state["pid"]}/status').read_text().splitlines()
                          if line.startswith(('VmRSS:', 'VmSwap:', 'VmHWM:'))}
                io = {line.split(':')[0]: int(line.split()[1])
                      for line in Path(f'/proc/{state["pid"]}/io').read_text().splitlines()}
                assert memory['VmSwap'] == 0
                cgroot = Path(state['cgroup']['path'])
                cgroup = {k: int((cgroot/k).read_text().strip()) for k in
                          ('memory.current', 'memory.peak', 'memory.swap.current', 'memory.max')}
                cgroup['events'] = dict(line.split() for line in (cgroot/'memory.events').read_text().splitlines())
                assert cgroup['memory.max'] == args.ram_budget_mib*2**20
                assert cgroup['memory.peak'] <= cgroup['memory.max']
                assert cgroup['memory.swap.current'] == 0
                assert int(cgroup['events']['oom_kill']) == 0
                weights = dict(reads=0, read_bytes=0, evictions=0, transient_reads=0, transient_read_bytes=0,
                               io_requests=0, io_bytes=0, merged_planes=0)
                for line in reversed(Path(state['log']).read_text().splitlines()):
                    if line.startswith('[dsv4f-ram-prefill]'):
                        weights = {k: int(v) for k, v in (field.split('=', 1) for field in line.split()[1:]) if k != 'layer'}
                        break
                device = ([int(n) for n in Path('/sys/block', args.block_device, 'stat').read_text().split()]
                          if args.block_device else None)
                return {'memory': memory, 'io': io, 'cgroup': cgroup, 'weights': weights, 'device': device}

            if args.preload == '1':
                report['preload_snapshot'] = snapshot()
                weights = report['preload_snapshot']['weights']
                assert weights['resident_bytes'] > 100*2**30
                assert 0 <= weights['ceiling_bytes']-weights['resident_bytes'] < 16*2**20
                assert weights['transient_bytes'] == 0
            for index in range(args.runs):
                before = snapshot()
                log_offset = len(Path(state['log']).read_text())
                samples = []; stop_sampling = threading.Event(); sampler = None
                if args.sample_vram:
                    devices = list(Path('/sys/class/drm').glob('card[0-9]*/device/mem_info_vram_total'))
                    if not devices:
                        raise RuntimeError('AMD VRAM counters unavailable')
                    counter = max(devices, key=lambda p: int(p.read_text()))
                    total = int(counter.read_text())
                    def sample():
                        while not stop_sampling.is_set():
                            samples.append(dict(monotonic=time.monotonic(), used_bytes=int(counter.with_name('mem_info_vram_used').read_text()), total_bytes=total))
                            stop_sampling.wait(.1)
                    sampler = threading.Thread(target=sample, daemon=True); sampler.start()
                try:
                    try:
                        result = generate(state['url'], 1, prompt, cache_prompt=False)
                    finally:
                        stop_sampling.set()
                        if sampler: sampler.join()
                except BaseException as error:
                    report['error'] = str(error) or type(error).__name__
                    report['failure_vram_samples'] = samples
                    cgroot=Path(state['cgroup']['path'])
                    try:
                        report['failure_cgroup']={k:(cgroot/k).read_text() for k in ['memory.events','memory.peak','memory.current','memory.swap.current']}
                    except OSError as cg_error:
                        report['failure_cgroup_error']=str(cg_error)
                    try:
                        report['failure_snapshot'] = snapshot()
                    except (OSError, AssertionError) as snapshot_error:
                        report['failure_snapshot_error'] = str(snapshot_error)
                    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
                    raise
                after = snapshot()
                assert result['timings']['cache_n'] == 0
                record = dict(index=index+1, timings=result['timings'], wall_seconds=result['wall_seconds'],
                              usage=result['usage'], content=result['content'], before=before, after=after,
                              read_bytes=after['io']['read_bytes']-before['io']['read_bytes'])
                record['expert_read_bytes'] = after['weights']['read_bytes']-before['weights']['read_bytes']
                record['expert_evictions'] = after['weights']['evictions']-before['weights']['evictions']
                record['io_requests'] = after['weights']['io_requests']-before['weights']['io_requests']
                record['io_bytes'] = after['weights']['io_bytes']-before['weights']['io_bytes']
                record['merged_planes'] = after['weights']['merged_planes']-before['weights']['merged_planes']
                text = Path(state['log']).read_text()
                batches = [int(n) for n in re.findall(r'\[dsv4f-ram-prefill\] layer=\S+ tokens=(\d+)', text) if int(n)]
                assert batches and max(batches) == int(args.prefill_ubatch), 'requested ubatch was not used'
                record['observed_prefill_ubatches'] = sorted(set(batches))
                record['workspace_log'] = [line for line in text.splitlines() if '[dsv4f-workspace]' in line]
                timers = dict(engram_read_ms=0.0, engram_wait_ms=0.0, input_ms=0.0)
                for line in text[log_offset:].splitlines():
                    values = {k: float(v) for k,v in re.findall(r'(\w+)=([\d.]+)', line)}
                    if line.startswith('[dsv4f-engram-prefetch-read]'):
                        timers['engram_read_ms'] += values.get('read_us', 0)/1000
                    elif line.startswith('[dsv4f-engram-async]'):
                        timers['engram_wait_ms'] += values.get('wait_us', 0)/1000
                    elif line.startswith('[dsv4f-input]'):
                        timers['input_ms'] += values.get('total_us', 0)/1000
                record.update(timers)
                if samples:
                    record['vram_samples'] = samples
                    record['vram_peak_bytes'] = max(x['used_bytes'] for x in samples)
                    record['vram_min_free_bytes'] = min(x['total_bytes']-x['used_bytes'] for x in samples)
                if args.block_device:
                    record['device_read_bytes'] = (after['device'][2]-before['device'][2])*512
                report['runs'].append(record)
                (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
                print('run', index+1, 'tokens', result['timings']['prompt_n'],
                      'prefill_tps', round(result['timings']['prompt_per_second'], 3),
                      'expert_read_gib', round(record['expert_read_bytes']/2**30, 3),
                      'evictions', record['expert_evictions'],
                      'io_requests', record['io_requests'], 'merged', record['merged_planes'],
                      'rss_gib', round(after['memory']['VmRSS']/2**30, 3), flush=True)
            values = [r['timings']['prompt_per_second'] for r in report['runs']]
            report['summary'] = dict(cold_prefill_tps=values[0],
                                     first_after_preload_tps=values[0] if args.preload == '1' else None,
                                     warm_median_prefill_tps=statistics.median(values[1:]) if len(values)>1 else None)
            (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
            print(report['summary'], flush=True)
            if args.validation_tokens:
                sample = generate(state['url'], args.validation_tokens, cache_prompt=False)
                report['validation_sample'] = sample
                (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
                print('Short-prompt validation:', sample['content'], flush=True)
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
