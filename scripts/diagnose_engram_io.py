#!/usr/bin/env python3
"""Measure Engram wall time and storage activity on an owned MTP-off server.

Keeps model/OS caches intact. Device counters include unrelated host I/O and
FUSE activity; pread bytes in the engine log are logical, not physical bytes.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import threading
import time
from urllib.request import urlopen

from benchmark import PROMPT, generate

ROOT = Path(__file__).resolve().parents[1]


def snapshot(pid):
    result = {'monotonic': time.monotonic(), 'devices': {}}
    for device in Path('/sys/block').iterdir():
        if device.name.startswith(('loop', 'ram')):
            continue
        values = list(map(int, (device/'stat').read_text().split()))
        result['devices'][device.name] = dict(reads=values[0], sectors=values[2],
                                            read_ms=values[3], io_ms=values[9])
    try:
        result['process_io'] = {k: int(v) for k, v in
                                (line.split(':') for line in Path(f'/proc/{pid}/io').read_text().splitlines())}
        result['memory'] = {line.split(':')[0]: int(line.split()[1])*1024
                            for line in Path(f'/proc/{pid}/status').read_text().splitlines()
                            if line.startswith(('VmRSS:', 'VmSwap:'))}
    except FileNotFoundError:
        pass
    return result


def summarize(text, timing):
    groups, tables = [], {0: [], 1: []}
    for line in text.splitlines():
        if line.startswith('[dsv4f-engram-profile]'):
            row = {k: float(v) for k, v in re.findall(r'(\w+)=([\d.]+)', line)}
            tables[int(row['slot'])].append(row)
        elif line.startswith('[dsv4f-input]'):
            values = {k: float(v) for k, v in re.findall(r'(\w+)=([\d.]+)', line)}
            groups.append(dict(input=values))
    # Async completion lines can follow their input marker. Each table keeps
    # call order, so pair by per-table call index rather than adjacent lines.
    if any(len(rows) != len(groups) for rows in tables.values()):
        raise RuntimeError('Incomplete two-table Engram profile')
    for index, group in enumerate(groups):
        group['tables'] = [tables[slot][index] for slot in (0, 1)]
        if any(row['tokens'] != group['input']['tokens'] for row in group['tables']):
            raise RuntimeError('Engram profile batch sizes do not match inputs')
    single = [g for g in groups if g['input']['tokens'] == 1]
    # Some server paths evaluate the final prompt token separately; others
    # include it in the last multi-token batch. Identify it from prompt_n.
    prefill_tokens = sum(g['input']['tokens'] for g in groups if g['input']['tokens'] > 1)
    prompt_singles = int(timing['prompt_n'] - prefill_tokens)
    if prompt_singles not in (0, 1):
        raise RuntimeError(f'Unexpected separately evaluated prompt tokens: {prompt_singles}')
    decode = single[prompt_singles:]
    expected = timing['predicted_n'] - 1
    if len(decode) != expected:
        raise RuntimeError(f'Expected {expected} decode evaluations, found {len(decode)}')
    if any(len(g['tables']) != 2 for g in groups):
        raise RuntimeError('Incomplete two-table Engram profile')
    result = dict(decode_evaluations=len(decode), single_token_inputs=len(single),
                  prompt_single_token_inputs=prompt_singles,
                  native_ms_per_token=timing['predicted_ms']/expected)
    for phase, selected in [('decode', decode), ('prefill', [g for g in groups if g['input']['tokens'] > 1])]:
        rows = [r for g in selected for r in g['tables']]
        tokens = sum(g['input']['tokens'] for g in selected)
        if not tokens:
            continue
        import statistics
        totals = [sum(r['total_us'] for r in g['tables']) for g in selected]
        result[phase] = dict(tokens=tokens, logical_bytes=sum(r['bytes'] for r in rows),
                             reads=sum(r['reads'] for r in rows),
                             cache_hit_percent=100*sum(r['hits'] for r in rows)/sum(r['rows'] for r in rows),
                             input_ms_per_token=sum(g['input']['total_us'] for g in selected)/tokens/1000,
                             engram_ms_per_token=sum(r['total_us'] for r in rows)/tokens/1000,
                             io_ms_per_token=sum(r['io_us'] for r in rows)/tokens/1000,
                             fp8_ms_per_token=sum(r['fp8_us'] for r in rows)/tokens/1000,
                             call_total_us_median=statistics.median(totals),
                             call_total_us_p95=sorted(totals)[int(.95*(len(totals)-1))],
                             call_total_us_max=max(totals))
    result['engram_fraction_percent'] = 100*result['decode']['engram_ms_per_token']/result['native_ms_per_token']
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--long-tokens', type=int, default=512)
    args = parser.parse_args()
    if args.long_tokens < 2:
        parser.error('--long-tokens must be at least 2')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    manifest = ROOT/'state/server.json'
    if manifest.exists():
        raise RuntimeError('Existing owned server; refusing to replace it')
    env = dict(os.environ, HOST='127.0.0.1', PORT='18089', DSV4F_MTP='0',
               DSV4F_INPUT_PROFILE='1', DSV4F_ENGRAM_PROFILE='1', DSV4F_SERVICE_PROFILE='1')
    command = ['systemd-run', '--user', '--scope', '--quiet', '-p', 'MemorySwapMax=0',
               '-p', 'MemoryMax=184G', './run-server.sh']
    state = None
    report = dict(date_utc=datetime.now(timezone.utc).isoformat(), requests=[])
    (out/'storage.txt').write_text(subprocess.check_output(
        ['findmnt', '-T', str(ROOT/'models/model-00047-of-00048.safetensors')], text=True))
    print('Loading owned server with launcher defaults, MTP=0', flush=True)
    with (out/'launcher.log').open('w') as log:
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
        try:
            for _ in range(480):
                if proc.poll() is not None:
                    raise RuntimeError(f'Server exited {proc.returncode}; see {out}/launcher.log')
                try:
                    s = json.loads(manifest.read_text())
                    assert s['url'] == 'http://127.0.0.1:18089'
                    assert s['engine_options']['DSV4F_MTP'] == '0'
                    assert Path(f'/proc/{s["pid"]}/exe').resolve() == ROOT/'build/bin/dsv4f-server'
                    with urlopen(s['url']+'/health', timeout=1) as response:
                        assert json.load(response)['status'] == 'ok'
                    state = s
                    break
                except (OSError, ValueError, AssertionError):
                    time.sleep(1)
            else:
                raise RuntimeError('Server load timeout')
            report['state'] = state
            server_log = Path(state['log'])
            print(f'Ready after {state["load_seconds"]:.1f}s', flush=True)
            long_prompt = (ROOT/'PROMPT_TEST.md').read_text()
            for label, tokens, prompt in [('short-warmup', 32, PROMPT), ('short-1', 128, PROMPT),
                                           ('short-2', 128, PROMPT), ('long', args.long_tokens, long_prompt)]:
                offset = server_log.stat().st_size
                samples, done = [], threading.Event()

                def monitor():
                    while True:
                        sample = snapshot(state['pid'])
                        sample['log_bytes'] = server_log.stat().st_size
                        samples.append(sample)
                        if done.wait(.1):
                            break

                worker = threading.Thread(target=monitor)
                worker.start()
                print(f'{label}: generating {tokens} tokens', flush=True)
                try:
                    result = generate(state['url'], tokens, prompt)
                finally:
                    done.set()
                    worker.join()
                samples.append(dict(snapshot(state['pid']), log_bytes=server_log.stat().st_size))
                with server_log.open('rb') as stream:
                    stream.seek(offset)
                    chunk = stream.read()
                (out/f'{label}.log').write_bytes(chunk)
                (out/f'{label}-samples.json').write_text(json.dumps(samples))
                result.update(label=label, prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
                (out/f'{label}-response.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
                result['summary'] = summarize(chunk.decode(errors='replace'), result['timings'])
                assert all(s.get('memory', {}).get('VmSwap', 0) == 0 for s in samples)
                report['requests'].append(result)
                (out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
                print(label, json.dumps(result['summary']), flush=True)
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
    print('Owned server stopped', flush=True)


if __name__ == '__main__':
    main()
