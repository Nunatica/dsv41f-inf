#!/usr/bin/env python3
"""Paired 128-GiB, preloaded 2.6bpw prefill I/O experiment."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    cases = [('baseline', '0', '0'), ('sort', '1', '0'),
             ('pipeline', '0', '1'), ('combined', '1', '1')]
    records = []
    for index, (name, sort, pipeline) in enumerate(cases + list(reversed(cases))):
        directory = out/f'{index+1:02d}-{name}'
        command = [sys.executable, str(ROOT/'scripts/benchmark_prefill_io.py'), str(directory),
                   '--batch-io', '1', '--ram-budget-mib', '131072', '--preload', '1',
                   '--io-sort', sort, '--io-pipeline', pipeline,
                   '--prompt-file', str(ROOT/'PROMPT_TEST2.md'), '--runs', '3',
                   '--model', str(ROOT/'models/DeepSeek-V4.1-Flash-1017-dsx-2.6bit.gguf'),
                   '--block-device', 'nvme0n1']
        print(f'Case {index+1}/8: {name}', flush=True)
        with (out/f'{index+1:02d}-{name}.log').open('w') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        report = json.loads((directory/'report.json').read_text())
        records.append({'case': name, 'directory': str(directory), 'report': report})
        print(name, report['summary'], flush=True)
        (out/'reports.json').write_text(json.dumps(records, indent=2)+'\n')
    assert len({r['report']['prompt_sha256'] for r in records}) == 1
    assert len({json.dumps(r['report']['binary_sha256'], sort_keys=True) for r in records}) == 1
    assert len({run['timings']['prompt_n'] for r in records for run in r['report']['runs']}) == 1
    summary = {}
    for name, _, _ in cases:
        reports = [r['report'] for r in records if r['case'] == name]
        row = {}
        for phase, runs in [('first', [r['runs'][0] for r in reports]),
                            ('repeated', [run for r in reports for run in r['runs'][1:]])]:
            mean_ms = statistics.mean(run['timings']['prompt_ms'] for run in runs)
            row[phase] = {'samples': len(runs), 'mean_seconds': mean_ms/1000,
                          'tps': runs[0]['timings']['prompt_n']*1000/mean_ms,
                          'mean_expert_read_gib': statistics.mean(run['expert_read_bytes']/2**30 for run in runs),
                          'mean_io_requests': statistics.mean(run['io_requests'] for run in runs),
                          'mean_merged_planes': statistics.mean(run['merged_planes'] for run in runs),
                          'mean_device_read_gib': statistics.mean(run['device_read_bytes']/2**30 for run in runs)}
        row['peak_memory_gib'] = max(run['after']['cgroup']['memory.peak']/2**30 for r in reports for run in r['runs'])
        row['preload_resident_gib'] = [r['preload_snapshot']['weights']['resident_bytes']/2**30 for r in reports]
        summary[name] = row
    for row in summary.values():
        for phase in ('first', 'repeated'):
            row[phase]['throughput_change_pct'] = (row[phase]['tps']/summary['baseline'][phase]['tps']-1)*100
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
