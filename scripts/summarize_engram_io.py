#!/usr/bin/env python3
"""Summarize diagnose_engram_io.py output and sampled device read counters."""
import argparse
import json
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    report = json.loads((args.directory/'report.json').read_text())
    complete_log = (args.directory/'server.log').read_bytes()
    summary = []
    for request in report['requests']:
        label = request['label']
        chunk = (args.directory/f'{label}.log').read_bytes()
        base = complete_log.find(chunk)
        if base < 0:
            raise ValueError(f'{label} log chunk missing from complete log')
        samples = json.loads((args.directory/f'{label}-samples.json').read_text())
        singles = list(re.finditer(rb'\[dsv4f-input\] tokens=1 .*?\n', chunk))
        # Sampling observes log writes after input preparation, including the
        # separately evaluated final prompt token. The boundary has ~100 ms
        # granularity and may include a little of the following evaluation.
        boundary = base + singles[0].end()
        first = next(i for i, s in enumerate(samples) if s['log_bytes'] >= boundary)
        a, b = samples[first], samples[-1]
        seconds = b['monotonic']-a['monotonic']
        devices = {}
        for name in b['devices']:
            delta = {k: b['devices'][name][k]-a['devices'][name][k]
                     for k in b['devices'][name]}
            delta.update(read_mib=delta['sectors']*512/(1 << 20),
                         read_mib_per_second=delta['sectors']*512/(1 << 20)/seconds,
                         average_read_request_kib=delta['sectors']*.5/max(1, delta['reads']))
            devices[name] = delta
        process = {k: b['process_io'][k]-a['process_io'][k] for k in b['process_io']}
        summary.append(dict(label=label, timings=request['timings'], **request['summary'],
                            sampled_decode_seconds=seconds, devices=devices,
                            process_io=process,
                            maximum_swap_bytes=max(s.get('memory', {}).get('VmSwap', 0) for s in samples)))
    (args.directory/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
