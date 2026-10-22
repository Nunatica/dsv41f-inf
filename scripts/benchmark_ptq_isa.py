#!/usr/bin/env python3
"""Compare CPU quantization ISA modes on an owned native llama-server."""
import argparse
import json
from pathlib import Path
import statistics
from benchmark import generate, PROMPT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--isa', choices=('auto', 'avx2', 'avx512'), required=True)
    p.add_argument('--ptq-kernel', choices=('auto', 'baseline', 'f16c', 'threshold'))
    p.add_argument('--quant-isa', choices=('auto', 'legacy', 'avx512'))
    p.add_argument('--q3-kernel', choices=('auto', 'signed', 'offset'))
    p.add_argument('--url', default='http://127.0.0.1:18081')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--runs', type=int, default=5)
    p.add_argument('--tokens', type=int, default=128)
    p.add_argument('--warmup-tokens', type=int, default=32)
    args = p.parse_args()
    if args.runs < 1 or args.tokens < 2:
        p.error('runs >= 1 and tokens >= 2 are required')
    root = Path(__file__).resolve().parents[1]
    state = json.loads((root / 'state/server.json').read_text())
    assert state['url'].rstrip('/') == args.url.rstrip('/'), 'Owned server URL differs'
    assert state['engine_options']['DSV4F_MTP'] == '0', 'Disable MTP for this test'
    assert state['engine_options']['DSV4F_CPU_PTQ_ISA'] == args.isa, 'Server ISA differs'
    if args.ptq_kernel:
        assert state['engine_options']['DSV4F_CPU_PTQ_KERNEL'] == args.ptq_kernel, 'Server PTQ kernel differs'
    if args.quant_isa:
        assert state['engine_options']['DSV4F_CPU_QUANT_ISA'] == args.quant_isa, 'Server quant ISA differs'
    if args.q3_kernel:
        assert state['engine_options']['DSV4F_CPU_Q3_K_KERNEL'] == args.q3_kernel, 'Server Q3 kernel differs'
    proc = Path('/proc') / str(state['pid'])
    assert (proc / 'exe').resolve() == root / 'build/bin/dsv4f-server'
    report = {'isa': args.isa, 'ptq_kernel': args.ptq_kernel, 'quant_isa': args.quant_isa, 'q3_kernel': args.q3_kernel, 'prompt': PROMPT, 'engine_options': state['engine_options'],
              'server_log': state['log'], 'pid': state['pid'], 'warmup_tokens': args.warmup_tokens, 'runs': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.warmup_tokens:
        report['warmup'] = generate(args.url, args.warmup_tokens, PROMPT)
    for i in range(args.runs):
        result = generate(args.url, args.tokens, PROMPT)
        result['memory'] = {line.split(':')[0]: int(line.split()[1])*1024
                            for line in (proc/'status').read_text().splitlines()
                            if line.startswith(('VmRSS:', 'VmSwap:'))}
        assert result['memory']['VmSwap'] == 0 and result['content']
        report['runs'].append(result)
        report['median_decode_tokens_per_second'] = statistics.median(
            r['timings']['predicted_per_second'] for r in report['runs'])
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        print(args.isa, i+1, result['timings']['predicted_per_second'], 'tokens/s', flush=True)
    print('median', report['median_decode_tokens_per_second'], flush=True)


if __name__ == '__main__':
    main()
