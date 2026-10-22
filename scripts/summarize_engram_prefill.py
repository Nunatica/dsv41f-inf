#!/usr/bin/env python3
"""Summarize A/B/A same-batch prefill tests without conflating cache warm-up."""
import argparse
import json
from pathlib import Path
import statistics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    args=p.parse_args()
    records={name:json.loads((args.root/name/'report.json').read_text()) for name in ('sync-run','async-run','sync-recheck')}
    first=records['sync-run']
    prompt_tokens=first['requests'][0]['timings']['prompt_n']
    expected={k:v for k,v in first['state']['engine_options'].items() if k!='DSV4F_ENGRAM_PREFILL_OVERLAP'}
    summaries=[]
    for name,record in records.items():
        assert record['prompt_sha256']==first['prompt_sha256']
        assert record['state']['command']==first['state']['command']
        actual={k:v for k,v in record['state']['engine_options'].items() if k!='DSV4F_ENGRAM_PREFILL_OVERLAP'}
        assert actual==expected
        rows=[]
        for i,r in enumerate(record['requests'],1):
            t=r['timings'];b=r['batch_timers']
            assert t['cache_n']==0 and t['prompt_n']==prompt_tokens
            assert r['after']['memory']['VmSwap']==0
            if record['prefill_overlap']=='1':
                assert len(b['async_rows'])==2*len(b['inputs'])
                for slot in (0,1):
                    entries=[q for q in b['async_rows'] if q['slot']==slot]
                    assert [q['tokens'] for q in entries]==[q['tokens'] for q in b['inputs']]
                    assert sum(q['bytes'] for q in entries)==prompt_tokens*24576
            else:
                assert not b['async_rows']
            by_slot=[]
            for slot in (0,1):
                entries=[q for q in b['async_rows'] if q['slot']==slot]
                by_slot.append(dict(slot=slot,read_ms=sum(q['read_us'] for q in entries)/1000,
                                    wait_ms=sum(q['wait_us'] for q in entries)/1000))
            rows.append(dict(request=i,prompt_ms=t['prompt_ms'],tokens_per_second=t['prompt_per_second'],
                             input_ms=b['input_ms'],async_read_ms=b['async_read_ms'],async_wait_ms=b['async_wait_ms'],
                             other_ms=t['prompt_ms']-b['input_ms']-b['async_wait_ms'],by_slot=by_slot,
                             rss_after_gib=r['after']['memory']['VmRSS']/2**30))
        summaries.append(dict(name=name,mode=record['prefill_overlap'],requests=rows,
                              median_prompt_ms=statistics.median(r['prompt_ms'] for r in rows)))
    result=dict(prompt_sha256=first['prompt_sha256'],prompt_tokens=prompt_tokens,
                note='Fresh-server request order retained. First and repeated requests have different Engram cache state; '
                     'do not interpret a single all-request median as a cache-independent speedup. '
                     'Worker read minus consumer wait estimates work completed before consumption, not a matched counterfactual latency saving.',
                measurements=summaries)
    (args.root/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    for group in summaries:
        print(group['name'],'seconds',[round(r['prompt_ms']/1000,3) for r in group['requests']])
        print('input seconds',[round(r['input_ms']/1000,3) for r in group['requests']],
              'wait seconds',[round(r['async_wait_ms']/1000,3) for r in group['requests']])


if __name__=='__main__':
    main()
