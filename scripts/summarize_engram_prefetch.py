#!/usr/bin/env python3
"""Compare one-batch Engram prefetch against same-batch overlap, A/B/A."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    args=p.parse_args()
    records={name:json.loads((args.root/name/'report.json').read_text())
             for name in ('control-before','prefetch','control-after')}
    first=records['control-before']
    expected={k:v for k,v in first['state']['engine_options'].items() if k!='DSV4F_ENGRAM_PREFETCH'}
    summary=[]
    for name,r in records.items():
        assert r['prompt_sha256']==first['prompt_sha256']
        assert r['state']['command']==first['state']['command']
        assert {k:v for k,v in r['state']['engine_options'].items() if k!='DSV4F_ENGRAM_PREFETCH'}==expected
        assert r['prefill_overlap']=='1' and not r['mtp']
        assert r['prefetch']==('1' if name=='prefetch' else '0')
        rows=[]
        for i,q in enumerate(r['requests'],1):
            b=q['batch_timers'];t=q['timings'];inputs=b['inputs']
            assert t['cache_n']==0 and t['prompt_n']==17778
            assert q['before']['memory']['VmSwap']==q['after']['memory']['VmSwap']==0
            assert [int(x['tokens']) for x in inputs]==[4096,4096,4096,4096,1394]
            assert len(b['async_rows'])==2*len(inputs)
            hits=b['prefetch'];reads=b['prefetch_reads']
            if name=='prefetch':
                assert [x['hit'] for x in hits]==[0,1,1,1,1]
                assert len(reads)==10 and not any(x['cancelled'] for x in reads)
            else:assert not hits and not reads
            by_batch=[]
            for j,x in enumerate(inputs):
                a=b['async_rows'][2*j:2*j+2]
                assert [z['slot'] for z in a]==[0,1] and all(z['tokens']==x['tokens'] for z in a)
                by_batch.append(dict(tokens=int(x['tokens']),wait_ms=sum(z['wait_us'] for z in a)/1000,
                                     wait_slot0_ms=a[0]['wait_us']/1000,wait_slot1_ms=a[1]['wait_us']/1000))
            rows.append(dict(request=i,prompt_ms=t['prompt_ms'],tokens_per_second=t['prompt_per_second'],
                             input_ms=b['input_ms'],consumer_wait_ms=b['async_wait_ms'],
                             pool_read_ms=sum(x['read_us'] for x in reads)/1000,
                             rss_after_gib=q['after']['memory']['VmRSS']/2**30,
                             by_batch=by_batch,output=q['content']))
        summary.append(dict(name=name,prefetch=r['prefetch'],requests=rows))
    result=dict(prompt_sha256=first['prompt_sha256'],prompt_tokens=17778,measurements=summary,
                note='Request 1 has fresh-process caches; OS caches are not flushed. Keep warm requests separate. '
                     'Async read_us includes job wait and pinned memcpy in prefetch mode; pool read_us measures I/O/decode. '
                     'No storage stall is subtracted from elapsed timings.')
    (args.root/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    for s in summary:
        print(s['name'],'prompt_s',[round(x['prompt_ms']/1000,3) for x in s['requests']],
              'wait_s',[round(x['consumer_wait_ms']/1000,3) for x in s['requests']])
        print('batch_wait_s',[[round(z['wait_ms']/1000,3) for z in x['by_batch']] for x in s['requests']])


if __name__=='__main__':main()
