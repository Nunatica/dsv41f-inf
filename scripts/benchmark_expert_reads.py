#!/usr/bin/env python3
"""Read-only O_DIRECT random expert-sized reads from an existing weight file."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import mmap
import os
from pathlib import Path
import random
import statistics
import time

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('model',type=Path)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--reads',type=int,default=128)
a=p.parse_args()
size=a.model.stat().st_size
block=3*5120*2304*144//256
assert block%4096==0
rng=random.Random(42)
offsets=[rng.randrange(4096,(size-block)//4096)*4096 for _ in range(a.reads)]
fd=os.open(a.model,os.O_RDONLY|os.O_DIRECT)
report=dict(path=str(a.model.resolve()),read_only=True,direct_io=True,block_bytes=block,reads=a.reads,runs=[])
try:
    for workers in [1,4]:
        def run_batch(batch):
            buf=mmap.mmap(-1,block)
            times=[]
            try:
                for off in batch:
                    start=time.perf_counter()
                    got=os.preadv(fd,[buf],off)
                    if got!=block:raise RuntimeError('short read')
                    times.append((time.perf_counter()-start)*1000)
            finally:buf.close()
            return times
        start=time.perf_counter()
        with ThreadPoolExecutor(workers) as pool:
            batches=list(pool.map(run_batch,[offsets[i::workers] for i in range(workers)]))
        elapsed=time.perf_counter()-start
        lat=sorted(x for b in batches for x in b)
        row=dict(workers=workers,seconds=elapsed,gib_per_second=block*len(offsets)/2**30/elapsed,
                 latency_mean_ms=statistics.mean(lat),latency_p50_ms=statistics.median(lat),latency_p95_ms=lat[int(len(lat)*.95)])
        report['runs'].append(row)
        print(row,flush=True)
finally:os.close(fd)
a.output.write_text(json.dumps(report,indent=2)+'\n')
