import functools
import itertools
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_expert_usage import belady_misses, load_routes, lru_misses


class ExpertUsageTest(unittest.TestCase):
    def test_cache_replay_against_exhaustive_optimum(self):
        # Exercise replacement and hits independently of the trace implementation.
        for seq in itertools.product(range(3), repeat=6):
            @functools.lru_cache(None)
            def optimum(index, cached):
                if index == len(seq): return 0
                key = seq[index]
                if key in cached: return optimum(index+1,cached)
                choices = [cached+(key,)] if len(cached)<2 else [tuple(k for k in cached if k!=victim)+(key,) for victim in cached]
                return 1+min(optimum(index+1,tuple(sorted(c))) for c in choices)
            self.assertEqual(sum(belady_misses(seq,2)),optimum(0,()))
        self.assertEqual(lru_misses([0,1,0,2,1],2),[1,1,0,1,1])
        self.assertEqual(belady_misses([0,1,0],0),[1,1,1])

    def test_trace_transposes_batches_and_rejects_incomplete_layers(self):
        data=b'R 0 2 2 0 1 1 2\nR 1 2 2 2 3 0 3\n'
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'trace';p.write_bytes(data)
            tokens,chunks=load_routes(p,0,len(data),2,4,2)
            self.assertEqual(tokens,[[0,1,6,7],[1,2,4,7]])
            self.assertEqual(chunks,[2])
            with self.assertRaisesRegex(ValueError,'Incomplete'):
                load_routes(p,0,data.index(b'R 1'),2,4,2)


if __name__=='__main__':unittest.main()
