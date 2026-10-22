import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from diagnose_engram_io import summarize


class EngramProfileTest(unittest.TestCase):
    def test_completion_order(self):
        def table(slot, tokens):
            return (f'[dsv4f-engram-profile] slot={slot} tokens={tokens} rows={tokens*24} '
                    'reads=2 bytes=132 hits=0 dedup=0 hash_us=1 io_us=90 fp8_us=2 copy_us=3 total_us=100')
        def inp(tokens):
            return f'[dsv4f-input] tokens={tokens} maps=0 map_bytes=0 maps_us=0 total_us=220'
        results=[]
        for mode in (0,1,2):
            lines=[]
            for tokens in (13,1,1):
                a,b,c=table(0,tokens),table(1,tokens),inp(tokens)
                # Prefill retains the synchronous path in all modes.
                lines += [a,b,c] if tokens>4 or mode==0 else [c,a,b] if mode==1 else [a,c,b]
            results.append(summarize('\n'.join(lines),dict(prompt_n=13,predicted_n=3,predicted_ms=150)))
        self.assertEqual(results[0],results[1])
        self.assertEqual(results[0],results[2])
        self.assertEqual(results[0]['decode_evaluations'],2)
        with self.assertRaisesRegex(RuntimeError,'Incomplete'):
            summarize('\n'.join(lines[:-1]),dict(prompt_n=13,predicted_n=3,predicted_ms=150))


if __name__=='__main__':
    unittest.main()
