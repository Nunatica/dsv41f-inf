import os
import json
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import model_info


class PruningMemoryEstimateTest(unittest.TestCase):
    def test_mixed_formats_and_sparse_history(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'Pruning.gguf'
            tensors = [('blk.0.ffn_exps_gu_type', [6], 26, 0)]
            offset = 32
            planes = {'q3_k': 28160, 'q2_0': 18432, 'ptq1_0': 14336}
            for suffix, typ in [('q3_k', 11), ('q2_0', 42), ('ptq1_0', 143)]:
                for role in ('gate', 'up', 'down'):
                    tensors.append((f'blk.0.ffn_{role}_exps.{suffix}.weight', [256, 256, 2], typ, offset))
                    offset += 2 * planes[suffix]
            with path.open('wb') as f:
                f.write(struct.pack('<6i', 11, 42, 143, 11, 42, 143))
                f.truncate(offset)
            history = {'version': 1, 'model': 'Pruning', 'experts': {
                'blk.0.ffn_exps.q3_k.weight': {'0': 1, '1': 2},
                'blk.0.ffn_exps.q2_0.weight': {'0': 11, '1': 12},
                'blk.0.ffn_exps.ptq1_0.weight': {'0': 101, '1': 102}}}
            table = Path(temp) / 'Pruning.routing.json'
            table.write_text(json.dumps(history))
            info = {'model': str(path), 'shards': [str(path)]}
            with patch.object(model_info, 'header', return_value=({}, tensors, 0)):
                self.assertEqual(model_info.resident_pruned_bytes(info, 0, temp), 0)
                self.assertEqual(model_info.resident_pruned_bytes(info, 1, temp), 0)
                self.assertEqual(model_info.resident_pruned_bytes(info, 25, temp), 3 * planes['q3_k'])
                self.assertEqual(model_info.resident_pruned_bytes(info, 50, temp), 6 * planes['q3_k'] + 3 * planes['q2_0'])
                del history['experts']['blk.0.ffn_exps.ptq1_0.weight']
                table.write_text(json.dumps(history))
                self.assertEqual(model_info.resident_pruned_bytes(info, 25, temp), 3 * planes['ptq1_0'])


class AdaptivePruningLauncherTest(unittest.TestCase):
    def test_pruning_range_and_mutual_exclusion(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith('DSV4F_')}
        with tempfile.TemporaryDirectory() as temp:
            model = Path(temp) / 'DeepSeek-V4.1-Flash-1017-dsx-2.6bit.gguf'
            def string(value):
                value = value.encode()
                return struct.pack('<Q', len(value)) + value
            model.write_bytes(b'GGUF' + struct.pack('<IQQ', 3, 0, 1)
                              + string('general.architecture') + struct.pack('<I', 8)
                              + string('deepseek41'))
            env.update(MODEL=str(model), DSV4F_ADAPTIVE_ROUTING='0')
            for value in ('0', '1', '25', '50'):
                with self.subTest(value=value):
                    result = subprocess.run([str(ROOT / 'run-server.sh'), '--dry-run'],
                        env={**env, 'DSV4F_ADAPTIVE_PRUNING': value}, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn(f'Adaptive expert pruning: {value}%', result.stdout)
            for value in ('-1', '51', '100', '1.5', 'abc', '01', '999999999999999999999'):
                with self.subTest(value=value):
                    result = subprocess.run([str(ROOT / 'run-server.sh'), '--dry-run'],
                        env={**env, 'DSV4F_ADAPTIVE_PRUNING': value}, capture_output=True, text=True)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn('integer in 0..50', result.stderr)
            result = subprocess.run([str(ROOT / 'run-server.sh'), '--dry-run'],
                env={**env, 'DSV4F_ADAPTIVE_ROUTING': '1', 'DSV4F_ADAPTIVE_PRUNING': '25'},
                capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('cannot both be enabled', result.stderr)
