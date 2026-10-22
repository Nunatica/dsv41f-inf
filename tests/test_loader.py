import importlib.util
from pathlib import Path
import struct
import json
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('model_info', Path(__file__).resolve().parents[1]/'scripts'/'model_info.py')
loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loader)


def string(s):
    b = s.encode()
    return struct.pack('<Q', len(b)) + b


def shard(path, index, count):
    metadata = {'general.architecture': 'deepseek4', 'split.no': index, 'split.count': count}
    data = b'GGUF' + struct.pack('<IQQ', 3, 1, len(metadata))
    for k, v in metadata.items():
        data += string(k)
        data += struct.pack('<I', 8 if isinstance(v, str) else 4)
        data += string(v) if isinstance(v, str) else struct.pack('<I', v)
    data += string(f'blk.{index}.ffn_up_exps.weight') + struct.pack('<IQQQIQ', 3, 32, 32, 2, 39, 0)
    data += bytes((-len(data)) % 32) + bytes(1024)
    path.write_bytes(data)


class LoaderTest(unittest.TestCase):
    def test_complete_shards_and_resident_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = [Path(folder)/f'model-{i:05d}-of-00002.gguf' for i in (1, 2)]
            for i, p in enumerate(paths):
                shard(p, i, 2)
            info = loader.inspect(paths[0])
            self.assertEqual(info['expert_bytes'], 2048)
            self.assertEqual(len(info['shards']), 2)
            self.assertEqual(info['model_bytes'], sum(p.stat().st_size for p in paths))
            with self.assertRaisesRegex(ValueError, 'first GGUF shard'):
                loader.inspect(paths[1])
            paths[1].unlink()
            with self.assertRaises(FileNotFoundError):
                loader.inspect(paths[0])

    def test_reject_inconsistent_and_truncated(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)/'model-00001-of-00002.gguf'
            q = Path(folder)/'model-00002-of-00002.gguf'
            shard(p, 0, 2)
            shard(q, 0, 2)
            with self.assertRaisesRegex(ValueError, 'Inconsistent'):
                loader.inspect(p)
            p.write_bytes(b'GGUF')
            with self.assertRaisesRegex(ValueError, 'Truncated'):
                loader.inspect(p)


class MtpLoaderTest(unittest.TestCase):
    def test_text_only_and_manifest_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            binary = Path(folder)/'dspark-xs2.bin'
            manifest = binary.with_suffix('.json')
            names = ['mtp.0.main_proj.weight', 'mtp.0.main_norm.weight', 'mtp.2.norm.weight',
                     'mtp.2.markov_head.embed.weight', 'mtp.2.markov_head.head.weight',
                     'mtp.2.confidence_head.proj.weight']
            tensors = [[name, 'BF16', i*64, 64, [1, 32]] for i, name in enumerate(names)]
            # Invalid and missing payloads in excluded vision sections are ignored.
            tensors += [['vision.skip', 'UNSUPPORTED', 1<<40, 1<<30, [1]],
                        ['mtp.0.ffn.gate.bias_vl', 'F32', 1<<40, 512, [128]]]
            binary.write_bytes(bytes(6*64))
            def save(): manifest.write_text(json.dumps({'tensors': tensors}))
            save()
            info = loader.inspect_mtp(binary)
            self.assertEqual(info['model_bytes'], 6*128)  # BF16 promoted to F32
            self.assertEqual(info['mtp_tensors'], 6)
            self.assertEqual(info['vision_tensors_loaded'], 0)
            tensors[0][3] -= 1; save()
            with self.assertRaisesRegex(ValueError, 'byte size'): loader.inspect_mtp(binary)
            tensors[0][3] += 1
            tensors[0][2] = 1<<32; save()
            with self.assertRaisesRegex(ValueError, 'Truncated'): loader.inspect_mtp(binary)
            tensors[0][2] = 0
            tensors[1][2] = 0; save()
            with self.assertRaisesRegex(ValueError, 'Overlapping'): loader.inspect_mtp(binary)
            tensors[1][2] = 64
            tensors.append(tensors[0]); save()
            with self.assertRaisesRegex(ValueError, 'Duplicate'): loader.inspect_mtp(binary)


if __name__ == '__main__':
    unittest.main()
