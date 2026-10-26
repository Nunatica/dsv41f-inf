import importlib.util
from pathlib import Path
import struct
import json
import os
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('model_info', Path(__file__).resolve().parents[1]/'scripts'/'model_info.py')
loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loader)


def string(s):
    b = s.encode()
    return struct.pack('<Q', len(b)) + b


def shard(path, index, count, extra=None, tensor=True):
    metadata = {'general.architecture': 'deepseek4', 'split.no': index, 'split.count': count}
    metadata.update(extra or {})
    data = b'GGUF' + struct.pack('<IQQ', 3, int(tensor), len(metadata))
    for k, v in metadata.items():
        data += string(k)
        typ = 8 if isinstance(v, str) else 9 if isinstance(v, list) else 4
        data += struct.pack('<I', typ)
        if typ == 9:
            data += struct.pack('<IQ', 4, len(v))+struct.pack('<'+'I'*len(v), *v)
        else:
            data += string(v) if typ == 8 else struct.pack('<I', v)
    if tensor:
        data += string(f'blk.{index}.ffn_up_exps.weight') + struct.pack('<IQQQIQ', 3, 32, 32, 2, 39, 0)
    data += bytes((-len(data)) % 32) + (bytes(1024) if tensor else b'')
    path.write_bytes(data)


class LoaderTest(unittest.TestCase):
    def test_exact_format_filenames(self):
        self.assertEqual(loader.model_format('/elsewhere/'+loader.DSX_FILENAME), 'dsx-2.6bit')
        self.assertEqual(loader.model_format('/elsewhere/'+loader.UD_FILENAME), 'ud-q4-k-xl')
        for name in ('model.gguf', loader.DSX_FILENAME.upper(), 'copy-'+loader.DSX_FILENAME,
                     loader.DSX_FILENAME.replace('1017', '1018'),
                     loader.UD_FILENAME.replace('00001', '00002'), 'copy-'+loader.UD_FILENAME):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'exactly'):
                loader.model_format(name)

    def test_native_missing_engram_shard_only(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {'DSV4F_ENGRAM_DIR': folder}):
            paths = [Path(folder)/loader.UD_FILENAME.replace('00001', f'{i:05d}') for i in range(1, 10)]
            meta = {'general.architecture': 'deepseek41', 'split.tensors.count': 8,
                    'deepseek41.block_count': 40, 'deepseek41.context_length': 1048576,
                    'deepseek41.ple.layers': [1], 'deepseek41.ple.ngram_size': 2,
                    'deepseek41.ple.heads_per_ngram': 1, 'deepseek41.ple.head_vocab_sizes': [2],
                    'deepseek41.embedding_length_per_layer_input': 32}
            for i, p in enumerate(paths):
                if i != 7:
                    shard(p, i, 9, meta if i == 0 else None, tensor=i != 0)
            table = Path(folder)/'original.safetensors'
            header = json.dumps({
                'layers.1.engram.embed.weight': {'dtype': 'F8_E4M3', 'shape': [2, 32], 'data_offsets': [0, 64]},
                'layers.1.engram.embed.scale': {'dtype': 'F8_E8M0', 'shape': [2, 1], 'data_offsets': [64, 66]},
            }).encode()
            table.write_bytes(struct.pack('<Q', len(header))+header+bytes(66))
            info = loader.inspect(paths[0])
            self.assertEqual(info['format'], 'ud-q4-k-xl')
            self.assertEqual(info['layers'], 40)
            self.assertEqual(info['skipped_engram_shards'], [str(paths[7])])
            self.assertEqual(len(info['shards']), 8)
            self.assertIn(str(paths[8]), info['shards'])
            self.assertEqual(len(info['original_engram']['tensors']), 2)
            paths[1].unlink()
            with self.assertRaises(FileNotFoundError):
                loader.inspect(paths[0])
            shard(paths[1], 1, 9)
            shard(paths[0], 0, 9, {**meta, 'split.tensors.count': 9}, tensor=False)
            with self.assertRaisesRegex(ValueError, 'tensor count'):
                loader.inspect(paths[0])
            shard(paths[0], 0, 9, meta, tensor=False)
            table.write_bytes(struct.pack('<Q', len(header))+header+bytes(65))
            with self.assertRaisesRegex(ValueError, 'dtype/shape/extent'):
                loader.inspect(paths[0])
            table.unlink()
            with self.assertRaisesRegex(ValueError, 'Missing original FP8 Engram'):
                loader.inspect(paths[0])

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
