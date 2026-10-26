import os
from pathlib import Path
import subprocess
import struct
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import memory_budget as memory


class MemoryBudgetTest(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(memory.budget_bytes({'DSV4F_RAM_BUDGET_MIB': '131072'}), 128 << 30)
        self.assertEqual(memory.budget_bytes({}), 0)
        for value in ('-1', '1.5', '', 'max', '１２', str(1 << 44)):
            with self.subTest(value=value), self.assertRaises(ValueError):
                memory.budget_bytes({'DSV4F_RAM_BUDGET_MIB': value})

    def test_ancestor_limit_includes_cgroup_root(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            leaf = root/'slice'/'scope'
            leaf.mkdir(parents=True)
            proc = root/'proc'
            proc.write_text('0::/slice/scope\n')
            (root/'memory.max').write_text(str(128 << 30))
            (leaf/'memory.max').write_text('max')
            (leaf/'memory.current').write_text('123')
            paths = list(memory.cgroup_paths(proc, root))
            self.assertEqual(paths, [leaf, leaf.parent, root])
            state = memory.memory_state(paths)
            self.assertTrue(memory.protected(128 << 30, state))
            self.assertFalse(memory.protected(64 << 30, state))
            self.assertEqual(state['memory.current'], 123)
            (leaf/'memory.max').write_text(str(32 << 30))
            self.assertEqual(memory.memory_state(paths)['limit_bytes'], 32 << 30)
            (root/'memory.max').write_text('max')
            (leaf/'memory.max').write_text('max')
            self.assertFalse(memory.protected(128 << 30, memory.memory_state(paths)))
            self.assertTrue(memory.protected(0, memory.memory_state(paths)))

    def test_effective_budget_obeys_tighter_parent(self):
        state = {'limit_bytes': 64 << 30}
        self.assertEqual(memory.effective_budget(128 << 30, state), 64 << 30)
        self.assertEqual(memory.effective_budget(0, {'limit_bytes': None}), 0)
        with self.assertRaises(RuntimeError):
            memory.effective_budget(128 << 30, {'limit_bytes': None})
        with self.assertRaises(RuntimeError):
            memory.effective_budget(128 << 30, {'limit_bytes': 100})

    def test_launch_defaults_and_validation(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith('DSV4F_')}
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        model = Path(temporary.name)/'fixture.gguf'
        def string(value):
            value = value.encode()
            return struct.pack('<Q', len(value))+value
        model.write_bytes(b'GGUF'+struct.pack('<IQQ',3,0,1)+string('general.architecture')+struct.pack('<I',8)+string('deepseek41'))
        env['MODEL'] = str(model)
        result = subprocess.run([str(ROOT/'run-server.sh'), '--dry-run'], env=env,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('RAM budget: 0 MiB; SSD payload I/O: resident', result.stdout)
        self.assertIn('--load-mode none', result.stdout)
        self.assertIn('--lazy-mode off', result.stdout)
        self.assertNotIn('--no-host', result.stdout)
        for value in ('-1', '123.4', '4096'):
            result = subprocess.run([str(ROOT/'run-server.sh'), '--dry-run'],
                                    env={**env, 'DSV4F_RAM_BUDGET_MIB': value}, capture_output=True)
            self.assertNotEqual(result.returncode, 0)

        legacy = subprocess.run([str(ROOT/'run-server.sh'), '--dry-run'],
                                env={**env, 'DSV4F_RAM_BUDGET_MIB': '0'}, capture_output=True, text=True)
        self.assertEqual(legacy.returncode, 0, legacy.stderr)
        self.assertIn('--load-mode none', legacy.stdout)
        self.assertNotIn('--no-host', legacy.stdout)

        tiered = subprocess.run([str(ROOT/'run-server.sh'), '--dry-run'],
                                env={**env, 'DSV4F_RAM_BUDGET_MIB': '163840'}, capture_output=True, text=True)
        self.assertEqual(tiered.returncode, 0, tiered.stderr)
        self.assertIn('RAM budget: 163840 MiB; SSD payload I/O: O_DIRECT', tiered.stdout)
        self.assertIn('--load-mode dio', tiered.stdout)
        self.assertIn('--no-host', tiered.stdout)
