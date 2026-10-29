import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from reasoning import DSV41F_REASONING_BUDGET_MESSAGE as MESSAGE, verify_off_reasoning


class ReasoningOffContractTest(unittest.TestCase):
    def test_forced_message_and_output_cap(self):
        verify_off_reasoning(MESSAGE)
        verify_off_reasoning(MESSAGE[:8], allow_partial=True)
        with self.assertRaises(RuntimeError):
            verify_off_reasoning(MESSAGE[:8])

    def test_reference_compatibility_is_explicit(self):
        verify_off_reasoning(None, allow_empty=True)
        for value in (None, '', 'Let me think.', MESSAGE + 'More thoughts'):
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                verify_off_reasoning(value)
        with self.assertRaises(RuntimeError):
            verify_off_reasoning('Let me think.', allow_empty=True, allow_partial=True)
