"""Tests for the source-only artifact's dynamic target-PC preparation."""

import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "artifact" / "prepare_case36_config.py"
SPEC = importlib.util.spec_from_file_location("prepare_case36_config", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PrepareCase36ConfigTests(unittest.TestCase):
    def test_reads_fuzzer_ready_pcs(self) -> None:
        output = 'waypoints\n[For SyzPilot-fuzzer:]\n["0x86e9f994","0x8829941c"]\n'
        self.assertEqual(
            MODULE.read_target_pcs(output), ["0x86e9f994", "0x8829941c"]
        )

    def test_rejects_invalid_pcs(self) -> None:
        for output in (
            "no target PCs",
            "[For SyzPilot-fuzzer:]\n[]\n",
            '[For SyzPilot-fuzzer:]\n["0x00000000"]\n',
            '[For SyzPilot-fuzzer:]\n["0x86e9f994","0x86e9f994"]\n',
            '[For SyzPilot-fuzzer:]\n["not-a-pc"]\n',
        ):
            with self.subTest(output=output), self.assertRaises(ValueError):
                MODULE.read_target_pcs(output)


if __name__ == "__main__":
    unittest.main()
