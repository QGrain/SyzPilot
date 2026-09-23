"""Regression tests for ordered SyzPilot target PC result collection."""

import json
import tempfile
import unittest
from pathlib import Path

from experiments.collect_syzkaller_results import (
    HOURS_24,
    HOURS_48,
    parse_bench_metrics,
    reachability_value,
    read_run_config,
)


class CollectSyzkallerResultsTest(unittest.TestCase):
    def test_read_run_config_uses_last_pc_as_final_target(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cfg_path = Path(temp_dir) / "run.cfg"
            cfg_path.write_text(
                json.dumps({
                    "SyzPilot": {
                        "task_name": "target-only-case",
                        "target_pcs": ["0x11111111", "0x22222222", "0x33333333"],
                    }
                }),
                encoding="utf-8",
            )

            task_name, target_pc = read_run_config(cfg_path)

            self.assertEqual(task_name, "target-only-case")
            self.assertEqual(target_pc, "0x33333333")

    def test_reachability_value_matches_strict_low_32_bits(self):
        obj = {"reachability 0x12345678": 9}

        self.assertEqual(
            reachability_value(obj, "0xffffffff12345678"),
            9,
        )
        self.assertIsNone(reachability_value(obj, "0x87654321"))
        self.assertIsNone(
            reachability_value({"reachability 0x11": 7}, "0x1")
        )

    def test_metrics_do_not_substitute_an_earlier_waypoint_for_target(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            bench_path = Path(temp_dir) / "bench.json"
            bench_path.write_text(
                json.dumps({
                    "uptime": 60,
                    "reachability 0x11111111": 7,
                    "reachability 0x33333333": 0,
                }),
                encoding="utf-8",
            )

            hit_24, hit_48, tth_24, tth_48 = parse_bench_metrics(
                bench_path,
                "0x33333333",
            )

            self.assertEqual((hit_24, hit_48), (0, 0))
            self.assertEqual((tth_24, tth_48), (HOURS_24, HOURS_48))

    def test_metrics_reject_log_without_final_target_stat(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            bench_path = Path(temp_dir) / "bench.json"
            bench_path.write_text(
                json.dumps({
                    "uptime": 60,
                    "reachability 0x11111111": 7,
                }),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "no reachability statistic for final target 0x33333333",
            ):
                parse_bench_metrics(bench_path, "0x33333333")


if __name__ == "__main__":
    unittest.main()
