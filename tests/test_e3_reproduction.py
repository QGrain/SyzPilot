#!/usr/bin/env python3
"""Unit tests for the bounded E3 experiment protocol."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import argparse
import json
import tempfile
import unittest
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = PROJECT_ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


harvester = load_script("harvest_e3_reproduction.py")
runner = load_script("run_e3_reproduction.py")


def record(
    case: int,
    arm: str,
    repetition: int,
    *,
    hits: int,
    executions: int,
    model_active: bool = False,
    crash: bool = False,
) -> dict:
    return {
        "name": f"case{case}_r{repetition}_a1_{arm}",
        "case": case,
        "arm": arm,
        "repetition": repetition,
        "config_errors": [],
        "exit": {"exit_code": 0, "elapsed_seconds": 4 * 3600},
        "final_hits": hits,
        "executions": executions,
        "final_hits_per_execution": hits / executions,
        "target_crash": crash,
        "model_inference_observed": model_active,
        "model_versions": ["v1"] if model_active else [],
    }


def complete_records(case36_hits: list[tuple[int, int]]) -> list[dict]:
    records = []
    for case in (21, 36):
        for repetition in range(1, 4):
            if case == 36:
                blank_hits, full_hits = case36_hits[repetition - 1]
            else:
                blank_hits, full_hits = 0, 1
            records.append(record(
                case,
                "blank",
                repetition,
                hits=blank_hits,
                executions=1_000,
            ))
            records.append(record(
                case,
                "full",
                repetition,
                hits=full_hits,
                executions=100_000,
                model_active=True,
            ))
    return records


class E3HarvesterTest(unittest.TestCase):
    def test_reachability_gate_uses_raw_nh_not_normalized_rate(self):
        result = harvester.summarize(
            complete_records([(10, 20), (10, 20), (0, 1)]), [], []
        )
        pairs = [
            pair for pair in result["paired_runs"] if pair["case"] == 36
        ]
        self.assertTrue(all(pair["full_final_hits_higher"] for pair in pairs))
        self.assertLess(sum(pair["full_rate_higher"] for pair in pairs), 2)
        self.assertTrue(
            result["predeclared_checks"]
            ["case36_reachability_direction_reproduced"]
        )

    def test_higher_rate_does_not_replace_raw_nh(self):
        records = complete_records([(10, 9), (10, 9), (0, 1)])
        for item in records:
            if item["case"] == 36 and item["arm"] == "full":
                item["executions"] = 10
                item["final_hits_per_execution"] = (
                    item["final_hits"] / item["executions"]
                )
        result = harvester.summarize(records, [], [])
        pairs = [
            pair for pair in result["paired_runs"] if pair["case"] == 36
        ]
        self.assertGreaterEqual(sum(pair["full_rate_higher"] for pair in pairs), 2)
        self.assertLess(sum(pair["full_final_hits_higher"] for pair in pairs), 2)
        self.assertFalse(
            result["predeclared_checks"]
            ["case36_reachability_direction_reproduced"]
        )

    def test_blank_and_full_config_boundaries(self):
        manifest = {
            "protocol": {
                "procs": 8,
                "reproduce": False,
                "vm_count": 1,
                "vm_cpu": 2,
                "vm_mem_mib": 4096,
            },
            "target_pcs": {"case_36": ["0x1", "0x2"]},
        }
        common = {
            "procs": 8,
            "reproduce": False,
            "vm": {"count": 1, "cpu": 2, "mem": 4096},
        }
        blank = {
            **common,
            "SyzPilot": {
                "target_pcs": ["0x1", "0x2"],
                "observe_target_coverage": True,
                "reach_filter": {"enable": False},
                "directed_corpus": {"enable": False},
            },
        }
        full = {
            **common,
            "SyzPilot": {
                "target_pcs": ["0x1", "0x2"],
                "observe_target_coverage": False,
                "reach_filter": {"enable": True},
                "directed_corpus": {"enable": True},
                "enable_online_guidance": True,
                "enable_sequence_guidance": True,
                "enable_attribution_guidance": True,
            },
        }
        self.assertEqual(harvester.validate_config(blank, manifest, 36, "blank"), [])
        self.assertEqual(harvester.validate_config(full, manifest, 36, "full"), [])
        blank["SyzPilot"]["reach_filter"]["enable"] = True
        self.assertIn(
            "Blank reach_filter must be disabled",
            harvester.validate_config(blank, manifest, 36, "blank"),
        )


class E3RunnerTest(unittest.TestCase):
    def test_resource_arguments(self):
        with mock.patch.object(
            runner,
            "gpu_inventory",
            return_value=[{"index": "0"}, {"index": "2"}],
        ):
            self.assertEqual(runner.parse_gpu_devices("2,0"), ("2", "0"))
        self.assertEqual(
            runner.parse_cpu_pairs("0-1;2-3;4-5;6-7"),
            ("0-1", "2-3", "4-5", "6-7"),
        )
        self.assertEqual(runner.duration_to_seconds("4h"), 4 * 3600)

    def test_rejects_duplicate_resource_assignments(self):
        with self.assertRaises(ValueError):
            runner.parse_gpu_devices("0,0")
        with self.assertRaises(ValueError):
            runner.parse_cpu_pairs("0-1;0-1;4-5;6-7")

    def test_recovers_only_interrupted_batches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            interrupted = root / "batch_r1_a1.json"
            completed = root / "batch_r2_a1.json"
            interrupted.write_text(json.dumps({
                "status": "launching",
                "runs": [{"supervisor_pid": 123}],
            }))
            completed.write_text(json.dumps({
                "status": "completed",
                "runs": [{"supervisor_pid": 456}],
            }))
            with mock.patch.object(runner, "stop_specs") as stop_specs:
                runner.recover_interrupted_batches(root)
            stop_specs.assert_called_once_with([{"supervisor_pid": 123}])
            self.assertEqual(
                json.loads(interrupted.read_text())["status"], "invalid"
            )
            self.assertEqual(
                json.loads(completed.read_text())["status"], "completed"
            )

    def test_cleanup_removes_only_owned_e3_containers(self):
        with (
            mock.patch.object(runner, "container_exists", return_value=True),
            mock.patch.object(runner, "run") as run,
        ):
            runner.cleanup(argparse.Namespace())
        self.assertEqual(
            [call.args[0] for call in run.call_args_list],
            [
                ["docker", "rm", "-f", runner.FUZZER_CONTAINER],
                ["docker", "rm", "-f", runner.BRAIN_CONTAINER],
            ],
        )


if __name__ == "__main__":
    unittest.main()
