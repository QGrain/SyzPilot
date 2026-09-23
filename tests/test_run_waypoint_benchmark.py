import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from openpyxl import Workbook

from analyzer import run_waypoint_benchmark as runner

class WaypointBenchmarkOrchestratorTests(unittest.TestCase):
    def test_execute_step_rebuilds_when_output_state_changes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output = root / "artifact.txt"
            manifest = {"steps": {}}
            manifest_path = root / "manifest.json"
            command = ["build-artifact"]

            def initial_build(_command, _log_path):
                output.write_text("original", encoding="utf-8")
                return 0

            with mock.patch.object(runner, "run_logged", side_effect=initial_build):
                runner.execute_step(
                    "build",
                    command,
                    [output],
                    manifest,
                    manifest_path,
                    root / "build.log",
                )

            output.write_text("tampered-with-same-schema", encoding="utf-8")

            def rebuild(_command, _log_path):
                output.write_text("rebuilt", encoding="utf-8")
                return 0

            with mock.patch.object(
                runner, "run_logged", side_effect=rebuild
            ) as run_logged:
                runner.execute_step(
                    "build",
                    command,
                    [output],
                    manifest,
                    manifest_path,
                    root / "build.log",
                )

            run_logged.assert_called_once()
            self.assertEqual(output.read_text(encoding="utf-8"), "rebuilt")

    def test_evaluation_reuse_validates_schema_rows_and_readme(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            payload_body = {
                "agentic_extractions": [{"case_id": "1"}],
                "evaluations": [{"chain": {"nodes": [{}]}}],
                "coverage_runs": [{"case_id": 1}],
            }
            evaluation_json = root / "evaluation.json"
            evaluation_json.write_text(
                json.dumps(
                    {"artifact_type": "waypoint_quality_evaluation", **payload_body}
                ),
                encoding="utf-8",
            )
            workbook = Workbook()
            workbook.remove(workbook.active)
            row_sheets = {
                "Cases",
                "Agentic_Extraction",
                "Evaluation",
                "Evaluation_Waypoints",
                "Coverage_Runs",
            }
            for sheet_name, headers in runner.PUBLISHED_WORKBOOK_COLUMNS.items():
                sheet = workbook.create_sheet(sheet_name)
                sheet.append(list(headers))
                if sheet_name in row_sheets:
                    sheet.append([""] * len(headers))
            workbook_path = root / "waypoints_evaluation.xlsx"
            workbook.save(workbook_path)
            readme = root / "README.md"
            readme.write_text(
                runner.workbook_readme_text(workbook, workbook_path.name),
                encoding="utf-8",
            )
            self.assertTrue(
                runner.evaluation_artifacts_are_readable(
                    evaluation_json, workbook_path, readme
                )
            )
            readme.write_text("tampered", encoding="utf-8")
            self.assertFalse(
                runner.evaluation_artifacts_are_readable(
                    evaluation_json, workbook_path, readme
                )
            )

    def test_static_resume_uses_lightweight_input_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            kernel = root / "kernel"
            (kernel / "arch/x86/boot").mkdir(parents=True)
            (kernel / "arch/x86/boot/bzImage").write_bytes(b"kernel")
            (kernel / "vmlinux").write_bytes(b"symbols")
            (kernel / "demo.c").write_text("int demo;\n", encoding="utf-8")
            title = root / "case.title"
            report = root / "case.report"
            title.write_text("BUG in demo\n", encoding="utf-8")
            report.write_text("Call Trace:\n demo+0x1/0x2\n", encoding="utf-8")
            state = {
                "title": runner.file_state(title),
                "report": runner.file_state(report),
                "bzimage": runner.file_state(kernel / "arch/x86/boot/bzImage"),
                "vmlinux_identity": runner.get_vmlinux_cache_identity(str(kernel)),
                "referenced_sources": {"demo.c": runner.file_state(kernel / "demo.c")},
            }
            body = {
                "artifact_type": "script_waypoint_extraction",
                "cases": [
                    {
                        "paths": {
                            "kernel_dir": str(kernel),
                            "title": str(title),
                            "report": str(report),
                        },
                        "input_state": state,
                    }
                ]
            }
            artifact = root / "static.json"
            artifact.write_text(
                json.dumps(body),
                encoding="utf-8",
            )
            self.assertTrue(runner.static_artifact_matches_inputs(artifact))
            report.write_text("changed report with another size\n", encoding="utf-8")
            self.assertFalse(runner.static_artifact_matches_inputs(artifact))

    def test_static_pipeline_gate_rejects_correctness_failure(self):
        stages = {
            key: {"chain": {"waypoints_target_to_entry": ["target@kernel/a.c:1"]}}
            for key, _, _ in runner.STAGES
        }
        body = {
            "artifact_type": "script_waypoint_extraction",
            "cases": [
                {
                    "case_id": "1",
                    "status": "ok",
                    "stages": stages,
                    "validations": [
                        {
                            "check": "final_pcs_resolve_nonzero",
                            "severity": "error",
                            "passed": False,
                        }
                    ],
                }
            ]
        }
        payload = body
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "static.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "correctness validation"):
                runner.validate_static_artifact_for_pipeline(path)

    def test_completed_agent_output_state_mismatch_forces_full_rebuild(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output = root / "agent.jsonl"
            output.write_text("original", encoding="utf-8")
            manifest = {
                "steps": {
                    "agent": {
                        "status": "complete",
                        "step_identity": {
                            "command": ["agent-command"],
                            "inputs": {},
                            "input_values": {},
                        },
                        "output_state": runner.file_state(output),
                    }
                }
            }
            output.write_text("tampered-longer", encoding="utf-8")
            manifest_path = root / "manifest.json"
            runner.write_json_atomic(manifest_path, manifest)

            def rebuild(command, _log_path):
                self.assertNotIn("--resume", command)
                self.assertFalse(output.exists())
                output.write_text("fresh", encoding="utf-8")
                return 0

            with mock.patch.object(runner, "run_logged", side_effect=rebuild):
                runner.execute_agent_step(
                    "agent",
                    ["agent-command"],
                    output,
                    0,
                    manifest,
                    manifest_path,
                    root / "agent.log",
                )

            self.assertEqual(output.read_text(encoding="utf-8"), "fresh")
            self.assertEqual(manifest["steps"]["agent"]["status"], "complete")
            self.assertEqual(
                manifest["steps"]["agent"]["output_state"],
                runner.file_state(output),
            )

    def test_completed_agent_step_reuses_matching_identity(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output = root / "agent.jsonl"
            output.write_text("complete", encoding="utf-8")
            identity = {
                "command": ["agent-command"],
                "inputs": {},
                "input_values": {},
            }
            manifest = {
                "steps": {
                    "agent": {
                        "status": "complete",
                        "step_identity": identity,
                        "output_state": runner.file_state(output),
                    }
                }
            }
            manifest_path = root / "manifest.json"
            with mock.patch.object(runner, "run_logged") as run_logged:
                runner.execute_agent_step(
                    "agent",
                    ["agent-command"],
                    output,
                    0,
                    manifest,
                    manifest_path,
                    root / "agent.log",
                )
            run_logged.assert_not_called()
            self.assertEqual(manifest["steps"]["agent"]["status"], "reused")

    def test_interrupted_agent_step_preserves_partial_resume_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output = root / "agent.jsonl"
            output.write_text("baseline", encoding="utf-8")
            manifest = {
                "steps": {
                    "agent": {
                        "status": "complete",
                    }
                }
            }
            manifest_path = root / "manifest.json"
            runner.write_json_atomic(manifest_path, manifest)

            def interrupt(command, _log_path):
                output.write_text("partial", encoding="utf-8")
                raise KeyboardInterrupt

            with mock.patch.object(runner, "run_logged", side_effect=interrupt):
                with self.assertRaises(KeyboardInterrupt):
                    runner.execute_agent_step(
                        "agent",
                        ["agent-command"],
                        output,
                        0,
                        manifest,
                        manifest_path,
                        root / "agent.log",
                    )
            self.assertEqual(manifest["steps"]["agent"]["status"], "running")

            def resume(command, _log_path):
                self.assertEqual(output.read_text(encoding="utf-8"), "partial")
                self.assertIn("--resume", command)
                output.write_text("complete", encoding="utf-8")
                return 0

            with mock.patch.object(runner, "run_logged", side_effect=resume):
                runner.execute_agent_step(
                    "agent",
                    ["agent-command"],
                    output,
                    0,
                    manifest,
                    manifest_path,
                    root / "agent.log",
                )
            self.assertEqual(manifest["steps"]["agent"]["status"], "complete")
            self.assertEqual(
                manifest["steps"]["agent"]["attempts"][0]["status"],
                "interrupted",
            )

    def test_resumed_agent_step_keeps_cumulative_failure_budget(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output = root / "agent.jsonl"
            output.write_text("partial", encoding="utf-8")
            manifest = {
                "steps": {
                    "agent": {
                        "status": "running",
                        "step_identity": {
                            "command": ["agent-command"],
                            "inputs": {},
                            "input_values": {},
                        },
                        "attempts": [
                            {"attempt": 1, "returncode": 1},
                            {"attempt": 2, "status": "running"},
                        ],
                    }
                }
            }
            manifest_path = root / "manifest.json"
            runner.write_json_atomic(manifest_path, manifest)
            with mock.patch.object(
                runner, "run_logged", return_value=1
            ) as run_logged:
                with self.assertRaisesRegex(
                    RuntimeError, "after 3 failed attempts"
                ):
                    runner.execute_agent_step(
                        "agent",
                        ["agent-command"],
                        output,
                        2,
                        manifest,
                        manifest_path,
                        root / "agent.log",
                    )
            self.assertEqual(run_logged.call_count, 2)
            self.assertEqual(
                manifest["steps"]["agent"]["attempts"][1]["status"],
                "interrupted",
            )
            self.assertEqual(
                [item["attempt"] for item in manifest["steps"]["agent"]["attempts"]],
                [1, 2, 3, 4],
            )


if __name__ == "__main__":
    unittest.main()
