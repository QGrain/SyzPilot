import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from openpyxl import Workbook

from analyzer import evaluate_waypoint_quality as quality


class WaypointQualityEvaluationTests(unittest.TestCase):
    def test_quality_module_help_works_in_clean_subprocess(self):
        result = subprocess.run(
            [sys.executable, "-m", "analyzer.evaluate_waypoint_quality", "--help"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def make_coverage_run(
        self,
        root: Path,
        calls_text: str,
        extra_text: str = "",
        **overrides,
    ):
        run_idx = int(overrides.get("run_idx", 1))
        run_dir = root / f"run_{run_idx:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        calls = run_dir / "calls_only.union.pc64"
        extra = run_dir / "extra.union.pc64"
        calls.write_text(calls_text, encoding="utf-8")
        extra.write_text(extra_text, encoding="utf-8")
        row = {
            "run_idx": run_idx,
            "preflight_status": "OK",
            "coverage_status": "COMPLETE",
            "kaslr_status": "disabled",
            "poc_fidelity_status": "EXACT",
            "calls_only_union_pc64": str(calls.relative_to(root)),
            "extra_union_pc64": str(extra.relative_to(root)),
            "calls_only_pc_count": len(quality.read_pc_file(calls)),
            "extra_pc_count": len(quality.read_pc_file(extra)),
            "result_dir": str(run_dir.relative_to(root)),
        }
        row.update(overrides)
        return row

    def test_dynamic_score_uses_ratio_and_target_hit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            runs = [
                self.make_coverage_run(
                    root,
                    "0xffffffff81000005\n0xffffffff81000025\n",
                    target_reproduced=False,
                )
            ]
            evidence = quality.coverage_evidence(
                root,
                runs,
                [
                    "0xffffffff8100000a",
                    "0xffffffff8100001a",
                    "0xffffffff8100002a",
                ],
            )

        self.assertEqual(evidence["Waypoint Hit Count"], 2)
        self.assertEqual(evidence["Target Hit"], True)
        self.assertEqual(evidence["Dynamic Coverage Score"], 70.3)
        self.assertEqual(
            evidence["Dynamic Evidence Confidence"],
            "non-reproducing lower-bound evidence",
        )
        self.assertEqual(evidence["PoC Evaluation Status"], "EVALUATED")
        self.assertEqual(
            json.loads(evidence["Coverage Comparison PCs64 (target->entry)"]),
            [
                "0xffffffff81000005",
                "0xffffffff81000015",
                "0xffffffff81000025",
            ],
        )

    def test_coverfile_comparison_does_not_use_unadjusted_fuzzer_pc(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            evidence = quality.coverage_evidence(
                root,
                [
                    self.make_coverage_run(
                        root, "0xffffffff8100000a\n"
                    )
                ],
                ["0xffffffff8100000a"],
            )
        self.assertEqual(evidence["Waypoint Hit Count"], 0)

    def test_extra_only_coverage_is_not_counted_as_fuzzer_reachability(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            evidence = quality.coverage_evidence(
                root,
                [
                    self.make_coverage_run(
                        root, "", "0xffffffff81000005\n"
                    )
                ],
                ["0xffffffff8100000a"],
            )
        self.assertEqual(evidence["Waypoint Hit Count"], 0)
        self.assertEqual(
            json.loads(evidence["Per-Waypoint Extra-only Hits (target->entry)"]),
            [True],
        )
        self.assertEqual(evidence["PoC Evaluation Status"], "EVALUATED_EXTRA_ONLY")
        self.assertEqual(evidence["Dynamic Coverage Score"], 5.0)

    def test_dynamic_score_rewards_any_hit_and_longer_full_chains(self):
        zero = quality.dynamic_coverage_score(0, 10, False)
        one = quality.dynamic_coverage_score(1, 10, False)
        short_full = quality.dynamic_coverage_score(3, 3, True)
        long_full = quality.dynamic_coverage_score(10, 10, True)

        self.assertGreater(one, zero)
        self.assertGreater(long_full, short_full)
        self.assertEqual(zero, 5.0)
        self.assertEqual(short_full, 85.0)
        self.assertEqual(long_full, 93.1)
        self.assertGreater(
            quality.dynamic_coverage_score(20, 20, True), long_full
        )

    def test_removed_configured_target_is_not_reassigned_to_next_node(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            evidence = quality.coverage_evidence(
                root,
                [self.make_coverage_run(root, "0xffffffff81000015\n")],
                ["0xffffffff8100001a"],
                configured_target_index=None,
            )
        self.assertFalse(evidence["Target Hit"])
        self.assertLessEqual(evidence["Dynamic Coverage Score"], 75.0)

    def test_target_miss_is_capped_below_single_target_hit(self):
        long_target_miss = quality.dynamic_coverage_score(9, 10, False)
        single_target_hit = quality.dynamic_coverage_score(1, 1, True)

        self.assertLess(long_target_miss, single_target_hit)
        self.assertLessEqual(long_target_miss, 75.0)

    def test_unknown_kaslr_run_does_not_block_disabled_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            runs = [
                {
                    "run_idx": 1,
                    "preflight_status": "OK",
                    "coverage_status": "UNAVAILABLE",
                    "kaslr_status": "unknown",
                },
                self.make_coverage_run(
                    root, "0xffffffff81000005\n", run_idx=2
                ),
            ]
            evidence = quality.coverage_evidence(
                root, runs, ["0xffffffff8100000a"]
            )
        self.assertEqual(evidence["PoC Evaluation Status"], "EVALUATED")
        self.assertEqual(evidence["Comparable Coverage Run Count"], 1)
        self.assertEqual(evidence["Waypoint Hit Count"], 1)

    def test_disabled_run_with_invalid_coverage_is_not_reported_as_kaslr_block(self):
        evidence = quality.coverage_evidence(
            Path("/unused"),
            [
                {
                    "run_idx": 1,
                    "preflight_status": "OK",
                    "coverage_status": "INVALID",
                    "kaslr_status": "disabled",
                }
            ],
            ["0xffffffff8100000a"],
        )
        self.assertEqual(
            evidence["PoC Evaluation Status"], "NO_USABLE_COVERAGE_RUN"
        )

    def test_setup_failure_is_not_reported_as_kaslr_block(self):
        evidence = quality.coverage_evidence(
            Path("/unused"),
            [
                {
                    "run_idx": 1,
                    "preflight_status": "OK",
                    "execution_status": "SETUP_ERROR",
                    "coverage_status": "UNAVAILABLE",
                    "kaslr_status": "unknown",
                }
            ],
            ["0xffffffff8100000a"],
        )
        self.assertEqual(
            evidence["PoC Evaluation Status"], "NO_USABLE_COVERAGE_RUN"
        )

    def test_missing_union_file_blocks_score(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            row = self.make_coverage_run(
                root, "0xffffffff81000005\n"
            )
            (root / row["calls_only_union_pc64"]).unlink()
            evidence = quality.coverage_evidence(
                root, [row], ["0xffffffff8100000a"]
            )
        self.assertEqual(
            evidence["PoC Evaluation Status"], "BLOCKED_ARTIFACT_MISMATCH"
        )
        self.assertIn(
            "missing coverage artifact",
            evidence["Artifact Compatibility Errors"],
        )

    def test_artifact_mismatch_blocks_score(self):
        evidence = quality.coverage_evidence(
            Path("/unused"),
            [
                {
                    "run_idx": 1,
                    "preflight_status": "OK",
                    "coverage_status": "COMPLETE",
                    "kaslr_status": "disabled",
                }
            ],
            ["0xffffffff8100000a"],
            compatibility_errors=["vmlinux hash mismatch"],
        )
        self.assertEqual(
            evidence["PoC Evaluation Status"], "BLOCKED_ARTIFACT_MISMATCH"
        )
        self.assertIsNone(evidence["Dynamic Coverage Score"])

    def test_coverage_compatibility_uses_semantics_not_kernel_identity(self):
        row = {
            "run_idx": 1,
            "preflight_status": "OK",
            "target_arch": "amd64",
            "coverage_pc_semantics_id": quality.COVERAGE_PC_SEMANTICS_ID,
            "vmlinux_identity": "deliberately-stale",
        }
        manifest = {
            "target_arch": "amd64",
            "coverage_pc_semantics_id": quality.COVERAGE_PC_SEMANTICS_ID,
        }
        self.assertEqual(
            quality.coverage_compatibility_errors("/unused", [row], manifest),
            [],
        )
        row["coverage_pc_semantics_id"] = "different-semantics"
        errors = quality.coverage_compatibility_errors(
            "/unused", [row], manifest
        )
        self.assertIn("run1 coverage PC semantics mismatch", errors)

    def test_evaluated_pc_gate_rejects_zero_and_duplicate_pc32(self):
        with self.assertRaisesRegex(ValueError, "zero evaluated PCs"):
            quality.validate_evaluated_pcs("1", ["a"], ["0x0"], [""])
        with self.assertRaisesRegex(ValueError, "duplicate evaluated PCs"):
            quality.validate_evaluated_pcs(
                "1",
                ["a", "b"],
                ["0xffffffff81000005", "0xffffffff81000005"],
                ["", ""],
            )

    def test_evaluated_pc_gate_allows_empty_only_for_missing_input(self):
        with self.assertRaisesRegex(ValueError, "empty evaluated chain"):
            quality.validate_evaluated_pcs("1", [], [], [])
        quality.validate_evaluated_pcs("62", [], [], [], allow_empty=True)

    def test_kaslr_blocks_direct_comparison(self):
        evidence = quality.coverage_evidence(
            Path("/unused"),
            [
                {
                    "run_idx": 1,
                    "preflight_status": "OK",
                    "coverage_status": "COMPLETE",
                    "kaslr_status": "enabled:0x100000",
                }
            ],
            ["0xffffffff81000005"],
        )
        self.assertEqual(
            evidence["PoC Evaluation Status"], "BLOCKED_KASLR_NOT_DISABLED"
        )
        self.assertIsNone(evidence["Dynamic Coverage Score"])

    def test_review_loader_rejects_duplicate_case(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "review.json"
            entry = {
                "case_id": 1,
                "extraction_type": "script extracted",
                "agentic_quality_score": 90,
            }
            path.write_text(json.dumps({"cases": [entry, entry]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate review"):
                quality.load_reviews([path])

    def test_review_loader_accepts_reviews_envelope(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "review.json"
            entry = {
                "case_id": 1,
                "extraction_type": "script extracted",
                "agentic_quality_score": 90,
            }
            path.write_text(
                json.dumps({"schema_version": "1.0", "reviews": [entry]}),
                encoding="utf-8",
            )
            reviews = quality.load_reviews([path])
        self.assertEqual(reviews, {"1": entry})

    def test_final_pc64_loader_preserves_target_order(self):
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "Waypoints_Long"
        sheet.append(["Case ID", "Stage", "Index target->entry", "PC64"])
        sheet.append(["1", "outlier_removal", 1, "0xffffffff81000015"])
        sheet.append(["1", "outlier_removal", 0, "0xffffffff81000005"])
        self.assertEqual(
            quality.load_final_pc64(workbook),
            {"1": ["0xffffffff81000005", "0xffffffff81000015"]},
        )


if __name__ == "__main__":
    unittest.main()
