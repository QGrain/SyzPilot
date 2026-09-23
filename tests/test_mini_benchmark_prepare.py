"""Tests for mini-benchmark configuration preparation."""

import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "mini-benchmark" / "prepare_manager.py"
SPEC = importlib.util.spec_from_file_location("mini_benchmark_prepare", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PrepareManagerTests(unittest.TestCase):
    def test_parses_fresh_pcs(self) -> None:
        output = '[For SyzPilot-fuzzer:]\n["0x12345678","0x87654321"]\n'
        self.assertEqual(MODULE.parse_target_pcs(output), ["0x12345678", "0x87654321"])

    def test_rejects_invalid_pcs(self) -> None:
        for output in (
            "no PCs",
            "[For SyzPilot-fuzzer:]\n[]\n",
            '[For SyzPilot-fuzzer:]\n["0x00000000"]\n',
            '[For SyzPilot-fuzzer:]\n["0x12345678","0x12345678"]\n',
        ):
            with self.subTest(output=output), self.assertRaises(ValueError):
                MODULE.parse_target_pcs(output)

    def test_all_cases_use_matching_paths(self) -> None:
        template = json.loads((ROOT / "artifact" / "case_36.manager.cfg").read_text())
        for case_id in MODULE.CASES:
            with self.subTest(case_id=case_id):
                config = MODULE.render_config(case_id, ["0x12345678"], template)
                case = f"case_{case_id}"
                self.assertEqual(config["SyzPilot"]["target_pcs"], ["0x12345678"])
                self.assertIn(case, config["kernel_obj"])
                self.assertIn(case, config["vm"]["kernel"])
                self.assertIn(case, config["SyzPilot"]["report_path"])
                self.assertEqual(config["sshkey"], "/artifact/assets/guest/bullseye.id_rsa")

    def test_case_25_enables_authenticated_generic_seeds_only(self) -> None:
        template = json.loads((ROOT / "artifact" / "case_36.manager.cfg").read_text())
        template["SyzPilot"]["cold_start_seed_dir"] = "/tmp/untrusted"
        template["SyzPilot"]["cold_start_seed_catalog"] = "/tmp/untrusted.json"
        template["SyzPilot"]["cold_start_auto_dependencies"] = True
        template["SyzPilot"]["directed_corpus"] = {"enable": False}
        self.assertIs(template["SyzPilot"]["observe_target_coverage"], True)
        for case_id in MODULE.CASES:
            with self.subTest(case_id=case_id):
                syzpilot = MODULE.render_config(case_id, ["0x12345678"], template)[
                    "SyzPilot"
                ]
                if case_id == 25:
                    self.assertIs(syzpilot["observe_target_coverage"], False)
                    self.assertEqual(
                        syzpilot["cold_start_seed_dir"],
                        "/root/SyzPilot-fuzzer/sys/linux/test",
                    )
                    self.assertEqual(
                        syzpilot["cold_start_seed_catalog"],
                        "/root/SyzPilot-fuzzer/syz-manager/"
                        "syzpilot_seed_catalog_linux_amd64.json",
                    )
                    self.assertIs(syzpilot["cold_start_auto_dependencies"], True)
                    self.assertEqual(
                        syzpilot["directed_corpus"],
                        {
                            "enable": True,
                            "capacity": 256,
                            "mutation_probability": 0.20,
                            "anchor_preserve_probability": 0.95,
                        },
                    )
                else:
                    self.assertIs(syzpilot["observe_target_coverage"], True)
                    self.assertNotIn("cold_start_seed_dir", syzpilot)
                    self.assertNotIn("cold_start_seed_catalog", syzpilot)
                    self.assertNotIn("cold_start_auto_dependencies", syzpilot)
                    self.assertNotIn("directed_corpus", syzpilot)
        self.assertEqual(template["SyzPilot"]["cold_start_seed_dir"], "/tmp/untrusted")
        self.assertEqual(
            template["SyzPilot"]["cold_start_seed_catalog"],
            "/tmp/untrusted.json",
        )
        self.assertIs(template["SyzPilot"]["cold_start_auto_dependencies"], True)
        self.assertEqual(template["SyzPilot"]["directed_corpus"], {"enable": False})
        self.assertIs(template["SyzPilot"]["observe_target_coverage"], True)


if __name__ == "__main__":
    unittest.main()
