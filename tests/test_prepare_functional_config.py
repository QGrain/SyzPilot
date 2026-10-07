import argparse
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "scripts" / "prepare_functional_config.py"
SPEC = importlib.util.spec_from_file_location("prepare_functional_config", MODULE_PATH)
prepare_config = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(prepare_config)


class PrepareFunctionalConfigTest(unittest.TestCase):
    def test_parse_and_render_distributed_config(self):
        pcs = prepare_config.parse_target_pcs(
            '[For SyzPilot-fuzzer:]\n["0x81234567", "0x89abcdef"]\n'
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            template = Path(temp_dir) / "template.json"
            template_config = json.loads(
                (PROJECT_ROOT / "artifact" / "case_36.manager.cfg").read_text()
            )
            template_config["SyzPilot"].update({
                "cold_start_seed_dir": "/stale/seed/path",
                "cold_start_seed_catalog": "/stale/catalog.json",
                "cold_start_auto_dependencies": True,
                "directed_corpus": {"enable": False, "capacity": 1},
            })
            template.write_text(json.dumps(template_config))
            for case in (21, 36):
                with self.subTest(case=case):
                    args = argparse.Namespace(
                        case=case,
                        template=template,
                        manager_port=39800 + case,
                        run_dir=f"/root/syzpilot-runs/case_{case}",
                        kernel_dir=f"/root/kernels/case_{case}",
                        image="/root/images/image-template/disk.img",
                        ssh_key="/root/images/image-template/disk.id_rsa",
                        syzkaller="/root/fuzzers/SyzPilot-fuzzer",
                        brain_host="10.0.0.10",
                        fuzzer_host="10.0.0.20",
                        report_path=(
                            f"/root/SyzPilot/benchmark/configs/case_{case}.report"
                        ),
                    )
                    config = prepare_config.render_config(args, pcs)

                    self.assertEqual(config["http"], f"0.0.0.0:{39800 + case}")
                    self.assertEqual(
                        config["kernel_obj"], f"/root/kernels/case_{case}"
                    )
                    self.assertEqual(
                        config["syzkaller"], "/root/fuzzers/SyzPilot-fuzzer"
                    )
                    self.assertEqual(config["SyzPilot"]["target_pcs"], pcs)
                    self.assertEqual(
                        config["SyzPilot"]["callback_ip"], "10.0.0.20"
                    )
                    self.assertEqual(
                        config["SyzPilot"]["reach_filter"]["controller"],
                        "10.0.0.10:48000",
                    )
                    self.assertEqual(config["SyzPilot"]["directed_corpus"], {
                        "enable": True,
                        "capacity": 256,
                        "mutation_probability": 0.20,
                        "anchor_preserve_probability": 0.95,
                    })
                    self.assertFalse(
                        config["SyzPilot"]["observe_target_coverage"]
                    )
                    for field in (
                        "cold_start_seed_dir",
                        "cold_start_seed_catalog",
                        "cold_start_auto_dependencies",
                    ):
                        self.assertNotIn(field, config["SyzPilot"])

    def test_case_25_uses_unified_seed_paths(self):
        args = argparse.Namespace(
            case=25,
            template=PROJECT_ROOT / "artifact" / "case_36.manager.cfg",
            manager_port=None,
            run_dir=None,
            kernel_dir=None,
            image="/root/images/image-template/disk.img",
            ssh_key="/root/images/image-template/disk.id_rsa",
            syzkaller="/root/fuzzers/SyzPilot-fuzzer",
            brain_host="10.0.0.10",
            fuzzer_host="10.0.0.20",
            report_path=None,
        )
        config = prepare_config.render_config(args, ["0x81234567"])
        syzpilot = config["SyzPilot"]
        self.assertEqual(config["http"], "0.0.0.0:39825")
        self.assertEqual(config["kernel_obj"], "/root/kernels/case_25")
        self.assertFalse(syzpilot["observe_target_coverage"])
        self.assertEqual(
            syzpilot["cold_start_seed_dir"],
            "/root/fuzzers/SyzPilot-fuzzer/sys/linux/test",
        )
        self.assertEqual(
            syzpilot["cold_start_seed_catalog"],
            "/root/fuzzers/SyzPilot-fuzzer/syz-manager/"
            "syzpilot_seed_catalog_linux_amd64.json",
        )
        self.assertTrue(syzpilot["cold_start_auto_dependencies"])
        self.assertEqual(syzpilot["directed_corpus"], {
            "enable": True,
            "capacity": 256,
            "mutation_probability": 0.20,
            "anchor_preserve_probability": 0.95,
        })

    def test_rejects_duplicate_or_zero_pcs(self):
        for output in (
            '[For SyzPilot-fuzzer:]\n["0x81234567", "0x81234567"]\n',
            '[For SyzPilot-fuzzer:]\n["0x00000000"]\n',
        ):
            with self.subTest(output=output), self.assertRaisesRegex(
                ValueError, "invalid or duplicate"
            ):
                prepare_config.parse_target_pcs(output)


if __name__ == "__main__":
    unittest.main()
