import argparse
import importlib.util
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
            template.write_text(
                (PROJECT_ROOT / "artifact" / "case_36.manager.cfg").read_text()
            )
            args = argparse.Namespace(
                case=36,
                template=template,
                manager_port=39836,
                run_dir="/root/syzpilot-runs/case_36",
                kernel_dir="/root/kernels/case_36",
                image="/root/images/image-template/disk.img",
                ssh_key="/root/images/image-template/disk.id_rsa",
                syzkaller="/root/fuzzers/SyzPilot-fuzzer",
                brain_host="10.0.0.10",
                fuzzer_host="10.0.0.20",
                report_path="/root/SyzPilot/benchmark/configs/case_36.report",
            )
            config = prepare_config.render_config(args, pcs)

        self.assertEqual(config["http"], "0.0.0.0:39836")
        self.assertEqual(config["kernel_obj"], "/root/kernels/case_36")
        self.assertEqual(config["syzkaller"], "/root/fuzzers/SyzPilot-fuzzer")
        self.assertEqual(config["SyzPilot"]["target_pcs"], pcs)
        self.assertEqual(config["SyzPilot"]["callback_ip"], "10.0.0.20")
        self.assertEqual(
            config["SyzPilot"]["reach_filter"]["controller"],
            "10.0.0.10:48000",
        )

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
        self.assertTrue(syzpilot["directed_corpus"]["enable"])

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
