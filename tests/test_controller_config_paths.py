"""Regression tests for portable Brain artifact paths."""

import asyncio
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "brain"))

from config import ControllerConfig


class ControllerConfigPathsTest(unittest.TestCase):
    def test_defaults_use_artifact_paths(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            config = ControllerConfig()

        self.assertEqual(
            config.base_model,
            "/opt/syzpilot/models/SyzEncoder_224w_full/best_model/",
        )
        self.assertEqual(
            config.tokenizer,
            "/artifact/assets/models/SyzTokenizer_224w/",
        )
        self.assertEqual(
            config.syzkaller_syslinux,
            "/artifact/assets/syzlang/sys/linux/",
        )
        self.assertEqual(
            config.guidance_report_roots,
            ("/artifact/assets",),
        )
        self.assertEqual(
            config.guidance_kallgraph_roots,
            ("/artifact/assets/kallgraph",),
        )
        self.assertEqual(config.report_paths, {})
        self.assertEqual(config.kallgraph_dirs, {})

    def test_environment_overrides_are_read_per_instance(self):
        variables = {
            "SYZPILOT_BASE_MODEL_PATH": "/artifact/model",
            "SYZPILOT_TOKENIZER_PATH": "/artifact/tokenizer",
            "SYZPILOT_SYZKALLER_SYSLINUX": "/artifact/syzkaller/sys/linux",
            "SYZPILOT_GUIDANCE_REPORT_ROOTS": (
                f"/artifact/reports{os.pathsep} /artifact/other-reports "
            ),
            "SYZPILOT_GUIDANCE_KALLGRAPH_ROOTS": "/artifact/kallgraph",
        }
        with mock.patch.dict(os.environ, variables):
            config = ControllerConfig()
            explicit = ControllerConfig(base_model="/explicit/model")

        self.assertEqual(config.base_model, "/artifact/model")
        self.assertEqual(config.tokenizer, "/artifact/tokenizer")
        self.assertEqual(
            config.syzkaller_syslinux, "/artifact/syzkaller/sys/linux"
        )
        self.assertEqual(
            config.guidance_report_roots,
            ("/artifact/reports", "/artifact/other-reports"),
        )
        self.assertEqual(
            config.guidance_kallgraph_roots, ("/artifact/kallgraph",)
        )
        self.assertEqual(explicit.base_model, "/explicit/model")

    def test_external_roots_do_not_inject_legacy_benchmark_paths(self):
        import controller

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report_root = root / "reports"
            graph_root = root / "graphs"
            report_root.mkdir()
            graph_root.mkdir()
            with mock.patch.dict(os.environ, {
                "SYZPILOT_GUIDANCE_REPORT_ROOTS": str(report_root),
                "SYZPILOT_GUIDANCE_KALLGRAPH_ROOTS": str(graph_root),
            }):
                config = ControllerConfig()
                explicit = ControllerConfig(
                    report_paths={"known": str(report_root / "known.report")},
                    kallgraph_dirs={"known": str(graph_root / "known")},
                )

            self.assertEqual(config.report_paths, {})
            self.assertEqual(config.kallgraph_dirs, {})
            self.assertEqual(
                explicit.report_paths,
                {"known": str(report_root / "known.report")},
            )
            self.assertEqual(
                explicit.kallgraph_dirs,
                {"known": str(graph_root / "known")},
            )

            payload = controller.RegistrationPayload(
                uuid="artifact-fuzzer",
                task_name="kernel BUG in validate_xmit_skb",
                mode="direct",
                host_ip="127.0.0.1",
                http_port=1234,
                target_os="linux",
                target_arch="amd64",
                target_revision="target-r1",
                producer_revision="fuzzer-r1",
                descriptions_mode="manual",
            )
            with mock.patch.object(controller, "config", config):
                self.assertEqual(
                    controller.resolve_guidance_context(payload),
                    ("validate_xmit_skb", "", ""),
                )
                instance = controller.Controller()
                with mock.patch.object(
                    instance,
                    "_alloc_port",
                    side_effect=RuntimeError("reached allocation"),
                ) as allocate_port:
                    with self.assertRaisesRegex(
                        RuntimeError, "reached allocation"
                    ):
                        asyncio.run(instance.register(payload))
                allocate_port.assert_called_once()

    def test_report_analyzer_uses_configured_syzkaller_path(self):
        import controller

        task = SimpleNamespace(
            task_id="test-task",
            report_text="report",
            target_arch="amd64",
            kallgraph_dir="",
        )
        analyzer = mock.Mock()
        analyzer.analyze_report.return_value = []
        with (
            mock.patch.object(
                controller.config,
                "syzkaller_syslinux",
                "/artifact/syzkaller/sys/linux",
            ),
            mock.patch("path_analyzer.PathBasedAnalyzer", return_value=analyzer)
            as analyzer_class,
        ):
            controller.Controller.__new__(controller.Controller)._run_static_analysis(
                task, mock.Mock()
            )

        analyzer_class.assert_called_once_with(
            syzkaller_syslinux_dir="/artifact/syzkaller/sys/linux"
        )


if __name__ == "__main__":
    unittest.main()
