"""CPU-only regression tests for startup GPU admission."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "brain"))
from gpu_admission import (
    AdmissionError,
    AdmissionPolicy,
    GpuStatus,
    check_service_ports,
    check_torchserve_pid_file,
    main,
    parse_gpu_status,
    probe_gpus,
    query_gpu_status,
    select_gpu_pair,
    selected_environment,
    validate_controller_command_port,
    validate_cuda_mapping,
    validate_existing_torchserve_config,
    validate_physical_gpu_namespace,
)


class GpuAdmissionTest(unittest.TestCase):
    def test_parse_rejects_malformed_and_duplicate_rows(self):
        row = "0, 00000000:12:00.0, Disabled, 4, 81920, 2"
        for output in ("", "0, N/A, Disabled, 4, 81920, 2",
                       "0, 00000000:12:00.0, Enabled, 4, 81920, 2",
                       "0, 00000000:12:00.0, Disabled, 4, 81920",
                       "0, 00000000:12:00.0, Disabled, 5, 4, 2",
                       row + "\n" + row):
            with self.subTest(output=output):
                with self.assertRaises(AdmissionError):
                    parse_gpu_status(output)

    def test_probe_uses_worst_status_across_stable_samples(self):
        outputs = [
            "0, 00000000:12:00.0, Disabled, 60000, 81920, 5\n"
            "1, 00000000:89:00.0, Disabled, 50000, 81920, 20",
            "0, 00000000:12:00.0, Disabled, 54000, 81920, 30\n"
            "1, 00000000:89:00.0, Disabled, 49000, 81920, 12",
            "0, 00000000:12:00.0, Disabled, 58000, 81920, 8\n"
            "1, 00000000:89:00.0, Disabled, 51000, 81920, 18",
        ]
        run = mock.Mock(side_effect=[
            subprocess.CompletedProcess([], 0, stdout=output, stderr="")
            for output in outputs
        ])
        sleep = mock.Mock()
        actual = probe_gpus(AdmissionPolicy(), run=run, sleep=sleep)
        self.assertEqual(actual[0], GpuStatus(
            0, "00000000:12:00.0", 54000, 81920, 30
        ))
        self.assertEqual(actual[1], GpuStatus(
            1, "00000000:89:00.0", 49000, 81920, 20
        ))
        self.assertEqual(sleep.call_count, 2)
        self.assertEqual(run.call_count, 3)

    def test_single_snapshot_rejects_failed_nvidia_smi(self):
        run = mock.Mock(return_value=subprocess.CompletedProcess(
            [], 9, stdout="", stderr="driver unavailable"
        ))
        with self.assertRaisesRegex(AdmissionError, "driver unavailable"):
            query_gpu_status(run=run)

    def test_direct_namespace_validates_order_identity_and_required_ids(self):
        output = (
            "0, 00000000:12:00.0, Disabled, 60000, 81920, 5\n"
            "1, 00000000:89:00.0, Disabled, 60000, 81920, 7"
        )
        run = mock.Mock(return_value=subprocess.CompletedProcess(
            [], 0, stdout=output, stderr=""
        ))
        cuda_query = mock.Mock(return_value={
            0: "0000:12:00.0", 1: "0000:89:00.0",
        })
        statuses = validate_physical_gpu_namespace(
            ("0", "1"),
            environment={"CUDA_DEVICE_ORDER": "PCI_BUS_ID"},
            run=run,
            cuda_query=cuda_query,
        )
        self.assertEqual(set(statuses), {0, 1})

        with self.assertRaisesRegex(AdmissionError, "PCI_BUS_ID"):
            validate_physical_gpu_namespace(
                ("0",), environment={}, run=run, cuda_query=cuda_query
            )
        with self.assertRaisesRegex(AdmissionError, "unavailable: 2"):
            validate_physical_gpu_namespace(
                ("2",),
                environment={"CUDA_DEVICE_ORDER": "PCI_BUS_ID"},
                run=run,
                cuda_query=cuda_query,
            )
        with self.assertRaisesRegex(AdmissionError, "does not match"):
            validate_physical_gpu_namespace(
                ("0",),
                environment={"CUDA_DEVICE_ORDER": "PCI_BUS_ID"},
                run=run,
                cuda_query=lambda: {
                    0: "0000:89:00.0", 1: "0000:12:00.0",
                },
            )

    def test_selects_distinct_least_busy_eligible_pair(self):
        statuses = {
            0: GpuStatus(0, "00000000:12:00.0", 60000, 81920, 35),
            1: GpuStatus(1, "00000000:89:00.0", 70000, 81920, 10),
            2: GpuStatus(2, "00000000:C1:00.0", 40000, 81920, 70),
        }
        self.assertEqual(select_gpu_pair(statuses, AdmissionPolicy()), (1, 0))

    def test_serving_only_eligible_device_is_reserved_for_serving(self):
        statuses = {
            0: GpuStatus(0, "00000000:12:00.0", 10000, 81920, 5),
            1: GpuStatus(1, "00000000:89:00.0", 60000, 81920, 20),
        }
        policy = AdmissionPolicy(serving_min_free_mib=8000)
        self.assertEqual(select_gpu_pair(statuses, policy), (1, 0))

    def test_refuses_one_card_or_insufficient_memory(self):
        policy = AdmissionPolicy()
        with self.assertRaisesRegex(AdmissionError, "distinct"):
            select_gpu_pair({0: GpuStatus(
                0, "00000000:12:00.0", 60000, 81920, 5
            )}, policy)
        with self.assertRaisesRegex(AdmissionError, "24000 MiB"):
            select_gpu_pair({
                0: GpuStatus(0, "00000000:12:00.0", 10000, 81920, 5),
                1: GpuStatus(1, "00000000:89:00.0", 10000, 81920, 5),
            }, policy)

    def test_cuda_mapping_requires_matching_pci_identity(self):
        statuses = {
            0: GpuStatus(0, "00000000:12:00.0", 60000, 81920, 5),
            1: GpuStatus(1, "00000000:89:00.0", 60000, 81920, 5),
        }
        validate_cuda_mapping(statuses, {
            0: "0000:12:00.0", 1: "0000:89:00.0",
        })
        with self.assertRaisesRegex(AdmissionError, "does not match"):
            validate_cuda_mapping(statuses, {
                0: "0000:89:00.0", 1: "0000:12:00.0",
            })

    def test_port_check_refuses_an_existing_service(self):
        with mock.patch("gpu_admission.socket.socket") as create_socket:
            create_socket.return_value.__enter__.return_value.bind.side_effect = (
                OSError("in use")
            )
            with self.assertRaisesRegex(AdmissionError, "unavailable"):
                check_service_ports()
        with self.assertRaisesRegex(AdmissionError, "differs"):
            validate_controller_command_port(
                ["python", "brain/controller.py", "--port", "48001"], 48000
            )
        with self.assertRaisesRegex(AdmissionError, "differs"):
            validate_controller_command_port(
                ["python", "brain/controller.py"], 48001
            )

    def test_torchserve_pid_file_blocks_launch_even_if_port_is_free(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch("gpu_admission.tempfile.gettempdir",
                            return_value=directory):
                check_torchserve_pid_file()
                (Path(directory) / ".model_server.pid").write_text("123\n")
                with self.assertRaisesRegex(AdmissionError, "PID file"):
                    check_torchserve_pid_file()

    def test_existing_torchserve_config_must_match_checked_ports(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch("gpu_admission.Path.cwd",
                            return_value=Path(directory)):
                validate_existing_torchserve_config()
                (Path(directory) / "config.properties").write_text(
                    "grpc_inference_port=39000\n"
                )
                with self.assertRaisesRegex(AdmissionError, "preflighted"):
                    validate_existing_torchserve_config()
                valid = (
                    "inference_address=http://0.0.0.0:37030\n"
                    "management_address=http://0.0.0.0:37031\n"
                    "metrics_address=http://0.0.0.0:37032\n"
                    "grpc_inference_port=37033\n"
                    "grpc_management_port=37034\n"
                )
                (Path(directory) / "config.properties").write_text(valid)
                validate_existing_torchserve_config()
                (Path(directory) / "config.properties").write_text(
                    valid + "grpc_inference_port: 39000\n"
                )
                with self.assertRaisesRegex(AdmissionError, "syntax"):
                    validate_existing_torchserve_config()
                (Path(directory) / "config.properties").write_text(
                    valid + "grpc_inference_port=39000\n"
                )
                with self.assertRaisesRegex(AdmissionError, "duplicate"):
                    validate_existing_torchserve_config()

    def test_selected_environment_preserves_two_gpu_namespace(self):
        result = selected_environment({}, AdmissionPolicy(), 1, 2)
        self.assertEqual(result["SYZPILOT_TRAINING_GPU_IDS"], "1")
        self.assertEqual(result["SYZPILOT_INFERENCE_GPU_ID"], "2")
        self.assertEqual(result["SYZPILOT_ATTRIBUTION_GPU_ID"], "2")
        self.assertEqual(result["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
        self.assertEqual(result["SYZPILOT_TRAINING_FALLBACK_GPU_IDS"], "")
        with self.assertRaisesRegex(AdmissionError, "manual GPU"):
            selected_environment({"SYZPILOT_TRAINING_GPU_IDS": "0"},
                                 AdmissionPolicy(), 1, 2)
        with self.assertRaisesRegex(AdmissionError, "CUDA_VISIBLE_DEVICES"):
            selected_environment({"CUDA_VISIBLE_DEVICES": "0,1"},
                                 AdmissionPolicy(), 1, 2)
        with self.assertRaisesRegex(AdmissionError, "CUDA_DEVICE_ORDER"):
            selected_environment({"CUDA_DEVICE_ORDER": "FASTEST_FIRST"},
                                 AdmissionPolicy(), 1, 2)

    def test_invalid_policy_is_rejected(self):
        for policy in (AdmissionPolicy(samples=4),
                       AdmissionPolicy(interval_seconds=-1),
                       AdmissionPolicy(training_min_free_mib=0),
                       AdmissionPolicy(serving_max_utilization=101)):
            with self.subTest(policy=policy):
                with self.assertRaises(AdmissionError):
                    policy.validate()

    def test_main_inherits_user_limit_and_passes_selected_gpu_environment(self):
        statuses = {
            0: GpuStatus(0, "00000000:12:00.0", 60000, 81920, 10),
            1: GpuStatus(1, "00000000:89:00.0", 60000, 81920, 15),
        }
        with (
            mock.patch.dict(os.environ, {
                "SYZPILOT_TRAINING_MAX_GPU_UTILIZATION": "20",
            }, clear=True),
            mock.patch("gpu_admission.probe_gpus", return_value=statuses),
            mock.patch("gpu_admission.cuda_device_bus_ids", return_value={
                0: "0000:12:00.0", 1: "0000:89:00.0",
            }),
            mock.patch("gpu_admission.validate_existing_torchserve_config"),
            mock.patch("gpu_admission.check_torchserve_pid_file"),
            mock.patch("gpu_admission.check_service_ports") as ports,
            mock.patch("gpu_admission.os.execvpe",
                       side_effect=OSError("test intercepted")) as execute,
        ):
            result = main(["--samples", "1", "--", "python",
                           "brain/controller.py", "--port", "48000"])
        self.assertEqual(result, 2)
        ports.assert_called_once_with(48000)
        self.assertEqual(
            execute.call_args.args[2][
                "SYZPILOT_TRAINING_MAX_GPU_UTILIZATION"
            ], "20"
        )
        self.assertEqual(
            execute.call_args.args[2]["SYZPILOT_INFERENCE_GPU_ID"], "1"
        )

    def test_main_rejects_without_launch_when_resources_are_insufficient(self):
        statuses = {
            0: GpuStatus(0, "00000000:12:00.0", 10000, 81920, 5),
            1: GpuStatus(1, "00000000:89:00.0", 10000, 81920, 5),
        }
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch("gpu_admission.probe_gpus", return_value=statuses),
            mock.patch("gpu_admission.cuda_device_bus_ids", return_value={
                0: "0000:12:00.0", 1: "0000:89:00.0",
            }),
            mock.patch("gpu_admission.validate_existing_torchserve_config"),
            mock.patch("gpu_admission.check_torchserve_pid_file"),
            mock.patch("gpu_admission.check_service_ports"),
            mock.patch("gpu_admission.os.execvpe") as execute,
        ):
            result = main(["--", "python", "brain/controller.py"])
        self.assertEqual(result, 2)
        execute.assert_not_called()

    def test_main_rejects_existing_torchserve_before_gpu_probe_or_launch(self):
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch("gpu_admission.validate_existing_torchserve_config"),
            mock.patch("gpu_admission.check_torchserve_pid_file",
                       side_effect=AdmissionError("existing PID file")),
            mock.patch("gpu_admission.probe_gpus") as probe,
            mock.patch("gpu_admission.os.execvpe") as execute,
        ):
            result = main(["--", "python", "brain/controller.py"])
        self.assertEqual(result, 2)
        probe.assert_not_called()
        execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
