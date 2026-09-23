"""TorchServe process-lifecycle regression tests."""

import os
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import psutil


REPO_ROOT = Path(__file__).resolve().parents[1]
BRAIN_DIR = REPO_ROOT / "brain"
sys.path.insert(0, str(BRAIN_DIR))

from ts_operators import ServeOperator


class TorchServeOperatorTest(unittest.TestCase):
    @staticmethod
    def configured_operator(temp_dir, **kwargs):
        operator = ServeOperator(temp_dir, **kwargs)
        operator.cwd = temp_dir
        return operator, operator.create_config()

    def test_register_model_defaults_to_low_latency_batch_delay(self):
        response = mock.Mock()
        with tempfile.TemporaryDirectory() as temp_dir:
            operator = ServeOperator(temp_dir, disable_auth=True)
            with mock.patch(
                    "ts_operators.api_request", return_value=response) as request:
                actual = operator.register_model("model")

        self.assertIs(actual, response)
        request.assert_called_once_with(
            "http://localhost",
            operator.management_port,
            "/models",
            {
                "url": "model.mar",
                "initial_workers": 2,
                "batch_size": 16,
                "max_batch_delay": 10,
                "synchronous": False,
            },
            None,
            "POST",
        )

    def test_public_ownership_and_readiness_reflect_owned_process(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            operator = ServeOperator(temp_dir, disable_auth=True)
            self.assertFalse(operator.has_owned_service)
            self.assertFalse(operator.is_service_ready())
            process = mock.sentinel.process
            operator._owned_process = process
            with mock.patch.object(
                    operator, "_management_api_is_ready",
                    return_value=True) as probe:
                self.assertTrue(operator.has_owned_service)
                self.assertTrue(operator.is_service_ready())
        probe.assert_called_once_with(process)

    def test_failed_initial_cleanup_retains_public_ownership(self):
        launcher = mock.Mock(pid=1234)
        with tempfile.TemporaryDirectory() as temp_dir:
            operator, config = self.configured_operator(
                temp_dir, disable_auth=True
            )
            runtime_dir = os.path.join(temp_dir, "runtime")
            os.mkdir(runtime_dir)
            with (
                mock.patch("ts_operators.subprocess.Popen",
                           return_value=launcher),
                mock.patch("ts_operators.tempfile.mkdtemp",
                           return_value=runtime_dir),
                mock.patch.object(operator, "_wait_for_management_api",
                                  return_value=False),
                mock.patch.object(
                    operator, "stop_service",
                    side_effect=RuntimeError("cleanup not confirmed"),
                ),
                self.assertRaisesRegex(RuntimeError, "cleanup not confirmed"),
            ):
                operator.start_service(config)

            self.assertTrue(operator.has_owned_service)
            self.assertIs(operator._owned_process, launcher)

    def test_management_readiness_requires_torchserve_models_payload(self):
        responses = [
            mock.Mock(status_code=404),
            mock.Mock(status_code=200),
            mock.Mock(status_code=200),
        ]
        responses[0].json.return_value = {"error": "not found"}
        responses[1].json.return_value = {"service": "unrelated"}
        responses[2].json.return_value = {"models": []}
        with tempfile.TemporaryDirectory() as temp_dir:
            operator = ServeOperator(temp_dir, disable_auth=True)
            with (
                mock.patch(
                    "ts_operators.requests.get", side_effect=responses
                ) as request,
                mock.patch("ts_operators.sleep"),
            ):
                ready = operator._wait_for_management_api(timeout=1)

        self.assertTrue(ready)
        self.assertEqual(request.call_count, 3)

    def test_management_readiness_rejects_unrelated_http_service(self):
        response = mock.Mock(status_code=200)
        response.json.return_value = {"status": "ok"}
        with tempfile.TemporaryDirectory() as temp_dir:
            operator = ServeOperator(temp_dir, disable_auth=True)
            with (
                mock.patch("ts_operators.requests.get", return_value=response),
                mock.patch("ts_operators.monotonic", side_effect=[0, 2]),
                mock.patch("ts_operators.sleep"),
            ):
                ready = operator._wait_for_management_api(timeout=1)

        self.assertFalse(ready)

    def test_management_readiness_uses_management_token_when_auth_enabled(self):
        response = mock.Mock(status_code=200)
        response.json.return_value = {"models": []}
        with tempfile.TemporaryDirectory() as temp_dir:
            operator = ServeOperator(temp_dir, disable_auth=False)
            with (
                mock.patch.object(
                    operator,
                    "_ServeOperator__read_key_file",
                    return_value=("management-token", "inference", "api"),
                ),
                mock.patch(
                    "ts_operators.requests.get", return_value=response
                ) as request,
            ):
                ready = operator._wait_for_management_api(timeout=1)

        self.assertTrue(ready)
        request.assert_called_once_with(
            f"http://127.0.0.1:{operator.management_port}/models",
            headers={"Authorization": "Bearer management-token"},
            timeout=0.5,
        )

    def test_start_service_keeps_owned_foreground_process(self):
        launcher = mock.Mock(pid=1234)
        launcher.poll.return_value = None
        with tempfile.TemporaryDirectory() as temp_dir:
            operator, config = self.configured_operator(
                temp_dir, disable_auth=True
            )
            with (
                mock.patch("ts_operators.subprocess.Popen", return_value=launcher),
                mock.patch("ts_operators.tempfile.mkdtemp", return_value=temp_dir),
                mock.patch.object(
                    operator, "_wait_for_management_api", return_value=True
                ) as ready,
                mock.patch.object(operator, "_owned_server_pid",
                                  return_value=os.getpid()),
                mock.patch("ts_operators.os.killpg") as killpg,
            ):
                result = operator.start_service(config)
                self.assertIs(result, launcher)
                self.assertIs(operator._owned_process, launcher)
                operator.stop_service()

        ready.assert_called_once_with(timeout=30, process=launcher)
        killpg.assert_called_once_with(1234, signal.SIGTERM)

    def test_start_service_restricts_torchserve_to_inference_gpu(self):
        launcher = mock.Mock(pid=1234)
        launcher.poll.return_value = None
        with tempfile.TemporaryDirectory() as temp_dir:
            operator, config = self.configured_operator(
                temp_dir, disable_auth=True, inference_gpu_id="1"
            )
            with (
                mock.patch.dict(
                    "ts_operators.os.environ",
                    {"CUDA_VISIBLE_DEVICES": "0,1,2",
                     "TS_CONFIG_FILE": "unvalidated.properties"},
                    clear=False,
                ),
                mock.patch(
                    "ts_operators.subprocess.Popen", return_value=launcher
                ) as popen,
                mock.patch("ts_operators.tempfile.mkdtemp", return_value=temp_dir),
                mock.patch.object(
                    operator, "_wait_for_management_api", return_value=True
                ),
                mock.patch.object(operator, "_owned_server_pid",
                                  return_value=os.getpid()),
                mock.patch("ts_operators.os.killpg"),
            ):
                result = operator.start_service(
                    config, models="named=archive.mar"
                )
                command = popen.call_args.args[0]
                private_config = command[command.index("--ts-config") + 1]
                original_config = Path(private_config).read_text()
                Path(config).write_text(
                    "management_address=http://0.0.0.0:9999\n"
                )
                self.assertEqual(Path(private_config).read_text(),
                                 original_config)
                self.assertEqual(
                    os.environ["CUDA_VISIBLE_DEVICES"], "0,1,2"
                )
                self.assertEqual(
                    os.environ["TS_CONFIG_FILE"], "unvalidated.properties"
                )
                operator.stop_service()

        self.assertIs(result, launcher)
        child_environment = popen.call_args.kwargs["env"]
        self.assertEqual(child_environment["CUDA_VISIBLE_DEVICES"], "1")
        self.assertEqual(child_environment["TMPDIR"], temp_dir)
        self.assertNotIn("TS_CONFIG_FILE", child_environment)
        self.assertIn("--foreground", popen.call_args.args[0])
        self.assertIn("named=archive.mar", popen.call_args.args[0])

    def test_readiness_rejects_foreign_management_listener(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            operator = ServeOperator(temp_dir, disable_auth=True)
            operator._runtime_dir = temp_dir
            (Path(temp_dir) / ".model_server.pid").write_text("4321\n")
            launcher = mock.Mock(pid=1234)
            launcher.poll.return_value = None
            server = mock.Mock(pid=4321)
            server.is_running.return_value = True
            server.cmdline.return_value = [
                "java", "org.pytorch.serve.ModelServer"
            ]
            server.net_connections.return_value = [
                mock.Mock(
                    laddr=mock.Mock(port=operator.inference_port),
                    status="LISTEN",
                ),
            ]
            with (
                mock.patch("ts_operators.psutil.Process", return_value=server),
                mock.patch("ts_operators.os.getpgid", return_value=1234),
            ):
                self.assertIsNone(operator._owned_server_pid(launcher))
                server.net_connections.return_value = [
                    mock.Mock(laddr=mock.Mock(port=port), status="LISTEN")
                    for port in (
                        operator.inference_port,
                        operator.management_port,
                        operator.metrics_port,
                        operator.grpc_inference_port,
                        operator.grpc_management_port,
                    )
                ]
                self.assertEqual(operator._owned_server_pid(launcher), 4321)

    def test_stop_waits_for_and_escalates_owned_process_group(self):
        program = (
            "import signal, time; "
            "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "print('ready', flush=True); time.sleep(60)"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            process = subprocess.Popen(
                [sys.executable, "-c", program],
                start_new_session=True,
                stdout=subprocess.PIPE,
                text=True,
            )
            try:
                self.assertEqual(process.stdout.readline().strip(), "ready")
                operator = ServeOperator(temp_dir, disable_auth=True)
                operator._runtime_dir = temp_dir
                operator._owned_process = process
                with (
                    mock.patch("ts_operators.TERM_GRACE_SECONDS", 0.2),
                    mock.patch("ts_operators.KILL_GRACE_SECONDS", 2),
                ):
                    operator.stop_service()
                self.assertIsNotNone(process.poll())
                self.assertIsNone(operator._owned_process)
                self.assertFalse(operator._live_group_members(process.pid))
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=2)
                process.stdout.close()

    def test_stop_discovers_unready_java_after_launcher_exits(self):
        child_program = (
            "import signal, time; "
            "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "print('ready', flush=True); time.sleep(60)"
        )
        launcher_program = (
            "import subprocess, sys; "
            "child=subprocess.Popen([sys.executable, '-c', sys.argv[1], "
            "'org.pytorch.serve.ModelServer'], stdout=subprocess.PIPE, "
            "text=True); "
            "child.stdout.readline(); print(child.pid, flush=True)"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            launcher = subprocess.Popen(
                [sys.executable, "-c", launcher_program, child_program],
                start_new_session=True,
                stdout=subprocess.PIPE,
                text=True,
            )
            java_pid = int(launcher.stdout.readline().strip())
            java = psutil.Process(java_pid)
            launcher.wait(timeout=3)
            try:
                operator = ServeOperator(temp_dir, disable_auth=True)
                operator._runtime_dir = temp_dir
                operator._owned_process = launcher
                (Path(temp_dir) / ".model_server.pid").write_text(
                    f"{java_pid}\n"
                )
                with (
                    mock.patch("ts_operators.TERM_GRACE_SECONDS", 0.2),
                    mock.patch("ts_operators.KILL_GRACE_SECONDS", 2),
                ):
                    operator.stop_service()
                self.assertIsNone(operator._owned_process)
                self.assertFalse(operator._live_group_members(launcher.pid))
            finally:
                try:
                    if (java.is_running() and
                            java.status() != psutil.STATUS_ZOMBIE):
                        java.kill()
                except psutil.Error:
                    pass
                launcher.stdout.close()

    def test_serve_model_uses_owned_startup(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            operator = ServeOperator(
                temp_dir, disable_auth=True, inference_gpu_id="1"
            )
            with mock.patch.object(
                operator, "start_service", return_value=mock.sentinel.process
            ) as start:
                result = operator.serve_model(
                    "model", "archive", "config.properties"
                )
        self.assertIs(result, mock.sentinel.process)
        start.assert_called_once_with(
            "config.properties", models="model=archive.mar"
        )

    def test_start_service_rejects_failed_launcher(self):
        launcher = mock.Mock(pid=1234)
        launcher.poll.return_value = 1
        with tempfile.TemporaryDirectory() as temp_dir:
            operator, config = self.configured_operator(
                temp_dir, disable_auth=True
            )
            with (
                mock.patch("ts_operators.subprocess.Popen", return_value=launcher),
                mock.patch("ts_operators.tempfile.mkdtemp", return_value=temp_dir),
                mock.patch.object(operator, "_wait_for_management_api",
                                  return_value=False),
                mock.patch("ts_operators.os.killpg") as killpg,
            ):
                result = operator.start_service(config)

        self.assertIsNone(result)
        killpg.assert_not_called()
        self.assertIsNone(operator._owned_process)

    def test_start_service_cleans_up_unready_owned_process(self):
        launcher = mock.Mock(pid=1234)
        launcher.poll.side_effect = [None, 0]
        with tempfile.TemporaryDirectory() as temp_dir:
            operator, config = self.configured_operator(
                temp_dir, disable_auth=True
            )
            with (
                mock.patch("ts_operators.subprocess.Popen", return_value=launcher),
                mock.patch("ts_operators.tempfile.mkdtemp", return_value=temp_dir),
                mock.patch("ts_operators.os.killpg") as killpg,
                mock.patch.object(operator, "_wait_for_management_api",
                                  return_value=False),
            ):
                result = operator.start_service(config)

        self.assertIsNone(result)
        killpg.assert_called_once_with(launcher.pid, signal.SIGTERM)
        self.assertIsNone(operator._owned_process)

    def test_start_service_cleans_up_after_post_launch_exception(self):
        launcher = mock.Mock(pid=1234)
        launcher.poll.side_effect = [None, 0]
        with tempfile.TemporaryDirectory() as temp_dir:
            operator, config = self.configured_operator(
                temp_dir, disable_auth=True
            )
            with (
                mock.patch("ts_operators.subprocess.Popen", return_value=launcher),
                mock.patch("ts_operators.tempfile.mkdtemp", return_value=temp_dir),
                mock.patch.object(
                    operator,
                    "_wait_for_management_api",
                    side_effect=RuntimeError("probe failed"),
                ),
                mock.patch("ts_operators.os.killpg") as killpg,
            ):
                result = operator.start_service(config)

        self.assertIsNone(result)
        killpg.assert_called_once_with(launcher.pid, signal.SIGTERM)

    def test_start_service_rejects_mismatched_config_before_spawning(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            operator, config = self.configured_operator(
                temp_dir, disable_auth=True
            )
            with open(config, "a", encoding="utf-8") as handle:
                handle.write("management_address=http://0.0.0.0:9999\n")
            with (
                mock.patch("ts_operators.subprocess.Popen") as popen,
            ):
                with self.assertRaisesRegex(RuntimeError,
                                            "duplicate TorchServe config key"):
                    operator.start_service(config)

        popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
