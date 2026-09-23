"""Regression tests for the container's direct-only Brain deployment."""

import asyncio
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from fastapi import HTTPException


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "brain"))

import controller


class DirectOnlyControllerTest(unittest.TestCase):
    def test_startup_skips_tunnel_helper_but_starts_torchserve(self):
        instance = controller.Controller()
        with (
            mock.patch.object(controller.config, "direct_only", True),
            mock.patch.object(controller.utils, "run_cmd") as run_cmd,
            mock.patch.object(instance, "_load_existing_keys") as load_keys,
            mock.patch.object(
                instance, "_validate_physical_gpu_mapping"
            ) as validate_gpus,
            mock.patch.object(instance, "_ensure_torchserve_started") as start_ts,
        ):
            instance.startup()

        validate_gpus.assert_called_once_with()
        run_cmd.assert_not_called()
        load_keys.assert_not_called()
        start_ts.assert_called_once_with()

    def test_gpu_identity_failure_prevents_torchserve_start(self):
        instance = controller.Controller()
        with (
            mock.patch.object(controller.config, "direct_only", True),
            mock.patch.object(
                instance,
                "_validate_physical_gpu_mapping",
                side_effect=RuntimeError("GPU identity mismatch"),
            ),
            mock.patch.object(instance, "_ensure_torchserve_started") as start,
            self.assertRaisesRegex(RuntimeError, "GPU identity mismatch"),
        ):
            instance.startup()

        start.assert_not_called()

    def test_isolated_registration_is_rejected_before_allocation(self):
        instance = controller.Controller()
        payload = SimpleNamespace(mode="isolated")
        with mock.patch.object(controller.config, "direct_only", True):
            with self.assertRaises(HTTPException) as raised:
                asyncio.run(instance.register(payload))

        self.assertEqual(raised.exception.status_code, 400)
        self.assertEqual(instance.global_tasks, {})

    def test_shutdown_skips_tunnel_key_cleanup(self):
        instance = controller.Controller()
        with (
            mock.patch.object(controller.config, "direct_only", True),
            mock.patch.object(instance, "_clear_authorized_keys") as clear_keys,
            mock.patch.object(instance, "_ensure_torchserve_stopped") as stop_ts,
        ):
            instance.shutdown()

        clear_keys.assert_not_called()
        stop_ts.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
