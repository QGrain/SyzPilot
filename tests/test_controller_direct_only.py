"""Regression tests for the container's direct-only Brain deployment."""

import asyncio
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from fastapi import HTTPException


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "brain"))

import controller


class DirectOnlyControllerTest(unittest.TestCase):
    def test_liveness_does_not_probe_torchserve(self):
        instance = controller.Controller()
        with mock.patch.object(
            instance.torchserve_operator,
            "is_service_ready",
            side_effect=AssertionError("liveness must not probe TorchServe"),
        ):
            payload = asyncio.run(instance.liveness())

        self.assertEqual(payload, {"alive": True, "active_tasks": 0})

    def test_liveness_route_is_distinct_from_readiness(self):
        instance = controller.Controller()
        routes = {
            route.path: route.endpoint
            for route in instance.app.routes
            if hasattr(route, "endpoint")
        }

        self.assertEqual(routes["/live"].__self__, instance)
        self.assertEqual(routes["/live"].__func__, instance.liveness.__func__)
        self.assertEqual(routes["/health"].__self__, instance)
        self.assertEqual(routes["/health"].__func__, instance.health.__func__)

    def test_readiness_returns_service_unavailable_until_torchserve_is_ready(self):
        instance = controller.Controller()
        with mock.patch.object(
            instance.torchserve_operator, "is_service_ready", return_value=False
        ):
            response = asyncio.run(instance.health())

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            json.loads(response.body),
            {"healthy": False, "torchserve": False, "active_tasks": 0},
        )

    def test_readiness_returns_payload_when_torchserve_is_ready(self):
        instance = controller.Controller()
        with mock.patch.object(
            instance.torchserve_operator, "is_service_ready", return_value=True
        ):
            payload = asyncio.run(instance.health())

        self.assertEqual(
            payload,
            {"healthy": True, "torchserve": True, "active_tasks": 0},
        )

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
