"""Training state-machine and checkpoint-selection regression tests."""

import asyncio
import concurrent.futures
import hashlib
import json
import math
import os
import pickle
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
BRAIN_DIR = REPO_ROOT / "brain"
sys.path.insert(0, str(BRAIN_DIR))
# The production entry points use script-local top-level module names. Remove
# any filter-side modules cached by another unittest module in this process.
sys.modules.pop("config", None)
sys.modules.pop("utils", None)

import controller as controller_module
from controller import (
    Controller,
    FuzzerTask,
    RegistrationPayload,
    build_training_split,
    list_committed_batch_indices,
    next_curriculum_stage,
    resolve_guidance_context,
    validate_task_identity,
    validate_guidance_context,
)
from guidance_engine import GuidanceConfig, GuidanceEngine


class FakeProcess:
    pid = 12345
    returncode = None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode

    def terminate(self):
        self.returncode = -15


class CompletedProcess(FakeProcess):
    returncode = 0


class FailedProcess(FakeProcess):
    returncode = 1


class SignalingLock:
    """Thread lock that exposes when an acquire attempt has started."""

    def __init__(self):
        self._lock = threading.Lock()
        self.acquire_started = threading.Event()

    def acquire(self, *args, **kwargs):
        self.acquire_started.set()
        return self._lock.acquire(*args, **kwargs)

    def release(self):
        self._lock.release()


class CloseFailureLog:
    def close(self):
        raise OSError("simulated close failure")


def passing_stage_one_promotion_metrics():
    return {
        "best_eval_accuracy": 0.95,
        "best_eval_weighted_f1": 0.95,
        "best_eval_macro_f1": 0.95,
        "validation_class_counts": {"0": 100, "1": 100},
        "validation_per_class_recall": {"0": 0.95, "1": 0.95},
        "validation_signature_count": 200,
        "validation_signature_sha256": "a" * 64,
        "loaded_checkpoint": None,
        "loaded_checkpoint_stage": 0,
    }


def write_committed_batch(data_dir, batch_id, labels=None):
    program = f"test${batch_id}()"
    signature = hashlib.sha1(program.encode("utf-8")).hexdigest()
    labels = labels or [True, False, False]
    with (data_dir / f"progs_batch_{batch_id}.pkl").open("wb") as handle:
        pickle.dump({signature: program}, handle)
    with (data_dir / f"labels_batch_{batch_id}.pkl").open("wb") as handle:
        pickle.dump({signature: labels}, handle)
    (data_dir / f"batch_{batch_id}.complete").write_text(
        "\n".join((
            "schema_version=1",
            f"batch_id={batch_id}",
            "unique_samples=1",
            "wire_samples=1",
            f"batch_digest={'0' * 64}",
            "",
        )),
        encoding="utf-8",
    )


def write_balanced_committed_batch(data_dir, batch_id, num_classes=3):
    programs = {}
    labels = {}
    for class_index in range(num_classes):
        program = f"test${batch_id}_{class_index}()"
        signature = hashlib.sha1(program.encode("utf-8")).hexdigest()
        label = [False] * num_classes
        label[class_index] = True
        programs[signature] = program
        labels[signature] = label
    with (data_dir / f"progs_batch_{batch_id}.pkl").open("wb") as handle:
        pickle.dump(programs, handle)
    with (data_dir / f"labels_batch_{batch_id}.pkl").open("wb") as handle:
        pickle.dump(labels, handle)
    (data_dir / f"batch_{batch_id}.complete").write_text(
        "\n".join((
            "schema_version=1",
            f"batch_id={batch_id}",
            f"unique_samples={num_classes}",
            f"wire_samples={num_classes}",
            f"batch_digest={'0' * 64}",
            "",
        )),
        encoding="utf-8",
    )


class TrainingStateTest(unittest.TestCase):
    def test_controller_rejects_ambiguous_cuda_device_namespace(self):
        with mock.patch.dict(
                os.environ, {"CUDA_VISIBLE_DEVICES": "2,0"}, clear=False):
            with self.assertRaisesRegex(
                    RuntimeError, "without CUDA_VISIBLE_DEVICES"):
                Controller()

    def test_controller_rejects_non_pci_cuda_device_order(self):
        with mock.patch.dict(
                os.environ, {"CUDA_DEVICE_ORDER": "FASTEST_FIRST"},
                clear=False):
            with self.assertRaisesRegex(RuntimeError, "PCI_BUS_ID"):
                Controller()

    def test_export_checkpoint_load_is_cpu_staged(self):
        torch_module = mock.Mock()
        torch_module.load.return_value = {"weight": object()}
        checkpoint = Path("checkpoint.pt")

        result = Controller._load_export_state_dict(
            torch_module, checkpoint
        )

        self.assertIs(result, torch_module.load.return_value)
        torch_module.load.assert_called_once_with(
            str(checkpoint), map_location="cpu", weights_only=True
        )

    def test_deployment_uses_latency_bounded_batching_config(self):
        instance = Controller()
        operator = mock.Mock()
        operator.model_dir = "/tmp/model-store"
        operator.create_index2name.return_value = "/tmp/index_to_name.json"
        operator.get_model_info.return_value = {"workers": [{"status": "READY"}]}
        instance.torchserve_operator = operator
        task = FuzzerTask(
            task_id="fuzzer@task@1", task_name="task", run_id=1,
            fuzzer_id="fuzzer", mode="direct", callback_addr="localhost:1",
        )

        with (
            mock.patch.object(
                controller_module.config, "ts_serving_batch_size", 8
            ),
            mock.patch.object(
                controller_module.config, "ts_max_batch_delay_ms", 7
            ),
        ):
            instance._deploy_to_torchserve(
                task, "model", "1", "/tmp/model.pt", 3, 1
            )

        operator.register_model.assert_called_once_with(
            model_name="model",
            init_worker=1,
            batch_size=8,
            max_batch_delay=7,
            sync=True,
        )
        operator.scale_worker.assert_not_called()

    def test_training_gpu_configuration_rejects_ambiguous_ids(self):
        with self.assertRaisesRegex(ValueError, "unique numeric"):
            controller_module.ControllerConfig(training_gpu_ids=("0", "0"))
        with self.assertRaisesRegex(ValueError, "unique numeric"):
            controller_module.ControllerConfig(training_gpu_ids=("GPU-uuid",))
        with self.assertRaisesRegex(ValueError, "unique numeric"):
            controller_module.ControllerConfig(training_gpu_ids=("00", "2"))
        with self.assertRaisesRegex(ValueError, "absent from primary"):
            controller_module.ControllerConfig(training_gpu_ids=("0", "1"))
        with self.assertRaisesRegex(ValueError, "inference_gpu_id"):
            controller_module.ControllerConfig(
                training_gpu_ids=("0",), inference_gpu_id="0"
            )
        with self.assertRaisesRegex(ValueError, "must match"):
            controller_module.ControllerConfig(
                training_gpu_ids=("0",),
                attribution_gpu_id="1",
                inference_gpu_id="2",
            )
        with self.assertRaisesRegex(ValueError, "internal_batch_size"):
            controller_module.ControllerConfig(
                attribution_internal_batch_size=0
            )
        with self.assertRaisesRegex(ValueError, "unique numeric"):
            controller_module.ControllerConfig(
                training_fallback_gpu_ids=("1", "1")
            )
        with self.assertRaisesRegex(ValueError, "must not overlap"):
            controller_module.ControllerConfig(
                training_gpu_ids=("0", "2"),
                training_fallback_gpu_ids=("2",),
            )

        shared = controller_module.ControllerConfig(
            training_gpu_ids=("0", "2"),
            training_fallback_gpu_ids=("1",),
            attribution_gpu_id="1",
        )
        self.assertEqual(shared.training_fallback_gpu_ids, ("1",))

        with self.assertRaisesRegex(ValueError, "attribution_slots"):
            controller_module.ControllerConfig(attribution_slots=0)
        with self.assertRaisesRegex(ValueError, "attribution_slots"):
            controller_module.ControllerConfig(attribution_slots=2)
        with self.assertRaisesRegex(ValueError, "training_slots_per_gpu"):
            controller_module.ControllerConfig(training_slots_per_gpu=2)
        with self.assertRaisesRegex(ValueError, "between 0 and 100"):
            controller_module.ControllerConfig(
                training_max_gpu_utilization=101
            )
        with self.assertRaisesRegex(ValueError, "probe_samples"):
            controller_module.ControllerConfig(training_gpu_probe_samples=0)
        with self.assertRaisesRegex(ValueError, "probe_interval"):
            controller_module.ControllerConfig(
                training_gpu_probe_interval_seconds=1.1
            )
        with self.assertRaisesRegex(ValueError, "batch_size"):
            controller_module.ControllerConfig(batch_size=0)
        with self.assertRaisesRegex(ValueError, "grad_acc_steps"):
            controller_module.ControllerConfig(grad_acc_steps=0)
        with self.assertRaisesRegex(ValueError, "ts_serving_batch_size"):
            controller_module.ControllerConfig(ts_serving_batch_size=0)
        with self.assertRaisesRegex(ValueError, "ts_max_batch_delay_ms"):
            controller_module.ControllerConfig(ts_max_batch_delay_ms=-1)
        with self.assertRaisesRegex(ValueError, "ts_max_batch_delay_ms"):
            controller_module.ControllerConfig(ts_max_batch_delay_ms=1001)

    def test_torchserve_batch_environment_overrides_in_fresh_process(self):
        environment = os.environ.copy()
        environment.update({
            "SYZPILOT_TS_SERVING_BATCH_SIZE": "8",
            "SYZPILOT_TS_MAX_BATCH_DELAY_MS": "7",
        })
        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import json; from config import ControllerConfig; "
                    "cfg = ControllerConfig(); "
                    "print(json.dumps([cfg.ts_serving_batch_size, "
                    "cfg.ts_max_batch_delay_ms]))"
                ),
            ],
            cwd=BRAIN_DIR,
            env=environment,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        self.assertEqual(json.loads(probe.stdout), [8, 7])

    def test_torchserve_batch_environment_rejects_non_integer(self):
        environment = os.environ.copy()
        environment["SYZPILOT_TS_MAX_BATCH_DELAY_MS"] = "not-an-integer"
        probe = subprocess.run(
            [sys.executable, "-c", "from config import ControllerConfig"],
            cwd=BRAIN_DIR,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        self.assertNotEqual(probe.returncode, 0)
        self.assertIn("invalid literal for int()", probe.stderr)
        self.assertIn("not-an-integer", probe.stderr)

    def test_training_waiter_lease_must_be_finite(self):
        for value in (float("nan"), float("inf"), float("-inf"), 14.9):
            with self.subTest(value=value), self.assertRaisesRegex(
                    ValueError, "training_waiter_lease_seconds"):
                controller_module.ControllerConfig(
                    training_waiter_lease_seconds=value
                )

    def test_training_profiles_are_explicit_and_validated(self):
        defaults = controller_module.ControllerConfig()
        self.assertEqual(
            (
                defaults.first_train_total_steps,
                defaults.first_train_test_interval,
                defaults.first_train_min_steps,
                defaults.first_train_patience,
            ),
            (1000, 200, 200, 3),
        )
        self.assertEqual(
            (
                defaults.continued_train_total_steps,
                defaults.continued_train_test_interval,
                defaults.continued_train_min_steps,
                defaults.continued_train_patience,
            ),
            (500, 100, 100, 2),
        )
        with self.assertRaisesRegex(ValueError, "first training profile"):
            controller_module.ControllerConfig(first_train_total_steps=0)
        with self.assertRaisesRegex(ValueError, "test interval"):
            controller_module.ControllerConfig(
                continued_train_total_steps=50,
                continued_train_test_interval=51,
            )
        with self.assertRaisesRegex(ValueError, "minimum steps"):
            controller_module.ControllerConfig(
                first_train_total_steps=100,
                first_train_test_interval=100,
                first_train_min_steps=101,
            )
        with self.assertRaisesRegex(ValueError, "warmup steps"):
            controller_module.ControllerConfig(num_warmup_steps=501)

    def test_attribution_gpu_is_serialized_across_tasks(self):
        instance = Controller()
        tasks = [
            FuzzerTask(
                task_id=f"fuzzer-{index}@task@1", task_name="task", run_id=1,
                fuzzer_id=f"fuzzer-{index}", mode="direct",
                callback_addr="localhost:1",
            )
            for index in range(2)
        ]
        for task in tasks:
            instance.global_tasks[task.task_id] = task
        active = 0
        maximum_active = 0
        state_lock = threading.Lock()
        first_entered = threading.Event()
        allow_first_exit = threading.Event()

        def reserved_body(task, engine, ckpt_path, num_classes, stage):
            nonlocal active, maximum_active
            try:
                with state_lock:
                    active += 1
                    maximum_active = max(maximum_active, active)
                first_entered.set()
                allow_first_exit.wait(timeout=1)
            finally:
                with state_lock:
                    active -= 1

        with mock.patch.object(
                instance, "_run_attribution_analysis_on_reserved_gpu",
                side_effect=reserved_body):
            workers = [
                threading.Thread(
                    target=instance._run_attribution_analysis,
                    args=(task, mock.Mock(), "checkpoint", 3, 2),
                )
                for task in tasks
            ]
            workers[0].start()
            self.assertTrue(first_entered.wait(timeout=1))
            workers[1].start()
            self.assertTrue(workers[1].is_alive())
            allow_first_exit.set()
            for worker in workers:
                worker.join(timeout=1)

        self.assertEqual(maximum_active, 1)
        self.assertTrue(all(not worker.is_alive() for worker in workers))

    def test_attribution_gpu_slot_is_released_after_failure(self):
        instance = Controller()
        task = FuzzerTask(
            task_id="fuzzer@task@1", task_name="task", run_id=1,
            fuzzer_id="fuzzer", mode="direct", callback_addr="localhost:1",
        )
        instance.global_tasks[task.task_id] = task

        with mock.patch.object(
                instance, "_run_attribution_analysis_on_reserved_gpu",
                side_effect=RuntimeError("boom")):
            instance._run_attribution_analysis(
                task, mock.Mock(), "checkpoint", 3, 2
            )

        self.assertTrue(
            instance._attribution_gpu_semaphore.acquire(blocking=False)
        )
        instance._attribution_gpu_semaphore.release()
        physical = instance._gpu_semaphores[
            controller_module.config.attribution_gpu_id
        ]
        self.assertTrue(physical.acquire(blocking=False))
        physical.release()

    def test_attribution_waits_for_shared_fallback_gpu_lease(self):
        instance = Controller()
        attribution_gpu = controller_module.config.attribution_gpu_id
        instance._training_fallback_gpu_ids = [attribution_gpu]
        task = FuzzerTask(
            task_id="fuzzer@task@1", task_name="task", run_id=1,
            fuzzer_id="fuzzer", mode="direct", callback_addr="localhost:1",
        )
        instance.global_tasks[task.task_id] = task
        physical = instance._gpu_semaphores[attribution_gpu]
        self.assertTrue(physical.acquire(blocking=False))
        entered = threading.Event()

        with mock.patch.object(
                instance, "_run_attribution_analysis_on_reserved_gpu",
                side_effect=lambda *args: entered.set()):
            worker = threading.Thread(
                target=instance._run_attribution_analysis,
                args=(task, mock.Mock(), "checkpoint", 3, 2),
            )
            worker.start()
            self.assertFalse(entered.wait(timeout=0.1))
            physical.release()
            self.assertTrue(entered.wait(timeout=1))
            worker.join(timeout=1)

        self.assertFalse(worker.is_alive())

    def test_attribution_yields_fallback_gpu_to_training_waiter(self):
        instance = Controller()
        attribution_gpu = controller_module.config.attribution_gpu_id
        instance._training_fallback_gpu_ids = [attribution_gpu]
        attribution_task = FuzzerTask(
            task_id="attribution", task_name="task", run_id=1,
            fuzzer_id="attribution", mode="direct",
            callback_addr="localhost:1",
        )
        training_task = FuzzerTask(
            task_id="training", task_name="task", run_id=2,
            fuzzer_id="training", mode="direct",
            callback_addr="localhost:2",
        )
        instance.global_tasks = {
            attribution_task.task_id: attribution_task,
            training_task.task_id: training_task,
        }
        instance._enqueue_training_waiter(training_task.task_id)
        entered = threading.Event()

        with mock.patch.object(
                instance, "_run_attribution_analysis_on_reserved_gpu",
                side_effect=lambda *args: entered.set()):
            worker = threading.Thread(
                target=instance._run_attribution_analysis,
                args=(attribution_task, mock.Mock(), "checkpoint", 3, 2),
            )
            worker.start()
            self.assertFalse(entered.wait(timeout=0.1))
            instance._remove_training_waiter(training_task.task_id)
            self.assertTrue(entered.wait(timeout=1.5))
            worker.join(timeout=1)

        self.assertFalse(worker.is_alive())

    def test_canceled_attribution_does_not_run_after_waiting_for_slot(self):
        instance = Controller()
        task = FuzzerTask(
            task_id="fuzzer@task@1", task_name="task", run_id=1,
            fuzzer_id="fuzzer", mode="direct", callback_addr="localhost:1",
        )
        instance.global_tasks[task.task_id] = task
        semaphore = instance._attribution_gpu_semaphore
        self.assertTrue(semaphore.acquire(blocking=False))
        waiter_entered = threading.Event()
        original_acquire = semaphore.acquire

        def signaling_acquire(*args, **kwargs):
            waiter_entered.set()
            return original_acquire(*args, **kwargs)

        with (
            mock.patch.object(
                semaphore, "acquire", side_effect=signaling_acquire
            ),
            mock.patch.object(
                instance, "_run_attribution_analysis_on_reserved_gpu"
            ) as reserved_body,
        ):
            worker = threading.Thread(
                target=instance._run_attribution_analysis,
                args=(task, mock.Mock(), "checkpoint", 3, 2),
            )
            worker.start()
            self.assertTrue(waiter_entered.wait(timeout=1))
            task.guidance_cancel.set()
            instance.global_tasks.pop(task.task_id)
            semaphore.release()
            worker.join(timeout=1)
            self.assertFalse(worker.is_alive())
            reserved_body.assert_not_called()

        self.assertTrue(semaphore.acquire(blocking=False))
        semaphore.release()

    def test_unregister_waits_for_running_attribution_gpu_release(self):
        instance = Controller()
        task = FuzzerTask(
            task_id="fuzzer@task@1", task_name="task", run_id=1,
            fuzzer_id="fuzzer", mode="direct", callback_addr="localhost:1",
            guidance_engine=mock.Mock(), grpc_port=31001,
        )
        instance.global_tasks[task.task_id] = task
        entered = threading.Event()
        observed_cancel = threading.Event()
        physical = instance._gpu_semaphores[
            controller_module.config.attribution_gpu_id
        ]

        def reserved_body(*args):
            entered.set()
            if task.guidance_cancel.wait(timeout=2):
                observed_cancel.set()

        def guidance_body(*args):
            instance._run_attribution_analysis(
                task, task.guidance_engine, "checkpoint", 3, 2
            )

        with (
            mock.patch.object(
                instance, "_run_guidance_pipeline_locked",
                side_effect=guidance_body,
            ),
            mock.patch.object(
                instance, "_run_attribution_analysis_on_reserved_gpu",
                side_effect=reserved_body,
            ),
            mock.patch.object(instance, "_release_port"),
            mock.patch.object(instance, "_remove_ssh_entry"),
        ):
            worker = threading.Thread(
                target=instance._run_guidance_pipeline,
                args=(task, 2, 3, "model", "checkpoint"),
            )
            worker.start()
            self.assertTrue(entered.wait(timeout=1))
            result = asyncio.run(instance.unregister(task.task_id))
            worker.join(timeout=1)

        self.assertEqual(result, {"status": "ok"})
        self.assertTrue(observed_cancel.is_set())
        self.assertFalse(worker.is_alive())
        self.assertNotIn(task.task_id, instance.global_tasks)
        self.assertTrue(
            instance._attribution_gpu_semaphore.acquire(blocking=False)
        )
        instance._attribution_gpu_semaphore.release()
        self.assertTrue(physical.acquire(blocking=False))
        physical.release()

    def test_canceled_unregister_reclaims_late_guidance_lock(self):
        instance = Controller()
        task = FuzzerTask(
            task_id="fuzzer@task@1", task_name="task", run_id=1,
            fuzzer_id="fuzzer", mode="direct", callback_addr="localhost:1",
        )
        task.guidance_lock = SignalingLock()
        self.assertTrue(task.guidance_lock.acquire(blocking=False))
        task.guidance_lock.acquire_started.clear()
        instance.global_tasks[task.task_id] = task

        async def scenario():
            unregister = asyncio.create_task(instance.unregister(task.task_id))
            while not task.guidance_lock.acquire_started.is_set():
                await asyncio.sleep(0.01)
            unregister.cancel()
            task.guidance_lock.release()
            with self.assertRaises(asyncio.CancelledError):
                await unregister

        asyncio.run(scenario())
        self.assertTrue(task.guidance_lock.acquire(blocking=False))
        task.guidance_lock.release()
        self.assertIn(task.task_id, instance.global_tasks)

    def test_canceled_unregister_releases_guidance_while_waiting_lifecycle(self):
        instance = Controller()
        task = FuzzerTask(
            task_id="fuzzer@task@1", task_name="task", run_id=1,
            fuzzer_id="fuzzer", mode="direct", callback_addr="localhost:1",
        )
        task.guidance_lock = SignalingLock()
        task.lifecycle_lock = SignalingLock()
        self.assertTrue(task.lifecycle_lock.acquire(blocking=False))
        task.lifecycle_lock.acquire_started.clear()
        instance.global_tasks[task.task_id] = task

        async def scenario():
            unregister = asyncio.create_task(instance.unregister(task.task_id))
            while not task.lifecycle_lock.acquire_started.is_set():
                await asyncio.sleep(0.01)
            unregister.cancel()
            task.lifecycle_lock.release()
            with self.assertRaises(asyncio.CancelledError):
                await unregister

        asyncio.run(scenario())
        self.assertTrue(task.guidance_lock.acquire(blocking=False))
        task.guidance_lock.release()
        self.assertTrue(task.lifecycle_lock.acquire(blocking=False))
        task.lifecycle_lock.release()
        self.assertIn(task.task_id, instance.global_tasks)

    def test_gpu_ranking_filters_low_memory_and_sorts_all_candidates(self):
        instance = Controller()
        instance._training_gpu_ids = ["0", "1", "2"]
        instance._gpu_semaphores = {
            gpu_id: threading.Semaphore(1)
            for gpu_id in instance._training_gpu_ids
        }
        instance._training_min_free_mib = 40000
        instance._training_max_gpu_utilization = 20
        instance._training_gpu_probe_samples = 1
        result = mock.Mock(
            returncode=0,
            stdout=(
                "0, 50000, 81920, 0\n"
                "1, 1000, 81920, 75\n"
                "2, 20000, 81920, 10\n"
            ),
        )

        with mock.patch.object(
                controller_module.subprocess, "run", return_value=result):
            self.assertEqual(instance._rank_training_gpus(), ["2"])

    def test_gpu_acquire_tries_next_memory_ranked_local_slot(self):
        instance = Controller()
        instance._training_gpu_ids = ["0", "1", "2"]
        instance._gpu_semaphores = {
            gpu_id: threading.Semaphore(1)
            for gpu_id in instance._training_gpu_ids
        }
        self.assertTrue(instance._gpu_semaphores["2"].acquire(blocking=False))

        with mock.patch.object(
                instance, "_rank_training_gpus", return_value=["2", "1", "0"]):
            selected = instance._acquire_training_gpu()

        self.assertEqual(selected, "1")
        instance._release_training_gpu("1")
        instance._release_training_gpu("2")

    def test_gpu_ranking_keeps_primary_tier_ahead_of_idle_fallback(self):
        instance = Controller()
        instance._training_gpu_ids = ["0"]
        instance._training_fallback_gpu_ids = ["1"]
        instance._gpu_semaphores = {
            gpu_id: threading.BoundedSemaphore(1)
            for gpu_id in ("0", "1")
        }
        instance._training_min_free_mib = 40000
        instance._training_max_gpu_utilization = 20
        instance._training_gpu_probe_samples = 1
        result = mock.Mock(
            returncode=0,
            stdout=(
                "0, 30000, 81920, 20\n"
                "1, 1000, 81920, 0\n"
            ),
        )

        with mock.patch.object(
                controller_module.subprocess, "run", return_value=result):
            self.assertEqual(instance._rank_training_gpus(), ["0", "1"])

    def test_gpu_acquire_uses_fallback_only_when_primary_slot_is_busy(self):
        instance = Controller()
        instance._training_gpu_ids = ["0"]
        instance._training_fallback_gpu_ids = ["1"]
        instance._gpu_semaphores = {
            gpu_id: threading.BoundedSemaphore(1)
            for gpu_id in ("0", "1")
        }
        self.assertTrue(
            instance._gpu_semaphores["0"].acquire(blocking=False)
        )

        with mock.patch.object(
                instance, "_rank_training_gpus", return_value=["0", "1"]):
            selected = instance._acquire_training_gpu()

        self.assertEqual(selected, "1")
        instance._release_training_gpu("1")
        instance._release_training_gpu("0")

    def test_training_wait_queue_is_fifo_and_idempotent(self):
        instance = Controller()
        for task_id in ("task-a", "task-b", "task-c"):
            instance.global_tasks[task_id] = FuzzerTask(
                task_id=task_id, task_name=task_id, run_id=1,
                fuzzer_id=task_id, mode="direct", callback_addr="localhost:1",
            )

        self.assertEqual(instance._enqueue_training_waiter("task-a"), 0)
        self.assertEqual(instance._enqueue_training_waiter("task-b"), 1)
        self.assertEqual(instance._enqueue_training_waiter("task-a"), 0)
        self.assertEqual(list(instance._training_wait_queue), [
            "task-a", "task-b",
        ])

        instance._remove_training_waiter("task-a")
        self.assertEqual(instance._enqueue_training_waiter("task-b"), 0)
        self.assertEqual(instance._enqueue_training_waiter("task-c"), 1)
        instance._remove_training_waiter("missing")
        self.assertEqual(list(instance._training_wait_queue), [
            "task-b", "task-c",
        ])

    def test_training_wait_queue_prunes_dead_receiver_head(self):
        instance = Controller()
        dead_receiver = FailedProcess()
        first = FuzzerTask(
            task_id="first", task_name="first", run_id=1,
            fuzzer_id="first", mode="direct", callback_addr="localhost:1",
            receiver_proc=FakeProcess(),
        )
        second = FuzzerTask(
            task_id="second", task_name="second", run_id=1,
            fuzzer_id="second", mode="direct", callback_addr="localhost:2",
            receiver_proc=FakeProcess(),
        )
        instance.global_tasks = {first.task_id: first, second.task_id: second}
        self.assertEqual(instance._enqueue_training_waiter(first.task_id), 0)
        self.assertEqual(instance._enqueue_training_waiter(second.task_id), 1)
        first.receiver_proc = dead_receiver

        self.assertEqual(instance._enqueue_training_waiter(second.task_id), 0)
        self.assertNotIn(first.task_id, instance._training_wait_set)

    def test_training_wait_queue_prunes_expired_live_head(self):
        instance = Controller()
        instance._training_waiter_lease_seconds = 30
        first = FuzzerTask(
            task_id="first", task_name="first", run_id=1,
            fuzzer_id="first", mode="direct", callback_addr="localhost:1",
        )
        second = FuzzerTask(
            task_id="second", task_name="second", run_id=1,
            fuzzer_id="second", mode="direct", callback_addr="localhost:2",
        )
        instance.global_tasks = {first.task_id: first, second.task_id: second}
        with mock.patch.object(
                controller_module.time, "monotonic", side_effect=(100, 100, 131)):
            self.assertEqual(instance._enqueue_training_waiter(first.task_id), 0)
            self.assertEqual(instance._enqueue_training_waiter(second.task_id), 1)
            self.assertEqual(instance._enqueue_training_waiter(second.task_id), 0)
        self.assertNotIn(first.task_id, instance._training_wait_set)

    def test_training_fifo_allows_two_tasks_to_acquire_distinct_gpus(self):
        instance = Controller()
        instance._training_gpu_ids = ["0", "1"]
        instance._gpu_semaphores = {
            gpu_id: threading.Semaphore(1)
            for gpu_id in instance._training_gpu_ids
        }
        tasks = []
        for task_id in ("first", "second"):
            task = FuzzerTask(
                task_id=task_id, task_name=task_id, run_id=1,
                fuzzer_id=task_id, mode="direct", callback_addr="localhost:1",
            )
            instance.global_tasks[task_id] = task
            tasks.append(task)
        self.assertEqual(instance._enqueue_training_waiter("first"), 0)
        self.assertEqual(instance._enqueue_training_waiter("second"), 1)
        with mock.patch.object(
                instance, "_rank_training_gpus", return_value=["0", "1"]):
            first_gpu = instance._acquire_training_gpu()
            instance._remove_training_waiter("first")
            self.assertEqual(instance._enqueue_training_waiter("second"), 0)
            second_gpu = instance._acquire_training_gpu()
        self.assertEqual((first_gpu, second_gpu), ("0", "1"))
        instance._release_training_gpu(first_gpu)
        instance._release_training_gpu(second_gpu)

    def test_unregister_removes_training_waiter_before_cleanup_retry(self):
        instance = Controller()
        task = FuzzerTask(
            task_id="fuzzer@task@1", task_name="task", run_id=1,
            fuzzer_id="fuzzer", mode="direct", callback_addr="localhost:1",
        )
        instance.global_tasks[task.task_id] = task
        instance._enqueue_training_waiter(task.task_id)

        with mock.patch.object(
                instance, "_stop_task_training", return_value=False):
            with self.assertRaises(controller_module.HTTPException):
                asyncio.run(instance.unregister(task.task_id))

        self.assertNotIn(task.task_id, instance._training_wait_set)
        self.assertEqual(list(instance._training_wait_queue), [])

    def test_preflight_failure_removes_head_retained_after_no_gpu(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir) / "data"
            data_dir = data_root / "task" / "1"
            data_dir.mkdir(parents=True)
            for batch_id in (1, 2, 3):
                write_balanced_committed_batch(data_dir, batch_id)
            instance = Controller()
            task = FuzzerTask(
                task_id="fuzzer@task@1", task_name="task", run_id=1,
                fuzzer_id="fuzzer", mode="direct", callback_addr="localhost:1",
                receiver_proc=FakeProcess(),
            )
            instance.global_tasks[task.task_id] = task

            with (
                mock.patch.object(controller_module.config, "data_root", str(data_root)),
                mock.patch.object(
                    instance, "_acquire_training_gpu_async",
                    new=mock.AsyncMock(return_value=None),
                ),
                self.assertRaisesRegex(
                    controller_module.HTTPException, "No GPU available"
                ),
            ):
                asyncio.run(instance.start_trainer(
                    task.task_id, batch_start=0, batch_end=3, stage=1
                ))

            self.assertEqual(list(instance._training_wait_queue), [task.task_id])
            self.assertFalse(task.training_in_progress)
            missing_root = Path(temp_dir) / "missing"
            with (
                mock.patch.object(
                    controller_module.config, "data_root", str(missing_root)
                ),
                self.assertRaisesRegex(
                    controller_module.HTTPException, "Data directory not found"
                ),
            ):
                asyncio.run(instance.start_trainer(
                    task.task_id, batch_start=0, batch_end=3, stage=1
                ))

            self.assertNotIn(task.task_id, instance._training_wait_set)
            self.assertEqual(list(instance._training_wait_queue), [])

    def test_gpu_ranking_defers_when_status_query_fails(self):
        instance = Controller()
        instance._training_gpu_ids = ["2", "0"]
        instance._gpu_semaphores = {
            gpu_id: threading.Semaphore(1)
            for gpu_id in instance._training_gpu_ids
        }
        result = mock.Mock(returncode=1, stdout="")

        with mock.patch.object(
                controller_module.subprocess, "run", return_value=result):
            self.assertEqual(instance._rank_training_gpus(), [])

    def test_gpu_ranking_fails_closed_for_low_or_malformed_rows(self):
        instance = Controller()
        instance._training_gpu_ids = ["0", "2"]
        instance._gpu_semaphores = {
            gpu_id: threading.Semaphore(1)
            for gpu_id in instance._training_gpu_ids
        }
        instance._training_min_free_mib = 40000
        instance._training_max_gpu_utilization = 20
        instance._training_gpu_probe_samples = 1

        low_result = mock.Mock(
            returncode=0,
            stdout="0, 41921, 81920, 0\n2, 81920, 81920, 0\n",
        )
        with mock.patch.object(
                controller_module.subprocess, "run", return_value=low_result):
            self.assertEqual(instance._rank_training_gpus(), [])

        threshold_result = mock.Mock(
            returncode=0,
            stdout=(
                "malformed\n"
                "0, 41920, 81920, 20\n"
                "2, not-a-number, 81920, 0\n"
            ),
        )
        with mock.patch.object(
                controller_module.subprocess, "run",
                return_value=threshold_result):
            self.assertEqual(instance._rank_training_gpus(), ["0"])

    def test_gpu_ranking_requires_stable_utilization(self):
        instance = Controller()
        instance._training_gpu_ids = ["0", "2"]
        instance._gpu_semaphores = {
            gpu_id: threading.Semaphore(1)
            for gpu_id in instance._training_gpu_ids
        }
        instance._training_min_free_mib = 24000
        instance._training_max_gpu_utilization = 20
        instance._training_gpu_probe_samples = 3
        instance._training_gpu_probe_interval_seconds = 1
        samples = [
            mock.Mock(
                returncode=0,
                stdout="0, 1000, 81920, 0\n2, 10000, 81920, 10\n",
            ),
            mock.Mock(
                returncode=0,
                stdout="0, 1000, 81920, 95\n2, 10000, 81920, 15\n",
            ),
            mock.Mock(
                returncode=0,
                stdout="0, 1000, 81920, 0\n2, 10000, 81920, 5\n",
            ),
        ]

        with (
            mock.patch.object(
                controller_module.subprocess, "run", side_effect=samples
            ) as run,
            mock.patch.object(threading.Event, "wait", return_value=False) as wait,
        ):
            self.assertEqual(
                instance._rank_training_gpus(threading.Event()), ["2"]
            )

        self.assertEqual(run.call_count, 3)
        self.assertEqual(wait.call_count, 2)

    def test_gpu_ranking_rejects_invalid_telemetry_values(self):
        instance = Controller()
        instance._training_gpu_ids = ["0", "1", "2", "3"]
        instance._gpu_semaphores = {
            gpu_id: threading.Semaphore(1)
            for gpu_id in instance._training_gpu_ids
        }
        instance._training_min_free_mib = 0
        instance._training_max_gpu_utilization = 100
        instance._training_gpu_probe_samples = 1
        result = mock.Mock(
            returncode=0,
            stdout=(
                "0, -1, 81920, 0\n"
                "1, 81921, 81920, 0\n"
                "2, 0, 0, 0\n"
                "3, 0, 81920, 101\n"
            ),
        )

        with mock.patch.object(
                controller_module.subprocess, "run", return_value=result):
            self.assertEqual(instance._rank_training_gpus(), [])

    def test_canceled_gpu_probe_stops_during_sampling_interval(self):
        instance = Controller()
        instance._training_gpu_ids = ["0"]
        instance._gpu_semaphores = {"0": threading.Semaphore(1)}
        instance._training_min_free_mib = 0
        instance._training_max_gpu_utilization = 100
        instance._training_gpu_probe_samples = 3
        instance._training_gpu_probe_interval_seconds = 1
        cancel_event = threading.Event()
        result = mock.Mock(returncode=0, stdout="0, 0, 81920, 0\n")

        def cancel_during_wait(_timeout):
            cancel_event.set()
            return True

        with (
            mock.patch.object(
                controller_module.subprocess, "run", return_value=result
            ) as run,
            mock.patch.object(
                cancel_event, "wait", side_effect=cancel_during_wait
            ),
        ):
            self.assertEqual(
                instance._rank_training_gpus(cancel_event), []
            )

        self.assertEqual(run.call_count, 1)

    def test_concurrent_gpu_probes_do_not_hold_selection_lock(self):
        instance = Controller()
        instance._training_gpu_ids = ["0"]
        instance._gpu_semaphores = {"0": threading.Semaphore(1)}
        instance._gpu_select_lock = threading.Lock()
        both_probing = threading.Barrier(2)

        def concurrent_rank(cancel_event=None):
            both_probing.wait(timeout=1)
            return ["0"]

        with mock.patch.object(
                instance, "_rank_training_gpus", side_effect=concurrent_rank):
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(
                    lambda _: instance._acquire_training_gpu(), range(2)
                ))

        self.assertEqual(
            sorted(results, key=lambda value: value or ""), [None, "0"]
        )
        instance._release_training_gpu("0")

    def test_async_gpu_acquire_does_not_block_event_loop(self):
        instance = Controller()
        started = threading.Event()
        release = threading.Event()

        def slow_acquire(cancel_event=None):
            started.set()
            release.wait(timeout=1)
            return None

        async def scenario():
            with mock.patch.object(
                    instance, "_acquire_training_gpu", side_effect=slow_acquire):
                acquire_task = asyncio.create_task(
                    instance._acquire_training_gpu_async(threading.Event())
                )
                while not started.is_set():
                    await asyncio.sleep(0)
                ticked = False

                async def ticker():
                    nonlocal ticked
                    await asyncio.sleep(0.01)
                    ticked = True
                    release.set()

                await asyncio.gather(acquire_task, ticker())
                return ticked

        self.assertTrue(asyncio.run(scenario()))

    def test_canceled_async_gpu_acquire_releases_late_slot(self):
        instance = Controller()
        started = threading.Event()
        release = threading.Event()

        def slow_acquire(cancel_event=None):
            started.set()
            release.wait(timeout=1)
            return "0"

        async def scenario():
            with (
                mock.patch.object(
                    instance, "_acquire_training_gpu", side_effect=slow_acquire
                ),
                mock.patch.object(instance, "_release_training_gpu") as release_gpu,
            ):
                acquire_task = asyncio.create_task(
                    instance._acquire_training_gpu_async(threading.Event())
                )
                while not started.is_set():
                    await asyncio.sleep(0)
                acquire_task.cancel()
                release.set()
                with self.assertRaises(asyncio.CancelledError):
                    await acquire_task
                release_gpu.assert_called_once_with("0")

        asyncio.run(scenario())

    def test_repeated_cancel_async_gpu_acquire_releases_late_slot(self):
        instance = Controller()
        started = threading.Event()
        release = threading.Event()

        def slow_acquire(cancel_event=None):
            started.set()
            release.wait(timeout=1)
            return "0"

        async def scenario():
            with (
                mock.patch.object(
                    instance, "_acquire_training_gpu", side_effect=slow_acquire
                ),
                mock.patch.object(instance, "_release_training_gpu") as release_gpu,
            ):
                acquire_task = asyncio.create_task(
                    instance._acquire_training_gpu_async(threading.Event())
                )
                while not started.is_set():
                    await asyncio.sleep(0)
                acquire_task.cancel()
                await asyncio.sleep(0)
                acquire_task.cancel()
                release.set()
                with self.assertRaises(asyncio.CancelledError):
                    await acquire_task
                release_gpu.assert_called_once_with("0")

        asyncio.run(scenario())

    def test_curriculum_progresses_from_binary_to_exact(self):
        self.assertEqual(next_curriculum_stage(2, 0, 2), 1)
        self.assertEqual(next_curriculum_stage(2, 1, 2), 2)
        self.assertEqual(next_curriculum_stage(2, 1, 3), 2)
        self.assertEqual(next_curriculum_stage(2, 2, 3), 2)
        with self.assertRaisesRegex(ValueError, "invalid requested"):
            next_curriculum_stage(3, 2, 3)

    def test_legacy_two_stage_state_preserves_exact_stage_two(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir) / "data"
            state_dir = data_root / "task" / "1"
            state_dir.mkdir(parents=True)
            (state_dir / "training_state.json").write_text(json.dumps({
                "schema_version": 1,
                "last_successful_stage": 2,
                "last_trained_batch": 10,
            }), encoding="utf-8")
            instance = Controller()
            task = FuzzerTask(
                task_id="fuzzer@task@1", task_name="task", run_id=1,
                fuzzer_id="fuzzer", mode="direct", callback_addr="localhost:1",
            )
            with mock.patch.object(
                    controller_module.config, "data_root", str(data_root)):
                instance._load_training_state(task)

            self.assertEqual(task.last_successful_stage, 2)
            self.assertEqual(task.last_trained_batch, 10)

    def test_experimental_stage_two_state_is_archived(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir) / "data"
            state_dir = data_root / "task" / "1"
            state_dir.mkdir(parents=True)
            state_path = state_dir / "training_state.json"
            state_path.write_text(json.dumps({
                "schema_version": 1,
                "curriculum_schema": 1,
                "last_successful_stage": 2,
                "last_trained_batch": 10,
            }), encoding="utf-8")
            instance = Controller()
            task = FuzzerTask(
                task_id="fuzzer@task@1", task_name="task", run_id=1,
                fuzzer_id="fuzzer", mode="direct", callback_addr="localhost:1",
            )
            with mock.patch.object(
                    controller_module.config, "data_root", str(data_root)):
                instance._load_training_state(task)

            self.assertEqual(task.last_successful_stage, 0)
            self.assertEqual(task.last_trained_batch, 0)
            self.assertFalse(state_path.exists())
            self.assertEqual(len(list(state_dir.glob(
                "training_state.three-stage-incompatible-*.json"
            ))), 1)

    def test_unknown_training_state_schema_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir) / "data"
            state_dir = data_root / "task" / "1"
            state_dir.mkdir(parents=True)
            (state_dir / "training_state.json").write_text(json.dumps({
                "schema_version": 99,
                "curriculum_schema": 2,
                "last_successful_stage": 2,
                "last_trained_batch": 10,
            }), encoding="utf-8")
            instance = Controller()
            task = FuzzerTask(
                task_id="fuzzer@task@1", task_name="task", run_id=1,
                fuzzer_id="fuzzer", mode="direct", callback_addr="localhost:1",
            )
            with mock.patch.object(
                    controller_module.config, "data_root", str(data_root)):
                instance._load_training_state(task)

            self.assertEqual(task.last_successful_stage, 0)
            self.assertEqual(task.last_trained_batch, 0)

    def test_rejected_evaluation_watermark_survives_restart(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir) / "data"
            instance = Controller()
            original = FuzzerTask(
                task_id="fuzzer@task@1", task_name="task", run_id=1,
                fuzzer_id="fuzzer", mode="direct", callback_addr="localhost:1",
                evaluation_watermarks={
                    1: {
                        "batch_end": 7,
                        "disposition": "rejected",
                        "reason_codes": ["low_class_recall"],
                    }
                },
            )
            recovered = FuzzerTask(
                task_id="fuzzer@task@1", task_name="task", run_id=1,
                fuzzer_id="fuzzer", mode="direct", callback_addr="localhost:1",
            )
            with mock.patch.object(
                    controller_module.config, "data_root", str(data_root)):
                instance._persist_training_state(original)
                instance._load_training_state(recovered)

            self.assertEqual(
                recovered.evaluation_watermarks[1]["batch_end"], 7
            )
            self.assertEqual(
                recovered.evaluation_watermarks[1]["reason_codes"],
                ["low_class_recall"],
            )

    def test_non_object_evaluation_watermarks_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir) / "data"
            state_dir = data_root / "task" / "1"
            state_dir.mkdir(parents=True)
            (state_dir / "training_state.json").write_text(json.dumps({
                "schema_version": 1,
                "curriculum_schema": 2,
                "last_successful_stage": 1,
                "last_trained_batch": 7,
                "evaluation_watermarks": [],
            }), encoding="utf-8")
            instance = Controller()
            task = FuzzerTask(
                task_id="fuzzer@task@1", task_name="task", run_id=1,
                fuzzer_id="fuzzer", mode="direct", callback_addr="localhost:1",
            )
            with mock.patch.object(
                    controller_module.config, "data_root", str(data_root)):
                instance._load_training_state(task)

            self.assertEqual(task.last_successful_stage, 0)
            self.assertEqual(task.last_trained_batch, 0)

    def test_stop_training_does_not_signal_reaped_successful_launcher(self):
        instance = Controller()
        task = FuzzerTask(
            task_id="fuzzer@task@1", task_name="task", run_id=1,
            fuzzer_id="fuzzer", mode="direct", callback_addr="localhost:1",
            trainer_proc=CompletedProcess(), training_in_progress=True,
            active_training_id="training-1", training_gpu="0",
            training_port=29999,
        )

        with (
            mock.patch.object(controller_module.utils, "kill_process") as kill,
            mock.patch.object(instance, "_release_training_gpu") as release_gpu,
            mock.patch.object(instance, "_release_port") as release_port,
        ):
            self.assertTrue(instance._stop_task_training(task))

        kill.assert_not_called()
        release_gpu.assert_called_once_with("0")
        release_port.assert_called_once_with(instance.training_port_pool, 29999)
        self.assertIsNone(task.trainer_proc)

    def test_stop_training_retains_slot_when_group_cleanup_is_unconfirmed(self):
        instance = Controller()
        process = FakeProcess()
        task = FuzzerTask(
            task_id="fuzzer@task@1", task_name="task", run_id=1,
            fuzzer_id="fuzzer", mode="direct", callback_addr="localhost:1",
            trainer_proc=process, training_in_progress=True,
            active_training_id="training-1", training_gpu="0",
            training_port=29999,
        )

        with (
            mock.patch.object(
                controller_module.utils, "kill_process", return_value=False
            ) as kill,
            mock.patch.object(instance, "_release_training_gpu") as release_gpu,
            mock.patch.object(instance, "_release_port") as release_port,
        ):
            self.assertFalse(instance._stop_task_training(task))

        kill.assert_called_once_with(process, process_group=True)
        release_gpu.assert_not_called()
        release_port.assert_not_called()
        self.assertIs(task.trainer_proc, process)
        self.assertEqual(task.training_gpu, "0")
        self.assertEqual(task.training_port, 29999)
        self.assertTrue(task.training_in_progress)

    def test_unregister_retains_task_when_trainer_cleanup_is_unconfirmed(self):
        instance = Controller()
        task = FuzzerTask(
            task_id="fuzzer@task@1", task_name="task", run_id=1,
            fuzzer_id="fuzzer", mode="direct", callback_addr="localhost:1",
        )
        instance.global_tasks[task.task_id] = task

        with mock.patch.object(
            instance, "_stop_task_training", return_value=False
        ):
            with self.assertRaises(controller_module.HTTPException) as raised:
                asyncio.run(instance.unregister(task.task_id))

        self.assertEqual(raised.exception.status_code, 503)
        self.assertIs(instance.global_tasks[task.task_id], task)

    def test_unregister_retains_task_and_ports_when_receiver_cleanup_fails(self):
        instance = Controller()
        receiver = FakeProcess()
        task = FuzzerTask(
            task_id="fuzzer@task@1", task_name="task", run_id=1,
            fuzzer_id="fuzzer", mode="direct", callback_addr="localhost:1",
            receiver_proc=receiver, grpc_port=31001,
        )
        instance.global_tasks[task.task_id] = task

        with (
            mock.patch.object(
                controller_module.utils, "kill_process", return_value=False
            ) as kill,
            mock.patch.object(instance, "_release_port") as release_port,
        ):
            with self.assertRaises(controller_module.HTTPException) as raised:
                asyncio.run(instance.unregister(task.task_id))

        self.assertEqual(raised.exception.status_code, 503)
        self.assertIs(instance.global_tasks[task.task_id], task)
        self.assertIs(task.receiver_proc, receiver)
        self.assertTrue(task.stopping)
        kill.assert_called_once_with(receiver)
        release_port.assert_not_called()

        with self.assertRaises(controller_module.HTTPException) as blocked:
            asyncio.run(instance.start_trainer(
                task.task_id, batch_start=0, batch_end=3, stage=1
            ))
        self.assertEqual(blocked.exception.status_code, 409)

    def test_stop_training_waits_for_launch_owner_before_cleanup(self):
        instance = Controller()
        process = FakeProcess()
        task = FuzzerTask(
            task_id="fuzzer@task@1", task_name="task", run_id=1,
            fuzzer_id="fuzzer", mode="direct", callback_addr="localhost:1",
            trainer_proc=process, training_in_progress=True,
            active_training_id="training-1", training_gpu="0",
            training_port=29999,
        )
        task.training_launch_done.clear()

        cleanup_started = threading.Event()
        result = []

        def stop_training():
            cleanup_started.set()
            result.append(instance._stop_task_training(task))

        with (
            mock.patch.object(
                controller_module.utils, "kill_process", return_value=True
            ) as kill,
            mock.patch.object(instance, "_release_training_gpu"),
            mock.patch.object(instance, "_release_port"),
        ):
            worker = threading.Thread(target=stop_training)
            worker.start()
            self.assertTrue(cleanup_started.wait(timeout=1))
            self.assertFalse(task.training_launch_done.is_set())
            self.assertTrue(task.training_cancel.is_set())
            kill.assert_not_called()
            task.training_launch_done.set()
            worker.join(timeout=5)

        self.assertFalse(worker.is_alive())
        self.assertEqual(result, [True])
        kill.assert_called_once_with(process, process_group=True)

    def test_unregister_cancels_training_dataset_preflight(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir) / "data"
            data_dir = data_root / "task" / "1"
            data_dir.mkdir(parents=True)
            for batch_id in (1, 2, 3):
                write_balanced_committed_batch(data_dir, batch_id)

            instance = Controller()
            task = FuzzerTask(
                task_id="fuzzer@task@1", task_name="task", run_id=1,
                fuzzer_id="fuzzer", mode="direct",
                callback_addr="localhost:1", grpc_port=31001,
            )
            instance.global_tasks[task.task_id] = task
            preflight_started = threading.Event()
            launch_errors = []
            real_pickle_load = pickle.load
            first_load = True

            def cancellable_pickle_load(file_handle):
                nonlocal first_load
                value = real_pickle_load(file_handle)
                if first_load:
                    first_load = False
                    preflight_started.set()
                    self.assertTrue(task.training_cancel.wait(timeout=5))
                return value

            def launch_training():
                try:
                    asyncio.run(instance.start_trainer(
                        task.task_id, batch_start=0, batch_end=3, stage=1
                    ))
                except Exception as error:
                    launch_errors.append(error)

            with (
                mock.patch.object(
                    controller_module.config, "data_root", str(data_root)
                ),
                mock.patch.object(
                    pickle, "load", side_effect=cancellable_pickle_load
                ),
                mock.patch.object(instance, "_release_port"),
                mock.patch.object(instance, "_remove_ssh_entry"),
            ):
                worker = threading.Thread(target=launch_training)
                worker.start()
                self.assertTrue(preflight_started.wait(timeout=5))

                response = asyncio.run(instance.unregister(task.task_id))
                worker.join(timeout=5)

            self.assertFalse(worker.is_alive())
            self.assertEqual(response, {"status": "ok"})
            self.assertNotIn(task.task_id, instance.global_tasks)
            self.assertTrue(task.training_cancel.is_set())
            self.assertTrue(task.training_launch_done.is_set())
            self.assertEqual(len(launch_errors), 1)
            self.assertIsInstance(
                launch_errors[0], controller_module.HTTPException
            )
            self.assertEqual(launch_errors[0].status_code, 409)

    def test_unregister_maps_gpu_wait_cancellation_to_conflict(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir) / "data"
            data_dir = data_root / "task" / "1"
            data_dir.mkdir(parents=True)
            for batch_id in (1, 2, 3):
                write_balanced_committed_batch(data_dir, batch_id)

            instance = Controller()
            task = FuzzerTask(
                task_id="fuzzer@task@1", task_name="task", run_id=1,
                fuzzer_id="fuzzer", mode="direct",
                callback_addr="localhost:1", grpc_port=31001,
            )
            instance.global_tasks[task.task_id] = task
            gpu_wait_started = threading.Event()
            launch_errors = []

            async def wait_for_cancel(cancel_event):
                gpu_wait_started.set()
                while not cancel_event.is_set():
                    await asyncio.sleep(0.01)
                return None

            def launch_training():
                try:
                    asyncio.run(instance.start_trainer(
                        task.task_id, batch_start=0, batch_end=3, stage=1
                    ))
                except Exception as error:
                    launch_errors.append(error)

            with (
                mock.patch.object(
                    controller_module.config, "data_root", str(data_root)
                ),
                mock.patch.object(
                    instance, "_acquire_training_gpu_async",
                    side_effect=wait_for_cancel,
                ),
                mock.patch.object(instance, "_release_port"),
                mock.patch.object(instance, "_remove_ssh_entry"),
            ):
                worker = threading.Thread(target=launch_training)
                worker.start()
                self.assertTrue(gpu_wait_started.wait(timeout=5))

                response = asyncio.run(instance.unregister(task.task_id))
                worker.join(timeout=5)

            self.assertFalse(worker.is_alive())
            self.assertEqual(response, {"status": "ok"})
            self.assertEqual(len(launch_errors), 1)
            self.assertIsInstance(
                launch_errors[0], controller_module.HTTPException
            )
            self.assertEqual(launch_errors[0].status_code, 409)

    def test_shutdown_stops_receiver_when_trainer_cleanup_is_unconfirmed(self):
        instance = Controller()
        trainer = FakeProcess()
        receiver = FakeProcess()
        attributor = FakeProcess()
        task = FuzzerTask(
            task_id="fuzzer@task@1", task_name="task", run_id=1,
            fuzzer_id="fuzzer", mode="direct", callback_addr="localhost:1",
            trainer_proc=trainer, receiver_proc=receiver,
            attributor_proc=attributor,
        )
        instance.global_tasks[task.task_id] = task

        with (
            mock.patch.object(
                instance, "_stop_task_training", return_value=False
            ),
            mock.patch.object(
                controller_module.utils, "kill_process", return_value=True
            ) as kill,
            mock.patch.object(instance, "_clear_authorized_keys"),
            mock.patch.object(
                instance, "_ensure_torchserve_stopped"
            ) as stop_torchserve,
            self.assertRaisesRegex(RuntimeError, "1 cleanup error"),
        ):
            instance.shutdown()

        self.assertIs(instance.global_tasks[task.task_id], task)
        self.assertEqual(
            kill.call_args_list,
            [mock.call(attributor), mock.call(receiver)],
        )
        stop_torchserve.assert_called_once_with()

    def test_shutdown_attempts_torchserve_after_task_cleanup_exception(self):
        instance = Controller()
        task = FuzzerTask(
            task_id="fuzzer@task@1", task_name="task", run_id=1,
            fuzzer_id="fuzzer", mode="direct", callback_addr="localhost:1",
        )
        instance.global_tasks[task.task_id] = task

        with (
            mock.patch.object(
                instance, "_remove_training_waiter",
                side_effect=RuntimeError("broken task state"),
            ),
            mock.patch.object(
                instance, "_stop_task_training", return_value=True
            ) as stop_training,
            mock.patch.object(
                instance, "_stop_task_auxiliary_processes", return_value=True
            ) as stop_auxiliary,
            mock.patch.object(controller_module.config, "direct_only", True),
            mock.patch.object(
                instance, "_ensure_torchserve_stopped"
            ) as stop_torchserve,
            self.assertRaisesRegex(RuntimeError, "1 cleanup error"),
        ):
            instance.shutdown()

        stop_training.assert_called_once_with(task)
        stop_auxiliary.assert_called_once_with(task)
        stop_torchserve.assert_called_once_with()

    def test_shutdown_reports_unconfirmed_auxiliary_cleanup(self):
        instance = Controller()
        task = FuzzerTask(
            task_id="fuzzer@task@1", task_name="task", run_id=1,
            fuzzer_id="fuzzer", mode="direct", callback_addr="localhost:1",
        )
        instance.global_tasks[task.task_id] = task

        with (
            mock.patch.object(instance, "_stop_task_training",
                              return_value=True),
            mock.patch.object(instance, "_stop_task_auxiliary_processes",
                              return_value=False),
            mock.patch.object(controller_module.config, "direct_only", True),
            mock.patch.object(
                instance, "_ensure_torchserve_stopped"
            ) as stop_torchserve,
            self.assertRaisesRegex(RuntimeError, "1 cleanup error"),
        ):
            instance.shutdown()

        self.assertIs(instance.global_tasks[task.task_id], task)
        stop_torchserve.assert_called_once_with()

    def test_shutdown_attempts_auxiliary_after_trainer_cleanup_raises(self):
        instance = Controller()
        task = FuzzerTask(
            task_id="fuzzer@task@1", task_name="task", run_id=1,
            fuzzer_id="fuzzer", mode="direct", callback_addr="localhost:1",
        )
        instance.global_tasks[task.task_id] = task

        with (
            mock.patch.object(
                instance, "_stop_task_training",
                side_effect=RuntimeError("trainer cleanup crashed"),
            ),
            mock.patch.object(
                instance, "_stop_task_auxiliary_processes", return_value=True
            ) as stop_auxiliary,
            mock.patch.object(controller_module.config, "direct_only", True),
            mock.patch.object(instance, "_ensure_torchserve_stopped"),
            self.assertRaisesRegex(RuntimeError, "2 cleanup error"),
        ):
            instance.shutdown()

        stop_auxiliary.assert_called_once_with(task)
        self.assertIs(instance.global_tasks[task.task_id], task)

    def test_shutdown_reports_ssh_cleanup_failure_after_torchserve_attempt(self):
        instance = Controller()
        instance.existing_ssh_entries.add("managed-key")
        ssh_error = subprocess.CalledProcessError(1, ["helper", "--clear"])
        with (
            mock.patch.object(controller_module.config, "direct_only", False),
            mock.patch.object(instance, "_clear_authorized_keys",
                              side_effect=ssh_error),
            mock.patch.object(
                instance, "_ensure_torchserve_stopped"
            ) as stop_torchserve,
            self.assertRaisesRegex(RuntimeError, "1 cleanup error"),
        ):
            instance.shutdown()

        self.assertTrue(instance._shutdown_cleanup_failed)
        self.assertEqual(instance.existing_ssh_entries, {"managed-key"})
        stop_torchserve.assert_called_once_with()

    def test_authorized_key_cleanup_clears_owned_entry_state(self):
        instance = Controller()
        instance.existing_ssh_entries.update(("key-a", "key-b"))
        with mock.patch.object(controller_module.utils, "run_cmd") as run_cmd:
            instance._clear_authorized_keys()

        run_cmd.assert_called_once_with([
            "sudo", "-n", "-u", controller_module.TUNNEL_USER,
            controller_module.HELPER_SCRIPT_PATH,
            controller_module.TUNNEL_USER, "--clear",
        ])
        self.assertEqual(instance.existing_ssh_entries, set())

    def test_partial_torchserve_start_is_retried_by_controller_cleanup(self):
        instance = Controller()
        with tempfile.TemporaryDirectory() as temp_dir:
            operator = controller_module.ServeOperator(
                temp_dir, disable_auth=True
            )
            operator.cwd = temp_dir
            config_path = operator.create_config()
            instance.torchserve_operator = operator
            launcher = mock.Mock(pid=1234)
            runtime_dir = os.path.join(temp_dir, "runtime")
            os.mkdir(runtime_dir)
            stop_attempts = 0

            def stop_with_first_failure():
                nonlocal stop_attempts
                stop_attempts += 1
                if stop_attempts == 1:
                    raise RuntimeError("first cleanup not confirmed")
                operator._owned_process = None
                operator._owned_java_process = None
                operator._cleanup_runtime_dir()

            with (
                mock.patch.object(operator, "create_config",
                                  return_value=config_path),
                mock.patch("ts_operators.subprocess.Popen",
                           return_value=launcher),
                mock.patch("ts_operators.tempfile.mkdtemp",
                           return_value=runtime_dir),
                mock.patch.object(operator, "_wait_for_management_api",
                                  return_value=False),
                mock.patch.object(operator, "stop_service",
                                  side_effect=stop_with_first_failure),
                self.assertRaisesRegex(RuntimeError,
                                      "first cleanup not confirmed"),
            ):
                instance._ensure_torchserve_started()

            self.assertFalse(instance.torchserve_started)
            self.assertTrue(operator.has_owned_service)
            with mock.patch.object(
                    operator, "stop_service",
                    side_effect=stop_with_first_failure) as retry:
                instance._ensure_torchserve_stopped()

            retry.assert_called_once_with()
            self.assertEqual(stop_attempts, 2)
            self.assertFalse(operator.has_owned_service)

    def test_run_translates_sighup_into_graceful_server_shutdown(self):
        if not hasattr(controller_module.signal, "SIGHUP"):
            self.skipTest("SIGHUP is unavailable on this platform")

        instance = Controller()
        instance.torchserve_started = True
        previous_sighup = controller_module.signal.getsignal(
            controller_module.signal.SIGHUP
        )

        class FakeServer:
            should_exit = False

            def run(self):
                handler = controller_module.signal.getsignal(
                    controller_module.signal.SIGHUP
                )
                handler(controller_module.signal.SIGHUP, None)
                self.assert_shutdown_requested()

            def assert_shutdown_requested(self):
                if not self.should_exit:
                    raise AssertionError("SIGHUP did not request server shutdown")

        server = FakeServer()

        def shutdown_during_second_sighup():
            server.should_exit = False
            handler = controller_module.signal.getsignal(
                controller_module.signal.SIGHUP
            )
            handler(controller_module.signal.SIGHUP, None)
            self.assertTrue(server.should_exit)

        with (
            mock.patch.object(controller_module.uvicorn, "Config"),
            mock.patch.object(
                controller_module.uvicorn, "Server", return_value=server
            ),
            mock.patch.object(
                instance,
                "shutdown",
                side_effect=shutdown_during_second_sighup,
            ) as shutdown,
        ):
            instance.run(port=48200)

        self.assertTrue(server.should_exit)
        shutdown.assert_called_once_with()
        self.assertIs(
            controller_module.signal.getsignal(controller_module.signal.SIGHUP),
            previous_sighup,
        )

    def test_run_cleans_live_resources_after_server_loop_failure(self):
        instance = Controller()
        instance.torchserve_started = True
        server = mock.Mock()
        server.run.side_effect = RuntimeError("server loop failed")

        with (
            mock.patch.object(controller_module.uvicorn, "Config"),
            mock.patch.object(
                controller_module.uvicorn, "Server", return_value=server
            ),
            mock.patch.object(instance, "shutdown") as shutdown,
            self.assertRaisesRegex(RuntimeError, "server loop failed"),
        ):
            instance.run(port=48200)

        shutdown.assert_called_once_with()

    def test_run_retries_partially_started_owned_torchserve(self):
        instance = Controller()
        instance.torchserve_started = False
        instance.torchserve_operator = mock.Mock(has_owned_service=True)
        server = mock.Mock()

        with (
            mock.patch.object(controller_module.uvicorn, "Config"),
            mock.patch.object(
                controller_module.uvicorn, "Server", return_value=server
            ),
            mock.patch.object(instance, "shutdown") as shutdown,
        ):
            instance.run(port=48200)

        shutdown.assert_called_once_with()

    def test_run_retries_ssh_only_lifespan_cleanup_failure(self):
        instance = Controller()
        instance.existing_ssh_entries.add("managed-key")
        clear_attempts = 0

        def clear_with_first_failure():
            nonlocal clear_attempts
            clear_attempts += 1
            if clear_attempts == 1:
                raise RuntimeError("temporary SSH helper failure")
            instance.existing_ssh_entries.clear()

        class FakeServer:
            should_exit = False

            def run(self):
                try:
                    instance.shutdown()
                except RuntimeError:
                    # Uvicorn records a lifespan shutdown failure and returns.
                    pass

        with (
            mock.patch.object(controller_module.config, "direct_only", False),
            mock.patch.object(controller_module.uvicorn, "Config"),
            mock.patch.object(controller_module.uvicorn, "Server",
                              return_value=FakeServer()),
            mock.patch.object(instance, "_clear_authorized_keys",
                              side_effect=clear_with_first_failure),
            mock.patch.object(instance, "_ensure_torchserve_stopped"),
        ):
            instance.run(port=48200)

        self.assertEqual(clear_attempts, 2)
        self.assertFalse(instance._shutdown_cleanup_failed)
        self.assertEqual(instance.existing_ssh_entries, set())

    def test_run_restores_sighup_handler_when_fallback_cleanup_fails(self):
        if not hasattr(controller_module.signal, "SIGHUP"):
            self.skipTest("SIGHUP is unavailable on this platform")

        instance = Controller()
        instance.torchserve_started = True
        previous_sighup = controller_module.signal.getsignal(
            controller_module.signal.SIGHUP
        )
        server = mock.Mock()

        with (
            mock.patch.object(controller_module.uvicorn, "Config"),
            mock.patch.object(
                controller_module.uvicorn, "Server", return_value=server
            ),
            mock.patch.object(
                instance,
                "shutdown",
                side_effect=RuntimeError("cleanup failed"),
            ),
            self.assertRaisesRegex(RuntimeError, "cleanup failed"),
        ):
            instance.run(port=48200)

        self.assertIs(
            controller_module.signal.getsignal(controller_module.signal.SIGHUP),
            previous_sighup,
        )

    def test_canceled_launch_registers_unconfirmed_resources_for_retry(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir) / "data"
            log_root = Path(temp_dir) / "logs"
            data_dir = data_root / "task" / "1"
            data_dir.mkdir(parents=True)
            for batch_id in (1, 2, 3):
                write_balanced_committed_batch(data_dir, batch_id)

            instance = Controller()
            task = FuzzerTask(
                task_id="fuzzer@task@1", task_name="task", run_id=1,
                fuzzer_id="fuzzer", mode="direct", callback_addr="localhost:1",
            )
            instance.global_tasks[task.task_id] = task
            process = FakeProcess()

            def cancel_during_popen(*args, **kwargs):
                with task.training_lock:
                    task.active_training_id = ""
                    task.training_cancel.set()
                return process

            with (
                mock.patch.object(controller_module.config, "data_root", str(data_root)),
                mock.patch.object(controller_module.config, "log_dir", str(log_root)),
                mock.patch.object(instance, "_acquire_training_gpu", return_value="0"),
                mock.patch.object(instance, "_alloc_port", return_value=29999),
                mock.patch.object(
                    controller_module.subprocess,
                    "Popen",
                    side_effect=cancel_during_popen,
                ),
                mock.patch.object(
                    controller_module.utils, "kill_process", return_value=False
                ),
                mock.patch.object(instance, "_release_training_gpu") as release_gpu,
                mock.patch.object(instance, "_release_port") as release_port,
            ):
                with self.assertRaises(controller_module.HTTPException):
                    asyncio.run(instance.start_trainer(
                        task.task_id, batch_start=0, batch_end=3, stage=1
                    ))

            self.assertTrue(task.training_launch_done.is_set())
            self.assertIs(task.trainer_proc, process)
            self.assertEqual(task.training_gpu, "0")
            self.assertEqual(task.training_port, 29999)
            self.assertTrue(task.training_in_progress)
            self.assertTrue(task.active_training_id)
            release_gpu.assert_not_called()
            release_port.assert_not_called()
            task.trainer_log_fh.close()
            task.trainer_log_fh = None

    def test_launch_rollback_opens_barrier_when_log_close_raises(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir) / "data"
            log_root = Path(temp_dir) / "logs"
            data_dir = data_root / "task" / "1"
            data_dir.mkdir(parents=True)
            for batch_id in (1, 2, 3):
                write_balanced_committed_batch(data_dir, batch_id)

            instance = Controller()
            task = FuzzerTask(
                task_id="fuzzer@task@1", task_name="task", run_id=1,
                fuzzer_id="fuzzer", mode="direct", callback_addr="localhost:1",
            )
            instance.global_tasks[task.task_id] = task
            process = FakeProcess()
            bad_log = CloseFailureLog()
            real_open = open

            def selective_open(path, *args, **kwargs):
                if str(path).endswith("trainer.log"):
                    return bad_log
                return real_open(path, *args, **kwargs)

            with (
                mock.patch.object(controller_module.config, "data_root", str(data_root)),
                mock.patch.object(controller_module.config, "log_dir", str(log_root)),
                mock.patch.object(instance, "_acquire_training_gpu", return_value="0"),
                mock.patch.object(instance, "_alloc_port", return_value=29999),
                mock.patch("builtins.open", side_effect=selective_open),
                mock.patch.object(
                    controller_module.subprocess, "Popen", return_value=process
                ),
                mock.patch.object(
                    instance, "_start_daemon_thread",
                    side_effect=RuntimeError("simulated thread start failure"),
                ),
                mock.patch.object(
                    controller_module.utils, "kill_process", return_value=True
                ) as kill,
                mock.patch.object(instance, "_release_training_gpu") as release_gpu,
                mock.patch.object(instance, "_release_port") as release_port,
            ):
                with self.assertRaises(controller_module.HTTPException):
                    asyncio.run(instance.start_trainer(
                        task.task_id, batch_start=0, batch_end=3, stage=1
                    ))

            self.assertTrue(task.training_launch_done.is_set())
            self.assertIsNone(task.trainer_proc)
            self.assertIsNone(task.trainer_log_fh)
            self.assertEqual(task.training_gpu, "")
            self.assertEqual(task.training_port, 0)
            self.assertFalse(task.training_in_progress)
            self.assertEqual(task.active_training_id, "")
            kill.assert_called_once_with(process, process_group=True)
            release_gpu.assert_called_once_with("0")
            release_port.assert_called_once_with(instance.training_port_pool, 29999)

    def test_failed_launcher_cleans_group_before_releasing_training_slot(self):
        instance = Controller()
        process = FailedProcess()
        task = FuzzerTask(
            task_id="fuzzer@task@1", task_name="task", run_id=1,
            fuzzer_id="fuzzer", mode="direct", callback_addr="localhost:1",
            trainer_proc=process, training_in_progress=True,
            active_training_id="training-1", training_gpu="0",
            training_port=29999,
        )
        instance.global_tasks[task.task_id] = task

        events = []
        with (
            mock.patch.object(
                controller_module.utils,
                "kill_process",
                side_effect=lambda *args, **kwargs: events.append("kill") or True,
            ) as kill,
            mock.patch.object(
                instance,
                "_release_training_gpu",
                side_effect=lambda gpu: events.append("release_gpu"),
            ),
            mock.patch.object(
                instance,
                "_release_port",
                side_effect=lambda pool, port: events.append("release_port"),
            ),
        ):
            instance._monitor_training_and_deploy(
                task, "training-1", process, 1, 3, "/unused",
                29999, "0", 1, 7,
            )

        kill.assert_called_once_with(process, process_group=True)
        self.assertEqual(events[0], "kill")
        self.assertCountEqual(events[1:], ["release_gpu", "release_port"])
        self.assertIsNone(task.trainer_proc)
        self.assertFalse(task.training_in_progress)

    def test_committed_batch_listing_ignores_partial_and_future_batches(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            for batch_id in (1, 2, 4):
                write_committed_batch(data_dir, batch_id)
            (data_dir / "batch_2.complete").unlink()

            self.assertEqual(list_committed_batch_indices(temp_dir, 3), [1])
            self.assertEqual(list_committed_batch_indices(temp_dir, 0), [1, 4])

    def test_guidance_loader_deduplicates_cross_batch_labels(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir)
            data_dir = data_root / "task" / "1"
            data_dir.mkdir(parents=True)
            write_committed_batch(
                data_dir, 1, labels=[False, False, True]
            )
            first_program = "test$1()"
            first_signature = hashlib.sha1(first_program.encode("utf-8")).hexdigest()
            with (data_dir / "progs_batch_2.pkl").open("wb") as handle:
                pickle.dump({first_signature: first_program}, handle)
            with (data_dir / "labels_batch_2.pkl").open("wb") as handle:
                pickle.dump({first_signature: [False, True, False]}, handle)
            (data_dir / "batch_2.complete").write_text(
                "\n".join((
                    "schema_version=1", "batch_id=2", "unique_samples=1",
                    "wire_samples=1", f"batch_digest={'1' * 64}", "",
                )),
                encoding="utf-8",
            )
            task = FuzzerTask(
                task_id="fuzzer@task@1", task_name="task", run_id=1,
                fuzzer_id="fuzzer", mode="direct", callback_addr="localhost:1",
            )
            instance = Controller()
            with mock.patch.object(controller_module.config, "data_root", str(data_root)):
                programs, labels = instance._load_training_data(task)

            self.assertEqual(programs, [first_program])
            self.assertEqual(labels, [[False, False, True]])

    def test_continued_split_uses_only_new_data_plus_old_replay(self):
        train_indices, test_indices = build_training_split(
            list(range(1, 16)),
            last_trained_batch=10,
            full_retrain=False,
            seed="stable",
        )
        new_train = [index for index in train_indices if index > 10]
        old_replay = [index for index in train_indices if index <= 10]
        self.assertEqual(set(new_train), {11, 12, 13, 14})
        self.assertEqual(test_indices, [15])
        self.assertEqual(len(old_replay), 2)
        self.assertTrue(set(old_replay).issubset(set(range(1, 11))))

    def test_committed_listing_rejects_corrupt_payload(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            write_committed_batch(data_dir, 11)
            (data_dir / "labels_batch_11.pkl").write_bytes(b"not-a-pickle")
            self.assertEqual(list_committed_batch_indices(temp_dir), [])

    def test_manifest_loader_selects_best_not_largest_step(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            best = model_dir / "step-200.pt"
            best.write_bytes(b"best")
            (model_dir / "step-400.pt").write_bytes(b"worse-final")
            manifest = {
                "schema_version": 1,
                "curriculum_schema": 2,
                "train_stage": 1,
                "num_classes": 3,
                "best_checkpoint": str(best),
                "checkpoint_sha256": hashlib.sha256(b"best").hexdigest(),
                "best_step": 200,
                "best_eval_loss": 0.2,
            }
            (model_dir / "training_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )

            _, loaded, checkpoint = Controller._load_training_manifest(
                temp_dir, stage=1, num_classes=3
            )
            self.assertEqual(loaded["best_step"], 200)
            self.assertEqual(checkpoint, best)

            best.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                Controller._load_training_manifest(temp_dir, stage=1, num_classes=3)

            manifest["best_eval_loss"] = math.nan
            best.write_bytes(b"best")
            (model_dir / "training_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "not finite"):
                Controller._load_training_manifest(temp_dir, stage=1, num_classes=3)

    def test_stage_two_request_launches_mandatory_stage_one_without_committing_round(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir) / "data"
            log_root = Path(temp_dir) / "logs"
            data_dir = data_root / "task" / "1"
            data_dir.mkdir(parents=True)
            for batch_id in (1, 2, 3):
                write_balanced_committed_batch(data_dir, batch_id)

            instance = Controller()
            task = FuzzerTask(
                task_id="fuzzer@task@1",
                task_name="task",
                run_id=1,
                fuzzer_id="fuzzer",
                mode="direct",
                callback_addr="localhost:1",
                model_name_prefix="reach_filter_test",
            )
            instance.global_tasks[task.task_id] = task
            fake_process = FakeProcess()
            with (
                mock.patch.object(controller_module.config, "data_root", str(data_root)),
                mock.patch.object(controller_module.config, "log_dir", str(log_root)),
                mock.patch.object(instance, "_acquire_training_gpu", return_value="0"),
                mock.patch.object(instance, "_alloc_port", return_value=29999),
                mock.patch.object(controller_module.subprocess, "Popen", return_value=fake_process) as popen,
                mock.patch.object(instance, "_start_daemon_thread") as start_thread,
            ):
                response = asyncio.run(instance.start_trainer(
                    task.task_id, batch_start=0, batch_end=3, stage=2
                ))

            command = popen.call_args.args[0]
            launch_env = popen.call_args.kwargs["env"]
            self.assertTrue(popen.call_args.kwargs["start_new_session"])
            self.assertEqual(launch_env["TOKENIZERS_PARALLELISM"], "true")
            self.assertEqual(
                launch_env["RAYON_NUM_THREADS"],
                str(controller_module.config.tokenizer_rayon_threads),
            )
            cache_index = command.index("--token_cache_entries") + 1
            self.assertEqual(
                command[cache_index],
                str(controller_module.config.token_cache_entries),
            )
            accumulation_index = command.index("--grad_acc_steps") + 1
            self.assertEqual(
                command[accumulation_index],
                str(controller_module.config.grad_acc_steps),
            )
            for option, expected in (
                ("--learning_rate", controller_module.config.learning_rate),
                ("--weight_decay", controller_module.config.weight_decay),
                ("--num_warmup_steps", controller_module.config.num_warmup_steps),
                ("--total_steps", controller_module.config.first_train_total_steps),
                (
                    "--test_interval",
                    controller_module.config.first_train_test_interval,
                ),
                ("--min_steps", controller_module.config.first_train_min_steps),
                ("--patience", controller_module.config.first_train_patience),
                ("--assigned_physical_gpu", "0"),
            ):
                option_index = command.index(option) + 1
                self.assertEqual(command[option_index], str(expected))
            stage_index = command.index("--train_stage") + 1
            self.assertEqual(command[stage_index], "1")
            self.assertIn("--is_first_train", command)
            self.assertNotIn("--load_path", command)
            self.assertEqual(response["stage"], 1)
            self.assertEqual(task.training_round, 0)
            self.assertTrue(task.training_in_progress)
            start_thread.assert_called_once()

            task.trainer_log_fh.close()
            task.trainer_log_fh = None
            task.training_in_progress = False
            task.active_training_id = ""

    def test_continued_training_excludes_all_checkpoint_history_from_test(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir) / "data"
            log_root = Path(temp_dir) / "logs"
            data_dir = data_root / "task" / "1"
            data_dir.mkdir(parents=True)
            for batch_id in range(1, 16):
                write_balanced_committed_batch(data_dir, batch_id)
            checkpoint = Path(temp_dir) / "previous.pt"
            checkpoint.write_bytes(b"checkpoint")

            instance = Controller()
            task = FuzzerTask(
                task_id="fuzzer@task@1", task_name="task", run_id=1,
                fuzzer_id="fuzzer", mode="direct", callback_addr="localhost:1",
                model_name_prefix="reach_filter_test", training_round=1,
                last_trained_batch=10, last_successful_stage=1,
                last_successful_checkpoint=str(checkpoint),
                seen_training_batches=list(range(1, 11)),
            )
            instance.global_tasks[task.task_id] = task
            fake_process = FakeProcess()
            with (
                mock.patch.object(controller_module.config, "data_root", str(data_root)),
                mock.patch.object(controller_module.config, "log_dir", str(log_root)),
                mock.patch.object(instance, "_acquire_training_gpu", return_value="0"),
                mock.patch.object(instance, "_alloc_port", return_value=29999),
                mock.patch.object(
                    controller_module.subprocess, "Popen", return_value=fake_process
                ) as popen,
                mock.patch.object(instance, "_start_daemon_thread"),
            ):
                asyncio.run(instance.start_trainer(
                    task.task_id, batch_start=11, batch_end=15, stage=1
                ))

            command = popen.call_args.args[0]
            canonical = command[command.index("--canonical_data_idx") + 1]
            excluded = command[command.index("--test_exclude_data_idx") + 1]
            test_indices = command[command.index("--test_data_idx") + 1]
            self.assertEqual(
                set(map(int, canonical.split(","))), set(range(1, 16))
            )
            self.assertEqual(
                set(map(int, excluded.split(","))), set(range(1, 15))
            )
            self.assertEqual(test_indices, "15")

            task.trainer_log_fh.close()
            task.trainer_log_fh = None
            task.training_in_progress = False
            task.active_training_id = ""

    def test_stage_two_same_boundary_reuses_unseen_holdout(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir) / "data"
            log_root = Path(temp_dir) / "logs"
            data_dir = data_root / "task" / "1"
            data_dir.mkdir(parents=True)
            for batch_id in (1, 2, 3):
                write_balanced_committed_batch(data_dir, batch_id)
            checkpoint = Path(temp_dir) / "stage1.pt"
            checkpoint.write_bytes(b"checkpoint")

            instance = Controller()
            task = FuzzerTask(
                task_id="fuzzer@task@1", task_name="task", run_id=1,
                fuzzer_id="fuzzer", mode="direct", callback_addr="localhost:1",
                model_name_prefix="reach_filter_test", training_round=1,
                last_trained_batch=3, last_successful_stage=1,
                last_successful_checkpoint=str(checkpoint),
                seen_training_batches=[1, 2],
            )
            instance.global_tasks[task.task_id] = task
            fake_process = FakeProcess()
            with (
                mock.patch.object(controller_module.config, "data_root", str(data_root)),
                mock.patch.object(controller_module.config, "log_dir", str(log_root)),
                mock.patch.object(instance, "_acquire_training_gpu", return_value="0"),
                mock.patch.object(instance, "_alloc_port", return_value=29999),
                mock.patch.object(
                    controller_module.subprocess, "Popen", return_value=fake_process
                ) as popen,
                mock.patch.object(instance, "_start_daemon_thread"),
            ):
                response = asyncio.run(instance.start_trainer(
                    task.task_id, batch_start=0, batch_end=3, stage=2
                ))

            command = popen.call_args.args[0]
            excluded = command[command.index("--test_exclude_data_idx") + 1]
            test_indices = command[command.index("--test_data_idx") + 1]
            self.assertEqual(response["stage"], 2)
            self.assertEqual(set(map(int, excluded.split(","))), {1, 2})
            self.assertEqual(test_indices, "3")

            task.trainer_log_fh.close()
            task.trainer_log_fh = None
            task.training_in_progress = False
            task.active_training_id = ""

    def test_single_waypoint_stage_two_launch_uses_exact_objective(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir) / "data"
            log_root = Path(temp_dir) / "logs"
            data_dir = data_root / "task" / "1"
            data_dir.mkdir(parents=True)
            for batch_id in (1, 2, 3):
                write_balanced_committed_batch(
                    data_dir, batch_id, num_classes=2
                )
            checkpoint = Path(temp_dir) / "stage1.pt"
            checkpoint.write_bytes(b"checkpoint")

            instance = Controller()
            task = FuzzerTask(
                task_id="fuzzer@task@1", task_name="task", run_id=1,
                fuzzer_id="fuzzer", mode="direct", callback_addr="localhost:1",
                model_name_prefix="reach_filter_test", training_round=1,
                last_trained_batch=3, last_successful_stage=1,
                last_successful_checkpoint=str(checkpoint),
                seen_training_batches=[1, 2],
            )
            instance.global_tasks[task.task_id] = task
            fake_process = FakeProcess()
            with (
                mock.patch.object(controller_module.config, "data_root", str(data_root)),
                mock.patch.object(controller_module.config, "log_dir", str(log_root)),
                mock.patch.object(instance, "_acquire_training_gpu", return_value="0"),
                mock.patch.object(instance, "_alloc_port", return_value=29999),
                mock.patch.object(
                    controller_module.subprocess, "Popen", return_value=fake_process
                ) as popen,
                mock.patch.object(instance, "_start_daemon_thread"),
            ):
                response = asyncio.run(instance.start_trainer(
                    task.task_id, batch_start=0, batch_end=3, stage=2
                ))

            command = popen.call_args.args[0]
            selected_stage = command[command.index("--train_stage") + 1]
            self.assertEqual(response["stage"], 2)
            self.assertEqual(selected_stage, "2")

            task.trainer_log_fh.close()
            task.trainer_log_fh = None
            task.training_in_progress = False
            task.active_training_id = ""

    def test_trainer_defers_snapshot_without_disjoint_validation_signatures(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir) / "data"
            data_dir = data_root / "task" / "1"
            data_dir.mkdir(parents=True)
            write_committed_batch(data_dir, 1)
            program = "test$1()"
            signature = hashlib.sha1(program.encode("utf-8")).hexdigest()
            with (data_dir / "progs_batch_2.pkl").open("wb") as handle:
                pickle.dump({signature: program}, handle)
            with (data_dir / "labels_batch_2.pkl").open("wb") as handle:
                pickle.dump({signature: [True, False, False]}, handle)
            (data_dir / "batch_2.complete").write_text("\n".join((
                "schema_version=1", "batch_id=2", "unique_samples=1",
                "wire_samples=1", f"batch_digest={'2' * 64}", "",
            )), encoding="utf-8")
            instance = Controller()
            task = FuzzerTask(
                task_id="fuzzer@task@1", task_name="task", run_id=1,
                fuzzer_id="fuzzer", mode="direct", callback_addr="localhost:1",
            )
            instance.global_tasks[task.task_id] = task

            with (
                mock.patch.object(controller_module.config, "data_root", str(data_root)),
                mock.patch.object(instance, "_acquire_training_gpu") as acquire_gpu,
            ):
                response = asyncio.run(instance.start_trainer(
                    task.task_id, batch_start=0, batch_end=2, stage=1
                ))

            self.assertEqual(response["status"], "deferred")
            self.assertFalse(task.training_in_progress)
            acquire_gpu.assert_not_called()

    def test_trainer_defers_snapshot_with_only_one_new_batch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir) / "data"
            data_dir = data_root / "task" / "1"
            data_dir.mkdir(parents=True)
            write_committed_batch(data_dir, 1)
            instance = Controller()
            task = FuzzerTask(
                task_id="fuzzer@task@1", task_name="task", run_id=1,
                fuzzer_id="fuzzer", mode="direct", callback_addr="localhost:1",
            )
            instance.global_tasks[task.task_id] = task

            with (
                mock.patch.object(controller_module.config, "data_root", str(data_root)),
                mock.patch.object(instance, "_acquire_training_gpu") as acquire_gpu,
            ):
                response = asyncio.run(instance.start_trainer(
                    task.task_id, batch_start=0, batch_end=1, stage=1
                ))

            self.assertEqual(response["status"], "deferred")
            self.assertIn("at least 2", response["reason"])
            self.assertFalse(task.training_in_progress)
            acquire_gpu.assert_not_called()

    def test_trainer_defers_when_split_omits_active_curriculum_classes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir) / "data"
            data_dir = data_root / "task" / "1"
            data_dir.mkdir(parents=True)
            write_committed_batch(
                data_dir, 1, labels=[True, False, False]
            )
            write_committed_batch(
                data_dir, 2, labels=[True, False, False]
            )
            write_committed_batch(
                data_dir, 3, labels=[False, True, False]
            )
            instance = Controller()
            task = FuzzerTask(
                task_id="fuzzer@task@1", task_name="task", run_id=1,
                fuzzer_id="fuzzer", mode="direct",
                callback_addr="localhost:1",
            )
            instance.global_tasks[task.task_id] = task

            with (
                mock.patch.object(
                    controller_module.config, "data_root", str(data_root)
                ),
                mock.patch.object(instance, "_acquire_training_gpu") as acquire_gpu,
            ):
                response = asyncio.run(instance.start_trainer(
                    task.task_id, batch_start=0, batch_end=3, stage=1
                ))

            self.assertEqual(response["status"], "deferred")
            self.assertIn("curriculum", response["reason"])
            self.assertEqual(response["train_class_counts"], {0: 2, 1: 0})
            self.assertEqual(
                response["validation_class_counts"], {0: 0, 1: 1}
            )
            self.assertFalse(task.training_in_progress)
            acquire_gpu.assert_not_called()

    def test_manual_attributor_endpoint_fails_closed(self):
        instance = Controller()
        task = FuzzerTask(
            task_id="fuzzer@task@1", task_name="task", run_id=1,
            fuzzer_id="fuzzer", mode="direct", callback_addr="localhost:1",
        )
        instance.global_tasks[task.task_id] = task

        with (
            mock.patch.object(controller_module.subprocess, "Popen") as popen,
            self.assertRaises(controller_module.HTTPException) as raised,
        ):
            asyncio.run(instance.start_attributor(
                task.task_id, model_version=1, top_k=10
            ))

        self.assertEqual(raised.exception.status_code, 501)
        popen.assert_not_called()

    def test_task_log_api_reads_bounded_file_backed_tail(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            instance = Controller()
            task = FuzzerTask(
                task_id="fuzzer@task name@1", task_name="task name", run_id=1,
                fuzzer_id="fuzzer", mode="direct", callback_addr="localhost:1",
            )
            instance.global_tasks[task.task_id] = task
            log_dir = Path(temp_dir) / task.log_dir_name / "1"
            log_dir.mkdir(parents=True)
            (log_dir / "receiver.log").write_text(
                "line one\nline two\nline three\n", encoding="utf-8"
            )

            with mock.patch.object(
                controller_module.config, "log_dir", temp_dir
            ):
                response = asyncio.run(instance.get_task_logs(
                    task.task_id, "receiver", lines=2
                ))

            self.assertEqual(response["logs"], ["line two", "line three"])
            self.assertEqual(response["total"], 2)
            self.assertEqual(response["source"], "file")

    def test_task_log_routes_require_dashboard_token(self):
        from fastapi.testclient import TestClient

        with tempfile.TemporaryDirectory() as temp_dir:
            instance = Controller()
            task = FuzzerTask(
                task_id="fuzzer@task@1", task_name="task", run_id=1,
                fuzzer_id="fuzzer", mode="direct", callback_addr="localhost:1",
            )
            instance.global_tasks[task.task_id] = task
            log_dir = Path(temp_dir) / task.log_dir_name / "1"
            log_dir.mkdir(parents=True)
            (log_dir / "receiver.log").write_text("secret\n", encoding="utf-8")

            with mock.patch.object(controller_module.config, "log_dir", temp_dir):
                client = TestClient(instance.app)
                missing = client.get(
                    f"/api/task/{task.task_id}/logs/receiver"
                )
                authorized = client.get(
                    f"/api/task/{task.task_id}/logs/receiver",
                    headers={
                        "Authorization": (
                            f"Bearer {controller_module.config.dashboard_token}"
                        )
                    },
                )

            self.assertEqual(missing.status_code, 401)
            self.assertEqual(authorized.status_code, 200)
            self.assertEqual(authorized.json()["logs"], ["secret"])

    def test_task_log_paths_are_collision_resistant(self):
        first = FuzzerTask(
            task_id="fuzzer-a@case 1@1", task_name="case 1", run_id=1,
            fuzzer_id="fuzzer-a", mode="direct", callback_addr="localhost:1",
        )
        second = FuzzerTask(
            task_id="fuzzer-b@case_1@1", task_name="case_1", run_id=1,
            fuzzer_id="fuzzer-b", mode="direct", callback_addr="localhost:1",
        )
        self.assertNotEqual(first.log_dir_name, second.log_dir_name)

    def test_task_log_tail_preserves_complete_boundary_line_and_cr_records(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            task = FuzzerTask(
                task_id="fuzzer@task@1", task_name="task", run_id=1,
                fuzzer_id="fuzzer", mode="direct", callback_addr="localhost:1",
            )
            log_dir = Path(temp_dir) / task.log_dir_name / "1"
            log_dir.mkdir(parents=True)
            log_path = log_dir / "trainer.log"
            tail = b"first\rsecond\rthird"
            log_path.write_bytes(b"discarded\n" + tail)

            with mock.patch.object(controller_module.config, "log_dir", temp_dir):
                lines = controller_module.read_task_log_tail(
                    task, "trainer", lines=3, max_bytes=len(tail)
                )

            self.assertEqual(lines, ["first", "second", "third"])

    def test_task_log_tail_consumes_crlf_at_truncation_boundaries(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            task = FuzzerTask(
                task_id="fuzzer@task@1", task_name="task", run_id=1,
                fuzzer_id="fuzzer", mode="direct", callback_addr="localhost:1",
            )
            log_dir = Path(temp_dir) / task.log_dir_name / "1"
            log_dir.mkdir(parents=True)
            log_path = log_dir / "trainer.log"
            window = b"\nfirst\r\nsecond"
            log_path.write_bytes(b"discarded\r" + window)

            with mock.patch.object(controller_module.config, "log_dir", temp_dir):
                boundary_lines = controller_module.read_task_log_tail(
                    task, "trainer", lines=3, max_bytes=len(window)
                )
                log_path.write_bytes(b"partial-text\r\nfirst")
                partial_lines = controller_module.read_task_log_tail(
                    task, "trainer", lines=3, max_bytes=len(b"text\r\nfirst")
                )

            self.assertEqual(boundary_lines, ["first", "second"])
            self.assertEqual(partial_lines, ["first"])

    def test_task_detail_reports_file_log_sizes_in_bytes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            instance = Controller()
            task = FuzzerTask(
                task_id="fuzzer@task@1", task_name="task", run_id=1,
                fuzzer_id="fuzzer", mode="direct", callback_addr="localhost:1",
            )
            instance.global_tasks[task.task_id] = task
            log_dir = Path(temp_dir) / task.log_dir_name / "1"
            log_dir.mkdir(parents=True)
            (log_dir / "receiver.log").write_bytes(b"receiver")
            (log_dir / "trainer.log").write_bytes(b"training-data")
            task.attributor_logs.extend(("one", "二"))

            with mock.patch.object(controller_module.config, "log_dir", temp_dir):
                response = asyncio.run(instance.get_task_detail(task.task_id))

            self.assertEqual(response["log_sizes"]["unit"], "bytes")
            self.assertEqual(response["log_sizes"]["receiver"], 8)
            self.assertEqual(response["log_sizes"]["trainer"], 13)
            self.assertEqual(response["log_sizes"]["attributor"], 6)

    def test_task_log_api_rejects_unbounded_line_request(self):
        instance = Controller()
        task = FuzzerTask(
            task_id="fuzzer@task@1", task_name="task", run_id=1,
            fuzzer_id="fuzzer", mode="direct", callback_addr="localhost:1",
        )
        instance.global_tasks[task.task_id] = task

        with self.assertRaises(controller_module.HTTPException) as raised:
            asyncio.run(instance.get_task_logs(
                task.task_id, "receiver", lines=1001
            ))

        self.assertEqual(raised.exception.status_code, 400)

    def test_monitor_commits_manifest_checkpoint_and_monotonic_deployment(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            checkpoint = model_dir / "step-200.pt"
            checkpoint.write_bytes(b"best")
            manifest = {
                "schema_version": 1,
                "curriculum_schema": 2,
                "train_stage": 1,
                "num_classes": 3,
                "best_checkpoint": str(checkpoint),
                "checkpoint_sha256": hashlib.sha256(b"best").hexdigest(),
                "best_step": 200,
                "best_eval_loss": 0.3,
                "seen_train_batch_indices": [1, 2, 3, 4, 5, 6],
                **passing_stage_one_promotion_metrics(),
            }
            (model_dir / "training_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )

            instance = Controller()
            task = FuzzerTask(
                task_id="fuzzer@task@1",
                task_name="task",
                run_id=1,
                fuzzer_id="fuzzer",
                mode="direct",
                callback_addr="localhost:1",
                model_name_prefix="reach_filter_test",
                training_in_progress=True,
                active_training_id="training-1",
                training_gpu="0",
                training_port=29999,
                trainer_proc=CompletedProcess(),
            )
            instance.global_tasks[task.task_id] = task
            with (
                mock.patch.object(
                    instance, "_release_training_gpu"
                ) as release_gpu,
                mock.patch.object(instance, "_release_port"),
                mock.patch.object(instance, "_persist_training_state"),
                mock.patch.object(
                    instance, "_export_torchscript"
                ) as export,
                mock.patch.object(instance, "_deploy_to_torchserve") as deploy,
                mock.patch.object(instance, "_notify_fuzzer_model_ready") as notify,
            ):
                def export_while_gpu_is_reserved(*_args):
                    self.assertEqual(task.training_gpu, "0")
                    release_gpu.assert_not_called()
                    return "model.pt"

                export.side_effect = export_while_gpu_is_reserved
                instance._monitor_training_and_deploy(
                    task,
                    "training-1",
                    task.trainer_proc,
                    1,
                    3,
                    str(model_dir),
                    29999,
                    "0",
                    1,
                    7,
                )

            self.assertEqual(task.training_round, 1)
            self.assertEqual(task.last_trained_batch, 7)
            self.assertEqual(task.last_successful_checkpoint, str(checkpoint))
            self.assertEqual(task.last_successful_stage, 1)
            self.assertEqual(task.model_name, "reach_filter_test_r1")
            self.assertEqual(task.model_version, "1")
            self.assertEqual(task.deployment_version, 1)
            self.assertEqual(task.pending_model_name, "")
            self.assertFalse(task.training_in_progress)
            export.assert_called_once_with(
                task, checkpoint, 3, 1, "reach_filter_test_r1", "0"
            )
            release_gpu.assert_called_once_with("0")
            deploy.assert_called_once_with(
                task, "reach_filter_test_r1", "1", "model.pt", 3, 1
            )
            notify.assert_called_once_with(task, "reach_filter_test_r1", "1")

    def test_monitor_rejects_low_quality_candidate_before_export(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            checkpoint = model_dir / "step-200.pt"
            checkpoint.write_bytes(b"weak")
            weak_metrics = passing_stage_one_promotion_metrics()
            weak_metrics.update({
                "best_eval_accuracy": 0.5,
                "best_eval_macro_f1": 0.33,
                "validation_per_class_recall": {"0": 1.0, "1": 0.0},
            })
            manifest = {
                "schema_version": 1,
                "curriculum_schema": 2,
                "train_stage": 1,
                "num_classes": 3,
                "best_checkpoint": str(checkpoint),
                "checkpoint_sha256": hashlib.sha256(b"weak").hexdigest(),
                "best_step": 200,
                "best_eval_loss": 0.7,
                "seen_train_batch_indices": [1, 2, 3],
                **weak_metrics,
            }
            (model_dir / "training_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            instance = Controller()
            task = FuzzerTask(
                task_id="fuzzer@task@1", task_name="task", run_id=1,
                fuzzer_id="fuzzer", mode="direct", callback_addr="localhost:1",
                model_name_prefix="reach_filter_test",
                training_in_progress=True,
                active_training_id="training-1",
                training_gpu="0",
                training_port=29999,
                trainer_proc=CompletedProcess(),
            )
            instance.global_tasks[task.task_id] = task
            with (
                mock.patch.object(instance, "_persist_training_state"),
                mock.patch.object(instance, "_release_training_resources"),
                mock.patch.object(instance, "_prune_unselected_checkpoints"),
                mock.patch.object(instance, "_export_torchscript") as export,
                mock.patch.object(instance, "_deploy_to_torchserve") as deploy,
                mock.patch.object(instance, "_notify_fuzzer_model_ready") as notify,
            ):
                instance._monitor_training_and_deploy(
                    task, "training-1", task.trainer_proc, 1, 3,
                    str(model_dir), 29999, "0", 1, 7,
                )

            export.assert_not_called()
            deploy.assert_not_called()
            notify.assert_not_called()
            self.assertEqual(task.training_round, 0)
            self.assertEqual(task.last_successful_stage, 0)
            self.assertEqual(
                task.evaluation_watermarks[1]["disposition"], "rejected"
            )
            decision = json.loads((
                model_dir / "promotion_decision.json"
            ).read_text(encoding="utf-8"))
            self.assertFalse(decision["accepted"])
            self.assertIn("low_class_recall", decision["reason_codes"])

    def test_invalid_promotion_evidence_is_terminal_for_snapshot(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            checkpoint = model_dir / "step-200.pt"
            checkpoint.write_bytes(b"best")
            manifest = {
                "schema_version": 1,
                "curriculum_schema": 2,
                "train_stage": 1,
                "num_classes": 3,
                "best_checkpoint": str(checkpoint),
                "checkpoint_sha256": hashlib.sha256(b"best").hexdigest(),
                "best_step": 200,
                "best_eval_loss": 0.3,
                "seen_train_batch_indices": [1, 2, 3],
                **passing_stage_one_promotion_metrics(),
            }
            (model_dir / "training_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            instance = Controller()
            task = FuzzerTask(
                task_id="fuzzer@task@1", task_name="task", run_id=1,
                fuzzer_id="fuzzer", mode="direct", callback_addr="localhost:1",
                training_in_progress=True,
                active_training_id="training-1",
                training_gpu="0",
                training_port=29999,
                trainer_proc=CompletedProcess(),
            )
            instance.global_tasks[task.task_id] = task
            with (
                mock.patch.object(instance, "_persist_training_state"),
                mock.patch.object(instance, "_release_training_resources"),
                mock.patch.object(instance, "_export_torchscript") as export,
            ):
                instance._monitor_training_and_deploy(
                    task, "training-1", task.trainer_proc, 1, 3,
                    str(model_dir), 29999, "0", 1, 7,
                    200, "b" * 64,
                )

            export.assert_not_called()
            self.assertEqual(
                task.evaluation_watermarks[1]["reason_codes"],
                ["invalid_evidence"],
            )
            decision = json.loads((
                model_dir / "promotion_decision.json"
            ).read_text(encoding="utf-8"))
            self.assertEqual(decision["reason_codes"], ["invalid_evidence"])
            self.assertIn("fingerprint", decision["evidence_error"])

            response = asyncio.run(instance.start_trainer(
                task.task_id, batch_start=0, batch_end=7, stage=1
            ))
            self.assertEqual(response["status"], "rejected")
            self.assertEqual(response["batch_end"], 7)

    def test_notification_failure_keeps_pending_without_consuming_boundary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            checkpoint = model_dir / "step-200.pt"
            checkpoint.write_bytes(b"best")
            manifest = {
                "schema_version": 1,
                "curriculum_schema": 2,
                "train_stage": 1,
                "num_classes": 3,
                "best_checkpoint": str(checkpoint),
                "checkpoint_sha256": hashlib.sha256(b"best").hexdigest(),
                "best_step": 200,
                "best_eval_loss": 0.3,
                "seen_train_batch_indices": [1, 2, 3, 4, 5, 6],
                **passing_stage_one_promotion_metrics(),
            }
            (model_dir / "training_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            instance = Controller()
            task = FuzzerTask(
                task_id="fuzzer@task@1", task_name="task", run_id=1,
                fuzzer_id="fuzzer", mode="direct",
                callback_addr="localhost:1",
                model_name_prefix="reach_filter_test",
                training_in_progress=True,
                active_training_id="training-1",
                training_gpu="0", training_port=29999,
                trainer_proc=CompletedProcess(),
            )
            instance.global_tasks[task.task_id] = task
            with (
                mock.patch.object(instance, "_release_training_gpu"),
                mock.patch.object(instance, "_release_port"),
                mock.patch.object(instance, "_persist_training_state"),
                mock.patch.object(
                    instance, "_export_torchscript", return_value="model.pt"
                ) as export,
                mock.patch.object(instance, "_deploy_to_torchserve"),
                mock.patch.object(
                    instance, "_notify_fuzzer_model_ready",
                    side_effect=RuntimeError("callback response unknown"),
                ),
                mock.patch.object(
                    instance.torchserve_operator, "unregister_model"
                ) as unregister,
            ):
                instance._monitor_training_and_deploy(
                    task, "training-1", task.trainer_proc, 1, 3,
                    str(model_dir), 29999, "0", 1, 7,
                )

            export.assert_called_once_with(
                task, checkpoint, 3, 1, "reach_filter_test_r1", "0"
            )

            self.assertEqual(task.training_round, 0)
            self.assertEqual(task.last_trained_batch, 0)
            self.assertEqual(task.last_successful_stage, 0)
            self.assertEqual(task.model_name, "")
            self.assertEqual(task.pending_model_name, "reach_filter_test_r1")
            self.assertEqual(task.pending_batch, 7)
            unregister.assert_not_called()

    def test_export_failure_releases_reserved_training_gpu(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            checkpoint = model_dir / "step-200.pt"
            checkpoint.write_bytes(b"best")
            manifest = {
                "schema_version": 1,
                "curriculum_schema": 2,
                "train_stage": 1,
                "num_classes": 3,
                "best_checkpoint": str(checkpoint),
                "checkpoint_sha256": hashlib.sha256(b"best").hexdigest(),
                "best_step": 200,
                "best_eval_loss": 0.3,
                "seen_train_batch_indices": [1, 2, 3],
                **passing_stage_one_promotion_metrics(),
            }
            (model_dir / "training_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            instance = Controller()
            task = FuzzerTask(
                task_id="fuzzer@task@1", task_name="task", run_id=1,
                fuzzer_id="fuzzer", mode="direct",
                callback_addr="localhost:1",
                model_name_prefix="reach_filter_test",
                training_in_progress=True,
                active_training_id="training-1",
                training_gpu="2", training_port=29999,
                trainer_proc=CompletedProcess(),
            )
            instance.global_tasks[task.task_id] = task
            with (
                mock.patch.object(
                    instance, "_release_training_gpu"
                ) as release_gpu,
                mock.patch.object(
                    instance, "_release_port"
                ) as release_port,
                mock.patch.object(
                    instance, "_export_torchscript",
                    side_effect=RuntimeError("trace failed"),
                ) as export,
                mock.patch.object(
                    instance, "_deploy_to_torchserve"
                ) as deploy,
            ):
                instance._monitor_training_and_deploy(
                    task, "training-1", task.trainer_proc, 1, 3,
                    str(model_dir), 29999, "2", 1, 7,
                )

            export.assert_called_once_with(
                task, checkpoint, 3, 1, "reach_filter_test_r1", "2"
            )
            release_gpu.assert_called_once_with("2")
            release_port.assert_called_once_with(
                instance.training_port_pool, 29999
            )
            deploy.assert_not_called()
            self.assertEqual(task.training_gpu, "")
            self.assertEqual(task.training_port, 0)
            self.assertFalse(task.training_in_progress)
            self.assertEqual(task.pending_model_name, "")

    def test_pending_deployment_request_retries_without_launching_trainer(self):
        instance = Controller()
        task = FuzzerTask(
            task_id="fuzzer@task@1", task_name="task", run_id=1,
            fuzzer_id="fuzzer", mode="direct", callback_addr="localhost:1",
            pending_model_name="candidate", pending_model_version="1",
            pending_checkpoint="best.pt", pending_manifest="manifest.json",
            pending_torchscript="candidate.pt",
            pending_stage=1, pending_batch=7, pending_training_round=1,
            pending_num_classes=3,
            pending_seen_training_batches=[1, 2, 3, 4, 5, 6],
        )
        instance.global_tasks[task.task_id] = task
        with mock.patch.object(instance, "_start_daemon_thread") as start_thread:
            response = asyncio.run(instance.start_trainer(
                task.task_id, batch_start=0, batch_end=7, stage=1
            ))
        self.assertTrue(response["deployment_retry"])
        self.assertEqual(response["batch_end"], 7)
        self.assertTrue(task.training_in_progress)
        start_thread.assert_called_once_with(
            instance._retry_pending_deployment,
            task,
            mock.ANY,
        )

    def test_training_state_recovers_seen_batches_from_manifests(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir) / "data"
            state_dir = data_root / "task" / "1"
            state_dir.mkdir(parents=True)
            active_manifest = Path(temp_dir) / "active.json"
            pending_manifest = Path(temp_dir) / "pending.json"
            active_manifest.write_text(json.dumps({
                "seen_train_batch_indices": [1, 2, 4],
            }), encoding="utf-8")
            pending_manifest.write_text(json.dumps({
                "seen_train_batch_indices": [1, 2, 4, 5],
            }), encoding="utf-8")
            (state_dir / "training_state.json").write_text(json.dumps({
                "schema_version": 1,
                "last_trained_batch": 4,
                "last_training_manifest": str(active_manifest),
                "pending_model_name": "candidate",
                "pending_manifest": str(pending_manifest),
            }), encoding="utf-8")

            instance = Controller()
            task = FuzzerTask(
                task_id="fuzzer@task@1", task_name="task", run_id=1,
                fuzzer_id="fuzzer", mode="direct", callback_addr="localhost:1",
            )
            with (
                mock.patch.object(controller_module.config, "data_root", str(data_root)),
                mock.patch.object(instance, "_persist_training_state") as persist,
            ):
                instance._load_training_state(task)

            self.assertEqual(task.seen_training_batches, [1, 2, 4])
            self.assertEqual(task.pending_seen_training_batches, [1, 2, 4, 5])
            persist.assert_called_once_with(task)

    def test_recovered_state_persist_failure_does_not_abort_loading(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir) / "data"
            state_dir = data_root / "task" / "1"
            state_dir.mkdir(parents=True)
            (state_dir / "training_state.json").write_text(json.dumps({
                "schema_version": 1,
                "last_trained_batch": 3,
            }), encoding="utf-8")
            instance = Controller()
            task = FuzzerTask(
                task_id="fuzzer@task@1", task_name="task", run_id=1,
                fuzzer_id="fuzzer", mode="direct", callback_addr="localhost:1",
            )

            with (
                mock.patch.object(controller_module.config, "data_root", str(data_root)),
                mock.patch.object(
                    instance, "_persist_training_state",
                    side_effect=OSError("disk full"),
                ),
            ):
                instance._load_training_state(task)

            self.assertEqual(task.seen_training_batches, [1, 2, 3])

    def test_pending_retry_rejects_missing_seen_batch_membership(self):
        instance = Controller()
        task = FuzzerTask(
            task_id="fuzzer@task@1", task_name="task", run_id=1,
            fuzzer_id="fuzzer", mode="direct", callback_addr="localhost:1",
            pending_model_name="candidate", pending_model_version="1",
            pending_checkpoint="best.pt", pending_manifest="manifest.json",
            pending_torchscript="candidate.pt", pending_stage=1,
            pending_batch=7, pending_training_round=1, pending_num_classes=3,
        )
        instance.global_tasks[task.task_id] = task

        with self.assertRaisesRegex(Exception, "incomplete pending deployment state"):
            asyncio.run(instance.start_trainer(
                task.task_id, batch_start=0, batch_end=7, stage=1
            ))
        self.assertFalse(task.training_in_progress)

    def test_start_trainer_rechecks_registration_while_claiming_lifecycle(self):
        instance = Controller()
        task = FuzzerTask(
            task_id="fuzzer@task@1", task_name="task", run_id=1,
            fuzzer_id="fuzzer", mode="direct", callback_addr="localhost:1",
        )
        instance.global_tasks[task.task_id] = task
        task.lifecycle_lock.acquire()
        result = {}

        def run_start():
            try:
                asyncio.run(instance.start_trainer(
                    task.task_id, batch_start=0, batch_end=1, stage=1
                ))
            except Exception as error:
                result["error"] = error

        worker = threading.Thread(target=run_start)
        worker.start()
        try:
            instance.global_tasks.pop(task.task_id)
        finally:
            task.lifecycle_lock.release()
        worker.join(timeout=5)

        self.assertFalse(worker.is_alive())
        self.assertEqual(getattr(result.get("error"), "status_code", None), 404)
        self.assertFalse(task.training_in_progress)

    def test_pending_deployment_recovery_runs_guidance_after_commit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "model"
            model_dir.mkdir()
            manifest_path = model_dir / "training_manifest.json"
            checkpoint_path = model_dir / "step-200.pt"
            instance = Controller()
            task = FuzzerTask(
                task_id="fuzzer@task@1", task_name="task", run_id=1,
                fuzzer_id="fuzzer", mode="direct", callback_addr="localhost:1",
                guidance_engine=object(), training_in_progress=True,
                active_training_id="retry-1",
                pending_model_name="candidate", pending_model_version="1",
                pending_checkpoint=str(checkpoint_path),
                pending_manifest=str(manifest_path),
                pending_torchscript=str(model_dir / "candidate.pt"),
                pending_stage=1, pending_batch=7, pending_training_round=1,
                pending_num_classes=3,
                pending_seen_training_batches=[1, 2, 3, 4, 5, 6],
            )
            instance.global_tasks[task.task_id] = task
            with (
                mock.patch.object(instance, "_notify_fuzzer_model_ready"),
                mock.patch.object(instance, "_persist_training_state"),
                mock.patch.object(instance, "_run_guidance_pipeline") as guidance,
            ):
                instance._retry_pending_deployment(task, "retry-1")

            guidance.assert_called_once_with(
                task, 1, 3, str(model_dir), str(checkpoint_path)
            )
            self.assertEqual(task.model_name, "candidate")
            self.assertEqual(task.pending_model_name, "")
            self.assertFalse(task.training_in_progress)

    def test_stage_one_guidance_skips_multiclass_attribution(self):
        instance = Controller()
        engine = mock.Mock()
        engine.version = 1
        engine.get_static_weights.return_value = {}
        engine.get_attribution_weights.return_value = {}
        engine.compute_guidance.return_value = {
            "version": 1,
            "syscall_weights": {},
            "mutation_templates": [],
        }
        task = FuzzerTask(
            task_id="fuzzer@task@1", task_name="task", run_id=1,
            fuzzer_id="fuzzer", mode="direct", callback_addr="localhost:1",
            guidance_engine=engine, static_analysis_done=True,
        )
        instance.global_tasks[task.task_id] = task
        labels = [[False, True, False] for _ in range(100)]
        programs = ["getpid()" for _ in labels]
        with (
            mock.patch.object(
                instance, "_load_training_data", return_value=(programs, labels)
            ),
            mock.patch.object(instance, "_run_attribution_analysis") as attribution,
            mock.patch.object(instance, "_run_sequence_mining") as sequence,
            mock.patch.object(
                instance, "_send_guidance_if_active", return_value=True
            ),
        ):
            instance._run_guidance_pipeline_locked(
                task, stage=1, num_classes=3,
                save_dir="model", ckpt_path="checkpoint.pt",
            )

        attribution.assert_not_called()
        sequence.assert_called_once()
        engine.update_attribution.assert_called_once_with({})

    def test_attribution_requires_selected_checkpoint_quality(self):
        instance = Controller()
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "training_manifest.json"
            manifest_path.write_text(json.dumps({
                "best_eval_accuracy": 0.84,
                "best_eval_weighted_f1": 0.95,
                "validation_class_counts": {"0": 10, "1": 5, "2": 5},
            }), encoding="utf-8")
            self.assertFalse(
                instance._model_quality_allows_attribution(temp_dir, 2, 3)
            )
            manifest_path.write_text(json.dumps({
                "best_eval_accuracy": 0.85,
                "best_eval_weighted_f1": 0.80,
                "validation_class_counts": {"0": 10, "1": 5, "2": 5},
            }), encoding="utf-8")
            self.assertTrue(
                instance._model_quality_allows_attribution(temp_dir, 2, 3)
            )
            manifest_path.write_text(json.dumps({
                "best_eval_accuracy": 2.0,
                "best_eval_weighted_f1": 0.80,
                "validation_class_counts": {"0": 10, "1": 5, "2": 5},
            }), encoding="utf-8")
            self.assertFalse(
                instance._model_quality_allows_attribution(temp_dir, 2, 3)
            )
            manifest_path.write_text(json.dumps({
                "best_eval_accuracy": 0.90,
                "best_eval_weighted_f1": 0.80,
                "validation_class_counts": {"0": 10, "1": 5, "2": 0},
            }), encoding="utf-8")
            self.assertFalse(
                instance._model_quality_allows_attribution(temp_dir, 2, 3)
            )

    def test_stage_two_guidance_uses_final_target_attribution(self):
        instance = Controller()
        engine = mock.Mock()
        engine.version = 1
        engine.get_static_weights.return_value = {}
        engine.get_attribution_weights.return_value = {}
        engine.compute_guidance.return_value = {
            "version": 1,
            "syscall_weights": {},
            "mutation_templates": [],
        }
        task = FuzzerTask(
            task_id="fuzzer@task@1", task_name="task", run_id=1,
            fuzzer_id="fuzzer", mode="direct", callback_addr="localhost:1",
            guidance_engine=engine, static_analysis_done=True,
        )
        instance.global_tasks[task.task_id] = task
        labels = [[False, False, True] for _ in range(100)]
        programs = ["getpid()" for _ in labels]
        with (
            mock.patch.object(
                instance, "_load_training_data", return_value=(programs, labels)
            ),
            mock.patch.object(
                instance, "_model_quality_allows_attribution", return_value=True
            ) as quality,
            mock.patch.object(instance, "_run_attribution_analysis") as attribution,
            mock.patch.object(instance, "_run_sequence_mining") as sequence,
            mock.patch.object(
                instance, "_send_guidance_if_active", return_value=True
            ),
        ):
            instance._run_guidance_pipeline_locked(
                task, stage=2, num_classes=3,
                save_dir="model", ckpt_path="checkpoint.pt",
            )

        attribution.assert_called_once_with(
            task, engine, "checkpoint.pt", 3, 2
        )
        quality.assert_called_once_with("model", 2, 3)
        sequence.assert_called_once_with(
            task, engine, programs, labels, 3, 2
        )
        engine.update_attribution.assert_called_once_with({})

    def test_stage_two_guidance_rejects_shallow_only_evidence(self):
        instance = Controller()
        engine = mock.Mock()
        engine.version = 1
        engine.get_static_weights.return_value = {}
        engine.get_attribution_weights.return_value = {}
        engine.compute_guidance.return_value = {
            "version": 1,
            "syscall_weights": {},
            "mutation_templates": [],
        }
        task = FuzzerTask(
            task_id="fuzzer@task@1", task_name="task", run_id=1,
            fuzzer_id="fuzzer", mode="direct", callback_addr="localhost:1",
            guidance_engine=engine, static_analysis_done=True,
        )
        instance.global_tasks[task.task_id] = task
        labels = [[False, True, False] for _ in range(100)]
        programs = ["getpid()" for _ in labels]
        with (
            mock.patch.object(
                instance, "_load_training_data", return_value=(programs, labels)
            ),
            mock.patch.object(
                instance, "_model_quality_allows_attribution", return_value=True
            ),
            mock.patch.object(instance, "_run_attribution_analysis") as attribution,
            mock.patch.object(instance, "_run_sequence_mining") as sequence,
            mock.patch.object(
                instance, "_send_guidance_if_active", return_value=True
            ),
        ):
            instance._run_guidance_pipeline_locked(
                task, stage=2, num_classes=3,
                save_dir="model", ckpt_path="checkpoint.pt",
            )

        attribution.assert_not_called()
        sequence.assert_not_called()
        engine.update_attribution.assert_called_once_with({})
        engine.update_sequence_patterns.assert_called_once_with([])

    def test_attribution_filters_manifest_before_top_k(self):
        instance = Controller()
        engine = mock.Mock()
        task = FuzzerTask(
            task_id="fuzzer@task@1", task_name="task", run_id=1,
            fuzzer_id="fuzzer", mode="direct", callback_addr="localhost:1",
            target_os="linux", target_arch="amd64",
            target_revision="target-r1", producer_revision="fuzzer-r1",
            descriptions_mode="manual",
        )
        instance.global_tasks[task.task_id] = task
        raw_scores = {
            **{f"unknown${index}": 100.0 - index for index in range(20)},
            "read": 3.0,
            "write": 4.0,
        }
        attribution_runner = mock.Mock(return_value=raw_scores)
        fake_attribution_module = mock.Mock(
            run_attribution_for_guidance=attribution_runner
        )
        filtered = mock.Mock(
            entries=(
                {"name": "read", "weight": 3.0},
                {"name": "write", "weight": 4.0},
            ),
            rejected={"unknown": 20},
        )
        manifest = mock.Mock()
        manifest.filter_generation_scores.return_value = filtered
        original_import = __import__

        def import_attribution(name, globals=None, locals=None,
                               fromlist=(), level=0):
            if name == "attribution_guidance":
                return fake_attribution_module
            return original_import(name, globals, locals, fromlist, level)

        with (
            mock.patch("builtins.__import__", side_effect=import_attribution),
            mock.patch.object(
                instance, "_get_data_dir", return_value="/data/task/1"
            ),
            mock.patch.object(
                controller_module, "list_committed_batch_indices",
                return_value=[1, 2],
            ),
            mock.patch.object(
                controller_module.SyzlangIndex, "load_cached",
                return_value=manifest,
            ) as load_manifest,
            mock.patch("torch.cuda.device_count", return_value=2),
            mock.patch("torch.cuda.device", return_value=mock.MagicMock()),
            mock.patch("torch.cuda.empty_cache"),
        ):
            instance._run_attribution_analysis_on_reserved_gpu(
                task, engine, "checkpoint.pt", 3, 2
            )

        self.assertEqual(len(
            manifest.filter_generation_scores.call_args.args[0]
        ), 22)
        manifest.filter_generation_scores.assert_called_once_with(
            raw_scores, descriptions_mode="manual"
        )
        load_manifest.assert_called_once_with(
            controller_module.config.syzlang_manifest_path,
            expected_os="linux",
            expected_arch="amd64",
            expected_revision="target-r1",
            expected_producer_revision="fuzzer-r1",
            max_bytes=controller_module.config.syzlang_manifest_max_bytes,
        )
        self.assertEqual(
            attribution_runner.call_args.kwargs["target_class"], 2
        )
        self.assertIsNone(attribution_runner.call_args.kwargs["top_k"])
        engine.update_attribution.assert_called_once_with({
            "write": 0.8,
            "read": 0.6,
        })

    def test_failed_attribution_quality_gate_clears_previous_snapshot(self):
        instance = Controller()
        engine = mock.Mock()
        engine.version = 1
        engine.get_static_weights.return_value = {}
        engine.get_attribution_weights.return_value = {}
        engine.compute_guidance.return_value = {
            "version": 1,
            "syscall_weights": {},
            "mutation_templates": [],
        }
        task = FuzzerTask(
            task_id="fuzzer@task@1", task_name="task", run_id=1,
            fuzzer_id="fuzzer", mode="direct", callback_addr="localhost:1",
            guidance_engine=engine, static_analysis_done=True,
        )
        instance.global_tasks[task.task_id] = task
        labels = [[False, True, False] for _ in range(100)]
        programs = ["getpid()" for _ in labels]
        with (
            mock.patch.object(
                instance, "_load_training_data", return_value=(programs, labels)
            ),
            mock.patch.object(
                instance, "_model_quality_allows_attribution", return_value=False
            ),
            mock.patch.object(instance, "_run_attribution_analysis") as attribution,
            mock.patch.object(instance, "_run_sequence_mining"),
            mock.patch.object(
                instance, "_send_guidance_if_active", return_value=True
            ),
        ):
            instance._run_guidance_pipeline_locked(
                task, stage=2, num_classes=3,
                save_dir="model", ckpt_path="checkpoint.pt",
            )

        attribution.assert_not_called()
        engine.update_attribution.assert_called_once_with({})

    def test_quality_gate_failure_sends_no_stale_attribution_weight(self):
        instance = Controller()
        engine = GuidanceEngine(GuidanceConfig())
        engine.update_attribution({"old$call": 1.0})
        task = FuzzerTask(
            task_id="fuzzer@task@1", task_name="task", run_id=1,
            fuzzer_id="fuzzer", mode="direct", callback_addr="localhost:1",
            guidance_engine=engine, static_analysis_done=True,
        )
        instance.global_tasks[task.task_id] = task
        labels = [[False, True, False] for _ in range(100)]
        programs = ["getpid()" for _ in labels]
        sent_payloads = []

        def capture_guidance(_task, _engine, guidance):
            sent_payloads.append(guidance)
            return True

        with (
            mock.patch.object(
                instance, "_load_training_data", return_value=(programs, labels)
            ),
            mock.patch.object(
                instance, "_model_quality_allows_attribution", return_value=False
            ),
            mock.patch.object(instance, "_run_attribution_analysis") as attribution,
            mock.patch.object(instance, "_run_sequence_mining"),
            mock.patch.object(
                instance, "_send_guidance_if_active",
                side_effect=capture_guidance,
            ),
        ):
            instance._run_guidance_pipeline_locked(
                task, stage=2, num_classes=3,
                save_dir="model", ckpt_path="checkpoint.pt",
            )

        attribution.assert_not_called()
        self.assertEqual(engine.get_attribution_weights(), {})
        self.assertEqual(len(sent_payloads), 1)
        self.assertNotIn("old$call", sent_payloads[0]["syscall_weights"])

    def test_model_notification_reconciles_lost_post_response(self):
        instance = Controller()
        task = FuzzerTask(
            task_id="fuzzer@task@1", task_name="task", run_id=1,
            fuzzer_id="fuzzer", mode="direct", callback_addr="localhost:1",
        )
        with (
            mock.patch("requests.post", side_effect=TimeoutError("lost")),
            mock.patch.object(
                instance, "_query_fuzzer_model", return_value=("candidate", "1")
            ),
        ):
            instance._notify_fuzzer_model_ready(task, "candidate", "1")

    def test_unregister_cancels_sleeping_registration_guidance(self):
        instance = Controller()
        engine = mock.Mock()
        task = FuzzerTask(
            task_id="fuzzer@task@1", task_name="task", run_id=1,
            fuzzer_id="fuzzer", mode="direct", callback_addr="localhost:1",
            report_path="report.txt", report_text="report",
            guidance_engine=engine, grpc_port=31001,
        )
        instance.global_tasks[task.task_id] = task
        with (
            mock.patch.object(instance, "_run_static_analysis") as static_analysis,
            mock.patch.object(instance, "_release_port"),
            mock.patch.object(instance, "_remove_ssh_entry"),
        ):
            worker = threading.Thread(
                target=instance._run_registration_guidance, args=(task,)
            )
            worker.start()
            asyncio.run(instance.unregister(task.task_id))
            worker.join(timeout=5)

        self.assertFalse(worker.is_alive())
        self.assertTrue(task.guidance_cancel.is_set())
        static_analysis.assert_not_called()
        engine.send_guidance.assert_not_called()

    def test_registration_guidance_accepts_kallgraph_only(self):
        instance = Controller()
        engine = mock.Mock()
        engine.version = 1
        engine.compute_guidance.return_value = {
            "syscall_weights": {"openat": 1.0},
            "mutation_templates": [],
        }
        task = FuzzerTask(
            task_id="fuzzer@task@1", task_name="task", run_id=1,
            fuzzer_id="fuzzer", mode="direct", callback_addr="localhost:1",
            kallgraph_dir="graph", target_func="target", guidance_engine=engine,
        )
        task.guidance_cancel = mock.Mock()
        task.guidance_cancel.is_set.return_value = False
        task.guidance_cancel.wait.return_value = False
        instance.global_tasks[task.task_id] = task

        def mark_static_done(_task, _engine):
            task.static_analysis_done = True

        with (
            mock.patch.object(
                instance, "_run_static_analysis", side_effect=mark_static_done
            ) as static_analysis,
            mock.patch.object(
                instance, "_send_guidance_if_active", return_value=True
            ) as send_guidance,
            mock.patch.object(controller_module.log_mgr, "log_guidance_result"),
        ):
            instance._run_registration_guidance(task)

        static_analysis.assert_called_once_with(task, engine)
        send_guidance.assert_called_once()

    def test_registration_payload_overrides_config_guidance_context(self):
        payload = RegistrationPayload(
            uuid="fuzzer", task_name="configured-task", mode="direct",
            target_func="payload-target", kallgraph_dir="payload-graph",
            report_path="payload-report",
            target_os="linux", target_arch="amd64",
            target_revision="target-r1", producer_revision="fuzzer-r1",
            descriptions_mode="manual",
        )
        with (
            mock.patch.dict(
                controller_module.config.target_funcs,
                {"configured-task": "config-target"}, clear=True,
            ),
            mock.patch.dict(
                controller_module.config.kallgraph_dirs,
                {"configured-task": "config-graph"}, clear=True,
            ),
            mock.patch.dict(
                controller_module.config.report_paths,
                {"configured-task": "config-report"}, clear=True,
            ),
        ):
            self.assertEqual(
                resolve_guidance_context(payload),
                ("payload-target", "payload-graph", "payload-report"),
            )

    def test_guidance_context_accepts_only_bounded_trusted_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report_root = root / "reports"
            graph_root = root / "graphs"
            report_root.mkdir()
            graph_root.mkdir()
            report = report_root / "case.report"
            report.write_text("report", encoding="utf-8")
            graph_dir = graph_root / "case_1" / "run"
            graph_dir.mkdir(parents=True)

            with (
                mock.patch.object(
                    controller_module.config,
                    "guidance_report_roots",
                    (str(report_root),),
                ),
                mock.patch.object(
                    controller_module.config,
                    "guidance_kallgraph_roots",
                    (str(graph_root),),
                ),
                mock.patch.object(
                    controller_module.config,
                    "guidance_max_report_bytes",
                    64,
                ),
            ):
                self.assertEqual(
                    validate_guidance_context(
                        " target ", str(graph_dir), str(report)
                    ),
                    (
                        "target", str(graph_dir.resolve()),
                        str(report.resolve()), "report",
                    ),
                )

                outside = root / "outside.report"
                outside.write_text("outside", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "outside configured roots"):
                    validate_guidance_context("target", "", str(outside))

                symlink = report_root / "escape.report"
                symlink.symlink_to(outside)
                with self.assertRaisesRegex(ValueError, "outside configured roots"):
                    validate_guidance_context("target", "", str(symlink))

                report.write_text("x" * 65, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "size limit"):
                    validate_guidance_context("target", "", str(report))

    def test_invalid_guidance_context_allocates_no_registration_resources(self):
        instance = Controller()
        payload = RegistrationPayload(
            uuid="fuzzer", task_name="task", mode="direct",
            host_ip="127.0.0.1", http_port=1234,
            report_path="/outside/trusted/roots.report",
            target_os="linux", target_arch="amd64",
            target_revision="target-r1", producer_revision="fuzzer-r1",
            descriptions_mode="manual",
        )

        with (
            mock.patch.object(instance, "_alloc_port") as allocate_port,
            mock.patch.object(instance, "_get_run_id") as get_run_id,
            mock.patch.object(controller_module, "start_receiver") as receiver,
        ):
            with self.assertRaisesRegex(
                controller_module.HTTPException, "unavailable|configured roots"
            ):
                asyncio.run(instance.register(payload))

        allocate_port.assert_not_called()
        get_run_id.assert_not_called()
        receiver.assert_not_called()

    def test_validated_report_is_immutable_after_path_replacement(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report_root = root / "reports"
            report_root.mkdir()
            report = report_root / "case.report"
            report.write_text("trusted report", encoding="utf-8")
            outside = root / "outside.report"
            outside.write_text("outside report", encoding="utf-8")

            with mock.patch.object(
                controller_module.config,
                "guidance_report_roots",
                (str(report_root),),
            ):
                _, _, validated_path, report_text = validate_guidance_context(
                    "target", "", str(report)
                )

            report.unlink()
            report.symlink_to(outside)
            self.assertNotEqual(validated_path, str(report.resolve()))
            self.assertEqual(report_text, "trusted report")

    def test_unregister_cancels_post_deployment_guidance_before_send(self):
        instance = Controller()
        engine = mock.Mock()
        task = FuzzerTask(
            task_id="fuzzer@task@1", task_name="task", run_id=1,
            fuzzer_id="fuzzer", mode="direct", callback_addr="localhost:1",
            guidance_engine=engine, static_analysis_done=True, grpc_port=31001,
        )
        instance.global_tasks[task.task_id] = task
        load_started = threading.Event()
        release_load = threading.Event()

        def load_training_data(_task):
            load_started.set()
            release_load.wait(timeout=5)
            return [], []

        with (
            mock.patch.object(
                instance, "_load_training_data", side_effect=load_training_data
            ),
            mock.patch.object(instance, "_release_port"),
            mock.patch.object(instance, "_remove_ssh_entry"),
        ):
            worker = threading.Thread(
                target=instance._run_guidance_pipeline,
                args=(task, 1, 3, "model", "checkpoint.pt"),
            )
            worker.start()
            self.assertTrue(load_started.wait(timeout=5))
            unregister_done = threading.Event()

            def unregister():
                asyncio.run(instance.unregister(task.task_id))
                unregister_done.set()

            unregister_worker = threading.Thread(target=unregister)
            unregister_worker.start()
            self.assertFalse(unregister_done.wait(timeout=0.1))
            release_load.set()
            worker.join(timeout=5)
            unregister_worker.join(timeout=5)

        self.assertFalse(worker.is_alive())
        self.assertFalse(unregister_worker.is_alive())
        self.assertTrue(unregister_done.is_set())
        engine.compute_guidance.assert_not_called()
        engine.send_guidance.assert_not_called()

    def test_guidance_send_holds_lifecycle_barrier_until_http_finishes(self):
        instance = Controller()
        send_started = threading.Event()
        release_send = threading.Event()
        engine = mock.Mock()
        engine.version = 1

        def send_guidance(_addr, guidance, cancel_event=None):
            send_started.set()
            release_send.wait(timeout=5)
            return True

        engine.send_guidance.side_effect = send_guidance
        task = FuzzerTask(
            task_id="fuzzer@task@1", task_name="task", run_id=1,
            fuzzer_id="fuzzer", mode="direct", callback_addr="localhost:1",
            guidance_engine=engine, grpc_port=31001,
        )
        instance.global_tasks[task.task_id] = task
        send_worker = threading.Thread(
            target=instance._send_guidance_if_active,
            args=(task, engine, {"version": 1}),
        )
        send_worker.start()
        self.assertTrue(send_started.wait(timeout=5))

        unregister_done = threading.Event()

        def unregister():
            asyncio.run(instance.unregister(task.task_id))
            unregister_done.set()

        with (
            mock.patch.object(instance, "_release_port"),
            mock.patch.object(instance, "_remove_ssh_entry"),
        ):
            unregister_worker = threading.Thread(target=unregister)
            unregister_worker.start()
            self.assertFalse(unregister_done.wait(timeout=0.1))
            release_send.set()
            send_worker.join(timeout=5)
            unregister_worker.join(timeout=5)

        self.assertFalse(send_worker.is_alive())
        self.assertFalse(unregister_worker.is_alive())
        self.assertEqual(task.guidance_version, 1)
        self.assertNotIn(task.task_id, instance.global_tasks)

    def test_checkpoint_pruning_keeps_selected_and_noncheckpoint_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir) / "data"
            run_dir = data_root / "task" / "1" / "models" / "round-1"
            run_dir.mkdir(parents=True)
            selected = run_dir / "step-200.pt"
            discarded = run_dir / "step-100.pt"
            scripted = run_dir / "model_scripted.pt"
            manifest = run_dir / "training_manifest.json"
            selected.write_bytes(b"selected")
            discarded.write_bytes(b"discarded")
            scripted.write_bytes(b"scripted")
            manifest.write_text("{}", encoding="utf-8")
            outside = Path(temp_dir) / "step-50.pt"
            outside.write_bytes(b"outside")

            instance = Controller()
            task = FuzzerTask(
                task_id="fuzzer@task@1", task_name="task", run_id=1,
                fuzzer_id="fuzzer", mode="direct",
                callback_addr="localhost:1",
            )
            with mock.patch.object(
                controller_module.config, "data_root", str(data_root)
            ):
                instance._prune_unselected_checkpoints(
                    task, str(run_dir), str(selected)
                )

            self.assertTrue(selected.is_file())
            self.assertFalse(discarded.exists())
            self.assertTrue(scripted.is_file())
            self.assertTrue(manifest.is_file())
            self.assertTrue(outside.is_file())

    def test_superseded_model_cleanup_removes_only_old_model_binaries(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir) / "data"
            model_store = Path(temp_dir) / "model_store"
            old_run = data_root / "task" / "1" / "models" / "round-old"
            new_run = data_root / "task" / "1" / "models" / "round-new"
            model_store.mkdir()
            old_run.mkdir(parents=True)
            new_run.mkdir(parents=True)
            old_manifest = old_run / "training_manifest.json"
            old_checkpoint = old_run / "step-200.pt"
            old_scripted = old_run / "model_scripted.pt"
            old_log = old_run / "train.log"
            new_checkpoint = new_run / "step-100.pt"
            old_checkpoint.write_bytes(b"old checkpoint")
            old_manifest.write_text(json.dumps({
                "best_checkpoint": str(old_checkpoint),
            }), encoding="utf-8")
            old_scripted.write_bytes(b"old scripted")
            old_log.write_text("metrics", encoding="utf-8")
            new_checkpoint.write_bytes(b"new checkpoint")
            old_mar = model_store / "old-model.mar"
            old_mar.write_bytes(b"old archive")

            instance = Controller()
            instance.torchserve_operator.model_dir = str(model_store)
            task = FuzzerTask(
                task_id="fuzzer@task@1", task_name="task", run_id=1,
                fuzzer_id="fuzzer", mode="direct",
                callback_addr="localhost:1", model_name="new-model",
            )
            with (
                mock.patch.object(
                    controller_module.config, "data_root", str(data_root)
                ),
                mock.patch.object(
                    instance.torchserve_operator, "unregister_model"
                ) as unregister,
            ):
                instance._remove_superseded_model(task, (
                    "old-model", "1", str(old_manifest),
                    str(old_checkpoint),
                ))

            unregister.assert_called_once_with("old-model", "1")
            self.assertFalse(old_mar.exists())
            self.assertFalse(old_checkpoint.exists())
            self.assertFalse(old_scripted.exists())
            self.assertTrue(old_manifest.is_file())
            self.assertTrue(old_log.is_file())
            self.assertTrue(new_checkpoint.is_file())

    def test_failed_unregister_preserves_superseded_model_binaries(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir) / "data"
            model_store = Path(temp_dir) / "model_store"
            old_run = data_root / "task" / "1" / "models" / "round-old"
            model_store.mkdir()
            old_run.mkdir(parents=True)
            old_manifest = old_run / "training_manifest.json"
            old_checkpoint = old_run / "step-200.pt"
            old_checkpoint.write_bytes(b"old checkpoint")
            old_manifest.write_text(json.dumps({
                "best_checkpoint": str(old_checkpoint),
            }), encoding="utf-8")
            old_mar = model_store / "old-model.mar"
            old_mar.write_bytes(b"old archive")

            instance = Controller()
            instance.torchserve_operator.model_dir = str(model_store)
            task = FuzzerTask(
                task_id="fuzzer@task@1", task_name="task", run_id=1,
                fuzzer_id="fuzzer", mode="direct",
                callback_addr="localhost:1", model_name="new-model",
            )
            with (
                mock.patch.object(
                    controller_module.config, "data_root", str(data_root)
                ),
                mock.patch.object(
                    instance.torchserve_operator, "unregister_model",
                    side_effect=RuntimeError("management API unavailable"),
                ),
            ):
                instance._remove_superseded_model(task, (
                    "old-model", "1", str(old_manifest),
                    str(old_checkpoint),
                ))

            self.assertTrue(old_mar.is_file())
            self.assertTrue(old_checkpoint.is_file())

    def test_artifact_cleanup_refuses_paths_outside_task_model_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir) / "data"
            task_models = data_root / "task" / "1" / "models"
            task_models.mkdir(parents=True)
            outside_run = Path(temp_dir) / "outside"
            outside_run.mkdir()
            manifest = outside_run / "training_manifest.json"
            checkpoint = outside_run / "step-100.pt"
            manifest.write_text("{}", encoding="utf-8")
            checkpoint.write_bytes(b"must remain")

            instance = Controller()
            task = FuzzerTask(
                task_id="fuzzer@task@1", task_name="task", run_id=1,
                fuzzer_id="fuzzer", mode="direct",
                callback_addr="localhost:1",
            )
            with mock.patch.object(
                controller_module.config, "data_root", str(data_root)
            ):
                instance._remove_superseded_run_binaries(
                    task, str(manifest), str(checkpoint)
                )
                instance._prune_unselected_checkpoints(
                    task, str(outside_run), str(checkpoint)
                )

            self.assertTrue(checkpoint.is_file())

    def test_task_identity_rejects_storage_and_task_id_delimiters(self):
        validate_task_identity("fuzzer-id", "kernel BUG in target")
        for task_name in (
            "..", "../outside", "/tmp/absolute", "bad@name", "bad?query",
            "bad#fragment", "bad%escape",
        ):
            with self.subTest(task_name=task_name):
                with self.assertRaises(ValueError):
                    validate_task_identity("fuzzer-id", task_name)
        with self.assertRaises(ValueError):
            validate_task_identity("bad@uuid", "safe-task")
        with self.assertRaises(ValueError):
            validate_task_identity("bad/uuid", "safe-task")

    def test_artifact_cleanup_refuses_escaped_task_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir) / "data"
            outside_run = Path(temp_dir) / "outside" / "1" / "models" / "run"
            outside_run.mkdir(parents=True)
            checkpoint = outside_run / "step-100.pt"
            manifest = outside_run / "training_manifest.json"
            checkpoint.write_bytes(b"must remain")
            manifest.write_text(json.dumps({
                "best_checkpoint": str(checkpoint),
            }), encoding="utf-8")

            instance = Controller()
            task = FuzzerTask(
                task_id="fuzzer@escaped@1", task_name="../outside", run_id=1,
                fuzzer_id="fuzzer", mode="direct",
                callback_addr="localhost:1",
            )
            with mock.patch.object(
                controller_module.config, "data_root", str(data_root)
            ):
                instance._remove_superseded_run_binaries(
                    task, str(manifest), str(checkpoint)
                )
                instance._prune_unselected_checkpoints(
                    task, str(outside_run), str(checkpoint)
                )

            self.assertTrue(checkpoint.is_file())


if __name__ == "__main__":
    unittest.main()
