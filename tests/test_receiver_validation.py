"""Receiver validation, idempotency, and atomic-commit regression tests."""

import hashlib
import json
import pickle
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock
from pathlib import Path

import grpc


REPO_ROOT = Path(__file__).resolve().parents[1]
BRAIN_DIR = REPO_ROOT / "brain"
sys.path.insert(0, str(BRAIN_DIR))

import transmission_pb2
from receiver import MLTrainingDataServer


def make_example(program, labels, signature=None):
    program_bytes = program.encode("utf-8")
    if signature is None:
        signature = hashlib.sha1(program_bytes).hexdigest()
    return transmission_pb2.TrainingExample(
        prog_data=program_bytes,
        prog_sig=signature,
        target_labels=labels,
    )


def make_batch(batch_id, examples):
    return transmission_pb2.TrainingBatch(
        batch_id=batch_id,
        examples=examples,
    )


class ReceiverValidationTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.receiver = MLTrainingDataServer(
            data_dir=self.temp_dir.name,
            task_id="fuzzer@test-task@1",
        )
        self.data_dir = Path(self.receiver.task_data_dir)

    def test_valid_batch_is_committed_and_retry_is_idempotent(self):
        example = make_example("test$one()", [False, True, False])
        batch = make_batch(7, [example, example])

        response = self.receiver.process_training_batch(batch)
        self.assertEqual(response.status, "success")
        self.assertEqual(response.processed_samples, 2)
        self.assertTrue((self.data_dir / "progs_batch_7.pkl").is_file())
        self.assertTrue((self.data_dir / "labels_batch_7.pkl").is_file())
        self.assertTrue((self.data_dir / "batch_7.complete").is_file())
        self.assertEqual(self.receiver.stats["total_batches"], 1)
        self.assertEqual(self.receiver.stats["total_samples"], 1)
        self.assertEqual(self.receiver.stats["label_distribution"], {1: 1})

        retry = self.receiver.process_training_batch(batch)
        self.assertEqual(retry.status, "success")
        self.assertEqual(retry.processed_samples, 2)
        self.assertEqual(self.receiver.stats["total_batches"], 1)
        self.assertEqual(self.receiver.stats["total_samples"], 1)

        conflicting = make_batch(
            7, [make_example("different()", [False, True, False])]
        )
        with self.assertRaisesRegex(ValueError, "different contents"):
            self.receiver.process_training_batch(conflicting)
        self.assertEqual(self.receiver.stats["total_batches"], 1)

    def test_duplicate_program_observations_keep_deepest_global_label(self):
        program = "nondeterministic$reach()"
        shallow = make_example(program, [False, True, False, False])
        deep = make_example(program, [False, False, False, True])

        first = self.receiver.process_training_batch(
            make_batch(1, [shallow, deep, shallow])
        )
        second = self.receiver.process_training_batch(
            make_batch(2, [shallow])
        )

        self.assertEqual(first.processed_samples, 3)
        self.assertEqual(second.processed_samples, 1)
        with (self.data_dir / "labels_batch_1.pkl").open("rb") as handle:
            first_labels = pickle.load(handle)
        with (self.data_dir / "labels_batch_2.pkl").open("rb") as handle:
            second_labels = pickle.load(handle)
        self.assertEqual(list(first_labels.values()), [[False, False, False, True]])
        self.assertEqual(list(second_labels.values()), [[False, True, False, False]])
        self.assertEqual(self.receiver.stats["total_batches"], 2)
        self.assertEqual(self.receiver.stats["total_samples"], 1)
        self.assertEqual(self.receiver.stats["label_distribution"], {3: 1})

        restarted = MLTrainingDataServer(
            data_dir=self.temp_dir.name,
            task_id="fuzzer@test-task@1",
        )
        self.assertEqual(restarted.stats["total_batches"], 2)
        self.assertEqual(restarted.stats["total_samples"], 1)
        self.assertEqual(restarted.stats["label_distribution"], {3: 1})

    def test_invalid_labels_signature_and_width_are_rejected_without_commit(self):
        valid = make_batch(1, [make_example("valid()", [True, False, False])])
        self.receiver.process_training_batch(valid)

        invalid_batches = [
            make_batch(2, [make_example("zero()", [False, False, False])]),
            make_batch(3, [make_example("multi()", [True, True, False])]),
            make_batch(4, [make_example("width()", [False, True])]),
            make_batch(5, [make_example("hash()", [False, True, False], "bad")]),
        ]
        for batch in invalid_batches:
            with self.subTest(batch_id=batch.batch_id):
                with self.assertRaises(ValueError):
                    self.receiver.process_training_batch(batch)
                self.assertFalse(
                    (self.data_dir / f"batch_{batch.batch_id}.complete").exists()
                )

        self.assertEqual(self.receiver.stats["total_batches"], 1)
        self.assertEqual(self.receiver.stats["total_samples"], 1)

    def test_stream_distinguishes_permanent_and_retryable_batch_errors(self):
        invalid = make_batch(
            20, [make_example("bad-signature()", [False, True], "bad")]
        )
        rejected = list(self.receiver.StreamTrainingData(iter([invalid]), None))
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0].status, "rejected")
        self.assertEqual(rejected[0].processed_samples, 0)
        self.assertFalse((self.data_dir / "batch_20.complete").exists())

        valid = make_batch(21, [make_example("retryable()", [False, True])])
        with mock.patch.object(
            self.receiver,
            "_atomic_pickle",
            side_effect=OSError("disk temporarily unavailable"),
        ):
            retryable = list(
                self.receiver.StreamTrainingData(iter([valid]), None)
            )
        self.assertEqual(len(retryable), 1)
        self.assertEqual(retryable[0].status, "retry")
        self.assertEqual(retryable[0].processed_samples, 0)
        self.assertFalse((self.data_dir / "batch_21.complete").exists())

    def test_stream_classifies_only_post_response_cancellation_as_normal(self):
        class CanceledRpc(grpc.RpcError):
            def code(self):
                return grpc.StatusCode.CANCELLED

            def details(self):
                return "client closed short stream"

        batch = make_batch(
            22, [make_example("committed-before-close()", [False, True])]
        )

        def cancel_after_batch():
            yield batch
            raise CanceledRpc()

        with mock.patch("builtins.print") as output:
            responses = list(
                self.receiver.StreamTrainingData(cancel_after_batch(), None)
            )
        messages = [str(call.args[0]) for call in output.call_args_list]
        self.assertEqual(len(responses), 1)
        self.assertEqual(responses[0].status, "success")
        self.assertTrue(any(
            "after 1 completed response(s)" in message
            for message in messages
        ))
        self.assertFalse(any("STREAM ERROR" in message for message in messages))

        def cancel_before_batch():
            raise CanceledRpc()
            yield  # pragma: no cover - keeps this function an iterator

        with mock.patch("builtins.print") as output:
            responses = list(
                self.receiver.StreamTrainingData(cancel_before_batch(), None)
            )
        messages = [str(call.args[0]) for call in output.call_args_list]
        self.assertEqual(responses, [])
        self.assertTrue(any(
            "STREAM ERROR: code=CANCELLED" in message and
            "completed_responses=0" in message
            for message in messages
        ))

    def test_restart_restores_committed_ids_width_and_statistics(self):
        self.receiver.process_training_batch(make_batch(
            3,
            [
                make_example("unreachable()", [True, False, False]),
                make_example("reached()", [False, False, True]),
            ],
        ))

        restarted = MLTrainingDataServer(
            data_dir=self.temp_dir.name,
            task_id="fuzzer@test-task@1",
        )
        self.assertEqual(restarted.processed_batch_ids, {3})
        self.assertEqual(restarted.label_width, 3)
        self.assertEqual(restarted.stats["total_batches"], 1)
        self.assertEqual(restarted.stats["total_samples"], 2)
        self.assertEqual(restarted.stats["label_distribution"], {0: 1, 2: 1})

    def test_concurrent_batches_receive_unique_wire_ids(self):
        errors = []

        def submit(batch_id):
            try:
                self.receiver.process_training_batch(make_batch(
                    batch_id,
                    [make_example(f"call${batch_id}()", [False, True])],
                ))
            except Exception as error:  # pragma: no cover - assertion below
                errors.append(error)

        threads = [threading.Thread(target=submit, args=(index,)) for index in range(1, 9)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(errors, [])
        self.assertEqual(self.receiver.processed_batch_ids, set(range(1, 9)))
        self.assertEqual(self.receiver.stats["total_batches"], 8)
        self.assertEqual(self.receiver.stats["total_samples"], 8)
        for batch_id in range(1, 9):
            with (self.data_dir / f"labels_batch_{batch_id}.pkl").open("rb") as file_handle:
                labels = pickle.load(file_handle)
            self.assertEqual(len(labels), 1)

    def test_stage_judgement_counts_missing_classes_as_zero(self):
        self.receiver.label_width = 5
        self.assertEqual(self.receiver._judge_stage({1: 1000}), 1)
        self.assertEqual(self.receiver._judge_stage({0: 900, 1: 100}), 1)
        self.assertEqual(
            self.receiver._judge_stage({0: 700, 1: 100, 2: 100, 3: 100}),
            1,
        )
        self.assertEqual(
            self.receiver._judge_stage(
                {0: 600, 1: 100, 2: 100, 3: 100, 4: 100}
            ),
            2,
        )
        self.assertEqual(
            self.receiver._judge_stage(
                {0: 99, 1: 226, 2: 225, 3: 225, 4: 225}
            ),
            1,
        )

    def test_experimental_stage_two_outbox_is_archived(self):
        Path(self.receiver._outbox_path).write_text(json.dumps({
            "schema_version": 1,
            "curriculum_schema": 1,
            "pending": None,
            "last_queued_samples": {"2": 0},
            "deferred_batch_end": {},
        }), encoding="utf-8")

        restarted = MLTrainingDataServer(
            data_dir=self.temp_dir.name,
            task_id="fuzzer@test-task@1",
        )

        self.assertIsNone(restarted.training_outbox)
        self.assertEqual(restarted.last_queued_samples, {})
        self.assertFalse(Path(restarted._outbox_path).exists())
        self.assertEqual(len(list(self.data_dir.glob(
            "training_outbox.json.three-stage-incompatible-*"
        ))), 1)

    def test_experimental_stage_one_outbox_migrates_to_schema_two(self):
        Path(self.receiver._outbox_path).write_text(json.dumps({
            "schema_version": 1,
            "curriculum_schema": 1,
            "pending": None,
            "last_queued_samples": {"1": 0},
            "deferred_batch_end": {},
        }), encoding="utf-8")

        restarted = MLTrainingDataServer(
            data_dir=self.temp_dir.name,
            task_id="fuzzer@test-task@1",
        )

        state = json.loads(Path(restarted._outbox_path).read_text(
            encoding="utf-8"
        ))
        self.assertEqual(state["curriculum_schema"], 2)
        self.assertEqual(restarted.last_queued_samples, {1: 0})

    def test_training_trigger_respects_warmup_gate(self):
        self.receiver.controller_addr = "localhost:1"
        self.receiver.api_token = "token"
        self.receiver.warmup_seconds = 1800
        self.receiver.label_width = 3
        self.receiver.processed_batch_ids = {1, 2}
        self.receiver.stats["total_samples"] = 1000
        self.receiver.stats["label_distribution"] = {0: 700, 1: 150, 2: 150}
        self.receiver.check_and_trigger_training()
        self.assertIsNone(self.receiver.training_outbox)

        self.receiver.stats["start_time"] = time.time() - 1801
        self.receiver.check_and_trigger_training()
        self.assertIsNotNone(self.receiver.training_outbox)

    def test_post_commit_trigger_failure_does_not_change_data_ack(self):
        batch = make_batch(9, [make_example("committed()", [False, True])])
        with mock.patch.object(
            self.receiver,
            "check_and_trigger_training",
            side_effect=RuntimeError("trigger unavailable"),
        ):
            response = self.receiver.process_training_batch(batch)
        self.assertEqual(response.status, "success")
        self.assertTrue((self.data_dir / "batch_9.complete").is_file())

        retry = self.receiver.process_training_batch(batch)
        self.assertEqual(retry.status, "success")
        self.assertEqual(self.receiver.stats["total_batches"], 1)

    def test_training_outbox_persists_and_already_running_stays_pending(self):
        self.receiver.controller_addr = "localhost:1"
        self.receiver.api_token = "token"
        self.receiver.label_width = 3
        self.receiver.processed_batch_ids = {1, 2, 3, 4, 5}
        self.receiver.stats["total_samples"] = 1000
        self.receiver.stats["label_distribution"] = {0: 700, 1: 150, 2: 150}

        self.receiver.check_and_trigger_training()
        self.assertEqual(self.receiver.training_outbox["stage"], 2)
        self.assertTrue(Path(self.receiver._outbox_path).is_file())

        with mock.patch.object(self.receiver, "_trigger_training", return_value=None):
            self.receiver._drain_training_outbox()
        self.assertIsNotNone(self.receiver.training_outbox)

        with mock.patch.object(
            self.receiver, "_trigger_training", return_value=(1, "queued")
        ):
            self.receiver._drain_training_outbox()
        self.assertIsNotNone(self.receiver.training_outbox)
        self.assertEqual(self.receiver.last_queued_samples, {})
        queued_snapshot = dict(self.receiver.training_outbox)
        self.receiver.outbox_wake.clear()
        self.receiver.processed_batch_ids.add(6)
        self.receiver.stats["total_samples"] = 1100
        self.receiver.check_and_trigger_training()
        self.assertEqual(self.receiver.training_outbox, queued_snapshot)
        self.assertFalse(
            self.receiver.outbox_wake.is_set(),
            "an existing queued request must retain the retry backoff",
        )

        with mock.patch.object(
            self.receiver, "_trigger_training", return_value=(1, "up_to_date")
        ):
            self.receiver._drain_training_outbox()
        self.assertIsNone(self.receiver.training_outbox)
        self.assertEqual(self.receiver.last_queued_samples, {1: 1000})

        restarted = MLTrainingDataServer(
            data_dir=self.temp_dir.name,
            task_id="fuzzer@test-task@1",
        )
        # This unit test injected in-memory counters without committed files;
        # restart must reject the resulting future watermark.
        self.assertEqual(restarted.last_queued_samples, {})

    def test_deferred_training_rebases_on_next_committed_snapshot(self):
        self.receiver.controller_addr = "localhost:1"
        self.receiver.api_token = "token"
        self.receiver.label_width = 3
        self.receiver.processed_batch_ids = {1, 2}
        self.receiver.stats["total_samples"] = 1000
        self.receiver.stats["label_distribution"] = {0: 800, 1: 100, 2: 100}
        self.receiver.check_and_trigger_training()
        self.assertEqual(self.receiver.training_outbox["stage"], 2)
        self.assertEqual(self.receiver.training_outbox["batch_end"], 2)

        with mock.patch.object(
            self.receiver, "_trigger_training", return_value=(1, "deferred")
        ):
            self.receiver._drain_training_outbox()
        self.assertIsNone(self.receiver.training_outbox)
        self.assertEqual(self.receiver.last_queued_samples, {})
        self.assertEqual(self.receiver.deferred_batch_end, {2: 2})

        # The periodic outbox loop must not recreate the same deferred
        # snapshot before a newer committed batch exists.
        self.receiver.check_and_trigger_training()
        self.assertIsNone(self.receiver.training_outbox)

        self.receiver.processed_batch_ids.add(3)
        self.receiver.stats["total_samples"] = 1001
        self.receiver.check_and_trigger_training()
        self.assertEqual(self.receiver.training_outbox["batch_end"], 3)

    def test_rejected_training_snapshot_is_consumed_without_retry_loop(self):
        self.receiver.controller_addr = "localhost:1"
        self.receiver.api_token = "token"
        self.receiver.label_width = 2
        self.receiver.processed_batch_ids = {1, 2}
        self.receiver.stats["total_samples"] = 1000
        self.receiver.stats["label_distribution"] = {0: 50, 1: 950}
        self.receiver.check_and_trigger_training()
        self.assertIsNotNone(self.receiver.training_outbox)

        with mock.patch.object(
            self.receiver, "_trigger_training", return_value=(1, "rejected")
        ):
            self.receiver._drain_training_outbox()

        self.assertIsNone(self.receiver.training_outbox)
        self.assertEqual(self.receiver.last_queued_samples, {1: 1000})
        self.receiver.check_and_trigger_training()
        self.assertIsNone(self.receiver.training_outbox)

        self.receiver.stats["total_samples"] = 1999
        self.receiver.check_and_trigger_training()
        self.assertIsNone(self.receiver.training_outbox)
        self.receiver.stats["total_samples"] = 2000
        self.receiver.processed_batch_ids.add(3)
        self.receiver.check_and_trigger_training()
        self.assertIsNotNone(self.receiver.training_outbox)

    def test_rejected_forced_stage_one_consumes_stage_two_request(self):
        self.receiver.controller_addr = "localhost:1"
        self.receiver.api_token = "token"
        self.receiver.label_width = 3
        self.receiver.processed_batch_ids = {1, 2}
        self.receiver.stats["total_samples"] = 1000
        self.receiver.stats["label_distribution"] = {0: 800, 1: 100, 2: 100}
        self.receiver.check_and_trigger_training()
        self.assertEqual(self.receiver.training_outbox["stage"], 2)

        with mock.patch.object(
            self.receiver, "_trigger_training", return_value=(1, "rejected")
        ):
            self.receiver._drain_training_outbox()

        self.assertEqual(
            self.receiver.last_queued_samples, {1: 1000, 2: 1000}
        )
        self.receiver.check_and_trigger_training()
        self.assertIsNone(self.receiver.training_outbox)

    def test_deferred_boundary_persist_failure_restores_pending_request(self):
        self.receiver.controller_addr = "localhost:1"
        self.receiver.api_token = "token"
        self.receiver.label_width = 3
        self.receiver.processed_batch_ids = {1, 2}
        self.receiver.stats["total_samples"] = 1000
        self.receiver.stats["label_distribution"] = {0: 800, 1: 200}
        self.receiver.check_and_trigger_training()
        pending = dict(self.receiver.training_outbox)

        with (
            mock.patch.object(
                self.receiver, "_trigger_training", return_value=(1, "deferred")
            ),
            mock.patch.object(
                self.receiver, "_persist_training_outbox_locked",
                side_effect=OSError("disk full"),
            ),
        ):
            self.receiver._drain_training_outbox()

        self.assertEqual(self.receiver.training_outbox, pending)
        self.assertEqual(self.receiver.deferred_batch_end, {})

    def test_corrupt_committed_state_is_ignored_and_retry_repairs_it(self):
        batch = make_batch(11, [make_example("repair()", [False, True])])
        self.receiver.process_training_batch(batch)
        labels_path = self.data_dir / "labels_batch_11.pkl"
        labels_path.write_bytes(b"not-a-pickle")

        restarted = MLTrainingDataServer(
            data_dir=self.temp_dir.name,
            task_id="fuzzer@test-task@1",
        )
        self.assertNotIn(11, restarted.processed_batch_ids)
        response = restarted.process_training_batch(batch)
        self.assertEqual(response.status, "success")
        self.assertIn(11, restarted.processed_batch_ids)

    def test_retry_ack_survives_repeated_trigger_enqueue_failure(self):
        batch = make_batch(12, [make_example("retry-safe()", [False, True])])
        first = self.receiver.process_training_batch(batch)
        self.assertEqual(first.status, "success")
        with mock.patch.object(
            self.receiver,
            "check_and_trigger_training",
            side_effect=RuntimeError("outbox unavailable"),
        ):
            retry = self.receiver.process_training_batch(batch)
        self.assertEqual(retry.status, "success")

    def test_corrupt_outbox_shapes_and_ranges_are_ignored(self):
        outbox_path = Path(self.receiver._outbox_path)
        invalid_states = [
            "[]",
            '{"schema_version": 1, "pending": {"stage": 3, '
            '"batch_end": 1, "total_samples": 1000}, '
            '"last_queued_samples": {}}',
            '{"schema_version": 1, "pending": null, '
            '"last_queued_samples": []}',
            '{"schema_version": 1, "pending": null, '
            '"last_queued_samples": {"9": 1000}}',
            '{"schema_version": 1, "pending": null, '
            '"last_queued_samples": {"1": 100000000000000000000}}',
        ]
        for state in invalid_states:
            with self.subTest(state=state):
                outbox_path.write_text(state, encoding="utf-8")
                restarted = MLTrainingDataServer(
                    data_dir=self.temp_dir.name,
                    task_id="fuzzer@test-task@1",
                )
                self.assertIsNone(restarted.training_outbox)
                self.assertEqual(restarted.last_queued_samples, {})

    def test_out_of_range_committed_wire_count_is_ignored(self):
        batch = make_batch(13, [make_example("wire-count()", [False, True])])
        self.receiver.process_training_batch(batch)
        marker_path = self.data_dir / "batch_13.complete"
        marker = marker_path.read_text(encoding="utf-8")
        marker_path.write_text(
            marker.replace("wire_samples=1", "wire_samples=10001"),
            encoding="utf-8",
        )

        restarted = MLTrainingDataServer(
            data_dir=self.temp_dir.name,
            task_id="fuzzer@test-task@1",
        )
        self.assertNotIn(13, restarted.processed_batch_ids)
        self.assertFalse(marker_path.exists())
        self.assertTrue(list(self.data_dir.glob("batch_13.complete.corrupt-*")))


if __name__ == "__main__":
    unittest.main()
