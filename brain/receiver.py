import os
import grpc
import time
import torch
import pickle
import requests
import threading
import hashlib
import tempfile
import sys
from pathlib import Path
from datetime import datetime
from concurrent import futures

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
from common.curriculum import CURRICULUM_SCHEMA_VERSION, infer_curriculum_stage

# Import grpc stubs
import transmission_pb2, transmission_pb2_grpc


class PermanentBatchError(ValueError):
    """A deterministic client batch error that cannot succeed on retry."""


class MLTrainingDataServer(transmission_pb2_grpc.MLTrainingDataServiceServicer):
    """ML Training Data Receiver"""

    def __init__(self, data_dir="./training_data", task_id="fuzzer_id@task_name@run_id",
                 controller_addr=None, api_token=None, warmup_seconds=0):
        self.task_id = task_id
        self.fuzzer_id, self.task_name, self.run_id = task_id.split('@')
        self.task_data_dir = os.path.join(data_dir, self.task_name, str(self.run_id))
        self.check_directories()
        self.batch_count = 0
        self.stage = 0
        self.stage1_threshold = 1000
        self.min_trainable_class_samples = 100
        self.warmup_seconds = max(0, int(warmup_seconds))
        self.controller_addr = controller_addr # supposed to be localhost:port
        self.api_token = api_token
        self.process_lock = threading.Lock()
        self.trigger_lock = threading.Lock()
        self.processed_batch_ids = set()
        self.label_width = None
        self.canonical_classes = {}

        self.stats = {
            'total_batches': 0,
            'total_samples': 0,
            'label_distribution': dict(),
            'start_time': time.time(),
        }
        self.stats_lock = threading.Lock()
        self.training_triggered = {}  # retained for dashboard compatibility
        self.training_outbox = None
        self.last_queued_samples = {}
        self.deferred_batch_end = {}
        self.outbox_stop = threading.Event()
        self.outbox_wake = threading.Event()
        self._load_committed_state()
        self._load_training_outbox()
        if self.controller_addr and self.api_token:
            threading.Thread(target=self._outbox_loop, daemon=True).start()

        print(f"[DataReceiver] Initialized for task: {self.task_name}, run: {self.run_id}")
        print(f"[DataReceiver] Data directory: {self.task_data_dir}")

    def check_directories(self):
        """Create necessary directories"""
        if not os.path.exists(self.task_data_dir):
            os.makedirs(self.task_data_dir)
            print(f"[DataReceiver] Created data directory: {self.task_data_dir}")
        else:
            print(f"[DataReceiver] Data directory already exists: {self.task_data_dir}")

    def _load_committed_state(self):
        """Restore idempotency and statistics from completed batch markers."""
        data_dir = Path(self.task_data_dir)
        for marker in sorted(data_dir.glob("batch_*.complete")):
            try:
                batch_id = int(marker.stem.split("_")[1])
            except (IndexError, ValueError):
                continue
            progs_file = data_dir / f"progs_batch_{batch_id}.pkl"
            labels_file = data_dir / f"labels_batch_{batch_id}.pkl"
            if not progs_file.is_file() or not labels_file.is_file():
                print(f"[DataReceiver] Ignoring incomplete marker for batch {batch_id}")
                continue
            try:
                marker_fields = self._read_marker_fields(marker)
                if int(marker_fields.get("schema_version", 0)) != 1:
                    raise ValueError("unsupported or missing marker schema")
                if int(marker_fields.get("batch_id", 0)) != batch_id:
                    raise ValueError("marker batch ID mismatch")
                batch_digest = marker_fields.get("batch_digest", "")
                if len(batch_digest) != 64 or any(
                    char not in "0123456789abcdef" for char in batch_digest
                ):
                    raise ValueError("invalid marker batch digest")
                with progs_file.open("rb") as file_handle:
                    programs = pickle.load(file_handle)  # nosec B301 - trusted local state
                with labels_file.open("rb") as file_handle:
                    labels = pickle.load(file_handle)  # nosec B301 - trusted local state
                if not isinstance(programs, dict) or not isinstance(labels, dict):
                    raise ValueError("committed payloads must be dictionaries")
                if programs.keys() != labels.keys():
                    raise ValueError("program and label keys differ")
                if not labels or len(labels) > 10000:
                    raise ValueError("committed payload sample count is out of range")
                if int(marker_fields.get("unique_samples", -1)) != len(labels):
                    raise ValueError("marker unique sample count mismatch")
                wire_samples = int(marker_fields.get("wire_samples", -1))
                if wire_samples < len(labels) or wire_samples > 10000:
                    raise ValueError("marker wire sample count is out of range")
                batch_label_width = self.label_width
                for signature, label in labels.items():
                    program = programs[signature]
                    if not isinstance(program, str):
                        raise ValueError("committed program is not UTF-8 text")
                    if hashlib.sha1(program.encode("utf-8")).hexdigest() != signature:
                        raise ValueError("committed program signature mismatch")
                    selected = self._validate_one_hot_label(label, batch_label_width)
                    if batch_label_width is None:
                        batch_label_width = len(label)
            except Exception as error:
                print(f"[DataReceiver] Ignoring corrupt committed batch {batch_id}: {error}")
                try:
                    quarantine = marker.with_name(
                        f"{marker.name}.corrupt-{time.time_ns()}"
                    )
                    os.replace(marker, quarantine)
                    directory_fd = os.open(self.task_data_dir, os.O_RDONLY)
                    try:
                        os.fsync(directory_fd)
                    finally:
                        os.close(directory_fd)
                    print(
                        f"[DataReceiver] Quarantined corrupt marker as "
                        f"{quarantine.name}"
                    )
                except OSError as quarantine_error:
                    print(
                        f"[DataReceiver] Failed to quarantine corrupt marker "
                        f"{marker}: {quarantine_error}"
                    )
                continue

            self.label_width = batch_label_width
            self.processed_batch_ids.add(batch_id)
            self.stats['total_batches'] += 1
            self._merge_canonical_label_stats(labels)
            self.stats['start_time'] = min(
                self.stats['start_time'], marker.stat().st_mtime
            )
        self.batch_count = len(self.processed_batch_ids)

    @staticmethod
    def _validate_one_hot_label(labels, expected_width=None):
        if not labels:
            raise PermanentBatchError("empty reachability label")
        if expected_width is not None and len(labels) != expected_width:
            raise PermanentBatchError(
                f"label width {len(labels)} does not match expected {expected_width}"
            )
        selected = [index for index, value in enumerate(labels) if bool(value)]
        if len(selected) != 1:
            raise PermanentBatchError(
                f"label must be exactly one-hot: {list(labels)}"
            )
        return selected[0]

    def _merge_canonical_label_stats(self, labels):
        """Merge committed rows into global unique-program training statistics."""
        new_programs = 0
        label_upgrades = 0
        distribution = self.stats['label_distribution']
        for signature, label in labels.items():
            selected_class = next(
                index for index, value in enumerate(label) if bool(value)
            )
            previous_class = self.canonical_classes.get(signature)
            if previous_class is None:
                self.canonical_classes[signature] = selected_class
                self.stats['total_samples'] += 1
                distribution[selected_class] = distribution.get(selected_class, 0) + 1
                new_programs += 1
            elif selected_class > previous_class:
                self.canonical_classes[signature] = selected_class
                distribution[previous_class] -= 1
                if distribution[previous_class] == 0:
                    del distribution[previous_class]
                distribution[selected_class] = distribution.get(selected_class, 0) + 1
                label_upgrades += 1
        return new_programs, label_upgrades

    @staticmethod
    def _atomic_pickle(path, value):
        directory = os.path.dirname(path)
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=directory, prefix=".receiver-", delete=False
            ) as file_handle:
                temp_path = file_handle.name
                pickle.dump(value, file_handle)
                file_handle.flush()
                os.fsync(file_handle.fileno())
            os.replace(temp_path, path)
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)

    @staticmethod
    def _atomic_marker(path, batch_id, unique_samples, wire_samples, batch_digest):
        directory = os.path.dirname(path)
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=directory,
                prefix=".receiver-", delete=False
            ) as file_handle:
                temp_path = file_handle.name
                file_handle.write("schema_version=1\n")
                file_handle.write(f"batch_id={batch_id}\n")
                file_handle.write(f"unique_samples={unique_samples}\n")
                file_handle.write(f"wire_samples={wire_samples}\n")
                file_handle.write(f"batch_digest={batch_digest}\n")
                file_handle.flush()
                os.fsync(file_handle.fileno())
            os.replace(temp_path, path)
            directory_fd = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)

    @staticmethod
    def _batch_digest(batch):
        """Return a stable identity for detecting conflicting batch-ID reuse."""
        wire_data = batch.SerializeToString(deterministic=True)
        return hashlib.sha256(wire_data).hexdigest()

    @staticmethod
    def _read_marker_fields(path):
        try:
            with open(path, encoding="utf-8") as file_handle:
                return dict(
                    line.rstrip("\n").split("=", 1)
                    for line in file_handle
                    if "=" in line
                )
        except OSError:
            return {}

    @staticmethod
    def _read_marker_count(path, key, fallback):
        try:
            fields = MLTrainingDataServer._read_marker_fields(path)
            return int(fields.get(key, fallback))
        except (TypeError, ValueError):
            return fallback

    @property
    def _outbox_path(self):
        return os.path.join(self.task_data_dir, "training_outbox.json")

    def _persist_training_outbox_locked(self):
        state = {
            "schema_version": 1,
            "curriculum_schema": CURRICULUM_SCHEMA_VERSION,
            "pending": self.training_outbox,
            "last_queued_samples": {
                str(stage): count
                for stage, count in self.last_queued_samples.items()
            },
            "deferred_batch_end": {
                str(stage): batch_end
                for stage, batch_end in self.deferred_batch_end.items()
            },
        }
        directory = self.task_data_dir
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=directory,
                prefix=".training-outbox-", delete=False,
            ) as file_handle:
                temp_path = file_handle.name
                import json
                json.dump(state, file_handle, indent=2, sort_keys=True)
                file_handle.write("\n")
                file_handle.flush()
                os.fsync(file_handle.fileno())
            os.replace(temp_path, self._outbox_path)
            directory_fd = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)

    def _load_training_outbox(self):
        if not os.path.isfile(self._outbox_path):
            return
        try:
            import json
            with open(self._outbox_path, encoding="utf-8") as file_handle:
                state = json.load(file_handle)
            if not isinstance(state, dict):
                raise ValueError("training outbox root must be an object")
            if state.get("schema_version") != 1:
                raise ValueError("unsupported training outbox schema")
            curriculum_schema = state.get("curriculum_schema")
            legacy_curriculum = curriculum_schema is None
            experimental_curriculum = (
                curriculum_schema is not None and int(curriculum_schema) == 1
            )
            if (curriculum_schema is not None and
                    int(curriculum_schema) not in (
                        1, CURRICULUM_SCHEMA_VERSION
                    )):
                raise ValueError("unsupported training outbox curriculum schema")
            pending = state.get("pending")
            if pending is not None:
                pending = {
                    "stage": int(pending["stage"]),
                    "batch_end": int(pending["batch_end"]),
                    "total_samples": int(pending["total_samples"]),
                }
                if pending["stage"] not in (1, 2, 3):
                    raise ValueError("training outbox stage is invalid")
                if pending["batch_end"] <= 0 or pending["total_samples"] <= 0:
                    raise ValueError("training outbox watermark is invalid")
                if pending["batch_end"] > max(self.processed_batch_ids, default=0):
                    raise ValueError("training outbox batch watermark is in the future")
                if pending["total_samples"] > self.stats["total_samples"]:
                    raise ValueError("training outbox sample watermark is in the future")
            self.training_outbox = pending
            last_queued_samples = {
                int(stage): int(count)
                for stage, count in state.get("last_queued_samples", {}).items()
            }
            if any(
                stage not in (1, 2, 3) or count < 0 or
                count > self.stats["total_samples"]
                for stage, count in last_queued_samples.items()
            ):
                raise ValueError("training outbox committed watermark is invalid")
            self.last_queued_samples = last_queued_samples
            deferred_batch_end = {
                int(stage): int(batch_end)
                for stage, batch_end in state.get(
                    "deferred_batch_end", {}
                ).items()
            }
            max_committed_batch = max(self.processed_batch_ids, default=0)
            if any(
                stage not in (1, 2, 3) or batch_end <= 0 or
                batch_end > max_committed_batch
                for stage, batch_end in deferred_batch_end.items()
            ):
                raise ValueError("training outbox deferred boundary is invalid")
            legacy_stages = set(last_queued_samples).union(deferred_batch_end)
            if pending is not None:
                legacy_stages.add(pending["stage"])
            if experimental_curriculum and any(
                    stage > 1 for stage in legacy_stages):
                archive_path = (
                    f"{self._outbox_path}.three-stage-incompatible-"
                    f"{time.time_ns()}"
                )
                os.replace(self._outbox_path, archive_path)
                print(
                    "[DataReceiver] Archived incompatible three-stage "
                    f"training outbox at {archive_path}; retained data will "
                    "be re-evaluated"
                )
                self.training_outbox = None
                self.last_queued_samples = {}
                self.deferred_batch_end = {}
                return
            if any(stage not in (1, 2) for stage in legacy_stages):
                raise ValueError("training outbox has an invalid curriculum stage")
            self.deferred_batch_end = deferred_batch_end
            if legacy_curriculum or experimental_curriculum:
                self._persist_training_outbox_locked()
        except (
            OSError, AttributeError, KeyError, TypeError, ValueError,
            json.JSONDecodeError,
        ) as error:
            print(f"[DataReceiver] Ignoring corrupt training outbox: {error}")
            self.training_outbox = None
            self.last_queued_samples = {}
            self.deferred_batch_end = {}

    def _outbox_loop(self):
        while not self.outbox_stop.is_set():
            self.outbox_wake.wait(5)
            self.outbox_wake.clear()
            if self.outbox_stop.is_set():
                return
            try:
                self._schedule_training_safely()
                self._drain_training_outbox()
            except Exception as error:
                print(f"[DataReceiver] Training outbox loop error: {error}")

    def stop(self):
        self.outbox_stop.set()
        self.outbox_wake.set()

    def StreamTrainingData(self, request_iterator, context):
        """Process streaming training data"""
        connection_id = int(time.time() * 1000000) % 1000000
        completed_responses = 0
        print(f"[DataReceiver] New connection established (ID: {connection_id})")


        try:
            for batch in request_iterator:
                print(f"\n[DataReceiver] ===== Processing Batch ID {batch.batch_id} =====")
                print(f"[DataReceiver] Batch ID: {batch.batch_id}")
                print(f"[DataReceiver] Examples count: {len(batch.examples)}")

                try:
                    # Process batch data
                    response = self.process_training_batch(batch)
                    yield response
                    completed_responses += 1

                except PermanentBatchError as error:
                    print(f"[DataReceiver] REJECTED batch: {error}")
                    yield transmission_pb2.BatchResponse(
                        batch_id=batch.batch_id,
                        status="rejected",
                        message=f"Permanent batch error: {error}",
                        processed_samples=0
                    )
                    completed_responses += 1
                except Exception as error:
                    print(f"[DataReceiver] RETRYABLE batch error: {error}")
                    yield transmission_pb2.BatchResponse(
                        batch_id=batch.batch_id,
                        status="retry",
                        message=f"Retryable processing error: {error}",
                        processed_samples=0
                    )
                    completed_responses += 1

        except grpc.RpcError as stream_error:
            code = stream_error.code()
            details = stream_error.details()
            if (code == grpc.StatusCode.CANCELLED and
                    completed_responses > 0):
                print(
                    f"[DataReceiver] Client closed connection {connection_id} "
                    f"after {completed_responses} completed response(s)"
                )
            else:
                print(
                    f"[DataReceiver] STREAM ERROR: code={code.name} "
                    f"details={details!r} completed_responses="
                    f"{completed_responses}"
                )
        except Exception as stream_error:
            print(
                f"[DataReceiver] STREAM ERROR: "
                f"type={type(stream_error).__name__} "
                f"details={str(stream_error)!r} completed_responses="
                f"{completed_responses}"
            )

        finally:
            print(f"[DataReceiver] Connection {connection_id} closed after {self.batch_count} batches")

    def process_training_batch(self, batch):
        """Process a single training batch"""
        with self.process_lock:
            return self._process_training_batch_locked(batch)

    def _process_training_batch_locked(self, batch):
        start_time = time.time()
        batch_id = int(batch.batch_id)
        examples = batch.examples
        if batch_id <= 0:
            raise PermanentBatchError(f"invalid batch ID: {batch_id}")
        if not examples:
            raise PermanentBatchError("empty training batch")
        if len(examples) > 10000:
            raise PermanentBatchError(
                f"training batch has too many examples: {len(examples)}"
            )
        batch_digest = self._batch_digest(batch)
        if batch_id in self.processed_batch_ids:
            progs_file = os.path.join(
                self.task_data_dir, f"progs_batch_{batch_id}.pkl"
            )
            with open(progs_file, "rb") as file_handle:
                sample_count = len(pickle.load(file_handle))  # nosec B301
            marker_file = os.path.join(
                self.task_data_dir, f"batch_{batch_id}.complete"
            )
            marker_fields = self._read_marker_fields(marker_file)
            if marker_fields.get("batch_digest") != batch_digest:
                raise PermanentBatchError(
                    f"batch ID {batch_id} was already committed with different contents"
                )
            processed_count = self._read_marker_count(
                marker_file, "wire_samples", sample_count
            )
            print(f"[DataReceiver] Batch {batch_id} already committed; acknowledging retry")
            self._schedule_training_safely()
            return transmission_pb2.BatchResponse(
                batch_id=batch_id,
                status="success",
                message=f"Batch {batch_id} already committed",
                processed_samples=processed_count,
            )

        # Print first 1 sample
        print(f"[DataReceiver] Showing first 1 sample:")
        for i, example in enumerate(examples[:1]):
            prog_str_preview = example.prog_data[:200].decode(
                'utf-8', errors='ignore'
            )
            print(f"  Sample {i+1}:")
            print(f"    Sig: {example.prog_sig}")
            print(f"    Labels: {list(example.target_labels)}")
            # Labels type: <class 'list'>, Raw Labels type: <class 'google._upb._message.RepeatedScalarContainer'>
            print(f"    Prog preview: {prog_str_preview}...")

        # Assemble data dictionary
        prog_dict = {}      # sig -> prog_str
        label_dict = {}     # sig -> one-hot list
        label_distribution = {} # label_index -> count
        duplicate_examples = 0
        label_upgrades = 0

        batch_label_width = self.label_width
        for example in examples:
            sig = example.prog_sig
            if len(example.prog_data) > 4 * 1024 * 1024:
                raise PermanentBatchError(f"program {sig!r} exceeds 4 MiB")
            computed_sig = hashlib.sha1(example.prog_data).hexdigest()
            if sig != computed_sig:
                raise PermanentBatchError(
                    f"program signature mismatch: received {sig!r}, computed {computed_sig}"
                )
            try:
                prog_str = example.prog_data.decode('utf-8')
            except UnicodeDecodeError as error:
                raise PermanentBatchError(
                    f"program {sig!r} is not valid UTF-8"
                ) from error
            one_hot_labels = list(example.target_labels)  # [False, True, False, False] format
            selected_class = self._validate_one_hot_label(
                one_hot_labels, batch_label_width
            )
            if batch_label_width is None:
                batch_label_width = len(one_hot_labels)
            if sig in prog_dict:
                if prog_dict[sig] != prog_str:
                    raise PermanentBatchError(
                        f"program hash collision for signature: {sig}"
                    )
                duplicate_examples += 1
                previous_class = next(
                    index for index, value in enumerate(label_dict[sig]) if value
                )
                # A syz-program can observe different reachability depths across
                # executions because kernel state and scheduling are nondeterministic.
                # The text-only classifier cannot distinguish those executions, so
                # retain the deepest observed label for this program in the batch.
                if selected_class > previous_class:
                    label_dict[sig] = one_hot_labels
                    label_distribution[previous_class] -= 1
                    if label_distribution[previous_class] == 0:
                        del label_distribution[previous_class]
                    label_distribution[selected_class] = (
                        label_distribution.get(selected_class, 0) + 1
                    )
                    label_upgrades += 1
                continue
            prog_dict[sig] = prog_str
            label_dict[sig] = one_hot_labels
            label_distribution[selected_class] = (
                label_distribution.get(selected_class, 0) + 1
            )

        # Save program dictionary
        progs_file = os.path.join(self.task_data_dir, f"progs_batch_{batch_id}.pkl")
        labels_file = os.path.join(self.task_data_dir, f"labels_batch_{batch_id}.pkl")
        marker_file = os.path.join(self.task_data_dir, f"batch_{batch_id}.complete")
        self._atomic_pickle(progs_file, prog_dict)
        self._atomic_pickle(labels_file, label_dict)
        self._atomic_marker(
            marker_file, batch_id, len(prog_dict), len(examples), batch_digest
        )
        self.label_width = batch_label_width
        self.processed_batch_ids.add(batch_id)
        self.batch_count = len(self.processed_batch_ids)

        # Update statistics
        with self.stats_lock:
            self.stats['total_batches'] += 1
            new_programs, global_label_upgrades = (
                self._merge_canonical_label_stats(label_dict)
            )
        processing_time = time.time() - start_time

        print(f"[DataReceiver] Batch {batch_id} processed successfully:")
        print(f"  - Saved {len(prog_dict)} programs to {progs_file}")
        print(f"  - Saved {len(label_dict)} label lists to {labels_file}")
        if duplicate_examples:
            print(
                f"  - Coalesced {duplicate_examples} duplicate observations "
                f"({label_upgrades} deepest-label upgrades)"
            )
        cross_batch_duplicates = len(prog_dict) - new_programs
        if cross_batch_duplicates:
            print(
                f"  - Observed {cross_batch_duplicates} cross-batch duplicates "
                f"({global_label_upgrades} global deepest-label upgrades)"
            )
        print(f"[DataReceiver] Total label distribution: {self.stats['label_distribution']}")
        print(f"  - Processing time: {processing_time:.3f}s")

        # Training delivery is an independent, durable outbox. It must never
        # turn an already committed data batch into an error acknowledgement.
        self._schedule_training_safely()

        return transmission_pb2.BatchResponse(
            batch_id=batch_id,
            status="success",
            message=f"Processed {len(prog_dict)} samples",
            processed_samples=len(examples)
        )

    def _schedule_training_safely(self):
        try:
            self.check_and_trigger_training()
        except Exception as error:
            print(f"[DataReceiver] Failed to enqueue training request: {error}")

    def check_and_trigger_training(self):
        """Create or retry a durable, idempotent training request."""
        if not self.controller_addr or not self.api_token:
            return

        with self.stats_lock:
            total_samples = self.stats['total_samples']
            label_dist = dict(self.stats['label_distribution'])
            elapsed = time.time() - self.stats['start_time']
        stage = self._judge_stage(label_dist)
        # Canonical labels can move to a deeper class after re-execution, which
        # may temporarily reduce a previously trainable group's count. Never
        # request a coarser objective after a finer stage has been committed.
        stage = max(stage, max(self.last_queued_samples, default=0))
        self.stage = stage
        if stage == 0:
            return
        if elapsed < self.warmup_seconds:
            return

        created = False
        with self.trigger_lock:
            if self.training_outbox is None:
                batch_end = max(self.processed_batch_ids, default=0)
                if batch_end <= self.deferred_batch_end.get(stage, 0):
                    return
                last_queued = self.last_queued_samples.get(stage, 0)
                if total_samples - last_queued < self.stage1_threshold:
                    return
                pending = {
                    "stage": stage,
                    "batch_end": batch_end,
                    "total_samples": total_samples,
                }
                self.training_outbox = pending
                try:
                    self._persist_training_outbox_locked()
                except Exception:
                    self.training_outbox = None
                    raise
                created = True
            # Once queued, retain an immutable snapshot. Samples that arrive
            # during training belong to the next threshold window.
        # Wake the worker only for a newly persisted request. Re-arming this
        # event for an existing queued request makes the worker's timed loop
        # spin without its five-second backoff while training is in progress.
        if created:
            self.outbox_wake.set()

    def _drain_training_outbox(self):
        with self.trigger_lock:
            if self.training_outbox is None:
                return
            pending = dict(self.training_outbox)
        result = self._trigger_training(pending)
        if result is None:
            return
        actual_stage, status = result
        if status not in ("queued", "up_to_date", "deferred", "rejected"):
            return
        if status == "queued":
            # Queued is not a commit: retain the outbox until the Controller
            # reports that this watermark was successfully trained.
            self.training_triggered[actual_stage] = True
            return
        if status == "deferred":
            with self.trigger_lock:
                if self.training_outbox != pending:
                    return
                previous_deferred = dict(self.deferred_batch_end)
                self.training_outbox = None
                requested_stage = pending["stage"]
                self.deferred_batch_end[requested_stage] = max(
                    self.deferred_batch_end.get(requested_stage, 0),
                    pending["batch_end"],
                )
                try:
                    self._persist_training_outbox_locked()
                except Exception as error:
                    self.training_outbox = pending
                    self.deferred_batch_end = previous_deferred
                    print(
                        f"[DataReceiver] Failed to persist deferred training "
                        f"boundary: {error}"
                    )
                    return
            print(
                f"[DataReceiver] Training request deferred at batch "
                f"{pending['batch_end']}; waiting for a newer data snapshot"
            )
            if max(self.processed_batch_ids, default=0) > pending["batch_end"]:
                self.outbox_wake.set()
            return
        with self.trigger_lock:
            if self.training_outbox != pending:
                return
            previous_samples = dict(self.last_queued_samples)
            previous_deferred = dict(self.deferred_batch_end)
            self.last_queued_samples[actual_stage] = max(
                self.last_queued_samples.get(actual_stage, 0),
                pending["total_samples"],
            )
            if status == "rejected":
                # The Controller may enforce Stage 1 for a request whose
                # observed label distribution already qualifies for Stage 2.
                # Consume both watermarks on rejection so the immutable
                # snapshot is not resubmitted every five seconds. A normal
                # up-to-date Stage 1 response intentionally leaves Stage 2
                # eligible on the same snapshot.
                requested_stage = pending["stage"]
                self.last_queued_samples[requested_stage] = max(
                    self.last_queued_samples.get(requested_stage, 0),
                    pending["total_samples"],
                )
            self.training_triggered[actual_stage] = True
            self.training_outbox = None
            self.deferred_batch_end.pop(actual_stage, None)
            self.deferred_batch_end.pop(pending["stage"], None)
            try:
                self._persist_training_outbox_locked()
            except Exception as error:
                self.training_outbox = pending
                self.last_queued_samples = previous_samples
                self.deferred_batch_end = previous_deferred
                print(
                    f"[DataReceiver] Failed to persist committed training "
                    f"request: {error}"
                )
                return
        print(
            f"[DataReceiver] Training request committed: "
            f"stage={actual_stage}, status={status}, batch_end={pending['batch_end']}"
        )

    def _judge_stage(self, label_dist):
        """
        Select the most detailed trainable curriculum objective.

        Stage 0 collects data, Stage 1 is binary reachability, and Stage 2
        learns every waypoint class independently.
        """
        if not label_dist:
            print(f"[DataReceiver] No label distribution found")
            return 0

        if not self.label_width:
            return 0
        return infer_curriculum_stage(
            label_dist,
            self.label_width,
            minimum_total=self.stage1_threshold,
            minimum_positive_class=self.min_trainable_class_samples,
        )

    def _trigger_training(self, pending):
        """Send training request to Controller"""
        try:
            url = f"http://{self.controller_addr}/start_trainer/{self.task_id}"
            params = {
                "batch_start": 0,
                "batch_end": pending["batch_end"],
                "stage": pending["stage"],
            }
            headers = {
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json"
            }

            response = requests.post(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()

            payload = response.json()
            status = payload.get("status")
            if status not in ("queued", "up_to_date", "deferred", "rejected"):
                print(f"[DataReceiver] Training request remains pending: {payload}")
                return None
            return int(payload.get("stage", pending["stage"])), status

        except Exception as e:
            print(f"[DataReceiver] ERROR triggering training: {e}")
            return None

    def HealthCheck(self, request, context):
        """Health check"""
        with self.stats_lock:
            total_batches = self.stats['total_batches']
            total_samples = self.stats['total_samples']
            runtime = time.time() - self.stats['start_time']

        message = f"Running for {runtime:.1f}s, received {total_samples} samples in {total_batches} batches"

        return transmission_pb2.HealthResponse(
            healthy=True,
            message=message,
            total_batches_received=total_batches,
            total_samples_received=total_samples
        )

def serve(port=50051, data_dir="./training_data", task_id="fuzzer_id@task_name@run_id",
          controller_addr=None, api_token=None, warmup_seconds=1800,
          enable_compression=False, log_file=None):
    """Start training data server"""
    # Redirect stdout to log file if specified (preserves all print() output)
    if log_file:
        import sys
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        log_fh = open(log_file, "a", encoding="utf-8", buffering=1)  # line-buffered
        sys.stdout = log_fh
        sys.stderr = log_fh

    fuzzer_id, task_name, run_id = task_id.split('@')
    print(f"[DataReceiver] Starting ML Training Data Server")
    print(f"[DataReceiver] Fuzzer ID: {fuzzer_id}")
    print(f"[DataReceiver] Task Name: {task_name}")
    print(f"[DataReceiver] Run ID: {run_id}")
    print(f"[DataReceiver] Port: {port}")

    # Set gRPC options, set larger gRPC message limit
    max_msg_size = 300 * 1024 * 1024  # 300MB
    options = [
        ('grpc.max_send_message_length', max_msg_size),
        ('grpc.max_receive_message_length', max_msg_size),
        ('grpc.keepalive_time_ms', 60000),
        ('grpc.keepalive_timeout_ms', 30000),
        ('grpc.keepalive_permit_without_calls', True),
        ('grpc.http2.max_pings_without_data', 0),
    ]

    print(f"[DataReceiver] Max message size: {max_msg_size / 1024 / 1024:.1f}MB")
    if enable_compression:
        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=10),
            options=options,
            compression=grpc.Compression.Gzip  # Enable compression to reduce transmission size
        )
        print(f"[DataReceiver] Compression enabled: Gzip")
    else:
        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=10),
            options=options
        )
        print(f"[DataReceiver] Compression disabled")


    # Add service
    training_service = MLTrainingDataServer(
        data_dir, task_id, controller_addr, api_token, warmup_seconds
    )
    transmission_pb2_grpc.add_MLTrainingDataServiceServicer_to_server(training_service, server)

    # Start server
    listen_addr = f'[::]:{port}'
    server.add_insecure_port(listen_addr)
    server.start()

    print(f"[DataReceiver] Server started on {listen_addr}")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("\n[DataReceiver] Shutting down...")
        server.stop(0)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='ML Training Data Receiver')
    parser.add_argument('--port', type=int, default=50051, help='Server port')
    parser.add_argument('--data_dir', type=str, default='./training_data', help='Data directory')
    parser.add_argument('--task_id', type=str, default='fuzzer_id@task_name@1', help='Task ID')
    parser.add_argument('--controller_addr', type=str, default=None, help='Controller address')
    parser.add_argument('--api_token', type=str, default=None, help='API token')
    parser.add_argument(
        '--warmup_seconds', type=int, default=1800,
        help='Minimum data-collection warm-up before online training',
    )
    parser.add_argument('--enable_compression', type=bool, default=False, help='Enable compression')
    parser.add_argument('--log_file', type=str, default=None, help='Log file path (default: stdout)')


    args = parser.parse_args()

    serve(
        args.port, args.data_dir, args.task_id, args.controller_addr,
        args.api_token, args.warmup_seconds, args.enable_compression,
        args.log_file,
    )
