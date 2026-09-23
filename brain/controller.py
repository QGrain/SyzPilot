import os
import sys
import time
import json
import hashlib
import math
import re
import httpx
import utils
import asyncio
import argparse
import threading
import subprocess
import stat
import signal

from datetime import datetime
from dataclasses import dataclass, asdict, field
from pathlib import Path
from collections import deque
from typing import Any, Dict, Optional, Set, Deque, List, Tuple

from ts_operators import ServeOperator
from gpu_admission import AdmissionError, validate_physical_gpu_namespace

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from fastapi import Request, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import HTMLResponse, RedirectResponse

from config import ControllerConfig
from guidance_engine import GuidanceEngine, GuidanceConfig, MutationTemplate
from log_manager import LogManager, init_log_manager, get_log_manager
from syzlang_manifest import SyzlangIndex
from model_promotion import PromotionThresholds, decide_model_promotion

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
from common.curriculum import (
    CURRICULUM_SCHEMA_VERSION,
    curriculum_class,
    curriculum_output_classes,
)
from common.label_contract import is_positive_one_hot, one_hot_label_class

# ====== Global configuration and constants ======
config = ControllerConfig()

# Initialize centralized logging
log_mgr = init_log_manager(config.log_dir)
logger = log_mgr.controller

# SSH helper (from demo-server)
TUNNEL_USER = "fuzzer-tunnel"
HELPER_SCRIPT_PATH = "/usr/local/bin/add-fuzzer-key"
TUNNEL_USER_HOME = Path(f"/home/{TUNNEL_USER}")
AUTHORIZED_KEYS_FILE = TUNNEL_USER_HOME / ".ssh" / "authorized_keys"

# TorchServe configuration (uses brain/config.properties to specify port)

# ====== Data classes and global state ======
def _completed_event() -> threading.Event:
    event = threading.Event()
    event.set()
    return event


def _canonical_task_log_dir_name(task_id: str, task_name: str) -> str:
    """Return a readable, collision-resistant directory for one task's logs."""
    readable = "".join(
        char if char.isascii() and (char.isalnum() or char in "-_") else "_"
        for char in task_name
    ).strip("._")[:48] or "task"
    digest = hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:16]
    return f"{readable}-{digest}"


@dataclass
class FuzzerTask:
    task_id: str
    task_name: str
    run_id: int
    fuzzer_id: str
    mode: str  # "isolated" | "direct"
    callback_addr: str
    target_os: str = ""
    target_arch: str = ""
    target_revision: str = ""
    producer_revision: str = ""
    descriptions_mode: str = ""
    # members with default values
    tunnel_port: Optional[int] = None  # isolated
    grpc_port: int = 0
    receiver_proc: Optional[subprocess.Popen] = None
    trainer_proc: Optional[subprocess.Popen] = None
    trainer_log_fh: Optional[Any] = None  # file handle for trainer log
    attributor_proc: Optional[subprocess.Popen] = None
    registered_at: float = field(default_factory=time.time)
    model_name: str = ""
    model_version: str = ""
    model_name_prefix: str = ""

    training_gpu: str = ""  # GPU assigned for training (for release after completion)
    training_port: int = 0
    training_round: int = 0  # Number of successfully completed training runs
    training_in_progress: bool = False
    active_training_id: str = ""
    training_launch_done: threading.Event = field(
        default_factory=_completed_event, repr=False
    )
    training_cancel: threading.Event = field(
        default_factory=threading.Event, repr=False
    )
    last_trained_batch: int = 0
    last_successful_checkpoint: str = ""
    last_successful_stage: int = 0
    last_training_manifest: str = ""
    seen_training_batches: List[int] = field(default_factory=list)
    deployment_version: int = 0
    training_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    lifecycle_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    stopping: bool = False
    pending_model_name: str = ""
    pending_model_version: str = ""
    pending_checkpoint: str = ""
    pending_manifest: str = ""
    pending_torchscript: str = ""
    pending_stage: int = 0
    pending_batch: int = 0
    pending_training_round: int = 0
    pending_num_classes: int = 0
    pending_seen_training_batches: List[int] = field(default_factory=list)
    evaluation_watermarks: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    ssh_entry: str = ""

    # Guidance system
    target_func: str = ""  # target vulnerability function name
    kallgraph_dir: str = ""  # path to KallGraph output for this case
    report_path: str = ""  # path to crash report for path-based analysis
    report_text: str = field(default="", repr=False)  # validated immutable report
    guidance_engine: Optional[GuidanceEngine] = field(default=None, repr=False)
    guidance_version: int = 0  # how many guidance rounds have been sent
    static_analysis_done: bool = False  # whether static analysis has been run
    guidance_cancel: threading.Event = field(
        default_factory=threading.Event, repr=False
    )
    guidance_lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False
    )

    # Log storage (not serialized)
    log_dir_name: str = ""
    receiver_logs: Deque[str] = field(default_factory=lambda: deque(maxlen=1000), repr=False)
    trainer_logs: Deque[str] = field(default_factory=lambda: deque(maxlen=1000), repr=False)
    attributor_logs: Deque[str] = field(default_factory=lambda: deque(maxlen=1000), repr=False)

    def __post_init__(self):
        if not self.log_dir_name:
            self.log_dir_name = _canonical_task_log_dir_name(
                self.task_id, self.task_name
            )

    def to_dict(self):
        """Convert to serializable dictionary"""
        # TODO
        return {}


def next_curriculum_stage(
        requested_stage: int, last_successful_stage: int, num_classes: int) -> int:
    """Return the next mandatory objective for a fixed-width classifier head."""
    if requested_stage not in (1, 2):
        raise ValueError(f"invalid requested curriculum stage: {requested_stage}")
    if last_successful_stage not in (0, 1, 2):
        raise ValueError(
            f"invalid last successful curriculum stage: {last_successful_stage}"
        )
    if num_classes < 2:
        raise ValueError("curriculum learning requires at least two output classes")

    valid_stages = (1, 2)
    if last_successful_stage == 0:
        return valid_stages[0]
    if last_successful_stage not in valid_stages:
        raise ValueError(
            f"stage {last_successful_stage} is invalid for {num_classes} classes"
        )
    if requested_stage <= last_successful_stage:
        return last_successful_stage
    current_index = valid_stages.index(last_successful_stage)
    return valid_stages[min(current_index + 1, len(valid_stages) - 1)]

class RegistrationPayload(BaseModel):
    uuid: str
    task_name: str
    mode: str  # "isolated" | "direct"
    public_key: Optional[str] = None
    host_ip: Optional[str] = None
    http_port: Optional[int] = None
    target_func: Optional[str] = None  # target vulnerability function name
    kallgraph_dir: Optional[str] = None  # path to KallGraph output
    report_path: Optional[str] = None  # path to crash report for path-based analysis
    target_os: str
    target_arch: str
    target_revision: str
    producer_revision: str
    descriptions_mode: str

class RegistrationResponse(BaseModel):
    task_id: str
    receiver_port: int
    torchserve_port: int
    tunnel_port: Optional[int] = None
    model_name: str = ""
    model_version: str = "1.0"


def validate_task_identity(fuzzer_id: str, task_name: str) -> None:
    """Reject identities that break task-id parsing or storage isolation."""
    for field_name, value in (("uuid", fuzzer_id), ("task_name", task_name)):
        if not value or len(value) > 255:
            raise ValueError(f"{field_name} must contain 1 to 255 characters")
        if "@" in value or "\x00" in value or any(
            ord(character) < 32 for character in value
        ):
            raise ValueError(f"{field_name} contains unsupported characters")
    if re.fullmatch(r"[A-Za-z0-9_.-]+", fuzzer_id) is None:
        raise ValueError("uuid must be a URL-safe identifier")
    if (task_name in (".", "..") or Path(task_name).is_absolute() or
            any(character in task_name for character in "/\\?#%")):
        raise ValueError("task_name must be a single storage-safe name")


def validate_fuzzer_build_identity(payload: RegistrationPayload) -> None:
    """Validate task-local compiled target identity before allocating resources."""
    for field_name in (
            "target_os", "target_arch", "target_revision", "producer_revision"):
        value = getattr(payload, field_name)
        if (not value or len(value) > 255 or "\x00" in value or
                any(ord(character) < 32 for character in value)):
            raise ValueError(
                f"{field_name} must contain 1 to 255 printable characters"
            )
    if payload.descriptions_mode not in ("manual", "auto", "any"):
        raise ValueError("descriptions_mode must be manual, auto, or any")


def resolve_guidance_context(payload: RegistrationPayload) -> Tuple[str, str, str]:
    """Resolve registration guidance inputs, preferring explicit payload values."""
    return (
        payload.target_func or (config.target_funcs or {}).get(payload.task_name, ""),
        payload.kallgraph_dir or (config.kallgraph_dirs or {}).get(payload.task_name, ""),
        payload.report_path or (config.report_paths or {}).get(payload.task_name, ""),
    )


def _trusted_guidance_path(raw_path: str, allowed_roots: Tuple[str, ...],
                           *, kind: str, require_file: bool) -> str:
    """Resolve a Brain-local guidance path without escaping configured roots."""
    if not raw_path:
        return ""
    try:
        resolved = Path(raw_path).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ValueError(f"{kind} path is unavailable: {raw_path}") from error

    trusted_roots = []
    for root in allowed_roots:
        try:
            trusted_roots.append(Path(root).expanduser().resolve(strict=True))
        except (OSError, RuntimeError):
            continue
    containing_root = next(
        (root for root in trusted_roots if resolved.is_relative_to(root)),
        None,
    )
    if containing_root is None:
        raise ValueError(f"{kind} path is outside configured roots")
    if not resolved.relative_to(containing_root).parts:
        raise ValueError(f"{kind} path must not be the configured root itself")
    if require_file and not resolved.is_file():
        raise ValueError(f"{kind} path must be a regular file")
    if not require_file and not resolved.is_dir():
        raise ValueError(f"{kind} path must be a directory")
    return str(resolved)


def _read_trusted_report(report_path: str, allowed_roots: Tuple[str, ...],
                         max_bytes: int) -> str:
    """Read one trusted report through the descriptor that was validated."""
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(report_path, flags)
    except OSError as error:
        raise ValueError(f"report path is unavailable: {report_path}") from error

    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("report path must be a regular file")
        if metadata.st_size > max_bytes:
            raise ValueError("report exceeds the configured size limit")

        try:
            opened_path = Path(f"/proc/self/fd/{descriptor}").resolve(strict=True)
            trusted_roots = [
                Path(root).expanduser().resolve(strict=True)
                for root in allowed_roots
            ]
        except (OSError, RuntimeError) as error:
            raise ValueError("report path could not be verified after opening") from error
        if not any(opened_path.is_relative_to(root) for root in trusted_roots):
            raise ValueError("report path is outside configured roots")

        chunks = []
        remaining = max_bytes + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        if len(payload) > max_bytes:
            raise ValueError("report exceeds the configured size limit")
        try:
            return payload.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("report is not valid UTF-8 text") from error
    finally:
        os.close(descriptor)


def validate_guidance_context(target_func: str, kallgraph_dir: str,
                              report_path: str) -> Tuple[str, str, str, str]:
    """Validate and materialize untrusted Brain-local guidance context."""
    target_func = target_func.strip()
    if len(target_func) > 256 or any(char in target_func for char in "\r\n\0"):
        raise ValueError("target_func is invalid")
    report_path = _trusted_guidance_path(
        report_path,
        tuple(config.guidance_report_roots),
        kind="report",
        require_file=True,
    )
    report_text = ""
    if report_path:
        report_text = _read_trusted_report(
            report_path,
            tuple(config.guidance_report_roots),
            config.guidance_max_report_bytes,
        )
    kallgraph_dir = _trusted_guidance_path(
        kallgraph_dir,
        tuple(config.guidance_kallgraph_roots),
        kind="KallGraph",
        require_file=False,
    )
    if kallgraph_dir and not target_func and not report_path:
        raise ValueError("target_func is required with KallGraph guidance")
    return target_func, kallgraph_dir, report_path, report_text


class _TrainingLaunchCanceled(RuntimeError):
    """Internal signal that task teardown canceled trainer preflight."""


def _check_training_launch_canceled(
        cancel_event: Optional[threading.Event]) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise _TrainingLaunchCanceled("training launch was canceled")


def list_committed_batch_indices(
        data_dir: str, batch_end: int = 0,
        cancel_event: Optional[threading.Event] = None) -> List[int]:
    """List fully committed Receiver batches within an optional snapshot boundary."""
    import pickle

    indices = []
    for marker in Path(data_dir).glob("batch_*.complete"):
        _check_training_launch_canceled(cancel_event)
        try:
            batch_index = int(marker.stem.removeprefix("batch_"))
        except ValueError:
            continue
        if batch_end > 0 and batch_index > batch_end:
            continue
        progs_path = Path(data_dir) / f"progs_batch_{batch_index}.pkl"
        labels_path = Path(data_dir) / f"labels_batch_{batch_index}.pkl"
        if not progs_path.is_file():
            continue
        if not labels_path.is_file():
            continue
        try:
            fields = {}
            for line in marker.read_text(encoding="utf-8").splitlines():
                if "=" in line:
                    key, value = line.split("=", 1)
                    fields[key] = value
            if int(fields.get("schema_version", 0)) != 1:
                raise ValueError("invalid marker schema")
            if int(fields.get("batch_id", 0)) != batch_index:
                raise ValueError("marker batch ID mismatch")
            digest = fields.get("batch_digest", "")
            if len(digest) != 64 or any(
                char not in "0123456789abcdef" for char in digest
            ):
                raise ValueError("invalid marker digest")
            with progs_path.open("rb") as file_handle:
                programs = pickle.load(file_handle)  # nosec B301 - local Receiver state
            _check_training_launch_canceled(cancel_event)
            with labels_path.open("rb") as file_handle:
                labels = pickle.load(file_handle)  # nosec B301 - local Receiver state
            _check_training_launch_canceled(cancel_event)
            if (not isinstance(programs, dict) or not isinstance(labels, dict) or
                    not labels or programs.keys() != labels.keys()):
                raise ValueError("invalid committed payload dictionaries")
            if int(fields.get("unique_samples", -1)) != len(labels):
                raise ValueError("marker unique count mismatch")
            wire_samples = int(fields.get("wire_samples", -1))
            if wire_samples < len(labels) or wire_samples > 10000:
                raise ValueError("marker wire count is invalid")
            label_width = None
            for item_index, (signature, label) in enumerate(labels.items()):
                if item_index % 256 == 0:
                    _check_training_launch_canceled(cancel_event)
                program = programs[signature]
                if (not isinstance(program, str) or
                        hashlib.sha1(program.encode("utf-8")).hexdigest() != signature):
                    raise ValueError("program signature mismatch")
                if (not isinstance(label, (list, tuple)) or
                        sum(value is True for value in label) != 1 or
                        any(not isinstance(value, bool) for value in label)):
                    raise ValueError("invalid one-hot label")
                if label_width is None:
                    label_width = len(label)
                elif len(label) != label_width:
                    raise ValueError("inconsistent label width")
        except _TrainingLaunchCanceled:
            raise
        except Exception as error:
            logger.warning(
                f"Ignoring invalid committed batch {batch_index}: {error}"
            )
            continue
        indices.append(batch_index)
    return sorted(set(indices))


def build_training_split(batch_indices: List[int], last_trained_batch: int,
                         full_retrain: bool, seed: str) -> Tuple[List[int], List[int]]:
    """Build an 80/20 new-data split with 30% old replay on continuations."""
    import random

    source_indices = list(batch_indices) if full_retrain else [
        index for index in batch_indices if index > last_trained_batch
    ]
    if len(source_indices) < 2:
        raise ValueError(
            f"need at least 2 new batches, found {len(source_indices)}"
        )

    split_point = max(1, int(len(source_indices) * 0.8))
    if split_point >= len(source_indices):
        split_point = len(source_indices) - 1
    new_train = source_indices[:split_point]
    test_indices = source_indices[split_point:]
    if full_retrain:
        return new_train, test_indices

    old_pool = [index for index in batch_indices if index <= last_trained_batch]
    # old / (new + old) ~= 0.30, hence old ~= new * 3/7.
    old_count = min(len(old_pool), max(1, round(len(new_train) * 3 / 7)))
    rng = random.Random(seed)
    old_replay = rng.sample(old_pool, old_count) if old_count else []
    train_indices = new_train + old_replay
    rng.shuffle(train_indices)
    return train_indices, test_indices


def load_batch_signatures(
        data_dir: str, batch_indices: List[int],
        cancel_event: Optional[threading.Event] = None) -> Set[str]:
    """Load program signatures for a validated committed batch subset."""
    import pickle

    signatures = set()
    for batch_index in sorted(set(batch_indices)):
        _check_training_launch_canceled(cancel_event)
        path = Path(data_dir) / f"progs_batch_{batch_index}.pkl"
        with path.open("rb") as file_handle:
            programs = pickle.load(file_handle)  # nosec B301 - local Receiver state
        _check_training_launch_canceled(cancel_event)
        if not isinstance(programs, dict):
            raise ValueError(f"batch {batch_index} programs must be a dictionary")
        signatures.update(programs)
    return signatures


def curriculum_split_class_counts(
        data_dir: str, num_classes: int, canonical_indices: List[int],
        member_indices: List[int], exclude_indices: List[int],
        stage: int,
        cancel_event: Optional[threading.Event] = None) -> Dict[int, int]:
    """Count active curriculum classes in a canonical batch membership view."""
    import pickle

    canonical_classes = {}
    member_signatures = set()
    excluded_signatures = set()
    member_set = set(member_indices)
    exclude_set = set(exclude_indices)
    for batch_index in sorted(set(canonical_indices)):
        _check_training_launch_canceled(cancel_event)
        path = Path(data_dir) / f"labels_batch_{batch_index}.pkl"
        with path.open("rb") as file_handle:
            labels = pickle.load(file_handle)  # nosec B301 - local Receiver state
        _check_training_launch_canceled(cancel_event)
        if not isinstance(labels, dict):
            raise ValueError(f"batch {batch_index} labels must be a dictionary")
        for item_index, (signature, label) in enumerate(labels.items()):
            if item_index % 256 == 0:
                _check_training_launch_canceled(cancel_event)
            selected_class = one_hot_label_class(label, num_classes)
            if selected_class is None:
                raise ValueError(
                    f"batch {batch_index} has an invalid one-hot label"
                )
            previous = canonical_classes.get(signature)
            if previous is None or selected_class > previous:
                canonical_classes[signature] = selected_class
            if batch_index in member_set:
                member_signatures.add(signature)
            if batch_index in exclude_set:
                excluded_signatures.add(signature)

    counts = {
        class_index: 0
        for class_index in range(
            curriculum_output_classes(num_classes, stage)
        )
    }
    for signature in member_signatures - excluded_signatures:
        exact_class = canonical_classes.get(signature)
        if exact_class is None:
            raise ValueError("member signature is missing from canonical data")
        counts[curriculum_class(exact_class, num_classes, stage)] += 1
    return counts

# Add log reading utility function
def capture_subprocess_output(proc: subprocess.Popen, log_buffer: Deque[str], prefix: str):
    """Capture subprocess output in background thread"""
    def _read_stream(stream, buffer, stream_name):
        try:
            for line in iter(stream.readline, b''):
                if line:
                    timestamp = datetime.now().strftime('%H:%M:%S')
                    log_line = f"[{timestamp}] {line.decode('utf-8', errors='ignore').rstrip()}"
                    buffer.append(log_line)
        except Exception as e:
            buffer.append(f"[ERROR] Failed to read {stream_name}: {e}")

    # Start threads for stdout and stderr
    threading.Thread(target=_read_stream, args=(proc.stdout, log_buffer, 'stdout'), daemon=True).start()
    threading.Thread(target=_read_stream, args=(proc.stderr, log_buffer, 'stderr'), daemon=True).start()


def task_log_path(task: FuzzerTask, process_type: str) -> Path:
    """Resolve one task-owned log path below the configured log root."""
    if process_type not in ("receiver", "trainer"):
        raise ValueError("unsupported file-backed process log")
    log_root = Path(config.log_dir).resolve()
    log_path = (
        log_root / task.log_dir_name / str(task.run_id) / f"{process_type}.log"
    ).resolve()
    if os.path.commonpath((str(log_root), str(log_path))) != str(log_root):
        raise ValueError("task log path escapes configured log root")
    return log_path


def task_log_size(task: FuzzerTask, process_type: str) -> int:
    """Return the current task log size in bytes, or zero when absent."""
    log_path = task_log_path(task, process_type)
    try:
        file_stat = log_path.stat()
    except FileNotFoundError:
        return 0
    if not stat.S_ISREG(file_stat.st_mode):
        raise ValueError("task log is not a regular file")
    return file_stat.st_size


def read_task_log_tail(task: FuzzerTask, process_type: str, lines: int,
                       max_bytes: int = 1024 * 1024) -> List[str]:
    """Read a bounded CR/LF-aware tail from one task-owned log file."""
    log_path = task_log_path(task, process_type)
    if not log_path.is_file():
        return []

    with log_path.open("rb") as log_file:
        log_file.seek(0, os.SEEK_END)
        size = log_file.tell()
        offset = max(0, size - max_bytes)
        preceding = b""
        if offset:
            log_file.seek(offset - 1)
            preceding = log_file.read(1)
        log_file.seek(offset)
        data = log_file.read(max_bytes)
    if offset and preceding == b"\r" and data.startswith(b"\n"):
        data = data[1:]
    elif offset and preceding not in (b"\n", b"\r"):
        separators = [
            index for index in (data.find(b"\n"), data.find(b"\r"))
            if index >= 0
        ]
        if not separators:
            return []
        separator = min(separators)
        separator_end = separator + 1
        if data[separator:separator + 2] == b"\r\n":
            separator_end += 1
        data = data[separator_end:]
    return data.decode("utf-8", errors="replace").splitlines()[-lines:]


def start_receiver(task_id: str, task_name: str, run_id: int, grpc_port: int,
                   controller_addr: str, api_token: str) -> subprocess.Popen:
    if controller_addr == "localhost:None":
        raise ValueError("Controller port is not set")
    # Call brain/receiver.py
    here = os.path.dirname(os.path.abspath(__file__))
    current_env = os.environ.copy()
    current_env["PYTHONUNBUFFERED"] = "1"

    # Construct per-task log file path
    log_dir_name = _canonical_task_log_dir_name(task_id, task_name)
    log_file = os.path.join(
        config.log_dir, log_dir_name, str(run_id), "receiver.log"
    )

    cmd = [
        sys.executable, "receiver.py",
        "--port", str(grpc_port),
        "--data_dir", config.data_root,
        "--task_id", task_id,
        "--controller_addr", controller_addr,
        "--api_token", api_token,
        "--warmup_seconds", str(config.training_warmup_seconds),
        "--log_file", log_file
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=here,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            text=False,
            env=current_env
        )
    except Exception as e:
        logger.error(f"Failed to start receiver process: {e}")
        proc = None
    return proc

class Controller:
    """SzyPilot-brain Controller"""

    @staticmethod
    def _validate_cuda_device_namespace():
        """Require controller CUDA ordinals to match physical GPU indices."""
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            raise RuntimeError(
                "SyzPilot Controller must start without CUDA_VISIBLE_DEVICES; "
                "GPU arbitration uses physical indices and isolates trainer "
                "subprocesses explicitly"
            )
        device_order = os.environ.get("CUDA_DEVICE_ORDER")
        if device_order is None:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        elif device_order != "PCI_BUS_ID":
            raise RuntimeError(
                "SyzPilot Controller requires "
                "CUDA_DEVICE_ORDER=PCI_BUS_ID so configured GPU indices map "
                "to nvidia-smi physical indices"
            )

    @staticmethod
    def _load_export_state_dict(torch_module, ckpt_path: Path):
        """Load a trainer checkpoint without allocating on any GPU."""
        return torch_module.load(
            str(ckpt_path), map_location="cpu", weights_only=True
        )

    def __init__(self):
        self._validate_cuda_device_namespace()
        self.app = FastAPI(lifespan=self.lifespan)
        # dict key is task_id = fuzzer_id+task_name+run_id
        self.global_tasks: Dict[str, FuzzerTask] = {}
        self.task_stats: Dict[str, Any] = {
            "total_runs": 0,  # total number of runs in the scope of this controller process
            "run_history": {} # key is task_name, value is the number of runs
        }
        self.task_stats_lock = threading.Lock()
        self.reserved_run_ids: Set[tuple] = set()
        self.tunnel_port_pool: Set[int] = set(range(21001, 22000)) # per task
        self.grpc_port_pool: Set[int] = set(range(31001, 32000)) # per task
        self.training_port_pool: Set[int] = set(range(29500, 30000)) # accelerate DDP port
        self.port_pool_lock = threading.Lock()  # guards alloc/release on all port pools
        self.existing_ssh_entries: Set[str] = set()
        self.ssh_entry_lock = threading.Lock()
        self.torchserve_operator = ServeOperator(
            config.ts_model_store,
            disable_auth=True,
            inference_gpu_id=config.inference_gpu_id,
        )
        self.torchserve_started = False
        self.torchserve_proc: Optional[subprocess.Popen] = None
        self.controller_port: Optional[int] = None
        self._shutdown_cleanup_failed = False

        # GPU resource management for training load balancing.  Concurrent
        # 1024-token trainers share compute poorly, so capacity is explicit.
        self._training_gpu_ids = list(config.training_gpu_ids)
        self._training_fallback_gpu_ids = list(
            config.training_fallback_gpu_ids
        )
        physical_gpu_ids = dict.fromkeys((
            *self._training_gpu_ids,
            *self._training_fallback_gpu_ids,
            config.attribution_gpu_id,
        ))
        self._gpu_semaphores: Dict[str, threading.BoundedSemaphore] = {
            gpu_id: threading.BoundedSemaphore(1)
            for gpu_id in physical_gpu_ids
        }
        self._training_min_free_mib = config.training_min_free_mib
        self._training_max_gpu_utilization = (
            config.training_max_gpu_utilization
        )
        self._training_gpu_probe_samples = config.training_gpu_probe_samples
        self._training_gpu_probe_interval_seconds = (
            config.training_gpu_probe_interval_seconds
        )
        self._gpu_select_lock = threading.Lock()  # serializes GPU selection queries
        # Receivers retry durable training requests independently. Preserve a
        # process-local FIFO across retries so a high-volume task cannot take
        # the only free trainer slot repeatedly and starve another task.
        self._training_wait_lock = threading.Lock()
        self._training_wait_queue: Deque[str] = deque()
        self._training_wait_set: Set[str] = set()
        self._training_wait_last_seen: Dict[str, float] = {}
        self._training_waiter_lease_seconds = (
            config.training_waiter_lease_seconds
        )
        self._attribution_gpu_semaphore = threading.BoundedSemaphore(
            config.attribution_slots
        )

        # Dashboard setup
        if config.dashboard_enabled:
            static_path = Path(__file__).parent / "dashboard" / "static"
            templates_path = Path(__file__).parent / "dashboard" / "templates"

            if static_path.exists():
                self.app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

            if templates_path.exists():
                self.templates = Jinja2Templates(directory=str(templates_path))
            else:
                self.templates = None

        self.security = HTTPBearer(auto_error=False)

        self._setup_routes()

    def _alloc_port(self, pool: Set[int]) -> int:
        """Thread-safe port allocation from a pool."""
        with self.port_pool_lock:
            return utils.alloc_port(pool)

    def _release_port(self, pool: Set[int], port: int):
        """Thread-safe port release back to a pool."""
        with self.port_pool_lock:
            utils.release_port(pool, port)

    @staticmethod
    def _start_daemon_thread(target, *args):
        """Start one controller-owned daemon thread and return its handle."""
        worker = threading.Thread(target=target, args=args, daemon=True)
        worker.start()
        return worker

    def _task_accepts_guidance(self, task: FuzzerTask) -> bool:
        """Return whether side effects still belong to this registered task."""
        return (
            not task.guidance_cancel.is_set()
            and self.global_tasks.get(task.task_id) is task
        )

    def _send_guidance_if_active(
            self, task: FuzzerTask, engine: GuidanceEngine, guidance: dict
    ) -> Optional[bool]:
        """Send while holding the lifecycle barrier, or cancel without I/O.

        Holding ``lifecycle_lock`` across the bounded HTTP operation prevents
        unregistration from releasing or reusing the callback/tunnel endpoint
        until this task's last possible send has finished.
        """
        with task.lifecycle_lock:
            if not self._task_accepts_guidance(task):
                logger.info(
                    f"[{task.task_id}] Guidance send canceled for inactive task"
                )
                return None
            send_ok = engine.send_guidance(
                task.callback_addr,
                guidance=guidance,
                cancel_event=task.guidance_cancel,
            )
            if send_ok:
                task.guidance_version = engine.version
            return send_ok

    @staticmethod
    def _training_state_path(task: FuzzerTask) -> Path:
        return Path(config.data_root) / task.task_name / str(task.run_id) / "training_state.json"

    @staticmethod
    def _manifest_seen_training_batches(manifest_path: str) -> List[int]:
        """Recover the exact cumulative training set from a model manifest."""
        if not manifest_path:
            return []
        try:
            manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
            values = manifest.get(
                "seen_train_batch_indices",
                manifest.get("test_exclude_batch_indices", []),
            )
            if not isinstance(values, list):
                raise TypeError("seen training batch indices must be a list")
            return sorted({int(value) for value in values if int(value) > 0})
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            logger.warning(
                "Failed to recover seen training batches from manifest %s: %s",
                manifest_path,
                error,
            )
            return []

    def _persist_training_state(self, task: FuzzerTask):
        """Persist committed training/deployment state for auditability and recovery."""
        with task.training_lock:
            state = {
                "schema_version": 1,
                "curriculum_schema": CURRICULUM_SCHEMA_VERSION,
                "training_round": task.training_round,
                "last_trained_batch": task.last_trained_batch,
                "last_successful_checkpoint": task.last_successful_checkpoint,
                "last_successful_stage": task.last_successful_stage,
                "last_training_manifest": task.last_training_manifest,
                "seen_training_batches": task.seen_training_batches,
                "deployment_version": task.deployment_version,
                "model_name": task.model_name,
                "model_version": task.model_version,
                "pending_model_name": task.pending_model_name,
                "pending_model_version": task.pending_model_version,
                "pending_checkpoint": task.pending_checkpoint,
                "pending_manifest": task.pending_manifest,
                "pending_torchscript": task.pending_torchscript,
                "pending_stage": task.pending_stage,
                "pending_batch": task.pending_batch,
                "pending_training_round": task.pending_training_round,
                "pending_num_classes": task.pending_num_classes,
                "pending_seen_training_batches": task.pending_seen_training_batches,
                "evaluation_watermarks": {
                    str(stage): watermark
                    for stage, watermark in task.evaluation_watermarks.items()
                },
            }
        state_path = self._training_state_path(task)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = state_path.with_suffix(".json.tmp")
        with temp_path.open("w", encoding="utf-8") as file_handle:
            json.dump(state, file_handle, indent=2, sort_keys=True)
            file_handle.write("\n")
            file_handle.flush()
            os.fsync(file_handle.fileno())
        os.replace(temp_path, state_path)

    def _load_training_state(self, task: FuzzerTask):
        state_path = self._training_state_path(task)
        if not state_path.is_file():
            return
        recovered_seen_batches = False
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if not isinstance(state, dict):
                raise ValueError("training state root must be an object")
            if state.get("schema_version") != 1:
                raise ValueError("unsupported training state schema")
            last_successful_stage = int(state.get("last_successful_stage", 0))
            pending_stage = int(state.get("pending_stage", 0))
            curriculum_schema = state.get("curriculum_schema")
            if curriculum_schema is None:
                if (last_successful_stage not in (0, 1, 2) or
                        pending_stage not in (0, 1, 2)):
                    raise ValueError("legacy two-stage state has an invalid stage")
                recovered_seen_batches = True
            elif int(curriculum_schema) == 1:
                if last_successful_stage > 1 or pending_stage > 1:
                    archive_path = state_path.with_name(
                        f"{state_path.stem}.three-stage-incompatible-"
                        f"{time.time_ns()}{state_path.suffix}"
                    )
                    os.replace(state_path, archive_path)
                    logger.warning(
                        "[%s] Archived incompatible three-stage training "
                        "state at %s; retained data will be retrained",
                        task.task_id,
                        archive_path,
                    )
                    return
                recovered_seen_batches = True
            elif int(curriculum_schema) != CURRICULUM_SCHEMA_VERSION:
                raise ValueError("unsupported curriculum schema")
            elif (last_successful_stage not in (0, 1, 2) or
                    pending_stage not in (0, 1, 2)):
                raise ValueError("training state has an invalid curriculum stage")
            raw_evaluation_watermarks = state.get(
                "evaluation_watermarks", {}
            )
            if not isinstance(raw_evaluation_watermarks, dict):
                raise ValueError("model evaluation watermarks must be an object")
            evaluation_watermarks = {}
            for raw_stage, raw_watermark in raw_evaluation_watermarks.items():
                watermark_stage = int(raw_stage)
                if (watermark_stage not in (1, 2) or
                        not isinstance(raw_watermark, dict)):
                    raise ValueError("invalid model evaluation watermark")
                watermark = dict(raw_watermark)
                watermark["batch_end"] = int(watermark["batch_end"])
                watermark["disposition"] = str(watermark["disposition"])
                reasons = watermark.get("reason_codes", [])
                if (watermark["batch_end"] <= 0 or
                        watermark["disposition"] != "rejected" or
                        not isinstance(reasons, list) or
                        any(not isinstance(reason, str) for reason in reasons)):
                    raise ValueError("invalid model evaluation watermark")
                watermark["reason_codes"] = reasons
                evaluation_watermarks[watermark_stage] = watermark
            with task.training_lock:
                task.training_round = int(state.get("training_round", 0))
                task.last_trained_batch = int(state.get("last_trained_batch", 0))
                task.last_successful_checkpoint = state.get(
                    "last_successful_checkpoint", ""
                )
                task.last_successful_stage = last_successful_stage
                task.last_training_manifest = state.get("last_training_manifest", "")
                task.seen_training_batches = sorted(set(map(
                    int, state.get("seen_training_batches", [])
                )))
                task.deployment_version = int(state.get("deployment_version", 0))
                task.model_name = state.get("model_name", "")
                task.model_version = state.get("model_version", "")
                task.pending_model_name = state.get("pending_model_name", "")
                task.pending_model_version = state.get("pending_model_version", "")
                task.pending_checkpoint = state.get("pending_checkpoint", "")
                task.pending_manifest = state.get("pending_manifest", "")
                task.pending_torchscript = state.get("pending_torchscript", "")
                task.pending_stage = pending_stage
                task.pending_batch = int(state.get("pending_batch", 0))
                task.pending_training_round = int(
                    state.get("pending_training_round", 0)
                )
                task.pending_num_classes = int(state.get("pending_num_classes", 0))
                task.pending_seen_training_batches = sorted(set(map(
                    int, state.get("pending_seen_training_batches", [])
                )))
                task.evaluation_watermarks = evaluation_watermarks

                if not task.seen_training_batches and task.last_trained_batch > 0:
                    task.seen_training_batches = self._manifest_seen_training_batches(
                        task.last_training_manifest
                    )
                    if not task.seen_training_batches:
                        # Legacy manifests did not record the cumulative training
                        # membership. Conservatively exclude the committed prefix
                        # rather than risk train/test signature leakage.
                        task.seen_training_batches = list(range(
                            1, task.last_trained_batch + 1
                        ))
                    recovered_seen_batches = True

                if (task.pending_model_name and
                        not task.pending_seen_training_batches):
                    task.pending_seen_training_batches = (
                        self._manifest_seen_training_batches(task.pending_manifest)
                    )
                    recovered_seen_batches = bool(
                        task.pending_seen_training_batches
                    ) or recovered_seen_batches
        except (
            OSError, AttributeError, KeyError, TypeError, ValueError,
            json.JSONDecodeError,
        ) as error:
            logger.error(f"[{task.task_id}] Failed to load training state: {error}")
            return

        if recovered_seen_batches:
            try:
                self._persist_training_state(task)
            except Exception as error:
                # The recovered in-memory state is safe to use. Do not abort
                # registration after Receiver/port resources were allocated
                # merely because the audit-state rewrite failed.
                logger.error(
                    f"[{task.task_id}] Failed to persist recovered training "
                    f"state: {error}"
                )

    def _rank_training_gpus(
            self, cancel_event: Optional[threading.Event] = None) -> List[str]:
        """Rank idle, memory-eligible configured GPUs for one trainer.

        A successful query enforces both the configured free-memory and GPU
        utilization gates. A failed or malformed query returns no candidates;
        the Receiver then retries later instead of selecting a GPU blindly.
        """
        import csv
        import io

        sampled_status = []
        for sample_index in range(self._training_gpu_probe_samples):
            if cancel_event is not None and cancel_event.is_set():
                return []
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu="
                     "index,memory.used,memory.total,utilization.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=2
                )
            except (OSError, subprocess.SubprocessError) as error:
                logger.warning(
                    "GPU status query failed (%s); deferring training",
                    error,
                )
                return []
            if cancel_event is not None and cancel_event.is_set():
                return []
            if result.returncode != 0:
                logger.warning(
                    "nvidia-smi exited with %d; deferring training",
                    result.returncode,
                )
                return []

            current_status = {}
            for line_number, line in enumerate(
                    csv.reader(io.StringIO(result.stdout)), start=1):
                if len(line) != 4:
                    logger.warning(
                        "Ignoring malformed nvidia-smi row %d", line_number
                    )
                    continue
                try:
                    gpu_idx = line[0].strip()
                    mem_used = int(line[1].strip())
                    mem_total = int(line[2].strip())
                    gpu_utilization = int(line[3].strip())
                except ValueError:
                    logger.warning(
                        "Ignoring malformed nvidia-smi row %d", line_number
                    )
                    continue
                if (mem_total <= 0 or mem_used < 0 or mem_used > mem_total or
                        not 0 <= gpu_utilization <= 100):
                    logger.warning(
                        "Ignoring invalid nvidia-smi row %d", line_number
                    )
                    continue
                if gpu_idx in self._gpu_semaphores:
                    current_status[gpu_idx] = (
                        mem_total - mem_used,
                        gpu_utilization,
                    )
            sampled_status.append(current_status)
            if sample_index + 1 < self._training_gpu_probe_samples:
                if cancel_event is None:
                    time.sleep(self._training_gpu_probe_interval_seconds)
                elif cancel_event.wait(
                        self._training_gpu_probe_interval_seconds):
                    return []

        gpu_status = {}
        training_gpu_ids = (
            self._training_gpu_ids + self._training_fallback_gpu_ids
        )
        for gpu_id in training_gpu_ids:
            observations = [
                sample[gpu_id] for sample in sampled_status
                if gpu_id in sample
            ]
            if len(observations) != self._training_gpu_probe_samples:
                continue
            gpu_status[gpu_id] = (
                min(free_mib for free_mib, _ in observations),
                max(utilization for _, utilization in observations),
            )

        def rank_tier(gpu_ids):
            return sorted(
                (
                    gpu_id for gpu_id in gpu_ids
                    if gpu_status.get(gpu_id, (-1, 101))[0] >=
                    self._training_min_free_mib and
                    gpu_status[gpu_id][1] <=
                    self._training_max_gpu_utilization
                ),
                key=lambda gpu_id: (
                    gpu_status[gpu_id][1],
                    -gpu_status[gpu_id][0],
                    gpu_ids.index(gpu_id),
                ),
            )

        ranked_primary = rank_tier(self._training_gpu_ids)
        ranked_fallback = rank_tier(self._training_fallback_gpu_ids)
        ranked = ranked_primary + ranked_fallback
        if ranked:
            logger.info(
                "Eligible training GPUs after %d stable samples: "
                "primary=[%s], fallback=[%s]",
                self._training_gpu_probe_samples,
                ", ".join(
                    f"{gpu_id}={gpu_status[gpu_id][1]}%/"
                    f"{gpu_status[gpu_id][0]}MiB"
                    for gpu_id in ranked_primary
                ),
                ", ".join(
                    f"{gpu_id}={gpu_status[gpu_id][1]}%/"
                    f"{gpu_status[gpu_id][0]}MiB"
                    for gpu_id in ranked_fallback
                ),
            )
        else:
            logger.warning(
                "No configured GPU stayed above %d MiB free memory and at "
                "or below %d%% utilization across %d samples",
                self._training_min_free_mib,
                self._training_max_gpu_utilization,
                self._training_gpu_probe_samples,
            )
        return ranked

    def _acquire_training_gpu(
            self, cancel_event: Optional[threading.Event] = None) -> Optional[str]:
        """Try once to acquire a slot on a memory-ranked eligible GPU.

        The Receiver owns the durable retry timer, so this method does not wait
        or poll when all GPUs are busy.  Every attempt tries all ranked GPUs,
        so a locally occupied best GPU cannot force selection of an externally
        loaded lower-numbered GPU.

        Returns the GPU index string, or None if no slot is currently eligible.
        """
        if cancel_event is not None and cancel_event.is_set():
            return None
        # GPU telemetry is read-only and can take a few seconds when stable
        # sampling is enabled. Probe outside the selection lock so concurrent
        # requests and canceled waiters cannot create head-of-line blocking.
        ranked_gpus = self._rank_training_gpus(cancel_event=cancel_event)
        if cancel_event is not None and cancel_event.is_set():
            return None
        with self._gpu_select_lock:
            for gpu_id in ranked_gpus:
                if cancel_event is not None and cancel_event.is_set():
                    return None
                if self._gpu_semaphores[gpu_id].acquire(blocking=False):
                    if cancel_event is not None and cancel_event.is_set():
                        self._gpu_semaphores[gpu_id].release()
                        return None
                    logger.info(f"Acquired training slot on GPU {gpu_id}")
                    return gpu_id
        logger.warning("No eligible training GPU slot is currently available")
        return None

    async def _acquire_training_gpu_async(
            self, cancel_event: threading.Event) -> Optional[str]:
        """Run the bounded GPU probe off-loop and reclaim on cancellation."""
        worker = asyncio.create_task(asyncio.to_thread(
            self._acquire_training_gpu, cancel_event=cancel_event
        ))
        try:
            return await asyncio.shield(worker)
        except asyncio.CancelledError as cancel_error:
            cancel_event.set()
            # A task may be canceled repeatedly while unregister waits for
            # cleanup. Keep the worker shielded until the thread observes the
            # cancel event; otherwise a late slot acquisition could leak its
            # semaphore permanently.
            while True:
                try:
                    acquired_gpu = await asyncio.shield(worker)
                    break
                except asyncio.CancelledError:
                    cancel_event.set()
            if acquired_gpu is not None:
                self._release_training_gpu(acquired_gpu)
            raise cancel_error

    @staticmethod
    async def _acquire_thread_lock_async(lock: threading.Lock):
        """Acquire a thread lock without leaking it on coroutine cancellation."""
        worker = asyncio.create_task(asyncio.to_thread(lock.acquire))
        try:
            await asyncio.shield(worker)
        except asyncio.CancelledError as cancel_error:
            # ``to_thread`` cannot stop a blocking lock.acquire(). Reclaim a
            # late acquisition before propagating cancellation to ASGI.
            while True:
                try:
                    acquired = await asyncio.shield(worker)
                    break
                except asyncio.CancelledError:
                    continue
            if acquired:
                lock.release()
            raise cancel_error

    def _release_training_gpu(self, gpu_id: str):
        """Release a training slot on the given GPU."""
        if gpu_id in self._gpu_semaphores:
            self._gpu_semaphores[gpu_id].release()
            logger.info(f"Released training slot on GPU {gpu_id}")

    def _enqueue_training_waiter(self, task_id: str) -> int:
        """Join the durable-retry FIFO and return the zero-based position."""
        with self._training_wait_lock:
            now = time.monotonic()
            self._prune_training_waiters_locked(now)
            if task_id not in self._training_wait_set:
                self._training_wait_queue.append(task_id)
                self._training_wait_set.add(task_id)
            self._training_wait_last_seen[task_id] = now
            return self._training_wait_queue.index(task_id)

    def _remove_training_waiter(self, task_id: str):
        """Remove a task from the trainer FIFO, including stale retries."""
        with self._training_wait_lock:
            self._discard_training_waiter_locked(task_id)

    def _has_valid_training_waiter(self) -> bool:
        """Return whether a live trainer request should precede fallback IG."""
        with self._training_wait_lock:
            self._prune_training_waiters_locked(time.monotonic())
            return bool(self._training_wait_queue)

    def _discard_training_waiter_locked(self, task_id: str):
        """Remove one waiter while the queue lock is held."""
        self._training_wait_last_seen.pop(task_id, None)
        if task_id not in self._training_wait_set:
            return
        self._training_wait_set.remove(task_id)
        self._training_wait_queue.remove(task_id)

    def _prune_training_waiters_locked(self, now: float):
        """Drop dead or expired FIFO heads while the queue lock is held."""
        while self._training_wait_queue:
            task_id = self._training_wait_queue[0]
            task = self.global_tasks.get(task_id)
            stale_reason = ""
            if task is None:
                stale_reason = "task is no longer registered"
            elif task.stopping:
                stale_reason = "task is stopping"
            elif (task.receiver_proc is not None and
                  task.receiver_proc.poll() is not None):
                stale_reason = "Receiver process exited"
            elif not task.training_in_progress:
                last_seen = self._training_wait_last_seen.get(task_id, 0.0)
                if now - last_seen > self._training_waiter_lease_seconds:
                    stale_reason = "waiter lease expired"
            if not stale_reason:
                return
            logger.warning(
                "Removing stale training waiter %s: %s", task_id, stale_reason
            )
            self._discard_training_waiter_locked(task_id)

    def _get_token_dependency(self):
        """Factory method to create token verification dependency"""
        async def _verify(credentials: HTTPAuthorizationCredentials = Depends(self.security)):
            return self.verify_token(credentials)
        return _verify

    def verify_token(self, credentials: Optional[HTTPAuthorizationCredentials]) -> bool:
        """Verify dashboard access token"""
        if not config.dashboard_enabled:
            raise HTTPException(status_code=404, detail="Dashboard disabled")

        if not credentials:
            raise HTTPException(
                status_code=401,
                detail="Missing authentication token",
                headers={"WWW-Authenticate": "Bearer"},
            )

        if credentials.credentials != config.dashboard_token:
            raise HTTPException(
                status_code=401,
                detail="Invalid authentication token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return True

    def _setup_routes(self):
        """Bind class methods to FastAPI routes"""
        token_dep = Depends(self._get_token_dependency())
        self.app.add_api_route("/register", self.register, methods=["POST"])
        self.app.add_api_route("/unregister/{task_id}", self.unregister, methods=["POST"])
        self.app.add_api_route("/ping_fuzzer/{task_id}", self.ping_fuzzer, methods=["GET"])
        self.app.add_api_route("/api/stats", self.get_stats, methods=["GET"])
        self.app.add_api_route("/list_tasks", self.list_tasks, methods=["GET"])
        # self.app.add_api_route("/start_trainer/{task_id}", self.start_trainer, methods=["POST"])
        # self.app.add_api_route("/start_attributor/{task_id}", self.start_attributor, methods=["POST"])
        self.app.add_api_route("/health", self.health, methods=["GET"])

        # Task log APIs
        self.app.add_api_route("/api/task/{task_id}/logs/{process_type}",
                              self.get_task_logs, methods=["GET"],
                              dependencies=[token_dep])
        self.app.add_api_route("/api/task/{task_id}/detail",
                              self.get_task_detail, methods=["GET"],
                              dependencies=[token_dep])

        # Dashboard routes
        if config.dashboard_enabled and self.templates:
            self.app.add_api_route("/", self.redirect_to_dashboard, methods=["GET"])
            self.app.add_api_route("/dashboard", self.dashboard_page, methods=["GET"], response_class=HTMLResponse)
            self.app.add_api_route("/login", self.login_page, methods=["GET"], response_class=HTMLResponse)
            self.app.add_api_route("/api/verify_token", self.verify_token_endpoint, methods=["POST"])

            # Protected routes - use dependencies parameter
            # self.app.add_api_route("/api/stats", self.get_stats, methods=["GET"], dependencies=[token_dep])
            # self.app.add_api_route("/list_tasks", self.list_tasks, methods=["GET"], dependencies=[token_dep])
            self.app.add_api_route("/start_trainer/{task_id}", self.start_trainer, methods=["POST"], dependencies=[token_dep])
            self.app.add_api_route("/start_attributor/{task_id}", self.start_attributor, methods=["POST"], dependencies=[token_dep])

    def startup(self):
        self._validate_physical_gpu_mapping()
        if config.direct_only:
            logger.info("Direct-only mode: skipping SSH tunnel helper preflight")
            self._ensure_torchserve_started()
            return
        logger.info("--- Running environment preflight check for tunnel service ---")
        # preflight check: helper script is executable with sudo privilege
        try:
            utils.run_cmd(["test", "-x", HELPER_SCRIPT_PATH])
            utils.run_cmd(["sudo", "-n", "-u", TUNNEL_USER, HELPER_SCRIPT_PATH, TUNNEL_USER, "--check"])
        except subprocess.CalledProcessError as e:
            # If you are in a direct-only scenario, you may choose to ignore this. But by default it must be available.
            raise RuntimeError(f"Helper preflight failed: {e.stderr or e.stdout}")
        logger.info("--- Preflight check for tunnel service passed ---")
        self._load_existing_keys()

        # start torchserve
        self._ensure_torchserve_started()

    def _validate_physical_gpu_mapping(self):
        """Fail closed when direct-start CUDA and NVML identities disagree."""
        required_gpu_ids = {
            *self._training_gpu_ids,
            *self._training_fallback_gpu_ids,
            config.attribution_gpu_id,
            config.inference_gpu_id,
        }
        try:
            validate_physical_gpu_namespace(required_gpu_ids)
        except AdmissionError as error:
            raise RuntimeError(
                f"Controller GPU identity validation failed: {error}"
            ) from error
        logger.info(
            "Controller GPU identity validation passed for physical GPUs: %s",
            ", ".join(sorted(required_gpu_ids, key=int)),
        )

    def _add_ssh_entry_if_not_exists(self, ssh_entry: str):
        logger.info(f"Adding ssh entry: {ssh_entry}")
        with self.ssh_entry_lock:
            logger.info(f"Existing SSH entries: {self.existing_ssh_entries}")
            if ssh_entry in self.existing_ssh_entries:
                logger.info(f"SSH entry already exists: {ssh_entry}")
                return
            utils.run_cmd(["sudo", "-n", "-u", TUNNEL_USER, HELPER_SCRIPT_PATH, TUNNEL_USER, ssh_entry])
            self.existing_ssh_entries.add(ssh_entry)

    def _load_existing_keys(self):
        # Pull all entries into memory set via helper '--read'
        out = utils.run_cmd(["sudo", "-n", "-u", TUNNEL_USER, HELPER_SCRIPT_PATH, TUNNEL_USER, "--read"]).stdout
        lines = [line for line in out.strip().split("\n") if line]
        self.existing_ssh_entries.clear()
        self.existing_ssh_entries.update(lines)

    def _clear_authorized_keys(self):
        utils.run_cmd([
            "sudo", "-n", "-u", TUNNEL_USER, HELPER_SCRIPT_PATH,
            TUNNEL_USER, "--clear",
        ])
        self.existing_ssh_entries.clear()

    def _remove_ssh_entry(self, ssh_entry: str):
        if not ssh_entry:
            return
        with self.ssh_entry_lock:
            if ssh_entry not in self.existing_ssh_entries:
                return
            try:
                utils.run_cmd([
                    "sudo", "-n", "-u", TUNNEL_USER, HELPER_SCRIPT_PATH,
                    TUNNEL_USER, "--remove", ssh_entry,
                ])
            except subprocess.CalledProcessError:
                # Compatibility with helpers installed before --remove was
                # added. Rebuild the small file while holding the global lock.
                retained = self.existing_ssh_entries - {ssh_entry}
                utils.run_cmd([
                    "sudo", "-n", "-u", TUNNEL_USER, HELPER_SCRIPT_PATH,
                    TUNNEL_USER, "--clear",
                ])
                for entry in sorted(retained):
                    utils.run_cmd([
                        "sudo", "-n", "-u", TUNNEL_USER, HELPER_SCRIPT_PATH,
                        TUNNEL_USER, entry,
                    ])
            self.existing_ssh_entries.discard(ssh_entry)

    def _ensure_torchserve_started(self):
        if self.torchserve_started:
            return
        self.torchserve_operator.create_config()
        self.torchserve_proc = self.torchserve_operator.start_service(self.torchserve_operator.config_path)
        if self.torchserve_proc is None:
            raise RuntimeError("Failed to start TorchServe")
        self.torchserve_started = True
        logger.info("--- TorchServe started ---")

    def _ensure_torchserve_stopped(self):
        if (not self.torchserve_started and
                not self.torchserve_operator.has_owned_service):
            return
        self.torchserve_operator.stop_service()
        self.torchserve_started = False
        self.torchserve_proc = None
        logger.info("--- TorchServe stopped ---")

    def _get_history_task_runs(self, task_name: str) -> int:
        """
        Get the number of history task runs for a given task name.
        Counts both on-disk directories and in-memory registered tasks
        to avoid run_id collisions from concurrent registrations.
        """
        task_dir = os.path.join(config.data_root, task_name)
        disk_count = 0
        if os.path.exists(task_dir):
            disk_count = len(os.listdir(task_dir))

        # Also count in-memory tasks with this task_name that haven't been
        # counted on disk yet (receiver hasn't created the directory)
        mem_count = sum(1 for t in self.global_tasks.values()
                       if t.task_name == task_name and t.run_id > disk_count)

        return max(disk_count, disk_count + mem_count)

    def _get_run_id(self, task_name: str) -> int:
        """
        Get the current run id for a given task name. Thread-safe: uses
        task_stats_lock to prevent concurrent registrations from getting
        the same run_id.
        """
        with self.task_stats_lock:
            # Recount under lock to avoid race conditions
            task_dir = os.path.join(config.data_root, task_name)
            disk_count = 0
            if os.path.exists(task_dir):
                disk_count = len(os.listdir(task_dir))

            # Count in-memory tasks that would bump the counter
            existing_run_ids = {t.run_id for t in self.global_tasks.values()
                               if t.task_name == task_name}
            max_existing = max(existing_run_ids, default=0)
            reserved_ids = {
                run_id for reserved_name, run_id in self.reserved_run_ids
                if reserved_name == task_name
            }
            max_reserved = max(reserved_ids, default=0)

            current_run_id = max(
                disk_count + 1, max_existing + 1, max_reserved + 1
            )
            self.reserved_run_ids.add((task_name, current_run_id))

            self.task_stats["run_history"][task_name] = current_run_id
            self.task_stats["total_runs"] += 1
        return current_run_id

    def _release_run_id_reservation(self, task_name: str, run_id: int):
        with self.task_stats_lock:
            self.reserved_run_ids.discard((task_name, run_id))

    # ====== Routes: register/unregister/probe ======
    async def redirect_to_dashboard(self):
        """Redirect root to dashboard"""
        return RedirectResponse(url="/dashboard")

    async def dashboard_page(self, request: Request):
        """Render dashboard page"""
        if not self.templates:
            raise HTTPException(status_code=404, detail="Dashboard not available")
        return self.templates.TemplateResponse("dashboard.html", {"request": request})

    async def login_page(self, request: Request):
        """Render login page"""
        if not self.templates:
            raise HTTPException(status_code=404, detail="Dashboard not available")
        return self.templates.TemplateResponse("login.html", {"request": request})

    async def verify_token_endpoint(self, credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer(auto_error=False))):
        """Endpoint to verify token validity"""
        try:
            self.verify_token(credentials)
            return {"valid": True}
        except HTTPException:
            return {"valid": False}

    async def get_task_logs(self, task_id: str, process_type: str, lines: int = 100):
        """Get logs for a specific process of a task"""
        task = self.global_tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        if lines < 1 or lines > 1000:
            raise HTTPException(
                status_code=400, detail="lines must be between 1 and 1000"
            )

        if process_type in ("receiver", "trainer"):
            try:
                logs = read_task_log_tail(task, process_type, lines)
            except (OSError, ValueError) as error:
                logger.error(
                    f"[{task.task_id}] Failed to read {process_type} log: {error}"
                )
                raise HTTPException(
                    status_code=500, detail=f"Failed to read {process_type} log"
                ) from error
            return {"logs": logs, "total": len(logs), "source": "file"}
        if process_type != "attributor":
            raise HTTPException(status_code=400, detail="Invalid process type")

        logs = list(task.attributor_logs)[-lines:]
        return {
            "logs": logs,
            "total": len(task.attributor_logs),
            "source": "memory",
        }

    async def get_task_detail(self, task_id: str):
        """Get detailed information about a task"""
        task = self.global_tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        # Get process status
        def get_proc_status(proc):
            if proc is None:
                return "not_started"
            elif proc.poll() is None:
                return "running"
            else:
                return f"exited_{proc.returncode}"

        return {
            "task_id": task.task_id,
            "task_name": task.task_name,
            "run_id": task.run_id,
            "fuzzer_id": task.fuzzer_id,
            "mode": task.mode,
            "grpc_port": task.grpc_port,
            "tunnel_port": task.tunnel_port,
            "callback_addr": task.callback_addr,
            "registered_at": task.registered_at,
            "fuzzer_build": {
                "target_os": task.target_os,
                "target_arch": task.target_arch,
                "target_revision": task.target_revision,
                "producer_revision": task.producer_revision,
                "descriptions_mode": task.descriptions_mode,
            },
            "processes": {
                "receiver": get_proc_status(task.receiver_proc),
                "trainer": get_proc_status(task.trainer_proc),
                "attributor": get_proc_status(task.attributor_proc),
            },
            "log_sizes": {
                "unit": "bytes",
                "receiver": task_log_size(task, "receiver"),
                "trainer": task_log_size(task, "trainer"),
                "attributor": sum(
                    len(entry.encode("utf-8")) for entry in task.attributor_logs
                ),
            },
            "guidance": {
                "target_func": task.target_func,
                "has_engine": task.guidance_engine is not None,
                "guidance_version": task.guidance_version,
                "static_analysis_done": task.static_analysis_done,
            },
        }

    async def get_stats(self):
        """Get global statistics for dashboard"""
        total_runs = 0
        with self.task_stats_lock:
            total_runs = self.task_stats["total_runs"]

        torchserve_ready = await asyncio.to_thread(
            self.torchserve_operator.is_service_ready
        )
        if torchserve_ready:
            torchserve_status = "running"
        elif self.torchserve_operator.has_owned_service:
            torchserve_status = "unhealthy"
        else:
            torchserve_status = "stopped"
        return {
            "active_tasks": len(self.global_tasks),
            "total_runs": total_runs,
            "torchserve_status": torchserve_status,
            "available_grpc_ports": len(self.grpc_port_pool),
            "available_tunnel_ports": len(self.tunnel_port_pool),
            "available_training_ports": len(self.training_port_pool),
        }

    async def register(self, payload: RegistrationPayload):
        if payload.mode not in ("isolated", "direct"):
            raise HTTPException(status_code=400, detail="mode must be 'isolated' or 'direct'")
        if config.direct_only and payload.mode != "direct":
            raise HTTPException(
                status_code=400,
                detail="isolated mode is unavailable in direct-only deployment",
            )
        if payload.mode == "direct" and (not payload.host_ip or not payload.http_port):
            raise HTTPException(
                status_code=400,
                detail="direct mode requires host_ip and http_port",
            )
        if payload.mode == "isolated" and not payload.public_key:
            raise HTTPException(
                status_code=400,
                detail="isolated mode requires public_key",
            )
        try:
            validate_task_identity(payload.uuid, payload.task_name)
            validate_fuzzer_build_identity(payload)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        # Guidance paths originate in an unauthenticated fuzzer payload. Fully
        # validate them, and materialize report text, before reserving a run ID,
        # allocating a port, or starting a Receiver process.
        try:
            target_func, kallgraph_dir, report_path, report_text = (
                validate_guidance_context(*resolve_guidance_context(payload))
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        # Allocate gRPC port, start receiver
        grpc_port = self._alloc_port(self.grpc_port_pool)
        run_id = self._get_run_id(payload.task_name)
        task_id = utils.generate_task_id(payload.uuid, payload.task_name, run_id)
        receiver_proc = start_receiver(
            task_id, payload.task_name, run_id, grpc_port,
            f"localhost:{self.controller_port}", config.dashboard_token,
        )
        # await for receiver_proc to start
        await asyncio.sleep(0.1)
        if receiver_proc is None or receiver_proc.poll() is not None:
            self._release_port(self.grpc_port_pool, grpc_port)
            self._release_run_id_reservation(payload.task_name, run_id)
            utils.kill_process(receiver_proc)
            raise HTTPException(status_code=500, detail="Failed to start receiver process")

        task = FuzzerTask(
            task_id=task_id,
            task_name=payload.task_name,
            fuzzer_id=payload.uuid,
            run_id=run_id,
            mode=payload.mode,
            grpc_port=grpc_port,
            receiver_proc=receiver_proc,
            callback_addr="",
            registered_at=time.time(),
            target_func=target_func,
            kallgraph_dir=kallgraph_dir,
            report_path=report_path,
            report_text=report_text,
            target_os=payload.target_os,
            target_arch=payload.target_arch,
            target_revision=payload.target_revision,
            producer_revision=payload.producer_revision,
            descriptions_mode=payload.descriptions_mode,
        )

        # Receiver logs are now written to per-task log file via --log_file
        # capture_subprocess_output is no longer needed

        # Reserve the task_id immediately to prevent concurrent registrations
        # from getting the same run_id (race condition in _get_run_id).
        self.global_tasks[task.task_id] = task
        self._release_run_id_reservation(task.task_name, task.run_id)

        # direct mode: requires host_ip + http_port
        if task.mode == "direct":
            task.callback_addr = f"{payload.host_ip}:{payload.http_port}"

        # isolated mode: allocate tunnel_port + write authorized_keys
        if task.mode == "isolated":
            try:
                task.tunnel_port = self._alloc_port(self.tunnel_port_pool)
                task.callback_addr = f"localhost:{task.tunnel_port}"
                ssh_entry = (
                    f'command="/bin/false",no-agent-forwarding,no-x11-forwarding,no-pty,'
                    f'permitopen="{task.callback_addr}" {payload.public_key.strip()}'
                )
                self._add_ssh_entry_if_not_exists(ssh_entry)
                task.ssh_entry = ssh_entry
                logger.info(f"SSH entry added successfully")
            except Exception as error:
                logger.error(f"Failed to configure isolated task: {error}")
                if task.tunnel_port:
                    self._release_port(self.tunnel_port_pool, task.tunnel_port)
                utils.kill_process(task.receiver_proc)
                self._release_port(self.grpc_port_pool, task.grpc_port)
                self.global_tasks.pop(task.task_id, None)
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to configure isolated task: {error}",
                ) from error

        # Initialize GuidanceEngine for this task
        if task.callback_addr:
            guidance_config = GuidanceConfig(
                fuzzer_callback_addr=task.callback_addr,
            )
            task.guidance_engine = GuidanceEngine(guidance_config)
            logger.info(f"[{task.task_id}] GuidanceEngine initialized (target={task.target_func or 'auto'})")

            # Run PathBasedAnalyzer at registration time if report is available.
            # This provides immediate guidance for cold-start, before any model
            # training occurs. The guidance is sent to the fuzzer right away.
            if task.report_text or task.kallgraph_dir:
                self._start_daemon_thread(self._run_registration_guidance, task)

        # Generate a unique model name for this task
        # Use task_id which is already globally unique (UUID@task_name@count)
        safe_task = task.task_id.replace('@', '_').replace(' ', '_')
        stable_suffix = hashlib.sha256(task.task_id.encode("utf-8")).hexdigest()[:12]
        task.model_name_prefix = f"reach_filter_{safe_task[:80]}_{stable_suffix}"
        self._load_training_state(task)

        logger.info(f"Task {task.task_id} registered successfully with data receiver server at port {task.grpc_port}, model prefix: {task.model_name_prefix} (will be deployed after training)")

        return RegistrationResponse(
            task_id=task.task_id,
            receiver_port=task.grpc_port,
            torchserve_port=config.ts_inference_port,
            tunnel_port=task.tunnel_port,
            model_name="",        # No model deployed yet — on-demand via /model_ready
            model_version=""       # No model deployed yet
        )

    async def unregister(self, task_id: str):
        t = self.global_tasks.get(task_id)
        if not t:
            raise HTTPException(status_code=404, detail="task not found")

        # Wake sleeping guidance workers before waiting for the lifecycle
        # barriers. Guidance takes guidance_lock before lifecycle_lock, so
        # unregistration must preserve the same ordering. This also guarantees
        # that a successful response means an in-flight IG run released its
        # shared physical GPU lease.
        t.guidance_cancel.set()
        guidance_acquired = False
        lifecycle_acquired = False
        try:
            await self._acquire_thread_lock_async(t.guidance_lock)
            guidance_acquired = True
            await self._acquire_thread_lock_async(t.lifecycle_lock)
            lifecycle_acquired = True
            if self.global_tasks.get(task_id) is not t:
                raise HTTPException(status_code=404, detail="task not found")
            t.stopping = True
            self._remove_training_waiter(task_id)
            if not self._stop_task_training(t):
                raise HTTPException(
                    status_code=503,
                    detail="trainer cleanup is incomplete; retry unregister",
                )
            if not self._stop_task_auxiliary_processes(t):
                raise HTTPException(
                    status_code=503,
                    detail="task subprocess cleanup is incomplete; retry unregister",
                )
            models = {
                (t.model_name, t.model_version),
                (t.pending_model_name, t.pending_model_version),
            }
            for model_name, model_version in models:
                if not model_name:
                    continue
                try:
                    self.torchserve_operator.unregister_model(
                        model_name, model_version
                    )
                except Exception as error:
                    logger.warning(
                        f"[{task_id}] Failed to unregister TorchServe model "
                        f"{model_name} v{model_version}: {error}"
                    )

            if self.global_tasks.pop(task_id, None) is not t:
                raise HTTPException(status_code=404, detail="task not found")
            self._remove_ssh_entry(t.ssh_entry)
            self._release_port(self.grpc_port_pool, t.grpc_port)
            if t.tunnel_port:
                self._release_port(self.tunnel_port_pool, t.tunnel_port)
        finally:
            if lifecycle_acquired:
                t.lifecycle_lock.release()
            if guidance_acquired:
                t.guidance_lock.release()

        logger.info(f"Task {task_id} unregistered successfully")
        return {"status": "ok"}

    async def ping_fuzzer(self, task_id: str):
        t = self.global_tasks.get(task_id)
        if not t:
            raise HTTPException(status_code=404, detail="fuzzer not found")

        url = f"http://{t.callback_addr}/"
        route = "TBD"
        url += route
        params = {"action": "ping"}
        try:
            async with httpx.AsyncClient() as client:
                r = await client.get(url, timeout=5, params=params)
            r.raise_for_status()
            return {"mode": t.mode, "response": r.text}
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"ping fuzzer failed: {e}")

    # ====== Routes: task/training/attribution (placeholders for further development) ======
    async def list_tasks(self):
        """List all active tasks"""
        tasks_data = []
        for task in self.global_tasks.values():
            # Manually construct serializable dict (exclude non-serializable fields)
            tasks_data.append({
                "task_id": task.task_id,
                "task_name": task.task_name,
                "run_id": task.run_id,
                "fuzzer_id": task.fuzzer_id,
                "mode": task.mode,
                "callback_addr": task.callback_addr,
                "tunnel_port": task.tunnel_port,
                "grpc_port": task.grpc_port,
                "registered_at": task.registered_at,
                # Process status instead of process objects
                "receiver_status": "running" if (task.receiver_proc and task.receiver_proc.poll() is None) else "stopped",
                "receiver_pid": task.receiver_proc.pid if task.receiver_proc else None,
                "trainer_pid": task.trainer_proc.pid if task.trainer_proc else None,
                "attributor_pid": task.attributor_proc.pid if task.attributor_proc else None,
                # Guidance status
                "target_func": task.target_func,
                "guidance_version": task.guidance_version,
                "stopping": task.stopping,
            })

        return {"list_tasks": tasks_data}

    async def start_trainer(self, task_id: str, batch_start: int, batch_end: int, stage: int):
        """Launch train_v2.py for the given task.

        Args:
            task_id: The task identifier (fuzzer_id@task_name@run_id)
            batch_start: Starting batch index (kept for API compatibility)
            batch_end: Highest committed Receiver batch ID in this snapshot
            stage: Training stage (1=binary, 2=exact waypoint-level)
        """
        t = self.global_tasks.get(task_id)
        if not t:
            self._remove_training_waiter(task_id)
            raise HTTPException(status_code=404, detail="task not found")
        if stage not in (1, 2):
            self._remove_training_waiter(task_id)
            raise HTTPException(status_code=400, detail=f"invalid training stage: {stage}")

        training_id = f"round-{time.time_ns()}"
        # Claim the launch while the task is known to be registered. Without
        # the lifecycle lock, unregister can remove and clean the task between
        # the lookup above and this claim, allowing a detached trainer to start.
        await asyncio.to_thread(t.lifecycle_lock.acquire)
        try:
            if self.global_tasks.get(task_id) is not t:
                self._remove_training_waiter(task_id)
                raise HTTPException(status_code=404, detail="task not found")
            if t.stopping:
                self._remove_training_waiter(task_id)
                raise HTTPException(
                    status_code=409,
                    detail="task is stopping and cannot start training",
                )
            with t.training_lock:
                if t.training_in_progress:
                    pid = t.trainer_proc.pid if t.trainer_proc else None
                    logger.info(f"[{task_id}] Training already in progress, skipping")
                    return {"status": "already_running", "pid": pid}
                if t.pending_model_name:
                    self._remove_training_waiter(task_id)
                    if not all((
                        t.pending_model_version, t.pending_checkpoint,
                        t.pending_manifest, t.pending_torchscript,
                        t.pending_stage, t.pending_batch,
                        t.pending_training_round, t.pending_num_classes,
                        t.pending_seen_training_batches,
                    )):
                        raise HTTPException(
                            status_code=500,
                            detail="incomplete pending deployment state",
                        )
                    t.training_in_progress = True
                    t.active_training_id = training_id
                    pending_stage = t.pending_stage
                    pending_round = t.pending_training_round
                    pending_batch = t.pending_batch
                    self._start_daemon_thread(
                        self._retry_pending_deployment, t, training_id
                    )
                    return {
                        "status": "queued",
                        "stage": pending_stage,
                        "round": pending_round,
                        "batch_end": pending_batch,
                        "deployment_retry": True,
                    }
                # Stages are mandatory and monotonic. A snapshot may already
                # satisfy a later distribution gate, but it must still train
                # each preceding objective before advancing.
                if t.last_successful_stage == 0:
                    effective_stage = 1
                else:
                    effective_stage = min(
                        max(stage, t.last_successful_stage),
                        t.last_successful_stage + 1,
                    )
                if (batch_end > 0 and batch_end <= t.last_trained_batch and
                        effective_stage <= t.last_successful_stage):
                    self._remove_training_waiter(task_id)
                    return {
                        "status": "up_to_date",
                        "stage": effective_stage,
                        "round": t.training_round,
                        "batch_end": t.last_trained_batch,
                    }
                rejected_watermark = t.evaluation_watermarks.get(
                    effective_stage
                )
                if (rejected_watermark is not None and batch_end > 0 and
                        batch_end <= rejected_watermark["batch_end"]):
                    self._remove_training_waiter(task_id)
                    return {
                        "status": "rejected",
                        "stage": effective_stage,
                        "round": t.training_round,
                        "batch_end": rejected_watermark["batch_end"],
                        "reason_codes": list(
                            rejected_watermark.get("reason_codes", [])
                        ),
                    }
                next_round = t.training_round + 1
                last_trained_batch = t.last_trained_batch
                load_path = t.last_successful_checkpoint
                loaded_checkpoint_stage = t.last_successful_stage
                seen_training_batches = list(t.seen_training_batches)
                rejected_batch_boundary = (
                    int(rejected_watermark["batch_end"])
                    if rejected_watermark is not None else 0
                )
                full_retrain = (
                    last_trained_batch == 0 or
                    effective_stage > t.last_successful_stage
                )
                t.training_in_progress = True
                t.active_training_id = training_id
                t.training_cancel.clear()
                t.training_launch_done.clear()
        finally:
            t.lifecycle_lock.release()

        training_gpu = None
        main_port = None
        trainer_log_fh = None
        trainer_proc = None
        launched = False
        retain_training_waiter = False
        try:
            data_dir = os.path.abspath(
                os.path.join(config.data_root, t.task_name, str(t.run_id))
            )
            if not os.path.isdir(data_dir):
                raise HTTPException(
                    status_code=400, detail=f"Data directory not found: {data_dir}"
                )
            batch_indices = list_committed_batch_indices(
                data_dir, batch_end, cancel_event=t.training_cancel
            )
            _check_training_launch_canceled(t.training_cancel)

            import pickle as _pickle
            sample_labels_file = os.path.join(
                data_dir, f"labels_batch_{batch_indices[0]}.pkl"
            )
            with open(sample_labels_file, "rb") as file_handle:
                sample_labels = _pickle.load(file_handle)  # nosec B301 - local receiver data
            _check_training_launch_canceled(t.training_cancel)
            if not sample_labels:
                raise HTTPException(status_code=400, detail="first label batch is empty")
            first_label = next(iter(sample_labels.values()))
            num_classes = len(first_label)
            try:
                effective_stage = next_curriculum_stage(
                    stage, t.last_successful_stage, num_classes
                )
            except ValueError as error:
                raise HTTPException(status_code=409, detail=str(error)) from error
            full_retrain = (
                last_trained_batch == 0 or
                effective_stage > t.last_successful_stage
            )

            try:
                if rejected_batch_boundary:
                    split_source_indices = [
                        index for index in batch_indices
                        if index > rejected_batch_boundary
                    ]
                    if len(split_source_indices) < 2:
                        logger.info(
                            f"Deferring training for task {t.task_id}: snapshot "
                            f"boundary {max(batch_indices)} has only "
                            f"{len(split_source_indices)} new committed batch(es)"
                        )
                        return {
                            "status": "deferred",
                            "stage": effective_stage,
                            "round": next_round,
                            "batch_end": max(batch_indices),
                            "reason": "need at least 2 new committed batches",
                        }
                    split_point = max(1, int(len(split_source_indices) * 0.8))
                    if split_point >= len(split_source_indices):
                        split_point = len(split_source_indices) - 1
                    fresh_train = split_source_indices[:split_point]
                    test_indices = split_source_indices[split_point:]
                    train_indices = [
                        index for index in batch_indices
                        if index <= rejected_batch_boundary
                    ] + fresh_train
                else:
                    train_indices, test_indices = build_training_split(
                        batch_indices,
                        last_trained_batch,
                        full_retrain,
                        seed=f"{task_id}:{next_round}:{effective_stage}",
                    )
            except ValueError as error:
                logger.info(
                    f"Deferring training for task {t.task_id}: {error}"
                )
                return {
                    "status": "deferred",
                    "stage": effective_stage,
                    "round": next_round,
                    "batch_end": max(batch_indices),
                    "reason": str(error),
                }
            test_exclude_indices = sorted(
                set(train_indices).union(seen_training_batches)
            )

            excluded_signatures = load_batch_signatures(
                data_dir, test_exclude_indices,
                cancel_event=t.training_cancel,
            )
            validation_signatures = (
                load_batch_signatures(
                    data_dir, test_indices,
                    cancel_event=t.training_cancel,
                )
                - excluded_signatures
            )
            if not validation_signatures:
                logger.info(
                    f"Deferring training for task {t.task_id}: snapshot "
                    f"boundary {max(batch_indices)} has no signature-disjoint "
                    "validation examples"
                )
                return {
                    "status": "deferred",
                    "stage": effective_stage,
                    "round": next_round,
                    "batch_end": max(batch_indices),
                    "reason": "no signature-disjoint validation examples",
                }
            validation_signature_count = len(validation_signatures)
            validation_signature_sha256 = hashlib.sha256(
                "\n".join(sorted(validation_signatures)).encode("utf-8")
            ).hexdigest()

            train_class_counts = curriculum_split_class_counts(
                data_dir,
                num_classes,
                batch_indices,
                train_indices,
                [],
                effective_stage,
                cancel_event=t.training_cancel,
            )
            validation_class_counts = curriculum_split_class_counts(
                data_dir,
                num_classes,
                batch_indices,
                test_indices,
                test_exclude_indices,
                effective_stage,
                cancel_event=t.training_cancel,
            )
            if (any(count == 0 for count in train_class_counts.values()) or
                    any(count == 0 for count in validation_class_counts.values())):
                logger.info(
                    f"Deferring training for task {t.task_id}: curriculum "
                    f"class coverage is incomplete; train={train_class_counts}, "
                    f"validation={validation_class_counts}"
                )
                return {
                    "status": "deferred",
                    "stage": effective_stage,
                    "round": next_round,
                    "batch_end": max(batch_indices),
                    "reason": "incomplete train/validation curriculum classes",
                    "train_class_counts": train_class_counts,
                    "validation_class_counts": validation_class_counts,
                }

            save_dir = os.path.abspath(
                os.path.join(config.data_root, t.task_name, str(t.run_id), "models")
            )
            os.makedirs(save_dir, exist_ok=True)
            session_name = f"TraceClassifier-v2.0-round-{next_round}-{time.time_ns()}"
            model_dir = os.path.join(save_dir, session_name)

            if load_path and not os.path.isfile(load_path):
                raise HTTPException(
                    status_code=500,
                    detail=f"Previous successful checkpoint is missing: {load_path}",
                )

            queue_position = self._enqueue_training_waiter(task_id)
            if queue_position != 0:
                retain_training_waiter = True
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "Training request is waiting for its FIFO turn at "
                        f"position {queue_position}"
                    ),
                )
            try:
                training_gpu = await self._acquire_training_gpu_async(
                    t.training_cancel
                )
            except BaseException:
                self._remove_training_waiter(task_id)
                raise
            if training_gpu is None:
                if t.training_cancel.is_set():
                    self._remove_training_waiter(task_id)
                    raise _TrainingLaunchCanceled(
                        "training launch was canceled"
                    )
                retain_training_waiter = True
                raise HTTPException(
                    status_code=503,
                    detail="No GPU available for training, retry later",
                )
            self._remove_training_waiter(task_id)
            main_port = self._alloc_port(self.training_port_pool)

            here = os.path.dirname(os.path.abspath(__file__))
            filter_dir = os.path.join(os.path.dirname(here), "filter")
            cmd = [
                "accelerate", "launch", "--num_processes", "1",
                "--gpu_ids", training_gpu,
                "--main_process_port", str(main_port), "train_v2.py",
                "--batch_size", str(config.batch_size),
                "--grad_acc_steps", str(config.grad_acc_steps),
                "--learning_rate", str(config.learning_rate),
                "--weight_decay", str(config.weight_decay),
                "--num_warmup_steps", str(config.num_warmup_steps),
                "--num_classes", str(num_classes),
                "--train_stage", str(effective_stage),
                "--base_model_path", config.base_model,
                "--tokenizer_path", config.tokenizer,
                "--freeze_layers",
                "--data_dir", data_dir,
                "--data_idx", ",".join(map(str, train_indices)),
                "--test_data_idx", ",".join(map(str, test_indices)),
                "--canonical_data_idx", ",".join(map(str, batch_indices)),
                "--test_exclude_data_idx", ",".join(
                    map(str, test_exclude_indices)
                ),
                "--log_dir", save_dir,
                "--session_name", session_name,
                "--token_cache_entries", str(config.token_cache_entries),
                "--disable_wandb",
            ]
            is_first_train = not load_path
            if load_path:
                cmd.extend([
                    "--load_path", load_path,
                    "--loaded_checkpoint_stage", str(loaded_checkpoint_stage),
                ])
            if is_first_train:
                cmd.extend([
                    "--total_steps", str(config.first_train_total_steps),
                    "--test_interval", str(config.first_train_test_interval),
                    "--min_steps", str(config.first_train_min_steps),
                    "--patience", str(config.first_train_patience),
                    "--is_first_train",
                ])
            else:
                cmd.extend([
                    "--total_steps", str(config.continued_train_total_steps),
                    "--test_interval",
                    str(config.continued_train_test_interval),
                    "--min_steps", str(config.continued_train_min_steps),
                    "--patience", str(config.continued_train_patience),
                ])
            cmd.extend(["--assigned_physical_gpu", training_gpu])

            logger.info(
                f"Starting training for task {t.task_id}: round={next_round}, "
                f"requested_stage={stage}, effective_stage={effective_stage}, "
                f"boundary={max(batch_indices)}, train={train_indices}, "
                f"test={test_indices}, load_path={load_path or 'none'}"
            )
            logger.info(f"Training command: {' '.join(cmd)}")

            current_env = os.environ.copy()
            current_env["PYTHONUNBUFFERED"] = "1"
            current_env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            # The trainer uses one process and DataLoader(num_workers=0), so a
            # bounded Rayon pool is safe and avoids serial re-tokenization.
            current_env["TOKENIZERS_PARALLELISM"] = "true"
            current_env["RAYON_NUM_THREADS"] = str(
                config.tokenizer_rayon_threads
            )
            current_env["CUDA_VISIBLE_DEVICES"] = training_gpu

            trainer_log_path = str(task_log_path(t, "trainer"))
            os.makedirs(os.path.dirname(trainer_log_path), exist_ok=True)
            with t.training_lock:
                if (t.active_training_id != training_id or
                        t.training_cancel.is_set()):
                    raise _TrainingLaunchCanceled(
                        "training launch was canceled"
                    )
            trainer_log_fh = open(trainer_log_path, "a", encoding="utf-8")
            trainer_proc = subprocess.Popen(
                cmd,
                cwd=filter_dir,
                stdout=trainer_log_fh,
                stderr=subprocess.STDOUT,
                bufsize=0,
                text=False,
                env=current_env,
                start_new_session=True,
            )
            with t.training_lock:
                if (t.active_training_id != training_id or
                        t.training_cancel.is_set()):
                    raise _TrainingLaunchCanceled(
                        "training launch was canceled"
                    )
                t.trainer_proc = trainer_proc
                t.trainer_log_fh = trainer_log_fh
                t.training_gpu = training_gpu
                t.training_port = main_port
            logger.info(
                f"Started training for task {t.task_id} "
                f"(PID: {trainer_proc.pid}, log: {trainer_log_path})"
            )

            self._start_daemon_thread(
                self._monitor_training_and_deploy,
                t, training_id, trainer_proc, effective_stage, num_classes,
                model_dir, main_port, training_gpu, next_round,
                max(batch_indices),
                validation_signature_count,
                validation_signature_sha256,
            )
            launched = True
            t.training_launch_done.set()
            return {
                "status": "queued",
                "pid": trainer_proc.pid,
                "stage": effective_stage,
                "round": next_round,
                "batch_end": max(batch_indices),
            }
        except _TrainingLaunchCanceled as error:
            logger.info(f"[{t.task_id}] {error}")
            raise HTTPException(status_code=409, detail=str(error)) from error
        except HTTPException:
            raise
        except Exception as error:
            logger.error(f"Failed to start trainer process: {error}")
            raise HTTPException(
                status_code=500, detail=f"Failed to start trainer: {error}"
            ) from error
        finally:
            if not retain_training_waiter:
                self._remove_training_waiter(task_id)
            if not launched:
                try:
                    try:
                        cleanup_confirmed = utils.kill_process(
                            trainer_proc, process_group=True
                        )
                    except Exception as error:
                        cleanup_confirmed = False
                        logger.error(
                            f"[{t.task_id}] Trainer cleanup raised during launch "
                            f"rollback: {error}"
                        )
                    if cleanup_confirmed:
                        with t.training_lock:
                            if t.trainer_proc is trainer_proc:
                                t.trainer_proc = None
                            if t.trainer_log_fh is trainer_log_fh:
                                t.trainer_log_fh = None
                            if t.training_gpu == (training_gpu or ""):
                                t.training_gpu = ""
                            if t.training_port == (main_port or 0):
                                t.training_port = 0
                            if t.active_training_id in ("", training_id):
                                t.training_in_progress = False
                                t.active_training_id = ""
                        if trainer_log_fh is not None:
                            try:
                                trainer_log_fh.close()
                            except Exception as error:
                                logger.error(
                                    f"[{t.task_id}] Failed to close canceled "
                                    f"trainer log: {error}"
                                )
                        if main_port is not None:
                            try:
                                self._release_port(
                                    self.training_port_pool, main_port
                                )
                            except Exception as error:
                                logger.error(
                                    f"[{t.task_id}] Failed to release canceled "
                                    f"training port {main_port}: {error}"
                                )
                        if training_gpu is not None:
                            try:
                                self._release_training_gpu(training_gpu)
                            except Exception as error:
                                logger.error(
                                    f"[{t.task_id}] Failed to release canceled "
                                    f"training GPU {training_gpu}: {error}"
                                )
                    else:
                        with t.training_lock:
                            if t.active_training_id in ("", training_id):
                                t.trainer_proc = trainer_proc
                                t.trainer_log_fh = trainer_log_fh
                                t.training_gpu = training_gpu or ""
                                t.training_port = main_port or 0
                                t.training_in_progress = True
                                t.active_training_id = training_id
                        logger.error(
                            f"[{t.task_id}] Failed to confirm trainer cleanup "
                            "after launch cancellation; retaining GPU and port "
                            "ownership"
                        )
                finally:
                    # Unregistration waits on this barrier. It must open even
                    # when best-effort log/resource cleanup itself raises.
                    t.training_launch_done.set()

    async def start_attributor(self, task_id: str, model_version: int, top_k: int = 10):
        t = self.global_tasks.get(task_id)
        if not t:
            raise HTTPException(status_code=404, detail="task not found")
        raise HTTPException(
            status_code=501,
            detail=(
                "manual attribution is not implemented; attribution runs "
                "through the quality-gated post-deployment guidance pipeline"
            ),
        )

    def _release_training_resources(self, task: FuzzerTask, training_id: str,
                                    training_port: int, training_gpu: str):
        """Release one launch's resources exactly once."""
        log_handle = None
        release_port = False
        release_gpu = False
        with task.training_lock:
            if task.active_training_id != training_id:
                return
            if task.training_port == training_port:
                task.training_port = 0
                release_port = True
            if task.training_gpu == training_gpu:
                task.training_gpu = ""
                release_gpu = True
            log_handle = task.trainer_log_fh
            task.trainer_log_fh = None
        if log_handle is not None:
            try:
                log_handle.close()
            except Exception:
                pass
        if release_gpu:
            self._release_training_gpu(training_gpu)
        if release_port:
            self._release_port(self.training_port_pool, training_port)

    def _stop_task_training(self, task: FuzzerTask):
        """Stop one task and release its trainer resources without double release."""
        wait_for_launch = None
        with task.training_lock:
            if (task.training_in_progress and
                    not task.training_launch_done.is_set()):
                task.training_cancel.set()
                wait_for_launch = task.training_launch_done
        if wait_for_launch is not None and not wait_for_launch.wait(timeout=15):
            logger.error(
                f"[{task.task_id}] Timed out waiting for trainer launch cleanup"
            )
            return False

        cleanup_confirmed = True
        with task.training_lock:
            proc = task.trainer_proc
            if proc is not None:
                returncode = proc.poll()
                if returncode is None or returncode != 0:
                    cleanup_confirmed = utils.kill_process(
                        proc, process_group=True
                    )
                if cleanup_confirmed:
                    task.trainer_proc = None
            if not cleanup_confirmed:
                logger.error(
                    f"[{task.task_id}] Failed to confirm trainer process-group "
                    "termination; retaining GPU and port ownership"
                )
                return False
            gpu = task.training_gpu
            port = task.training_port
            log_handle = task.trainer_log_fh
            task.training_gpu = ""
            task.training_port = 0
            task.trainer_log_fh = None
            task.training_in_progress = False
            task.active_training_id = ""
        if log_handle is not None:
            try:
                log_handle.close()
            except Exception:
                pass
        if gpu:
            self._release_training_gpu(gpu)
        if port:
            self._release_port(self.training_port_pool, port)
        return True

    @staticmethod
    def _stop_task_auxiliary_processes(task: FuzzerTask) -> bool:
        """Best-effort stop all non-trainer children without skipping siblings."""
        cleanup_confirmed = True
        for attribute, role in (
                ("attributor_proc", "attributor"),
                ("receiver_proc", "receiver")):
            proc = getattr(task, attribute)
            if proc is None:
                continue
            try:
                stopped = utils.kill_process(proc)
            except Exception as error:
                stopped = False
                logger.error(
                    f"[{task.task_id}] {role} cleanup raised: {error}"
                )
            if stopped:
                setattr(task, attribute, None)
            else:
                cleanup_confirmed = False
                logger.error(
                    f"[{task.task_id}] Failed to confirm {role} termination"
                )
        return cleanup_confirmed

    @staticmethod
    def _load_training_manifest(model_dir: str, stage: int, num_classes: int):
        manifest_path = Path(model_dir) / "training_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 1:
            raise ValueError("unsupported training manifest schema")
        if manifest.get("curriculum_schema") != CURRICULUM_SCHEMA_VERSION:
            raise ValueError("unsupported training manifest curriculum schema")
        if int(manifest.get("train_stage", 0)) != stage:
            raise ValueError("training manifest stage mismatch")
        if int(manifest.get("num_classes", 0)) != num_classes:
            raise ValueError("training manifest class-count mismatch")
        best_eval_loss = float(manifest.get("best_eval_loss", math.nan))
        if not math.isfinite(best_eval_loss):
            raise ValueError("training manifest best eval loss is not finite")
        for metric_name in ("best_eval_accuracy", "best_eval_weighted_f1"):
            if metric_name in manifest:
                metric_value = float(manifest[metric_name])
                if not math.isfinite(metric_value) or not 0.0 <= metric_value <= 1.0:
                    raise ValueError(f"training manifest {metric_name} is invalid")
        validation_counts = manifest.get("validation_class_counts")
        if validation_counts is not None:
            expected_classes = curriculum_output_classes(num_classes, stage)
            normalized_counts = {
                str(int(class_index)): int(count)
                for class_index, count in validation_counts.items()
            }
            if (set(normalized_counts) != {
                    str(index) for index in range(expected_classes)
                } or any(count < 0 for count in normalized_counts.values())):
                raise ValueError("training manifest validation class counts are invalid")
        checkpoint = Path(manifest["best_checkpoint"]).resolve()
        expected_parent = Path(model_dir).resolve()
        if checkpoint.parent != expected_parent or not checkpoint.is_file():
            raise ValueError("training manifest checkpoint is missing or outside run directory")
        digest_builder = hashlib.sha256()
        with checkpoint.open("rb") as file_handle:
            for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
                digest_builder.update(chunk)
        digest = digest_builder.hexdigest()
        if digest != manifest.get("checkpoint_sha256"):
            raise ValueError("training manifest checkpoint hash mismatch")
        return manifest_path, manifest, checkpoint

    @staticmethod
    def _promotion_thresholds() -> PromotionThresholds:
        return PromotionThresholds(
            majority_margin=config.promotion_majority_margin,
            stage1_min_support=config.promotion_stage1_min_support,
            stage1_min_macro_f1=config.promotion_stage1_min_macro_f1,
            stage1_min_recall=config.promotion_stage1_min_recall,
            stage2_min_support=config.promotion_stage2_min_support,
            stage2_min_macro_f1=config.promotion_stage2_min_macro_f1,
            stage2_min_unreachable_recall=(
                config.promotion_stage2_min_unreachable_recall
            ),
            stage2_min_reached_recall=(
                config.promotion_stage2_min_reached_recall
            ),
            stage2_min_final_recall=config.promotion_stage2_min_final_recall,
        )

    @staticmethod
    def _write_promotion_decision(model_dir: str, decision: Dict[str, Any]) -> Path:
        """Atomically retain the quality decision beside the model manifest."""
        decision_path = Path(model_dir) / "promotion_decision.json"
        temp_path = decision_path.with_suffix(".json.tmp")
        with temp_path.open("w", encoding="utf-8") as file_handle:
            json.dump(decision, file_handle, indent=2, sort_keys=True)
            file_handle.write("\n")
            file_handle.flush()
            os.fsync(file_handle.fileno())
        os.replace(temp_path, decision_path)
        directory_fd = os.open(decision_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return decision_path

    def _record_rejected_evaluation(
            self, task: FuzzerTask, training_id: str, stage: int,
            batch_boundary: int, manifest_path: Optional[Path],
            manifest: Dict[str, Any],
            decision: Dict[str, Any]) -> None:
        """Persist a terminal rejected snapshot without advancing active state."""
        watermark = {
            "batch_end": int(batch_boundary),
            "disposition": "rejected",
            "reason_codes": list(decision["reason_codes"]),
            "manifest_path": str(manifest_path) if manifest_path else "",
            "candidate_checkpoint_sha256": str(
                manifest.get("checkpoint_sha256", "")
            ),
            "evaluated_at": datetime.now().astimezone().isoformat(),
        }
        with task.training_lock:
            if task.active_training_id != training_id:
                raise RuntimeError("stale training cannot record an evaluation")
            previous = task.evaluation_watermarks.get(stage)
            task.evaluation_watermarks[stage] = watermark
        try:
            self._persist_training_state(task)
        except Exception:
            with task.training_lock:
                if previous is None:
                    task.evaluation_watermarks.pop(stage, None)
                else:
                    task.evaluation_watermarks[stage] = previous
            raise

    def _record_invalid_evaluation(
            self, task: FuzzerTask, training_id: str, stage: int,
            batch_boundary: int, model_dir: str, error: Exception,
            manifest_path: Optional[Path] = None,
            manifest: Optional[Dict[str, Any]] = None) -> Path:
        """Make deterministic artifact-validation failures terminal."""
        decision = {
            "schema_version": 1,
            "accepted": False,
            "reason_codes": ["invalid_evidence"],
            "stage": stage,
            "task_id": task.task_id,
            "training_id": training_id,
            "batch_end": batch_boundary,
            "manifest_path": str(manifest_path) if manifest_path else "",
            "evidence_error_type": type(error).__name__,
            "evidence_error": str(error)[:1000],
            "decided_at": datetime.now().astimezone().isoformat(),
        }
        decision_path = self._write_promotion_decision(model_dir, decision)
        self._record_rejected_evaluation(
            task,
            training_id,
            stage,
            batch_boundary,
            manifest_path,
            manifest or {},
            decision,
        )
        return decision_path

    @staticmethod
    def _validate_promotion_provenance(
            task: FuzzerTask, manifest: Dict[str, Any]) -> None:
        """Bind baseline metrics to the exact active checkpoint they evaluate."""
        loaded_checkpoint = manifest.get("loaded_checkpoint")
        if not loaded_checkpoint:
            if int(manifest.get("loaded_checkpoint_stage", 0)) != 0:
                raise ValueError("manifest has a stage without a loaded checkpoint")
            return
        if not task.last_successful_checkpoint:
            raise ValueError("manifest loaded a checkpoint but task has no active model")
        loaded_path = Path(loaded_checkpoint).resolve()
        active_path = Path(task.last_successful_checkpoint).resolve()
        if loaded_path != active_path:
            raise ValueError("manifest baseline checkpoint is not the active checkpoint")
        if int(manifest.get("loaded_checkpoint_stage", 0)) != task.last_successful_stage:
            raise ValueError("manifest baseline checkpoint stage mismatch")
        expected_digest = str(manifest.get("loaded_checkpoint_sha256", ""))
        digest = hashlib.sha256()
        with active_path.open("rb") as file_handle:
            for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected_digest:
            raise ValueError("manifest baseline checkpoint hash mismatch")

    def _finish_training(self, task: FuzzerTask, training_id: str):
        with task.training_lock:
            if task.active_training_id != training_id:
                return
            task.training_in_progress = False
            task.active_training_id = ""
            task.trainer_proc = None

    def _commit_pending_deployment(self, task: FuzzerTask, training_id: str):
        """Atomically make a fuzzer-acknowledged candidate the active watermark."""
        with task.training_lock:
            if (task.active_training_id != training_id or
                    not task.pending_model_name):
                return None
            rollback = {
                field_name: getattr(task, field_name)
                for field_name in (
                    "training_round", "last_trained_batch",
                    "last_successful_checkpoint", "last_successful_stage",
                    "last_training_manifest", "seen_training_batches",
                    "model_name", "model_version",
                    "deployment_version", "pending_model_name",
                    "pending_model_version", "pending_checkpoint",
                    "pending_manifest", "pending_torchscript", "pending_stage",
                    "pending_batch", "pending_training_round",
                    "pending_num_classes", "pending_seen_training_batches",
                    "evaluation_watermarks",
                )
            }
            rollback["evaluation_watermarks"] = dict(
                task.evaluation_watermarks
            )
            old_model = (
                task.model_name,
                task.model_version,
                task.last_training_manifest,
                task.last_successful_checkpoint,
            )
            task.training_round = task.pending_training_round
            task.last_trained_batch = task.pending_batch
            task.last_successful_checkpoint = task.pending_checkpoint
            task.last_successful_stage = task.pending_stage
            task.last_training_manifest = task.pending_manifest
            task.seen_training_batches = list(task.pending_seen_training_batches)
            task.model_name = task.pending_model_name
            task.model_version = task.pending_model_version
            task.deployment_version = int(task.pending_model_version)
            task.evaluation_watermarks.pop(task.pending_stage, None)
            task.pending_model_name = ""
            task.pending_model_version = ""
            task.pending_checkpoint = ""
            task.pending_manifest = ""
            task.pending_torchscript = ""
            task.pending_stage = 0
            task.pending_batch = 0
            task.pending_training_round = 0
            task.pending_num_classes = 0
            task.pending_seen_training_batches = []
        try:
            self._persist_training_state(task)
        except Exception:
            with task.training_lock:
                for field_name, value in rollback.items():
                    setattr(task, field_name, value)
            raise
        return old_model

    def _remove_superseded_model(self, task: FuzzerTask, old_model):
        old_name, old_version, old_manifest, old_checkpoint = (
            old_model or ("", "", "", "")
        )
        if not old_name or old_name == task.model_name:
            return
        try:
            self.torchserve_operator.unregister_model(old_name, old_version)
        except Exception as error:
            logger.warning(
                f"[{task.task_id}] Failed to remove superseded model "
                f"{old_name} v{old_version}: {error}"
            )
            return

        # TorchServe no longer owns the old archive after a successful
        # unregister. Keep manifests and scalar logs for auditability, but
        # remove superseded model binaries that otherwise grow by roughly one
        # gigabyte per online round.
        if re.fullmatch(r"[A-Za-z0-9_.-]+", old_name):
            model_store = Path(self.torchserve_operator.model_dir).resolve()
            mar_path = model_store / f"{old_name}.mar"
            try:
                if (mar_path.parent == model_store and mar_path.is_file() and
                        not mar_path.is_symlink()):
                    mar_path.unlink()
            except OSError as error:
                logger.warning(
                    f"[{task.task_id}] Failed to remove superseded MAR "
                    f"{mar_path}: {error}"
                )
        self._remove_superseded_run_binaries(
            task, old_manifest, old_checkpoint
        )

    def _model_runs_root(self, task: FuzzerTask) -> Path:
        data_root = Path(config.data_root).expanduser().resolve()
        task_root = (
            data_root / task.task_name / str(task.run_id)
        ).resolve()
        if task_root == data_root or not task_root.is_relative_to(data_root):
            raise ValueError("task model root escapes configured data root")
        return task_root / "models"

    def _remove_superseded_run_binaries(
            self, task: FuzzerTask, manifest_path: str,
            checkpoint_path: str):
        """Remove old checkpoint/TorchScript binaries within one task root."""
        if not manifest_path or not checkpoint_path:
            return
        try:
            runs_root = self._model_runs_root(task)
            manifest_source = Path(manifest_path)
            checkpoint_source = Path(checkpoint_path)
            manifest = manifest_source.resolve()
            run_dir = manifest.parent
            checkpoint = checkpoint_source.resolve()
            if (not run_dir.is_relative_to(runs_root) or
                    checkpoint.parent != run_dir or
                    not manifest_source.is_file() or
                    manifest_source.is_symlink() or
                    not checkpoint_source.is_file() or
                    checkpoint_source.is_symlink()):
                logger.warning(
                    f"[{task.task_id}] Refusing superseded artifact cleanup "
                    "outside the task model root"
                )
                return
            manifest_data = json.loads(
                manifest_source.read_text(encoding="utf-8")
            )
            recorded_checkpoint = Path(
                manifest_data["best_checkpoint"]
            ).resolve()
            if recorded_checkpoint != checkpoint:
                logger.warning(
                    f"[{task.task_id}] Refusing cleanup for a checkpoint "
                    "that does not match its training manifest"
                )
                return
            for artifact in run_dir.glob("*.pt"):
                if artifact.is_file() and not artifact.is_symlink():
                    artifact.unlink()
        except (
            OSError, RuntimeError, KeyError, TypeError, ValueError,
            json.JSONDecodeError,
        ) as error:
            logger.warning(
                f"[{task.task_id}] Failed to clean superseded model binaries: "
                f"{error}"
            )

    def _prune_unselected_checkpoints(
            self, task: FuzzerTask, model_dir: str,
            best_checkpoint: str):
        """Keep only the manifest-selected checkpoint in an active run."""
        try:
            runs_root = self._model_runs_root(task)
            run_dir = Path(model_dir).resolve()
            selected = Path(best_checkpoint).resolve()
            if (not run_dir.is_relative_to(runs_root) or
                    selected.parent != run_dir or not selected.is_file()):
                logger.warning(
                    f"[{task.task_id}] Refusing checkpoint pruning outside "
                    "the task model root"
                )
                return
            for checkpoint in run_dir.glob("step-*.pt"):
                if (checkpoint.resolve() != selected and
                        checkpoint.is_file() and
                        not checkpoint.is_symlink()):
                    checkpoint.unlink()
        except (OSError, RuntimeError, ValueError) as error:
            logger.warning(
                f"[{task.task_id}] Failed to prune unselected checkpoints: "
                f"{error}"
            )

    def _retry_pending_deployment(self, task: FuzzerTask, training_id: str):
        """Retry acknowledgement of a deployed candidate without retraining."""
        guidance_args = None
        try:
            with task.lifecycle_lock:
                if self.global_tasks.get(task.task_id) is not task:
                    return
                with task.training_lock:
                    if (task.active_training_id != training_id or
                            not task.pending_model_name):
                        return
                    candidate_name = task.pending_model_name
                    candidate_version = task.pending_model_version
                    pending_checkpoint = task.pending_checkpoint
                    pending_manifest = task.pending_manifest
                    pending_stage = task.pending_stage
                    pending_num_classes = task.pending_num_classes
                try:
                    self._notify_fuzzer_model_ready(
                        task, candidate_name, candidate_version
                    )
                except Exception as error:
                    logger.error(
                        f"[{task.task_id}] Pending deployment remains unacknowledged: "
                        f"{error}"
                    )
                    return
                try:
                    old_model = self._commit_pending_deployment(task, training_id)
                except Exception as error:
                    logger.error(
                        f"[{task.task_id}] Failed to persist active deployment: {error}"
                    )
                    return
                if old_model is None:
                    return
                self._remove_superseded_model(task, old_model)
                self._prune_unselected_checkpoints(
                    task, str(Path(pending_manifest).parent),
                    pending_checkpoint,
                )
                logger.info(
                    f"[{task.task_id}] Pending deployment {candidate_name} "
                    f"v{candidate_version} committed"
                )
                guidance_args = (
                    pending_stage,
                    pending_num_classes,
                    str(Path(pending_manifest).parent),
                    pending_checkpoint,
                )

            if (guidance_args is not None and
                    self.global_tasks.get(task.task_id) is task and
                    task.guidance_engine is not None):
                try:
                    self._run_guidance_pipeline(task, *guidance_args)
                except Exception as error:
                    logger.error(
                        f"[{task.task_id}] Guidance pipeline failed after "
                        f"pending deployment recovery: {error}"
                    )
        finally:
            self._finish_training(task, training_id)

    def _monitor_training_and_deploy(
            self, task: FuzzerTask, training_id: str, proc: subprocess.Popen,
            stage: int, num_classes: int, model_dir: str, training_port: int,
            training_gpu: str, training_round: int, batch_boundary: int,
            expected_validation_signature_count: Optional[int] = None,
            expected_validation_signature_sha256: Optional[str] = None):
        """Deploy the manifest-selected model and commit only after fuzzer ACK."""
        logger.info(f"[{task.task_id}] Monitoring training process (PID: {proc.pid})...")
        cleanup_confirmed = True
        try:
            proc.wait()
            returncode = proc.returncode
            # Serialize process-group retirement with unregister. A successful
            # launcher has already waited for its worker; an abnormal launcher
            # may have left workers behind and must clean the group before the
            # GPU slot is advertised as available.
            with task.training_lock:
                if task.trainer_proc is proc:
                    if returncode != 0:
                        cleanup_confirmed = utils.kill_process(
                            proc, process_group=True
                        )
                    if cleanup_confirmed:
                        task.trainer_proc = None
            if not cleanup_confirmed:
                logger.error(
                    f"[{task.task_id}] Failed to confirm cleanup after trainer "
                    f"exit {returncode}; retaining GPU and port ownership"
                )
                return
            if self.global_tasks.get(task.task_id) is not task:
                return
            with task.training_lock:
                if (task.active_training_id != training_id or
                        task.stopping):
                    return
            logger.info(f"[{task.task_id}] Training process exited with code {returncode}")
            if returncode != 0:
                logger.error(f"[{task.task_id}] Training failed (exit code {returncode})")
                return

            try:
                manifest_path, manifest, best_checkpoint = self._load_training_manifest(
                    model_dir, stage, num_classes
                )
            except Exception as error:
                logger.error(f"[{task.task_id}] Invalid training manifest: {error}")
                try:
                    self._record_invalid_evaluation(
                        task,
                        training_id,
                        stage,
                        batch_boundary,
                        model_dir,
                        error,
                        manifest_path=Path(model_dir) / "training_manifest.json",
                    )
                except Exception as persist_error:
                    logger.error(
                        f"[{task.task_id}] Failed to persist invalid model "
                        f"evidence: {persist_error}"
                    )
                return

            try:
                if ((expected_validation_signature_count is None) !=
                        (expected_validation_signature_sha256 is None)):
                    raise ValueError(
                        "incomplete expected validation fingerprint"
                    )
                if expected_validation_signature_count is not None:
                    if (int(manifest.get(
                            "validation_signature_count", 0
                        )) != expected_validation_signature_count or
                            str(manifest.get(
                                "validation_signature_sha256", ""
                            )) != expected_validation_signature_sha256):
                        raise ValueError(
                            "training manifest validation fingerprint "
                            "does not match the Controller split"
                        )
                self._validate_promotion_provenance(task, manifest)
                promotion_decision = decide_model_promotion(
                    manifest,
                    stage,
                    num_classes,
                    thresholds=self._promotion_thresholds(),
                )
                promotion_decision.update({
                    "task_id": task.task_id,
                    "training_id": training_id,
                    "batch_end": batch_boundary,
                    "manifest_path": str(manifest_path),
                    "candidate_checkpoint": str(best_checkpoint),
                    "candidate_checkpoint_sha256": manifest[
                        "checkpoint_sha256"
                    ],
                    "decided_at": datetime.now().astimezone().isoformat(),
                })
                decision_path = self._write_promotion_decision(
                    model_dir, promotion_decision
                )
            except Exception as error:
                logger.error(
                    f"[{task.task_id}] Invalid model-promotion evidence: {error}"
                )
                try:
                    self._record_invalid_evaluation(
                        task,
                        training_id,
                        stage,
                        batch_boundary,
                        model_dir,
                        error,
                        manifest_path=manifest_path,
                        manifest=manifest,
                    )
                except Exception as persist_error:
                    logger.error(
                        f"[{task.task_id}] Failed to persist invalid model "
                        f"evidence: {persist_error}"
                    )
                return

            if not promotion_decision["accepted"]:
                try:
                    self._record_rejected_evaluation(
                        task,
                        training_id,
                        stage,
                        batch_boundary,
                        manifest_path,
                        manifest,
                        promotion_decision,
                    )
                except Exception as error:
                    logger.error(
                        f"[{task.task_id}] Failed to persist rejected model "
                        f"evaluation: {error}"
                    )
                    return
                self._prune_unselected_checkpoints(
                    task, model_dir, str(best_checkpoint)
                )
                logger.warning(
                    f"[{task.task_id}] Rejected model candidate before export: "
                    f"reasons={promotion_decision['reason_codes']}, "
                    f"decision={decision_path}"
                )
                return

            with task.lifecycle_lock:
                if self.global_tasks.get(task.task_id) is not task:
                    return
                with task.training_lock:
                    if task.active_training_id != training_id:
                        return
                    next_deployment = task.deployment_version + 1

                model_prefix = task.model_name_prefix or (
                    f"reach_filter_{hashlib.sha256(task.task_id.encode()).hexdigest()[:12]}"
                )
                candidate_name = f"{model_prefix}_r{next_deployment}"
                candidate_version = str(next_deployment)
                candidate_registered = False
                try:
                    try:
                        torchscript_path = self._export_torchscript(
                            task, best_checkpoint, num_classes, stage,
                            candidate_name, training_gpu,
                        )
                    finally:
                        # Export loads and traces the model on the same GPU as
                        # the trainer. Keep the admission slot until tracing is
                        # complete, then release it before TorchServe startup
                        # and the post-deployment guidance pipeline.
                        self._release_training_resources(
                            task, training_id, training_port, training_gpu
                        )
                    candidate_registered = True
                    self._deploy_to_torchserve(
                        task, candidate_name, candidate_version,
                        torchscript_path, num_classes, stage,
                    )
                    with task.training_lock:
                        if task.active_training_id != training_id:
                            raise RuntimeError("deployment was canceled")
                        task.pending_model_name = candidate_name
                        task.pending_model_version = candidate_version
                        task.pending_checkpoint = str(best_checkpoint)
                        task.pending_manifest = str(manifest_path)
                        task.pending_torchscript = str(torchscript_path)
                        task.pending_stage = stage
                        task.pending_batch = batch_boundary
                        task.pending_training_round = training_round
                        task.pending_num_classes = num_classes
                        task.pending_seen_training_batches = sorted(set(map(
                            int, manifest.get(
                                "seen_train_batch_indices",
                                manifest.get("test_exclude_batch_indices", []),
                            )
                        )))
                    self._persist_training_state(task)
                except Exception as error:
                    logger.error(
                        f"[{task.task_id}] Model deployment transaction failed: {error}"
                    )
                    if candidate_registered:
                        with task.training_lock:
                            if task.pending_model_name == candidate_name:
                                task.pending_model_name = ""
                                task.pending_model_version = ""
                                task.pending_checkpoint = ""
                                task.pending_manifest = ""
                                task.pending_torchscript = ""
                                task.pending_stage = 0
                                task.pending_batch = 0
                                task.pending_training_round = 0
                                task.pending_num_classes = 0
                                task.pending_seen_training_batches = []
                        try:
                            self.torchserve_operator.unregister_model(
                                candidate_name, candidate_version
                            )
                        except Exception:
                            pass
                    return

                try:
                    self._notify_fuzzer_model_ready(
                        task, candidate_name, candidate_version
                    )
                except Exception as error:
                    # Do not unregister: an HTTP response may have been lost
                    # after the fuzzer activated the candidate. The durable
                    # pending state lets the Receiver retry acknowledgement.
                    logger.error(
                        f"[{task.task_id}] Candidate remains pending: {error}"
                    )
                    return

                try:
                    old_model = self._commit_pending_deployment(task, training_id)
                except Exception as error:
                    logger.error(
                        f"[{task.task_id}] Failed to persist active deployment: {error}"
                    )
                    return
                if old_model is None:
                    return
                self._remove_superseded_model(task, old_model)
                self._prune_unselected_checkpoints(
                    task, model_dir, str(best_checkpoint)
                )
                logger.info(
                    f"[{task.task_id}] Committed training round {training_round}: "
                    f"best_step={manifest['best_step']}, "
                    f"eval_loss={manifest['best_eval_loss']}, "
                    f"batch_end={batch_boundary}, model={candidate_name}"
                )

            if task.guidance_engine is not None:
                try:
                    self._run_guidance_pipeline(
                        task, stage, num_classes, model_dir, str(best_checkpoint)
                    )
                except Exception as error:
                    logger.error(f"[{task.task_id}] Guidance pipeline failed: {error}")
        finally:
            if cleanup_confirmed:
                self._release_training_resources(
                    task, training_id, training_port, training_gpu
                )
                self._finish_training(task, training_id)

    # Module-level lock for sys.modules manipulation (not thread-safe)
    _export_lock = threading.Lock()

    def _export_torchscript(self, task: FuzzerTask, ckpt_path: Path,
                            num_classes: int, stage: int,
                            artifact_name: str, training_gpu: str) -> str:
        """Export trained model checkpoint to TorchScript format."""
        import importlib
        import sys
        import torch

        # Add filter directory to path and handle module name collision.
        # brain/utils.py is cached as 'utils' in sys.modules, which shadows
        # filter/utils.py (needed by model_v2.py). Temporarily swap it out.
        here = os.path.dirname(os.path.abspath(__file__))
        filter_dir = os.path.join(os.path.dirname(here), "filter")

        with Controller._export_lock:
            if filter_dir not in sys.path:
                sys.path.insert(0, filter_dir)

            # Save and replace the cached 'utils' module so filter/utils.py loads correctly
            brain_utils = sys.modules.pop("utils", None)
            # Also clear any cached model_v2 / config from filter to force reimport
            for mod_name in ["model_v2", "config", "utils"]:
                sys.modules.pop(mod_name, None)

            try:
                from model_v2 import (
                    TraceClassifierServingWrapper,
                    TraceClassifierV2,
                )
            finally:
                # Restore brain/utils.py after importing model_v2
                if brain_utils is not None:
                    sys.modules["utils"] = brain_utils

        logger.info(f"[{task.task_id}] Exporting TorchScript from {ckpt_path}")

        # Load model
        model = TraceClassifierV2(config.base_model, num_classes, stage=stage)
        state_dict = self._load_export_state_dict(torch, ckpt_path)
        try:
            model.load_state_dict(state_dict)
        finally:
            del state_dict
        model.eval()
        serving_model = TraceClassifierServingWrapper(model, stage=stage)
        serving_model.eval()

        device_index = int(training_gpu)
        if device_index >= torch.cuda.device_count():
            raise RuntimeError(
                f"assigned export GPU {device_index} is unavailable"
            )
        device = torch.device(f"cuda:{device_index}")
        input_ids = None
        attention_mask = None
        scripted_model = None
        try:
            serving_model.to(device)

            # Create example inputs for tracing
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)
            tokenizer.pad_token = tokenizer.eos_token

            # Use dummy inputs for tracing
            dummy_texts = ["test input"] * 4
            tokenized = tokenizer(
                dummy_texts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=1024,
            )
            input_ids = tokenized["input_ids"].to(device)
            attention_mask = tokenized["attention_mask"].to(device)

            # Trace model
            with torch.no_grad():
                scripted_model = torch.jit.trace(
                    serving_model, (input_ids, attention_mask)
                )

            # Save TorchScript model
            output_path = os.path.join(
                str(ckpt_path.parent), f"{artifact_name}_scripted.pt"
            )
            scripted_model.save(output_path)
            logger.info(
                f"[{task.task_id}] TorchScript model saved to {output_path}"
            )
            return output_path
        finally:
            # The Controller is long-lived. Drop export-only tensors and
            # return cached blocks so later admission probes observe the real
            # post-export free memory rather than this process's cache.
            del scripted_model
            del input_ids
            del attention_mask
            del serving_model
            del model
            try:
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
            except Exception as error:
                logger.warning(
                    "[%s] Failed to release export CUDA cache on %s: %s",
                    task.task_id,
                    device,
                    error,
                )

    def _deploy_to_torchserve(self, task: FuzzerTask, model_name: str,
                              model_version: str, torchscript_path: str,
                              num_classes: int, stage: int):
        """Pack .mar and register model on TorchServe."""
        # Create index_to_name.json
        index2name_path = self.torchserve_operator.create_index2name(
            num_classes,
            os.path.join(self.torchserve_operator.model_dir, model_name),
            stage=stage,
        )

        # Pack .mar file (force=True to allow re-deployment of retrained models)
        self.torchserve_operator.pack_model(
            model_name=model_name,
            index2name_path=index2name_path,
            torch_script_path=torchscript_path,
            handler_path=config.ts_handler_path,
            version=model_version,
            force=True
        )

        # Register model on TorchServe
        self.torchserve_operator.register_model(
            model_name=model_name,
            init_worker=1,
            batch_size=config.ts_serving_batch_size,
            max_batch_delay=config.ts_max_batch_delay_ms,
            sync=True
        )

        # Verify workers started, fallback to scale_worker if needed
        try:
            model_info = self.torchserve_operator.get_model_info(model_name)
            if not model_info or not model_info.get('workers'):
                logger.warning(f"[{task.task_id}] Workers=0 after register, falling back to scale_worker")
                self.torchserve_operator.scale_worker(
                    model_name=model_name, min_worker=1, sync=True
                )
                model_info = self.torchserve_operator.get_model_info(model_name)
            if not model_info or not model_info.get('workers'):
                raise RuntimeError(f"TorchServe model {model_name} has no workers")
        except Exception as e:
            raise RuntimeError(f"failed to verify TorchServe workers: {e}") from e

        logger.info(f"[{task.task_id}] Model deployed to TorchServe: {model_name} v{model_version}")
        log_mgr.log_model_version(task.task_id, model_name, model_version, "deploy", True)

    def _notify_fuzzer_model_ready(self, task: FuzzerTask, model_name: str,
                                   model_version: str):
        """Activate a model and reconcile a possibly lost callback response."""
        if not task.callback_addr:
            logger.warning(f"[{task.task_id}] No callback address, skipping notification")
            return

        url = f"http://{task.callback_addr}/model_ready"
        payload = {
            "model_name": model_name,
            "model_version": model_version,
        }

        logger.info(f"[{task.task_id}] Notifying fuzzer at {url}: {payload}")

        import requests
        last_error = None
        for attempt in range(5):
            try:
                resp = requests.post(url, json=payload, timeout=10)
                resp.raise_for_status()
                acknowledgement = resp.json()
                if (acknowledgement.get("status") != "active" or
                        acknowledgement.get("model_name") != model_name or
                        str(acknowledgement.get("model_version")) != model_version):
                    raise RuntimeError(
                        f"invalid model activation acknowledgement: {acknowledgement}"
                    )
                logger.info(
                    f"[{task.task_id}] Fuzzer activated model: {acknowledgement}"
                )
                return
            except Exception as error:
                last_error = error
                active_name, active_version = self._query_fuzzer_model(task)
                if active_name == model_name and active_version == model_version:
                    logger.info(
                        f"[{task.task_id}] Reconciled active model after lost callback response"
                    )
                    return
                if attempt < 4:
                    time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(f"fuzzer did not acknowledge model {model_name}: {last_error}")

    @staticmethod
    def _query_fuzzer_model(task: FuzzerTask):
        if not task.callback_addr:
            return "", ""
        import requests
        try:
            response = requests.get(
                f"http://{task.callback_addr}/model_status", timeout=5
            )
            response.raise_for_status()
            payload = response.json()
            return (
                str(payload.get("model_name", "")),
                str(payload.get("model_version", "")),
            )
        except Exception:
            return "", ""

    # ====== Guidance Pipeline ======

    def _run_registration_guidance(self, task: FuzzerTask):
        """Run configured static analysis at registration for cold-start guidance.

        This runs in a background thread so it doesn't block registration.
        It prefers a KallGraph target query and falls back to crash-report path
        analysis before sending guidance to the fuzzer.
        """
        if (not task.report_text and not task.kallgraph_dir) or not task.guidance_engine:
            return

        # Wait briefly for the fuzzer's HTTP server to be ready, but wake
        # immediately when unregistration cancels this task.
        if task.guidance_cancel.wait(timeout=3):
            return

        with task.guidance_lock:
            if not self._task_accepts_guidance(task):
                return
            engine = task.guidance_engine
            logger.info(
                f"[{task.task_id}] Running registration-time static guidance..."
            )
            self._run_static_analysis(task, engine)
            if not self._task_accepts_guidance(task):
                return

            if task.static_analysis_done:
                guidance = engine.compute_guidance()
                send_ok = self._send_guidance_if_active(
                    task, engine, guidance
                )
                if send_ok is None:
                    return
                if send_ok:
                    logger.info(
                        f"[{task.task_id}] Registration guidance sent: "
                        f"{len(guidance['syscall_weights'])} weights "
                        f"v{task.guidance_version}"
                    )
                else:
                    logger.warning(
                        f"[{task.task_id}] Failed to send registration guidance"
                    )

                # Log to guidance.log
                log_mgr.log_guidance_result(
                    task_id=task.task_id,
                    version=engine.version,
                    static_weights=engine.get_static_weights(),
                    attribution_weights=engine.get_attribution_weights(),
                    merged_weights=guidance.get("syscall_weights", {}),
                    templates=guidance.get("mutation_templates", []),
                    send_success=send_ok
                )
            else:
                logger.warning(
                    f"[{task.task_id}] PathBasedAnalyzer found no results"
                )

    def _get_data_dir(self, task: FuzzerTask) -> str:
        """Get the data directory for a task."""
        data_dir = os.path.join(config.data_root, task.task_name, str(task.run_id))
        return os.path.abspath(data_dir)

    def _load_training_data(self, task: FuzzerTask):
        """Load programs and labels from batch PKL files.

        Note: Uses pickle for ML training data (standard format for numpy/torch datasets).
        Only loads from task-specific data directories created by the receiver.
        """
        import pickle  # nosec B403 — ML training data in PKL format is standard practice

        data_dir = self._get_data_dir(task)
        canonical = {}
        batch_indices = list_committed_batch_indices(data_dir)
        for idx in batch_indices:
            prog_file = os.path.join(data_dir, f"progs_batch_{idx}.pkl")
            label_file = os.path.join(data_dir, f"labels_batch_{idx}.pkl")
            try:
                with open(prog_file, 'rb') as f:
                    progs = pickle.load(f)  # nosec B301 — trusted internal data
                with open(label_file, 'rb') as f:
                    labs = pickle.load(f)  # nosec B301 — trusted internal data
                for sig in labs:
                    if sig in progs:
                        label = list(labs[sig])
                        selected_class = one_hot_label_class(label, len(label))
                        if selected_class is None:
                            raise ValueError(f"invalid one-hot label for {sig}")
                        previous = canonical.get(sig)
                        if previous is None:
                            canonical[sig] = (progs[sig], label, selected_class)
                        elif previous[0] != progs[sig]:
                            raise ValueError(f"program hash collision for {sig}")
                        elif selected_class > previous[2]:
                            canonical[sig] = (progs[sig], label, selected_class)
            except Exception as e:
                logger.warning(f"[{task.task_id}] Failed to load batch {idx}: {e}")

        programs = [record[0] for record in canonical.values()]
        labels = [record[1] for record in canonical.values()]
        logger.info(
            f"[{task.task_id}] Loaded {len(programs)} canonical program/label rows"
        )
        return programs, labels

    def _count_positive_samples(self, labels: list, num_classes: int) -> int:
        """Count valid one-hot labels whose selected class is not Unreachable."""
        return sum(
            1 for label in labels
            if is_positive_one_hot(label, num_classes=num_classes)
        )

    @staticmethod
    def _count_exact_class_samples(
            labels: list, num_classes: int, target_class: int) -> int:
        """Count canonical rows assigned to one exact curriculum class."""
        if not 0 <= target_class < num_classes:
            raise ValueError("target class is outside the label schema")
        return sum(
            one_hot_label_class(label, num_classes) == target_class
            for label in labels
        )

    @staticmethod
    def _model_quality_allows_attribution(
            save_dir: str, stage: int, num_classes: int) -> bool:
        """Fail closed unless the selected checkpoint has reliable metrics."""
        try:
            manifest = json.loads(
                (Path(save_dir) / "training_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            accuracy = float(manifest["best_eval_accuracy"])
            weighted_f1 = float(manifest["best_eval_weighted_f1"])
            validation_counts = {
                int(class_index): int(count)
                for class_index, count in manifest[
                    "validation_class_counts"
                ].items()
            }
            expected_classes = curriculum_output_classes(num_classes, stage)
        except (
            OSError, AttributeError, KeyError, TypeError, ValueError,
            json.JSONDecodeError,
        ):
            return False
        return (
            math.isfinite(accuracy)
            and math.isfinite(weighted_f1)
            and config.attribution_min_accuracy <= accuracy <= 1.0
            and 0.0 <= weighted_f1 <= 1.0
            and set(validation_counts) == set(range(expected_classes))
            and all(count > 0 for count in validation_counts.values())
        )

    def _run_guidance_pipeline(self, task: FuzzerTask, stage: int,
                                num_classes: int, save_dir: str, ckpt_path: str):
        """Run the full guidance pipeline after model deployment.

        Steps:
        1. Static analysis (once per task, cold-start bootstrapper)
        2. Attribution analysis (if positive samples exist)
        3. Sequence pattern mining (if positive samples exist)
        4. Merge all sources via GuidanceEngine
        5. Send guidance to fuzzer via POST /guidance
        """
        if not self._task_accepts_guidance(task):
            return
        with task.guidance_lock:
            if not self._task_accepts_guidance(task):
                return
            self._run_guidance_pipeline_locked(
                task, stage, num_classes, save_dir, ckpt_path
            )

    def _run_guidance_pipeline_locked(
            self, task: FuzzerTask, stage: int,
            num_classes: int, save_dir: str, ckpt_path: str):
        """Execute one serialized guidance round for an active task."""
        engine = task.guidance_engine
        if engine is None:
            return
        logger.info(f"[{task.task_id}] Starting guidance pipeline (stage={stage})")

        # Load training data for sequence mining and counting
        programs, labels = self._load_training_data(task)
        if not self._task_accepts_guidance(task):
            return
        n_positive = self._count_positive_samples(labels, num_classes)
        final_target_class = num_classes - 1
        n_final_target = self._count_exact_class_samples(
            labels, num_classes, final_target_class
        )

        # Step 1: Static analysis (run once per task)
        if not task.static_analysis_done:
            self._run_static_analysis(task, engine)
        if not self._task_accepts_guidance(task):
            return

        # Step 2: Attribution belongs to the newly deployed model snapshot.
        # Invalidate the previous snapshot before applying quality/data gates,
        # so a skipped or failed recomputation cannot be published as evidence
        # from the current stage/model version. The Fuzzer keeps its previously
        # acknowledged guidance until the complete new payload is sent.
        engine.update_attribution({})

        # Stage 1 only learns generic reachability. Stage 2 provides exact
        # waypoint-specific positive predictions.
        if stage == 1:
            logger.info(
                f"[{task.task_id}] Skipping attribution: Stage {stage} "
                "does not learn reached-class distinctions"
            )
        elif not self._model_quality_allows_attribution(
                save_dir, stage, num_classes):
            logger.info(
                f"[{task.task_id}] Skipping attribution: selected checkpoint "
                f"did not meet accuracy threshold "
                f"{config.attribution_min_accuracy:.2f}"
            )
        elif n_final_target >= 100:
            self._run_attribution_analysis(
                task, engine, ckpt_path, num_classes, stage
            )
        else:
            logger.info(
                f"[{task.task_id}] Skipping attribution: only "
                f"{n_final_target} final-target samples (need 100+)"
            )
        if not self._task_accepts_guidance(task):
            return

        # Step 3: Sequence pattern mining (needs 10+ positive samples)
        sequence_evidence = (
            n_positive if stage == 1 else n_final_target
        )
        if sequence_evidence >= 10:
            self._run_sequence_mining(
                task, engine, programs, labels, num_classes, stage
            )
        else:
            engine.update_sequence_patterns([])
            tier_name = "reached" if stage == 1 else "final-target"
            logger.info(
                f"[{task.task_id}] Skipping sequence mining: only "
                f"{sequence_evidence} {tier_name} samples (need 10+)"
            )
        if not self._task_accepts_guidance(task):
            return

        # Step 4: Compute and send guidance
        guidance = engine.compute_guidance()
        logger.info(f"[{task.task_id}] Guidance v{engine.version}: "
                     f"{len(guidance['syscall_weights'])} syscall weights, "
                     f"{len(guidance['mutation_templates'])} templates")

        send_ok = self._send_guidance_if_active(task, engine, guidance)
        if send_ok is None:
            return
        if send_ok:
            logger.info(f"[{task.task_id}] Guidance v{engine.version} sent to fuzzer")
        else:
            logger.error(f"[{task.task_id}] Failed to send guidance to fuzzer")

        # Log full guidance result to guidance.log
        log_mgr.log_guidance_result(
            task_id=task.task_id,
            version=engine.version,
            static_weights=engine.get_static_weights(),
            attribution_weights=engine.get_attribution_weights(),
            merged_weights=guidance.get("syscall_weights", {}),
            templates=guidance.get("mutation_templates", []),
            send_success=send_ok
        )

    def _run_static_analysis(self, task: FuzzerTask, engine: GuidanceEngine):
        """Run static analysis to find relevant syscalls.

        Report-derived tasks use PathBasedAnalyzer exclusively. KallGraph is a
        fallback only when no report is configured, keeping report-driven and
        target-only cold-start evidence from being mixed silently.
        """
        entries = []
        static_templates = []
        dispatch_observation = None
        analysis_started = time.monotonic()

        if task.report_text:
            try:
                from path_analyzer import PathBasedAnalyzer

                syzkaller_dir = os.path.expanduser(config.syzkaller_syslinux)
                pa = PathBasedAnalyzer(syzkaller_syslinux_dir=syzkaller_dir)
                entries = pa.analyze_report(task.report_text)
                dispatch_observation = pa.extract_report_dispatch_constant(
                    task.report_text, task.target_arch
                )
                logger.info(f"[{task.task_id}] Path-based analysis: {len(entries)} entries")
            except Exception as e:
                logger.error(f"[{task.task_id}] Path-based analysis failed: {e}")
        elif task.kallgraph_dir and os.path.isdir(task.kallgraph_dir):
            try:
                from static_analyzer import StaticAnalyzer
                analyzer = StaticAnalyzer(
                    kallgraph_output_dir=task.kallgraph_dir,
                    target_func=task.target_func,
                    max_callgraph_bytes=config.guidance_max_callgraph_bytes,
                    trusted_roots=tuple(config.guidance_kallgraph_roots),
                )
                if analyzer.load_callgraph():
                    entries = analyzer.find_reachable_syscall_entries()
                    if entries:
                        logger.info(
                            f"[{task.task_id}] KallGraph analysis: "
                            f"{len(entries)} entries"
                        )
            except Exception as e:
                logger.error(f"[{task.task_id}] KallGraph analysis failed: {e}")

        if entries:
            try:
                manifest = SyzlangIndex.load_cached(
                    config.syzlang_manifest_path,
                    expected_os=task.target_os,
                    expected_arch=task.target_arch,
                    expected_revision=task.target_revision,
                    expected_producer_revision=task.producer_revision,
                    max_bytes=config.syzlang_manifest_max_bytes,
                )
                if dispatch_observation is not None:
                    refinement = manifest.refine_candidates_by_report_constant(
                        entries,
                        call_name=dispatch_observation.call_name,
                        syscall_nr=dispatch_observation.syscall_nr,
                        fixed_arg_index=(
                            dispatch_observation.fixed_arg_index
                        ),
                        value=dispatch_observation.value,
                        descriptions_mode=task.descriptions_mode,
                    )
                    entries = list(refinement.entries)
                    logger.info(
                        f"[{task.task_id}] Report dispatch refinement: "
                        f"call={dispatch_observation.call_name}, "
                        f"nr=0x{dispatch_observation.syscall_nr:x}, "
                        f"arg={dispatch_observation.fixed_arg_index}, "
                        f"value=0x{dispatch_observation.value:x}, "
                        f"status={refinement.status}, "
                        f"matched={refinement.matched_name}"
                    )
                filtered = manifest.filter_candidates(
                    entries,
                    descriptions_mode=task.descriptions_mode,
                )
                entries = list(filtered.entries)
                if task.report_text:
                    template_result = manifest.build_report_static_templates(
                        entries,
                        descriptions_mode=task.descriptions_mode,
                    )
                    static_templates = list(template_result.templates)
                    entries_by_name = {
                        entry["name"]: index
                        for index, entry in enumerate(entries)
                    }
                    for producer in template_result.producer_entries:
                        existing_index = entries_by_name.get(producer["name"])
                        if existing_index is None:
                            entries_by_name[producer["name"]] = len(entries)
                            entries.append(dict(producer))
                            continue
                        existing = entries[existing_index]
                        if existing.get("guidance_role") == "entry_exact":
                            continue
                        promoted = dict(existing)
                        promoted["guidance_role"] = "resource_producer"
                        existing_weight = existing.get("weight", 0.0)
                        if (isinstance(existing_weight, bool) or
                                not isinstance(existing_weight, (int, float)) or
                                not math.isfinite(float(existing_weight))):
                            existing_weight = 0.0
                        promoted["weight"] = max(
                            float(existing_weight),
                            float(producer["weight"]),
                        )
                        entries[existing_index] = promoted
                logger.info(
                    f"[{task.task_id}] Compiled Syzlang filter: "
                    f"accepted={len(entries)}, rejected={dict(filtered.rejected)}, "
                    f"static_templates={len(static_templates)}, "
                    f"target_revision={manifest.target_revision}, "
                    f"sha256={manifest.sha256[:16]}"
                )
            except (OSError, ValueError) as error:
                entries = []
                logger.error(
                    f"[{task.task_id}] Compiled Syzlang filtering failed "
                    f"closed: {error}"
                )

        if entries:
            engine.update_static_analysis(entries)
            engine.update_static_templates(static_templates)
            task.static_analysis_done = True
            level_counts = {
                level: sum(
                    entry.get("guidance_level") == level
                    for entry in entries
                )
                for level in ("system_call", "syz_call")
            }
            logger.info(
                f"[{task.task_id}] Static analysis complete in "
                f"{time.monotonic() - analysis_started:.3f}s: "
                f"{len(entries)} entries, levels={level_counts}"
            )
        else:
            engine.update_static_templates([])
            logger.warning(f"[{task.task_id}] No static analysis results (no KallGraph data or crash report)")

    def _run_attribution_analysis(self, task: FuzzerTask, engine: GuidanceEngine,
                                   ckpt_path: str, num_classes: int, stage: int):
        """Run Captum IG attribution to find syscalls that drive target prediction."""
        global_acquired = False
        physical_acquired = False
        attribution_gpu = config.attribution_gpu_id
        physical_slot = self._gpu_semaphores[attribution_gpu]
        is_training_fallback = (
            attribution_gpu in self._training_fallback_gpu_ids
        )
        try:
            while self._task_accepts_guidance(task):
                global_acquired = self._attribution_gpu_semaphore.acquire(
                    timeout=0.5
                )
                if global_acquired:
                    break
            if not global_acquired:
                logger.info(
                    f"[{task.task_id}] Attribution canceled while waiting for GPU"
                )
                return

            while self._task_accepts_guidance(task):
                if (is_training_fallback and
                        self._has_valid_training_waiter()):
                    logger.info(
                        f"[{task.task_id}] Attribution yielding fallback GPU "
                        f"{attribution_gpu} to a queued trainer"
                    )
                    task.guidance_cancel.wait(timeout=0.5)
                    continue
                physical_acquired = physical_slot.acquire(timeout=0.5)
                if not physical_acquired:
                    continue
                # Close the check/acquire race in favor of durable training.
                if (is_training_fallback and
                        self._has_valid_training_waiter()):
                    physical_slot.release()
                    physical_acquired = False
                    task.guidance_cancel.wait(timeout=0.5)
                    continue
                break
            if not physical_acquired:
                logger.info(
                    f"[{task.task_id}] Attribution canceled while waiting for "
                    f"physical GPU {attribution_gpu}"
                )
                return
            if not self._task_accepts_guidance(task):
                logger.info(
                    f"[{task.task_id}] Attribution canceled after acquiring GPU"
                )
                return
            self._run_attribution_analysis_on_reserved_gpu(
                task, engine, ckpt_path, num_classes, stage
            )
        except Exception as e:
            logger.error(f"[{task.task_id}] Attribution analysis failed: {e}")
        finally:
            if physical_acquired:
                physical_slot.release()
            if global_acquired:
                self._attribution_gpu_semaphore.release()

    def _run_attribution_analysis_on_reserved_gpu(
            self, task: FuzzerTask, engine: GuidanceEngine,
            ckpt_path: str, num_classes: int, stage: int):
        """Compute attribution while the controller-wide GPU slot is held."""
        import sys as _sys
        filter_dir = os.path.abspath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "filter"
        ))
        with Controller._export_lock:
            if filter_dir not in _sys.path:
                _sys.path.insert(0, filter_dir)
            brain_utils = _sys.modules.pop("utils", None)
            _sys.modules.pop("attribution_guidance", None)
            try:
                from attribution_guidance import run_attribution_for_guidance
            finally:
                if brain_utils is not None:
                    _sys.modules["utils"] = brain_utils

        data_dir = self._get_data_dir(task)
        batch_indices = list_committed_batch_indices(data_dir)
        if not batch_indices:
            logger.warning(f"[{task.task_id}] No batch data for attribution")
            return

        import torch as _torch
        attribution_index = int(config.attribution_gpu_id)
        if attribution_index >= _torch.cuda.device_count():
            raise RuntimeError(
                f"configured attribution GPU {attribution_index} is unavailable"
            )
        try:
            scores = run_attribution_for_guidance(
                model_path=ckpt_path,
                base_model_path=config.base_model,
                tokenizer_path=config.tokenizer,
                data_dir=data_dir,
                data_indices=batch_indices,
                num_classes=num_classes,
                stage=stage,
                target_class=num_classes - 1,
                top_k=None,
                max_samples=50,
                max_length=1024,
                device=f"cuda:{attribution_index}",
                internal_batch_size=(
                    config.attribution_internal_batch_size
                ),
                cancel_event=task.guidance_cancel,
            )
        finally:
            # Release cached model allocations before the outer physical GPU
            # lease becomes available to a fallback trainer.
            with _torch.cuda.device(attribution_index):
                _torch.cuda.empty_cache()
        if scores and self._task_accepts_guidance(task):
            try:
                manifest = SyzlangIndex.load_cached(
                    config.syzlang_manifest_path,
                    expected_os=task.target_os,
                    expected_arch=task.target_arch,
                    expected_revision=task.target_revision,
                    expected_producer_revision=task.producer_revision,
                    max_bytes=config.syzlang_manifest_max_bytes,
                )
                filtered = manifest.filter_generation_scores(
                    scores,
                    descriptions_mode=task.descriptions_mode,
                )
            except (OSError, ValueError) as error:
                logger.error(
                    f"[{task.task_id}] Attribution manifest filtering "
                    f"failed closed: {error}"
                )
                return
            filtered_scores = {
                entry["name"]: float(entry["weight"])
                for entry in filtered.entries
            }
            norm = math.sqrt(sum(
                score * score for score in filtered_scores.values()
            ))
            normalized_scores = {
                name: score / norm
                for name, score in filtered_scores.items()
            } if norm > 0 else {}
            scores = dict(sorted(
                normalized_scores.items(),
                key=lambda item: (-item[1], item[0]),
            )[:15])
            engine.update_attribution(scores)
            logger.info(
                f"[{task.task_id}] Attribution manifest filter: "
                f"accepted={len(scores)}, rejected={dict(filtered.rejected)}, "
                f"top={list(scores.keys())[:5]}"
            )

    def _run_sequence_mining(self, task: FuzzerTask, engine: GuidanceEngine,
                              programs: list, labels: list, num_classes: int,
                              stage: int):
        """Mine frequent syscall sequences from reaching programs."""
        try:
            import sys as _sys
            analyzer_dir = os.path.abspath(os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "..", "analyzer"
            ))
            if analyzer_dir not in _sys.path:
                _sys.path.insert(0, analyzer_dir)
            from sequence_miner import SequencePatternMiner

            miner = SequencePatternMiner(
                programs, labels, num_classes=num_classes
            )
            if stage == 1:
                target_class = -1
            elif stage == 2:
                target_class = num_classes - 1
            else:
                raise ValueError(f"invalid curriculum stage: {stage}")
            templates = miner.generate_templates(
                target_class=target_class,
                min_support=0.3,
                max_gap=3,
                max_templates=16,
                min_programs=10,
                max_frontier=512,
                deadline_seconds=10.0,
            )
            engine.update_sequence_patterns(templates)
            logger.info(
                f"[{task.task_id}] Sequence mining: {len(templates)} "
                f"templates for class threshold {target_class}"
            )
        except Exception as e:
            engine.update_sequence_patterns([])
            logger.error(f"[{task.task_id}] Sequence mining failed: {e}")

    async def health(self):
        torchserve_ready = await asyncio.to_thread(
            self.torchserve_operator.is_service_ready
        )
        return {
            "healthy": torchserve_ready,
            "torchserve": torchserve_ready,
            "active_tasks": len(self.global_tasks),
        }

    def shutdown(self):
        cleanup_errors = []
        # Attempt every task cleanup even if one task has malformed state.
        for task_id, t in list(self.global_tasks.items()):
            try:
                t.guidance_cancel.set()
                with t.guidance_lock:
                    with t.lifecycle_lock:
                        t.stopping = True
                        task_cleanup_errors = []
                        try:
                            self._remove_training_waiter(task_id)
                        except Exception as error:
                            task_cleanup_errors.append(error)
                            logger.exception(
                                "[%s] Failed to remove training waiter during "
                                "shutdown", task_id
                            )
                        try:
                            training_cleanup_confirmed = (
                                self._stop_task_training(t)
                            )
                        except Exception as error:
                            training_cleanup_confirmed = False
                            task_cleanup_errors.append(error)
                            logger.exception(
                                "[%s] Trainer cleanup raised during shutdown",
                                task_id,
                            )
                        if not training_cleanup_confirmed:
                            task_cleanup_errors.append(RuntimeError(
                                f"[{task_id}] trainer cleanup was not confirmed"
                            ))
                            logger.critical(
                                f"[{task_id}] Controller shutdown could not "
                                "confirm trainer cleanup; task ownership is "
                                "retained"
                            )
                        try:
                            auxiliary_cleanup_confirmed = (
                                self._stop_task_auxiliary_processes(t)
                            )
                        except Exception as error:
                            auxiliary_cleanup_confirmed = False
                            task_cleanup_errors.append(error)
                            logger.exception(
                                "[%s] Auxiliary subprocess cleanup raised "
                                "during shutdown", task_id
                            )
                        if not auxiliary_cleanup_confirmed:
                            task_cleanup_errors.append(RuntimeError(
                                f"[{task_id}] auxiliary subprocess cleanup "
                                "was not confirmed"
                            ))
                            logger.critical(
                                f"[{task_id}] Controller shutdown could not "
                                "confirm auxiliary subprocess cleanup; task "
                                "ownership is retained"
                            )
                        if (not task_cleanup_errors and
                                training_cleanup_confirmed and
                                auxiliary_cleanup_confirmed):
                            self.global_tasks.pop(task_id, None)
                        cleanup_errors.extend(task_cleanup_errors)
            except Exception as error:
                cleanup_errors.append(error)
                logger.exception(
                    "[%s] Unexpected task cleanup failure; continuing with "
                    "remaining owned resources", task_id
                )
        try:
            if not config.direct_only:
                self._clear_authorized_keys()
        except Exception as error:
            cleanup_errors.append(error)
            logger.exception(
                "Failed to clear Controller-managed SSH keys during shutdown"
            )
        try:
            self._ensure_torchserve_stopped()
        except Exception as error:
            cleanup_errors.append(error)
            logger.exception("Failed to stop owned TorchServe service")
        self._shutdown_cleanup_failed = bool(cleanup_errors)
        if cleanup_errors:
            raise RuntimeError(
                f"Controller shutdown encountered {len(cleanup_errors)} "
                "cleanup error(s)"
            ) from cleanup_errors[0]

    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        self.startup()

        yield

        self.shutdown()

    def run(self, host: str = "0.0.0.0", port: int = 48000):
        logger.info(f"Starting SyzPilot-brain Controller on {host}:{port}")
        self.controller_port = port
        server_config = uvicorn.Config(
            self.app,
            host=host,
            port=port,
            # log_config=log_config
        )
        server = uvicorn.Server(server_config)

        previous_sighup = None
        can_handle_sighup = (
            hasattr(signal, "SIGHUP") and
            threading.current_thread() is threading.main_thread()
        )
        if can_handle_sighup:
            previous_sighup = signal.getsignal(signal.SIGHUP)

            def request_graceful_shutdown(_signum, _frame):
                server.should_exit = True

            signal.signal(signal.SIGHUP, request_graceful_shutdown)

        try:
            server.run()
        finally:
            try:
                # A normal lifespan shutdown has already released both of
                # these. This fallback covers startup failures and server-loop
                # exceptions. Keep the graceful SIGHUP handler installed until
                # potentially slow child-process cleanup is complete.
                if (self.global_tasks or self.torchserve_started or
                        self._shutdown_cleanup_failed or
                        self.existing_ssh_entries or
                        self.torchserve_operator.has_owned_service):
                    self.shutdown()
            finally:
                if can_handle_sighup:
                    signal.signal(signal.SIGHUP, previous_sighup)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SyzPilot-brain Controller')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=48000, help='Port to bind to')
    args = parser.parse_args()
    controller = Controller()
    controller.run(host=args.host, port=args.port)
