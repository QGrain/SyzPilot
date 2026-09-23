"""Select separate training and serving GPUs before launching the Controller.

This optional launcher does not create a model or training context. It samples
every GPU visible to ``nvidia-smi``, verifies CUDA ordinal identities, and
passes physical indices to the existing Controller, which continues to
arbitrate the selected training GPU before each training round.
"""

import argparse
import csv
import ctypes
import ctypes.util
import io
import logging
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


LOG = logging.getLogger("syzpilot.gpu_admission")
GPU_QUERY = (
    "nvidia-smi",
    "--query-gpu=index,pci.bus_id,mig.mode.current,"
    "memory.free,memory.total,utilization.gpu",
    "--format=csv,noheader,nounits",
)
MANUAL_GPU_VARS = (
    "SYZPILOT_TRAINING_GPU_IDS",
    "SYZPILOT_TRAINING_FALLBACK_GPU_IDS",
    "SYZPILOT_ATTRIBUTION_GPU_ID",
    "SYZPILOT_INFERENCE_GPU_ID",
)


class AdmissionError(RuntimeError):
    """No safe GPU assignment can be made from the available evidence."""


@dataclass(frozen=True)
class GpuStatus:
    index: int
    pci_bus_id: str
    free_mib: int
    total_mib: int
    utilization: int


@dataclass(frozen=True)
class AdmissionPolicy:
    training_min_free_mib: int = 24000
    training_max_utilization: int = 60
    serving_min_free_mib: int = 12000
    serving_max_utilization: int = 85
    samples: int = 3
    interval_seconds: float = 1.0

    def validate(self):
        if min(self.training_min_free_mib, self.serving_min_free_mib) <= 0:
            raise AdmissionError("GPU free-memory budgets must be positive")
        if not all(0 <= value <= 100 for value in (
                self.training_max_utilization, self.serving_max_utilization)):
            raise AdmissionError("GPU utilization limits must be in [0, 100]")
        if not 1 <= self.samples <= 3:
            raise AdmissionError("GPU probe samples must be in [1, 3]")
        if not 0 <= self.interval_seconds <= 1:
            raise AdmissionError("GPU probe interval must be in [0, 1] seconds")


def parse_gpu_status(output: str) -> dict[int, GpuStatus]:
    """Fail closed on malformed NVML rows or duplicate device indices."""
    statuses = {}
    for row in csv.reader(io.StringIO(output)):
        if len(row) != 6:
            raise AdmissionError(f"malformed nvidia-smi GPU row: {row!r}")
        try:
            index, free_mib, total_mib, utilization = (
                int(row[position].strip()) for position in (0, 3, 4, 5)
            )
        except ValueError as error:
            raise AdmissionError(f"non-numeric nvidia-smi GPU row: {row!r}") from error
        pci_bus_id = row[1].strip()
        _pci_key(pci_bus_id)
        if row[2].strip().lower() not in ("disabled", "n/a", "[n/a]"):
            raise AdmissionError("MIG devices are not supported by automatic "
                                 "GPU admission")
        if (index < 0 or free_mib < 0 or total_mib <= 0 or
                free_mib > total_mib or not 0 <= utilization <= 100 or
                index in statuses):
            raise AdmissionError(f"invalid nvidia-smi GPU row: {row!r}")
        statuses[index] = GpuStatus(
            index, pci_bus_id, free_mib, total_mib, utilization
        )
    if not statuses:
        raise AdmissionError("nvidia-smi reported no visible GPUs")
    return statuses


def _pci_key(pci_bus_id: str) -> tuple[int, int, int, int]:
    """Normalize CUDA and nvidia-smi PCI addresses with differing domain width."""
    match = re.fullmatch(
        r"([0-9a-fA-F]+):([0-9a-fA-F]+):([0-9a-fA-F]+)\.([0-7])",
        pci_bus_id,
    )
    if match is None:
        raise AdmissionError(f"invalid GPU PCI bus ID: {pci_bus_id!r}")
    return tuple(int(component, 16) for component in match.groups())


def cuda_device_bus_ids() -> dict[int, str]:
    """Read CUDA ordinals without creating a model or CUDA training context."""
    library_name = ctypes.util.find_library("cudart")
    if not library_name:
        raise AdmissionError("CUDA runtime library is unavailable")
    try:
        runtime = ctypes.CDLL(library_name)
        runtime.cudaGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        runtime.cudaGetDeviceCount.restype = ctypes.c_int
        runtime.cudaDeviceGetPCIBusId.argtypes = [
            ctypes.c_char_p, ctypes.c_int, ctypes.c_int,
        ]
        runtime.cudaDeviceGetPCIBusId.restype = ctypes.c_int
        count = ctypes.c_int()
        if runtime.cudaGetDeviceCount(ctypes.byref(count)) != 0:
            raise AdmissionError("CUDA runtime could not enumerate GPUs")
        bus_ids = {}
        for ordinal in range(count.value):
            buffer = ctypes.create_string_buffer(32)
            if runtime.cudaDeviceGetPCIBusId(buffer, len(buffer), ordinal) != 0:
                raise AdmissionError(
                    f"CUDA runtime could not identify device {ordinal}"
                )
            bus_ids[ordinal] = buffer.value.decode("ascii")
        return bus_ids
    except (OSError, AttributeError, UnicodeDecodeError) as error:
        raise AdmissionError(f"CUDA device identity query failed: {error}") from error


def validate_cuda_mapping(statuses: dict[int, GpuStatus],
                          cuda_bus_ids: dict[int, str]):
    """Require the Controller's unmasked CUDA ordinal to match NVML index."""
    if set(statuses) != set(cuda_bus_ids):
        raise AdmissionError(
            "CUDA and nvidia-smi expose different GPU index sets; automatic "
            "admission cannot safely map Controller attribution"
        )
    for index, gpu in statuses.items():
        if _pci_key(gpu.pci_bus_id) != _pci_key(cuda_bus_ids[index]):
            raise AdmissionError(
                f"CUDA ordinal {index} does not match nvidia-smi GPU {index} "
                "by PCI bus ID; automatic admission refused"
            )


def query_gpu_status(*, run=None) -> dict[int, GpuStatus]:
    """Read one complete physical-GPU snapshot from ``nvidia-smi``."""
    run = subprocess.run if run is None else run
    try:
        result = run(
            GPU_QUERY, capture_output=True, text=True, timeout=3,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise AdmissionError(f"nvidia-smi probe failed: {error}") from error
    if result.returncode != 0:
        raise AdmissionError(
            f"nvidia-smi exited with {result.returncode}: "
            f"{result.stderr.strip()[:200]}"
        )
    return parse_gpu_status(result.stdout)


def validate_physical_gpu_namespace(required_indices, *, environment=None,
                                    run=None, cuda_query=None):
    """Verify that configured GPU indices denote the same physical devices.

    The Controller passes physical indices to independently masked trainer and
    serving subprocesses. Therefore a direct launch must enforce the same PCI
    ordering and CUDA/NVML identity contract as the optional admission helper.
    """
    environment = os.environ if environment is None else environment
    cuda_query = cuda_device_bus_ids if cuda_query is None else cuda_query
    if "CUDA_VISIBLE_DEVICES" in environment:
        raise AdmissionError(
            "unset CUDA_VISIBLE_DEVICES; physical GPU indices are required"
        )
    if environment.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise AdmissionError("CUDA_DEVICE_ORDER must be PCI_BUS_ID")
    statuses = query_gpu_status(run=run)
    validate_cuda_mapping(statuses, cuda_query())
    try:
        required = {int(index) for index in required_indices}
    except (TypeError, ValueError) as error:
        raise AdmissionError(
            "configured GPU indices must be integers"
        ) from error
    missing = sorted(required.difference(statuses))
    if missing:
        raise AdmissionError(
            "configured physical GPUs are unavailable: "
            + ", ".join(str(index) for index in missing)
        )
    return statuses


def probe_gpus(policy: AdmissionPolicy, *, run=None, sleep=None) \
        -> dict[int, GpuStatus]:
    """Use worst observed free memory and utilization across stable samples."""
    run = subprocess.run if run is None else run
    sleep = time.sleep if sleep is None else sleep
    policy.validate()
    samples = []
    for attempt in range(policy.samples):
        samples.append(query_gpu_status(run=run))
        if attempt + 1 < policy.samples:
            sleep(policy.interval_seconds)

    stable_indices = set.intersection(*(set(sample) for sample in samples))
    if len(stable_indices) < 2:
        raise AdmissionError("fewer than two GPUs were visible in every sample")
    stable = {
        index: GpuStatus(
            index=index,
            pci_bus_id=samples[0][index].pci_bus_id,
            free_mib=min(sample[index].free_mib for sample in samples),
            total_mib=min(sample[index].total_mib for sample in samples),
            utilization=max(sample[index].utilization for sample in samples),
        )
        for index in sorted(stable_indices)
        if len({_pci_key(sample[index].pci_bus_id) for sample in samples}) == 1
    }
    if len(stable) < 2:
        raise AdmissionError("fewer than two stable physical GPUs were observed")
    return stable


def select_gpu_pair(statuses: dict[int, GpuStatus],
                    policy: AdmissionPolicy) -> tuple[int, int]:
    """Prefer the least busy eligible trainer, then the least busy server."""
    policy.validate()
    ranked = sorted(
        statuses.values(),
        key=lambda gpu: (gpu.utilization, -gpu.free_mib, gpu.index),
    )
    training = [
        gpu for gpu in ranked
        if (gpu.free_mib >= policy.training_min_free_mib and
            gpu.utilization <= policy.training_max_utilization)
    ]
    serving = [
        gpu for gpu in ranked
        if (gpu.free_mib >= policy.serving_min_free_mib and
            gpu.utilization <= policy.serving_max_utilization)
    ]
    for trainer in training:
        for server in serving:
            if server.index != trainer.index:
                return trainer.index, server.index
    summary = ", ".join(
        f"GPU {gpu.index}: {gpu.free_mib} MiB free, "
        f"{gpu.utilization}% utilized"
        for gpu in ranked
    )
    raise AdmissionError(
        "no distinct training/serving GPU pair satisfies "
        f"training >= {policy.training_min_free_mib} MiB free and "
        f"<= {policy.training_max_utilization}% utilized, "
        f"serving >= {policy.serving_min_free_mib} MiB free and "
        f"<= {policy.serving_max_utilization}% utilized; observed: {summary}. "
        "Free GPU resources or adjust the explicit workload budgets."
    )


def selected_environment(base: dict[str, str], policy: AdmissionPolicy,
                         training_gpu: int, serving_gpu: int) -> dict[str, str]:
    """Keep the Controller's existing two-GPU physical-index contract."""
    if "CUDA_VISIBLE_DEVICES" in base:
        raise AdmissionError(
            "unset CUDA_VISIBLE_DEVICES before GPU admission; the Controller "
            "uses the physical GPU indices reported by nvidia-smi"
        )
    if base.get("CUDA_DEVICE_ORDER", "PCI_BUS_ID") != "PCI_BUS_ID":
        raise AdmissionError(
            "CUDA_DEVICE_ORDER must be PCI_BUS_ID for automatic admission"
        )
    overridden = [name for name in MANUAL_GPU_VARS if name in base]
    if overridden:
        raise AdmissionError(
            "unset manual GPU assignment before automatic admission: "
            + ", ".join(overridden)
        )
    selected = base.copy()
    selected.update({
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "SYZPILOT_TRAINING_GPU_IDS": str(training_gpu),
        "SYZPILOT_TRAINING_FALLBACK_GPU_IDS": "",
        "SYZPILOT_ATTRIBUTION_GPU_ID": str(serving_gpu),
        "SYZPILOT_INFERENCE_GPU_ID": str(serving_gpu),
        "SYZPILOT_TRAINING_MIN_FREE_MIB": str(policy.training_min_free_mib),
        "SYZPILOT_TRAINING_MAX_GPU_UTILIZATION": str(
            policy.training_max_utilization
        ),
        "SYZPILOT_TRAINING_GPU_PROBE_SAMPLES": str(policy.samples),
        "SYZPILOT_TRAINING_GPU_PROBE_INTERVAL_SECONDS": str(
            policy.interval_seconds
        ),
    })
    return selected


def check_service_ports(controller_port: int = 48000,
                        torchserve_base_port: int = 37030):
    """Refuse a launch that could interfere with an already running service."""
    ports = (controller_port, *(torchserve_base_port + offset
                                for offset in range(5)))
    for port in ports:
        if not 1 <= port <= 65535:
            raise AdmissionError(f"invalid service port {port}")
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.bind(("0.0.0.0", port))
        except OSError as error:
            raise AdmissionError(
                f"service port {port} is unavailable ({error}); "
                "do not start another Controller over an existing instance"
            ) from error


def check_torchserve_pid_file():
    """Conservatively refuse an existing global TorchServe instance."""
    pid_file = Path(tempfile.gettempdir()) / ".model_server.pid"
    if pid_file.exists() or pid_file.is_symlink():
        raise AdmissionError(
            f"TorchServe PID file {pid_file} already exists; inspect its "
            "owner and process before launching another Controller. "
            "Automatic admission will not delete or stop it."
        )


def validate_existing_torchserve_config(
        torchserve_base_port: int = 37030,
        config_path: Path | None = None):
    """Do not preflight different ports from an existing TorchServe config."""
    config_path = (Path.cwd() / "config.properties" if config_path is None
                   else Path(config_path))
    if not config_path.exists():
        return
    try:
        lines = config_path.read_text().splitlines()
    except OSError as error:
        raise AdmissionError(
            f"cannot inspect existing TorchServe config {config_path}: {error}"
        ) from error
    settings = {}
    for line_number, line in enumerate(lines, start=1):
        if not line.strip() or line.startswith(("#", "!")):
            continue
        # Java Properties supports ':' and whitespace separators, escaped
        # keys, continuations, and last-definition-wins. Accept only the
        # unambiguous form emitted by ServeOperator.create_config().
        if (line.count("=") != 1 or "\\" in line or
                line[:1].isspace()):
            raise AdmissionError(
                f"unsupported TorchServe config syntax at line {line_number}"
            )
        name, value = line.split("=", 1)
        if (not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", name) or
                not re.fullmatch(r"[A-Za-z0-9_./:-]+", value) or
                name in settings):
            raise AdmissionError(
                f"invalid or duplicate TorchServe config key at line "
                f"{line_number}"
            )
        settings[name] = value
    expected = {
        "inference_address": f"http://0.0.0.0:{torchserve_base_port}",
        "management_address": f"http://0.0.0.0:{torchserve_base_port + 1}",
        "metrics_address": f"http://0.0.0.0:{torchserve_base_port + 2}",
        "grpc_inference_port": str(torchserve_base_port + 3),
        "grpc_management_port": str(torchserve_base_port + 4),
    }
    if any(settings.get(name) != value for name, value in expected.items()):
        raise AdmissionError(
            f"existing {config_path} does not use the five preflighted "
            "TorchServe ports; inspect it before launching"
        )


def validate_controller_command_port(command: list[str], expected_port: int):
    """Prevent checking one Controller port while launching on another."""
    actual = "48000"  # brain/controller.py CLI default
    for position, argument in enumerate(command):
        if argument == "--port":
            if position + 1 >= len(command):
                raise AdmissionError("Controller --port has no value")
            actual = command[position + 1]
        elif argument.startswith("--port="):
            actual = argument.partition("=")[2]
        else:
            continue
    if actual != str(expected_port):
        raise AdmissionError(
            f"Controller command port {actual!r} differs from admission "
            f"port {expected_port}"
        )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-min-free-mib", type=int, default=int(
        os.getenv("SYZPILOT_TRAINING_MIN_FREE_MIB", "24000")))
    parser.add_argument("--training-max-utilization", type=int, default=int(
        os.getenv("SYZPILOT_TRAINING_MAX_GPU_UTILIZATION", "60")))
    parser.add_argument("--serving-min-free-mib", type=int, default=int(
        os.getenv("SYZPILOT_SERVING_MIN_FREE_MIB", "12000")))
    parser.add_argument("--serving-max-utilization", type=int, default=int(
        os.getenv("SYZPILOT_SERVING_MAX_GPU_UTILIZATION", "85")))
    parser.add_argument("--samples", type=int, default=int(
        os.getenv("SYZPILOT_TRAINING_GPU_PROBE_SAMPLES", "3")))
    parser.add_argument("--interval-seconds", type=float, default=float(
        os.getenv("SYZPILOT_TRAINING_GPU_PROBE_INTERVAL_SECONDS", "1.0")))
    parser.add_argument("--controller-port", type=int, default=48000)
    parser.add_argument("command", nargs=argparse.REMAINDER,
                        help="optional command after --; omission is dry-run")
    args = parser.parse_args(argv)
    policy = AdmissionPolicy(
        training_min_free_mib=args.training_min_free_mib,
        training_max_utilization=args.training_max_utilization,
        serving_min_free_mib=args.serving_min_free_mib,
        serving_max_utilization=args.serving_max_utilization,
        samples=args.samples,
        interval_seconds=args.interval_seconds,
    )
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        environment = os.environ.copy()
        # Reject a masked or manually assigned namespace before probing.
        selected_environment(environment, policy, 0, 1)
        if command:
            validate_controller_command_port(command, args.controller_port)
            validate_existing_torchserve_config()
            check_torchserve_pid_file()
            check_service_ports(args.controller_port)
        # The Controller's attribution path is unmasked, so compare its CUDA
        # ordinal to nvidia-smi's physical index under an explicit PCI order.
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        statuses = probe_gpus(policy)
        validate_cuda_mapping(statuses, cuda_device_bus_ids())
        training_gpu, serving_gpu = select_gpu_pair(statuses, policy)
        environment = selected_environment(
            environment, policy, training_gpu, serving_gpu
        )
    except AdmissionError as error:
        LOG.error("GPU admission refused: %s", error)
        return 2

    LOG.info(
        "GPU admission accepted: training GPU %d; serving/attribution GPU "
        "%d (two GPUs only)", training_gpu, serving_gpu
    )
    for gpu in statuses.values():
        LOG.info("GPU %d: at least %d MiB free, at most %d%% utilized",
                 gpu.index, gpu.free_mib, gpu.utilization)
    if not command:
        LOG.info("Dry run only; provide -- COMMAND to launch the Controller")
        return 0
    try:
        os.execvpe(command[0], command, environment)
    except OSError as error:
        LOG.error("Failed to launch Controller command: %s", error)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
