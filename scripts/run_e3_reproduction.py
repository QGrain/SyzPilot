#!/usr/bin/env python3
"""Run the evaluator-facing E3 mini-reproduction from one artifact revision."""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request


DEFAULT_IMAGE = "syzpilot:e3-candidate"
DEFAULT_OUTPUT_ROOT = Path(
    os.environ.get(
        "SYZPILOT_E3_ROOT",
        str(Path.cwd() / "run/e3-results"),
    )
)
DEFAULT_KERNEL_ROOT = Path(
    os.environ.get("SYZPILOT_E3_KERNEL_ROOT", "./kernel-cases")
)
BRAIN_CONTAINER = "syzpilot-e3-brain"
FUZZER_CONTAINER = "syzpilot-e3-fuzzer"
CONTAINER_RUN_ROOT = Path("/root/e3-runs")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
E3_TOOL_PATHS = {
    "scripts/run_e3_reproduction.py": PROJECT_ROOT
    / "scripts/run_e3_reproduction.py",
    "scripts/harvest_e3_reproduction.py": PROJECT_ROOT
    / "scripts/harvest_e3_reproduction.py",
    "artifact/E3_REPRODUCTION.md": PROJECT_ROOT / "artifact/E3_REPRODUCTION.md",
}

CASES = {
    21: {
        "title": "WARNING in cfg80211_connect",
    },
    36: {
        "title": "WARNING in rds_rdma_extra_size",
    },
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_e3_tools(output_root: Path) -> dict[str, str]:
    frozen_root = output_root / "provenance/e3-tools"
    frozen_root.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, source in E3_TOOL_PATHS.items():
        target = frozen_root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        hashes[name] = sha256_file(target)
    return hashes


def verify_e3_tools(output_root: Path, expected: dict[str, str]) -> None:
    for name, expected_hash in expected.items():
        current = PROJECT_ROOT / name
        frozen = output_root / "provenance/e3-tools" / name
        if not current.is_file() or sha256_file(current) != expected_hash:
            raise RuntimeError(f"current E3 tool changed after setup: {name}")
        if not frozen.is_file() or sha256_file(frozen) != expected_hash:
            raise RuntimeError(f"frozen E3 tool changed after setup: {name}")


def verify_tools_in_revision(expected_revision: str) -> None:
    head = run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], capture=True
    ).stdout.strip()
    if head != expected_revision:
        raise RuntimeError(
            f"E3 runner checkout {head} != expected revision {expected_revision}"
        )
    for relative, current in E3_TOOL_PATHS.items():
        committed = run(
            [
                "git",
                "-C",
                str(PROJECT_ROOT),
                "show",
                f"{expected_revision}:{relative}",
            ],
            capture=True,
        ).stdout
        if current.read_text() != committed:
            raise RuntimeError(
                f"E3 tool differs from {expected_revision}: {relative}"
            )


def acquire_run_lock(output_root: Path):
    lock_path = output_root / ".e3-run.lock"
    stream = lock_path.open("w")
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        stream.close()
        raise RuntimeError(
            f"another E3 pilot/formal runner holds {lock_path}"
        ) from None
    stream.write(f"pid={os.getpid()}\nstarted_at={utc_now()}\n")
    stream.flush()
    return stream


def run(
    command: list[str], *, check: bool = True, capture: bool = False,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        check=check,
        text=True,
        capture_output=capture,
    )


def docker_exec(
    container: str,
    command: str,
    *,
    check: bool = True,
    capture: bool = False,
    detach: bool = False,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    argv = ["docker", "exec"]
    argv.append("-d" if detach else "-i")
    for key, value in (env or {}).items():
        argv.extend(["-e", f"{key}={value}"])
    argv.extend([container, "bash", "-lc", command])
    return run(argv, check=check, capture=capture)


def container_exists(name: str) -> bool:
    result = run(
        ["docker", "ps", "-aq", "--filter", f"name=^{name}$"],
        capture=True,
    )
    return bool(result.stdout.strip())


def ensure_absent_container(name: str) -> None:
    if container_exists(name):
        raise RuntimeError(
            f"container {name} already exists; inspect it or run cleanup"
        )


def image_metadata(image: str) -> dict:
    result = run(["docker", "image", "inspect", image], capture=True)
    data = json.loads(result.stdout)[0]
    return {
        "reference": image,
        "image_id": data["Id"],
        "repo_digests": data.get("RepoDigests") or [],
        "revision": (data.get("Config", {}).get("Labels") or {}).get(
            "org.opencontainers.image.revision", ""
        ),
    }


def elf_build_id(path: Path) -> str:
    output = run(["eu-readelf", "-n", str(path)], capture=True).stdout
    match = re.search(r"Build ID:\s*([0-9a-f]+)", output)
    if match is None:
        raise RuntimeError(f"no GNU build ID in {path}")
    return match.group(1)


def kernel_build_ids(kernel_root: Path, case: int) -> tuple[str, str]:
    case_root = kernel_root / f"case_{case}"
    vmlinux_id = elf_build_id(case_root / "vmlinux")
    with tempfile.NamedTemporaryFile(prefix=f"case_{case}-", suffix="-vmlinux") as out:
        run_result = subprocess.run(
            [
                str(case_root / "scripts/extract-vmlinux"),
                str(case_root / "arch/x86/boot/bzImage"),
            ],
            check=True,
            stdout=out,
        )
        del run_result
        out.flush()
        bzimage_id = elf_build_id(Path(out.name))
    return vmlinux_id, bzimage_id


def parse_waypoint_pcs(path: Path) -> list[str]:
    match = re.search(
        r"\[For SyzPilot-fuzzer:\]\s*\n(\[[^\n]+\])",
        path.read_text(),
    )
    if match is None:
        raise RuntimeError(f"no fuzzer-ready target PCs in {path}")
    return json.loads(match.group(1))


def list_tasks() -> list[dict]:
    with urllib.request.urlopen(
        "http://127.0.0.1:48000/list_tasks", timeout=10
    ) as response:
        return json.load(response)["list_tasks"]


def list_tasks_if_running() -> list[dict]:
    try:
        return list_tasks()
    except Exception:
        return []


def task_detail(task_id: str, dashboard_token: str) -> dict:
    encoded = urllib.parse.quote(task_id, safe="")
    request = urllib.request.Request(
        f"http://127.0.0.1:48000/api/task/{encoded}/detail",
        headers={"Authorization": f"Bearer {dashboard_token}"},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.load(response)


def wait_brain(timeout: int = 240) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(
                "http://127.0.0.1:48000/health", timeout=3
            ) as response:
                payload = json.load(response)
            if payload.get("healthy") and payload.get("torchserve"):
                return
        except Exception:
            pass
        time.sleep(3)
    raise RuntimeError("Brain did not become healthy")


def parse_csv_rows(text: str, fields: tuple[str, ...]) -> list[dict]:
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        values = [value.strip() for value in line.split(",", len(fields) - 1)]
        if len(values) == len(fields):
            rows.append(dict(zip(fields, values)))
    return rows


def gpu_inventory() -> list[dict]:
    output = run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture=True,
    ).stdout
    return parse_csv_rows(
        output,
        (
            "index",
            "uuid",
            "name",
            "memory_total_mib",
            "memory_used_mib",
            "utilization_pct",
        ),
    )


def compute_apps() -> list[dict]:
    result = run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture=True,
    )
    return parse_csv_rows(
        result.stdout,
        ("gpu_uuid", "pid", "process_name", "memory_used_mib"),
    )


def container_pids(name: str) -> set[int]:
    if not container_exists(name):
        return set()
    result = run(["docker", "top", name, "-eo", "pid"], capture=True)
    pids = set()
    for line in result.stdout.splitlines()[1:]:
        value = line.strip().split()[0] if line.strip() else ""
        if value.isdigit():
            pids.add(int(value))
    return pids


def pid_belongs_to_container(pid: int, name: str) -> bool:
    if not container_exists(name):
        return False
    container_id = run(
        ["docker", "inspect", "--format", "{{.Id}}", name], capture=True
    ).stdout.strip()
    cgroup = Path(f"/proc/{pid}/cgroup")
    if not cgroup.exists():
        return False
    text = cgroup.read_text(errors="replace")
    return container_id in text or container_id[:12] in text


def gpu_snapshot(selected_indices: tuple[str, str]) -> dict:
    inventory = gpu_inventory()
    selected = {
        row["uuid"]: row
        for row in inventory
        if row["index"] in selected_indices
    }
    first_apps = [app for app in compute_apps() if app["gpu_uuid"] in selected]
    time.sleep(0.2)
    apps = [app for app in compute_apps() if app["gpu_uuid"] in selected]
    stable_keys = {(app["gpu_uuid"], app["pid"]) for app in apps}
    vanished_apps = [
        app
        for app in first_apps
        if (app["gpu_uuid"], app["pid"]) not in stable_keys
    ]
    brain_pids = container_pids(BRAIN_CONTAINER)
    for app in apps:
        pid = int(app["pid"])
        app["owned_by_brain_container"] = (
            pid in brain_pids or pid_belongs_to_container(pid, BRAIN_CONTAINER)
        )
    return {
        "timestamp": utc_now(),
        "gpus": list(selected.values()),
        "compute_apps": apps,
        "vanished_compute_apps": vanished_apps,
        "external_compute_apps": [
            app for app in apps if not app["owned_by_brain_container"]
        ],
    }


def parse_gpu_devices(value: str) -> tuple[str, str]:
    devices = tuple(item.strip() for item in value.split(",") if item.strip())
    if len(devices) != 2 or len(set(devices)) != 2:
        raise ValueError("--gpu-devices must name two distinct physical GPU indices")
    available = {row["index"] for row in gpu_inventory()}
    if any(device not in available for device in devices):
        raise ValueError(
            f"requested GPUs {devices} are not a subset of {sorted(available)}"
        )
    return devices  # type: ignore[return-value]


def parse_cpu_pairs(value: str) -> tuple[str, str, str, str]:
    pairs = tuple(item.strip() for item in value.split(";") if item.strip())
    if len(pairs) != 4:
        raise ValueError("--cpu-pairs must contain four semicolon-separated pairs")
    used = set()
    allowed = os.sched_getaffinity(0)
    for pair in pairs:
        cpus = [int(item) for item in re.split(r"[-,]", pair)]
        if len(cpus) != 2 or len(set(cpus)) != 2:
            raise ValueError(f"invalid two-core CPU pair: {pair}")
        if used.intersection(cpus):
            raise ValueError("CPU pairs overlap")
        if any(cpu not in allowed for cpu in cpus):
            raise ValueError(f"CPU pair {pair} is outside the allowed host cpuset")
        used.update(cpus)
    return pairs  # type: ignore[return-value]


def port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(("127.0.0.1", port)) != 0


def load_manifest(output_root: Path) -> dict:
    path = output_root / "manifest.json"
    if not path.is_file():
        raise RuntimeError(f"missing setup manifest: {path}")
    return json.loads(path.read_text())


def docker_proxy_env_args() -> list[str]:
    """Let Docker copy an evaluator's optional proxy variables by name."""
    args = []
    for name in (
        "http_proxy", "https_proxy", "no_proxy",
        "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY",
    ):
        if os.environ.get(name):
            args.extend(["-e", name])
    return args


def start_containers(
    args: argparse.Namespace, devices: tuple[str, str]
) -> None:
    try:
        run(
            [
                "docker", "run", "-d", "--init", "--name", BRAIN_CONTAINER,
                "--network", "host", "--gpus", f"\"device={','.join(devices)}\"",
                "-v", f"{args.output_root / 'runs'}:{CONTAINER_RUN_ROOT}",
                "-v", (
                    f"{args.output_root / 'brain/receiver_data'}:"
                    "/root/SyzPilot/brain/receiver_data"
                ),
                "-v", (
                    f"{args.output_root / 'brain/logs'}:"
                    "/root/SyzPilot/brain/logs"
                ),
                "-v", (
                    f"{args.output_root / 'brain/model_store'}:"
                    "/root/SyzPilot/brain/model_store"
                ),
                args.image, "sleep", "infinity",
            ]
        )
        run(
            [
                "docker", "run", "-d", "--init", "--name", FUZZER_CONTAINER,
                "--network", "host", "--device", "/dev/kvm",
                *docker_proxy_env_args(),
                "-v", f"{args.output_root / 'runs'}:{CONTAINER_RUN_ROOT}",
                "-v", f"{args.output_root / 'fuzzers'}:/root/fuzzers",
                "-v", (
                    f"{args.kernel_root / 'case_21'}:"
                    "/root/kernels/case_21:ro"
                ),
                "-v", (
                    f"{args.kernel_root / 'case_36'}:"
                    "/root/kernels/case_36:ro"
                ),
                args.image, "sleep", "infinity",
            ]
        )
    except Exception:
        for name in (FUZZER_CONTAINER, BRAIN_CONTAINER):
            if container_exists(name):
                run(["docker", "rm", "-f", name], check=False)
        raise


def setup(args: argparse.Namespace) -> None:
    if not args.expected_revision:
        raise ValueError("setup requires --expected-revision")
    verify_tools_in_revision(args.expected_revision)
    ensure_absent_container(BRAIN_CONTAINER)
    ensure_absent_container(FUZZER_CONTAINER)
    if list_tasks_if_running():
        raise RuntimeError("another Brain is already listening on port 48000")
    if not port_is_free(48000):
        raise RuntimeError("port 48000 is already in use")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise RuntimeError(
            f"output root is not empty: {args.output_root}; use a fresh path"
        )

    metadata = image_metadata(args.image)
    if metadata["revision"] != args.expected_revision:
        raise RuntimeError(
            f"image revision {metadata['revision']} != {args.expected_revision}"
        )
    devices = parse_gpu_devices(args.gpu_devices)
    preflight = gpu_snapshot(devices)
    if preflight["external_compute_apps"] and not args.allow_shared_gpus:
        raise RuntimeError(
            "selected GPUs have external compute processes; choose exclusive GPUs"
        )

    build_ids = {}
    for case in CASES:
        case_root = args.kernel_root / f"case_{case}"
        for path in (case_root / "vmlinux", case_root / "arch/x86/boot/bzImage"):
            if not path.is_file():
                raise RuntimeError(f"missing kernel artifact: {path}")
        vmlinux_id, bzimage_id = kernel_build_ids(args.kernel_root, case)
        if vmlinux_id != bzimage_id:
            raise RuntimeError(
                f"case_{case} vmlinux build ID {vmlinux_id} does not match "
                f"the kernel embedded in bzImage ({bzimage_id})"
            )
        build_ids[f"case_{case}"] = vmlinux_id

    for relative in (
        "runs",
        "provenance/e3-tools",
        "brain/receiver_data",
        "brain/logs",
        "brain/model_store",
        "fuzzers",
    ):
        (args.output_root / relative).mkdir(parents=True, exist_ok=True)

    start_containers(args, devices)

    try:
        for container in (BRAIN_CONTAINER, FUZZER_CONTAINER):
            image_head = docker_exec(
                container,
                "git -C /root/SyzPilot rev-parse HEAD",
                capture=True,
            ).stdout.strip()
            if image_head != args.expected_revision:
                raise RuntimeError(
                    f"{container} source {image_head} != {args.expected_revision}"
                )

        mapping = docker_exec(
            BRAIN_CONTAINER,
            "python - <<'PY'\n"
            "import json, torch\n"
            "print(json.dumps({'count': torch.cuda.device_count(), "
            "'names': [torch.cuda.get_device_name(i) for i in "
            "range(torch.cuda.device_count())]}))\n"
            "PY",
            capture=True,
        ).stdout.strip()
        if json.loads(mapping)["count"] != 2:
            raise RuntimeError(f"Brain container did not receive two GPUs: {mapping}")

        docker_exec(
            FUZZER_CONTAINER,
            "cd /root/SyzPilot && scripts/artifact_eval.sh prepare-fuzzer",
        )

        preparation = args.output_root / "runs/preparation"
        runtime_pcs = {}
        published_waypoint_references = {}
        for case in CASES:
            (preparation / f"case_{case}").mkdir(parents=True, exist_ok=True)
            output = (
                CONTAINER_RUN_ROOT
                / "preparation"
                / f"case_{case}"
                / "waypoints.txt"
            )
            docker_exec(
                FUZZER_CONTAINER,
                "cd /root/SyzPilot && "
                f"python analyzer/waypoints_extractor.py -k /root/kernels/case_{case} "
                f"-t benchmark/configs/case_{case}.title "
                f"-r benchmark/configs/case_{case}.report "
                f"| tee {shlex.quote(str(output))}",
            )
            observed = parse_waypoint_pcs(
                preparation / f"case_{case}" / "waypoints.txt"
            )
            published_record = json.loads(
                docker_exec(
                    FUZZER_CONTAINER,
                    f"cat /root/SyzPilot/benchmark/waypoints/case_{case}.json",
                    capture=True,
                ).stdout
            )
            if not observed:
                raise RuntimeError(f"case_{case} extractor returned no target PCs")
            published_build_id = published_record[
                "kernel_build_identity"
            ].removeprefix("build-id:")
            same_published_build = (
                build_ids[f"case_{case}"] == published_build_id
            )
            if same_published_build and observed != published_record["target_pcs"]:
                raise RuntimeError(
                    f"case_{case} extracted PCs {observed} != published "
                    f"{published_record['target_pcs']} for the same kernel build"
                )
            runtime_pcs[f"case_{case}"] = observed
            published_waypoint_references[f"case_{case}"] = {
                "kernel_build_identity": published_record["kernel_build_identity"],
                "report_target": published_record["report_target"],
                "same_kernel_build": same_published_build,
                "pc_match_checked": same_published_build,
            }

        docker_exec(
            BRAIN_CONTAINER,
            "cd /root/SyzPilot && scripts/artifact_eval.sh start-brain",
            detach=True,
            env={
                "SYZPILOT_RUN_ROOT": "/root/e3-runs/brain",
                "SYZPILOT_TRAINING_GPU_IDS": "0",
                "SYZPILOT_INFERENCE_GPU_ID": "1",
                "SYZPILOT_ATTRIBUTION_GPU_ID": "1",
                "DASHBOARD_TOKEN": args.dashboard_token,
            },
        )
        wait_brain()

        patch_hash = docker_exec(
            FUZZER_CONTAINER,
            "sha256sum /root/SyzPilot/fuzzer/SyzPilot-fuzzer.diff | cut -d' ' -f1",
            capture=True,
        ).stdout.strip()
        binary_hash_output = docker_exec(
            FUZZER_CONTAINER,
            "cd /root/fuzzers/SyzPilot-fuzzer/bin && "
            "sha256sum syz-manager linux_amd64/syz-executor",
            capture=True,
        ).stdout
        binary_hashes = {
            line.split()[1]: line.split()[0]
            for line in binary_hash_output.splitlines()
        }
        e3_tool_hashes = freeze_e3_tools(args.output_root)
        formal_matrix = [
            {"case": case, "arm": arm, "repetition": repetition}
            for repetition in range(1, 4)
            for case in CASES
            for arm in ("blank", "full")
        ]
        manifest = {
            "created_at": utc_now(),
            "artifact_image": metadata,
            "artifact_revision": args.expected_revision,
            "container_source_revision": args.expected_revision,
            "fuzzer_patch_sha256": patch_hash,
            "fuzzer_binary_sha256": binary_hashes,
            "e3_tool_sha256": e3_tool_hashes,
            "kernel_build_ids": build_ids,
            "target_pcs": runtime_pcs,
            "published_waypoint_references": published_waypoint_references,
            "host_gpu_devices": list(devices),
            "container_gpu_roles": {"training": 0, "inference_attribution": 1},
            "setup_gpu_snapshot": preflight,
            "formal_matrix": formal_matrix,
            "protocol": {
                "duration": "4h",
                "vm_count": 1,
                "vm_cpu": 2,
                "vm_mem_mib": 4096,
                "procs": 8,
                "reproduce": False,
            },
        }
        (args.output_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
    except (Exception, KeyboardInterrupt):
        cleanup(args)
        raise


def make_config(
    args: argparse.Namespace,
    case: int,
    arm: str,
    repetition: int,
    attempt: int,
    phase: str,
    port: int,
) -> tuple[Path, str]:
    name = f"case{case}_r{repetition}_a{attempt}_{arm}"
    host_root = args.output_root / "runs" / phase / name
    container_root = CONTAINER_RUN_ROOT / phase / name
    host_root.mkdir(parents=True, exist_ok=False)
    waypoint_path = (
        CONTAINER_RUN_ROOT / "preparation" / f"case_{case}" / "waypoints.txt"
    )
    config_path = host_root / "manager.cfg"
    docker_exec(
        FUZZER_CONTAINER,
        "cd /root/SyzPilot && "
        "python scripts/prepare_functional_config.py "
        f"--case {case} --waypoints-output {shlex.quote(str(waypoint_path))} "
        f"--output {shlex.quote(str(container_root / 'manager.cfg'))} "
        f"--run-dir {shlex.quote(str(container_root))} "
        f"--kernel-dir /root/kernels/case_{case} "
        "--image /root/images/image-template/disk.img "
        "--ssh-key /root/images/image-template/disk.id_rsa "
        "--syzkaller /root/fuzzers/SyzPilot-fuzzer "
        "--brain-host 127.0.0.1 --fuzzer-host 127.0.0.1 "
        f"--manager-port {port}",
    )
    config = json.loads(config_path.read_text())
    config["vm"]["snapshot"] = True
    config["SyzPilot"]["task_name"] = f"ae_e3_{phase}_{name}"
    config["SyzPilot"]["enable_dump"] = False
    config["SyzPilot"]["persist_target_evidence"] = False
    config["SyzPilot"]["enable_diverse_cover"] = False
    config["SyzPilot"]["enable_online_guidance"] = arm == "full"
    config["SyzPilot"]["enable_sequence_guidance"] = arm == "full"
    config["SyzPilot"]["enable_attribution_guidance"] = arm == "full"
    if arm == "blank":
        config["SyzPilot"]["observe_target_coverage"] = True
        config["SyzPilot"]["reach_filter"]["enable"] = False
        config["SyzPilot"]["reach_filter"]["controller"] = ""
        config["SyzPilot"]["directed_corpus"] = {"enable": False}
        config["SyzPilot"]["template_insert_weight"] = 0
        config["SyzPilot"]["allow_standalone_guidance"] = False
        for field in (
            "cold_start_seed_dir",
            "cold_start_seed_catalog",
            "cold_start_auto_dependencies",
        ):
            config["SyzPilot"].pop(field, None)
    rendered_path = config_path.with_suffix(".cfg.tmp")
    rendered_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    rendered_path.replace(config_path)
    return config_path, name


def duration_to_seconds(value: str) -> int:
    match = re.fullmatch(r"([1-9][0-9]*)([mh])", value)
    if match is None:
        raise ValueError(f"invalid duration: {value}")
    number, unit = match.groups()
    return int(number) * {"m": 60, "h": 3600}[unit]


def write_manager_wrapper(
    host_root: Path,
    container_root: Path,
    name: str,
    cpus: str,
    duration: str,
) -> None:
    wrapper = host_root / "run_manager.sh"
    command = (
        f"taskset -c {shlex.quote(cpus)} "
        "/root/fuzzers/SyzPilot-fuzzer/bin/syz-manager "
        f"-config {shlex.quote(str(container_root / 'manager.cfg'))} "
        f"-timeout {shlex.quote(duration)} "
        f"-bench {shlex.quote(str(container_root / f'{name}.bench.log'))} "
        f"> {shlex.quote(str(container_root / 'fuzzer.log'))} 2>&1"
    )
    wrapper.write_text(
        "#!/usr/bin/env bash\n"
        "set +e\n"
        "started_epoch=$(date +%s)\n"
        "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)\n"
        f"{command}\n"
        "status=$?\n"
        "ended_epoch=$(date +%s)\n"
        "ended_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)\n"
        f"printf 'name={name}\\nstarted_epoch=%s\\nstarted_at=%s\\n"
        "ended_epoch=%s\\nended_at=%s\\nexit_code=%s\\n' "
        '"$started_epoch" "$started_at" "$ended_epoch" "$ended_at" "$status" '
        f"> {shlex.quote(str(container_root / 'manager.exit'))}\n"
        "exit \"$status\"\n"
    )
    wrapper.chmod(0o755)


def read_exit_file(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text().splitlines():
        key, value = line.split("=", 1)
        result[key] = value
    return result


def process_group_alive(pid: int) -> bool:
    return docker_exec(
        FUZZER_CONTAINER, f"kill -0 -- -{pid}", check=False
    ).returncode == 0


def unregister_e3_tasks() -> None:
    for task in list_tasks_if_running():
        if not task.get("task_name", "").startswith("ae_e3_"):
            continue
        encoded = urllib.parse.quote(task["task_id"], safe="")
        request = urllib.request.Request(
            f"http://127.0.0.1:48000/unregister/{encoded}", method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=60):
                pass
        except Exception:
            pass


def stop_specs(specs: list[dict]) -> None:
    for signal_name in ("TERM", "KILL"):
        for spec in specs:
            pid = spec.get("supervisor_pid")
            if pid and process_group_alive(pid):
                docker_exec(
                    FUZZER_CONTAINER,
                    f"kill -{signal_name} -- -{pid}",
                    check=False,
                )
        time.sleep(5 if signal_name == "TERM" else 1)
    deadline = time.time() + 30
    while time.time() < deadline and any(
        process_group_alive(spec["supervisor_pid"])
        for spec in specs
        if spec.get("supervisor_pid")
    ):
        time.sleep(1)
    survivors = [
        spec["supervisor_pid"]
        for spec in specs
        if spec.get("supervisor_pid")
        and process_group_alive(spec["supervisor_pid"])
    ]
    unregister_e3_tasks()
    deadline = time.time() + 90
    while time.time() < deadline:
        remaining = [
            task
            for task in list_tasks_if_running()
            if task.get("task_name", "").startswith("ae_e3_")
        ]
        if not remaining:
            if survivors:
                raise RuntimeError(
                    "E3 process groups survived cleanup: "
                    + ", ".join(map(str, survivors))
                )
            return
        time.sleep(2)
    raise RuntimeError("E3 process or Controller task cleanup did not finish")


def append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def final_bench_row(path: Path) -> dict:
    text = path.read_text()
    decoder = json.JSONDecoder()
    rows = []
    offset = 0
    while offset < len(text):
        while offset < len(text) and text[offset].isspace():
            offset += 1
        if offset == len(text):
            break
        row, offset = decoder.raw_decode(text, offset)
        rows.append(row)
    if not rows:
        raise RuntimeError(f"bench log is empty: {path}")
    return rows[-1]


def snapshot_brain(path: Path, dashboard_token: str) -> dict:
    tasks = list_tasks_if_running()
    details = []
    for task in tasks:
        try:
            details.append(task_detail(task["task_id"], dashboard_token))
        except Exception as error:
            details.append({"task_id": task["task_id"], "error": str(error)})
    snapshot = {"timestamp": utc_now(), "tasks": details}
    append_jsonl(path, snapshot)
    return snapshot


def assert_exclusive_gpus(args: argparse.Namespace) -> dict:
    manifest = load_manifest(args.output_root)
    devices = tuple(manifest["host_gpu_devices"])
    snapshot = gpu_snapshot(devices)  # type: ignore[arg-type]
    if snapshot["external_compute_apps"]:
        raise RuntimeError(
            "formal E3 requires exclusive selected GPUs; external processes: "
            + json.dumps(snapshot["external_compute_apps"])
        )
    return snapshot


def launch_batch(
    args: argparse.Namespace,
    phase: str,
    repetition: int,
    attempt: int,
    duration: str,
    cases: tuple[int, ...],
) -> dict:
    manifest = load_manifest(args.output_root)
    verify_e3_tools(args.output_root, manifest["e3_tool_sha256"])
    expected_revision = manifest["artifact_revision"]
    current_image = image_metadata(manifest["artifact_image"]["reference"])
    if current_image["image_id"] != manifest["artifact_image"]["image_id"]:
        raise RuntimeError("artifact image ID changed after setup")
    for container in (BRAIN_CONTAINER, FUZZER_CONTAINER):
        head = docker_exec(
            container, "git -C /root/SyzPilot rev-parse HEAD", capture=True
        ).stdout.strip()
        if head != expected_revision:
            raise RuntimeError(f"{container} source revision changed")
    patch_hash = docker_exec(
        FUZZER_CONTAINER,
        "sha256sum /root/SyzPilot/fuzzer/SyzPilot-fuzzer.diff | cut -d' ' -f1",
        capture=True,
    ).stdout.strip()
    binary_hash_output = docker_exec(
        FUZZER_CONTAINER,
        "cd /root/fuzzers/SyzPilot-fuzzer/bin && "
        "sha256sum syz-manager linux_amd64/syz-executor",
        capture=True,
    ).stdout
    binary_hashes = {
        line.split()[1]: line.split()[0]
        for line in binary_hash_output.splitlines()
    }
    if patch_hash != manifest["fuzzer_patch_sha256"]:
        raise RuntimeError("Fuzzer patch changed after setup")
    if binary_hashes != manifest["fuzzer_binary_sha256"]:
        raise RuntimeError("Fuzzer binaries changed after setup")
    if list_tasks():
        raise RuntimeError("Brain still owns tasks before batch launch")

    cpu_pairs = parse_cpu_pairs(args.cpu_pairs)
    base_logical = [(case, arm) for case in cases for arm in ("blank", "full")]
    rotation = (repetition - 1) % len(base_logical)
    logical = base_logical[rotation:] + base_logical[:rotation]
    ordered_pairs = cpu_pairs
    phase_root = args.output_root / "runs" / phase
    phase_root.mkdir(parents=True, exist_ok=True)
    batch_path = phase_root / f"batch_r{repetition}_a{attempt}.json"
    telemetry_path = phase_root / f"gpu_r{repetition}_a{attempt}.jsonl"
    brain_path = phase_root / f"brain_r{repetition}_a{attempt}.jsonl"
    if batch_path.exists() or telemetry_path.exists() or brain_path.exists():
        raise RuntimeError(f"batch r{repetition} already has output under {phase_root}")

    preflight = gpu_snapshot(tuple(manifest["host_gpu_devices"]))
    if phase == "formal" and preflight["external_compute_apps"]:
        raise RuntimeError("formal batch preflight found external GPU compute processes")
    append_jsonl(telemetry_path, preflight)

    specs: list[dict] = []
    batch = {
        "phase": phase,
        "repetition": repetition,
        "attempt": attempt,
        "duration": duration,
        "status": "launching",
        "started_at": utc_now(),
        "runs": specs,
    }
    batch_path.write_text(json.dumps(batch, indent=2, sort_keys=True) + "\n")
    try:
        for index, ((case, arm), cpus) in enumerate(zip(logical, ordered_pairs)):
            port = (
                (50000 if phase == "pilot" else 52000)
                + repetition * 100
                + attempt * 10
                + index
            )
            if not port_is_free(port):
                raise RuntimeError(f"manager port {port} is already in use")
            config_path, name = make_config(
                args, case, arm, repetition, attempt, phase, port
            )
            host_root = config_path.parent
            container_root = CONTAINER_RUN_ROOT / phase / name
            write_manager_wrapper(host_root, container_root, name, cpus, duration)
            command = (
                "unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; "
                "export no_proxy=localhost,127.0.0.1; export NO_PROXY=$no_proxy; "
                f"nohup setsid {shlex.quote(str(container_root / 'run_manager.sh'))} "
                f"> {shlex.quote(str(container_root / 'supervisor.log'))} "
                "2>&1 < /dev/null & echo $!"
            )
            pid = int(
                docker_exec(FUZZER_CONTAINER, command, capture=True).stdout.strip()
            )
            spec = {
                "case": case,
                "arm": arm,
                "name": name,
                "cpus": cpus,
                "port": port,
                "config": str(config_path),
                "started_at": utc_now(),
                "supervisor_pid": pid,
            }
            specs.append(spec)
            batch_path.write_text(
                json.dumps(batch, indent=2, sort_keys=True) + "\n"
            )
            time.sleep(2)

        time.sleep(30)
        for spec in specs:
            exit_path = (
                args.output_root / "runs" / phase / spec["name"] / "manager.exit"
            )
            alive = docker_exec(
                FUZZER_CONTAINER,
                f"kill -0 {spec['supervisor_pid']}",
                check=False,
            ).returncode == 0
            if not alive or exit_path.exists():
                raise RuntimeError(f"{spec['name']} exited during startup")

        duration_seconds = duration_to_seconds(duration)
        deadline = time.time() + duration_seconds + 20 * 60
        pending = {spec["name"]: spec for spec in specs}
        last_snapshot = 0.0
        while pending and time.time() < deadline:
            if time.time() - last_snapshot >= 60:
                snapshot = gpu_snapshot(tuple(manifest["host_gpu_devices"]))
                append_jsonl(telemetry_path, snapshot)
                snapshot_brain(brain_path, args.dashboard_token)
                last_snapshot = time.time()
                if phase == "formal" and snapshot["external_compute_apps"]:
                    raise RuntimeError("external GPU compute process appeared")
            for name, spec in list(pending.items()):
                exit_path = (
                    args.output_root / "runs" / phase / name / "manager.exit"
                )
                if not exit_path.exists():
                    continue
                exit_data = read_exit_file(exit_path)
                elapsed = int(exit_data["ended_epoch"]) - int(
                    exit_data["started_epoch"]
                )
                if int(exit_data["exit_code"]) != 0 or elapsed < duration_seconds - 300:
                    raise RuntimeError(
                        f"{name} ended abnormally: exit={exit_data['exit_code']} "
                        f"elapsed={elapsed}s"
                    )
                spec["exit"] = exit_data
                del pending[name]
            if pending:
                time.sleep(30)
        if pending:
            raise RuntimeError("managers exceeded deadline: " + ", ".join(pending))

        snapshot_brain(brain_path, args.dashboard_token)

        duration_seconds = duration_to_seconds(duration)
        for spec in specs:
            run_root = args.output_root / "runs" / phase / spec["name"]
            bench_path = run_root / f"{spec['name']}.bench.log"
            final = final_bench_row(bench_path)
            if int(final.get("exec total", 0)) <= 0:
                raise RuntimeError(f"{spec['name']} completed without executions")
            if int(final.get("uptime", 0)) < duration_seconds - 600:
                raise RuntimeError(
                    f"{spec['name']} has insufficient bench uptime: "
                    f"{final.get('uptime')}"
                )
            missing_targets = [
                pc
                for pc in manifest["target_pcs"][f"case_{spec['case']}"]
                if f"reachability {pc}" not in final
            ]
            if missing_targets:
                raise RuntimeError(
                    f"{spec['name']} is missing target observation metrics: "
                    + ", ".join(missing_targets)
                )
            spec["final_exec_total"] = int(final["exec total"])
            spec["final_bench_uptime"] = int(final["uptime"])
        batch["status"] = "completed"
        batch["ended_at"] = utc_now()
        batch_path.write_text(json.dumps(batch, indent=2, sort_keys=True) + "\n")

        task_deadline = time.time() + 240
        while list_tasks_if_running() and time.time() < task_deadline:
            time.sleep(5)
        if list_tasks_if_running():
            raise RuntimeError("Brain task cleanup did not complete after batch")

        if phase == "pilot":
            full_spec = next(spec for spec in specs if spec["arm"] == "full")
            full_root = args.output_root / "runs" / phase / full_spec["name"]
            if "Registration successful" not in (
                full_root / "fuzzer.log"
            ).read_text(errors="replace"):
                raise RuntimeError("pilot Full run did not register with the Brain")
            receiver_files = list(
                (
                    args.output_root
                    / "brain/receiver_data"
                    / f"ae_e3_{phase}_{full_spec['name']}"
                ).rglob("*.pkl")
            )
            if not receiver_files:
                raise RuntimeError("pilot Full run produced no Receiver batches")
            snapshots = load_jsonl(brain_path)
            matching_details = [
                task
                for snapshot in snapshots
                for task in snapshot.get("tasks", [])
                if task.get("task_name")
                == f"ae_e3_{phase}_{full_spec['name']}"
            ]
            if not matching_details or any("error" in task for task in matching_details):
                raise RuntimeError("pilot could not read the protected Brain task detail")
            passed = {
                "passed": True,
                "created_at": utc_now(),
                "artifact_revision": manifest["artifact_revision"],
                "image_id": manifest["artifact_image"]["image_id"],
                "batch": batch_path.name,
                "attempt": attempt,
                "runs": [spec["name"] for spec in specs],
            }
            (args.output_root / "pilot_passed.json").write_text(
                json.dumps(passed, indent=2, sort_keys=True) + "\n"
            )
        return batch
    except (Exception, KeyboardInterrupt) as error:
        cleanup_error = None
        try:
            stop_specs(specs)
        except Exception as stop_error:
            cleanup_error = str(stop_error)
        batch["status"] = "invalid"
        batch["error"] = str(error)
        if cleanup_error:
            batch["cleanup_error"] = cleanup_error
        batch["ended_at"] = utc_now()
        batch_path.write_text(json.dumps(batch, indent=2, sort_keys=True) + "\n")
        if cleanup_error:
            raise RuntimeError(
                f"{error}; cleanup also failed: {cleanup_error}"
            ) from error
        raise


def recover_interrupted_batches(formal_root: Path) -> None:
    for batch_path in sorted(formal_root.glob("batch_r*_a*.json")):
        batch = json.loads(batch_path.read_text())
        if batch.get("status") != "launching":
            continue
        stop_specs(batch.get("runs", []))
        batch["status"] = "invalid"
        batch["error"] = "recovered after an interrupted host runner"
        batch["ended_at"] = utc_now()
        batch_path.write_text(json.dumps(batch, indent=2, sort_keys=True) + "\n")


def pilot(args: argparse.Namespace) -> None:
    duration = args.duration or "30m"
    if duration != "30m":
        raise ValueError("the E3 pilot is frozen at 30m")
    launch_batch(args, "pilot", 1, args.attempt, duration, (36,))


def formal(args: argparse.Namespace) -> None:
    duration = args.duration or "4h"
    if duration != "4h" or args.repetitions != 3:
        raise ValueError("formal E3 is frozen at 3 repetitions of 4h")
    manifest = load_manifest(args.output_root)
    pilot_path = args.output_root / "pilot_passed.json"
    if not pilot_path.is_file():
        raise RuntimeError("formal E3 requires a successful 30-minute pilot")
    pilot_state = json.loads(pilot_path.read_text())
    if (
        not pilot_state.get("passed")
        or pilot_state.get("artifact_revision") != manifest["artifact_revision"]
        or pilot_state.get("image_id") != manifest["artifact_image"]["image_id"]
    ):
        raise RuntimeError("pilot evidence does not match the current artifact")

    selection_path = args.output_root / "runs/formal/selected_attempts.json"
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    recover_interrupted_batches(selection_path.parent)
    assert_exclusive_gpus(args)
    selected = (
        json.loads(selection_path.read_text()) if selection_path.is_file() else {}
    )
    for repetition in range(1, args.repetitions + 1):
        if str(repetition) in selected:
            continue
        last_error = None
        existing_attempts = [
            int(match.group(1))
            for path in selection_path.parent.glob(f"batch_r{repetition}_a*.json")
            if (match := re.fullmatch(
                rf"batch_r{repetition}_a(\d+)\.json", path.name
            ))
        ]
        first_attempt = max(existing_attempts, default=0) + 1
        for attempt in range(
            first_attempt, first_attempt + args.max_attempts
        ):
            try:
                assert_exclusive_gpus(args)
                launch_batch(
                    args, "formal", repetition, attempt, duration, (21, 36)
                )
                selected[str(repetition)] = attempt
                selection_path.write_text(
                    json.dumps(selected, indent=2, sort_keys=True) + "\n"
                )
                break
            except Exception as error:
                last_error = error
        else:
            raise RuntimeError(
                f"formal repetition {repetition} exhausted "
                f"{args.max_attempts} attempts: {last_error}"
            )


def cleanup(args: argparse.Namespace) -> None:
    for name in (FUZZER_CONTAINER, BRAIN_CONTAINER):
        if container_exists(name):
            run(["docker", "rm", "-f", name])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("setup", "pilot", "formal", "cleanup"))
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--expected-revision")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--kernel-root", type=Path, default=DEFAULT_KERNEL_ROOT)
    parser.add_argument("--gpu-devices", default="0,1")
    parser.add_argument("--allow-shared-gpus", action="store_true")
    parser.add_argument(
        "--dashboard-token",
        default=os.environ.get("SYZPILOT_E3_DASHBOARD_TOKEN", "e3-evaluator-token"),
    )
    parser.add_argument("--duration")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--cpu-pairs", default="0-1;2-3;4-5;6-7")
    args = parser.parse_args()
    lock = (
        acquire_run_lock(args.output_root)
        if args.command in {"pilot", "formal"}
        else None
    )
    try:
        {
            "setup": setup,
            "pilot": pilot,
            "formal": formal,
            "cleanup": cleanup,
        }[args.command](args)
    finally:
        if lock is not None:
            lock.close()


if __name__ == "__main__":
    main()
