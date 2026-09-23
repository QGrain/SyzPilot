#!/usr/bin/env python3

import argparse
import logging
import json
import os
import shutil
import time
import copy
import socket
import subprocess
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed


LOGGER = logging.getLogger(__name__)
LOG_FORMAT = "%(asctime)s %(levelname)s [%(processName)s] %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# VM Image Configuration
IMAGE_RELEASE = "bullseye"
# IMAGE_PATH = os.path.join(IMAGE_DIR, f"{IMAGE_RELEASE}.img")
# SSHKEY_PATH = os.path.join(IMAGE_DIR, f"{IMAGE_RELEASE}.id_rsa")

# Syzkaller Fuzzer Configuration
SYZKALLER_PATH = "/root/fuzzers/SyzPilot-fuzzer-syzkaller/"
RUN_ARTIFACT_SUFFIXES = (".cfg", ".bench.log", ".console.log")


def configure_logging():
    """Configure timestamped logs for both the main process and workers."""
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format=LOG_FORMAT,
            datefmt=LOG_DATE_FORMAT,
        )


def is_port_available(port):
    """Check if a port is available for binding"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        try:
            s.bind(("localhost", port))
            return True
        except socket.error:
            return False


def find_next_available_port(start_port):
    """Find the next available port starting from start_port"""
    port = start_port
    while port < 65535:
        if is_port_available(port):
            return port
        port += 1
    raise RuntimeError(f"No available port found starting from {start_port}")


def validate_case_name(case_name):
    """Reject case names that could escape the case output directory."""
    if not case_name or case_name in {".", ".."}:
        raise ValueError("case_name must be a non-empty relative name")
    if os.path.isabs(case_name) or os.sep in case_name:
        raise ValueError(f"invalid case_name: {case_name}")
    if os.altsep and os.altsep in case_name:
        raise ValueError(f"invalid case_name: {case_name}")


def get_existing_run_ids(case_dir):
    """Return run ids already represented by workdirs or per-run files."""
    if not os.path.isdir(case_dir):
        return []

    existing_runs = set()
    for item in os.listdir(case_dir):
        item_path = os.path.join(case_dir, item)
        if os.path.isdir(item_path) and item.isdigit():
            existing_runs.add(int(item))
            continue

        for suffix in RUN_ARTIFACT_SUFFIXES:
            if item.endswith(suffix):
                run_id = item[: -len(suffix)]
                if run_id.isdigit():
                    existing_runs.add(int(run_id))
                break

    return sorted(existing_runs)


def get_next_run_id(case_dir):
    """Get the next run_id by checking existing run artifacts."""
    if not os.path.exists(case_dir):
        os.makedirs(case_dir, exist_ok=True)
        return 1

    existing_runs = get_existing_run_ids(case_dir)

    if not existing_runs:
        return 1
    return max(existing_runs) + 1


def reset_case_dir(case_dir, work_root, case_name):
    """Remove one target case output directory before a deliberate rerun."""
    work_root_abs = os.path.abspath(work_root)
    case_dir_abs = os.path.abspath(case_dir)
    if os.path.commonpath([work_root_abs, case_dir_abs]) != work_root_abs:
        raise ValueError(f"refusing to rerun outside work root: {case_dir}")

    if os.path.exists(case_dir):
        LOGGER.warning(
            "%s: Removing existing case output directory for rerun: %s",
            case_name,
            case_dir,
        )
        shutil.rmtree(case_dir)
    os.makedirs(case_dir, exist_ok=True)


def load_template_config(template_path):
    """Load the syzkaller configuration template"""
    with open(template_path, "r") as f:
        return json.load(f)


def load_targets_csv(csv_path):
    """Load targets from CSV file"""
    # CSV file has comments starting with #, no header row
    # Format: case_name,kernel_dir,task_name,target_pc
    df = pd.read_csv(
        csv_path,
        comment="#",
        header=None,
        names=["case_name", "kernel_dir", "task_name", "target_pc"],
    )
    targets = []
    for _, row in df.iterrows():
        if pd.isna(row["case_name"]):
            raise ValueError("case_name must be a non-empty relative name")
        case_name = str(row["case_name"]).strip()
        validate_case_name(case_name)
        targets.append(
            {
                "case_name": case_name,
                "kernel_dir": row["kernel_dir"],
                "task_name": row["task_name"],
                "target_pc": row["target_pc"],
            }
        )
    return targets


def create_fuzzing_config(
    template_config,
    image_dir,
    kernel_dir,
    task_name,
    target_pc,
    workdir,
    http_port,
    dump_dir,
):
    """Create a fuzzing configuration for a specific run"""
    run_config = copy.deepcopy(template_config)

    # Fill in the configuration
    run_config["image"] = os.path.join(image_dir, f"{IMAGE_RELEASE}.img")
    run_config["sshkey"] = os.path.join(image_dir, f"{IMAGE_RELEASE}.id_rsa")
    run_config["syzkaller"] = SYZKALLER_PATH
    run_config["kernel_obj"] = kernel_dir
    run_config["workdir"] = workdir
    run_config["http"] = f"0.0.0.0:{http_port}"

    # Find bzImage in kernel_dir
    bzimage_path = os.path.join(kernel_dir, "arch/x86/boot/bzImage")
    if not os.path.exists(bzimage_path):
        # Try alternative path
        bzimage_path = os.path.join(kernel_dir, "bzImage")
        if not os.path.exists(bzimage_path):
            raise FileNotFoundError(f"bzImage not found in {kernel_dir}")

    run_config["vm"]["kernel"] = bzimage_path

    # Configure SyzPilot settings
    run_config["SyzPilot"]["dump_dir"] = dump_dir
    run_config["SyzPilot"]["task_name"] = task_name
    run_config["SyzPilot"]["target_pcs"] = [target_pc]

    return run_config


def write_config_file(config, config_path):
    """Write configuration to file"""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)


def load_cfg_overrides(overrides_path):
    """Load per-case config overrides."""
    with open(overrides_path, "r") as f:
        overrides = json.load(f)

    if not isinstance(overrides, dict):
        raise ValueError("cfg overrides must be a JSON object")

    for case_name, case_override in overrides.items():
        if not isinstance(case_name, str):
            raise ValueError("cfg override case names must be strings")
        if not isinstance(case_override, dict):
            raise ValueError(f"cfg override for {case_name} must be a JSON object")

    return overrides


def deep_update_config(config, override):
    """Recursively apply override values to config.

    This is deliberately override semantics, not append semantics. If a key is
    missing in the generated config, it is created. If a key already exists, the
    override replaces it unless both values are JSON objects, in which case the
    same rule is applied recursively. Users who want to preserve an existing
    scalar/list/string value must put the full desired value in the override.
    Keeping this rule uniform avoids hidden field-specific behavior.
    """
    for key, value in override.items():
        if isinstance(config.get(key), dict) and isinstance(value, dict):
            deep_update_config(config[key], value)
        else:
            config[key] = copy.deepcopy(value)


def flatten_override_paths(override, prefix=""):
    """Return dotted paths for logging, e.g. vm.cmdline."""
    paths = []
    for key, value in override.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict) and value:
            paths.extend(flatten_override_paths(value, path))
        else:
            paths.append(path)
    return paths


def parse_cpu_pool(cpu_pool):
    """Parse a taskset-style CPU list, e.g. 0,4,7-11."""
    cpus = []
    seen = set()

    for token in cpu_pool.split(","):
        token = token.strip()
        if not token:
            raise ValueError("CPU pool contains an empty item")

        if "-" in token:
            bounds = token.split("-")
            if len(bounds) != 2 or not bounds[0] or not bounds[1]:
                raise ValueError(f"Invalid CPU range: {token}")
            try:
                start = int(bounds[0])
                end = int(bounds[1])
            except ValueError as exc:
                raise ValueError(f"Invalid CPU range: {token}") from exc
            if start < 0 or end < 0:
                raise ValueError(f"CPU ids must be non-negative: {token}")
            if start > end:
                raise ValueError(f"CPU range start is greater than end: {token}")
            values = range(start, end + 1)
        else:
            try:
                value = int(token)
            except ValueError as exc:
                raise ValueError(f"Invalid CPU id: {token}") from exc
            if value < 0:
                raise ValueError(f"CPU ids must be non-negative: {token}")
            values = [value]

        for cpu in values:
            if cpu in seen:
                raise ValueError(f"Duplicate CPU id in pool: {cpu}")
            seen.add(cpu)
            cpus.append(cpu)

    if not cpus:
        raise ValueError("CPU pool is empty")
    return cpus


def format_cpu_set(cpus):
    """Format CPU ids as a compact taskset-compatible list."""
    if not cpus:
        return ""

    parts = []
    start = cpus[0]
    previous = cpus[0]

    for cpu in cpus[1:]:
        if cpu == previous + 1:
            previous = cpu
            continue

        if start == previous:
            parts.append(str(start))
        else:
            parts.append(f"{start}-{previous}")
        start = cpu
        previous = cpu

    if start == previous:
        parts.append(str(start))
    else:
        parts.append(f"{start}-{previous}")

    return ",".join(parts)


def get_instance_cpu_requirement(template_config):
    """Return the configured VM CPU demand for one syz-manager instance."""
    vm_config = template_config.get("vm", {})
    vm_count = int(vm_config.get("count", 1))
    vm_cpu = int(vm_config.get("cpu", 1))
    if vm_count <= 0 or vm_cpu <= 0:
        raise ValueError("Template vm.count and vm.cpu must be positive integers")
    return vm_count * vm_cpu


def allocate_cpu_sets(cpu_pool, task_count, cpus_per_instance):
    """Allocate a fixed CPU set for every fuzzer instance."""
    if cpus_per_instance <= 0:
        raise ValueError("--cpus-per-instance must be a positive integer")

    required_cpus = task_count * cpus_per_instance
    if required_cpus > len(cpu_pool):
        raise ValueError(
            "CPU pool is too small: "
            f"{task_count} tasks x {cpus_per_instance} CPUs/task = "
            f"{required_cpus} CPUs required, but pool has {len(cpu_pool)} CPUs"
        )

    cpu_sets = []
    for task_index in range(task_count):
        start = task_index * cpus_per_instance
        end = start + cpus_per_instance
        cpu_sets.append(format_cpu_set(cpu_pool[start:end]))
    return cpu_sets


def run_fuzzer_instance(
    fuzzer_bin, config_path, timeout, console_log_path, bench_log_path, cpu_set=None
):
    """Run a single fuzzer instance with output redirected to log file"""
    configure_logging()

    command = [
        fuzzer_bin,
        f"-config={config_path}",
        f"-bench={bench_log_path}",
        f"-timeout={timeout}",
    ]
    if cpu_set:
        command = ["taskset", "-c", cpu_set] + command

    LOGGER.info("Starting: %s", " ".join(command))
    LOGGER.info("Console logging to: %s", console_log_path)
    LOGGER.info("Bench logging to: %s", bench_log_path)
    if cpu_set:
        LOGGER.info("CPU set: %s", cpu_set)

    try:
        with open(console_log_path, "w") as console_log_file:
            process = subprocess.run(
                command,
                stdout=console_log_file,
                stderr=subprocess.STDOUT,  # Merge stderr into stdout
                text=True,
            )
            ret = process.returncode
    except Exception as e:
        LOGGER.error("Failed to run %s: %s", config_path, e)
        return -1

    LOGGER.info("Finished: %s (exit code: %s)", config_path, ret)
    LOGGER.info("Console log saved to: %s", console_log_path)
    LOGGER.info("Bench log saved to: %s", bench_log_path)
    return ret


def main():
    configure_logging()

    parser = argparse.ArgumentParser(
        description="Run batch syzkaller fuzzing experiments"
    )
    parser.add_argument(
        "--targets", required=True, help="Path to CSV file with target configurations"
    )
    parser.add_argument(
        "--template",
        required=True,
        help="Path to syzkaller base configuration template",
    )
    parser.add_argument(
        "--image-dir",
        required=True,
        help="Image name (will be used to construct image path)",
    )
    parser.add_argument(
        "--workdir",
        required=True,
        help="Working directory name (will be created under fuzzer path)",
    )
    parser.add_argument(
        "--start-port", type=int, default=12630, help="Start HTTP port number"
    )
    parser.add_argument(
        "--timeout", default="24h", help="Fuzzing timeout duration (e.g., 24h, 48h)"
    )
    parser.add_argument(
        "--rounds", type=int, default=3, help="New runs to prepare for each target"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum number of parallel fuzzing instances (default: total number of instances)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only generate config files without running fuzzer",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Delete existing output for target cases before preparing new runs",
    )
    parser.add_argument(
        "--cpu-pool",
        default=None,
        help="CPU ids available for fuzzing instances, e.g. 0-89 or 0,4,7-11",
    )
    parser.add_argument(
        "--cpus-per-instance",
        type=int,
        default=None,
        help="CPU ids assigned to each syz-manager instance (default: vm.count * vm.cpu when --cpu-pool is set)",
    )
    parser.add_argument(
        "--cfg-overrides",
        default=None,
        help="Path to per-case generated-config overrides JSON",
    )

    args = parser.parse_args()
    if args.rounds <= 0:
        parser.error("--rounds must be a positive integer")
    if args.max_workers is not None and args.max_workers <= 0:
        parser.error("--max-workers must be a positive integer")
    if args.cpus_per_instance is not None and args.cpus_per_instance <= 0:
        parser.error("--cpus-per-instance must be a positive integer")
    if args.cpus_per_instance is not None and not args.cpu_pool:
        parser.error("--cpus-per-instance requires --cpu-pool")
    if args.rerun and args.dry_run:
        parser.error(
            "--rerun cannot be combined with --dry-run because it deletes "
            "existing case output"
        )

    LOGGER.info("Loading template config from %s", args.template)
    template_config = load_template_config(args.template)
    try:
        instance_cpu_requirement = get_instance_cpu_requirement(template_config)
    except ValueError as exc:
        parser.error(str(exc))

    LOGGER.info("Loading targets from %s", args.targets)
    try:
        targets = load_targets_csv(args.targets)
    except ValueError as exc:
        parser.error(str(exc))
    LOGGER.info("Found %s targets", len(targets))
    expected_task_count = len(targets) * args.rounds
    target_case_names = {target["case_name"] for target in targets}

    cfg_overrides = {}
    if args.cfg_overrides:
        LOGGER.info("Loading cfg overrides from %s", args.cfg_overrides)
        try:
            cfg_overrides = load_cfg_overrides(args.cfg_overrides)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            parser.error(f"failed to load --cfg-overrides: {exc}")

        unused_override_cases = sorted(set(cfg_overrides) - target_case_names)
        for case_name in unused_override_cases:
            LOGGER.warning(
                "cfg override for %s does not match any target in this run",
                case_name,
            )

    cpu_sets = []
    cpus_per_instance = args.cpus_per_instance
    if args.cpu_pool:
        if cpus_per_instance is None:
            cpus_per_instance = instance_cpu_requirement
        try:
            cpu_pool = parse_cpu_pool(args.cpu_pool)
            cpu_sets = allocate_cpu_sets(
                cpu_pool, expected_task_count, cpus_per_instance
            )
        except ValueError as exc:
            parser.error(str(exc))

        LOGGER.info(
            "CPU pool: %s (%s CPUs)",
            format_cpu_set(cpu_pool),
            len(cpu_pool),
        )
        LOGGER.info(
            "CPU binding: %s instances x %s CPUs = %s CPUs",
            expected_task_count,
            cpus_per_instance,
            expected_task_count * cpus_per_instance,
        )
        if cpus_per_instance != instance_cpu_requirement:
            LOGGER.warning(
                "--cpus-per-instance differs from template vm.count x vm.cpu (%s)",
                instance_cpu_requirement,
            )

    # Prepare work directory structure
    work_root = os.path.join(SYZKALLER_PATH, args.workdir, "syzkaller")
    fuzzer_bin = os.path.join(SYZKALLER_PATH, "bin", "syz-manager")

    # Prepare all fuzzing tasks
    fuzzing_tasks = []
    current_port = args.start_port
    reset_cases = set()

    if args.rerun:
        LOGGER.warning(
            "Rerun mode enabled: existing output for target cases will be deleted"
        )

    for target in targets:
        case_name = target["case_name"]
        kernel_dir = target["kernel_dir"]
        task_name = target["task_name"]
        target_pc = target["target_pc"]

        LOGGER.info("Preparing %s: %s", case_name, task_name)

        # Get case directory and next run_id
        case_dir = os.path.join(work_root, case_name)
        if args.rerun and case_name not in reset_cases:
            try:
                reset_case_dir(case_dir, work_root, case_name)
            except ValueError as exc:
                parser.error(str(exc))
            reset_cases.add(case_name)

        next_run_id = get_next_run_id(case_dir)

        LOGGER.info("%s: Starting from run_id %s", case_name, next_run_id)

        # Create configs for each round
        for round_idx in range(args.rounds):
            run_id = next_run_id + round_idx

            # Allocate workdir and port
            run_workdir = os.path.join(case_dir, str(run_id))
            run_dump_dir = os.path.join(run_workdir, "dump")
            config_path = os.path.join(case_dir, f"{run_id}.cfg")

            # Find next available port
            http_port = find_next_available_port(current_port)
            current_port = http_port + 1
            task_cpu_set = cpu_sets[len(fuzzing_tasks)] if cpu_sets else None

            if task_cpu_set:
                LOGGER.info(
                    "%s/run%s: workdir=%s, port=%s, cpus=%s",
                    case_name,
                    run_id,
                    run_workdir,
                    http_port,
                    task_cpu_set,
                )
            else:
                LOGGER.info(
                    "%s/run%s: workdir=%s, port=%s",
                    case_name,
                    run_id,
                    run_workdir,
                    http_port,
                )

            # Create configuration
            run_config = create_fuzzing_config(
                template_config,
                args.image_dir,
                kernel_dir,
                task_name,
                target_pc,
                run_workdir,
                http_port,
                run_dump_dir,
            )
            case_override = cfg_overrides.get(case_name)
            if case_override:
                override_paths = ", ".join(flatten_override_paths(case_override))
                LOGGER.info(
                    "%s/run%s: Applying cfg override: %s",
                    case_name,
                    run_id,
                    override_paths,
                )
                deep_update_config(run_config, case_override)

            # Write configuration file
            write_config_file(run_config, config_path)
            LOGGER.info(
                "%s/run%s: Config written to %s", case_name, run_id, config_path
            )

            # Add to task list
            console_log_path = config_path.replace(".cfg", ".console.log")
            bench_log_path = config_path.replace(".cfg", ".bench.log")
            fuzzing_tasks.append(
                (
                    fuzzer_bin,
                    config_path,
                    args.timeout,
                    console_log_path,
                    bench_log_path,
                    task_cpu_set,
                )
            )

    LOGGER.info("Total fuzzing tasks prepared: %s", len(fuzzing_tasks))

    # Set max_workers to total tasks if not specified
    max_workers = (
        args.max_workers if args.max_workers is not None else len(fuzzing_tasks)
    )
    total_cpu_cores = len(fuzzing_tasks) * instance_cpu_requirement

    LOGGER.info(
        "Parallelism: %s workers for %s instances",
        max_workers,
        len(fuzzing_tasks),
    )
    LOGGER.info(
        "CPU requirement: %s instances x %s cores = %s total cores",
        len(fuzzing_tasks),
        instance_cpu_requirement,
        total_cpu_cores,
    )
    if cpu_sets:
        LOGGER.info(
            "CPU affinity: enabled, %s CPUs per instance", cpus_per_instance
        )

    if args.dry_run:
        LOGGER.info("Dry run mode - config files generated but fuzzing not started")
        return
    if not fuzzing_tasks:
        LOGGER.info("No fuzzing tasks to run")
        return

    LOGGER.info("Starting fuzzing...")

    # Run fuzzing tasks in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for task in fuzzing_tasks:
            futures.append(executor.submit(run_fuzzer_instance, *task))
            time.sleep(2)  # Stagger the starts slightly

        # Wait for all tasks to complete
        for future in as_completed(futures):
            try:
                result = future.result()
                LOGGER.info("Task completed with result: %s", result)
            except Exception as e:
                LOGGER.error("Task failed with exception: %s", e)

    LOGGER.info("All fuzzing tasks completed")


if __name__ == "__main__":
    main()


# python3 run_syzkaller.py \
#   --targets batch/batch_3.csv \
#   --template syzkaller_base.cfg \
#   --image-dir image-1 \
#   --workdir exp \
#   --rounds 3 \
#   --cpu-pool 0-89 \
#   --cpus-per-instance 2 \
#   --cfg-overrides cfg_overrides.json
