#!/usr/bin/env python3
"""Validate and summarize all predeclared E3 runs without selecting outcomes."""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import statistics


DEFAULT_ROOT = Path(
    os.environ.get("SYZPILOT_E3_ROOT", str(Path.cwd() / "run/e3-results"))
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_TIME = re.compile(r"^(\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})")
PREDICTOR_NOTICE = re.compile(r"Predictor enabled: model=\S+ version=([^,\s]+)")
TARGET_TITLES = {
    21: "WARNING in cfg80211_connect",
    36: "WARNING in rds_rdma_extra_size",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_e3_tools(root: Path, manifest: dict) -> None:
    expected = manifest.get("e3_tool_sha256", {})
    if not expected:
        raise RuntimeError("manifest does not freeze the E3 tools")
    for name, expected_hash in expected.items():
        current = PROJECT_ROOT / name
        frozen = root / "provenance/e3-tools" / name
        if not current.is_file() or sha256_file(current) != expected_hash:
            raise RuntimeError(f"current E3 tool differs from setup: {name}")
        if not frozen.is_file() or sha256_file(frozen) != expected_hash:
            raise RuntimeError(f"frozen E3 tool differs from setup: {name}")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def load_bench(path: Path) -> list[dict]:
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
        raise ValueError(f"empty bench log: {path}")
    return rows


def first_positive_interval(rows: list[dict], key: str) -> list[int] | None:
    previous = 0
    for row in rows:
        uptime = int(row.get("uptime", 0))
        if int(row.get(key, 0)) > 0:
            return [previous, uptime]
        previous = uptime
    return None


def crash_titles(workdir: Path) -> dict[str, int]:
    titles = []
    for path in workdir.glob("crashes/*/description"):
        title = " ".join(path.read_text(errors="replace").split())
        if title:
            titles.append(title)
    return dict(sorted(collections.Counter(titles).items()))


def log_summary(path: Path, target_title: str) -> dict:
    start = None
    target_crash_seconds = None
    model_versions = []
    first_model_ready_seconds = None
    prediction_failures = 0
    for line in path.read_text(errors="replace").splitlines():
        timestamp = None
        match = LOG_TIME.match(line)
        if match:
            timestamp = dt.datetime.strptime(match.group(1), "%Y/%m/%d %H:%M:%S")
            if start is None:
                start = timestamp
        crash_match = re.search(r"crash:\s+(.+?)\s*$", line)
        if (
            crash_match
            and " ".join(crash_match.group(1).split()) == target_title
            and timestamp
            and start
        ):
            target_crash_seconds = target_crash_seconds or (
                timestamp - start
            ).total_seconds()
        notice = PREDICTOR_NOTICE.search(line)
        if notice:
            model_versions.append(notice.group(1))
            if timestamp and start and first_model_ready_seconds is None:
                first_model_ready_seconds = (timestamp - start).total_seconds()
        if "predictions failed" in line:
            prediction_failures += 1
    return {
        "target_crash_seconds": target_crash_seconds,
        "model_versions": list(dict.fromkeys(model_versions)),
        "first_model_ready_seconds": first_model_ready_seconds,
        "prediction_failure_events": prediction_failures,
    }


def parse_exit(path: Path) -> dict:
    values = {}
    for line in path.read_text().splitlines():
        key, value = line.split("=", 1)
        values[key] = value
    values["elapsed_seconds"] = (
        int(values["ended_epoch"]) - int(values["started_epoch"])
    )
    values["exit_code"] = int(values["exit_code"])
    return values


def brain_lifecycle(
    root: Path,
    phase: str,
    repetition: int,
    attempt: int,
    task_name: str,
) -> dict:
    path = root / "runs" / phase / f"brain_r{repetition}_a{attempt}.jsonl"
    versions = []
    stages = []
    first_seen = None
    if not path.is_file():
        return {"versions": [], "max_stage": 0, "first_model_snapshot": None}
    for snapshot in load_jsonl(path):
        for task in snapshot.get("tasks", []):
            if task.get("task_name") != task_name:
                continue
            model = task.get("model", {})
            if model.get("version"):
                versions.append(str(model["version"]))
                first_seen = first_seen or snapshot["timestamp"]
            if model.get("stage") is not None:
                stages.append(int(model["stage"]))
    return {
        "versions": list(dict.fromkeys(versions)),
        "max_stage": max(stages, default=0),
        "first_model_snapshot": first_seen,
    }


def validate_config(config: dict, manifest: dict, case: int, arm: str) -> list[str]:
    errors = []
    protocol = manifest["protocol"]
    vm = config.get("vm", {})
    expected_common = {
        "procs": protocol["procs"],
        "reproduce": protocol["reproduce"],
    }
    for field, expected in expected_common.items():
        if config.get(field) != expected:
            errors.append(f"{field}={config.get(field)!r}, expected {expected!r}")
    for field, expected in (
        ("count", protocol["vm_count"]),
        ("cpu", protocol["vm_cpu"]),
        ("mem", protocol["vm_mem_mib"]),
    ):
        if vm.get(field) != expected:
            errors.append(f"vm.{field}={vm.get(field)!r}, expected {expected!r}")

    syzpilot = config.get("SyzPilot", {})
    if syzpilot.get("target_pcs") != manifest["target_pcs"][f"case_{case}"]:
        errors.append("target_pcs differ from the frozen manifest")
    for field in ("enable_dump", "persist_target_evidence", "enable_diverse_cover"):
        if syzpilot.get(field, False):
            errors.append(f"{field} must be false")

    reach = syzpilot.get("reach_filter", {})
    directed = syzpilot.get("directed_corpus", {})
    if arm == "blank":
        if reach.get("enable"):
            errors.append("Blank reach_filter must be disabled")
        if not syzpilot.get("observe_target_coverage"):
            errors.append("Blank compact target observation must be enabled")
        if directed.get("enable"):
            errors.append("Blank directed corpus must be disabled")
        for field in (
            "enable_online_guidance",
            "enable_sequence_guidance",
            "enable_attribution_guidance",
            "allow_standalone_guidance",
        ):
            if syzpilot.get(field, False):
                errors.append(f"Blank {field} must be false")
    else:
        if not reach.get("enable"):
            errors.append("Full reach_filter must be enabled")
        if syzpilot.get("observe_target_coverage"):
            errors.append("Full must use reach-filter target observation semantics")
        if not directed.get("enable"):
            errors.append("Full directed corpus must be enabled")
        for field in (
            "enable_online_guidance",
            "enable_sequence_guidance",
            "enable_attribution_guidance",
        ):
            if not syzpilot.get(field):
                errors.append(f"Full {field} must be enabled")
    return errors


def analyze_run(root: Path, manifest: dict, run_root: Path) -> dict:
    match = re.fullmatch(
        r"case(\d+)_r(\d+)_a(\d+)_(blank|full)", run_root.name
    )
    if match is None:
        raise ValueError(f"unexpected run directory: {run_root}")
    case = int(match.group(1))
    repetition = int(match.group(2))
    attempt = int(match.group(3))
    arm = match.group(4)
    config = json.loads((run_root / "manager.cfg").read_text())
    exit_data = parse_exit(run_root / "manager.exit")
    bench_path = run_root / f"{run_root.name}.bench.log"
    rows = load_bench(bench_path)
    final = rows[-1]
    final_pc = config["SyzPilot"]["target_pcs"][-1]
    hit_key = f"reachability {final_pc}"
    missing_metrics = [
        pc
        for pc in manifest["target_pcs"][f"case_{case}"]
        if f"reachability {pc}" not in final
    ]
    if missing_metrics:
        raise ValueError(
            f"{run_root.name} lacks target observation metrics: {missing_metrics}"
        )
    executions = int(final.get("exec total", 0))
    final_hits = int(final.get(hit_key, 0))
    titles = crash_titles(run_root / "workdir")
    logs = log_summary(run_root / "fuzzer.log", TARGET_TITLES[case])
    target_crash = TARGET_TITLES[case] in titles
    lifecycle = brain_lifecycle(
        root,
        "formal",
        repetition,
        attempt,
        config["SyzPilot"]["task_name"],
    )
    versions = list(
        dict.fromkeys(logs["model_versions"] + lifecycle["versions"])
    )
    rfilter_filtered = int(final.get("rfilter filtered total", 0))
    return {
        "name": run_root.name,
        "case": case,
        "arm": arm,
        "repetition": repetition,
        "attempt": attempt,
        "config_errors": validate_config(config, manifest, case, arm),
        "exit": exit_data,
        "uptime_seconds": int(final.get("uptime", 0)),
        "executions": executions,
        "executions_per_second": (
            executions / int(final["uptime"]) if int(final.get("uptime", 0)) else 0
        ),
        "final_pc": final_pc,
        "final_hits": final_hits,
        "final_hits_per_execution": final_hits / executions if executions else 0,
        "first_final_interval_seconds": first_positive_interval(rows, hit_key),
        "target_crash": target_crash,
        "target_crash_seconds": logs["target_crash_seconds"],
        "target_crash_time_is_interval_censored": (
            target_crash and logs["target_crash_seconds"] is None
        ),
        "crash_titles": titles,
        "vm_restarts": int(final.get("vm restarts", 0)),
        "model_versions": versions,
        "first_model_ready_seconds": logs["first_model_ready_seconds"],
        "max_curriculum_stage": lifecycle["max_stage"],
        "rfilter_filtered_total": rfilter_filtered,
        "prediction_failure_events": logs["prediction_failure_events"],
        # The Fuzzer logs "Predictor enabled" only after its live inference
        # probe succeeds, before installing the ReachFilterSource.
        "model_inference_observed": bool(logs["model_versions"]),
        "reachability": {
            key.removeprefix("reachability "): int(value)
            for key, value in final.items()
            if key.startswith("reachability ")
        },
    }


def validate_matrix(
    root: Path, manifest: dict
) -> tuple[list[dict], list[str], list[str]]:
    errors = []
    warnings = []
    expected = {
        (entry["case"], entry["arm"], entry["repetition"])
        for entry in manifest["formal_matrix"]
    }
    selection_path = root / "runs/formal/selected_attempts.json"
    if not selection_path.is_file():
        return [], ["missing selected_attempts.json"], []
    selected = {
        int(repetition): int(attempt)
        for repetition, attempt in json.loads(selection_path.read_text()).items()
    }
    if set(selected) != {1, 2, 3}:
        errors.append(f"selected attempts do not cover repetitions 1--3: {selected}")

    records = []
    actual = set()
    formal_root = root / "runs/formal"
    for run_root in sorted(formal_root.glob("case*_r*_*")):
        if not run_root.is_dir():
            continue
        match = re.fullmatch(
            r"case(\d+)_r(\d+)_a(\d+)_(blank|full)", run_root.name
        )
        if match is None:
            continue
        repetition = int(match.group(2))
        attempt = int(match.group(3))
        if selected.get(repetition) != attempt:
            continue
        record = analyze_run(root, manifest, run_root)
        key = (record["case"], record["arm"], record["repetition"])
        if key in actual:
            errors.append(f"duplicate run matrix entry: {key}")
        actual.add(key)
        records.append(record)
    if actual != expected:
        errors.append(
            f"matrix mismatch: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )

    for repetition in range(1, 4):
        attempt = selected.get(repetition, 0)
        batch_path = formal_root / f"batch_r{repetition}_a{attempt}.json"
        if not batch_path.is_file():
            errors.append(f"missing {batch_path.name}")
        elif json.loads(batch_path.read_text()).get("status") != "completed":
            errors.append(f"{batch_path.name} is not completed")
        gpu_path = formal_root / f"gpu_r{repetition}_a{attempt}.jsonl"
        if not gpu_path.is_file():
            errors.append(f"missing {gpu_path.name}")
        else:
            snapshots = load_jsonl(gpu_path)
            if any(row.get("external_compute_apps") for row in snapshots):
                errors.append(
                    f"external GPU process observed in repetition {repetition}"
                )
            if any(row.get("vanished_compute_apps") for row in snapshots):
                warnings.append(
                    "short-lived GPU process evidence requires manual review in "
                    f"repetition {repetition}"
                )

    cpus_by_rep = collections.defaultdict(list)
    ports_by_rep = collections.defaultdict(list)
    for repetition in range(1, 4):
        attempt = selected.get(repetition, 0)
        batch_path = formal_root / f"batch_r{repetition}_a{attempt}.json"
        if not batch_path.is_file():
            continue
        for spec in json.loads(batch_path.read_text()).get("runs", []):
            cpus_by_rep[repetition].append(spec["cpus"])
            ports_by_rep[repetition].append(spec["port"])
    for repetition in range(1, 4):
        if len(set(cpus_by_rep[repetition])) != 4:
            errors.append(f"CPU allocation is not independent in repetition {repetition}")
        if len(set(ports_by_rep[repetition])) != 4:
            errors.append(f"manager ports are not unique in repetition {repetition}")
    return records, errors, warnings


def summarize(
    records: list[dict], matrix_errors: list[str], matrix_warnings: list[str]
) -> dict:
    groups = {}
    for case in (21, 36):
        for arm in ("blank", "full"):
            runs = [
                record
                for record in records
                if record["case"] == case and record["arm"] == arm
            ]
            rates = [record["final_hits_per_execution"] for record in runs]
            groups[f"case_{case}/{arm}"] = {
                "runs": len(runs),
                "final_hit_runs": sum(record["final_hits"] > 0 for record in runs),
                "target_crash_runs": sum(record["target_crash"] for record in runs),
                "total_final_hits": sum(record["final_hits"] for record in runs),
                "total_executions": sum(record["executions"] for record in runs),
                "aggregate_final_hits_per_execution": (
                    sum(record["final_hits"] for record in runs)
                    / sum(record["executions"] for record in runs)
                    if sum(record["executions"] for record in runs) else 0
                ),
                "median_run_final_hits_per_execution": (
                    statistics.median(rates) if rates else 0
                ),
                "model_active_runs": sum(
                    record["model_inference_observed"] for record in runs
                ),
                "model_versions": sum(len(record["model_versions"]) for record in runs),
            }

    pairs = []
    for case in (21, 36):
        for repetition in range(1, 4):
            by_arm = {
                record["arm"]: record
                for record in records
                if record["case"] == case
                and record["repetition"] == repetition
            }
            if set(by_arm) != {"blank", "full"}:
                continue
            blank, full = by_arm["blank"], by_arm["full"]
            pairs.append({
                "case": case,
                "repetition": repetition,
                "blank_final_hits": blank["final_hits"],
                "full_final_hits": full["final_hits"],
                "full_final_hits_higher": (
                    full["final_hits"] > blank["final_hits"]
                ),
                "blank_rate": blank["final_hits_per_execution"],
                "full_rate": full["final_hits_per_execution"],
                "rate_difference": (
                    full["final_hits_per_execution"]
                    - blank["final_hits_per_execution"]
                ),
                "full_rate_higher": (
                    full["final_hits_per_execution"]
                    > blank["final_hits_per_execution"]
                ),
                "blank_target_crash": blank["target_crash"],
                "full_target_crash": full["target_crash"],
            })

    run_errors = {
        record["name"]: record["config_errors"]
        for record in records
        if record["config_errors"]
    }
    abnormal = [
        record["name"]
        for record in records
        if record["exit"]["exit_code"] != 0
        or record["exit"]["elapsed_seconds"] < 4 * 3600 - 300
    ]
    case36_pairs = [pair for pair in pairs if pair["case"] == 36]
    full_model_active = sum(
        record["model_inference_observed"]
        for record in records
        if record["arm"] == "full"
    )
    model_active_per_case = {
        case: any(
            record["model_inference_observed"]
            for record in records
            if record["case"] == case and record["arm"] == "full"
        )
        for case in (21, 36)
    }
    full_crashes = sum(
        record["target_crash"] for record in records if record["arm"] == "full"
    )
    blank_crashes = sum(
        record["target_crash"] for record in records if record["arm"] == "blank"
    )
    frozen_runs_valid = not matrix_errors and not run_errors and not abnormal
    model_lifecycle_supported = (
        full_model_active >= 4 and all(model_active_per_case.values())
    )
    case36_reachability_supported = (
        groups.get("case_36/full", {}).get("final_hit_runs", 0)
        > groups.get("case_36/blank", {}).get("final_hit_runs", 0)
        and sum(pair["full_final_hits_higher"] for pair in case36_pairs) >= 2
    )
    bug_reproduction_supported = full_crashes > blank_crashes
    return {
        "groups": groups,
        "paired_runs": pairs,
        "validity": {
            "matrix_errors": matrix_errors,
            "manual_review_warnings": matrix_warnings,
            "config_errors": run_errors,
            "abnormal_runs": abnormal,
        },
        "predeclared_checks": {
            "all_frozen_runs_valid": frozen_runs_valid,
            "online_model_lifecycle_observed": model_lifecycle_supported,
            "case36_reachability_direction_reproduced": (
                case36_reachability_supported
            ),
            "combined_bug_reproduction_direction_reproduced": (
                bug_reproduction_supported
            ),
        },
        "claim_boundary": (
            "reachability_and_bug_reproduction"
            if frozen_runs_valid
            and model_lifecycle_supported
            and case36_reachability_supported
            and bug_reproduction_supported
            else "reachability_only"
            if frozen_runs_valid
            and model_lifecycle_supported
            and case36_reachability_supported
            else "not_supported"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    manifest = json.loads((args.root / "manifest.json").read_text())
    verify_e3_tools(args.root, manifest)
    records, matrix_errors, matrix_warnings = validate_matrix(args.root, manifest)
    result = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "manifest": manifest,
        "records": records,
        "formal_summary": summarize(records, matrix_errors, matrix_warnings),
    }
    output = args.root / "results.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["formal_summary"], indent=2, sort_keys=True))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
