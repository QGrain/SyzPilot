#!/usr/bin/env python3
"""Extract publishable report-derived waypoint PCs for benchmark cases."""

import argparse
import json
import logging
import re
from pathlib import Path

try:
    from .get_targets import (
        canonicalize_kernel_source_path,
        get_vmlinux_cache_identity,
    )
    from .waypoints_extractor import WaypointsExtractor
except ImportError:
    from get_targets import canonicalize_kernel_source_path, get_vmlinux_cache_identity
    from waypoints_extractor import WaypointsExtractor


LOGGER = logging.getLogger(__name__)
CASE_TITLE_PATTERN = re.compile(r"case_(\d+)\.title$")
RESOLUTION_POLICY = (
    "report-relative-position-closest-preceding-outer-function-pc; "
    "inline-function-and-file-context-corroboration; "
    "symbol-less-inline-source-line-fallback; "
    "explicit-nearest-instrumented-crash-caller-proxy; "
    "no-offset-lowest-address-debug-fallback"
)


def discover_case_ids(config_dir: Path) -> list[int]:
    """Return case IDs that have benchmark title metadata."""
    case_ids = []
    for title_path in config_dir.glob("case_*.title"):
        match = CASE_TITLE_PATTERN.fullmatch(title_path.name)
        if match:
            case_ids.append(int(match.group(1)))
    return sorted(set(case_ids))


def portable_target(target: str, kernel_dir: Path) -> str:
    """Convert a resolved target to a repository-portable source path."""
    func_name, location = target.split("@", 1)
    source_path, line = location.rsplit(":", 1)
    relative_path = (
        canonicalize_kernel_source_path(str(kernel_dir), source_path)
        if Path(source_path).is_absolute() else
        source_path
    )
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("target source is outside the kernel tree")
    return f"{func_name}@{relative_path}:{line}"


def extract_case(
    case_id: int,
    benchmark_dir: Path,
    kernel_root: Path,
) -> dict:
    """Extract one case and return its portable waypoint document."""
    config_dir = benchmark_dir / "configs"
    case_name = f"case_{case_id}"
    kernel_dir = kernel_root / case_name
    title_path = config_dir / f"{case_name}.title"
    report_path = config_dir / f"{case_name}.report"
    commit_path = config_dir / f"{case_name}.commit"
    vmlinux_path = kernel_dir / "vmlinux"

    missing = [
        str(path.name)
        for path in (title_path, report_path, commit_path, vmlinux_path)
        if not path.is_file()
    ]
    if missing:
        return {
            "case": case_name,
            "status": "skipped",
            "reason": f"missing required input: {', '.join(missing)}",
        }

    extractor = WaypointsExtractor(
        str(kernel_dir), str(title_path), str(report_path)
    )
    extractor.extract()
    full_targets, full_pcs = extractor.get_pcs()
    waypoints = []
    for index, (target, pc, resolution_method) in enumerate(
        zip(full_targets, full_pcs, extractor.pc_resolution_methods),
        start=1,
    ):
        waypoints.append(
            {
                "class_index": index,
                "target": portable_target(target, kernel_dir),
                "pc": f"0x{int(pc, 16) & 0xffffffff:08x}",
                "full_pc": f"0x{int(pc, 16):016x}",
                "resolution_method": resolution_method,
            }
        )

    build_identity = get_vmlinux_cache_identity(str(kernel_dir))
    if not build_identity.startswith("build-id:"):
        raise ValueError("vmlinux has no GNU ELF build ID")

    return {
        "schema_version": 2,
        "case": case_name,
        "status": "ok",
        "title": title_path.read_text(encoding="utf-8").strip(),
        "kernel_commit": commit_path.read_text(encoding="utf-8").strip(),
        "kernel_build_identity": build_identity,
        "source": "report-derived",
        "target_resolution_policy": RESOLUTION_POLICY,
        "report_target": portable_target(
            extractor.report_target, kernel_dir
        ),
        "final_target": waypoints[-1]["target"],
        "final_target_kind": extractor.final_target_kind,
        "waypoints": waypoints,
        "target_pcs": [waypoint["pc"] for waypoint in waypoints],
    }


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract report-derived waypoint PCs for benchmark cases."
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=Path("benchmark"),
        help="benchmark directory containing configs/",
    )
    parser.add_argument(
        "--kernel-root",
        type=Path,
        required=True,
        help="directory containing compiled case_N kernel trees",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="output directory (default: <benchmark-dir>/waypoints)",
    )
    parser.add_argument(
        "--cases",
        type=int,
        nargs="*",
        help="optional case IDs; default: all cases with title metadata",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.DEBUG if args.debug else logging.INFO)
    benchmark_dir = args.benchmark_dir.expanduser().resolve()
    kernel_root = args.kernel_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else benchmark_dir / "waypoints"
    )
    case_ids = sorted(set(
        args.cases or discover_case_ids(benchmark_dir / "configs")
    ))
    if args.cases and args.output_dir is None:
        parser.error("--cases requires a dedicated --output-dir")
    requested_names = {f"case_{case_id}.json" for case_id in case_ids}
    existing_names = {
        path.name for path in output_dir.glob("case_*.json")
    }
    unrelated = sorted(existing_names - requested_names)
    if args.cases and unrelated:
        parser.error(
            "subset output directory contains unrelated case JSON: "
            + ", ".join(unrelated[:5])
        )
    if not args.cases:
        for stale_name in unrelated:
            (output_dir / stale_name).unlink()

    cases = []
    for case_id in case_ids:
        LOGGER.info("Extracting case_%d", case_id)
        case_output = output_dir / f"case_{case_id}.json"
        case_output.unlink(missing_ok=True)
        try:
            result = extract_case(case_id, benchmark_dir, kernel_root)
        except Exception as error:
            LOGGER.exception("case_%d extraction failed", case_id)
            result = {
                "case": f"case_{case_id}",
                "status": "failed",
                "reason": f"{type(error).__name__}: extraction failed; see local logs",
            }
        cases.append(result)
        if result["status"] == "ok":
            write_json(case_output, result)

    summary = {
        "schema_version": 2,
        "target_resolution_policy": RESOLUTION_POLICY,
        "requested_cases": len(case_ids),
        "successful_cases": sum(case["status"] == "ok" for case in cases),
        "skipped_cases": sum(case["status"] == "skipped" for case in cases),
        "failed_cases": sum(case["status"] == "failed" for case in cases),
        "cases": cases,
    }
    write_json(output_dir / "manifest.json", summary)
    LOGGER.info(
        "Finished: %d successful, %d skipped, %d failed",
        summary["successful_cases"],
        summary["skipped_cases"],
        summary["failed_cases"],
    )
    return 1 if summary["failed_cases"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
