#!/usr/bin/env python3
"""Create a mini-benchmark manager config for the unified image."""

import argparse
import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE = PROJECT_ROOT / "artifact" / "case_36.manager.cfg"
CASES = {
    21: ("cfg80211_connect", 39821),
    25: ("f2fs_is_valid_blkaddr", 39825),
    36: ("rds_rdma_extra_size", 39836),
}
PC_LIST_PATTERN = re.compile(r"\[For SyzPilot-fuzzer:\]\s*\n(\[[^\n]+\])")
PC_PATTERN = re.compile(r"0x[0-9a-fA-F]{8}\Z")


def parse_target_pcs(output: str) -> list[str]:
    """Extract the fuzzer-ready PC list from waypoint extractor output."""
    match = PC_LIST_PATTERN.search(output)
    if match is None:
        raise ValueError("waypoint output has no fuzzer-ready PC list")
    pcs = json.loads(match.group(1))
    if (
        not isinstance(pcs, list)
        or not pcs
        or any(not isinstance(pc, str) or PC_PATTERN.fullmatch(pc) is None for pc in pcs)
        or len(pcs) != len(set(pcs))
        or any(int(pc, 16) == 0 for pc in pcs)
    ):
        raise ValueError("waypoint output contains invalid or duplicate target PCs")
    return pcs


def render_config(args: argparse.Namespace, pcs: list[str]) -> dict:
    """Populate portable runtime paths and distributed network addresses."""
    target_func, default_port = CASES[args.case]
    case_name = f"case_{args.case}"
    kernel_dir = args.kernel_dir or f"/root/kernels/{case_name}"
    run_dir = args.run_dir or f"/root/syzpilot-runs/{case_name}"
    report_path = (
        args.report_path
        or f"/root/SyzPilot/benchmark/configs/{case_name}.report"
    )
    config = json.loads(args.template.read_text())
    config["http"] = f"0.0.0.0:{args.manager_port or default_port}"
    config["workdir"] = f"{run_dir}/workdir"
    config["kernel_obj"] = kernel_dir
    config["image"] = args.image
    config["sshkey"] = args.ssh_key
    config["syzkaller"] = args.syzkaller
    config["vm"]["kernel"] = f"{kernel_dir}/arch/x86/boot/bzImage"

    syzpilot = config["SyzPilot"]
    syzpilot["task_name"] = f"{case_name}_artifact_functional"
    syzpilot["callback_ip"] = args.fuzzer_host
    syzpilot["dump_dir"] = f"{run_dir}/target_evidence"
    syzpilot["report_path"] = report_path
    syzpilot["target_func"] = target_func
    syzpilot["target_pcs"] = pcs
    syzpilot["reach_filter"]["controller"] = f"{args.brain_host}:48000"
    for field in (
        "cold_start_seed_dir",
        "cold_start_seed_catalog",
        "cold_start_auto_dependencies",
        "directed_corpus",
    ):
        syzpilot.pop(field, None)
    if args.case == 25:
        syzpilot["observe_target_coverage"] = False
        syzpilot["cold_start_seed_dir"] = (
            "/root/fuzzers/SyzPilot-fuzzer/sys/linux/test"
        )
        syzpilot["cold_start_seed_catalog"] = (
            "/root/fuzzers/SyzPilot-fuzzer/syz-manager/"
            "syzpilot_seed_catalog_linux_amd64.json"
        )
        syzpilot["cold_start_auto_dependencies"] = True
        syzpilot["directed_corpus"] = {
            "enable": True,
            "capacity": 256,
            "mutation_probability": 0.20,
            "anchor_preserve_probability": 0.95,
        }
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=int, choices=sorted(CASES), default=36)
    parser.add_argument("--waypoints-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--brain-host", required=True)
    parser.add_argument("--fuzzer-host", required=True)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--manager-port", type=int)
    parser.add_argument("--kernel-dir")
    parser.add_argument("--image", default="/root/images/image-template/disk.img")
    parser.add_argument(
        "--ssh-key", default="/root/images/image-template/disk.id_rsa"
    )
    parser.add_argument("--syzkaller", default="/root/fuzzers/SyzPilot-fuzzer")
    parser.add_argument("--run-dir")
    parser.add_argument(
        "--report-path",
        help="Brain-local path included in the registration payload",
    )
    args = parser.parse_args()

    pcs = parse_target_pcs(args.waypoints_output.read_text())
    config = render_config(args, pcs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, indent=2) + "\n")
    print(f"Wrote {len(pcs)} target PCs to {args.output}")


if __name__ == "__main__":
    main()
