#!/usr/bin/env python3
"""Prepare one mini-benchmark manager config using PCs from its kernel build."""

import argparse
import json
import re
from pathlib import Path


CASES = {
    21: ("cfg80211_connect", 39821),
    25: ("f2fs_is_valid_blkaddr", 39825),
    36: ("rds_rdma_extra_size", 39836),
}
CASE_25_GENERIC_SEED_DIR = "/root/SyzPilot-fuzzer/sys/linux/test"
CASE_25_GENERIC_SEED_CATALOG = (
    "/root/SyzPilot-fuzzer/syz-manager/syzpilot_seed_catalog_linux_amd64.json"
)
PC_LIST_PATTERN = re.compile(r"\[For SyzPilot-fuzzer:\]\s*\n(\[[^\n]+\])")
PC_PATTERN = re.compile(r"0x[0-9a-fA-F]{8}\Z")
DEFAULT_TEMPLATE = (
    Path(__file__).resolve().parents[1] / "artifact" / "case_36.manager.cfg"
)


def parse_target_pcs(output: str) -> list[str]:
    """Extract the fuzzer-ready list printed by waypoints_extractor.py."""
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


def render_config(case_id: int, pcs: list[str], template: dict) -> dict:
    """Adapt the reviewed one-guest config without reusing its kernel PCs."""
    target_func, http_port = CASES[case_id]
    case = f"case_{case_id}"
    config = json.loads(json.dumps(template))
    config["http"] = f"127.0.0.1:{http_port}"
    config["workdir"] = f"/artifact_runs/{case}/workdir"
    config["kernel_obj"] = f"/artifact/assets/kernels/{case}"
    config["image"] = "/artifact/assets/guest/bullseye.img"
    config["sshkey"] = "/artifact/assets/guest/bullseye.id_rsa"
    config["vm"]["kernel"] = f"/artifact/assets/kernels/{case}/arch/x86/boot/bzImage"
    syzpilot = config["SyzPilot"]
    syzpilot["task_name"] = f"{case}_mini_functional"
    syzpilot["dump_dir"] = f"/artifact_runs/{case}/target_evidence"
    syzpilot["report_path"] = f"/artifact/assets/{case}/configs/{case}.report"
    syzpilot["target_func"] = target_func
    syzpilot["target_pcs"] = pcs
    syzpilot.pop("cold_start_seed_dir", None)
    syzpilot.pop("cold_start_seed_catalog", None)
    syzpilot.pop("cold_start_auto_dependencies", None)
    syzpilot.pop("directed_corpus", None)
    if case_id == 25:
        # Directed scheduling is an active treatment, not an observation-only
        # control. Keep the two modes mutually exclusive as required by the
        # Fuzzer configuration contract.
        syzpilot["observe_target_coverage"] = False
        syzpilot["cold_start_seed_dir"] = CASE_25_GENERIC_SEED_DIR
        syzpilot["cold_start_seed_catalog"] = CASE_25_GENERIC_SEED_CATALOG
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
    parser.add_argument("--case", type=int, required=True, choices=sorted(CASES))
    parser.add_argument("--waypoints-output", type=Path, required=True)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pcs = parse_target_pcs(args.waypoints_output.read_text())
    template = json.loads(args.template.read_text())
    config = render_config(args.case, pcs, template)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, indent=2) + "\n")
    print(f"Wrote {len(pcs)} target PCs for case_{args.case} to {args.output}")


if __name__ == "__main__":
    main()
