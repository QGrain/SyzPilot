#!/usr/bin/env python3
"""Fill the example manager config with PCs from a freshly built kernel."""

import argparse
import json
import re
from pathlib import Path


PC_LIST_PATTERN = re.compile(r"\[For SyzPilot-fuzzer:\]\s*\n(\[[^\n]+\])")
PC_PATTERN = re.compile(r"0x[0-9a-fA-F]{8}\Z")


def read_target_pcs(output: str) -> list[str]:
    """Parse the fuzzer-ready PC list printed by waypoints_extractor.py."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--waypoints-output", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pcs = read_target_pcs(args.waypoints_output.read_text())
    config = json.loads(args.template.read_text())
    config["SyzPilot"]["target_pcs"] = pcs
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, indent=2) + "\n")
    print(f"Wrote {len(pcs)} target PCs to {args.output}")


if __name__ == "__main__":
    main()
