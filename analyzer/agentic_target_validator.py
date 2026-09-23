#!/usr/bin/env python3
"""Resolve one agentic configured target with the canonical PC resolver."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path

if __package__:
    from .evaluate_waypoint_quality import resolve_agentic_chain
    from .waypoint_schema import AgenticTargetResolution
else:
    from evaluate_waypoint_quality import resolve_agentic_chain
    from waypoint_schema import AgenticTargetResolution


class TargetUnresolvable(ValueError):
    """The resolver ran correctly but found no instrumentation for the target."""


def resolve_target(kernel_dir: Path, target: str) -> AgenticTargetResolution:
    diagnostics = io.StringIO()
    with contextlib.redirect_stdout(diagnostics):
        resolved, pcs64, pcs32, errors = resolve_agentic_chain(
            str(kernel_dir), [target]
        )
    if errors[0]:
        raise TargetUnresolvable(errors[0])
    return AgenticTargetResolution(
        proposed_target=target,
        resolved_target=resolved[0],
        pc64=pcs64[0],
        pc32=pcs32[0],
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Resolve an agentic configured target to a nonzero kernel PC."
    )
    parser.add_argument("--kernel-dir", required=True, type=Path)
    parser.add_argument("--target", required=True)
    args = parser.parse_args()
    try:
        resolution = resolve_target(args.kernel_dir, args.target)
    except TargetUnresolvable as exc:
        print(
            json.dumps(
                {
                    "status": "unresolvable",
                    "error": f"{type(exc).__name__}: {exc}",
                },
                sort_keys=True,
            )
        )
        return 2
    except (Exception, SystemExit) as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                },
                sort_keys=True,
            )
        )
        return 1
    print(
        json.dumps(
            {"status": "ok", "resolution": resolution.model_dump()},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
