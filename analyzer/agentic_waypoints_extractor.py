#!/usr/bin/env python3
"""Extract report-grounded waypoint candidates with isolated Codex threads."""

from __future__ import annotations

import argparse
import asyncio
import csv
import importlib.metadata
import json
import os
import signal
from pathlib import Path
from typing import Any
from pydantic import ValidationError

if __package__:
    from .waypoint_schema import (
        AgenticExtractionDecision,
        AgenticExtractionRecord,
        AgenticTargetResolution,
    )
else:
    from waypoint_schema import (
        AgenticExtractionDecision,
        AgenticExtractionRecord,
        AgenticTargetResolution,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT = Path(__file__).with_name("prompts") / "agentic_waypoint_extractor.md"


def load_benchmark(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def parse_case_ids(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def case_request(
    row: dict[str, str], title_path: Path, report_path: Path, kernel_dir: Path
) -> str:
    return "\n".join(
        [
            f"Analyze benchmark case {row['ID']}.",
            f"Title metadata: {row.get('Title', '')}",
            f"Configured Bug Position: {row.get('Bug Position', '')}",
            f"Title file: {title_path}",
            f"Report file: {report_path}",
            f"Kernel directory: {kernel_dir}",
            "Inspect the actual files with local tools before returning the schema.",
        ]
    )


def target_repair_request(target: str, error: str) -> str:
    return "\n".join(
        [
            "The canonical instrumentation resolver rejected the first configured target:",
            f"Rejected target: {target}",
            f"Resolver error: {error}",
            "Return the complete decision again with a resolvable configured target at index 0.",
            "For an inlined Bug Position, use the observable outer function's own report-backed callsite line, not the inlined callee's source line paired with the outer function name.",
            "Keep the configured target's behavior and source evidence; make only evidence-supported chain changes.",
        ]
    )


async def resolve_configured_target(
    python: Path,
    kernel_dir: Path,
    target: str,
    timeout_seconds: float,
) -> tuple[AgenticTargetResolution | None, str, bool]:
    """Return resolution, error, and whether semantic repair is appropriate."""
    process = await asyncio.create_subprocess_exec(
        str(python),
        "-m",
        "analyzer.agentic_target_validator",
        "--kernel-dir",
        str(kernel_dir),
        "--target",
        target,
        cwd=str(PROJECT_ROOT),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    communication = asyncio.create_task(process.communicate())
    try:
        stdout, stderr = await asyncio.wait_for(
            asyncio.shield(communication), timeout=timeout_seconds
        )
    except asyncio.TimeoutError:
        await terminate_process_group(process, process.pid, communication)
        return (
            None,
            f"resolver timed out after {timeout_seconds:g} seconds",
            False,
        )
    except asyncio.CancelledError:
        await terminate_process_group(process, process.pid, communication)
        raise
    output = stdout.decode(errors="replace").strip()
    diagnostics = stderr.decode(errors="replace").strip()
    try:
        payload = json.loads(output)
    except json.JSONDecodeError:
        return (
            None,
            f"resolver returned invalid JSON (exit={process.returncode}): "
            f"stdout={output!r}, stderr={diagnostics!r}",
            False,
        )
    status = payload.get("status")
    if status == "unresolvable":
        if process.returncode != 2:
            return (
                None,
                "resolver protocol mismatch: unresolvable status requires "
                f"exit 2, got {process.returncode}",
                False,
            )
        return (
            None,
            str(payload.get("error") or "target is unresolvable"),
            True,
        )
    if process.returncode != 0 or status != "ok":
        return (
            None,
            str(payload.get("error") or diagnostics or "unknown error"),
            False,
        )
    try:
        return (
            AgenticTargetResolution.model_validate(payload["resolution"]),
            "",
            False,
        )
    except (KeyError, ValidationError) as exc:
        return None, f"invalid resolver payload: {exc}", False


async def terminate_process_group(
    process: asyncio.subprocess.Process,
    process_group_id: int,
    communication: asyncio.Task[tuple[bytes, bytes]],
    grace_seconds: float = 5.0,
) -> None:
    """Terminate a resolver and reap its whole subprocess group."""
    leader_wait = (
        asyncio.create_task(process.wait())
        if process.returncode is None
        else None
    )
    await signal_process_group(process_group_id, signal.SIGTERM)
    if not await wait_for_process_group_exit(process_group_id, grace_seconds):
        await signal_process_group(process_group_id, signal.SIGKILL)
        await wait_for_process_group_exit(process_group_id, grace_seconds)
    if leader_wait is not None:
        await leader_wait
    try:
        await asyncio.wait_for(
            asyncio.shield(communication), timeout=grace_seconds
        )
    except asyncio.TimeoutError:
        communication.cancel()
        await asyncio.gather(communication, return_exceptions=True)


async def signal_process_group(process_group_id: int, sig: signal.Signals) -> None:
    try:
        os.killpg(process_group_id, sig)
    except ProcessLookupError:
        return


def process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


async def wait_for_process_group_exit(
    process_group_id: int, timeout_seconds: float
) -> bool:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_seconds
    while process_group_exists(process_group_id):
        if loop.time() >= deadline:
            return False
        await asyncio.sleep(min(0.05, max(deadline - loop.time(), 0.0)))
    return True


def require_present_input_status(decision: AgenticExtractionDecision) -> None:
    """Reject an agent's missing-input claim after the orchestrator found inputs."""
    if decision.status == "missing_input":
        raise RuntimeError(
            "agent returned missing_input although title and report files exist"
        )


async def validate_and_repair_configured_target(
    thread: Any,
    decision: AgenticExtractionDecision,
    resolver_python: Path,
    kernel_dir: Path,
    target_resolution_retries: int,
    resolver_timeout_seconds: float,
    effort: Any,
    sandbox: Any,
    approval_mode: Any,
) -> tuple[AgenticExtractionDecision, AgenticTargetResolution | None]:
    """Resolve the first target and request bounded evidence-preserving repairs."""
    require_present_input_status(decision)
    target_resolution = None
    for repair_attempt in range(target_resolution_retries + 1):
        if decision.status != "ok":
            break
        configured_target = decision.waypoints_target_to_entry[0].target
        (
            target_resolution,
            resolution_error,
            semantic_repair_allowed,
        ) = await resolve_configured_target(
            resolver_python,
            kernel_dir,
            configured_target,
            resolver_timeout_seconds,
        )
        if target_resolution is not None:
            break
        if not semantic_repair_allowed:
            raise RuntimeError(
                f"configured target resolver infrastructure failure: {resolution_error}"
            )
        if repair_attempt == target_resolution_retries:
            raise RuntimeError(
                "configured target remains unresolved after "
                f"{repair_attempt + 1} attempts: {resolution_error}"
            )
        repair = await thread.run(
            target_repair_request(configured_target, resolution_error),
            effort=effort,
            sandbox=sandbox,
            approval_mode=approval_mode,
            output_schema=AgenticExtractionDecision.model_json_schema(),
        )
        if repair.final_response is None:
            raise RuntimeError("configured target repair returned no response")
        decision = AgenticExtractionDecision.model_validate_json(
            repair.final_response
        )
        require_present_input_status(decision)
    if decision.status == "ok" and target_resolution is None:
        raise RuntimeError(
            "successful extraction lacks configured target resolution"
        )
    return decision, target_resolution


def load_existing(path: Path) -> dict[str, AgenticExtractionRecord]:
    if not path.is_file():
        return {}
    records: dict[str, AgenticExtractionRecord] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if line.strip():
            try:
                record = AgenticExtractionRecord.model_validate_json(line)
            except (ValidationError, json.JSONDecodeError, TypeError):
                print(
                    f"Ignoring incompatible cached agentic record "
                    f"at {path}:{line_number}",
                    flush=True,
                )
                continue
            records[record.case_id] = record
    return records


def write_records(path: Path, records: dict[str, AgenticExtractionRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.writing")
    with temporary.open("w", encoding="utf-8") as stream:
        for case_id in sorted(records, key=int):
            stream.write(records[case_id].model_dump_json() + "\n")
    os.replace(temporary, path)


def cached_extraction_matches(
    cached: AgenticExtractionRecord | None,
    row: dict[str, str],
    inputs_present: bool,
    model: str,
    effort: str,
) -> bool:
    if cached is None or cached.decision.status == "failed":
        return False
    if cached.title != row.get("Title", ""):
        return False
    if cached.bug_position != row.get("Bug Position", ""):
        return False
    if cached.model != model or cached.effort != effort:
        return False
    if cached.decision.status == "missing_input":
        return not inputs_present
    return inputs_present


async def run_all(args: argparse.Namespace) -> dict[str, AgenticExtractionRecord]:
    prompt = args.prompt.read_text(encoding="utf-8")
    try:
        detected_sdk_version = importlib.metadata.version("openai-codex")
    except importlib.metadata.PackageNotFoundError:
        detected_sdk_version = "not_installed"
    selected = parse_case_ids(args.case_ids)
    rows = load_benchmark(args.benchmark)
    if selected is not None:
        rows = [row for row in rows if row["ID"] in selected]
    existing = load_existing(args.output) if args.resume else {}
    records: dict[str, AgenticExtractionRecord] = dict(existing)
    pending: list[tuple[dict[str, str], Path, Path, Path]] = []

    for row in rows:
        case_id = row["ID"]
        title_path = args.configs_dir / f"case_{case_id}.title"
        report_path = args.configs_dir / f"case_{case_id}.report"
        kernel_dir = args.cases_root / f"case_{case_id}"
        cached = existing.get(case_id)
        inputs_present = title_path.is_file() and report_path.is_file()
        if cached_extraction_matches(
            cached, row, inputs_present, args.model, args.effort
        ):
            records[case_id] = cached
            continue
        records.pop(case_id, None)
        if not inputs_present:
            decision = AgenticExtractionDecision(
                status="missing_input",
                report_kind="unknown",
                concurrency_class="unknown",
                confidence="low",
                waypoints_target_to_entry=[],
                rationale="Required title or report input is missing.",
                unresolved_questions=[],
            )
            records[case_id] = AgenticExtractionRecord(
                case_id=case_id,
                title=row.get("Title", ""),
                bug_position=row.get("Bug Position", ""),
                model=args.model,
                effort=args.effort,
                codex_sdk_version=detected_sdk_version,
                thread_id="",
                decision=decision,
                configured_target_resolution=None,
            )
            continue
        pending.append((row, title_path, report_path, kernel_dir))

    write_records(args.output, records)
    if not pending:
        return records

    try:
        from openai_codex import ApprovalMode, AsyncCodex, CodexConfig, Sandbox
        from openai_codex import __version__ as codex_sdk_version
        from openai_codex.types import ReasoningEffort
        from codex_cli_bin import bundled_codex_path
    except ImportError as exc:
        raise RuntimeError(
            "openai-codex is required; run this script in the syzpilot-agent environment"
        ) from exc

    wrapper = Path(__file__).with_name("codex_blind_wrapper.sh")
    config = CodexConfig(
        cwd="/tmp",
        codex_bin=str(wrapper),
        env={
            "CODEX_HOME": str(args.codex_home.expanduser().resolve()),
            "SYZPILOT_CODEX_REAL_BIN": str(bundled_codex_path()),
            "SYZPILOT_MASKED_REPO": str(PROJECT_ROOT),
            "SYZPILOT_ALLOWED_ROOTS": os.pathsep.join(
                [
                    str(args.cases_root.expanduser().resolve()),
                    str(args.configs_dir.expanduser().resolve()),
                ]
            ),
        },
    )
    effort = ReasoningEffort(args.effort)
    semaphore = asyncio.Semaphore(args.max_concurrency)

    async with AsyncCodex(config) as codex:
        available_models = {item.model for item in (await codex.models()).data}
        if args.model not in available_models:
            raise RuntimeError(
                f"model {args.model!r} is unavailable; available={sorted(available_models)}"
            )

        async def extract_one(item):
            row, title_path, report_path, kernel_dir = item
            thread_id = ""
            target_resolution = None
            try:
                async with semaphore:
                    thread = await codex.thread_start(
                        cwd="/tmp",
                        model=args.model,
                        sandbox=Sandbox.read_only,
                        approval_mode=ApprovalMode.deny_all,
                        ephemeral=True,
                        config={"web_search": "disabled"},
                        developer_instructions=prompt,
                    )
                    thread_id = thread.id
                    result = await thread.run(
                        case_request(row, title_path, report_path, kernel_dir),
                        effort=effort,
                        sandbox=Sandbox.read_only,
                        approval_mode=ApprovalMode.deny_all,
                        output_schema=AgenticExtractionDecision.model_json_schema(),
                    )
                    if result.final_response is None:
                        raise RuntimeError(f"case {row['ID']} returned no final response")
                    decision = AgenticExtractionDecision.model_validate_json(
                        result.final_response
                    )
                    decision, target_resolution = (
                        await validate_and_repair_configured_target(
                            thread,
                            decision,
                            args.resolver_python,
                            kernel_dir,
                            args.target_resolution_retries,
                            args.resolver_timeout_seconds,
                            effort,
                            Sandbox.read_only,
                            ApprovalMode.deny_all,
                        )
                    )
            except Exception as exc:
                decision = AgenticExtractionDecision(
                    status="failed",
                    report_kind="unknown",
                    concurrency_class="unknown",
                    confidence="low",
                    waypoints_target_to_entry=[],
                    rationale=f"Agent extraction failed: {type(exc).__name__}: {exc}",
                    unresolved_questions=[],
                )
                target_resolution = None
            return AgenticExtractionRecord(
                case_id=row["ID"],
                title=row.get("Title", ""),
                bug_position=row.get("Bug Position", ""),
                model=args.model,
                effort=args.effort,
                codex_sdk_version=codex_sdk_version,
                thread_id=thread_id,
                decision=decision,
                configured_target_resolution=target_resolution,
            )

        tasks = [asyncio.create_task(extract_one(item)) for item in pending]
        for task in asyncio.as_completed(tasks):
            record = await task
            records[record.case_id] = record
            write_records(args.output, records)
            print(
                f"case_{record.case_id}: {record.decision.status} "
                f"nodes={len(record.decision.waypoints_target_to_entry)}",
                flush=True,
            )

    write_records(args.output, records)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run independent Codex-based waypoint extraction for benchmark reports."
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=PROJECT_ROOT / "benchmark" / "benchmark.csv",
    )
    parser.add_argument(
        "--cases-root",
        type=Path,
        default=Path.home() / "kernels" / "SyzPilot-experiments" / "cases",
    )
    parser.add_argument(
        "--configs-dir",
        type=Path,
        default=Path.home() / "kernels" / "SyzPilot-experiments" / "configs",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--codex-home", type=Path, default=Path.home() / ".codex_jjy")
    parser.add_argument(
        "--resolver-python",
        type=Path,
        default=Path.home() / "miniconda3/envs/syzpilot/bin/python",
    )
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--effort", choices=("low", "medium", "high", "xhigh"), default="xhigh")
    parser.add_argument("--max-concurrency", type=int, default=6)
    parser.add_argument("--target-resolution-retries", type=int, default=2)
    parser.add_argument("--resolver-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--case-ids")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.max_concurrency < 1:
        parser.error("--max-concurrency must be at least 1")
    if args.target_resolution_retries < 0:
        parser.error("--target-resolution-retries must be nonnegative")
    if args.resolver_timeout_seconds <= 0:
        parser.error("--resolver-timeout-seconds must be positive")
    if not args.resolver_python.is_file():
        parser.error(f"resolver Python executable not found: {args.resolver_python}")
    records = asyncio.run(run_all(args))
    print(f"Wrote {args.output}")
    selected = parse_case_ids(args.case_ids)
    considered = (
        records.values()
        if selected is None
        else (record for case_id, record in records.items() if case_id in selected)
    )
    return 1 if any(record.decision.status == "failed" for record in considered) else 0


if __name__ == "__main__":
    raise SystemExit(main())
