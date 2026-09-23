#!/usr/bin/env python3
"""Blindly score anonymous waypoint candidates with isolated Codex threads."""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import os
from pathlib import Path

from pydantic import ValidationError

if __package__:
    from .waypoint_schema import (
        BlindScoreDecision,
        BlindScoreRecord,
        ScoringCaseInput,
    )
else:
    from waypoint_schema import BlindScoreDecision, BlindScoreRecord, ScoringCaseInput


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT = Path(__file__).with_name("prompts") / "agentic_waypoint_scorer.md"
def load_input(path: Path) -> list[ScoringCaseInput]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases") if isinstance(payload, dict) else None
    if not isinstance(cases, list):
        raise ValueError("scoring input must contain a cases list")
    validated = [ScoringCaseInput.model_validate(case) for case in cases]
    case_ids = [case.case_id for case in validated]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("case_id values must be unique in scoring input")
    return validated


def load_existing(path: Path) -> dict[str, BlindScoreRecord]:
    if not path.is_file():
        return {}
    records: dict[str, BlindScoreRecord] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                record = BlindScoreRecord.model_validate_json(line)
            except ValidationError:
                continue
            records[record.case_id] = record
    return records


def write_records(path: Path, records: dict[str, BlindScoreRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.writing")
    with temporary.open("w", encoding="utf-8") as stream:
        for case_id in sorted(records, key=int):
            stream.write(records[case_id].model_dump_json() + "\n")
    os.replace(temporary, path)


def scoring_request(case: ScoringCaseInput) -> str:
    anonymous_candidates = [
        {
            "candidate_id": candidate.candidate_id,
            "waypoints_target_to_entry": candidate.waypoints_target_to_entry,
        }
        for candidate in sorted(case.candidates, key=lambda item: item.candidate_id)
    ]
    visible = {
        "case_id": case.case_id,
        "title": case.title,
        "bug_position": case.bug_position,
        "report_path": case.report_path,
        "kernel_dir": case.kernel_dir,
        "candidates": anonymous_candidates,
    }
    return (
        "Inspect the report and source, then review every anonymous candidate in "
        "this JSON bundle:\n" + json.dumps(visible, indent=2, sort_keys=True)
    )


def candidate_maps(
    case: ScoringCaseInput,
) -> tuple[dict[str, str], dict[str, list[str]]]:
    method_map = {
        candidate.candidate_id: candidate.method_key for candidate in case.candidates
    }
    chain_map = {
        candidate.candidate_id: candidate.waypoints_target_to_entry
        for candidate in case.candidates
    }
    return method_map, chain_map


def cached_score_matches(
    cached: BlindScoreRecord | None,
    case: ScoringCaseInput,
    model: str,
    effort: str,
) -> bool:
    if cached is None or cached.status == "failed":
        return False
    method_map, chain_map = candidate_maps(case)
    return (
        cached.model == model
        and cached.effort == effort
        and cached.candidate_method_map == method_map
        and cached.candidate_chain_map == chain_map
    )


def evidence_roots(cases: list[ScoringCaseInput]) -> list[str]:
    return sorted(
        {
            str(Path(case.kernel_dir).expanduser().resolve()) for case in cases
        }
        | {
            str(Path(case.report_path).expanduser().resolve().parent)
            for case in cases
        }
    )


async def run_all(args: argparse.Namespace) -> dict[str, BlindScoreRecord]:
    prompt = args.prompt.read_text(encoding="utf-8")
    try:
        detected_sdk_version = importlib.metadata.version("openai-codex")
    except importlib.metadata.PackageNotFoundError:
        detected_sdk_version = "not_installed"
    cases = load_input(args.input)
    selected = {item.strip() for item in args.case_ids.split(",")} if args.case_ids else None
    if selected is not None:
        cases = [case for case in cases if case.case_id in selected]
    existing = load_existing(args.output) if args.resume else {}
    records: dict[str, BlindScoreRecord] = dict(existing)
    pending: list[ScoringCaseInput] = []
    for case in cases:
        case_id = case.case_id
        cached = existing.get(case_id)
        if cached_score_matches(cached, case, args.model, args.effort):
            records[case_id] = cached
        else:
            records.pop(case_id, None)
            pending.append(case)

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
            "SYZPILOT_ALLOWED_ROOTS": os.pathsep.join(evidence_roots(pending)),
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

        async def score_one(item):
            case = item
            case_id = case.case_id
            candidates = case.candidates
            candidate_ids = {candidate.candidate_id for candidate in candidates}
            method_map, chain_map = candidate_maps(case)
            thread_id = ""
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
                        scoring_request(case),
                        effort=effort,
                        sandbox=Sandbox.read_only,
                        approval_mode=ApprovalMode.deny_all,
                        output_schema=BlindScoreDecision.model_json_schema(),
                    )
                    if result.final_response is None:
                        raise RuntimeError(f"case {case_id} returned no final response")
                    decision = BlindScoreDecision.model_validate_json(
                        result.final_response
                    )
                    returned_ids = {review.candidate_id for review in decision.reviews}
                    if (
                        returned_ids != candidate_ids
                        or len(decision.reviews) != len(candidates)
                    ):
                        raise ValueError(
                            f"case {case_id} scorer candidate mismatch: "
                            f"expected={sorted(candidate_ids)} "
                            f"returned={sorted(returned_ids)}"
                        )
            except Exception as exc:
                return BlindScoreRecord(
                    case_id=case_id,
                    model=args.model,
                    effort=args.effort,
                    codex_sdk_version=codex_sdk_version,
                    thread_id=thread_id,
                    candidate_method_map=method_map,
                    candidate_chain_map=chain_map,
                    status="failed",
                    error=f"Agent scoring failed: {type(exc).__name__}: {exc}",
                    decision=None,
                )
            return BlindScoreRecord(
                case_id=case_id,
                model=args.model,
                effort=args.effort,
                codex_sdk_version=codex_sdk_version,
                thread_id=thread_id,
                candidate_method_map=method_map,
                candidate_chain_map=chain_map,
                status="ok",
                error="",
                decision=decision,
            )

        tasks = [asyncio.create_task(score_one(item)) for item in pending]
        for task in asyncio.as_completed(tasks):
            record = await task
            records[record.case_id] = record
            write_records(args.output, records)
            print(
                f"case_{record.case_id}: {record.status} "
                f"reviews={len(record.decision.reviews) if record.decision else 0}",
                flush=True,
            )

    write_records(args.output, records)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Blindly score anonymous waypoint candidates with Codex."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--codex-home", type=Path, default=Path.home() / ".codex_jjy")
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--effort", choices=("low", "medium", "high", "xhigh"), default="xhigh")
    parser.add_argument("--max-concurrency", type=int, default=6)
    parser.add_argument("--case-ids")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.max_concurrency < 1:
        parser.error("--max-concurrency must be at least 1")
    records = asyncio.run(run_all(args))
    print(f"Wrote {args.output}")
    selected = (
        {item.strip() for item in args.case_ids.split(",") if item.strip()}
        if args.case_ids
        else None
    )
    considered = (
        records.values()
        if selected is None
        else (record for case_id, record in records.items() if case_id in selected)
    )
    return 1 if any(record.status == "failed" for record in considered) else 0


if __name__ == "__main__":
    raise SystemExit(main())
