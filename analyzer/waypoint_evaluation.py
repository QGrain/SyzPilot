#!/usr/bin/env python3
"""Build blind scoring inputs and the canonical waypoint evaluation artifact."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from openpyxl import load_workbook
from openpyxl.workbook import Workbook

if __package__:
    from .agentic_waypoint_scorer import load_existing as load_score_records
    from .agentic_waypoints_extractor import load_existing as load_extraction_records
    from .evaluate_waypoint_quality import (
        ZERO_PC32,
        ZERO_PC64,
        coverage_compatibility_errors,
        coverage_evidence,
        load_coverage_results,
        parse_json_cell,
        parse_pc64,
        resolve_agentic_chain,
    )
    from .evaluate_waypoints_extractor import (
        STAGES,
        append_mapping_sheet,
        save_workbook_atomic,
        style_sheet,
    )
    from .waypoint_schema import (
        AGENTIC_EXTRACTION_SCHEMA_VERSION,
        ScoringCaseInput,
        semantic_score,
    )
    from .waypoints_extractor import WaypointsExtractor
else:
    from agentic_waypoint_scorer import load_existing as load_score_records
    from agentic_waypoints_extractor import load_existing as load_extraction_records
    from evaluate_waypoint_quality import (
        ZERO_PC32,
        ZERO_PC64,
        coverage_compatibility_errors,
        coverage_evidence,
        load_coverage_results,
        parse_json_cell,
        parse_pc64,
        resolve_agentic_chain,
    )
    from evaluate_waypoints_extractor import (
        STAGES,
        append_mapping_sheet,
        save_workbook_atomic,
        style_sheet,
    )
    from waypoint_schema import (
        AGENTIC_EXTRACTION_SCHEMA_VERSION,
        ScoringCaseInput,
        semantic_score,
    )
    from waypoints_extractor import WaypointsExtractor


STATIC_SCHEMA_VERSION = "2.0"
BLIND_SCORING_INPUT_SCHEMA_VERSION = "2.0"
SCHEMA_VERSION = "2.4"
AGENTIC_METHOD_KEY = "agentic:final"
WEIGHT_SEMANTIC = 0.30
WEIGHT_HIT_QUALITY = 0.20
WEIGHT_EFFECTIVE_LENGTH = 0.50
HIT_QUALITY_FLOOR = 50.0
K_MIN = WaypointsExtractor.MIN_WAYPOINTS_LENGTH


@dataclass(frozen=True)
class QualityWeights:
    """Validated weights for the paper-facing linear quality score."""

    semantic: float
    hit_quality: float
    effective_length: float

    def __post_init__(self) -> None:
        values = (self.semantic, self.hit_quality, self.effective_length)
        if not all(math.isfinite(value) and value >= 0 for value in values):
            raise ValueError("quality weights must be finite and nonnegative")
        if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("quality weights must sum to 1.0")

    def as_dict(self) -> dict[str, float]:
        return {
            "semantic": self.semantic,
            "hit_quality": self.hit_quality,
            "effective_length": self.effective_length,
        }

    def formula(self) -> str:
        def number(value: float) -> str:
            return f"{value:.12g}"

        return (
            f"{number(self.semantic)}*Semantic + "
            f"{number(self.hit_quality)}*Hit + "
            f"{number(self.effective_length)}*Length"
        )


DEFAULT_QUALITY_WEIGHTS = QualityWeights(
    semantic=WEIGHT_SEMANTIC,
    hit_quality=WEIGHT_HIT_QUALITY,
    effective_length=WEIGHT_EFFECTIVE_LENGTH,
)


def add_quality_weight_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--semantic-weight",
        type=float,
        default=WEIGHT_SEMANTIC,
        help="Semantic component weight (default: 0.30).",
    )
    parser.add_argument(
        "--hit-weight",
        type=float,
        default=WEIGHT_HIT_QUALITY,
        help="KCOV hit-quality component weight (default: 0.20).",
    )
    parser.add_argument(
        "--length-weight",
        type=float,
        default=WEIGHT_EFFECTIVE_LENGTH,
        help=(
            "Candidate-length component weight (default: 0.50); all three "
            "weights must be nonnegative and sum to 1."
        ),
    )


def quality_weights_from_args(args: argparse.Namespace) -> QualityWeights:
    return QualityWeights(
        semantic=float(getattr(args, "semantic_weight", WEIGHT_SEMANTIC)),
        hit_quality=float(getattr(args, "hit_weight", WEIGHT_HIT_QUALITY)),
        effective_length=float(
            getattr(args, "length_weight", WEIGHT_EFFECTIVE_LENGTH)
        ),
    )


def quality_weights_from_payload(payload: dict[str, Any]) -> QualityWeights:
    values = payload.get("quality_weights")
    if not isinstance(values, dict):
        return DEFAULT_QUALITY_WEIGHTS
    return QualityWeights(
        semantic=float(values["semantic"]),
        hit_quality=float(values["hit_quality"]),
        effective_length=float(values["effective_length"]),
    )


PUBLISHED_WORKBOOK_COLUMNS = {
    "Run_Metadata": ("Key", "Value"),
    "Cases": (
        "ID",
        "Title",
        "Version",
        "Bug Position",
        "Bug Position Instrumentation Target",
        "Bug Position PC64",
        "Bug Position Resolution Class",
        "Report Type",
        "Detected Report Sections",
        "Status",
        "Failed Stage",
        "Error",
        "Warnings",
        *(
            column
            for _, label, _ in STAGES
            for column in (
                f"{label} Waypoints (target->entry)",
                f"{label} Listed PCs (target->entry)",
                f"{label} Count",
            )
        ),
        "Final Waypoints (target->entry)",
        "Final Listed PCs (target->entry)",
        "Fuzzer target_pcs (entry->target)",
        "Final Count",
        "Bug Position Function Relation",
        "Bug Position Function Index (target=0)",
        "Validation Error Count",
        "Validation Warning Count",
    ),
    "Waypoints_Long": (
        "Case ID",
        "Stage",
        "Stage Order",
        "Index target->entry",
        "Report Target",
        "Resolved Target",
        "PC64",
        "PC32",
        "PC Resolution Error",
        "Inline",
        "BB Offset",
        "BB Count",
        "Hot Entry Score",
        "Value Score",
        "Present Next Stage",
        "Drop Transition",
        "Causal Phase",
    ),
    "Validation": ("Case ID", "Check", "Severity", "Passed", "Details"),
    "Summary": ("Category", "Metric", "Value", "Note"),
    "Agentic_Extraction": (
        "ID",
        "Title",
        "Bug Position",
        "Status",
        "Report Kind",
        "Concurrency Class",
        "Confidence",
        "Proposed Waypoints (target->entry)",
        "Proposed Causal Phases (target->entry)",
        "Final Waypoints (target->entry)",
        "Final PCs64 (target->entry)",
        "Final target_pcs (entry->target)",
        "Verified Target Proxy",
        "Verified Target PC64",
        "Dropped or Unresolved Nodes",
        "Rationale",
        "Unresolved Questions",
        "Model",
        "Effort",
    ),
    "Agentic_Waypoints": (
        "Case ID",
        "Proposed Index target->entry",
        "Waypoint",
        "Causal Phase",
        "Report Evidence",
        "Proxy Reason",
    ),
    "Evaluation": (
        "ID",
        "Method Key",
        "Proposed Length",
        "Resolved Unique Length",
        "Configured Target Retained",
        "Waypoints (target->entry)",
        "PCs64 (target->entry)",
        "Target Fidelity",
        "Section Fidelity",
        "Causal Coherence",
        "Parsimony",
        "Agent Evidence",
        "Agent Rationale",
        "Observability Score",
        "Semantic Quality Score",
        "PoC Evaluation Status",
        "Waypoint Hit Count",
        "Waypoint Total",
        "Waypoint Hit Ratio",
        "Target Hit",
        "Dynamic Coverage Score",
        "Hit Quality Score",
        "Effective Length Score",
        "Quality Score",
        "Dynamic Evidence Confidence",
        "Artifact Compatibility Errors",
    ),
    "Evaluation_Waypoints": (
        "Case ID",
        "Method Key",
        "Index target->entry",
        "Configured Target",
        "Causal Phase",
        "Waypoint",
        "Resolved Target",
        "PC64",
        "PC32",
        "Coverfile Comparison PC64",
        "PoC Covered",
        "PoC Extra-only Covered",
    ),
    "Rule_Level_Summary": (
        "Method Key",
        "Extraction Method",
        "Stage",
        "Stage Type",
        "Case Count",
        "Changed vs Previous N",
        "Changed vs Previous Rate",
        "Mean Candidate Length",
        "Candidate Compression vs Raw",
        "Mean Operational Label Length",
        "Operational Compression vs Raw",
        "Configured Target Retained N",
        "Configured Target Retained Rate",
        "Coverage Case Count",
        "Mean Waypoint Hit Count",
        "Mean Waypoint Total",
        "Overall Waypoint Hit Rate",
        "Target Hit N",
        "Target Hit Rate",
        "Quality Score N",
        "Mean Quality Score",
    ),
    "Coverage_Runs": (
        "case_id",
        "run_idx",
        "preflight_status",
        "boot_status",
        "execution_status",
        "coverage_status",
        "crash_status",
        "crash_title",
        "target_reproduced",
        "poc_fidelity_status",
        "kaslr_status",
        "calls_only_pc_count",
        "extra_pc_count",
        "coverage_pc_count",
        "boot_seconds",
        "exec_seconds",
        "exit_code",
        "result_dir",
        "poc_path",
        "errors",
    ),
}

def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.writing")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def load_static(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != STATIC_SCHEMA_VERSION:
        raise ValueError(f"unsupported static schema: {payload.get('schema_version')!r}")
    if payload.get("artifact_type") != "script_waypoint_extraction":
        raise ValueError("input is not a script waypoint extraction artifact")
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise ValueError("static artifact cases must be a list")
    case_ids = [str(case.get("case_id", "")) for case in cases]
    if any(not case_id.isdigit() for case_id in case_ids):
        raise ValueError("static case IDs must be decimal digits")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("static case IDs must be unique")
    return payload


def correctness_errors(case: dict[str, Any]) -> list[str]:
    return [
        str(validation.get("check", "unknown"))
        for validation in case.get("validations", [])
        if validation.get("severity") == "error" and not validation.get("passed")
    ]


def quality_eligible(case: dict[str, Any]) -> bool:
    return case.get("status") == "ok" and not correctness_errors(case)


def static_chain(case: dict[str, Any], stage_key: str) -> dict[str, Any] | None:
    if stage_key == "outlier_removal":
        final = case.get("final")
        if isinstance(final, dict):
            return final
    stage = case.get("stages", {}).get(stage_key)
    return stage.get("chain") if isinstance(stage, dict) else None


def candidate_methods(
    case: dict[str, Any], extraction: Any | None
) -> dict[str, list[str]]:
    methods = {}
    for stage_key, _, _ in STAGES:
        chain = static_chain(case, stage_key)
        if chain and chain.get("waypoints_target_to_entry"):
            methods[f"script:{stage_key}"] = list(
                chain["waypoints_target_to_entry"]
            )
    if extraction is not None and extraction.decision.status == "ok":
        methods[AGENTIC_METHOD_KEY] = [
            waypoint.target
            for waypoint in extraction.decision.waypoints_target_to_entry
        ]
    return methods


def require_complete_candidates(
    case: dict[str, Any], extraction: Any | None
) -> dict[str, list[str]]:
    methods = candidate_methods(case, extraction)
    required = {f"script:{stage_key}" for stage_key, _, _ in STAGES}
    required.add(AGENTIC_METHOD_KEY)
    missing = sorted(required - set(methods))
    if missing:
        raise ValueError(
            f"case {case['case_id']} is missing required quality candidates: {missing}"
        )
    if set(methods) != required:
        raise ValueError(f"case {case['case_id']} has unexpected quality candidates")
    return methods


def build_scoring_input(
    static: dict[str, Any],
    extractions: dict[str, Any],
    seed: str,
) -> dict[str, Any]:
    cases = []
    for case in sorted(static["cases"], key=lambda item: int(item["case_id"])):
        case_id = str(case["case_id"])
        if not quality_eligible(case):
            continue
        extraction = extractions.get(case_id)
        methods = list(require_complete_candidates(case, extraction).items())
        seed_value = sum(
            (index + 1) * ord(character)
            for index, character in enumerate(f"{seed}:{case_id}")
        )
        random.Random(seed_value).shuffle(methods)
        candidates = []
        for index, (method_key, targets) in enumerate(methods, start=1):
            candidates.append(
                {
                    "candidate_id": f"c_{index:016x}",
                    "method_key": method_key,
                    "waypoints_target_to_entry": targets,
                }
            )
        if not candidates:
            continue
        scoring_case = ScoringCaseInput(
            case_id=case_id,
            title=str(case.get("title", "")),
            bug_position=str(case.get("bug_position", "")),
            report_path=str(case.get("paths", {}).get("report", "")),
            kernel_dir=str(case.get("paths", {}).get("kernel_dir", "")),
            candidates=sorted(candidates, key=lambda item: item["candidate_id"]),
        )
        cases.append(scoring_case.model_dump(mode="json"))
    return {
        "schema_version": BLIND_SCORING_INPUT_SCHEMA_VERSION,
        "artifact_type": "blind_waypoint_scoring_input",
        "static_run_id": static.get("run_metadata", {}).get("Run ID", ""),
        "blind_seed": seed,
        "cases": cases,
    }


def normalized_chain(
    targets: list[str],
    phases: list[str],
    resolved_targets: list[str],
    pcs64: list[str | None],
    errors: list[str | None],
    configured_target_proposed_index: int | None = 0,
) -> dict[str, Any]:
    """Normalize PCs while preserving the explicitly identified target node."""
    if not (
        len(targets)
        == len(phases)
        == len(resolved_targets)
        == len(pcs64)
        == len(errors)
    ):
        raise ValueError("waypoint chain arrays are not aligned")
    if configured_target_proposed_index is not None and not (
        0 <= configured_target_proposed_index < len(targets)
    ):
        raise ValueError("configured target index is outside the proposed chain")
    valid_nodes: dict[int, dict[str, Any]] = {}
    dropped = []
    indices_by_pc32: dict[int, list[int]] = defaultdict(list)
    for index, (target, phase, resolved, pc64, error) in enumerate(
        zip(targets, phases, resolved_targets, pcs64, errors)
    ):
        value = parse_pc64(pc64)
        pc32_value = value & 0xFFFFFFFF
        if error or not value or not pc32_value:
            dropped.append(
                {
                    "index_target_to_entry": index,
                    "waypoint": target,
                    "reason": error or "zero_or_missing_pc",
                }
            )
            continue
        indices_by_pc32[pc32_value].append(index)
        valid_nodes[index] = {
            "proposed_index_target_to_entry": index,
            "waypoint": target,
            "causal_phase": phase,
            "resolved_target": resolved,
            "pc64": f"0x{value:016x}",
            "pc32": f"0x{pc32_value:08x}",
        }
    retained_indices = set()
    for pc32_value, indices in indices_by_pc32.items():
        retained = (
            configured_target_proposed_index
            if configured_target_proposed_index is not None
            and configured_target_proposed_index in indices
            else min(indices)
        )
        retained_indices.add(retained)
        for index in indices:
            if index == retained:
                continue
            dropped.append(
                {
                    "index_target_to_entry": index,
                    "waypoint": targets[index],
                    "reason": f"duplicate_pc32:0x{pc32_value:08x}",
                }
            )
    nodes = [valid_nodes[index] for index in sorted(retained_indices)]
    for index, node in enumerate(nodes):
        node["index_target_to_entry"] = index
        node["index_entry_to_target"] = len(nodes) - index - 1
    configured_target_index = next(
        (
            index
            for index, node in enumerate(nodes)
            if node["proposed_index_target_to_entry"]
            == configured_target_proposed_index
        ),
        None,
    )
    configured_target_retained = configured_target_index is not None
    return {
        "proposed_length": len(targets),
        "resolved_unique_length": len(nodes),
        "waypoints_target_to_entry": [node["waypoint"] for node in nodes],
        "causal_phases_target_to_entry": [node["causal_phase"] for node in nodes],
        "resolved_targets_target_to_entry": [
            node["resolved_target"] for node in nodes
        ],
        "pcs64_target_to_entry": [node["pc64"] for node in nodes],
        "pcs32_target_to_entry": [node["pc32"] for node in nodes],
        "fuzzer_pcs32_entry_to_target": [
            node["pc32"] for node in reversed(nodes)
        ],
        "configured_target_retained": configured_target_retained,
        "configured_target_proposed_index": configured_target_proposed_index,
        "configured_target_index_target_to_entry": configured_target_index,
        "configured_target_waypoint": (
            targets[configured_target_proposed_index]
            if configured_target_proposed_index is not None
            else None
        ),
        "configured_target_pc64": (
            nodes[configured_target_index]["pc64"]
            if configured_target_index is not None
            else None
        ),
        "nodes": nodes,
        "dropped_nodes": dropped,
}


def script_configured_target_index(
    case: dict[str, Any], chain: dict[str, Any]
) -> int | None:
    targets = list(chain.get("waypoints_target_to_entry", []))
    pcs64 = list(chain.get("pcs64_target_to_entry", []))
    resolution = case.get("bug_position_resolution", {})
    configured_pc32 = parse_pc64(resolution.get("pc64")) & 0xFFFFFFFF
    if configured_pc32:
        for index, pc64 in enumerate(pcs64):
            if parse_pc64(pc64) & 0xFFFFFFFF == configured_pc32:
                return index
    configured_function = str(resolution.get("function") or "")
    if configured_function:
        for index, target in enumerate(targets):
            if target.split("@", 1)[0] == configured_function:
                return index
    return None


def agentic_configured_target_index(phases: list[str]) -> int:
    indices = [
        index for index, phase in enumerate(phases) if phase == "configured_target"
    ]
    if indices != [0]:
        raise ValueError(
            "agentic target-to-entry chain must identify the configured target "
            "exactly once at index zero"
        )
    return 0


def resolve_candidate(
    case: dict[str, Any], method_key: str, extraction: Any | None
) -> dict[str, Any]:
    if method_key.startswith("script:"):
        stage_key = method_key.split(":", 1)[1]
        chain = static_chain(case, stage_key)
        if chain is None:
            raise ValueError(f"case {case['case_id']} lacks stage {stage_key}")
        targets = list(chain.get("waypoints_target_to_entry", []))
        phases = ["" for _ in targets]
        resolved = list(chain.get("resolved_targets_target_to_entry", []))
        pcs64 = list(chain.get("pcs64_target_to_entry", []))
        errors = list(chain.get("resolution_errors_target_to_entry", []))
        configured_target_index = script_configured_target_index(case, chain)
    elif method_key == AGENTIC_METHOD_KEY:
        if extraction is None or extraction.decision.status != "ok":
            raise ValueError(f"case {case['case_id']} has no successful agentic chain")
        waypoints = extraction.decision.waypoints_target_to_entry
        targets = [waypoint.target for waypoint in waypoints]
        phases = [waypoint.causal_phase for waypoint in waypoints]
        try:
            resolved, pcs64, _, errors = resolve_agentic_chain(
                str(case["paths"]["kernel_dir"]), targets
            )
        except (Exception, SystemExit) as exc:
            resolved = list(targets)
            pcs64 = [None for _ in targets]
            errors = [f"resolution_failed:{type(exc).__name__}:{exc}" for _ in targets]
        configured_target_index = agentic_configured_target_index(phases)
    else:
        raise ValueError(f"unsupported method key: {method_key}")
    chain = normalized_chain(
        targets,
        phases,
        resolved,
        pcs64,
        errors,
        configured_target_proposed_index=configured_target_index,
    )
    if method_key == AGENTIC_METHOD_KEY:
        if not chain["configured_target_retained"]:
            raise ValueError(
                f"case {case['case_id']} agentic configured target has no observable PC"
            )
        recorded = extraction.configured_target_resolution
        if recorded is None:
            raise ValueError(
                f"case {case['case_id']} lacks verified agentic target resolution"
            )
        if chain["configured_target_pc64"].lower() != recorded.pc64.lower():
            raise ValueError(
                f"case {case['case_id']} agentic target resolution changed"
            )
    return chain


def structured_coverage(evidence: dict[str, Any]) -> dict[str, Any]:
    list_fields = {
        "Coverage Statuses",
        "KASLR Statuses",
        "Per-Waypoint Hits (target->entry)",
        "Per-Waypoint Extra-only Hits (target->entry)",
        "Coverage Comparison PCs64 (target->entry)",
        "Missing Waypoint PCs",
        "PoC Fidelity Statuses",
    }
    structured = {}
    for key, value in evidence.items():
        if key in list_fields:
            structured[key] = parse_json_cell(value)
        else:
            structured[key] = value
    return structured


def review_by_candidate(score_record: Any | None) -> dict[str, Any]:
    if score_record is None or score_record.status != "ok":
        return {}
    return {
        review.candidate_id: review
        for review in score_record.decision.reviews
    }


def dynamic_configured_target_index(
    method_key: str, chain: dict[str, Any], review: Any | None
) -> int | None:
    """Grant target-hit credit only when target semantics are independently eligible."""
    configured_index = chain["configured_target_index_target_to_entry"]
    if method_key != AGENTIC_METHOD_KEY:
        return configured_index
    if review is None or review.target_fidelity not in {
        "exact",
        "resolvable_proxy",
    }:
        return None
    return configured_index


def validate_scoring_bundle(
    static: dict[str, Any],
    scoring_input: dict[str, Any],
    extractions: dict[str, Any],
    scores: dict[str, Any],
) -> None:
    if scoring_input.get("schema_version") != BLIND_SCORING_INPUT_SCHEMA_VERSION:
        raise ValueError(
            "unsupported blind scoring input schema: "
            f"{scoring_input.get('schema_version')!r}"
        )
    if scoring_input.get("artifact_type") != "blind_waypoint_scoring_input":
        raise ValueError("input is not a blind waypoint scoring artifact")
    static_run_id = static.get("run_metadata", {}).get("Run ID", "")
    if scoring_input.get("static_run_id") != static_run_id:
        raise ValueError("scoring input belongs to a different static run")
    static_cases = {str(case["case_id"]): case for case in static["cases"]}
    scoring_cases = {
        str(case["case_id"]): ScoringCaseInput.model_validate(case)
        for case in scoring_input.get("cases", [])
    }
    expected_case_ids = {
        case_id
        for case_id, case in static_cases.items()
        if quality_eligible(case)
    }
    if set(scoring_cases) != expected_case_ids:
        raise ValueError(
            "scoring input case set does not match current successful static cases"
        )
    for case_id, scoring_case in scoring_cases.items():
        expected = require_complete_candidates(
            static_cases[case_id], extractions.get(case_id)
        )
        actual = {
            candidate.method_key: candidate.waypoints_target_to_entry
            for candidate in scoring_case.candidates
        }
        if actual != expected:
            raise ValueError(f"case {case_id} scoring candidates are stale")
        record = scores.get(case_id)
        if record is None:
            raise ValueError(f"case {case_id} has no blind-score record")
        if record.status != "ok":
            raise ValueError(f"case {case_id} blind scoring failed: {record.error}")
        expected_map = {
            candidate.candidate_id: candidate.method_key
            for candidate in scoring_case.candidates
        }
        expected_chains = {
            candidate.candidate_id: candidate.waypoints_target_to_entry
            for candidate in scoring_case.candidates
        }
        if record.candidate_method_map != expected_map:
            raise ValueError(f"case {case_id} blind-score candidate map is stale")
        if record.candidate_chain_map != expected_chains:
            raise ValueError(f"case {case_id} blind-score candidate chains are stale")
        returned_ids = [review.candidate_id for review in record.decision.reviews]
        if len(returned_ids) != len(set(returned_ids)) or set(returned_ids) != set(
            expected_map
        ):
            raise ValueError(f"case {case_id} blind-score review IDs are invalid")


def mean_or_none(values: Iterable[Any]) -> float | None:
    numeric = [float(value) for value in values if isinstance(value, (int, float))]
    return round(mean(numeric), 3) if numeric else None


def hit_quality_score(coverage: dict[str, Any]) -> float | None:
    """Score direct KCOV evidence without treating a PoC miss as zero quality."""
    hit_count = coverage.get("Waypoint Hit Count")
    total = coverage.get("Waypoint Total")
    if not isinstance(hit_count, (int, float)) or not isinstance(
        total, (int, float)
    ) or total <= 0:
        return None
    ratio = min(1.0, max(0.0, float(hit_count) / float(total)))
    absolute_credit = min(1.0, max(0.0, float(hit_count) / K_MIN))
    target_credit = 1.0 if coverage.get("Target Hit") is True else 0.0
    score = (
        HIT_QUALITY_FLOOR
        + 30.0 * ratio
        + 10.0 * absolute_credit
        + 10.0 * target_credit
    )
    return round(min(100.0, score), 1)


def effective_length_score(candidate_length: Any) -> float | None:
    """Reward a compact rule-stage candidate chain relative to K_min."""
    if not isinstance(candidate_length, (int, float)):
        return None
    length = float(candidate_length)
    if length <= 0:
        return None
    return round(100.0 * min(1.0, K_MIN / length), 1)


def combined_quality_score(
    semantic: Any,
    hit_quality: Any,
    length_quality: Any,
    weights: QualityWeights = DEFAULT_QUALITY_WEIGHTS,
) -> float | None:
    """Combine semantic, execution-hit, and candidate-length evidence."""
    if not all(
        isinstance(value, (int, float))
        for value in (semantic, hit_quality, length_quality)
    ):
        return None
    score = (
        weights.semantic * float(semantic)
        + weights.hit_quality * float(hit_quality)
        + weights.effective_length * float(length_quality)
    )
    return round(min(100.0, max(0.0, score)), 1)


def enrich_reporting_metrics(
    payload: dict[str, Any],
    weights: QualityWeights = DEFAULT_QUALITY_WEIGHTS,
) -> dict[str, Any]:
    """Add deterministic reporting-only metrics to current or historical JSON."""
    payload.pop("content_id", None)
    for item in payload.get("evaluations", []):
        semantic = item.get(
            "semantic_quality_score", item.get("agentic_quality_score")
        )
        coverage = item.get("coverage", {})
        length_quality = effective_length_score(
            item.get("chain", {}).get("proposed_length")
        )
        hit_quality = hit_quality_score(coverage)
        item["semantic_quality_score"] = semantic
        item["hit_quality_score"] = hit_quality
        item["effective_length_score"] = length_quality
        item["quality_score"] = combined_quality_score(
            semantic,
            hit_quality,
            length_quality,
            weights,
        )
        for legacy_key in (
            "agentic_quality_score",
            "weighted_quality_score",
            "positive_dynamic_evidence",
            "positive_evidence_sensitivity_score",
        ):
            item.pop(legacy_key, None)
    semantics = payload.setdefault("score_semantics", {})
    semantics.clear()
    semantics["semantic"] = (
        "blind target/section/causal/parsimony judgment plus deterministic "
        "PC observability"
    )
    semantics["hit_quality"] = (
        "50 + 30*h/r + 10*min(h/K_min,1) + 10*target_hit"
    )
    semantics["effective_length"] = (
        "100*min(K_min/n,1), using the unnormalized rule-stage candidate count"
    )
    semantics["quality"] = (
        f"{weights.semantic:.12g}*semantic + "
        f"{weights.hit_quality:.12g}*hit_quality + "
        f"{weights.effective_length:.12g}*effective_length"
    )
    payload["quality_weights"] = weights.as_dict()
    payload["schema_version"] = SCHEMA_VERSION
    return payload


def build_evaluation(
    static: dict[str, Any],
    scoring_input: dict[str, Any],
    extractions: dict[str, Any],
    scores: dict[str, Any],
    coverage_results: Path | None,
    weights: QualityWeights = DEFAULT_QUALITY_WEIGHTS,
) -> dict[str, Any]:
    coverage_root, coverage_runs, coverage_manifest = load_coverage_results(
        coverage_results
    )
    static_cases = {str(case["case_id"]): case for case in static["cases"]}
    evaluations = []
    scoring_cases = {
        str(case["case_id"]): ScoringCaseInput.model_validate(case)
        for case in scoring_input["cases"]
    }
    for case_id in sorted(scoring_cases, key=int):
        case = static_cases[case_id]
        extraction = extractions.get(case_id)
        score_record = scores.get(case_id)
        expected_method_map = {
            candidate.candidate_id: candidate.method_key
            for candidate in scoring_cases[case_id].candidates
        }
        expected_chain_map = {
            candidate.candidate_id: candidate.waypoints_target_to_entry
            for candidate in scoring_cases[case_id].candidates
        }
        if score_record is not None and score_record.candidate_method_map != expected_method_map:
            raise ValueError(f"case {case_id} blind-score candidate map is stale")
        if score_record is not None and score_record.candidate_chain_map != expected_chain_map:
            raise ValueError(f"case {case_id} blind-score candidate chains are stale")
        reviews = review_by_candidate(score_record)
        case_runs = coverage_runs.get(case_id, [])
        compatibility = coverage_compatibility_errors(
            str(case["paths"]["kernel_dir"]),
            case_runs,
            coverage_manifest,
        )
        for candidate in scoring_cases[case_id].candidates:
            chain = resolve_candidate(case, candidate.method_key, extraction)
            if not chain["resolved_unique_length"]:
                raise ValueError(
                    f"case {case_id} {candidate.method_key} has no observable PCs"
                )
            review = reviews.get(candidate.candidate_id)
            observability = round(
                15.0
                * chain["resolved_unique_length"]
                / max(chain["proposed_length"], 1),
                1,
            )
            semantic = semantic_score(review, observability) if review else None
            dynamic_target_index = dynamic_configured_target_index(
                candidate.method_key, chain, review
            )
            evidence = structured_coverage(
                coverage_evidence(
                    coverage_root,
                    case_runs,
                    chain["pcs64_target_to_entry"],
                    compatibility_errors=compatibility,
                    configured_target_index=dynamic_target_index,
                )
            )
            hit_quality = hit_quality_score(evidence)
            length_quality = effective_length_score(
                chain["proposed_length"]
            )
            quality = combined_quality_score(
                semantic,
                hit_quality,
                length_quality,
                weights,
            )
            family, stage = candidate.method_key.split(":", 1)
            evaluations.append(
                {
                    "case_id": case_id,
                    "title": case.get("title", ""),
                    "bug_position": case.get("bug_position", ""),
                    "candidate_id": candidate.candidate_id,
                    "method_key": candidate.method_key,
                    "extraction_method": family,
                    "stage": stage,
                    "proposed_waypoints_target_to_entry": list(
                        candidate.waypoints_target_to_entry
                    ),
                    "chain": chain,
                    "observability_score": observability,
                    "semantic_judgment": (
                        review.model_dump(mode="json") if review else None
                    ),
                    "dynamic_target_bonus_eligible": (
                        dynamic_target_index is not None
                    ),
                    "semantic_quality_score": semantic,
                    "coverage": evidence,
                    "hit_quality_score": hit_quality,
                    "effective_length_score": length_quality,
                    "quality_score": quality,
                }
            )
    extraction_payload = []
    for case_id in sorted(static_cases, key=int):
        record = extractions.get(case_id)
        extraction_payload.append(
            record.model_dump(mode="json")
            if record is not None
            else {"case_id": case_id, "decision": {"status": "not_run"}}
        )
    body = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "waypoint_quality_evaluation",
        "score_semantics": {
            "semantic": (
                "target 30 + section 20 + causal coherence 25 + parsimony 10 "
                "+ deterministic PC observability 15"
            ),
            "hit_quality": (
                "50 + 30*h/r + 10*min(h/K_min,1) + 10*target_hit"
            ),
            "effective_length": (
                "100*min(K_min/n,1), using the unnormalized rule-stage "
                "candidate count"
            ),
            "quality": weights.formula(),
        },
        "provenance": {
            "static_run_id": static.get("run_metadata", {}).get("Run ID", ""),
            "coverage_results": str(coverage_results.resolve()) if coverage_results else "",
            "coverage_manifest": coverage_manifest,
        },
        "agentic_extractions": extraction_payload,
        "excluded_cases": [
            {
                "case_id": case_id,
                "static_status": static_cases[case_id].get("status", ""),
                "failed_stage": static_cases[case_id].get("failed_stage"),
                "error": static_cases[case_id].get("error"),
                "correctness_errors": correctness_errors(static_cases[case_id]),
            }
            for case_id in sorted(static_cases, key=int)
            if not quality_eligible(static_cases[case_id])
        ],
        "evaluations": evaluations,
        "coverage_runs": [
            row
            for case_id in sorted(set(coverage_runs) & set(static_cases), key=int)
            for row in coverage_runs[case_id]
        ],
    }
    return enrich_reporting_metrics(body, weights)


def excel_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, separators=(",", ":"), sort_keys=True)
    return value


def evaluation_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in payload["evaluations"]:
        chain = item["chain"]
        coverage = item["coverage"]
        judgment = item["semantic_judgment"] or {}
        rows.append(
            {
                "ID": item["case_id"],
                "Title": item["title"],
                "Extraction Method": item["extraction_method"],
                "Stage": item["stage"],
                "Method Key": item["method_key"],
                "Anonymous Candidate ID": item["candidate_id"],
                "Proposed Length": chain["proposed_length"],
                "Resolved Unique Length": chain["resolved_unique_length"],
                "Configured Target Retained": chain[
                    "configured_target_retained"
                ],
                "Configured Target Waypoint": chain[
                    "configured_target_waypoint"
                ],
                "Configured Target PC64": chain["configured_target_pc64"],
                "Dynamic Target Bonus Eligible": item.get(
                    "dynamic_target_bonus_eligible", ""
                ),
                "Waypoints (target->entry)": "\n".join(
                    chain["waypoints_target_to_entry"]
                ),
                "Causal Phases (target->entry)": excel_value(
                    chain["causal_phases_target_to_entry"]
                ),
                "PCs64 (target->entry)": excel_value(chain["pcs64_target_to_entry"]),
                "Fuzzer target_pcs (entry->target)": excel_value(
                    chain["fuzzer_pcs32_entry_to_target"]
                ),
                "Dropped or Unresolved Nodes": excel_value(chain["dropped_nodes"]),
                "Target Fidelity": judgment.get("target_fidelity", ""),
                "Section Fidelity": judgment.get("section_fidelity", ""),
                "Causal Coherence": judgment.get("causal_coherence", ""),
                "Parsimony": judgment.get("parsimony", ""),
                "Agent Confidence": judgment.get("confidence", ""),
                "Agent Evidence": excel_value(judgment.get("evidence", [])),
                "Agent Rationale": judgment.get("rationale", ""),
                "Observability Score": item["observability_score"],
                "Semantic Quality Score": item["semantic_quality_score"],
                "PoC Evaluation Status": coverage["PoC Evaluation Status"],
                "Waypoint Hit Count": coverage["Waypoint Hit Count"],
                "Waypoint Total": coverage["Waypoint Total"],
                "Waypoint Hit Ratio": coverage["Waypoint Hit Ratio"],
                "Target Hit": coverage["Target Hit"],
                "Dynamic Coverage Score": coverage["Dynamic Coverage Score"],
                "Hit Quality Score": item["hit_quality_score"],
                "Effective Length Score": item["effective_length_score"],
                "Quality Score": item["quality_score"],
                "Dynamic Evidence Confidence": coverage[
                    "Dynamic Evidence Confidence"
                ],
                "Artifact Compatibility Errors": coverage[
                    "Artifact Compatibility Errors"
                ],
            }
        )
    return rows


def evaluation_waypoint_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in payload["evaluations"]:
        coverage = item["coverage"]
        hits = coverage["Per-Waypoint Hits (target->entry)"]
        extra_hits = coverage["Per-Waypoint Extra-only Hits (target->entry)"]
        comparison = coverage["Coverage Comparison PCs64 (target->entry)"]
        hit_sources = coverage.get("Hit Sources", {})
        extra_sources = coverage.get("Extra Hit Sources", {})
        for index, node in enumerate(item["chain"]["nodes"]):
            rows.append(
                {
                    "Case ID": item["case_id"],
                    "Method Key": item["method_key"],
                    "Extraction Method": item["extraction_method"],
                    "Stage": item["stage"],
                    "Anonymous Candidate ID": item["candidate_id"],
                    "Index target->entry": index,
                    "Index entry->target": len(item["chain"]["nodes"]) - index - 1,
                    "Configured Target": index
                    == item["chain"]["configured_target_index_target_to_entry"],
                    "Causal Phase": node["causal_phase"],
                    "Waypoint": node["waypoint"],
                    "Resolved Target": node["resolved_target"],
                    "PC64": node["pc64"],
                    "PC32": node["pc32"],
                    "Coverfile Comparison PC64": comparison[index],
                    "PoC Covered": hits[index] if hits else "",
                    "Coverage Sources": excel_value(
                        hit_sources.get(node["pc64"], [])
                    ),
                    "PoC Extra-only Covered": extra_hits[index] if extra_hits else "",
                    "Extra Coverage Sources": excel_value(
                        extra_sources.get(node["pc64"], [])
                    ),
                }
            )
    return rows


def extraction_rows(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cases = []
    waypoints = []
    final_agentic = {
        item["case_id"]: item
        for item in payload["evaluations"]
        if item["method_key"] == AGENTIC_METHOD_KEY
    }
    for record in payload["agentic_extractions"]:
        decision = record.get("decision", {})
        case_id = str(record["case_id"])
        evaluation = final_agentic.get(case_id)
        chain = evaluation["chain"] if evaluation else None
        proposed = decision.get("waypoints_target_to_entry", [])
        target_resolution = record.get("configured_target_resolution") or {}
        cases.append(
            {
                "ID": case_id,
                "Title": record.get("title", ""),
                "Bug Position": record.get("bug_position", ""),
                "Extraction Type": (
                    "agentic extracted" if decision.get("status") == "ok" else "missing input"
                    if decision.get("status") == "missing_input" else "agentic failed"
                ),
                "Status": decision.get("status", "not_run"),
                "Report Kind": decision.get("report_kind", ""),
                "Concurrency Class": decision.get("concurrency_class", ""),
                "Confidence": decision.get("confidence", ""),
                "Proposed Waypoints (target->entry)": "\n".join(
                    item.get("target", "") for item in proposed
                ),
                "Proposed Causal Phases (target->entry)": excel_value(
                    [item.get("causal_phase", "") for item in proposed]
                ),
                "Final Waypoints (target->entry)": (
                    "\n".join(chain["waypoints_target_to_entry"]) if chain else ""
                ),
                "Final PCs64 (target->entry)": (
                    excel_value(chain["pcs64_target_to_entry"]) if chain else ""
                ),
                "Final target_pcs (entry->target)": (
                    excel_value(chain["fuzzer_pcs32_entry_to_target"])
                    if chain
                    else ""
                ),
                "Verified Target Proxy": target_resolution.get(
                    "proposed_target", ""
                ),
                "Verified Target PC64": target_resolution.get("pc64", ""),
                "Verified Target PC32": target_resolution.get("pc32", ""),
                "Dynamic Target Bonus Eligible": (
                    evaluation.get("dynamic_target_bonus_eligible", "")
                    if evaluation
                    else ""
                ),
                "Dropped or Unresolved Nodes": (
                    excel_value(chain["dropped_nodes"]) if chain else ""
                ),
                "Rationale": decision.get("rationale", ""),
                "Unresolved Questions": excel_value(
                    decision.get("unresolved_questions", [])
                ),
                "Model": record.get("model", ""),
                "Effort": record.get("effort", ""),
                "Thread ID": record.get("thread_id", ""),
            }
        )
        for index, waypoint in enumerate(proposed):
            waypoints.append(
                {
                    "Case ID": case_id,
                    "Proposed Index target->entry": index,
                    "Waypoint": waypoint.get("target", ""),
                    "Causal Phase": waypoint.get("causal_phase", ""),
                    "Report Evidence": waypoint.get("report_evidence", ""),
                    "Proxy Reason": waypoint.get("proxy_reason", ""),
                }
            )
    return cases, waypoints


def summary_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in payload["evaluations"]:
        groups[item["method_key"]].append(item)
    order = [f"script:{stage[0]}" for stage in STAGES] + [AGENTIC_METHOD_KEY]
    stage_types = {key: stage_type for key, _, stage_type in STAGES}
    raw_by_case = {
        str(item["case_id"]): item for item in groups.get("script:call_trace", [])
    }
    rows = []
    previous_script_by_case = raw_by_case
    for method_key in order:
        items = groups.get(method_key, [])
        if not items:
            continue
        if method_key == "script:call_trace":
            changed_n = 0
        elif method_key.startswith("script:"):
            changed_n = sum(
                tuple(item["proposed_waypoints_target_to_entry"])
                != tuple(
                    previous_script_by_case[str(item["case_id"])][
                        "proposed_waypoints_target_to_entry"
                    ]
                )
                for item in items
                if str(item["case_id"]) in previous_script_by_case
            )
        else:
            changed_n = None
        paired_items = [
            item for item in items if str(item["case_id"]) in raw_by_case
        ]
        raw_resolved_total = sum(
            raw_by_case[str(item["case_id"])]["chain"]["resolved_unique_length"]
            for item in paired_items
        )
        resolved_total = sum(
            item["chain"]["resolved_unique_length"] for item in paired_items
        )
        raw_proposed_total = sum(
            raw_by_case[str(item["case_id"])]["chain"]["proposed_length"]
            for item in paired_items
        )
        proposed_total = sum(
            item["chain"]["proposed_length"] for item in paired_items
        )
        proposed_compression = (
            1.0 - proposed_total / raw_proposed_total if raw_proposed_total else None
        )
        compression = (
            1.0 - resolved_total / raw_resolved_total if raw_resolved_total else None
        )
        coverage_items = [
            item
            for item in items
            if isinstance(item.get("hit_quality_score"), (int, float))
        ]
        quality_items = [
            item
            for item in items
            if isinstance(item.get("quality_score"), (int, float))
        ]
        target_retained_n = sum(
            bool(item["chain"]["configured_target_retained"]) for item in items
        )
        hit_total = sum(
            int(item["coverage"]["Waypoint Hit Count"] or 0)
            for item in coverage_items
        )
        waypoint_total = sum(
            int(item["coverage"]["Waypoint Total"] or 0)
            for item in coverage_items
        )
        stage = items[0]["stage"]
        row = {
            "Method Key": method_key,
            "Extraction Method": items[0]["extraction_method"],
            "Stage": stage,
            "Stage Type": (
                stage_types.get(stage, "optional")
                if method_key.startswith("script:")
                else "optional"
            ),
            "Case Count": len(items),
            "Changed vs Previous N": changed_n,
            "Changed vs Previous Rate": (
                round(changed_n / len(items), 4)
                if isinstance(changed_n, int) and items
                else None
            ),
            "Mean Candidate Length": mean_or_none(
                item["chain"]["proposed_length"] for item in items
            ),
            "Candidate Compression vs Raw": (
                round(proposed_compression, 4)
                if proposed_compression is not None
                else None
            ),
            "Mean Operational Label Length": mean_or_none(
                item["chain"]["resolved_unique_length"] for item in items
            ),
            "Operational Compression vs Raw": (
                round(compression, 4) if compression is not None else None
            ),
            "Configured Target Retained N": target_retained_n,
            "Configured Target Retained Rate": round(target_retained_n / len(items), 4),
            "Coverage Case Count": len(coverage_items),
            "Mean Waypoint Hit Count": mean_or_none(
                item["coverage"]["Waypoint Hit Count"] for item in coverage_items
            ),
            "Mean Waypoint Total": mean_or_none(
                item["coverage"]["Waypoint Total"] for item in coverage_items
            ),
            "Overall Waypoint Hit Rate": (
                round(hit_total / waypoint_total, 4) if waypoint_total else None
            ),
            "Target Hit N": sum(
                item["coverage"]["Target Hit"] is True for item in coverage_items
            ),
            "Target Hit Rate": (
                round(
                    sum(
                        item["coverage"]["Target Hit"] is True
                        for item in coverage_items
                    )
                    / len(coverage_items),
                    4,
                )
                if coverage_items
                else None
            ),
            "Quality Score N": len(quality_items),
            "Mean Quality Score": mean_or_none(
                item["quality_score"] for item in quality_items
            ),
        }
        rows.append(row)
        if method_key.startswith("script:"):
            previous_script_by_case = {
                str(item["case_id"]): item for item in items
            }
    return rows


def workbook_agentic_schema(workbook: Workbook) -> str:
    """Read explicit schema metadata or conservatively classify an old workbook."""
    if "Run_Metadata" in workbook.sheetnames:
        for key, value in workbook["Run_Metadata"].iter_rows(
            min_row=2, values_only=True
        ):
            if key == "Agentic Extraction Schema Version" and value:
                return str(value)
    if "Agentic_Extraction" not in workbook.sheetnames:
        return "legacy"
    sheet = workbook["Agentic_Extraction"]
    headers = [cell.value for cell in sheet[1]]
    if "Verified Target Proxy" not in headers:
        return "legacy"
    proxy_column = headers.index("Verified Target Proxy") + 1
    if any(
        sheet.cell(row_index, proxy_column).value
        for row_index in range(2, sheet.max_row + 1)
    ):
        return AGENTIC_EXTRACTION_SCHEMA_VERSION
    return "legacy"


def set_metadata_value(workbook: Workbook, key: str, value: Any) -> None:
    if "Run_Metadata" not in workbook.sheetnames:
        return
    metadata = workbook["Run_Metadata"]
    for row_index in range(2, metadata.max_row + 1):
        if metadata.cell(row_index, 1).value == key:
            metadata.cell(row_index, 2, value)
            return
    metadata.append([key, value])


def remove_metadata_keys(workbook: Workbook, keys: set[str]) -> None:
    if "Run_Metadata" not in workbook.sheetnames:
        return
    metadata = workbook["Run_Metadata"]
    for row_index in range(metadata.max_row, 1, -1):
        if metadata.cell(row_index, 1).value in keys:
            metadata.delete_rows(row_index)


def compact_workbook(workbook: Workbook) -> Workbook:
    """Keep the paper-analysis workbook small without changing source artifacts."""
    agentic_schema = workbook_agentic_schema(workbook)
    for sheet_name in tuple(workbook.sheetnames):
        if sheet_name not in PUBLISHED_WORKBOOK_COLUMNS:
            workbook.remove(workbook[sheet_name])

    if "Summary" in workbook.sheetnames:
        summary = workbook["Summary"]
        for row_index in range(summary.max_row, 1, -1):
            if summary.cell(row_index, 1).value == "Rule quality":
                summary.delete_rows(row_index)
    set_metadata_value(
        workbook, "Agentic Extraction Schema Version", agentic_schema
    )

    for sheet_name, wanted_columns in PUBLISHED_WORKBOOK_COLUMNS.items():
        if sheet_name not in workbook.sheetnames:
            continue
        sheet = workbook[sheet_name]
        rows = list(sheet.iter_rows(values_only=True))
        headers = list(rows[0]) if rows else []
        projected_rows = []
        for values in rows[1:]:
            row = dict(zip(headers, values))
            if (
                sheet_name == "Evaluation_Waypoints"
                and not row.get("Method Key")
                and row.get("Extraction Method")
                and row.get("Stage")
            ):
                row["Method Key"] = (
                    f"{row['Extraction Method']}:{row['Stage']}"
                )
            projected_rows.append([row.get(column) for column in wanted_columns])
        if sheet.max_row:
            sheet.delete_rows(1, sheet.max_row)
        sheet.append(list(wanted_columns))
        for values in projected_rows:
            sheet.append(values)
        style_sheet(sheet)
    desired_order = [
        name for name in PUBLISHED_WORKBOOK_COLUMNS if name in workbook.sheetnames
    ]
    for target_index, sheet_name in enumerate(desired_order):
        current_index = workbook.sheetnames.index(sheet_name)
        workbook.move_sheet(
            workbook[sheet_name], offset=target_index - current_index
        )
    return workbook


def workbook_readme_text(workbook: Workbook, workbook_name: str) -> str:
    """Describe the published workbook for paper-revision agents."""
    metadata = {}
    if "Run_Metadata" in workbook.sheetnames:
        metadata = {
            str(key): value
            for key, value in workbook["Run_Metadata"].iter_rows(
                min_row=2, values_only=True
            )
            if key
        }
    agent_schema = str(
        metadata.get("Agentic Extraction Schema Version", "legacy")
    )
    legacy_note = ""
    if not agent_schema.startswith(AGENTIC_EXTRACTION_SCHEMA_VERSION):
        legacy_note = (
            "\n## Agentic Legacy Caveat\n\n"
            "The `agentic:*` comparison and Agentic sheets predate configured-target "
            "PC verification and strict agentic target-to-entry order enforcement, "
            "so they are preliminary rather than a publication agentic baseline. "
            "The script extraction stages and script rows are unaffected and remain "
            "the paper-facing rule-level baseline.\n"
        )
    run_id = metadata.get("Run ID", "unknown")
    quality_formula = metadata.get(
        "Quality Formula", DEFAULT_QUALITY_WEIGHTS.formula()
    )
    return f"""# Waypoint Evaluation Workbook

Workbook: `{workbook_name}`
Run ID: `{run_id}`

## Reading Guide

- `Cases`: script-extracted waypoint chains, lengths, and aligned PCs after each rule stage.
- `Waypoints_Long`: per-node script trajectory and stage-retention details.
- `Validation`: deterministic correctness checks and diagnostics.
- `Summary`: run-level extraction counts, lengths, and validation totals.
- `Agentic_Extraction` and `Agentic_Waypoints`: independent agent candidate, causal phases, evidence, resolved PCs, and normalization drops.
- `Evaluation` and `Evaluation_Waypoints`: per-method quality scores and per-waypoint PoC coverage outcomes.
- `Rule_Level_Summary`: compact rule-stage candidate lengths, change prevalence, candidate and operational compression, KCOV hit statistics, target retention, and combined quality. Filter `Extraction Method=script` for the paper table; `agentic:final` is an artifact-only optional comparison.
- `Coverage_Runs`: compact execution provenance for the KCOV evidence used by scoring.

All displayed waypoint chains and aligned PC arrays use target-to-entry order. `Fuzzer target_pcs` uses entry-to-target order. The method-blind agent scores target fidelity, report-section fidelity, causal coherence, and parsimony; deterministic code adds PC observability to form `Semantic Quality Score`.

For `h` KCOV-hit labels among `r` unique operational labels, `Hit = 50 + 30*h/r + 10*min(h/4,1) + 10*target_hit`. For the unnormalized rule-stage candidate count `n`, `Length = 100*min(4/n,1)`. The combined score is the uncapped linear formula `Quality = {quality_formula}`. Missing or incompatible KCOV evidence leaves Hit and Quality blank rather than treating collection failure as a miss. `Overall Waypoint Hit Rate` is `sum(h)/sum(r)` across coverage-compatible cases, so each evaluated operational PC contributes equally.

For the revision, use the script rows in `Rule_Level_Summary` for aggregate rule-level tables, `Cases` for selected extraction case studies, and `Evaluation` for per-case component auditing. Agentic extraction remains an independent optional artifact method and is not part of the paper rule pipeline.
{legacy_note}"""


def set_quality_metadata(workbook: Workbook, weights: QualityWeights) -> None:
    set_metadata_value(workbook, "Semantic Weight", weights.semantic)
    set_metadata_value(workbook, "Hit Quality Weight", weights.hit_quality)
    set_metadata_value(
        workbook, "Effective Length Weight", weights.effective_length
    )
    set_metadata_value(workbook, "Quality Formula", weights.formula())


def write_workbook_readme(workbook: Workbook, workbook_path: Path) -> Path:
    readme_path = workbook_path.parent / "README.md"
    temporary = readme_path.with_name(f".{readme_path.name}.writing")
    temporary.write_text(
        workbook_readme_text(workbook, workbook_path.name), encoding="utf-8"
    )
    os.replace(temporary, readme_path)
    return readme_path


def build_workbook_view(
    static_workbook: Path,
    static: dict[str, Any],
    payload: dict[str, Any],
):
    weights = quality_weights_from_payload(payload)
    workbook = load_workbook(static_workbook)
    metadata = {
        row[0]: row[1]
        for row in workbook["Run_Metadata"].iter_rows(min_row=2, values_only=True)
        if row[0]
    }
    if metadata.get("Run ID") != payload["provenance"]["static_run_id"]:
        raise ValueError("static workbook and static JSON have different Run IDs")
    case_rows = workbook["Cases"].iter_rows(values_only=True)
    case_headers = list(next(case_rows))
    workbook_cases = {
        str(values[case_headers.index("ID")]): dict(zip(case_headers, values))
        for values in case_rows
    }
    workbook_case_ids = set(workbook_cases)
    payload_case_ids = {
        str(record["case_id"]) for record in payload["agentic_extractions"]
    }
    if workbook_case_ids != payload_case_ids:
        raise ValueError("static workbook and static JSON have different case sets")
    static_cases = {str(case["case_id"]): case for case in static["cases"]}
    for case_id, case in static_cases.items():
        workbook_case = workbook_cases[case_id]
        if (workbook_case.get("Status") or "") != case.get("status", ""):
            raise ValueError(
                f"case {case_id} static workbook and JSON statuses differ"
            )
        for stage_key, stage_label, _ in STAGES:
            stage = case.get("stages", {}).get(stage_key)
            chain = stage.get("chain") if isinstance(stage, dict) else None
            if chain is None:
                continue
            prefix = stage_label
            workbook_targets = (
                workbook_case.get(f"{prefix} Waypoints (target->entry)") or ""
            ).splitlines()
            if workbook_targets != chain["waypoints_target_to_entry"]:
                raise ValueError(
                    f"case {case_id} static workbook and JSON {stage_key} chains differ"
                )
            workbook_pcs = parse_json_cell(
                workbook_case.get(f"{prefix} Listed PCs (target->entry)")
            )
            workbook_pcs = [
                None if pc in (None, ZERO_PC32) else pc for pc in workbook_pcs
            ]
            if workbook_pcs != chain["pcs32_target_to_entry"]:
                raise ValueError(
                    f"case {case_id} static workbook and JSON {stage_key} PCs differ"
                )
            workbook_resolved = (
                workbook_case.get(f"{prefix} Resolved Targets (target->entry)")
                or ""
            ).splitlines()
            if workbook_resolved != chain["resolved_targets_target_to_entry"]:
                raise ValueError(
                    f"case {case_id} static workbook and JSON {stage_key} resolved targets differ"
                )
            workbook_errors = (
                workbook_case.get(f"{prefix} PC Resolution Errors") or ""
            ).splitlines()
            json_errors = [
                error
                for error in chain["resolution_errors_target_to_entry"]
                if error
            ]
            if workbook_errors != json_errors:
                raise ValueError(
                    f"case {case_id} static workbook and JSON {stage_key} errors differ"
                )
        final = case.get("final")
        if isinstance(final, dict):
            final_columns = {
                "Final Waypoints (target->entry)": final[
                    "waypoints_target_to_entry"
                ],
                "Final Resolved Targets (target->entry)": final[
                    "resolved_targets_target_to_entry"
                ],
            }
            for column, expected in final_columns.items():
                if (workbook_case.get(column) or "").splitlines() != expected:
                    raise ValueError(
                        f"case {case_id} static workbook and JSON {column} differ"
                    )
            if parse_json_cell(
                workbook_case.get("Final Listed PCs (target->entry)")
            ) != final["pcs32_target_to_entry"]:
                raise ValueError(
                    f"case {case_id} static workbook and JSON final PCs differ"
                )
            if parse_json_cell(
                workbook_case.get("Fuzzer target_pcs (entry->target)")
            ) != final["fuzzer_pcs32_entry_to_target"]:
                raise ValueError(
                    f"case {case_id} static workbook and JSON fuzzer PCs differ"
                )
    new_sheets = (
        "Agentic_Extraction",
        "Agentic_Waypoints",
        "Evaluation",
        "Evaluation_Waypoints",
        "Rule_Level_Summary",
        "Coverage_Runs",
    )
    for sheet_name in new_sheets:
        if sheet_name in workbook.sheetnames:
            workbook.remove(workbook[sheet_name])
    agent_cases, agent_waypoints = extraction_rows(payload)
    append_mapping_sheet(workbook, "Agentic_Extraction", agent_cases)
    append_mapping_sheet(workbook, "Agentic_Waypoints", agent_waypoints)
    append_mapping_sheet(workbook, "Evaluation", evaluation_rows(payload))
    append_mapping_sheet(
        workbook, "Evaluation_Waypoints", evaluation_waypoint_rows(payload)
    )
    rule_rows = summary_rows(payload)
    append_mapping_sheet(workbook, "Rule_Level_Summary", rule_rows)
    coverage_rows = [
        {key: excel_value(value) for key, value in row.items()}
        for row in payload["coverage_runs"]
    ]
    append_mapping_sheet(workbook, "Coverage_Runs", coverage_rows)

    if "Run_Metadata" in workbook.sheetnames:
        remove_metadata_keys(
            workbook,
            {
                "Agentic Weight",
                "Dynamic Weight",
                "Weighted Score Role",
                "Positive-Evidence Sensitivity Formula",
                "Semantic Weight",
                "Hit Quality Weight",
                "Effective Length Weight",
                "Quality Formula",
            },
        )
        metadata = workbook["Run_Metadata"]
        set_metadata_value(workbook, "Evaluation Schema Version", SCHEMA_VERSION)
        set_metadata_value(
            workbook,
            "Agentic Extraction Schema Version",
            AGENTIC_EXTRACTION_SCHEMA_VERSION,
        )
        set_quality_metadata(workbook, weights)
        style_sheet(metadata)
    return compact_workbook(workbook)


def prepare_command(args: argparse.Namespace) -> int:
    if args.output.resolve() in {
        args.static_json.resolve(),
        args.agentic_extractions.resolve(),
    }:
        raise ValueError("prepare output must not overwrite an input artifact")
    static = load_static(args.static_json)
    extractions = load_extraction_records(args.agentic_extractions)
    seed = args.blind_seed or static.get("run_metadata", {}).get("Run ID", "default")
    payload = build_scoring_input(static, extractions, seed)
    write_json_atomic(args.output, payload)
    print(f"Wrote {args.output}")
    return 0


def build_command(args: argparse.Namespace) -> int:
    weights = quality_weights_from_args(args)
    static = load_static(args.static_json)
    scoring_input = json.loads(args.scoring_input.read_text(encoding="utf-8"))
    extractions = load_extraction_records(args.agentic_extractions)
    scores = load_score_records(args.blind_scores)
    json_output = args.json_output or args.output.with_suffix(".json")
    input_paths = {
        args.static_json.resolve(),
        args.static_workbook.resolve(),
        args.agentic_extractions.resolve(),
        args.scoring_input.resolve(),
        args.blind_scores.resolve(),
    }
    if args.coverage_results is not None:
        input_paths.add(args.coverage_results.resolve())
    if args.output.resolve() == json_output.resolve():
        raise ValueError("XLSX and JSON outputs must be distinct")
    if args.output.resolve() in input_paths or json_output.resolve() in input_paths:
        raise ValueError("build outputs must not overwrite input artifacts")
    temporary_json = json_output.with_name(f".{json_output.name}.writing")
    temporary_xlsx = args.output.with_name(
        f".{args.output.stem}.writing{args.output.suffix}"
    )
    output_paths = {
        args.output.resolve(),
        json_output.resolve(),
        temporary_json.resolve(),
        temporary_xlsx.resolve(),
    }
    if len(output_paths) != 4 or output_paths & input_paths:
        raise ValueError("build final and temporary paths must all be distinct")
    validate_scoring_bundle(static, scoring_input, extractions, scores)
    payload = build_evaluation(
        static,
        scoring_input,
        extractions,
        scores,
        args.coverage_results,
        weights,
    )
    workbook = build_workbook_view(args.static_workbook, static, payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    workbook.save(temporary_xlsx)
    os.replace(temporary_json, json_output)
    os.replace(temporary_xlsx, args.output)
    readme = write_workbook_readme(workbook, args.output)
    print(f"Wrote {json_output}")
    print(f"Wrote {args.output}")
    print(f"Wrote {readme}")
    return 0


def compact_command(args: argparse.Namespace) -> int:
    if not args.input.is_file():
        raise FileNotFoundError(f"workbook not found: {args.input}")
    destination = args.output or args.input
    workbook = compact_workbook(load_workbook(args.input))
    save_workbook_atomic(workbook, destination)
    readme = write_workbook_readme(workbook, destination)
    manifest_path = destination.parent / "manifest.json"
    if manifest_path.is_file():
        original_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        workbook_artifact = original_manifest.get("artifacts", {}).get(
            "evaluation_workbook"
        )
        recorded_path = (
            Path(str(workbook_artifact.get("path", "")))
            if isinstance(workbook_artifact, dict)
            else None
        )
        if recorded_path is not None and not recorded_path.is_absolute():
            recorded_path = PROJECT_ROOT / recorded_path
        if (
            isinstance(workbook_artifact, dict)
            and recorded_path is not None
            and recorded_path.resolve() == destination.resolve()
        ):
            manifest = original_manifest
            workbook_artifact = manifest["artifacts"]["evaluation_workbook"]
            workbook_artifact["size"] = destination.stat().st_size
            write_json_atomic(manifest_path, manifest)
    print(f"Wrote {destination}")
    print(f"Wrote {readme}")
    return 0


def validate_reporting_refresh_inputs(
    workbook: Workbook, payload: dict[str, Any]
) -> None:
    required_sheets = {"Run_Metadata", "Cases", "Evaluation"}
    missing = sorted(required_sheets - set(workbook.sheetnames))
    if missing:
        raise ValueError(f"workbook is missing refresh source sheets: {missing}")
    metadata = {
        str(row[0]): row[1]
        for row in workbook["Run_Metadata"].iter_rows(
            min_row=2, values_only=True
        )
        if row and row[0]
    }
    static_run_id = str(payload.get("provenance", {}).get("static_run_id", ""))
    if not static_run_id or str(metadata.get("Run ID", "")) != static_run_id:
        raise ValueError("evaluation JSON and workbook have different Run IDs")

    case_rows = workbook["Cases"].iter_rows(values_only=True)
    case_headers = list(next(case_rows))
    if "ID" not in case_headers:
        raise ValueError("Cases sheet lacks ID column")
    workbook_case_ids = {
        str(row[case_headers.index("ID")]) for row in case_rows
    }
    payload_case_ids = {
        str(record["case_id"])
        for record in payload.get("agentic_extractions", [])
    }
    if workbook_case_ids != payload_case_ids:
        raise ValueError("evaluation JSON and workbook have different case sets")

    evaluation_rows_iter = workbook["Evaluation"].iter_rows(values_only=True)
    evaluation_headers = list(next(evaluation_rows_iter))
    if not {"ID", "Method Key"}.issubset(evaluation_headers):
        raise ValueError("Evaluation sheet lacks ID or Method Key column")
    workbook_candidates = {
        (
            str(row[evaluation_headers.index("ID")]),
            str(row[evaluation_headers.index("Method Key")]),
        )
        for row in evaluation_rows_iter
    }
    payload_candidates = {
        (str(item["case_id"]), str(item["method_key"]))
        for item in payload.get("evaluations", [])
    }
    if workbook_candidates != payload_candidates:
        raise ValueError(
            "evaluation JSON and workbook have different candidate sets"
        )


def replace_reporting_sheet(
    workbook: Workbook,
    sheet_name: str,
    rows: list[dict[str, Any]],
    index: int,
) -> None:
    sheet = workbook.create_sheet(sheet_name, index)
    columns = PUBLISHED_WORKBOOK_COLUMNS[sheet_name]
    sheet.append(list(columns))
    for row in rows:
        sheet.append([row.get(column) for column in columns])
    style_sheet(sheet)


def manifest_artifact_matches_path(artifact: Any, path: Path) -> bool:
    if not isinstance(artifact, dict) or not artifact.get("path"):
        return False
    recorded = Path(str(artifact["path"]))
    if not recorded.is_absolute():
        recorded = PROJECT_ROOT / recorded
    return recorded.resolve() == path.resolve()


def refresh_reporting_command(args: argparse.Namespace) -> int:
    """Refresh reporting metrics from an existing canonical JSON artifact."""
    if not args.json.is_file() or not args.workbook.is_file():
        raise FileNotFoundError("evaluation JSON and workbook must both exist")
    payload = json.loads(args.json.read_text(encoding="utf-8"))
    if payload.get("artifact_type") != "waypoint_quality_evaluation":
        raise ValueError("JSON is not a waypoint quality evaluation artifact")
    weights = quality_weights_from_args(args)
    payload = enrich_reporting_metrics(payload, weights)
    workbook = load_workbook(args.workbook)
    validate_reporting_refresh_inputs(workbook, payload)
    evaluation_index = workbook.sheetnames.index("Evaluation")
    rule_index = workbook.sheetnames.index("Rule_Level_Summary")
    workbook.remove(workbook["Evaluation"])
    workbook.remove(workbook["Rule_Level_Summary"])
    replace_reporting_sheet(
        workbook, "Evaluation", evaluation_rows(payload), evaluation_index
    )
    replace_reporting_sheet(
        workbook, "Rule_Level_Summary", summary_rows(payload), rule_index
    )
    remove_metadata_keys(
        workbook,
        {
            "Agentic Weight",
            "Dynamic Weight",
            "Weighted Score Role",
            "Positive-Evidence Sensitivity Formula",
        },
    )
    set_metadata_value(workbook, "Evaluation Schema Version", SCHEMA_VERSION)
    set_quality_metadata(workbook, weights)

    output_dir = getattr(args, "output_dir", None)
    if output_dir is not None:
        json_output = output_dir / args.json.name
        workbook_output = output_dir / args.workbook.name
    else:
        json_output = args.json
        workbook_output = args.workbook
    temporary_json = json_output.with_name(f".{json_output.name}.reporting")
    temporary_workbook = workbook_output.with_name(
        f".{workbook_output.stem}.reporting{workbook_output.suffix}"
    )
    if output_dir is not None:
        source_paths = {args.json.resolve(), args.workbook.resolve()}
        destination_paths = {
            json_output.resolve(),
            workbook_output.resolve(),
            temporary_json.resolve(),
            temporary_workbook.resolve(),
        }
        if len(destination_paths) != 4 or source_paths & destination_paths:
            raise ValueError(
                "--output-dir artifacts must not overwrite source or each other"
            )
        output_dir.mkdir(parents=True, exist_ok=True)
    temporary_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    workbook.save(temporary_workbook)
    os.replace(temporary_json, json_output)
    os.replace(temporary_workbook, workbook_output)
    readme = write_workbook_readme(workbook, workbook_output)

    manifest_path = workbook_output.parent / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key, path in (
            ("evaluation_json", json_output),
            ("evaluation_workbook", workbook_output),
        ):
            artifact = manifest.get("artifacts", {}).get(key)
            if manifest_artifact_matches_path(artifact, path):
                artifact["size"] = path.stat().st_size
        write_json_atomic(manifest_path, manifest)
    print(f"Wrote {json_output}")
    print(f"Wrote {workbook_output}")
    print(f"Wrote {readme}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare blind waypoint scoring and build canonical evaluation artifacts."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="Build blind scoring input.")
    prepare.add_argument("--static-json", required=True, type=Path)
    prepare.add_argument("--agentic-extractions", required=True, type=Path)
    prepare.add_argument("--output", required=True, type=Path)
    prepare.add_argument("--blind-seed")
    prepare.set_defaults(handler=prepare_command)

    build = subparsers.add_parser("build", help="Build final JSON and XLSX outputs.")
    build.add_argument("--static-json", required=True, type=Path)
    build.add_argument("--static-workbook", required=True, type=Path)
    build.add_argument("--agentic-extractions", required=True, type=Path)
    build.add_argument("--scoring-input", required=True, type=Path)
    build.add_argument("--blind-scores", required=True, type=Path)
    build.add_argument("--coverage-results", type=Path)
    build.add_argument("--output", required=True, type=Path)
    build.add_argument("--json-output", type=Path)
    add_quality_weight_arguments(build)
    build.set_defaults(handler=build_command)

    compact = subparsers.add_parser(
        "compact", help="Compact an existing evaluation workbook and write README.md."
    )
    compact.add_argument("--input", required=True, type=Path)
    compact.add_argument("--output", type=Path)
    compact.set_defaults(handler=compact_command)

    refresh = subparsers.add_parser(
        "refresh-reporting",
        help="Refresh summary metrics from existing evaluation JSON without agents or QEMU.",
    )
    refresh.add_argument("--json", required=True, type=Path)
    refresh.add_argument("--workbook", required=True, type=Path)
    refresh.add_argument(
        "--output-dir",
        type=Path,
        help="Write a sensitivity variant without replacing the source artifacts.",
    )
    add_quality_weight_arguments(refresh)
    refresh.set_defaults(handler=refresh_reporting_command)
    args = parser.parse_args()
    try:
        quality_weights_from_args(args)
    except ValueError as error:
        parser.error(str(error))
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
