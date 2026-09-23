"""Shared schemas for reproducible script and agentic waypoint evaluation."""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


SCHEMA_VERSION = "2.0"
AGENTIC_EXTRACTION_SCHEMA_VERSION = "2.2"
BLIND_SCORE_SCHEMA_VERSION = "2.2"

ExtractionStatus = Literal["ok", "missing_input", "failed"]
Confidence = Literal["high", "medium", "low"]
CausalPhase = Literal[
    "configured_target",
    "trigger",
    "trigger_path",
    "memory_access",
    "object_deallocation",
    "deallocation_handoff",
    "object_allocation",
    "allocation_origin",
    "syscall_entry",
    "reported_path",
    "unknown",
]
TargetFidelity = Literal[
    "exact",
    "resolvable_proxy",
    "function_only",
    "unsupported",
]
SectionFidelity = Literal[
    "complete",
    "minor_omission",
    "partial",
    "mixed_or_wrong",
]
CausalCoherence = Literal[
    "coherent",
    "mostly_coherent",
    "partial",
    "incoherent",
]
Parsimony = Literal[
    "concise",
    "minor_redundancy",
    "substantial_redundancy",
    "unusable",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AgenticWaypoint(StrictModel):
    target: str = Field(description="func@relative/source/path:line")
    causal_phase: CausalPhase
    report_evidence: str
    proxy_reason: str

    @field_validator("target")
    @classmethod
    def validate_target(cls, value: str) -> str:
        match = re.fullmatch(r"([^@\s]+)@([^:\s]+):(\d+)", value)
        if match is None:
            raise ValueError("target must use func@file:line format")
        source_path = PurePosixPath(match.group(2))
        if source_path.is_absolute() or ".." in source_path.parts:
            raise ValueError("target source path must be relative and normalized")
        if int(match.group(3)) < 1:
            raise ValueError("target source line must be positive")
        return value


class AgenticExtractionDecision(StrictModel):
    status: ExtractionStatus
    report_kind: str
    concurrency_class: str
    confidence: Confidence
    waypoints_target_to_entry: list[AgenticWaypoint]
    rationale: str
    unresolved_questions: list[str]

    @field_validator("waypoints_target_to_entry")
    @classmethod
    def validate_waypoint_count(
        cls, value: list[AgenticWaypoint]
    ) -> list[AgenticWaypoint]:
        if len(value) > 10:
            raise ValueError("agentic waypoint chain exceeds K_max=10")
        return value

    @model_validator(mode="after")
    def validate_status_chain(self) -> "AgenticExtractionDecision":
        if self.status == "ok" and not self.waypoints_target_to_entry:
            raise ValueError("ok extraction must contain at least one waypoint")
        if self.status != "ok" and self.waypoints_target_to_entry:
            raise ValueError("non-ok extraction must not contain waypoints")
        configured_targets = sum(
            waypoint.causal_phase == "configured_target"
            for waypoint in self.waypoints_target_to_entry
        )
        if self.status == "ok" and configured_targets != 1:
            raise ValueError(
                "ok extraction must identify exactly one configured target"
            )
        if (
            self.status == "ok"
            and self.waypoints_target_to_entry[0].causal_phase
            != "configured_target"
        ):
            raise ValueError(
                "target-to-entry chain must place the configured target first"
            )
        return self


class AgenticTargetResolution(StrictModel):
    proposed_target: str
    resolved_target: str
    pc64: str
    pc32: str

    @model_validator(mode="after")
    def validate_nonzero_pcs(self) -> "AgenticTargetResolution":
        if re.fullmatch(r"0x[0-9a-fA-F]{16}", self.pc64) is None:
            raise ValueError("pc64 must contain exactly 16 hexadecimal digits")
        if re.fullmatch(r"0x[0-9a-fA-F]{8}", self.pc32) is None:
            raise ValueError("pc32 must contain exactly 8 hexadecimal digits")
        if int(self.pc64, 16) == 0 or int(self.pc32, 16) == 0:
            raise ValueError("configured target PCs must be nonzero")
        if int(self.pc64, 16) & 0xFFFFFFFF != int(self.pc32, 16):
            raise ValueError("pc32 must be the low 32 bits of pc64")
        return self


class AgenticExtractionRecord(StrictModel):
    schema_version: Literal["2.2"] = AGENTIC_EXTRACTION_SCHEMA_VERSION
    method: Literal["agentic"] = "agentic"
    case_id: str
    title: str
    bug_position: str
    model: str
    effort: str
    codex_sdk_version: str
    thread_id: str
    decision: AgenticExtractionDecision
    configured_target_resolution: AgenticTargetResolution | None = None

    @field_validator("case_id")
    @classmethod
    def validate_case_id(cls, value: str) -> str:
        if not value.isdigit():
            raise ValueError("case_id must contain decimal digits only")
        return value

    @model_validator(mode="after")
    def validate_target_resolution(self) -> "AgenticExtractionRecord":
        if self.decision.status == "ok":
            if self.configured_target_resolution is None:
                raise ValueError(
                    "ok extraction must include configured target resolution"
                )
            proposed = self.decision.waypoints_target_to_entry[0].target
            if self.configured_target_resolution.proposed_target != proposed:
                raise ValueError(
                    "configured target resolution does not match the first waypoint"
                )
        elif self.configured_target_resolution is not None:
            raise ValueError(
                "non-ok extraction must not include configured target resolution"
            )
        return self


class CandidateSemanticJudgment(StrictModel):
    candidate_id: str
    target_fidelity: TargetFidelity
    section_fidelity: SectionFidelity
    causal_coherence: CausalCoherence
    parsimony: Parsimony
    confidence: Confidence
    evidence: list[str]
    rationale: str
    unresolved_questions: list[str]


class BlindScoreDecision(StrictModel):
    reviews: list[CandidateSemanticJudgment]


class ScoringCandidateInput(StrictModel):
    """Candidate input with a private method key excluded from the agent view."""

    candidate_id: str
    method_key: str
    waypoints_target_to_entry: list[str]

    @field_validator("candidate_id")
    @classmethod
    def validate_candidate_id(cls, value: str) -> str:
        if re.fullmatch(r"c_[0-9a-f]{16}", value) is None:
            raise ValueError("candidate_id must be an opaque c_<16 hex> identifier")
        return value


class ScoringCaseInput(StrictModel):
    case_id: str
    title: str
    bug_position: str
    report_path: str
    kernel_dir: str
    candidates: list[ScoringCandidateInput]

    @field_validator("case_id")
    @classmethod
    def validate_case_id(cls, value: str) -> str:
        if not value.isdigit():
            raise ValueError("case_id must contain decimal digits only")
        return value

    @model_validator(mode="after")
    def validate_unique_candidates(self) -> "ScoringCaseInput":
        candidate_ids = [candidate.candidate_id for candidate in self.candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("candidate_id values must be unique within a case")
        if not candidate_ids:
            raise ValueError("each scoring case must contain at least one candidate")
        return self


class BlindScoreRecord(StrictModel):
    schema_version: Literal["2.2"] = BLIND_SCORE_SCHEMA_VERSION
    case_id: str
    model: str
    effort: str
    codex_sdk_version: str
    thread_id: str
    candidate_method_map: dict[str, str]
    candidate_chain_map: dict[str, list[str]]
    status: Literal["ok", "failed"] = "ok"
    error: str = ""
    decision: BlindScoreDecision | None

    @field_validator("case_id")
    @classmethod
    def validate_case_id(cls, value: str) -> str:
        if not value.isdigit():
            raise ValueError("case_id must contain decimal digits only")
        return value

    @model_validator(mode="after")
    def validate_status_decision(self) -> "BlindScoreRecord":
        if self.status == "ok" and self.decision is None:
            raise ValueError("ok scoring record must contain a decision")
        if self.status == "failed" and self.decision is not None:
            raise ValueError("failed scoring record must not contain a decision")
        return self


TARGET_SCORES: dict[str, int] = {
    "exact": 30,
    "resolvable_proxy": 24,
    "function_only": 12,
    "unsupported": 0,
}
SECTION_SCORES: dict[str, int] = {
    "complete": 20,
    "minor_omission": 16,
    "partial": 8,
    "mixed_or_wrong": 0,
}
CAUSAL_SCORES: dict[str, int] = {
    "coherent": 25,
    "mostly_coherent": 20,
    "partial": 10,
    "incoherent": 0,
}
PARSIMONY_SCORES: dict[str, int] = {
    "concise": 10,
    "minor_redundancy": 7,
    "substantial_redundancy": 3,
    "unusable": 0,
}


def semantic_score(
    judgment: CandidateSemanticJudgment, observability_score: float
) -> float:
    """Map categorical agent judgments plus deterministic observability to 0-100."""
    if not 0 <= observability_score <= 15:
        raise ValueError("observability_score must be in [0, 15]")
    score = (
        TARGET_SCORES[judgment.target_fidelity]
        + SECTION_SCORES[judgment.section_fidelity]
        + CAUSAL_SCORES[judgment.causal_coherence]
        + PARSIMONY_SCORES[judgment.parsimony]
        + observability_score
    )
    return round(float(score), 1)
