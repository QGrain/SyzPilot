"""Task-local fusion of stage-aware attribution evidence."""

from dataclasses import dataclass
import math
from numbers import Real
from typing import Dict, Iterable, Mapping, Optional, Tuple


_STAGE_WEIGHTS = {1: 0.15, 2: 0.40, 3: 1.00}
_TTL_MULTIPLIERS = (1.0, 0.5, 0.25)
_LOWER_STAGE_CAP = 0.25
_MIN_BUCKET_SAMPLES = 8


def _normalized(scores: Mapping[str, float]) -> Dict[str, float]:
    positive = {
        name: float(score)
        for name, score in scores.items()
        if (isinstance(name, str) and name and
            not isinstance(score, bool) and isinstance(score, Real) and
            math.isfinite(float(score)) and score > 0)
    }
    norm = math.sqrt(sum(score * score for score in positive.values()))
    if norm == 0:
        return {}
    return {name: score / norm for name, score in positive.items()}


@dataclass(frozen=True)
class AttributionSnapshot:
    """One successfully analyzed curriculum objective."""

    stage: int
    objective: str
    canonical_members: Tuple[int, ...]
    active_exact_classes: Tuple[int, ...]
    deployment_version: int
    selected_signature_digest: str
    analyzed_count: int
    examined_count: int
    depth_weight: float
    scores: Dict[str, float]

    @classmethod
    def from_result(cls, result: Mapping[str, object]):
        snapshot = cls(
            stage=int(result["stage"]),
            objective=str(result["objective"]),
            canonical_members=tuple(int(value) for value in result[
                "canonical_members"
            ]),
            active_exact_classes=tuple(int(value) for value in result.get(
                "active_exact_classes", ()
            )),
            deployment_version=int(result["deployment_version"]),
            selected_signature_digest=str(result[
                "selected_signature_digest"
            ]),
            analyzed_count=int(result["analyzed_count"]),
            examined_count=int(result.get(
                "examined_count", result["analyzed_count"]
            )),
            depth_weight=float(result["depth_weight"]),
            scores=_normalized(result["scores"]),
        )
        if snapshot.stage not in _STAGE_WEIGHTS:
            raise ValueError("invalid attribution stage")
        if (not snapshot.objective or not snapshot.canonical_members or
                snapshot.analyzed_count < _MIN_BUCKET_SAMPLES or
                snapshot.examined_count < snapshot.analyzed_count or
                not math.isfinite(snapshot.depth_weight) or
                snapshot.depth_weight <= 0 or not snapshot.scores):
            raise ValueError("invalid attribution snapshot")
        return snapshot


@dataclass(frozen=True)
class FusedAttribution:
    """Normalized syscall scores and their most specific supporting stage."""

    scores: Dict[str, float]
    specificity: Dict[str, int]


class HierarchicalAttributionStore:
    """Keep bounded attribution snapshots for one fuzzing task."""

    def __init__(self):
        self._deployment_version = 0
        self._current_stage = 0
        self._active_exact_classes: Tuple[int, ...] = ()
        self._snapshots: Dict[Tuple[int, str], AttributionSnapshot] = {}

    @property
    def deployment_version(self) -> int:
        return self._deployment_version

    def begin_deployment(
            self, deployment_version: int, stage: int,
            active_exact_classes: Optional[Iterable[int]] = None):
        """Advance snapshot age and drop evidence incompatible with a model."""
        version = int(deployment_version)
        if version < self._deployment_version:
            raise ValueError("deployment version cannot move backwards")
        if stage not in _STAGE_WEIGHTS:
            raise ValueError("invalid attribution stage")
        self._deployment_version = version
        self._current_stage = stage
        self._active_exact_classes = tuple(sorted(set(
            int(value) for value in active_exact_classes or ()
        )))

        if stage == 3:
            active = frozenset(self._active_exact_classes)
            self._snapshots = {
                key: snapshot
                for key, snapshot in self._snapshots.items()
                if not (
                    snapshot.stage == 3 and
                    any(member not in active
                        for member in snapshot.canonical_members)
                )
            }
        self._drop_expired()

    def replace(self, results: Iterable[Mapping[str, object]]) -> int:
        """Replace each non-empty stage/objective snapshot atomically."""
        replacements = {}
        for result in results:
            snapshot = AttributionSnapshot.from_result(result)
            if snapshot.deployment_version != self._deployment_version:
                raise ValueError("snapshot deployment version is not current")
            self._validate_current_objective(snapshot)
            replacements[(snapshot.stage, snapshot.objective)] = snapshot
        self._snapshots.update(replacements)
        return len(replacements)

    def _validate_current_objective(self, snapshot: AttributionSnapshot):
        """Reject evidence that does not describe the deployed objective."""
        if snapshot.stage != self._current_stage:
            raise ValueError("snapshot stage is not the deployed stage")
        if any(member <= 0 for member in snapshot.canonical_members):
            raise ValueError("attribution requires reached canonical classes")
        if snapshot.stage == 1:
            valid = (
                snapshot.objective == "reachable" and
                not snapshot.active_exact_classes
            )
        elif snapshot.stage == 2:
            valid = (
                snapshot.objective in ("shallow", "deep") and
                not snapshot.active_exact_classes
            )
        else:
            member = snapshot.canonical_members[0]
            valid = (
                len(snapshot.canonical_members) == 1 and
                snapshot.objective == f"exact:{member}" and
                member in self._active_exact_classes and
                snapshot.active_exact_classes == self._active_exact_classes
            )
        if not valid:
            raise ValueError("snapshot does not match the deployed objective")

    def _age_multiplier(self, snapshot: AttributionSnapshot) -> float:
        age = self._deployment_version - snapshot.deployment_version
        if age < 0 or age >= len(_TTL_MULTIPLIERS):
            return 0.0
        return _TTL_MULTIPLIERS[age]

    def _drop_expired(self):
        self._snapshots = {
            key: snapshot
            for key, snapshot in self._snapshots.items()
            if self._age_multiplier(snapshot) > 0
        }

    def fuse(self) -> FusedAttribution:
        """Fuse buckets while preserving objective specificity."""
        stage_vectors: Dict[int, Dict[str, float]] = {}
        stage_freshness: Dict[int, float] = {}
        for snapshot in self._snapshots.values():
            freshness = self._age_multiplier(snapshot)
            if freshness == 0:
                continue
            vector = stage_vectors.setdefault(snapshot.stage, {})
            bucket_weight = snapshot.depth_weight * freshness
            for name, score in snapshot.scores.items():
                vector[name] = vector.get(name, 0.0) + score * bucket_weight
            stage_freshness[snapshot.stage] = max(
                stage_freshness.get(snapshot.stage, 0.0), freshness
            )

        contributions = {}
        for stage, vector in stage_vectors.items():
            normalized = _normalized(vector)
            scale = _STAGE_WEIGHTS[stage] * stage_freshness[stage]
            contributions[stage] = {
                name: score * scale for name, score in normalized.items()
            }

        fused = {}
        specificity = {}
        names = set().union(*(vector for vector in contributions.values()))
        for name in names:
            supporting = [
                stage for stage, vector in contributions.items()
                if vector.get(name, 0.0) > 0
            ]
            highest = max(supporting)
            primary = contributions[highest][name]
            lower = sum(
                vector.get(name, 0.0)
                for stage, vector in contributions.items()
                if stage < highest
            )
            fused[name] = primary + min(lower, primary * _LOWER_STAGE_CAP)
            specificity[name] = highest

        return FusedAttribution(_normalized(fused), specificity)

    def publication_scores(
            self, allowed_names: Optional[Iterable[str]] = None,
            limit: Optional[int] = None) -> Dict[str, float]:
        """Return specificity-first scores for the existing guidance API."""
        fused = self.fuse()
        allowed = None if allowed_names is None else frozenset(allowed_names)
        ordered = sorted(
            (
                (name, score) for name, score in fused.scores.items()
                if allowed is None or name in allowed
            ),
            key=lambda item: (
                -fused.specificity[item[0]], -item[1], item[0]
            ),
        )
        if limit is not None:
            ordered = ordered[:limit]
        return dict(ordered)

    def audit_state(self):
        """Return concise metadata for logs and tests."""
        state = []
        for snapshot in sorted(
                self._snapshots.values(),
                key=lambda item: (item.stage, item.objective)):
            state.append({
                "stage": snapshot.stage,
                "objective": snapshot.objective,
                "deployment_version": snapshot.deployment_version,
                "effective_multiplier": self._age_multiplier(snapshot),
                "analyzed_count": snapshot.analyzed_count,
                "examined_count": snapshot.examined_count,
                "canonical_members": list(snapshot.canonical_members),
                "selected_signature_digest": (
                    snapshot.selected_signature_digest
                ),
            })
        return state
