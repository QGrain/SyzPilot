"""
SequencePatternMiner: Mine common syscall sequences from reaching programs.

Identifies frequent syscall subsequences that correlate with reaching
target waypoints. Generates mutation templates for the fuzzer.
"""

import logging
import io
import math
import os
import re
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
from common.label_contract import one_hot_label_class

logger = logging.getLogger(__name__)

_SYSCALL_LINE_RE = re.compile(
    r"^(?:r[0-9]+\s*=\s*)?"
    r"([A-Za-z_][A-Za-z0-9_]*(?:\$[A-Za-z0-9_]+)*)\s*\("
)


def extract_syscalls_from_program(prog_text: str) -> List[str]:
    """Extract ordered syscall names from top-level syz-program calls.

    Result assignments are recognized only at the start of a line. Argument
    expressions commonly contain ``=`` and must never be treated as the
    assignment delimiter for the enclosing call.
    """
    syscalls, _ = _extract_syscalls_with_deadline(prog_text, None)
    return syscalls


def _extract_syscalls_with_deadline(
    prog_text: str, deadline: Optional[float]
) -> Tuple[List[str], bool]:
    """Extract calls while honoring a best-effort monotonic deadline."""
    syscalls = []
    for line_index, line in enumerate(io.StringIO(prog_text)):
        if (deadline is not None and line_index % 256 == 0 and
                time.monotonic() >= deadline):
            return syscalls, True
        line = line.strip()
        match = _SYSCALL_LINE_RE.match(line)
        if match is not None:
            syscalls.append(match.group(1))
    return syscalls, False


class SequencePatternMiner:
    """Mine common syscall sequences from reaching programs."""

    def __init__(
        self,
        programs: List[str],
        labels: List[List[bool]],
        num_classes: int = None,
    ):
        """
        Args:
            programs: List of syz-program texts
            labels: List of reachability labels (one-hot bool lists)
            num_classes: Expected label width, if known
        """
        self.programs = programs
        self.labels = labels
        self.num_classes = num_classes

    def _partition_sequences(
        self, target_class: int, deadline: Optional[float]
    ) -> Tuple[List[List[str]], List[List[str]], bool]:
        """Return valid positive and comparison syscall sequences.

        Labels encode the deepest reached waypoint. A class threshold therefore
        represents reaching that waypoint *or a deeper one*. ``target_class=-1``
        retains the Stage-1 interpretation of any reached waypoint versus
        Unreachable.
        """
        if len(self.programs) != len(self.labels):
            raise ValueError("program and label counts differ")
        if target_class == 0:
            raise ValueError("target_class 0 denotes Unreachable")
        if self.num_classes is not None and target_class >= self.num_classes:
            raise ValueError("target_class is outside the label width")

        positives = []
        comparisons = []
        for program_index, (program, label) in enumerate(zip(
            self.programs, self.labels
        )):
            if (deadline is not None and program_index % 64 == 0 and
                    time.monotonic() >= deadline):
                return positives, comparisons, True
            selected_class = one_hot_label_class(label, self.num_classes)
            if selected_class is None:
                continue
            sequence, expired = _extract_syscalls_with_deadline(
                program, deadline
            )
            if expired:
                return positives, comparisons, True
            if not sequence:
                continue
            is_positive = (
                selected_class > 0 if target_class < 0
                else selected_class >= target_class
            )
            (positives if is_positive else comparisons).append(sequence)
        return positives, comparisons, False

    @staticmethod
    def _initial_projections(
        sequences: List[List[str]], deadline: Optional[float],
    ) -> Tuple[Dict[str, Dict[int, Tuple[int, ...]]], bool]:
        projections = defaultdict(lambda: defaultdict(list))
        for sequence_index, sequence in enumerate(sequences):
            if (deadline is not None and sequence_index % 64 == 0 and
                    time.monotonic() >= deadline):
                return {}, True
            for position, syscall in enumerate(sequence):
                if (deadline is not None and position % 256 == 0 and
                        time.monotonic() >= deadline):
                    return {}, True
                projections[syscall][sequence_index].append(position)
        return ({
            syscall: {
                sequence_index: tuple(positions)
                for sequence_index, positions in by_sequence.items()
            }
            for syscall, by_sequence in projections.items()
        }, False)

    @staticmethod
    def _extend_all(
        projection: Dict[int, Tuple[int, ...]],
        sequences: List[List[str]],
        max_gap: int,
        deadline: Optional[float],
    ) -> Tuple[Dict[str, Dict[int, Tuple[int, ...]]], bool]:
        """Project every one-syscall extension under the paper's gap rule."""
        extensions = defaultdict(lambda: defaultdict(set))
        for projection_index, (sequence_index, end_positions) in enumerate(
            projection.items()
        ):
            if (deadline is not None and projection_index % 64 == 0 and
                    time.monotonic() >= deadline):
                return {}, True
            sequence = sequences[sequence_index]
            for end_index, end_position in enumerate(end_positions):
                if (deadline is not None and end_index % 256 == 0 and
                        time.monotonic() >= deadline):
                    return {}, True
                # delta_max counts intervening calls, so the next call may be
                # at end_position + max_gap + 1.
                stop = min(len(sequence), end_position + max_gap + 2)
                for next_position in range(end_position + 1, stop):
                    extensions[sequence[next_position]][sequence_index].add(
                        next_position
                    )
        return ({
            syscall: {
                sequence_index: tuple(sorted(positions))
                for sequence_index, positions in by_sequence.items()
            }
            for syscall, by_sequence in extensions.items()
        }, False)

    @staticmethod
    def _extend_one(
        projection: Dict[int, Tuple[int, ...]],
        sequences: List[List[str]],
        max_gap: int,
        next_syscall: str,
        deadline: Optional[float],
    ) -> Tuple[Dict[int, Tuple[int, ...]], bool]:
        extension = {}
        for projection_index, (sequence_index, end_positions) in enumerate(
            projection.items()
        ):
            if (deadline is not None and projection_index % 64 == 0 and
                    time.monotonic() >= deadline):
                return {}, True
            sequence = sequences[sequence_index]
            next_positions = set()
            for end_index, end_position in enumerate(end_positions):
                if (deadline is not None and end_index % 256 == 0 and
                        time.monotonic() >= deadline):
                    return {}, True
                stop = min(len(sequence), end_position + max_gap + 2)
                for next_position in range(end_position + 1, stop):
                    if sequence[next_position] == next_syscall:
                        next_positions.add(next_position)
            if next_positions:
                extension[sequence_index] = tuple(sorted(next_positions))
        return extension, False

    def mine_frequent_subsequences(
        self,
        target_class: int = -1,
        min_support: float = 0.3,
        max_gap: int = 3,
        min_length: int = 2,
        max_length: int = 10,
        max_patterns: int = 64,
        min_programs: int = 1,
        min_support_gain: float = 0.0,
        min_lift: float = 1.0,
        max_frontier: int = 4096,
        deadline_seconds: Optional[float] = 10.0,
    ) -> List[Tuple[Tuple[str, ...], float]]:
        """Find frequent subsequences in programs that reach target class.

        Uses bounded breadth-first pattern growth with projected end positions.

        Args:
            target_class: Reached-class threshold (-1 = any reached class)
            min_support: Minimum frequency to include
            max_gap: Max gap between items in subsequence
            min_length: Minimum subsequence length
            max_length: Maximum subsequence length
            max_patterns: Maximum number of highest-support patterns to return
            min_programs: Minimum absolute number of supporting positives
            min_support_gain: Minimum positive-minus-comparison support
            min_lift: Minimum positive/comparison support ratio
            max_frontier: Maximum frequent prefixes retained at each depth
            deadline_seconds: Optional monotonic runtime budget

        Returns:
            List of (subsequence_tuple, support_score) sorted by support
        """
        if min_length < 1 or max_length < min_length:
            raise ValueError("invalid subsequence length bounds")
        if max_gap < 0:
            raise ValueError("max_gap must be non-negative")
        if max_patterns < 1:
            raise ValueError("max_patterns must be at least 1")
        if min_programs < 1:
            raise ValueError("min_programs must be at least 1")
        if not 0.0 <= min_support <= 1.0:
            raise ValueError("min_support must be within [0, 1]")
        if min_support_gain < 0.0 or min_lift < 0.0:
            raise ValueError("contrast thresholds must be non-negative")
        if max_frontier < 1:
            raise ValueError("max_frontier must be at least 1")
        if deadline_seconds is not None and deadline_seconds <= 0.0:
            raise ValueError("deadline_seconds must be positive")

        started = time.monotonic()
        deadline = (
            started + deadline_seconds
            if deadline_seconds is not None else None
        )
        positive_sequences, comparison_sequences, expired = (
            self._partition_sequences(target_class, deadline)
        )
        if expired:
            logger.warning(
                "[SequenceMiner] Deadline expired while parsing programs; "
                "discarding the incomplete snapshot"
            )
            return []
        if not positive_sequences:
            logger.warning(f"[SequenceMiner] No programs for target class {target_class}")
            return []

        total_positive = len(positive_sequences)
        minimum_count = max(
            min_programs,
            math.ceil(total_positive * min_support),
        )
        positive_roots, expired = self._initial_projections(
            positive_sequences, deadline
        )
        if expired:
            logger.warning(
                "[SequenceMiner] Deadline expired while indexing positives; "
                "discarding the incomplete search"
            )
            return []
        comparison_roots, expired = self._initial_projections(
            comparison_sequences, deadline
        )
        if expired:
            logger.warning(
                "[SequenceMiner] Deadline expired while indexing comparisons; "
                "discarding the incomplete search"
            )
            return []
        frontier = {}
        for syscall, positive_projection in positive_roots.items():
            if len(positive_projection) >= minimum_count:
                frontier[(syscall,)] = (
                    positive_projection,
                    comparison_roots.get(syscall, {}),
                )

        candidates = []
        truncated = False

        def score_layer(layer, length):
            if length < min_length:
                return True
            layer_candidates = []
            for pattern_index, (
                pattern, (positive_projection, comparison_projection)
            ) in enumerate(layer.items()):
                if (deadline is not None and pattern_index % 256 == 0 and
                        time.monotonic() >= deadline):
                    return False
                positive_support = len(positive_projection) / total_positive
                comparison_support = (
                    len(comparison_projection) / len(comparison_sequences)
                    if comparison_sequences else 0.0
                )
                support_gain = positive_support - comparison_support
                lift = (
                    positive_support / comparison_support
                    if comparison_support > 0.0 else float("inf")
                )
                if (support_gain > 0.0 and
                        support_gain >= min_support_gain and
                        lift >= min_lift):
                    layer_candidates.append((
                        pattern, support_gain, positive_support,
                        comparison_support,
                    ))
            candidates.extend(layer_candidates)
            return True

        if not score_layer(frontier, 1):
            logger.warning(
                "[SequenceMiner] Deadline expired while scoring roots; "
                "discarding the incomplete search"
            )
            return []
        for length in range(1, max_length):
            if not frontier:
                break
            if deadline is not None and time.monotonic() >= deadline:
                truncated = True
                break

            next_frontier = {}
            layer_complete = True
            for pattern in sorted(frontier):
                positive_projection, comparison_projection = frontier[pattern]
                extensions, expired = self._extend_all(
                    positive_projection, positive_sequences, max_gap, deadline
                )
                if expired:
                    layer_complete = False
                    break
                for syscall, next_positive in extensions.items():
                    if len(next_positive) < minimum_count:
                        continue
                    next_comparison, expired = self._extend_one(
                        comparison_projection,
                        comparison_sequences,
                        max_gap,
                        syscall,
                        deadline,
                    )
                    if expired:
                        layer_complete = False
                        break
                    next_frontier[pattern + (syscall,)] = (
                        next_positive, next_comparison
                    )
                if not layer_complete:
                    break

            if not layer_complete:
                truncated = True
                break

            # Score the complete layer before limiting how many prefixes may
            # consume the next expansion budget. The cap must not change the
            # current layer's contrast-ranked result set.
            if not score_layer(next_frontier, length + 1):
                truncated = True
                break

            if len(next_frontier) > max_frontier:
                truncated = True
                ranked_frontier = sorted(
                    next_frontier.items(),
                    key=lambda item: (-len(item[1][0]), item[0]),
                )[:max_frontier]
                next_frontier = dict(ranked_frontier)
            frontier = next_frontier

        candidates.sort(
            key=lambda item: (-item[1], -item[2], item[0])
        )
        results = [
            (pattern, score)
            for pattern, score, _, _ in candidates[:max_patterns]
        ]
        if truncated:
            logger.warning(
                "[SequenceMiner] Search hit a runtime/frontier bound; "
                "returning the deterministic best completed patterns"
            )
        logger.info(
            "[SequenceMiner] Mined %d patterns from %d positive and %d "
            "comparison programs in %.3fs",
            len(results), total_positive, len(comparison_sequences),
            time.monotonic() - started,
        )
        return results

    def compute_syscall_cooccurrence(self, target_class: int = -1) -> Dict[str, Dict[str, float]]:
        """Compute syscall co-occurrence matrix for reaching programs.

        Returns:
            {syscall_a: {syscall_b: cooccurrence_score, ...}, ...}
        """
        positive_sequences, _, _ = self._partition_sequences(
            target_class, None
        )

        cooccur = defaultdict(lambda: defaultdict(int))
        for seq in positive_sequences:
            unique = list(set(seq))
            for i in range(len(unique)):
                for j in range(i + 1, len(unique)):
                    cooccur[unique[i]][unique[j]] += 1
                    cooccur[unique[j]][unique[i]] += 1

        # Normalize by count of reaching programs
        total = max(1, len(positive_sequences))
        normalized = {}
        for sc_a, neighbors in cooccur.items():
            normalized[sc_a] = {sc_b: count / total for sc_b, count in neighbors.items()}

        return normalized

    def generate_templates(
        self,
        min_support: float = 0.3,
        max_gap: int = 3,
        max_templates: int = 64,
        target_class: int = -1,
        min_programs: int = 1,
        max_frontier: int = 512,
        deadline_seconds: Optional[float] = 10.0,
    ) -> List[Dict]:
        """Generate mutation templates from mined patterns.

        Returns:
            List of template dicts ready for GuidanceEngine
        """
        patterns = self.mine_frequent_subsequences(
            target_class=target_class,
            min_support=min_support,
            max_gap=max_gap,
            max_patterns=max_templates,
            min_programs=min_programs,
            max_frontier=max_frontier,
            deadline_seconds=deadline_seconds,
        )

        templates = []
        for pattern, support in patterns:
            templates.append({
                "type": "sequence",
                "syscalls": list(pattern),
                "priority": support,
                "insert_mode": "prefix",
            })

        return templates
