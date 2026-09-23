"""
GuidanceEngine: Combines guidance from multiple sources and sends to fuzzer.

Sources:
1. Static analysis (KallGraph callgraph) — cold-start bootstrapper
2. Token-level attribution (Captum IG) — model-driven
3. Sequence pattern mining — data-driven
4. PoC pattern analysis — oracle (experimental only)

The engine merges all sources with configurable weights and produces
a unified GuidancePayload for the fuzzer's /guidance endpoint.
"""

import json
import logging
import math
import time
from dataclasses import dataclass, field
from numbers import Real
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


# Report-derived exact entry calls carry operation/resource semantics that a
# primitive syscall or lexical subsystem peer does not. When such an entry is
# generatable, preserve a small fallback/exploration signal without letting the
# broader candidates consume most of the guided-choice probability mass.
_STATIC_ROLE_MULTIPLIERS = {
    "entry_exact": 1.0,
    "resource_producer": 0.65,
    "primitive_fallback": 0.25,
    "subsystem_peer": 0.10,
    "heuristic_peer": 0.10,
}
_STATIC_EXACT_BUDGET = 0.70
_MAX_MERGED_TEMPLATES = 16
_MAX_TEMPLATE_SYSCALLS = 32
_MAX_TEMPLATE_ARG_HINTS = 64
_VALID_TEMPLATE_TYPES = frozenset({"sequence", "prefix", "splice"})
_VALID_TEMPLATE_INSERT_MODES = frozenset({"prefix", "suffix", "replace"})


@dataclass
class ArgConstraint:
    """A fixed argument value for a specific syscall in a template."""
    syscall: str   # syscall name
    arg_idx: int   # 0-based argument index
    value: int     # fixed value to set

@dataclass
class MutationTemplate:
    """A mutation template for the fuzzer."""
    type: str  # "sequence", "prefix", "splice"
    syscalls: List[str]
    priority: float
    insert_mode: str = "prefix"  # "prefix", "suffix", "replace"
    arg_hints: List[ArgConstraint] = field(default_factory=list)  # argument constraints


@dataclass
class GuidanceConfig:
    """Configuration for guidance computation."""
    # Per-channel source budgets. Syscall sources and template sources are
    # normalized independently because the Fuzzer samples them in separate
    # mechanisms. The four values therefore do not need to sum to 1.0.
    static_analysis_weight: float = 0.3
    attribution_weight: float = 0.3
    sequence_weight: float = 0.2
    poc_weight: float = 0.2

    # Guidance parameters
    max_syscall_weights: int = 30  # max number of syscalls to send
    # Minimum source-relative score to include before allocating its budget.
    min_weight_threshold: float = 0.1
    guidance_decay_lambda: float = 0.03  # weight decay over time

    # Fuzzer endpoint
    fuzzer_callback_addr: str = ""  # e.g., "localhost:21007"
    fuzzer_http_port: int = 12630
    pending_ack_timeout_seconds: float = 120.0
    pending_poll_seconds: float = 5.0


class GuidanceEngine:
    """Combines all guidance sources and produces unified guidance for fuzzer."""

    def __init__(self, config: GuidanceConfig):
        self.config = config
        self.version = 0
        self.last_guidance_time = 0.0

        # Guidance sources
        self._static_weights: Dict[str, float] = {}
        self._static_seed_hints: Dict[str, float] = {}
        self._static_role_aware = False
        self._static_exact_names = frozenset()
        self._static_templates: List[MutationTemplate] = []
        self._attribution_weights: Dict[str, float] = {}
        self._sequence_templates: List[MutationTemplate] = []
        self._poc_templates: List[MutationTemplate] = []

        # Merged state
        self._current_weights: Dict[str, float] = {}
        self._current_templates: List[MutationTemplate] = []
        self._last_ack: Dict[str, object] = {}

    def update_static_analysis(self, syscall_entries: List[Dict]):
        """Update from KallGraph static analysis results.

        Args:
            syscall_entries: List of dicts with 'name', 'weight', 'path_length' keys
        """
        validated_entries = []
        for entry in syscall_entries:
            if not isinstance(entry, dict):
                logger.warning(
                    "[GuidanceEngine] Ignoring malformed static entry: %r",
                    entry,
                )
                continue
            name = entry.get("name")
            weight = entry.get("weight", 1.0)
            if (not isinstance(name, str) or not name or
                    isinstance(weight, bool) or
                    not isinstance(weight, Real) or
                    not math.isfinite(float(weight)) or weight <= 0):
                logger.warning(
                    "[GuidanceEngine] Ignoring malformed static entry: %r",
                    entry,
                )
                continue
            validated_entries.append((entry, name, float(weight)))

        has_generatable_exact = any(
            entry.get("guidance_role") == "entry_exact" and
            entry.get("delivery_mode", "generatable") == "generatable"
            for entry, _, _ in validated_entries
        )
        raw_exact_weights = {}
        raw_other_weights = {}
        raw_seed_hints = {}
        for entry, name, weight in validated_entries:
            delivery_mode = entry.get("delivery_mode", "generatable")
            if delivery_mode == "generatable":
                role = entry.get("guidance_role")
                if has_generatable_exact:
                    weight *= _STATIC_ROLE_MULTIPLIERS.get(role, 0.10)
                destination = (
                    raw_exact_weights
                    if role == "entry_exact" else raw_other_weights
                )
                destination[name] = max(destination.get(name, 0.0), weight)
            elif delivery_mode == "seed_only":
                raw_seed_hints[name] = max(
                    raw_seed_hints.get(name, 0.0), weight
                )
        if has_generatable_exact:
            # Exact evidence wins when the same syscall was also discovered by
            # a weaker heuristic role.
            raw_other_weights = {
                name: weight for name, weight in raw_other_weights.items()
                if name not in raw_exact_weights
            }
        raw_weights = {**raw_other_weights, **raw_exact_weights}
        # Path-distance scores can all be below the global merge threshold
        # (for example 1/(distance+1) <= 0.125). Normalize each source before
        # applying its configured source weight, as attribution already does.
        max_seed_hint = max(raw_seed_hints.values(), default=0.0)
        self._static_role_aware = has_generatable_exact
        if has_generatable_exact:
            exact_total = sum(raw_exact_weights.values())
            other_total = sum(raw_other_weights.values())
            exact_budget = (
                _STATIC_EXACT_BUDGET if other_total > 0 else 1.0
            )
            other_budget = 1.0 - exact_budget
            self._static_weights = {
                name: exact_budget * weight / exact_total
                for name, weight in raw_exact_weights.items()
            }
            if other_total > 0:
                self._static_weights.update({
                    name: other_budget * weight / other_total
                    for name, weight in raw_other_weights.items()
                })
        else:
            max_weight = max(raw_weights.values(), default=0.0)
            self._static_weights = {
                name: weight / max_weight
                for name, weight in raw_weights.items()
            } if max_weight > 0 else {}
        self._static_exact_names = frozenset(raw_exact_weights)
        self._static_seed_hints = {
            name: weight / max_seed_hint
            for name, weight in raw_seed_hints.items()
        } if max_seed_hint > 0 else {}
        logger.info(
            "[GuidanceEngine] Static analysis updated: %d generatable, "
            "%d seed-only syscalls, role_aware=%s",
            len(self._static_weights), len(self._static_seed_hints),
            self._static_role_aware,
        )

    def update_attribution(self, syscall_scores: Dict[str, float]):
        """Update from Captum token-level attribution.

        Args:
            syscall_scores: Dict of {syscall_name: attribution_score}
        """
        # An empty result is an explicit snapshot clear. This prevents weights
        # from a previous model or curriculum stage from being republished when
        # the current attribution round has no eligible evidence.
        valid_scores = {}
        if isinstance(syscall_scores, dict):
            for name, score in syscall_scores.items():
                if (isinstance(name, str) and name and
                        not isinstance(score, bool) and
                        isinstance(score, Real) and
                        math.isfinite(float(score)) and score > 0):
                    valid_scores[name] = float(score)
                else:
                    logger.warning(
                        "[GuidanceEngine] Ignoring malformed attribution "
                        "entry: %r=%r", name, score,
                    )

        # Empty or wholly invalid input is an explicit snapshot clear. This
        # prevents weights from a previous model or curriculum stage from
        # being republished when the current round has no eligible evidence.
        max_score = max(valid_scores.values(), default=0.0)
        self._attribution_weights = {
            name: score / max_score
            for name, score in valid_scores.items()
        } if max_score > 0 else {}
        logger.info(f"[GuidanceEngine] Attribution updated: {len(self._attribution_weights)} syscalls")

    @staticmethod
    def _positive_budget(value) -> float:
        """Return a finite positive source budget, otherwise zero."""
        if (isinstance(value, bool) or not isinstance(value, Real) or
                not math.isfinite(float(value)) or value <= 0):
            return 0.0
        return float(value)

    @staticmethod
    def _l1_allocate(scores: Dict[str, float], budget: float) -> Dict[str, float]:
        """Allocate a fixed source budget proportionally across candidates."""
        total = sum(scores.values())
        if total <= 0 or budget <= 0:
            return {}
        return {
            name: budget * score / total
            for name, score in scores.items()
        }

    @staticmethod
    def _parse_templates(templates: List[object]) -> List[MutationTemplate]:
        parsed = []
        for template in templates:
            if isinstance(template, MutationTemplate):
                syscalls = template.syscalls
                priority = template.priority
                template_type = template.type
                insert_mode = template.insert_mode
                raw_hints = template.arg_hints
            elif isinstance(template, dict):
                syscalls = template.get("syscalls", [])
                priority = template.get("priority", 0.5)
                template_type = template.get("type", "sequence")
                insert_mode = template.get("insert_mode", "prefix")
                raw_hints = template.get("arg_hints", [])
            else:
                continue
            if (not isinstance(template_type, str) or
                    template_type not in _VALID_TEMPLATE_TYPES or
                    not isinstance(insert_mode, str) or
                    insert_mode not in _VALID_TEMPLATE_INSERT_MODES or
                    not isinstance(syscalls, list) or not syscalls or
                    len(syscalls) > _MAX_TEMPLATE_SYSCALLS or
                    any(not isinstance(name, str) or not name
                        for name in syscalls) or
                    isinstance(priority, bool) or
                    not isinstance(priority, Real) or
                    not math.isfinite(float(priority)) or priority <= 0 or
                    not isinstance(raw_hints, list) or
                    len(raw_hints) > _MAX_TEMPLATE_ARG_HINTS):
                continue

            parsed_hints = []
            seen_hints = set()
            valid_hints = True
            for hint in raw_hints:
                if isinstance(hint, ArgConstraint):
                    syscall = hint.syscall
                    arg_idx = hint.arg_idx
                    value = hint.value
                elif isinstance(hint, dict):
                    syscall = hint.get("syscall")
                    arg_idx = hint.get("arg_idx")
                    value = hint.get("value")
                else:
                    valid_hints = False
                    break
                hint_key = (syscall, arg_idx)
                if (not isinstance(syscall, str) or not syscall or
                        syscall not in syscalls or
                        isinstance(arg_idx, bool) or
                        not isinstance(arg_idx, int) or arg_idx < 0 or
                        isinstance(value, bool) or
                        not isinstance(value, int) or
                        value < 0 or value >= (1 << 64) or
                        hint_key in seen_hints):
                    valid_hints = False
                    break
                seen_hints.add(hint_key)
                parsed_hints.append(ArgConstraint(
                    syscall=syscall,
                    arg_idx=arg_idx,
                    value=value,
                ))
            if not valid_hints:
                continue
            parsed_hints.sort(
                key=lambda hint: (hint.syscall, hint.arg_idx, hint.value)
            )
            parsed.append(MutationTemplate(
                type=template_type,
                syscalls=list(syscalls),
                priority=float(priority),
                insert_mode=insert_mode,
                arg_hints=parsed_hints,
            ))
        return parsed

    def update_static_templates(self, templates: List[Dict]):
        """Replace bounded report-derived resource-aware templates."""
        self._static_templates = self._parse_templates(templates)
        logger.info(
            "[GuidanceEngine] Static templates updated: %d templates",
            len(self._static_templates),
        )

    def update_sequence_patterns(self, templates: List[Dict]):
        """Update from sequence pattern mining.

        Args:
            templates: List of template dicts with 'type', 'syscalls', 'priority' keys
        """
        self._sequence_templates = self._parse_templates(templates)
        logger.info(f"[GuidanceEngine] Sequence patterns updated: {len(self._sequence_templates)} templates")

    def update_poc_patterns(self, templates: List[object]):
        """Update from explicitly enabled oracle PoC pattern analysis."""
        self._poc_templates = self._parse_templates(templates)
        logger.info(f"[GuidanceEngine] PoC patterns updated: {len(self._poc_templates)} templates")

    def compute_guidance(self) -> Dict:
        """Compute unified guidance payload for fuzzer.

        Merges all sources with configurable weights.

        Returns:
            GuidancePayload dict ready to send via POST /guidance
        """
        self.version += 1
        self.last_guidance_time = time.time()

        static_budget = self._positive_budget(
            self.config.static_analysis_weight
        )
        attribution_budget = self._positive_budget(
            self.config.attribution_weight
        )
        threshold = self.config.min_weight_threshold
        if (isinstance(threshold, bool) or not isinstance(threshold, Real) or
                not math.isfinite(float(threshold))):
            threshold = 0.0
        threshold = max(0.0, float(threshold))

        # Apply the threshold before source budgeting. Role-aware static
        # candidates are evidence-bearing and intentionally exempt: their
        # exact/setup split already bounds weak fallback candidates.
        static_candidates = {
            name: score for name, score in self._static_weights.items()
            if static_budget > 0 and score > 0 and (
                self._static_role_aware or score >= threshold
            )
        }
        attribution_candidates = {
            name: score for name, score in self._attribution_weights.items()
            if attribution_budget > 0 and score >= threshold
        }
        preliminary_static = self._l1_allocate(
            static_candidates, static_budget
        )
        preliminary_attribution = self._l1_allocate(
            attribution_candidates, attribution_budget
        )
        merged_weights = {}
        for name, weight in preliminary_static.items():
            merged_weights[name] = merged_weights.get(name, 0.0) + weight
        for name, weight in preliminary_attribution.items():
            merged_weights[name] = merged_weights.get(name, 0.0) + weight

        # Select top-K while reserving the evidence-bearing exact entry and at
        # least one setup/exploration call when both groups are available.
        sorted_weights = sorted(
            merged_weights.items(), key=lambda item: (-item[1], item[0])
        )
        max_weights = max(0, self.config.max_syscall_weights)
        selected_names = []
        if self._static_role_aware and max_weights > 0:
            exact_candidates = [
                name for name, _ in sorted_weights
                if name in self._static_exact_names
            ]
            other_static_candidates = [
                name for name, _ in sorted_weights
                if name in self._static_weights and
                name not in self._static_exact_names
            ]
            if exact_candidates:
                selected_names.append(exact_candidates[0])
            if (max_weights >= 2 and other_static_candidates and
                    len(selected_names) < max_weights):
                selected_names.append(other_static_candidates[0])
            attribution_candidates_sorted = [
                name for name, _ in sorted_weights
                if name in attribution_candidates
            ]
            selected_has_attribution = any(
                name in attribution_candidates for name in selected_names
            )
            if (max_weights >= 3 and not selected_has_attribution and
                    len(selected_names) < max_weights):
                reserved_attribution = next(
                    (name for name in attribution_candidates_sorted
                     if name not in selected_names),
                    None,
                )
                if reserved_attribution is not None:
                    selected_names.append(reserved_attribution)
            for name, _ in sorted_weights:
                if len(selected_names) >= max_weights:
                    break
                if name not in selected_names:
                    selected_names.append(name)
        else:
            # When capacity permits, retain at least one candidate from each
            # active source before filling the remaining slots by merged score.
            active_sources = [
                source for source in (
                    static_candidates, attribution_candidates
                ) if source
            ]
            if max_weights >= len(active_sources):
                for source in active_sources:
                    best = min(
                        source,
                        key=lambda name: (-merged_weights[name], name),
                    )
                    if best not in selected_names:
                        selected_names.append(best)
            for name, _ in sorted_weights:
                if len(selected_names) >= max_weights:
                    break
                if name not in selected_names:
                    selected_names.append(name)

        # Top-K can remove most members of a source. Reapply each source's
        # fixed budget over its retained candidates so candidate count cannot
        # silently change the configured source share.
        selected_static = {
            name: static_candidates[name] for name in selected_names
            if name in static_candidates
        }
        selected_attribution = {
            name: attribution_candidates[name] for name in selected_names
            if name in attribution_candidates
        }
        if self._static_role_aware:
            selected_exact = [
                name for name in selected_names
                if name in selected_static and name in self._static_exact_names
            ]
            selected_other_static = [
                name for name in selected_names
                if name in selected_static and
                name not in self._static_exact_names
            ]
            exact_total = sum(
                selected_static[name] for name in selected_exact
            )
            other_total = sum(
                selected_static[name]
                for name in selected_other_static
            )
            if exact_total > 0 and other_total > 0:
                exact_budget = _STATIC_EXACT_BUDGET
                other_budget = 1.0 - exact_budget
            elif exact_total > 0:
                exact_budget, other_budget = 1.0, 0.0
            else:
                exact_budget, other_budget = 0.0, 1.0

            allocated_static = {}
            if exact_total > 0:
                allocated_static.update({
                    name: static_budget * exact_budget *
                    selected_static[name] / exact_total
                    for name in selected_exact
                })
            if other_total > 0:
                allocated_static.update({
                    name: static_budget * other_budget *
                    selected_static[name] / other_total
                    for name in selected_other_static
                })
        else:
            allocated_static = self._l1_allocate(
                selected_static, static_budget
            )
        allocated_attribution = self._l1_allocate(
            selected_attribution, attribution_budget
        )
        selected_weights = {}
        for name, weight in allocated_static.items():
            selected_weights[name] = selected_weights.get(name, 0.0) + weight
        for name, weight in allocated_attribution.items():
            selected_weights[name] = selected_weights.get(name, 0.0) + weight
        top_weights = dict(sorted(
            selected_weights.items(),
            key=lambda item: (-item[1], item[0]),
        ))

        # Merge templates from all sources. Static templates are retained when
        # online sequence mining refreshes, then all sources are deduplicated
        # and hard-capped before crossing the Brain/Fuzzer boundary.
        def template_dict(template):
            result = {
                "type": template.type,
                "syscalls": template.syscalls,
                "priority": template.priority,
                "insert_mode": template.insert_mode,
            }
            if template.arg_hints:
                result["arg_hints"] = [
                    {
                        "syscall": hint.syscall,
                        "arg_idx": hint.arg_idx,
                        "value": hint.value,
                    }
                    for hint in template.arg_hints
                ]
            return result

        def template_key(template):
            hint_key = tuple(
                (hint["syscall"], hint["arg_idx"], hint["value"])
                for hint in template.get("arg_hints", [])
            )
            return (
                template["type"], tuple(template["syscalls"]),
                template["insert_mode"], hint_key,
            )

        template_sources = []
        for source_name, templates, source_budget in (
                ("static", self._static_templates,
                 self.config.static_analysis_weight),
                ("sequence", self._sequence_templates,
                 self.config.sequence_weight),
                ("poc", self._poc_templates, self.config.poc_weight)):
            budget = self._positive_budget(source_budget)
            if budget <= 0:
                continue
            deduplicated = {}
            for template in templates:
                candidate = template_dict(template)
                key = template_key(candidate)
                current = deduplicated.get(key)
                if (current is None or
                        candidate["priority"] > current["priority"]):
                    deduplicated[key] = candidate
            if deduplicated:
                template_sources.append((
                    source_name, budget, deduplicated
                ))

        preliminary_templates = {}
        for source_name, budget, candidates in template_sources:
            total = sum(template["priority"] for template in candidates.values())
            for key, template in candidates.items():
                entry = preliminary_templates.setdefault(key, {
                    "template": template,
                    "score": 0.0,
                    "sources": {},
                })
                entry["score"] += budget * template["priority"] / total
                entry["sources"][source_name] = template["priority"]

        sorted_template_keys = sorted(
            preliminary_templates,
            key=lambda key: (-preliminary_templates[key]["score"], key),
        )
        selected_template_keys = []
        if _MAX_MERGED_TEMPLATES >= len(template_sources):
            for source_name, _, candidates in template_sources:
                best = min(
                    candidates,
                    key=lambda key: (
                        -preliminary_templates[key]["score"], key
                    ),
                )
                if best not in selected_template_keys:
                    selected_template_keys.append(best)
        for key in sorted_template_keys:
            if len(selected_template_keys) >= _MAX_MERGED_TEMPLATES:
                break
            if key not in selected_template_keys:
                selected_template_keys.append(key)

        selected_template_priorities = {
            key: 0.0 for key in selected_template_keys
        }
        for source_name, budget, candidates in template_sources:
            selected_for_source = {
                key: candidates[key]["priority"]
                for key in selected_template_keys if key in candidates
            }
            total = sum(selected_for_source.values())
            if total <= 0:
                continue
            for key, priority in selected_for_source.items():
                selected_template_priorities[key] += (
                    budget * priority / total
                )

        merged_templates = []
        for key in selected_template_keys:
            template = dict(preliminary_templates[key]["template"])
            template["priority"] = selected_template_priorities[key]
            merged_templates.append(template)
        merged_templates.sort(key=lambda template: (
            -template["priority"],
            template["type"],
            tuple(template["syscalls"]),
            template["insert_mode"],
        ))

        # Build generation hints
        preferred_scores = dict(sorted_weights)
        for syscall, score in self._static_seed_hints.items():
            preferred_scores[syscall] = max(
                preferred_scores.get(syscall, 0.0),
                self.config.static_analysis_weight * score,
            )
        preferred = [
            syscall for syscall, _ in sorted(
                preferred_scores.items(), key=lambda item: (-item[1], item[0])
            )[:10]
        ]
        generation_hints = {
            "preferred_syscalls": preferred,
            "preferred_ratio": 0.3,
        }

        guidance = {
            "version": self.version,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "syscall_weights": top_weights,
            "mutation_templates": merged_templates,
            "generation_hints": generation_hints,
        }

        self._current_weights = top_weights
        self._current_templates = []
        for t in merged_templates:
            arg_hints = []
            if "arg_hints" in t and t["arg_hints"]:
                arg_hints = [ArgConstraint(**h) for h in t["arg_hints"]]
            self._current_templates.append(MutationTemplate(
                type=t["type"],
                syscalls=t["syscalls"],
                priority=t["priority"],
                insert_mode=t.get("insert_mode", "prefix"),
                arg_hints=arg_hints,
            ))

        logger.info(f"[GuidanceEngine] Guidance computed: v{self.version}, "
                    f"{len(top_weights)} syscall weights, {len(merged_templates)} templates")

        return guidance

    def send_guidance(
            self, callback_addr: str = "", guidance: dict = None,
            cancel_event=None) -> bool:
        """Send guidance to fuzzer via POST /guidance.

        Args:
            callback_addr: Fuzzer callback address (e.g., "localhost:21007")
            guidance: Pre-computed guidance dict (if None, calls compute_guidance())
            cancel_event: Optional threading.Event that interrupts retries.

        Returns:
            True if the fuzzer applies the guidance, False otherwise
        """
        addr = callback_addr or self.config.fuzzer_callback_addr

        if not addr:
            logger.error("[GuidanceEngine] No fuzzer callback address configured")
            return False

        if guidance is None:
            guidance = self.compute_guidance()
        self._last_ack = {}

        url = f"http://{addr}/guidance"
        last_error = None
        error_attempt = 0
        pending_deadline = (
            time.monotonic()
            + max(0.0, self.config.pending_ack_timeout_seconds)
        )
        while error_attempt < 5:
            if cancel_event is not None and cancel_event.is_set():
                logger.info(
                    f"[GuidanceEngine] Guidance v{guidance['version']} "
                    "send canceled before request"
                )
                return False
            remaining = pending_deadline - time.monotonic()
            if remaining <= 0:
                if last_error is None:
                    last_error = TimeoutError(
                        "guidance acknowledgement deadline expired"
                    )
                break
            try:
                resp = requests.post(
                    url,
                    json=guidance,
                    timeout=max(0.001, min(10.0, remaining)),
                )
                resp.raise_for_status()
                payload = resp.json()
                if int(payload.get("version", -1)) != int(guidance["version"]):
                    raise RuntimeError(
                        f"guidance acknowledgement version mismatch: {payload}"
                    )
                status = payload.get("status")
                if status == "pending":
                    last_error = RuntimeError(
                        f"guidance is not active yet: {payload}"
                    )
                    remaining = pending_deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    delay = min(
                        max(0.1, self.config.pending_poll_seconds), remaining
                    )
                    if cancel_event is None:
                        time.sleep(delay)
                    elif cancel_event.wait(delay):
                        logger.info(
                            f"[GuidanceEngine] Guidance v{guidance['version']} "
                            "send canceled while pending"
                        )
                        return False
                    continue
                if status not in ("applied", "already_applied"):
                    raise RuntimeError(
                        f"guidance is not active yet: {payload}"
                    )
                logger.info(
                    f"[GuidanceEngine] Guidance v{guidance['version']} "
                    f"acknowledged by {url}: {payload}"
                )
                self._last_ack = dict(payload)
                accepted_seeds = payload.get("accepted_seeds")
                if (status == "applied" and self._static_seed_hints and
                        accepted_seeds == 0):
                    logger.warning(
                        "[GuidanceEngine] Seed-only guidance was applied but "
                        "the fuzzer injected no cold-start seed: %s",
                        payload,
                    )
                return True
            except Exception as error:
                last_error = error
                error_attempt += 1
                if error_attempt < 5:
                    remaining = pending_deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    delay = min(2 ** (error_attempt - 1), 8, remaining)
                    if cancel_event is None:
                        time.sleep(delay)
                    elif cancel_event.wait(delay):
                        logger.info(
                            f"[GuidanceEngine] Guidance v{guidance['version']} "
                            "send canceled during retry"
                        )
                        return False
        logger.error(
            f"[GuidanceEngine] Guidance v{guidance['version']} was not "
            f"acknowledged within its retry window: {last_error}"
        )
        return False

    def get_current_weights(self) -> Dict[str, float]:
        """Get the current merged syscall weights."""
        return dict(self._current_weights)

    def get_current_templates(self) -> List[MutationTemplate]:
        """Get the current merged templates."""
        return list(self._current_templates)

    def get_static_weights(self) -> Dict[str, float]:
        """Get the static analysis syscall weights."""
        return dict(self._static_weights)

    def get_last_ack(self) -> Dict[str, object]:
        """Return the last active Fuzzer acknowledgement, including seed counts."""
        return dict(self._last_ack)

    def get_attribution_weights(self) -> Dict[str, float]:
        """Get the attribution-based syscall weights."""
        return dict(self._attribution_weights)
