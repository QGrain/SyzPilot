"""Validated index for compiled Syzlang call metadata.

Raw ``sys/linux/*.txt`` files may propose report-derived candidates, but this
module is the authority for whether a call exists in the compiled target and
whether it is generatable, seed-only, or disabled.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import threading
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Dict, Iterable, Mapping, Optional, Tuple


SCHEMA_NAME = "syzpilot.compiled-syzlang-manifest"
SCHEMA_VERSION = 2
DELIVERY_GENERATABLE = "generatable"
DELIVERY_SEED_ONLY = "seed_only"
DELIVERY_DISABLED = "disabled"
DELIVERY_MODES = frozenset({
    DELIVERY_GENERATABLE,
    DELIVERY_SEED_ONLY,
    DELIVERY_DISABLED,
})
DESCRIPTION_MODES = frozenset({"manual", "auto", "any"})
DEFAULT_MAX_MANIFEST_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class CallRecord:
    """The call metadata needed at the Brain/Fuzzer guidance boundary."""

    name: str
    call_name: str
    nr: int
    fixed_arguments: Tuple[Tuple[int, int], ...]
    delivery_mode: str
    automatic: bool
    automatic_helper: bool
    required_resources: Tuple[str, ...]
    declared_output_resources: Tuple[str, ...]

    def allowed_by_description_mode(self, mode: Optional[str]) -> bool:
        """Mirror syzkaller description-mode eligibility when mode is known."""
        if mode is None or mode == "any":
            return True
        if mode == "manual":
            return not self.automatic
        if mode == "auto":
            return self.automatic or self.automatic_helper
        raise ValueError(f"unsupported descriptions mode: {mode!r}")


@dataclass(frozen=True)
class CandidateFilterResult:
    """Compiled candidates and explicit rejection counts."""

    entries: Tuple[dict, ...]
    rejected: Mapping[str, int]


@dataclass(frozen=True)
class StaticTemplateResult:
    """Bounded report-derived templates and their setup-call guidance."""

    templates: Tuple[dict, ...]
    producer_entries: Tuple[dict, ...]


@dataclass(frozen=True)
class ConstantRefinementResult:
    """Candidates after a conservative report-constant refinement."""

    entries: Tuple[dict, ...]
    matched_name: Optional[str]
    status: str


class SyzlangIndex:
    """Immutable, content-identified index over one compiled target manifest."""

    _cache_lock = threading.Lock()
    _cache: "OrderedDict[tuple, SyzlangIndex]" = OrderedDict()
    _cache_capacity = 4

    def __init__(
            self, *, target_os: str, target_arch: str, target_revision: str,
            producer_revision: str, sha256: str,
            by_name: Mapping[str, CallRecord],
            by_call_name: Mapping[str, Tuple[CallRecord, ...]],
            resource_constructors: Mapping[str, Tuple[str, ...]],
            resource_consumer_counts: Mapping[str, int],
            constant_names_by_value: Mapping[int, Tuple[str, ...]]):
        self.target_os = target_os
        self.target_arch = target_arch
        self.target_revision = target_revision
        self.producer_revision = producer_revision
        self.sha256 = sha256
        self._by_name = MappingProxyType(dict(by_name))
        self._by_call_name = MappingProxyType(dict(by_call_name))
        self._resource_constructors = MappingProxyType(
            dict(resource_constructors)
        )
        self._resource_consumer_counts = MappingProxyType(
            dict(resource_consumer_counts)
        )
        self._constant_names_by_value = MappingProxyType(
            dict(constant_names_by_value)
        )

    @property
    def identity(self) -> Mapping[str, object]:
        """Return the immutable fields that identify compiled call semantics."""
        return MappingProxyType({
            "schema_version": SCHEMA_VERSION,
            "target_os": self.target_os,
            "target_arch": self.target_arch,
            "target_revision": self.target_revision,
            "producer_revision": self.producer_revision,
            "sha256": self.sha256,
        })

    def get(self, exact_name: str) -> Optional[CallRecord]:
        return self._by_name.get(exact_name)

    def variants(self, call_name: str) -> Tuple[CallRecord, ...]:
        return self._by_call_name.get(call_name, ())

    def refine_candidates_by_report_constant(
            self, entries: Iterable[dict], *, call_name: str,
            syscall_nr: int, fixed_arg_index: int, value: int,
            descriptions_mode: Optional[str] = None,
    ) -> ConstantRefinementResult:
        """Promote one path-supported variant selected by a named constant.

        Numeric values alone are commonly ambiguous across kernel subsystems.
        A refinement is therefore accepted only when a manifest constant name
        exactly matches a variant suffix already proposed from report paths,
        exactly one eligible variant remains, and no conflicting exact report
        evidence exists. Ambiguity leaves the original candidates unchanged.
        """
        materialized = tuple(entries)
        if not isinstance(call_name, str) or not call_name:
            raise ValueError("call_name must be a non-empty string")
        for field_name, number in (
                ("syscall_nr", syscall_nr),
                ("fixed_arg_index", fixed_arg_index),
                ("value", value)):
            if (isinstance(number, bool) or not isinstance(number, int) or
                    number < 0 or
                    (field_name != "fixed_arg_index" and
                     number > (1 << 64) - 1)):
                raise ValueError(f"report {field_name} is invalid")
        if (descriptions_mode is not None and
                descriptions_mode not in DESCRIPTION_MODES):
            raise ValueError(
                f"unsupported descriptions mode: {descriptions_mode!r}"
            )

        constant_names = set(self._constant_names_by_value.get(value, ()))
        if not constant_names:
            return ConstantRefinementResult(
                materialized, None, "unknown_constant"
            )
        candidate_names = {
            entry.get("name")
            for entry in materialized
            if isinstance(entry, dict) and
            entry.get("source") == "path_analysis"
        }
        matches = []
        for record in self.variants(call_name):
            if (record.name not in candidate_names or
                    record.nr != syscall_nr or
                    (fixed_arg_index, value) not in record.fixed_arguments or
                    record.delivery_mode != DELIVERY_GENERATABLE or
                    not record.allowed_by_description_mode(descriptions_mode)):
                continue
            base_name, separator, suffix = record.name.partition("$")
            if separator and base_name == call_name and suffix in constant_names:
                matches.append(record)
        if len(matches) != 1:
            status = "ambiguous" if matches else "unsupported_by_report_path"
            return ConstantRefinementResult(materialized, None, status)

        exact = matches[0]
        for entry in materialized:
            if (not isinstance(entry, dict) or
                    entry.get("source") != "path_analysis" or
                    entry.get("guidance_role") != "entry_exact" or
                    entry.get("name") == exact.name):
                continue
            record = self.get(entry.get("name"))
            if record is not None and record.call_name == call_name:
                return ConstantRefinementResult(
                    materialized, None, "conflicting_exact_evidence"
                )

        refined = []
        for entry in materialized:
            if not isinstance(entry, dict):
                refined.append(entry)
                continue
            name = entry.get("name")
            record = self.get(name) if isinstance(name, str) else None
            if name == exact.name:
                promoted = dict(entry)
                promoted.update({
                    "kernel_name": (
                        f"report_register:{call_name}:0x{value:x}"
                    ),
                    "weight": 0.99,
                    "guidance_level": "syz_call",
                    "guidance_role": "entry_exact",
                })
                refined.append(promoted)
                continue
            if (entry.get("source") == "path_analysis" and
                    record is not None and
                    record.call_name == call_name and
                    "$" in record.name and
                    entry.get("guidance_role") == "subsystem_peer"):
                continue
            refined.append(entry)
        return ConstantRefinementResult(
            tuple(refined), exact.name, "matched"
        )

    def filter_candidates(
            self, entries: Iterable[dict], *,
            descriptions_mode: Optional[str] = None,
            enabled_names: Optional[frozenset[str]] = None,
    ) -> CandidateFilterResult:
        """Join heuristic candidates to compiled calls and annotate delivery.

        Runtime enabled-call membership is intentionally reported as unknown
        unless the registering fuzzer supplied an exact enabled set.
        """
        if (descriptions_mode is not None and
                descriptions_mode not in DESCRIPTION_MODES):
            raise ValueError(
                f"unsupported descriptions mode: {descriptions_mode!r}"
            )
        accepted = []
        rejected = Counter()
        for source in entries:
            if not isinstance(source, dict):
                rejected["malformed"] += 1
                continue
            name = source.get("name")
            if not isinstance(name, str) or not name:
                rejected["malformed"] += 1
                continue
            record = self.get(name)
            if record is None:
                rejected["unknown"] += 1
                continue
            if record.delivery_mode == DELIVERY_DISABLED:
                rejected["disabled"] += 1
                continue
            if not record.allowed_by_description_mode(descriptions_mode):
                rejected["description_mode"] += 1
                continue
            if enabled_names is not None and name not in enabled_names:
                rejected["runtime_disabled"] += 1
                continue
            entry = dict(source)
            entry["delivery_mode"] = record.delivery_mode
            entry["runtime_eligibility"] = (
                "enabled" if enabled_names is not None else "unknown"
            )
            accepted.append(entry)
        return CandidateFilterResult(
            entries=tuple(accepted),
            rejected=MappingProxyType(dict(sorted(rejected.items()))),
        )

    def filter_generation_scores(
            self, scores: Mapping[str, float], *,
            descriptions_mode: Optional[str] = None,
    ) -> CandidateFilterResult:
        """Keep only positive scores for calls the current target can generate."""
        if not isinstance(scores, Mapping):
            raise ValueError("generation scores must be a mapping")
        if (descriptions_mode is not None and
                descriptions_mode not in DESCRIPTION_MODES):
            raise ValueError(
                f"unsupported descriptions mode: {descriptions_mode!r}"
            )
        accepted = []
        rejected = Counter()
        for name, score in scores.items():
            if (not isinstance(name, str) or not name or
                    isinstance(score, bool) or
                    not isinstance(score, (int, float)) or
                    not math.isfinite(float(score)) or float(score) <= 0):
                rejected["malformed"] += 1
                continue
            record = self.get(name)
            if record is None:
                rejected["unknown"] += 1
                continue
            if record.delivery_mode == DELIVERY_DISABLED:
                rejected["disabled"] += 1
                continue
            if record.delivery_mode == DELIVERY_SEED_ONLY:
                rejected["seed_only"] += 1
                continue
            if not record.allowed_by_description_mode(descriptions_mode):
                rejected["description_mode"] += 1
                continue
            accepted.append({
                "name": name,
                "weight": float(score),
                "delivery_mode": DELIVERY_GENERATABLE,
            })
        return CandidateFilterResult(
            entries=tuple(accepted),
            rejected=MappingProxyType(dict(sorted(rejected.items()))),
        )

    def build_report_static_templates(
            self, entries: Iterable[dict], *,
            descriptions_mode: Optional[str] = None,
            max_templates: int = 4, max_named_calls: int = 4,
            max_constructors: int = 3,
            max_state_variants: int = 2,
            max_resource_consumer_fanout: int = 32,
    ) -> StaticTemplateResult:
        """Build small resource-aware templates from exact report evidence.

        The compiled resource-dependency table is authoritative for setup-call
        selection. State-establishing bind/connect/listen calls are optional,
        low-confidence variants and must already be present in the report's
        filtered candidate set.
        """
        limits = (
            max_templates, max_named_calls, max_constructors,
            max_state_variants, max_resource_consumer_fanout,
        )
        if any(not isinstance(limit, int) or limit < 0 for limit in limits):
            raise ValueError("static template limits must be non-negative integers")
        if not max_templates or not max_named_calls:
            return StaticTemplateResult((), ())
        if (descriptions_mode is not None and
                descriptions_mode not in DESCRIPTION_MODES):
            raise ValueError(
                f"unsupported descriptions mode: {descriptions_mode!r}"
            )

        report_entries = []
        report_names = set()
        for entry in entries:
            if (not isinstance(entry, dict) or
                    entry.get("source") != "path_analysis"):
                continue
            name = entry.get("name")
            record = self.get(name) if isinstance(name, str) else None
            if (record is None or
                    record.delivery_mode != DELIVERY_GENERATABLE or
                    not record.allowed_by_description_mode(descriptions_mode)):
                continue
            report_entries.append((entry, record))
            report_names.add(record.name)

        def entry_weight(entry: dict) -> float:
            value = entry.get("weight", 0.0)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return 0.0
            value = float(value)
            return value if value > 0 else 0.0

        exact_entries = sorted(
            (
                (entry, record) for entry, record in report_entries
                if entry.get("guidance_role") == "entry_exact"
            ),
            key=lambda item: (-entry_weight(item[0]), item[1].name),
        )
        if not exact_entries:
            return StaticTemplateResult((), ())

        templates = []
        seen_templates = set()
        producer_weights: Dict[str, float] = {}

        def eligible_constructor(name: str) -> Optional[CallRecord]:
            record = self.get(name)
            if (record is None or
                    record.delivery_mode != DELIVERY_GENERATABLE or
                    not record.allowed_by_description_mode(descriptions_mode)):
                return None
            return record

        def choose_constructors(exact: CallRecord) -> Tuple[CallRecord, ...]:
            selected: Dict[str, CallRecord] = {}
            for resource in exact.required_resources:
                candidates = []
                for name in self._resource_constructors.get(resource, ()):
                    record = eligible_constructor(name)
                    if record is not None and record.name != exact.name:
                        candidates.append(record)
                if not candidates:
                    continue
                chosen = min(
                    candidates,
                    key=lambda record: (
                        record.name not in report_names,
                        len(record.required_resources),
                        record.name,
                    ),
                )
                selected.setdefault(chosen.name, chosen)
            selected_records = sorted(
                selected.values(),
                key=lambda record: (
                    len(record.required_resources), record.name
                ),
            )[:max_constructors]
            return self._topological_constructor_order(selected_records)

        def add_template(syscalls: Iterable[str], priority: float) -> None:
            ordered = tuple(dict.fromkeys(syscalls))
            if (not ordered or len(ordered) > max_named_calls or
                    ordered in seen_templates or
                    len(templates) >= max_templates):
                return
            seen_templates.add(ordered)
            templates.append({
                "type": "sequence",
                "syscalls": list(ordered),
                "priority": priority,
                "insert_mode": "prefix",
            })

        state_call_names = {"bind", "connect", "listen"}
        for exact_entry, exact in exact_entries:
            if len(templates) >= max_templates:
                break
            constructors = choose_constructors(exact)
            constructor_names = tuple(record.name for record in constructors)
            exact_weight = max(entry_weight(exact_entry), 0.01)
            for constructor in constructors:
                producer_weights[constructor.name] = max(
                    producer_weights.get(constructor.name, 0.0),
                    exact_weight,
                )
            add_template((*constructor_names, exact.name), 1.0)

            exact_resources = set(exact.required_resources)
            peers = []
            for peer_entry, peer in report_entries:
                if (peer.name == exact.name or
                        peer.call_name not in state_call_names):
                    continue
                shared_resources = exact_resources.intersection(
                    peer.required_resources
                )
                if not any(
                        self._resource_consumer_counts.get(resource, 0) <=
                        max_resource_consumer_fanout
                        for resource in shared_resources):
                    continue
                peers.append((peer_entry, peer))
            peers.sort(key=lambda item: (
                {"bind": 0, "connect": 1, "listen": 2}[item[1].call_name],
                -entry_weight(item[0]),
                item[1].name,
            ))
            for peer_index, (_, peer) in enumerate(
                    peers[:max_state_variants]):
                add_template(
                    (*constructor_names, peer.name, exact.name),
                    0.8 - 0.1 * peer_index,
                )

        producer_entries = tuple({
            "name": name,
            "kernel_name": "compiled_resource_dependency",
            "path_length": 1,
            "weight": weight,
            "source": "path_analysis_resource_dependency",
            "guidance_level": (
                "syz_call" if "$" in name or name.startswith("syz_")
                else "system_call"
            ),
            "guidance_role": "resource_producer",
            "delivery_mode": DELIVERY_GENERATABLE,
            "runtime_eligibility": "unknown",
        } for name, weight in sorted(producer_weights.items()))
        return StaticTemplateResult(tuple(templates), producer_entries)

    def _topological_constructor_order(
            self, records: Iterable[CallRecord]) -> Tuple[CallRecord, ...]:
        """Order named constructors using compiled dependency edges."""
        by_name = {record.name: record for record in records}
        dependencies = {name: set() for name in by_name}
        for consumer in by_name.values():
            for resource in consumer.required_resources:
                for producer in self._resource_constructors.get(resource, ()):
                    if producer in by_name and producer != consumer.name:
                        dependencies[consumer.name].add(producer)

        ordered = []
        remaining = set(by_name)
        while remaining:
            ready = sorted(
                (name for name in remaining if not dependencies[name] & remaining),
                key=lambda name: (
                    len(by_name[name].required_resources), name
                ),
            )
            if not ready:
                ready = [min(remaining)]
            for name in ready:
                ordered.append(by_name[name])
                remaining.remove(name)
        return tuple(ordered)

    @classmethod
    def load(
            cls, path: str | Path, *, expected_os: str,
            expected_arch: str, expected_revision: Optional[str] = None,
            expected_producer_revision: Optional[str] = None,
            max_bytes: int = DEFAULT_MAX_MANIFEST_BYTES,
    ) -> "SyzlangIndex":
        """Load and validate a manifest from one stable regular-file snapshot."""
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        manifest_path = Path(path).expanduser()

        descriptor = cls._open(manifest_path)
        try:
            return cls._load_descriptor(
                descriptor,
                expected_os=expected_os,
                expected_arch=expected_arch,
                expected_revision=expected_revision,
                expected_producer_revision=expected_producer_revision,
                max_bytes=max_bytes,
            )
        finally:
            os.close(descriptor)

    @staticmethod
    def _open(path: Path) -> int:
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        return os.open(path, flags)

    @classmethod
    def _load_descriptor(
            cls, descriptor: int, *, expected_os: str,
            expected_arch: str, expected_revision: Optional[str],
            expected_producer_revision: Optional[str], max_bytes: int,
    ) -> "SyzlangIndex":
        try:
            before = os.fstat(descriptor)
            cls._validate_metadata(before, max_bytes)
            chunks = []
            remaining = max_bytes + 1
            while remaining > 0:
                chunk = os.read(descriptor, min(1024 * 1024, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            data = b"".join(chunks)
            if len(data) > max_bytes:
                raise ValueError("Syzlang manifest exceeds the configured limit")
            after = os.fstat(descriptor)
            if cls._file_identity(before) != cls._file_identity(after):
                raise ValueError("Syzlang manifest changed while it was read")
        except OSError as error:
            raise ValueError(f"failed to read Syzlang manifest: {error}") from error

        try:
            document = json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid Syzlang manifest JSON: {error}") from error
        return cls._from_document(
            document,
            expected_os=expected_os,
            expected_arch=expected_arch,
            expected_revision=expected_revision,
            expected_producer_revision=expected_producer_revision,
            sha256=hashlib.sha256(data).hexdigest(),
        )

    @classmethod
    def load_cached(cls, path: str | Path, **kwargs) -> "SyzlangIndex":
        """Reuse an immutable index while the manifest file identity is stable."""
        manifest_path = Path(path).expanduser()
        max_bytes = kwargs.get("max_bytes", DEFAULT_MAX_MANIFEST_BYTES)
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        descriptor = cls._open(manifest_path)
        try:
            metadata = os.fstat(descriptor)
            cls._validate_metadata(metadata, max_bytes)
            key = (
                str(manifest_path.absolute()),
                *cls._file_identity(metadata),
                kwargs.get("expected_os"),
                kwargs.get("expected_arch"),
                kwargs.get("expected_revision"),
                kwargs.get("expected_producer_revision"),
                max_bytes,
            )
            with cls._cache_lock:
                cached = cls._cache.get(key)
                if cached is not None:
                    cls._cache.move_to_end(key)
                    return cached
            loaded = cls._load_descriptor(
                descriptor,
                expected_os=kwargs["expected_os"],
                expected_arch=kwargs["expected_arch"],
                expected_revision=kwargs.get("expected_revision"),
                expected_producer_revision=kwargs.get(
                    "expected_producer_revision"
                ),
                max_bytes=max_bytes,
            )
            with cls._cache_lock:
                cls._cache[key] = loaded
                cls._cache.move_to_end(key)
                while len(cls._cache) > cls._cache_capacity:
                    cls._cache.popitem(last=False)
            return loaded
        finally:
            os.close(descriptor)

    @staticmethod
    def _validate_metadata(metadata: os.stat_result, max_bytes: int) -> None:
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("Syzlang manifest must be a regular file")
        if metadata.st_size <= 0 or metadata.st_size > max_bytes:
            raise ValueError(
                "Syzlang manifest size is outside the configured limit: "
                f"{metadata.st_size} bytes"
            )

    @staticmethod
    def _file_identity(metadata: os.stat_result) -> tuple:
        return (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        )

    @classmethod
    def _from_document(
            cls, document: object, *, expected_os: str,
            expected_arch: str, expected_revision: Optional[str],
            expected_producer_revision: Optional[str],
            sha256: str,
    ) -> "SyzlangIndex":
        if not isinstance(document, dict):
            raise ValueError("Syzlang manifest root must be an object")
        if document.get("schema") != SCHEMA_NAME:
            raise ValueError("unsupported Syzlang manifest schema")
        if document.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported Syzlang manifest schema version")
        target = document.get("target")
        producer = document.get("producer")
        calls = document.get("calls")
        constants = document.get("constants")
        resource_dependencies = document.get("resource_dependencies")
        if not isinstance(target, dict) or not isinstance(producer, dict):
            raise ValueError("manifest target and producer must be objects")
        target_os = target.get("os")
        target_arch = target.get("arch")
        target_revision = target.get("revision")
        producer_revision = producer.get("syzkaller_git_revision")
        for field_name, value in (
                ("target.os", target_os),
                ("target.arch", target_arch),
                ("target.revision", target_revision),
                ("producer.syzkaller_git_revision", producer_revision)):
            if not isinstance(value, str) or not value:
                raise ValueError(f"manifest {field_name} must be a non-empty string")
        if target_os != expected_os or target_arch != expected_arch:
            raise ValueError(
                "manifest target mismatch: "
                f"got {target_os}/{target_arch}, "
                f"want {expected_os}/{expected_arch}"
            )
        if expected_revision is not None and target_revision != expected_revision:
            raise ValueError(
                "manifest target revision mismatch: "
                f"got {target_revision}, want {expected_revision}"
            )
        if (expected_producer_revision is not None and
                producer_revision != expected_producer_revision):
            raise ValueError(
                "manifest producer revision mismatch: "
                f"got {producer_revision}, want {expected_producer_revision}"
            )
        if not isinstance(calls, list) or not calls:
            raise ValueError("manifest calls must be a non-empty array")
        if not isinstance(constants, list):
            raise ValueError("manifest constants must be an array")
        if not isinstance(resource_dependencies, list):
            raise ValueError("manifest resource_dependencies must be an array")

        constant_names_by_value: Dict[int, list[str]] = defaultdict(list)
        seen_constant_names = set()
        previous_constant_name = ""
        for item in constants:
            if not isinstance(item, dict):
                raise ValueError("manifest constants must be objects")
            name = item.get("name")
            encoded = item.get("value")
            if (not isinstance(name, str) or not name or
                    name in seen_constant_names or
                    name < previous_constant_name or
                    not isinstance(encoded, str)):
                raise ValueError("manifest constants are invalid or unsorted")
            try:
                value = int(encoded, 16)
            except ValueError as error:
                raise ValueError(
                    f"manifest constant {name} has an invalid value"
                ) from error
            if (value < 0 or value > (1 << 64) - 1 or
                    encoded != f"0x{value:x}"):
                raise ValueError(
                    f"manifest constant {name} has a non-canonical value"
                )
            seen_constant_names.add(name)
            previous_constant_name = name
            constant_names_by_value[value].append(name)

        by_name: Dict[str, CallRecord] = {}
        grouped: Dict[str, list[CallRecord]] = defaultdict(list)
        previous_name = ""
        for item in calls:
            if not isinstance(item, dict):
                raise ValueError("manifest call entries must be objects")
            name = item.get("name")
            call_name = item.get("call_name")
            encoded_nr = item.get("nr")
            delivery_mode = item.get("delivery_mode")
            attrs = item.get("attrs")
            if (not isinstance(name, str) or not name or
                    not isinstance(call_name, str) or not call_name):
                raise ValueError("manifest call names must be non-empty strings")
            if not isinstance(encoded_nr, str):
                raise ValueError(f"call {name} has an invalid syscall number")
            try:
                nr = int(encoded_nr, 16)
            except ValueError as error:
                raise ValueError(
                    f"call {name} has an invalid syscall number"
                ) from error
            if (nr < 0 or nr > (1 << 64) - 1 or
                    encoded_nr != f"0x{nr:x}"):
                raise ValueError(
                    f"call {name} has a non-canonical syscall number"
                )
            if name in by_name:
                raise ValueError(f"duplicate manifest call: {name}")
            if name < previous_name:
                raise ValueError("manifest calls are not sorted by name")
            previous_name = name
            if delivery_mode not in DELIVERY_MODES:
                raise ValueError(
                    f"call {name} has invalid delivery mode {delivery_mode!r}"
                )
            if not isinstance(attrs, dict):
                raise ValueError(f"call {name} attrs must be an object")
            disabled = attrs.get("disabled")
            no_generate = attrs.get("no_generate")
            automatic = attrs.get("automatic")
            automatic_helper = attrs.get("automatic_helper")
            required_resources = item.get("required_resources")
            declared_output_resources = item.get("declared_output_resources")
            fixed_arguments = item.get("fixed_arguments")
            if not all(isinstance(value, bool) for value in (
                    disabled, no_generate, automatic, automatic_helper)):
                raise ValueError(
                    f"call {name} has invalid delivery-related attributes"
                )
            for field_name, resources in (
                    ("required_resources", required_resources),
                    ("declared_output_resources", declared_output_resources)):
                if (not isinstance(resources, list) or
                        any(not isinstance(resource, str) or not resource
                            for resource in resources) or
                        len(resources) != len(set(resources))):
                    raise ValueError(
                        f"call {name} has invalid {field_name}"
                    )
            if not isinstance(fixed_arguments, list):
                raise ValueError(f"call {name} has invalid fixed_arguments")
            parsed_fixed_arguments = []
            previous_fixed_index = -1
            for argument in fixed_arguments:
                if not isinstance(argument, dict):
                    raise ValueError(
                        f"call {name} has invalid fixed_arguments"
                    )
                index = argument.get("index")
                encoded_value = argument.get("value")
                if (isinstance(index, bool) or not isinstance(index, int) or
                        index <= previous_fixed_index or
                        not isinstance(encoded_value, str)):
                    raise ValueError(
                        f"call {name} has invalid fixed_arguments"
                    )
                try:
                    fixed_value = int(encoded_value, 16)
                except ValueError as error:
                    raise ValueError(
                        f"call {name} has invalid fixed_arguments"
                    ) from error
                if (fixed_value < 0 or fixed_value > (1 << 64) - 1 or
                        encoded_value != f"0x{fixed_value:x}"):
                    raise ValueError(
                        f"call {name} has invalid fixed_arguments"
                    )
                parsed_fixed_arguments.append((index, fixed_value))
                previous_fixed_index = index
            expected_delivery = (
                DELIVERY_DISABLED if disabled else
                DELIVERY_SEED_ONLY if no_generate else
                DELIVERY_GENERATABLE
            )
            if delivery_mode != expected_delivery:
                raise ValueError(
                    f"call {name} delivery mode contradicts its attributes"
                )
            record = CallRecord(
                name=name,
                call_name=call_name,
                nr=nr,
                fixed_arguments=tuple(parsed_fixed_arguments),
                delivery_mode=delivery_mode,
                automatic=automatic,
                automatic_helper=automatic_helper,
                required_resources=tuple(required_resources),
                declared_output_resources=tuple(declared_output_resources),
            )
            by_name[name] = record
            grouped[call_name].append(record)
        by_call_name = {
            name: tuple(sorted(records, key=lambda record: record.name))
            for name, records in grouped.items()
        }
        constructor_map: Dict[str, Tuple[str, ...]] = {}
        for dependency in resource_dependencies:
            if not isinstance(dependency, dict):
                raise ValueError("resource dependency entries must be objects")
            resource = dependency.get("name")
            constructors = dependency.get("precise_constructors")
            if (not isinstance(resource, str) or not resource or
                    resource in constructor_map or
                    not isinstance(constructors, list) or
                    any(not isinstance(name, str) or not name
                        for name in constructors) or
                    len(constructors) != len(set(constructors))):
                raise ValueError("manifest resource dependency is invalid")
            unknown = [name for name in constructors if name not in by_name]
            if unknown:
                raise ValueError(
                    f"resource {resource} references unknown constructors"
                )
            constructor_map[resource] = tuple(constructors)
        consumer_counts = Counter(
            resource
            for record in by_name.values()
            for resource in record.required_resources
        )
        return cls(
            target_os=target_os,
            target_arch=target_arch,
            target_revision=target_revision,
            producer_revision=producer_revision,
            sha256=sha256,
            by_name=by_name,
            by_call_name=by_call_name,
            resource_constructors=constructor_map,
            resource_consumer_counts=consumer_counts,
            constant_names_by_value={
                value: tuple(names)
                for value, names in constant_names_by_value.items()
            },
        )
