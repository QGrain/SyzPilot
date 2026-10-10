#!/usr/bin/env python
# coding: utf-8
"""
Attribution-guided syscall scoring for SyzPilot.

Provides run_attribution_for_guidance() which analyzes positive samples
to identify which syscalls contribute most to the target class prediction.
Used by the Brain controller to generate guidance for the fuzzer.
"""

import os
import sys
import hashlib
import logging
import math
from dataclasses import dataclass, field
from typing import Mapping, Optional, Tuple
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer
from captum.attr import LayerIntegratedGradients

logger = logging.getLogger(__name__)

# Add filter dir to path for local imports
filter_dir = os.path.dirname(os.path.abspath(__file__))
if filter_dir not in sys.path:
    sys.path.insert(0, filter_dir)
repo_root = os.path.dirname(filter_dir)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

try:
    from .model_v2 import TraceClassifierV2
    from .model_v2 import TraceClassifierServingWrapper
    from .dataset_v2 import load_canonical_records
except ImportError:
    # Controller loads this module with filter/ on sys.path when it runs as a
    # script, while tests and package users import it as filter.*.
    from model_v2 import TraceClassifierV2
    from model_v2 import TraceClassifierServingWrapper
    from dataset_v2 import load_canonical_records
from common.curriculum import (
    is_exact_curriculum_objective,
    normalize_stage3_active_classes,
    stage2_deep_class_start,
)


_CANDIDATE_SCAN_FACTOR = 8


def replace_tokens(tokens):
    """Normalize tokenizer whitespace and newline markers in place."""
    for index, token in enumerate(tokens):
        tokens[index] = token.replace(
            "Ġ", " "
        ).replace("Ċ", "\n").replace("<|endoftext|>", "")


def split_invocations(tokens, token_attributions):
    """Group token attribution scores by syz-program invocation."""
    if len(tokens) != len(token_attributions):
        raise ValueError("tokens and attributions must have equal length")
    grouped_tokens = []
    grouped_attributions = []
    current_tokens = []
    current_attributions = []
    for token, attribution in zip(tokens, token_attributions):
        current_tokens.append(token)
        current_attributions.append(attribution)
        if "\n" in token:
            grouped_tokens.append(current_tokens)
            grouped_attributions.append(current_attributions)
            current_tokens = []
            current_attributions = []
    if current_tokens:
        grouped_tokens.append(current_tokens)
        grouped_attributions.append(current_attributions)
    return grouped_tokens, grouped_attributions


def get_syscall_attr(tokenizer, invocations_tokens, invocations_attrs):
    """Aggregate token attribution scores over syscall-name subtokens."""
    if len(invocations_tokens) != len(invocations_attrs):
        raise ValueError("invocations and attributions must have equal length")
    syscall_attr = {}
    for invocation_tokens, invocation_attrs in zip(
            invocations_tokens, invocations_attrs):
        invocation = "".join(invocation_tokens)
        if "(" not in invocation:
            continue
        syscall = invocation.split("(", 1)[0].strip().split(" ")[-1].strip()
        if not syscall:
            continue
        syscall_attr.setdefault(syscall, 0)
        syscall_tokens = tokenizer.tokenize(syscall)
        width = len(syscall_tokens)
        for offset in range(len(invocation_tokens) - width + 1):
            if all(
                    syscall_tokens[index] ==
                    invocation_tokens[offset + index].strip()
                    for index in range(width)):
                syscall_attr[syscall] += sum(
                    invocation_attrs[offset:offset + width]
                )
                break
    return syscall_attr


def merge_syscall_attr(total_syscall_attr, new_syscall_attr):
    """Merge one sample's syscall attribution into an aggregate mapping."""
    for syscall, attribution in new_syscall_attr.items():
        total_syscall_attr[syscall] = (
            total_syscall_attr.get(syscall, 0) + attribution
        )


@dataclass(frozen=True)
class AttributionCandidate:
    """One canonical program eligible for a curriculum objective."""

    signature: str
    program: str
    canonical_class: int
    depth_factor: float


@dataclass(frozen=True)
class AttributionCohort:
    """A deterministic candidate order for one attribution objective."""

    stage: int
    objective: str
    target_output: int
    canonical_members: Tuple[int, ...]
    active_exact_classes: Tuple[int, ...]
    max_samples: int
    depth_weight: float
    candidates: Tuple[AttributionCandidate, ...]


@dataclass
class _AttributionCohortState:
    """Mutable progress for fair round-robin attribution analysis."""

    cohort: AttributionCohort
    total_syscall_attr: dict = field(default_factory=dict)
    selected_signatures: list = field(default_factory=list)
    next_candidate: int = 0
    examined: int = 0

    def has_candidate(self):
        candidate_budget = self.cohort.max_samples * _CANDIDATE_SCAN_FACTOR
        return (
            len(self.selected_signatures) < self.cohort.max_samples and
            self.next_candidate < len(self.cohort.candidates) and
            self.next_candidate < candidate_budget
        )


def _cohort_rank(deployment_version, signature):
    payload = f"{deployment_version}\0{signature}".encode("utf-8")
    return hashlib.blake2b(payload, digest_size=8).digest(), signature


def _candidate_depth_factor(class_index, num_classes):
    return 0.5 + 0.5 * class_index / (num_classes - 1)


def select_attribution_cohorts(
        records: Mapping[str, tuple], num_classes: int, stage: int,
        deployment_version: int, active_classes=None):
    """Build the fixed stage-aware attribution cohorts from canonical data."""
    if num_classes < 2 or stage not in (1, 2, 3):
        raise ValueError("invalid curriculum schema")
    active = (
        normalize_stage3_active_classes(active_classes, num_classes)
        if stage == 3 else ()
    )

    if stage == 1:
        specifications = [
            ("reachable", 1, tuple(range(1, num_classes)), 48, 1.0)
        ]
    elif stage == 2:
        deep_start = stage2_deep_class_start(num_classes)
        specifications = [
            ("shallow", 1, tuple(range(1, deep_start)), 24, 0.60),
            ("deep", 2, tuple(range(deep_start, num_classes)), 24, 1.00),
        ]
    else:
        reached = [class_index for class_index in active if class_index > 0]
        if len(reached) > 16:
            final_class = num_classes - 1
            rotating = [
                class_index for class_index in reached
                if class_index != final_class
            ]
            offset = int(deployment_version) % len(rotating)
            rotating = rotating[offset:] + rotating[:offset]
            reached = sorted(rotating[:15] + [final_class])
        specifications = [
            (
                f"exact:{class_index}", active.index(class_index),
                (class_index,), 8,
                _candidate_depth_factor(class_index, num_classes),
            )
            for class_index in reached
        ]

    cohorts = []
    for objective, target_output, members, cap, depth_weight in specifications:
        candidates = []
        member_set = frozenset(members)
        for signature, record in records.items():
            program, _, canonical_class = record
            if canonical_class not in member_set:
                continue
            sample_depth = (
                _candidate_depth_factor(canonical_class, num_classes)
                if stage == 1 else 1.0
            )
            candidates.append(AttributionCandidate(
                signature=str(signature),
                program=program,
                canonical_class=canonical_class,
                depth_factor=sample_depth,
            ))
        candidates.sort(
            key=lambda item: _cohort_rank(
                deployment_version, item.signature
            )
        )
        cohorts.append(AttributionCohort(
            stage=stage,
            objective=objective,
            target_output=target_output,
            canonical_members=members,
            active_exact_classes=active,
            max_samples=cap,
            depth_weight=depth_weight,
            candidates=tuple(candidates),
        ))
    return cohorts


def _normalize_positive_scores(scores):
    positive = {
        name: float(score)
        for name, score in scores.items()
        if math.isfinite(float(score)) and score > 0
    }
    norm = math.sqrt(sum(score * score for score in positive.values()))
    return {
        name: score / norm for name, score in positive.items()
    } if norm > 0 else {}


class AttributionProbabilityWrapper(nn.Module):
    """Expose curriculum probabilities as the Integrated Gradients target."""

    def __init__(self, model, stage, active_classes=None, cancel_event=None):
        super().__init__()
        self.serving_model = TraceClassifierServingWrapper(
            model, stage=stage, active_classes=active_classes
        )
        self.cancel_event = cancel_event

    def forward(self, input_ids, attention_mask=None):
        _raise_if_attribution_canceled(self.cancel_event)
        return F.softmax(
            self.serving_model(input_ids, attention_mask), dim=-1
        )


class AttributionCanceled(RuntimeError):
    """Raised when the owning fuzzing task stops during IG analysis."""


def _raise_if_attribution_canceled(cancel_event):
    """Interrupt attribution promptly when its task no longer owns the work."""
    if cancel_event is not None and cancel_event.is_set():
        raise AttributionCanceled("attribution canceled by task lifecycle")


def run_attribution_for_guidance(
    model_path,
    base_model_path,
    tokenizer_path,
    data_dir,
    data_indices,
    num_classes,
    stage,
    active_classes=None,
    target_class=None,
    top_k=10,
    max_samples=50,
    max_length=1024,
    device="cuda:0",
    internal_batch_size=5,
    cancel_event=None,
):
    """Run attribution analysis and return syscall-level scores for guidance.

    Args:
        model_path: Path to the trained model checkpoint (.pt file)
        base_model_path: Path to the pretrained encoder used by TrainerV2
        tokenizer_path: Path to the tokenizer
        data_dir: Directory containing progs_batch_*.pkl and labels_batch_*.pkl
        data_indices: List of batch indices to use for attribution
        num_classes: Number of classes in the model
        stage: Exact-waypoint Stage 3, or Stage 1 for a single waypoint
        active_classes: Canonical exact labels enabled by the Stage-3 model
        target_class: Optional reached class to analyze. None analyzes all
            reached classes.
        top_k: Number of top syscalls to return. None returns every positive
            score so a caller can apply a compiled-call trust boundary first.
        max_samples: Maximum number of samples to analyze
        max_length: Token sequence length. This must match training and serving.
        device: Device to use for computation
        internal_batch_size: Number of IG interpolation examples evaluated per
            forward pass. This bounds peak memory without changing n_steps.
        cancel_event: Optional threading-compatible event owned by the task.

    Returns:
        Dict of {syscall_name: normalized_attribution_score}
    """
    if not is_exact_curriculum_objective(stage, num_classes):
        raise ValueError("attribution requires an exact curriculum objective")
    active_classes = (
        normalize_stage3_active_classes(active_classes, num_classes)
        if stage == 3 else None
    )
    if target_class is not None and not 0 < target_class < num_classes:
        raise ValueError(
            f"target_class must be a reached class in [1, {num_classes - 1}]"
        )
    if (stage == 3 and target_class is not None and
            target_class not in active_classes):
        raise ValueError("target_class is inactive in the Stage-3 model")
    if internal_batch_size <= 0:
        raise ValueError("internal_batch_size must be positive")
    if top_k is not None and (
            isinstance(top_k, bool) or not isinstance(top_k, int) or
            top_k <= 0):
        raise ValueError("top_k must be a positive integer or None")
    _raise_if_attribution_canceled(cancel_event)

    # Load data
    programs = []
    true_classes = []
    records = load_canonical_records(data_dir, num_classes, data_indices)
    _raise_if_attribution_canceled(cancel_event)
    for program, label_bools, class_idx in records.values():
        _raise_if_attribution_canceled(cancel_event)
        if (class_idx == 0 or
                (stage == 3 and class_idx not in active_classes)):
            continue
        if target_class is not None and class_idx != target_class:
            continue
        programs.append(program)
        true_classes.append(class_idx)

    if not programs:
        logger.warning("[Attribution] No data found for attribution")
        return {}

    logger.info(
        "[Attribution] Found %d valid positive candidates%s",
        len(programs),
        f" for class {target_class}" if target_class is not None else "",
    )

    # Load tokenizer
    _raise_if_attribution_canceled(cancel_event)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token

    # TrainerV2 checkpoints contain the full TraceClassifierV2 state dict, but
    # their parent directory is not a Hugging Face model directory. Recreate
    # the training architecture from the configured base encoder first.
    model = TraceClassifierV2(
        base_model_path,
        num_classes,
        stage=stage,
        active_classes=active_classes,
    )
    _raise_if_attribution_canceled(cancel_event)
    model.load_state_dict(
        torch.load(model_path, map_location="cpu", weights_only=True)
    )
    _raise_if_attribution_canceled(cancel_event)
    model.to(device)
    model.eval()
    attribution_model = AttributionProbabilityWrapper(
        model,
        stage=stage,
        active_classes=active_classes,
        cancel_event=cancel_event,
    )
    attribution_model.to(device)
    attribution_model.eval()

    lig = LayerIntegratedGradients(
        attribution_model, model.base_model.embeddings
    )

    total_syscall_attr = {}
    analyzed = 0

    for prog_text, true_class in zip(programs, true_classes):
        _raise_if_attribution_canceled(cancel_event)
        if analyzed >= max_samples:
            break
        encoding = tokenizer(
            prog_text, max_length=max_length, padding='max_length',
            truncation=True, return_tensors='pt'
        )
        input_ids = encoding['input_ids'].long().to(device)
        attention_mask = encoding['attention_mask'].long().to(device)

        with torch.no_grad():
            logits = attribution_model(input_ids, attention_mask)
            pred_class = torch.argmax(logits, dim=1).item()
        _raise_if_attribution_canceled(cancel_event)

        target_output = (
            active_classes.index(true_class) if stage == 3
            else true_class
        )

        # Attribution guidance must only use correctly predicted, reached
        # samples. A positive sample misclassified as another reached class is
        # not evidence for either class's guidance.
        if pred_class == 0 or pred_class != target_output:
            continue

        attributions, delta = lig.attribute(
            inputs=input_ids,
            baselines=torch.zeros_like(input_ids),
            additional_forward_args=(attention_mask,),
            target=target_output,
            n_steps=50,
            internal_batch_size=internal_batch_size,
            return_convergence_delta=True
        )
        _raise_if_attribution_canceled(cancel_event)

        token_attributions = attributions.sum(dim=-1).squeeze(0).tolist()
        tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).tolist())
        replace_tokens(tokens)

        invocation_tokens, invocation_attrs = split_invocations(tokens, token_attributions)
        syscall_attr = get_syscall_attr(tokenizer, invocation_tokens, invocation_attrs)
        merge_syscall_attr(total_syscall_attr, syscall_attr)
        analyzed += 1

    _raise_if_attribution_canceled(cancel_event)
    if not total_syscall_attr:
        logger.warning("[Attribution] No syscall attributions computed")
        return {}

    # Mutation guidance consumes positive evidence. Apply the paper's L2
    # normalization after discarding non-positive contributions.
    positive_scores = {
        name: float(score)
        for name, score in total_syscall_attr.items()
        if math.isfinite(float(score)) and score > 0
    }
    norm = math.sqrt(sum(score * score for score in positive_scores.values()))
    normalized = {
        name: score / norm for name, score in positive_scores.items()
    } if norm > 0 else {}

    sorted_attrs = sorted(normalized.items(), key=lambda x: x[1], reverse=True)
    result = dict(sorted_attrs if top_k is None else sorted_attrs[:top_k])

    logger.info(f"[Attribution] Computed attribution for {analyzed} samples, "
                f"returned {len(result)} syscall scores")

    return result


def run_hierarchical_attribution_for_guidance(
    model_path,
    base_model_path,
    tokenizer_path,
    data_dir,
    data_indices,
    num_classes,
    stage,
    deployment_version,
    active_classes=None,
    max_length=1024,
    device="cuda:0",
    internal_batch_size=5,
    cancel_event=None,
):
    """Compute one bounded IG snapshot for each supported stage objective."""
    if stage not in (1, 2, 3):
        raise ValueError("invalid curriculum stage")
    if internal_batch_size <= 0:
        raise ValueError("internal_batch_size must be positive")
    active = (
        normalize_stage3_active_classes(active_classes, num_classes)
        if stage == 3 else None
    )
    _raise_if_attribution_canceled(cancel_event)

    records = load_canonical_records(data_dir, num_classes, data_indices)
    _raise_if_attribution_canceled(cancel_event)
    cohorts = select_attribution_cohorts(
        records, num_classes, stage, deployment_version, active
    )
    if not any(cohort.candidates for cohort in cohorts):
        logger.warning("[Attribution] No reached candidates for Stage %d", stage)
        return []

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = TraceClassifierV2(
        base_model_path,
        num_classes,
        stage=stage,
        active_classes=active,
    )
    model.load_state_dict(
        torch.load(model_path, map_location="cpu", weights_only=True)
    )
    _raise_if_attribution_canceled(cancel_event)
    model.to(device)
    model.eval()
    attribution_model = AttributionProbabilityWrapper(
        model,
        stage=stage,
        active_classes=active,
        cancel_event=cancel_event,
    )
    attribution_model.to(device)
    attribution_model.eval()
    lig = LayerIntegratedGradients(
        attribution_model, model.base_model.embeddings
    )

    states = [_AttributionCohortState(cohort) for cohort in cohorts]
    while True:
        progressed = False
        for state in states:
            if not state.has_candidate():
                continue
            progressed = True
            _raise_if_attribution_canceled(cancel_event)
            cohort = state.cohort
            candidate = cohort.candidates[state.next_candidate]
            state.next_candidate += 1
            state.examined += 1
            encoding = tokenizer(
                candidate.program,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = encoding["input_ids"].long().to(device)
            attention_mask = encoding["attention_mask"].long().to(device)

            with torch.no_grad():
                probabilities = attribution_model(input_ids, attention_mask)
                predicted = torch.argmax(probabilities, dim=1).item()
            _raise_if_attribution_canceled(cancel_event)
            if predicted != cohort.target_output:
                continue

            attributions, _ = lig.attribute(
                inputs=input_ids,
                baselines=torch.zeros_like(input_ids),
                additional_forward_args=(attention_mask,),
                target=cohort.target_output,
                n_steps=50,
                internal_batch_size=internal_batch_size,
                return_convergence_delta=True,
            )
            _raise_if_attribution_canceled(cancel_event)
            token_attributions = (
                attributions.sum(dim=-1).squeeze(0).tolist()
            )
            tokens = tokenizer.convert_ids_to_tokens(
                input_ids.squeeze(0).tolist()
            )
            replace_tokens(tokens)
            invocation_tokens, invocation_attrs = split_invocations(
                tokens, token_attributions
            )
            syscall_attr = get_syscall_attr(
                tokenizer, invocation_tokens, invocation_attrs
            )
            merge_syscall_attr(state.total_syscall_attr, {
                name: score * candidate.depth_factor
                for name, score in syscall_attr.items()
            })
            state.selected_signatures.append(candidate.signature)
        if not progressed:
            break

    results = []
    for state in states:
        cohort = state.cohort
        scores = _normalize_positive_scores(state.total_syscall_attr)
        if len(state.selected_signatures) < 8 or not scores:
            logger.info(
                "[Attribution] Stage %d objective %s skipped: "
                "%d/%d correctly predicted/examined samples",
                stage, cohort.objective, len(state.selected_signatures),
                state.examined,
            )
            continue
        signature_digest = hashlib.sha256(
            "\n".join(state.selected_signatures).encode("utf-8")
        ).hexdigest()
        results.append({
            "stage": stage,
            "objective": cohort.objective,
            "canonical_members": list(cohort.canonical_members),
            "active_exact_classes": list(cohort.active_exact_classes),
            "deployment_version": int(deployment_version),
            "selected_signature_digest": signature_digest,
            "analyzed_count": len(state.selected_signatures),
            "examined_count": state.examined,
            "depth_weight": cohort.depth_weight,
            "scores": scores,
        })
        logger.info(
            "[Attribution] Stage %d objective %s examined %d candidates, "
            "analyzed %d samples, and produced %d syscall scores",
            stage, cohort.objective, state.examined,
            len(state.selected_signatures), len(scores),
        )
    return results
