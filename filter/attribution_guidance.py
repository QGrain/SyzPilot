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
import logging
import math
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

from model_v2 import TraceClassifierV2
from model_v2 import TraceClassifierServingWrapper
from common.curriculum import curriculum_class
from dataset_v2 import load_canonical_records
from attribution_wrapped import (
    split_invocations, replace_tokens, get_syscall_attr, merge_syscall_attr
)


class AttributionProbabilityWrapper(nn.Module):
    """Expose curriculum probabilities as the Integrated Gradients target."""

    def __init__(self, model, stage, cancel_event=None):
        super().__init__()
        self.serving_model = TraceClassifierServingWrapper(model, stage=stage)
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
        stage: Active exact-waypoint curriculum stage (2)
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
    if stage != 2:
        raise ValueError("attribution requires exact-waypoint curriculum stage 2")
    if target_class is not None and not 0 < target_class < num_classes:
        raise ValueError(
            f"target_class must be a reached class in [1, {num_classes - 1}]"
        )
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
        if class_idx == 0:
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
    model = TraceClassifierV2(base_model_path, num_classes, stage=stage)
    _raise_if_attribution_canceled(cancel_event)
    model.load_state_dict(
        torch.load(model_path, map_location="cpu", weights_only=True)
    )
    _raise_if_attribution_canceled(cancel_event)
    model.to(device)
    model.eval()
    attribution_model = AttributionProbabilityWrapper(
        model, stage=stage, cancel_event=cancel_event
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

        grouped_true_class = curriculum_class(
            true_class, num_classes, stage
        )

        # Attribution guidance must only use correctly predicted, reached
        # samples. A positive sample misclassified as another reached class is
        # not evidence for either class's guidance.
        if pred_class == 0 or pred_class != grouped_true_class:
            continue

        attributions, delta = lig.attribute(
            inputs=input_ids,
            baselines=torch.zeros_like(input_ids),
            additional_forward_args=(attention_mask,),
            target=grouped_true_class,
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
