"""Shared two-stage curriculum label semantics for SyzPilot."""

from collections.abc import Mapping


VALID_CURRICULUM_STAGES = (1, 2)
# Schema 1 was the short-lived three-stage experiment. Schema 2 restores the
# paper-aligned binary-then-exact objective and must not reuse ternary models.
CURRICULUM_SCHEMA_VERSION = 2


def curriculum_class(class_index: int, num_classes: int, stage: int) -> int:
    """Map a waypoint-level class to the class used by a curriculum stage."""
    if stage not in VALID_CURRICULUM_STAGES:
        raise ValueError(f"invalid curriculum stage: {stage}")
    if not 0 <= class_index < num_classes:
        raise ValueError(
            f"class index {class_index} is outside [0, {num_classes - 1}]"
        )
    if stage == 1:
        return int(class_index > 0)
    return class_index


def curriculum_output_classes(num_classes: int, stage: int) -> int:
    """Return the number of logits/classes exposed by a curriculum stage."""
    if stage == 1:
        if num_classes < 2:
            raise ValueError("Stage 1 requires at least two output classes")
        return 2
    if stage == 2:
        if num_classes < 2:
            raise ValueError("Stage 2 requires at least two output classes")
        return num_classes
    raise ValueError(f"invalid curriculum stage: {stage}")


def infer_curriculum_stage(
    label_counts: Mapping[int, int],
    num_classes: int,
    *,
    minimum_total: int,
    minimum_positive_class: int,
) -> int:
    """Infer the most detailed trainable stage from canonical label counts."""
    if num_classes < 2:
        return 0
    counts = [int(label_counts.get(index, 0)) for index in range(num_classes)]
    if any(count < 0 for count in counts):
        raise ValueError("label counts must be non-negative")
    total = sum(counts)
    positive_total = sum(counts[1:])
    if total < minimum_total or positive_total < minimum_positive_class:
        return 0

    if any(count < minimum_positive_class for count in counts):
        return 1
    return 2
