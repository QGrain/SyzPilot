"""Validation helpers for SyzPilot's exactly-one-hot training labels."""

from typing import Optional, Sequence


def one_hot_label_class(
    labels: Sequence[bool], num_classes: Optional[int] = None
) -> Optional[int]:
    """Return the selected class, or ``None`` for an invalid label.

    Class 0 is the explicit Unreachable class. Classes greater than zero are
    reached waypoint classes. Empty, all-zero, wrong-width, and multi-hot
    labels are invalid.
    """
    if labels is None:
        return None
    if num_classes is not None and len(labels) != num_classes:
        return None

    selected = [index for index, value in enumerate(labels) if bool(value)]
    if len(selected) != 1:
        return None
    return selected[0]


def is_positive_one_hot(
    labels: Sequence[bool], num_classes: Optional[int] = None
) -> bool:
    """Return whether a valid label selects a reached (non-zero) class."""
    selected = one_hot_label_class(labels, num_classes=num_classes)
    return selected is not None and selected > 0
