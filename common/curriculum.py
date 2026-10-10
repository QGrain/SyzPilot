"""Shared three-stage curriculum label semantics for SyzPilot."""

from collections.abc import Iterable, Mapping


VALID_CURRICULUM_STAGES = (1, 2, 3)
CURRICULUM_SCHEMA_VERSION = 4
DENSE_PAPER_MODE = "dense-paper"
SPARSE_ADAPTIVE_MODE = "sparse-adaptive"
VALID_CURRICULUM_MODES = (DENSE_PAPER_MODE, SPARSE_ADAPTIVE_MODE)


def normalize_curriculum_mode(mode: str) -> str:
    """Return one explicit curriculum mode or raise on ambiguous input."""
    normalized = str(mode).strip().lower()
    if normalized not in VALID_CURRICULUM_MODES:
        raise ValueError(
            f"invalid curriculum mode {mode!r}; expected one of "
            f"{VALID_CURRICULUM_MODES}"
        )
    return normalized


def is_exact_curriculum_objective(stage: int, num_classes: int) -> bool:
    """Return whether serving labels identify every reachable target exactly."""
    if stage not in VALID_CURRICULUM_STAGES:
        raise ValueError(f"invalid curriculum stage: {stage}")
    if num_classes < 2:
        raise ValueError("curriculum learning requires at least two classes")
    return stage == 3 or (stage == 1 and num_classes == 2)


def stage2_deep_class_start(num_classes: int) -> int:
    """Return the first exact class assigned to the Stage-2 Deep group."""
    if num_classes < 3:
        raise ValueError("Stage 2 requires at least two reached classes")
    return 1 + (num_classes - 1) // 2


def normalize_stage3_active_classes(
    active_classes: Iterable[int] | None,
    num_classes: int,
) -> tuple[int, ...]:
    """Validate the exact classes exposed by a Stage-3 objective."""
    if num_classes < 2:
        raise ValueError("Stage 3 requires at least two output classes")
    if active_classes is None:
        raise ValueError("Stage 3 requires active exact classes")
    normalized = tuple(sorted(set(int(index) for index in active_classes)))
    if (not normalized or normalized[0] < 0 or
            normalized[-1] >= num_classes):
        raise ValueError("Stage-3 active classes are outside the label schema")
    if 0 not in normalized or num_classes - 1 not in normalized:
        raise ValueError(
            "Stage 3 requires Unreachable and the final target class"
        )
    return normalized


def curriculum_class(
    class_index: int,
    num_classes: int,
    stage: int,
    active_classes: Iterable[int] | None = None,
) -> int:
    """Map an exact waypoint class to one active curriculum class."""
    if stage not in VALID_CURRICULUM_STAGES:
        raise ValueError(f"invalid curriculum stage: {stage}")
    if not 0 <= class_index < num_classes:
        raise ValueError(
            f"class index {class_index} is outside [0, {num_classes - 1}]"
        )
    if stage == 1:
        return int(class_index > 0)
    if stage == 2:
        deep_start = stage2_deep_class_start(num_classes)
        if class_index == 0:
            return 0
        return 1 if class_index < deep_start else 2

    active = normalize_stage3_active_classes(active_classes, num_classes)
    try:
        return active.index(class_index)
    except ValueError:
        # Sparse Stage 3 keeps one catch-all reached class after the supported
        # exact classes. It preserves reachability evidence without exposing
        # an unsupported waypoint depth to the fuzzer.
        return len(active)


def curriculum_output_classes(
    num_classes: int,
    stage: int,
    active_classes: Iterable[int] | None = None,
) -> int:
    """Return the number of logits/classes used by a curriculum objective."""
    if stage == 1:
        if num_classes < 2:
            raise ValueError("Stage 1 requires at least two output classes")
        return 2
    if stage == 2:
        stage2_deep_class_start(num_classes)
        return 3
    if stage == 3:
        active = normalize_stage3_active_classes(
            active_classes, num_classes
        )
        return len(active) + int(len(active) < num_classes)
    raise ValueError(f"invalid curriculum stage: {stage}")


def curriculum_metric_class_members(
    num_classes: int,
    stage: int,
    active_classes: Iterable[int] | None = None,
) -> dict[int, tuple[int, ...]]:
    """Map compact metric indices to their canonical exact label members."""
    if stage == 1:
        if num_classes < 2:
            raise ValueError("Stage 1 requires at least two output classes")
        return {0: (0,), 1: tuple(range(1, num_classes))}
    if stage == 2:
        deep_start = stage2_deep_class_start(num_classes)
        return {
            0: (0,),
            1: tuple(range(1, deep_start)),
            2: tuple(range(deep_start, num_classes)),
        }
    if stage == 3:
        active = normalize_stage3_active_classes(
            active_classes, num_classes
        )
        members = {
            metric_index: (exact_class,)
            for metric_index, exact_class in enumerate(active)
        }
        inactive = tuple(
            exact_class for exact_class in range(num_classes)
            if exact_class not in active
        )
        if inactive:
            members[len(active)] = inactive
        return members
    raise ValueError(f"invalid curriculum stage: {stage}")


def stage3_active_classes(
    label_counts: Mapping[int, int],
    num_classes: int,
    *,
    minimum_class_samples: int,
    train_counts: Mapping[int, int] | None = None,
    validation_counts: Mapping[int, int] | None = None,
    minimum_validation_samples: int = 1,
    previous_active_classes: Iterable[int] | None = None,
    curriculum_mode: str = DENSE_PAPER_MODE,
) -> tuple[int, ...]:
    """Select exact classes that have enough support for Stage 3.

    Dense-paper mode requires every canonical class. Sparse-adaptive mode
    requires Unreachable and the final target, while supported intermediate
    waypoints enter independently. A previously active set is monotonic.
    """
    mode = normalize_curriculum_mode(curriculum_mode)
    if (num_classes < 3 or minimum_class_samples <= 0 or
            minimum_validation_samples <= 0):
        return ()
    counts = [int(label_counts.get(index, 0)) for index in range(num_classes)]
    if any(count < 0 for count in counts):
        raise ValueError("label counts must be non-negative")
    if (counts[0] < minimum_class_samples or
            counts[-1] < minimum_class_samples):
        return ()

    split_counts_supplied = (
        train_counts is not None or validation_counts is not None
    )
    if split_counts_supplied:
        if train_counts is None or validation_counts is None:
            raise ValueError(
                "train and validation counts must be provided together"
            )
        train = [
            int(train_counts.get(index, 0)) for index in range(num_classes)
        ]
        validation = [
            int(validation_counts.get(index, 0))
            for index in range(num_classes)
        ]
        if any(count < 0 for count in train + validation):
            raise ValueError("split label counts must be non-negative")
    else:
        train = validation = None

    previous = ()
    if previous_active_classes is not None:
        previous = normalize_stage3_active_classes(
            previous_active_classes, num_classes
        )

    mandatory = set(previous)
    if mode == DENSE_PAPER_MODE:
        mandatory.update(range(num_classes))
    else:
        mandatory.update((0, num_classes - 1))
    if any(counts[index] < minimum_class_samples for index in mandatory):
        return ()
    if split_counts_supplied and any(
            train[index] <= 0 or
            validation[index] < minimum_validation_samples
            for index in mandatory):
        return ()

    selected = set(mandatory)
    for index, count in enumerate(counts):
        if count < minimum_class_samples:
            continue
        if (split_counts_supplied and
                (train[index] <= 0 or
                 validation[index] < minimum_validation_samples)):
            # A new optional waypoint joins in a later round once its holdout
            # evidence is ready. It must not disable an established objective.
            continue
        selected.add(index)
    return tuple(sorted(selected))


def infer_curriculum_stage(
    label_counts: Mapping[int, int],
    num_classes: int,
    *,
    minimum_total: int,
    minimum_positive_class: int,
    curriculum_mode: str = DENSE_PAPER_MODE,
) -> int:
    """Infer the most detailed trainable stage from canonical label counts."""
    mode = normalize_curriculum_mode(curriculum_mode)
    if num_classes < 2:
        return 0
    if minimum_total <= 0 or minimum_positive_class <= 0:
        raise ValueError("curriculum support thresholds must be positive")
    counts = [int(label_counts.get(index, 0)) for index in range(num_classes)]
    if any(count < 0 for count in counts):
        raise ValueError("label counts must be non-negative")
    total = sum(counts)
    positive_total = sum(counts[1:])
    if total < minimum_total or positive_total < minimum_positive_class:
        return 0
    if num_classes < 3:
        return 1

    deep_start = stage2_deep_class_start(num_classes)
    stage2_counts = (
        counts[0],
        sum(counts[1:deep_start]),
        sum(counts[deep_start:]),
    )
    if min(stage2_counts) < minimum_positive_class:
        return 1

    active = stage3_active_classes(
        label_counts,
        num_classes,
        minimum_class_samples=minimum_positive_class,
        curriculum_mode=mode,
    )
    if active:
        return 3

    return 2
