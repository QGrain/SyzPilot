"""Regression tests for guidance-side reachability label handling."""

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYZER_DIR = REPO_ROOT / "analyzer"
sys.path.insert(0, str(ANALYZER_DIR))

from common.label_contract import is_positive_one_hot, one_hot_label_class
from sequence_miner import SequencePatternMiner, extract_syscalls_from_program


def test_exact_one_hot_label_contract():
    assert one_hot_label_class([True, False, False], 3) == 0
    assert one_hot_label_class([False, True, False], 3) == 1
    assert one_hot_label_class([False, False, True], 3) == 2
    assert one_hot_label_class([False, False, False], 3) is None
    assert one_hot_label_class([True, True, False], 3) is None
    assert one_hot_label_class([False, True], 3) is None
    assert not is_positive_one_hot([True, False, False], 3)
    assert is_positive_one_hot([False, True, False], 3)


def test_sequence_miner_excludes_unreachable_and_invalid_labels():
    programs = [
        "unreachable$only()",
        "positive$one()\npositive$two()",
        "unknown$only()",
        "invalid$only()",
    ]
    labels = [
        [True, False, False],
        [False, True, False],
        [False, False, False],
        [True, True, False],
    ]

    patterns = SequencePatternMiner(
        programs, labels, num_classes=3
    ).mine_frequent_subsequences(
        min_support=1.0,
    )

    assert patterns == [(('positive$one', 'positive$two'), 1.0)]


def test_sequence_support_counts_each_program_once():
    repeated = "a()\nb()\na()\nb()"
    patterns = SequencePatternMiner(
        [repeated, "a()\nb()"],
        [[False, True], [False, True]],
        num_classes=2,
    ).mine_frequent_subsequences(min_support=0.0, max_gap=3)

    assert patterns
    assert all(support <= 1.0 for _, support in patterns)
    assert dict(patterns)[("a", "b")] == 1.0


def test_sequence_length_bounds_are_respected():
    patterns = SequencePatternMiner(
        ["a()\nb()\nc()"],
        [[False, True]],
        num_classes=2,
    ).mine_frequent_subsequences(
        min_support=1.0,
        min_length=3,
        max_length=3,
    )

    assert patterns == [(('a', 'b', 'c'), 1.0)]


def test_sequence_miner_supports_declared_length_four_and_five():
    patterns = SequencePatternMiner(
        ["a()\nb()\nc()\nd()\ne()"],
        [[False, True]],
        num_classes=2,
    ).mine_frequent_subsequences(
        min_support=1.0,
        max_gap=0,
        min_length=4,
        max_length=5,
    )

    assert patterns == [
        (("a", "b", "c", "d"), 1.0),
        (("a", "b", "c", "d", "e"), 1.0),
        (("b", "c", "d", "e"), 1.0),
    ]


def test_sequence_miner_rejects_wrong_label_width():
    patterns = SequencePatternMiner(
        ["wrong$width()\na()"],
        [[False, True]],
        num_classes=3,
    ).mine_frequent_subsequences(min_support=0.0)

    assert patterns == []


def test_sequence_miner_caps_sorted_patterns():
    patterns = SequencePatternMiner(
        ["a()\nb()\nc()\nd()"],
        [[False, True]],
        num_classes=2,
    ).mine_frequent_subsequences(
        min_support=1.0,
        min_length=2,
        max_length=4,
        max_patterns=2,
    )

    assert patterns == [
        (("a", "b"), 1.0),
        (("a", "b", "c"), 1.0),
    ]


def test_sequence_templates_require_absolute_positive_support():
    programs = ["a()\nb()" for _ in range(9)]
    labels = [[False, False, True] for _ in programs]
    miner = SequencePatternMiner(programs, labels, num_classes=3)

    assert miner.generate_templates(
        target_class=2,
        min_support=0.0,
        min_programs=10,
    ) == []


def test_sequence_parser_does_not_split_on_argument_assignments():
    program = "\n".join((
        "r0 = openat$foo(0xffffffffffffff9c, &(0x7f0000000000)="
        "'./file0\\x00', 0x0, 0x0)",
        "mount(&(0x7f0000000040)=@loop, &(0x7f0000000080)="
        "'./file0\\x00', &(0x7f00000000c0)='f2fs\\x00', 0x0, "
        "&(0x7f0000000100)={&r0})",
        "syz_mount_image$f2fs(&(0x7f0000000140)='f2fs\\x00', "
        "&(0x7f0000000180)='./file0\\x00', 0x0)",
        "r1 = accept4$netrom(r0, 0x0, 0x0, 0x0)",
        "r2 = ioctl$IOMMU_OPTION$IOMMU_OPTION_HUGE_PAGES(r0, 0x0)",
        "ioctl$IOMMU_DESTROY$ioas(r2, 0x0)",
    ))

    assert extract_syscalls_from_program(program) == [
        "openat$foo",
        "mount",
        "syz_mount_image$f2fs",
        "accept4$netrom",
        "ioctl$IOMMU_OPTION$IOMMU_OPTION_HUGE_PAGES",
        "ioctl$IOMMU_DESTROY$ioas",
    ]


def test_sequence_parser_ignores_non_call_lines():
    assert extract_syscalls_from_program(
        "# comment\n&(0x7f0000000000)=@loop\nnot a call\n"
    ) == []


def test_sequence_gap_counts_intervening_calls_like_paper_definition():
    patterns = dict(SequencePatternMiner(
        [
            "a()\nx()\nx()\nx()\nb()",
            "a()\nx()\nx()\nx()\nx()\nb()",
        ],
        [[False, True], [False, True]],
        num_classes=2,
    ).mine_frequent_subsequences(
        min_support=0.5,
        max_gap=3,
        min_length=2,
        max_length=2,
    ))

    assert patterns[("a", "b")] == 0.5


def test_sequence_class_threshold_includes_deeper_labels_and_uses_contrast():
    patterns = dict(SequencePatternMiner(
        [
            "common()\ndeep()",
            "common()\ndeep()",
            "common()\nshallow()",
            "common()\nunreachable()",
        ],
        [
            [False, False, True, False],
            [False, False, False, True],
            [False, True, False, False],
            [True, False, False, False],
        ],
        num_classes=4,
    ).mine_frequent_subsequences(
        target_class=2,
        min_support=1.0,
        max_gap=0,
        min_length=1,
        max_length=2,
    ))

    assert patterns[("deep",)] == 1.0
    assert ("common",) not in patterns
    assert ("common", "deep") in patterns


def test_sequence_miner_rejects_mismatched_program_and_label_counts():
    miner = SequencePatternMiner(
        ["a()"], [], num_classes=2
    )
    try:
        miner.mine_frequent_subsequences()
    except ValueError as error:
        assert "counts differ" in str(error)
    else:
        raise AssertionError("mismatched program and label counts were accepted")


def test_frontier_cap_does_not_drop_current_layer_contrast_winner():
    patterns = dict(SequencePatternMiner(
        [
            "a()\nx()\nb()\ny()",
            "a()\nx()",
            "a()\nx()",
            "a()\nx()",
        ],
        [
            [False, True],
            [False, True],
            [True, False],
            [True, False],
        ],
        num_classes=2,
    ).mine_frequent_subsequences(
        min_support=0.5,
        max_gap=0,
        min_length=2,
        max_length=2,
        max_frontier=1,
    ))

    assert patterns[("b", "y")] == 0.5
    assert ("a", "x") not in patterns


def test_sequence_deadline_can_abort_during_program_parsing():
    large_program = "\n".join("a()" for _ in range(100_000))
    patterns = SequencePatternMiner(
        [large_program], [[False, True]], num_classes=2
    ).mine_frequent_subsequences(deadline_seconds=1e-6)

    assert patterns == []
