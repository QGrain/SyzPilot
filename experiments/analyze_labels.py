#!/usr/bin/env python3
"""
Analyze training label distribution from receiver data.

Usage:
  python3 analyze_labels.py brain/receiver_data/kernel\\ BUG\\ in\\ validate_xmit_skb/3
  python3 analyze_labels.py brain/receiver_data/kernel\\ BUG\\ in\\ validate_xmit_skb/3 --target-pcs 0x870b2a58 0x87ebeb3a ...
  python3 analyze_labels.py brain/receiver_data/ --all
  python3 analyze_labels.py brain/receiver_data/ --all --compact
"""

import argparse
import pickle  # noqa: S403 — internal tool, reading our own training data
import sys
from collections import Counter
from pathlib import Path

DEFAULT_TARGET_PCS = [
    "0x870b2a58", "0x87ebeb3a", "0x8714da05", "0x8714e856",
    "0x8730e5b3", "0x84cf851c", "0x87925004", "0x8714d14e", "0x8714b0d2",
]


def load_labels(data_dir: Path) -> list:
    """Load all labels from labels_batch_*.pkl files in a directory."""
    label_files = sorted(data_dir.glob("labels_batch_*.pkl"))
    if not label_files:
        return []

    all_labels = []
    for lf in label_files:
        with open(lf, "rb") as f:
            batch = pickle.load(f)  # noqa: S301 — trusted internal data
        for prog_hash, label in batch.items():
            if isinstance(label, (list, tuple)):
                all_labels.append(tuple(bool(x) for x in label))
            else:
                all_labels.append(label)
    return all_labels


def analyze_distribution(all_labels: list, target_pcs: list, show_detail: bool = True):
    """Analyze and print label distribution."""
    if not all_labels:
        print("No labels found.")
        return

    total = len(all_labels)
    dim = len(all_labels[0])

    # Count per-class
    class_counts = [0] * dim
    all_zero_count = 0
    multi_label_count = 0

    for v in all_labels:
        active = [i for i, x in enumerate(v) if x]
        if len(active) == 0:
            all_zero_count += 1
        elif len(active) > 1:
            multi_label_count += 1
        for i in active:
            class_counts[i] += 1

    # Build class names
    class_names = []
    for i in range(dim):
        if i == 0:
            class_names.append("Unreachable")
        else:
            pc = target_pcs[i - 1] if i - 1 < len(target_pcs) else f"PC_{i}"
            class_names.append(f"Reach_Func{i} ({pc})")

    # Print summary
    labeled = total - all_zero_count
    print(f"Total samples:     {total:,}")
    print(f"Labeled samples:   {labeled:,} ({100 * labeled / total:.1f}%)")
    print(f"Unlabeled (zero):  {all_zero_count:,} ({100 * all_zero_count / total:.1f}%)")
    if multi_label_count > 0:
        print(f"Multi-label:       {multi_label_count:,}")
    print(f"Label dimension:   {dim}")
    print()

    if show_detail:
        print(f"{'Class':<35} {'Count':>8} {'%Total':>7} {'%Labeled':>9}")
        print("-" * 62)
        for i in range(dim):
            cnt = class_counts[i]
            pct_total = 100.0 * cnt / total if total > 0 else 0
            pct_labeled = 100.0 * cnt / labeled if labeled > 0 else 0
            print(f"{class_names[i]:<35} {cnt:>8,} {pct_total:>6.1f}% {pct_labeled:>8.1f}%")
        print("-" * 62)

    # Class balance assessment
    if labeled > 0:
        positive_counts = [class_counts[i] for i in range(1, dim)]
        unreachable_ratio = class_counts[0] / labeled
        max_positive = max(positive_counts) if positive_counts else 0
        min_positive = min(c for c in positive_counts if c > 0) if any(c > 0 for c in positive_counts) else 0

        print()
        print("Balance (among labeled samples):")
        print(f"  Unreachable:     {unreachable_ratio:.1%}")
        print(f"  Positive classes: {sum(1 for c in positive_counts if c > 0)}/{len(positive_counts)} non-zero")
        if min_positive > 0:
            print(f"  Imbalance ratio: {max_positive / min_positive:.1f}x (max/min)")
        zero_classes = [i + 1 for i, c in enumerate(positive_counts) if c == 0]
        if zero_classes:
            print(f"  Zero-sample:     {', '.join(f'Func{i}' for i in zero_classes)}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze training label distribution from SyzPilot receiver data"
    )
    parser.add_argument(
        "data_path",
        type=Path,
        help="Run directory (e.g., brain/receiver_data/task_name/run_id) "
             "or task_name directory with --all",
    )
    parser.add_argument(
        "--target-pcs",
        nargs="+",
        default=DEFAULT_TARGET_PCS,
        help="Target PC addresses for waypoint naming (default: case_1 PCs)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Analyze all run directories under data_path",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Compact output (summary only, no per-class detail)",
    )
    args = parser.parse_args()

    target_pcs = args.target_pcs

    if args.all:
        run_dirs = sorted(set(p.parent for p in args.data_path.rglob("labels_batch_*.pkl")))
        if not run_dirs:
            print(f"No label files found under {args.data_path}")
            sys.exit(1)
        print(f"Found {len(run_dirs)} run(s)\n")
        for run_dir in run_dirs:
            print(f"{'=' * 60}")
            print(f"Run: {run_dir}")
            print("=" * 60)
            labels = load_labels(run_dir)
            if labels:
                analyze_distribution(labels, target_pcs, show_detail=not args.compact)
            else:
                print("  No labels found.")
            print()
    else:
        if not args.data_path.exists():
            print(f"Error: {args.data_path} not found")
            sys.exit(1)

        pkl_files = list(args.data_path.glob("labels_batch_*.pkl"))
        if pkl_files:
            labels = load_labels(args.data_path)
            analyze_distribution(labels, target_pcs, show_detail=not args.compact)
        else:
            run_dirs = sorted(
                d for d in args.data_path.iterdir()
                if d.is_dir() and list(d.glob("labels_batch_*.pkl"))
            )
            if run_dirs:
                print(f"Found {len(run_dirs)} run(s)\n")
                for run_dir in run_dirs:
                    print(f"{'=' * 60}")
                    print(f"Run: {run_dir.name}")
                    print("=" * 60)
                    labels = load_labels(run_dir)
                    analyze_distribution(labels, target_pcs, show_detail=not args.compact)
                    print()
            else:
                print(f"No label files found in {args.data_path}")
                sys.exit(1)


if __name__ == "__main__":
    main()
