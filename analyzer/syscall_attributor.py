import os
import re
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from functools import reduce


def read_prog(prog_path: str) -> str:
    with open(prog_path, 'r') as f:
        return f.read()


def extract_syscalls(prog_str: str) -> list[str]:
    prog_lines = prog_str.splitlines()
    valid_syscalls = []
    for line in prog_lines:
        line = line.strip()
        if line.startswith("#"):
            continue
        # find the first syscall name in the line
        line_syscalls = re.findall(r'(?<![\\])\b([A-Za-z0-9_$]+)\(', line)
        if len(line_syscalls) > 0 and line_syscalls[0] != '' and not line_syscalls[0].isdigit():
            valid_syscalls.append(line_syscalls[0])
    return valid_syscalls


def process_single_prog(prog_path: str) -> tuple[dict[str, int], list[str]]:
    """
    Process a single prog file and return syscall distribution and sequence
    """
    prog_str = read_prog(prog_path)
    syscalls = extract_syscalls(prog_str)

    syscalls_distribution = {}
    for syscall in syscalls:
        syscalls_distribution[syscall] = syscalls_distribution.get(syscall, 0) + 1

    return syscalls_distribution, syscalls


def stat_syscalls_distribution(prog_dir: str, use_multiprocessing: bool = True) -> tuple[dict[str, int], list[list[str]]]:
    """
    Statistics of syscall distribution across all progs.
    Returns: (syscalls_distribution, all_syscall_sequences)
    """
    prog_files = [os.path.join(prog_dir, f) for f in os.listdir(prog_dir) if os.path.isfile(os.path.join(prog_dir, f))]

    if use_multiprocessing and len(prog_files) > 1:
        # Use multiprocessing pool
        num_processes = min(cpu_count(), len(prog_files))
        with Pool(processes=num_processes) as pool:
            results = list(tqdm(
                pool.imap(process_single_prog, prog_files),
                total=len(prog_files),
                desc="Processing progs"
            ))
    else:
        # Single process with progress bar
        results = [process_single_prog(prog_path) for prog_path in tqdm(prog_files, desc="Processing progs")]

    # Aggregate results
    syscalls_distribution = {}
    all_syscall_sequences = []

    for dist, seq in results:
        all_syscall_sequences.append(seq)
        for syscall, count in dist.items():
            syscalls_distribution[syscall] = syscalls_distribution.get(syscall, 0) + count

    return syscalls_distribution, all_syscall_sequences


def contains_subsequence(sequence: list[str], pattern: tuple[str, ...]) -> bool:
    """
    Check if sequence contains pattern as a subsequence (preserving relative order).
    """
    pattern_idx = 0
    for item in sequence:
        if pattern_idx < len(pattern) and item == pattern[pattern_idx]:
            pattern_idx += 1
    return pattern_idx == len(pattern)


def find_high_frequency_subsequences(
    sequences: list[list[str]],
    min_support: float = 0.5,
    min_length: int = 3,
    max_length: int = 6,
    max_gap: int = 3,
    top_k: int = 20
) -> list[tuple[tuple[str, ...], int, float]]:
    """
    Find high-frequency non-contiguous subsequences using sliding window approach.
    More efficient than exhaustive enumeration - uses gap-constrained mining.

    Args:
        sequences: List of syscall sequences
        min_support: Minimum ratio of sequences that must contain the pattern (0.0-1.0)
        min_length: Minimum length of subsequence
        max_length: Maximum length of subsequence
        max_gap: Maximum gap allowed between consecutive elements in pattern
        top_k: Return top-k most frequent subsequences

    Returns:
        List of (pattern, count, support_ratio) sorted by frequency
    """
    from collections import defaultdict

    print(f"Mining frequent subsequences (min_support={min_support}, min_length={min_length}, max_length={max_length}, max_gap={max_gap})...")

    # Count how many sequences contain each subsequence
    pattern_support = defaultdict(int)
    total_sequences = len(sequences)
    min_count = int(total_sequences * min_support)

    def extract_patterns_from_seq(seq: list[str]) -> set[tuple[str, ...]]:
        """Extract gap-constrained subsequences from a single sequence"""
        patterns = set()
        n = len(seq)

        # Use dynamic programming to build patterns incrementally
        # dp[i][pattern] = True if pattern can be formed ending at position i
        for start in range(n):
            # Build patterns starting from position 'start'
            # Use BFS-like approach with gap constraint
            queue = [(start, tuple([seq[start]]))]
            visited = {(start, tuple([seq[start]]))}

            while queue:
                pos, pattern = queue.pop(0)

                # Add pattern if it meets length requirement
                if len(pattern) >= min_length:
                    patterns.add(pattern)

                # Extend pattern if not too long
                if len(pattern) < max_length:
                    # Try to extend with elements within max_gap distance
                    for next_pos in range(pos + 1, min(pos + max_gap + 2, n)):
                        next_pattern = pattern + (seq[next_pos],)
                        state = (next_pos, next_pattern)
                        if state not in visited:
                            visited.add(state)
                            queue.append(state)

        return patterns

    for seq in tqdm(sequences, desc="Mining patterns"):
        patterns = extract_patterns_from_seq(seq)
        for pattern in patterns:
            pattern_support[pattern] += 1

    # Filter by minimum support and sort by frequency
    frequent_patterns = [
        (pattern, count, count / total_sequences)
        for pattern, count in pattern_support.items()
        if count >= min_count
    ]

    # Sort by count (descending), then by length (descending)
    frequent_patterns.sort(key=lambda x: (-x[1], -len(x[0])))

    return frequent_patterns[:top_k]


def find_high_frequency_subsequences_with_must_include(
    sequences: list[list[str]],
    must_include: str,
    min_support: float = 0.5,
    min_length: int = 3,
    max_length: int = 10,
    max_gap: int = 3,
    top_k: int = 10
) -> list[tuple[tuple[str, ...], int, float]]:
    """
    Find high-frequency subsequences that MUST include a specific syscall.

    Args:
        sequences: List of syscall sequences
        must_include: The syscall that must be included
        min_support: Minimum support ratio (0.0-1.0)
        min_length: Minimum length of subsequence
        max_length: Maximum length of subsequence
        max_gap: Maximum gap between consecutive elements
        top_k: Return top-k most frequent subsequences

    Returns:
        List of (pattern, count, support_ratio) containing must_include syscall
    """
    # Filter sequences that contain the must_include syscall
    filtered_sequences = [seq for seq in sequences if must_include in seq]

    if not filtered_sequences:
        print(f"Warning: No sequences contain '{must_include}'")
        return []

    print(f"Filtered to {len(filtered_sequences)}/{len(sequences)} sequences containing '{must_include}'")

    # Find frequent patterns on filtered sequences
    patterns = find_high_frequency_subsequences(
        filtered_sequences,
        min_support=min_support,
        min_length=min_length,
        max_length=max_length,
        max_gap=max_gap,
        top_k=top_k * 2  # Get more candidates for filtering
    )

    # Filter patterns that actually contain must_include
    filtered_patterns = [(p, c, s) for p, c, s in patterns if must_include in p]

    return filtered_patterns[:top_k]


def find_frequent_patterns(sequences: list[list[str]], min_length: int = 3, top_k: int = 20) -> list[tuple[tuple[str, ...], int]]:
    """
    Find frequent subsequence patterns (continuous substrings) across all sequences.
    Returns top_k most frequent patterns of length >= min_length.
    """
    from collections import Counter

    pattern_counter = Counter()

    print(f"Extracting frequent patterns from {len(sequences)} sequences...")
    for seq in tqdm(sequences, desc="Extracting patterns"):
        # Extract all continuous subsequences of length >= min_length
        for length in range(min_length, len(seq) + 1):
            for i in range(len(seq) - length + 1):
                pattern = tuple(seq[i:i+length])
                pattern_counter[pattern] += 1

    # Get top_k most frequent patterns
    return pattern_counter.most_common(top_k)


def main():
    # prog_dir = "/root/fuzzers/SyzPilot-fuzzer-syzkaller/workdir/syzkaller/case_32/2/dump/0x89d7bdad/programs"
    prog_dir = "/root/fuzzers/SyzPilot-fuzzer-syzkaller/workdir/syzkaller/case_32/run2_dump_100"

    print(f"Analyzing progs in: {prog_dir}\n")

    # Step 1: Stat syscalls distribution with multiprocessing
    syscalls_distribution, all_syscall_sequences = stat_syscalls_distribution(prog_dir)

    # Get top-K syscalls
    sorted_syscalls_distribution = sorted(syscalls_distribution.items(), key=lambda x: x[1], reverse=True)
    top_k = 5
    top_k_syscalls = [syscall for syscall, _ in sorted_syscalls_distribution[:top_k]]

    print(f"\n{'='*60}")
    print(f"Syscall Distribution (Top {top_k} most frequent):")
    print(f"{'='*60}")
    for i, (syscall, count) in enumerate(sorted_syscalls_distribution[:top_k], 1):
        print(f"{i}. {syscall}: {count}")

    # Step 2: Find high-frequency subsequences (non-contiguous, preserving order)
    print(f"\n{'='*60}")
    print(f"[Strategy 1] High-Frequency Subsequences:")
    print(f"{'='*60}")

    # Use lower min_support for larger datasets (more tolerant to outliers)
    min_support = 0.1 if len(all_syscall_sequences) > 50 else 0.3

    high_freq_patterns = find_high_frequency_subsequences(
        all_syscall_sequences,
        min_support=min_support,
        min_length=3,
        max_length=6,
        max_gap=3,
        top_k=10
    )

    print(f"\nTop-10 High-Frequency Subsequences:")
    for i, (pattern, count, support) in enumerate(high_freq_patterns, 1):
        print(f"\nPattern #{i}:")
        print(f"  Appears in: {count}/{len(all_syscall_sequences)} sequences ({support*100:.1f}%)")
        print(f"  Length: {len(pattern)}")
        print(f"  Sequence: {' -> '.join(pattern)}")

    # Step 3: Find high-frequency patterns for top-K syscalls
    print(f"\n{'='*60}")
    print(f"[Strategy 2] High-Frequency Patterns for Top-{top_k} Syscalls:")
    print(f"{'='*60}")

    for syscall in top_k_syscalls:
        print(f"\n\nAnalyzing patterns containing '{syscall}':")
        print(f"{'-'*60}")

        patterns = find_high_frequency_subsequences_with_must_include(
            all_syscall_sequences,
            syscall,
            min_support=min_support,
            min_length=3,
            max_length=5,
            max_gap=2,
            top_k=5
        )

        if patterns:
            for i, (pattern, count, support) in enumerate(patterns, 1):
                print(f"\n  Pattern #{i}:")
                print(f"    Appears in: {count} sequences ({support*100:.1f}%)")
                print(f"    Length: {len(pattern)}")
                print(f"    Sequence: {' -> '.join(pattern)}")
        else:
            print(f"  No frequent patterns found for '{syscall}'")

    # Step 4: Find frequent continuous patterns (for comparison)
    print(f"\n{'='*60}")
    print(f"[Strategy 3] Frequent Continuous Patterns (for comparison):")
    print(f"{'='*60}")

    frequent_patterns = find_frequent_patterns(all_syscall_sequences, min_length=3, top_k=10)

    for i, (pattern, count) in enumerate(frequent_patterns, 1):
        print(f"\nPattern #{i}:")
        print(f"  Appears: {count} times ({count/len(all_syscall_sequences)*100:.1f}%)")
        print(f"  Length: {len(pattern)}")
        print(f"  Sequence: {' -> '.join(pattern)}")


if __name__ == "__main__":
    main()