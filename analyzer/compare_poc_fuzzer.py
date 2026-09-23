#!/usr/bin/env python3
"""
Compare PoC programs with fuzzer-discovered programs that reach target waypoints.

Analyzes the gap between what the PoC does and what the fuzzer generates,
identifying missing factors needed for vulnerability reproduction.

Usage:
    python compare_poc_fuzzer.py \
        --poc ~/kernels/SyzPilot-experiments/configs/case_1.poc.syz \
        --dump_dir /path/to/dump/0x8714b0d2/programs/ \
        --output report.json
"""

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


# === Syscall extraction (reuse from syscall_attribution.py pattern) ===

SYSCALL_RE = re.compile(r'^(\w[\w$]*)\(')

def extract_syscalls(prog_str: str) -> List[str]:
    """Extract ordered list of syscall names from a syz-program."""
    syscalls = []
    for line in prog_str.strip().split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        m = SYSCALL_RE.match(line)
        if m:
            syscalls.append(m.group(1))
    return syscalls


def extract_resource_flow(prog_str: str) -> Dict[str, List[str]]:
    """Extract resource production/consumption relationships.

    Returns: {resource_var: [consuming_syscall_line_numbers]}
    """
    producers = {}  # resource_var -> producer_syscall
    consumers = defaultdict(list)  # resource_var -> [consumer_syscalls]

    for i, line in enumerate(prog_str.strip().split('\n')):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        m = SYSCALL_RE.match(line)
        if not m:
            continue
        syscall = m.group(1)

        # Find resource production: r0 = syscall(...)
        prod_match = re.match(r'(\w+)\s*=\s*\w+', line)
        if prod_match:
            res_var = prod_match.group(1)
            producers[res_var] = syscall

        # Find resource consumption: syscall(..., r0, ...)
        for res_var in re.findall(r'\br(\d+)\b', line):
            full_var = f'r{res_var}'
            if full_var in producers:
                consumers[full_var].append(syscall)

    return dict(consumers)


def extract_args(prog_str: str) -> Dict[str, List[str]]:
    """Extract syscall arguments (simplified)."""
    args = {}
    for line in prog_str.strip().split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        m = re.match(r'(\w[\w$]*)\((.*)\)', line)
        if m:
            syscall = m.group(1)
            arg_str = m.group(2)
            args[syscall] = arg_str
    return args


# === PoC Analysis ===

class PoCAnalyzer:
    """Analyze a PoC syz-program to extract critical factors."""

    def __init__(self, poc_path: str):
        with open(poc_path, 'r') as f:
            content = f.read()
        # Remove comments
        lines = []
        for line in content.split('\n'):
            stripped = line.strip()
            if stripped.startswith('#'):
                continue
            lines.append(line)
        self.poc_text = '\n'.join(lines)
        self.syscalls = extract_syscalls(self.poc_text)
        self.args = extract_args(self.poc_text)
        self.resource_flow = extract_resource_flow(self.poc_text)

    def get_required_syscalls(self) -> List[str]:
        """Return the ordered list of PoC syscalls."""
        return self.syscalls

    def get_socket_types(self) -> Dict[str, Tuple[int, int, int]]:
        """Extract socket domain/type/protocol from socket calls."""
        sockets = {}
        for line in self.poc_text.split('\n'):
            m = re.match(r'(\w+)\s*=\s*socket(?:\$\w+)?\((\w+),\s*(\w+),\s*(\w+)\)', line.strip())
            if m:
                var, domain, stype, proto = m.group(1), m.group(2), m.group(3), m.group(4)
                sockets[var] = (domain, stype, proto)
        return sockets

    def get_critical_values(self) -> Dict[str, Dict]:
        """Extract critical argument values that must be correct."""
        critical = {}
        for syscall, arg_str in self.args.items():
            if 'socket' in syscall.lower():
                critical[syscall] = {'args': arg_str, 'type': 'socket_config'}
            elif 'setsockopt' in syscall.lower():
                critical[syscall] = {'args': arg_str, 'type': 'socket_option'}
            elif 'sendto' in syscall.lower() or 'sendmsg' in syscall.lower():
                critical[syscall] = {'args': arg_str, 'type': 'send_data'}
            elif 'bind' in syscall.lower():
                critical[syscall] = {'args': arg_str, 'type': 'bind_config'}
            elif 'getsockname' in syscall.lower():
                critical[syscall] = {'args': arg_str, 'type': 'name_query'}
        return critical


# === Fuzzer Program Analysis ===

class FuzzerProgramAnalyzer:
    """Analyze a fuzzer-discovered program against PoC factors."""

    def __init__(self, prog_path: str, poc: PoCAnalyzer):
        with open(prog_path, 'r') as f:
            self.prog_text = f.read()
        self.syscalls = extract_syscalls(self.prog_text)
        self.args = extract_args(self.prog_text)
        self.resource_flow = extract_resource_flow(self.prog_text)
        self.poc = poc

    def compute_syscall_overlap(self) -> Dict[str, bool]:
        """Check which PoC syscalls are present in the fuzzer program."""
        poc_syscalls = set(self.poc.get_required_syscalls())
        fuzzer_syscalls = set(self.syscalls)
        return {s: (s in fuzzer_syscalls) for s in poc_syscalls}

    def compute_order_preservation(self) -> float:
        """Score how well the fuzzer program preserves PoC syscall order.

        Returns 0.0 (no order preserved) to 1.0 (perfect order).
        """
        poc_seq = self.poc.get_required_syscalls()
        fuzzer_seq = self.syscalls

        # Find longest common subsequence
        # Use set intersection for efficiency
        poc_set = set(poc_seq)
        fuzzer_filtered = [s for s in fuzzer_seq if s in poc_set]

        if not fuzzer_filtered or not poc_seq:
            return 0.0

        # LCS length
        m, n = len(poc_seq), len(fuzzer_filtered)
        # For very long sequences, use approximate method
        if m * n > 10000:
            return self._approximate_order_score(poc_seq, fuzzer_filtered)

        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if poc_seq[i-1] == fuzzer_filtered[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        lcs_len = dp[m][n]
        return lcs_len / m if m > 0 else 0.0

    def _approximate_order_score(self, poc_seq: List[str], fuzzer_seq: List[str]) -> float:
        """Approximate order preservation score for long sequences."""
        # Greedy match: for each PoC syscall, find its next occurrence in fuzzer
        fuzzer_idx = 0
        matched = 0
        for poc_call in poc_seq:
            while fuzzer_idx < len(fuzzer_seq):
                if fuzzer_seq[fuzzer_idx] == poc_call:
                    matched += 1
                    fuzzer_idx += 1
                    break
                fuzzer_idx += 1
        return matched / len(poc_seq) if poc_seq else 0.0

    def compute_resource_wiring_score(self) -> float:
        """Score how well resource dependencies match the PoC.

        Checks if the same resource flow pattern exists.
        """
        poc_flow = self.poc.resource_flow
        fuzzer_flow = self.resource_flow

        if not poc_flow:
            return 1.0  # No resources to match

        matched = 0
        total = len(poc_flow)
        for res_var, poc_consumers in poc_flow.items():
            # We can't match exact resource vars (they're local),
            # so match the pattern: producer_syscall -> consumer_syscall
            # Check if fuzzer has any resource that flows between same syscall types
            for f_res, f_consumers in fuzzer_flow.items():
                # Check if any consumer matches
                if any(c in poc_consumers for c in f_consumers):
                    matched += 1
                    break

        return matched / total if total > 0 else 0.0

    def compute_socket_setup_score(self) -> float:
        """Score how well socket setup matches the PoC."""
        poc_sockets = self.poc.get_socket_types()
        if not poc_sockets:
            return 1.0

        fuzzer_sockets = {}
        for line in self.prog_text.split('\n'):
            m = re.match(r'(\w+)\s*=\s*socket(?:\$\w+)?\((\w+),\s*(\w+),\s*(\w+)\)', line.strip())
            if m:
                var, domain, stype, proto = m.group(1), m.group(2), m.group(3), m.group(4)
                fuzzer_sockets[var] = (domain, stype, proto)

        # Check if any fuzzer socket matches any PoC socket
        matched = 0
        for poc_var, poc_cfg in poc_sockets.items():
            for f_var, f_cfg in fuzzer_sockets.items():
                if f_cfg == poc_cfg:
                    matched += 1
                    break

        return matched / len(poc_sockets) if poc_sockets else 0.0

    def has_critical_syscall(self, syscall_pattern: str) -> bool:
        """Check if program contains a syscall matching the pattern."""
        for s in self.syscalls:
            if syscall_pattern in s:
                return True
        return False

    def compute_gap_score(self) -> Dict[str, float]:
        """Compute overall gap analysis."""
        syscall_overlap = self.compute_syscall_overlap()
        overlap_ratio = sum(syscall_overlap.values()) / len(syscall_overlap) if syscall_overlap else 0.0

        return {
            'syscall_overlap_ratio': overlap_ratio,
            'order_preservation': self.compute_order_preservation(),
            'resource_wiring': self.compute_resource_wiring_score(),
            'socket_setup': self.compute_socket_setup_score(),
            'program_length': len(self.syscalls),
            'poc_length': len(self.poc.get_required_syscalls()),
        }


# === Batch Comparison ===

class BatchComparator:
    """Compare all fuzzer programs against a PoC."""

    def __init__(self, poc_path: str, dump_dir: str):
        self.poc = PoCAnalyzer(poc_path)
        self.dump_dir = Path(dump_dir)
        self.programs = sorted(self.dump_dir.glob('*'))
        # Filter to actual files (not directories)
        self.programs = [p for p in self.programs if p.is_file() and not p.name.startswith('.')]

    def run(self, max_programs: int = 0) -> Dict:
        """Run comparison on all programs."""
        programs = self.programs
        if max_programs > 0:
            programs = programs[:max_programs]

        results = []
        for prog_path in programs:
            try:
                analyzer = FuzzerProgramAnalyzer(str(prog_path), self.poc)
                score = analyzer.compute_gap_score()
                score['program'] = prog_path.name
                score['has_sendto'] = analyzer.has_critical_syscall('sendto')
                score['has_packet_socket'] = analyzer.has_critical_syscall('socket$packet')
                score['has_bind'] = analyzer.has_critical_syscall('bind$packet')
                score['has_getsockname'] = analyzer.has_critical_syscall('getsockname')
                score['has_nl_route'] = analyzer.has_critical_syscall('nl_route_sched')
                results.append(score)
            except Exception as e:
                print(f"Error processing {prog_path.name}: {e}", file=sys.stderr)

        return self._summarize(results)

    def _summarize(self, results: List[Dict]) -> Dict:
        """Produce summary statistics."""
        if not results:
            return {'error': 'No programs analyzed'}

        # Aggregate scores
        metrics = ['syscall_overlap_ratio', 'order_preservation', 'resource_wiring', 'socket_setup']
        summary = {}
        for metric in metrics:
            values = [r[metric] for r in results]
            summary[metric] = {
                'mean': sum(values) / len(values),
                'max': max(values),
                'min': min(values),
                'above_50pct': sum(1 for v in values if v > 0.5) / len(values),
            }

        # Critical syscall presence
        critical = ['has_sendto', 'has_packet_socket', 'has_bind', 'has_getsockname', 'has_nl_route']
        summary['critical_syscalls'] = {}
        for crit in critical:
            count = sum(1 for r in results if r.get(crit, False))
            summary['critical_syscalls'][crit] = {
                'count': count,
                'ratio': count / len(results),
            }

        # Top-k closest programs
        results.sort(key=lambda r: r['syscall_overlap_ratio'] + r['order_preservation'], reverse=True)
        summary['top_10_closest'] = results[:10]

        # Length distribution
        lengths = [r['program_length'] for r in results]
        summary['length_distribution'] = {
            'mean': sum(lengths) / len(lengths),
            'min': min(lengths),
            'max': max(lengths),
            'poc_length': len(self.poc.get_required_syscalls()),
        }

        summary['total_programs'] = len(results)
        summary['poc_syscalls'] = self.poc.get_required_syscalls()

        return summary


def main():
    parser = argparse.ArgumentParser(description='Compare PoC with fuzzer-discovered programs')
    parser.add_argument('--poc', required=True, help='Path to PoC .syz file')
    parser.add_argument('--dump_dir', required=True, help='Path to dump/programs/ directory')
    parser.add_argument('--output', '-o', default=None, help='Output JSON file')
    parser.add_argument('--max_programs', type=int, default=0, help='Max programs to analyze (0=all)')
    args = parser.parse_args()

    if not os.path.exists(args.poc):
        print(f"Error: PoC file not found: {args.poc}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.dump_dir):
        print(f"Error: Dump directory not found: {args.dump_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"PoC: {args.poc}")
    print(f"Dump: {args.dump_dir}")

    comparator = BatchComparator(args.poc, args.dump_dir)
    results = comparator.run(max_programs=args.max_programs)

    # Print summary
    print(f"\n{'=' * 80}")
    print("PoC vs Fuzzer Gap Analysis")
    print(f"{'=' * 80}")
    print(f"Total programs analyzed: {results.get('total_programs', 0)}")
    print(f"PoC syscalls: {results.get('poc_syscalls', [])}")

    print(f"\n--- Score Distributions ---")
    for metric in ['syscall_overlap_ratio', 'order_preservation', 'resource_wiring', 'socket_setup']:
        if metric in results:
            s = results[metric]
            print(f"  {metric}: mean={s['mean']:.3f} max={s['max']:.3f} >50%={s['above_50pct']:.1%}")

    print(f"\n--- Critical Syscall Presence ---")
    if 'critical_syscalls' in results:
        for crit, info in results['critical_syscalls'].items():
            print(f"  {crit}: {info['count']}/{results['total_programs']} ({info['ratio']:.1%})")

    print(f"\n--- Top 5 Closest Programs ---")
    for i, prog in enumerate(results.get('top_10_closest', [])[:5]):
        print(f"  {i+1}. {prog['program']}: overlap={prog['syscall_overlap_ratio']:.3f} "
              f"order={prog['order_preservation']:.3f} len={prog['program_length']}")

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nFull report written to: {args.output}")


if __name__ == '__main__':
    main()
