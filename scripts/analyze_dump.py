#!/usr/bin/env python3
"""
Analyze dump directory to check if syzkaller dump functionality is working properly.
This script checks programs and coverage files for various issues and provides a summary.
"""

import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Set, Tuple

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Warning: tqdm not available. Install with 'pip install tqdm' for progress bars.")


class DumpAnalyzer:
    def __init__(self, dump_dir: str, max_examples: int = 5):
        self.dump_dir = Path(dump_dir)
        self.programs_dir = self.dump_dir / "programs"
        self.coverage_dir = self.dump_dir / "coverage"
        self.max_examples = max_examples

        # Analysis results
        self.programs_without_coverage: List[str] = []
        self.empty_program_files: List[str] = []
        self.programs_with_empty_coverage: List[str] = []
        self.programs_with_single_empty_coverage: List[str] = []

        # Statistics
        self.total_programs = 0
        self.total_coverage_files = 0
        self.programs_with_coverage = 0

    def analyze(self) -> Dict:
        """Perform complete analysis of dump directory."""
        print(f"Analyzing dump directory: {self.dump_dir}")
        print("=" * 60)

        # Validate directories
        if not self._validate_directories():
            return {}

        # Get all files
        program_files = self._get_files(self.programs_dir, "programs")
        coverage_files = self._get_files(self.coverage_dir, "coverage")

        self.total_programs = len(program_files)
        self.total_coverage_files = len(coverage_files)

        print(f"Found {self.total_programs} program files and {self.total_coverage_files} coverage files")
        print()

        # Perform all checks
        self._check_programs_without_coverage(program_files, coverage_files)
        self._check_empty_program_files(program_files)
        self._check_programs_with_empty_coverage(program_files, coverage_files)

        return self._generate_summary()

    def _validate_directories(self) -> bool:
        """Validate that required directories exist."""
        dirs_to_check = [
            (self.dump_dir, "Dump directory"),
            (self.programs_dir, "Programs directory"),
            (self.coverage_dir, "Coverage directory")
        ]

        for dir_path, dir_name in dirs_to_check:
            if not dir_path.exists():
                print(f"Error: {dir_name} '{dir_path}' does not exist!")
                return False
        return True

    def _get_files(self, directory: Path, desc: str) -> List[Path]:
        """Get all files from a directory with optional progress bar."""
        files = []
        print(f"Scanning {desc} directory...")

        items = list(directory.iterdir())
        iterator = tqdm(items, desc=f"Scanning {desc}") if TQDM_AVAILABLE else items

        for file_path in iterator:
            if file_path.is_file():
                files.append(file_path)

        return files

    def _iterate_with_progress(self, items: List, desc: str):
        """Helper to iterate with optional progress bar."""
        return tqdm(items, desc=desc) if TQDM_AVAILABLE else items

    def _check_programs_without_coverage(self, program_files: List[Path], coverage_files: List[Path]):
        """Check if each program has at least one corresponding coverage file."""
        print("1. Checking programs without coverage...")

        # Extract coverage signatures
        coverage_sigs = set()
        for coverage_file in self._iterate_with_progress(coverage_files, "Processing coverage files"):
            filename = coverage_file.name
            if '-' in filename:
                sig = filename.split('-')[0]
                coverage_sigs.add(sig)

        # Check programs
        for program_file in self._iterate_with_progress(program_files, "Checking program coverage"):
            program_sig = program_file.name
            if program_sig not in coverage_sigs:
                self.programs_without_coverage.append(program_sig)

        self.programs_with_coverage = self.total_programs - len(self.programs_without_coverage)
        self._print_results("programs without coverage", self.programs_without_coverage, "All programs have coverage files ✓")

    def _check_empty_program_files(self, program_files: List[Path]):
        """Check for empty program files."""
        print("2. Checking for empty program files...")

        for program_file in self._iterate_with_progress(program_files, "Checking empty programs"):
            if program_file.stat().st_size == 0:
                self.empty_program_files.append(program_file.name)

        self._print_results("empty program files", self.empty_program_files, "No empty program files found ✓")

    def _check_programs_with_empty_coverage(self, program_files: List[Path], coverage_files: List[Path]):
        """Check if all coverage files for a program are empty."""
        print("3. Checking programs with empty coverage...")

        # Group coverage files by program signature
        coverage_by_program = defaultdict(list)
        for coverage_file in self._iterate_with_progress(coverage_files, "Grouping coverage files"):
            filename = coverage_file.name
            if '-' in filename:
                sig = filename.split('-')[0]
                coverage_by_program[sig].append(coverage_file)

        # Check each program's coverage files
        for program_file in self._iterate_with_progress(program_files, "Checking empty coverage"):
            program_sig = program_file.name
            if program_sig in coverage_by_program:
                coverage_files_for_program = coverage_by_program[program_sig]

                # Check if all coverage files are empty
                all_empty = all(cov.stat().st_size == 0 for cov in coverage_files_for_program)

                if all_empty:
                    self.programs_with_empty_coverage.append(program_sig)

                    # Check if it's a single empty coverage file (sig-0)
                    if (len(coverage_files_for_program) == 1 and
                        coverage_files_for_program[0].name.endswith('-0')):
                        self.programs_with_single_empty_coverage.append(program_sig)

        self._print_results("programs with empty coverage", self.programs_with_empty_coverage, "No programs with empty coverage found ✓")

        # Print detailed statistics for single empty coverage
        if self.programs_with_single_empty_coverage:
            print(f"   Programs with single empty coverage (sig-0): {len(self.programs_with_single_empty_coverage)}")
            for i, sig in enumerate(self.programs_with_single_empty_coverage):
                if i >= self.max_examples:
                    remaining = len(self.programs_with_single_empty_coverage) - self.max_examples
                    print(f"     ... and {remaining} more")
                    break
                print(f"     - {sig}")
        print()

    def _print_results(self, issue_type: str, issues: List[str], success_message: str):
        """Helper to print results with consistent formatting."""
        if issues:
            print(f"   Found {len(issues)} {issue_type}:")
            for i, item in enumerate(issues):
                if i >= self.max_examples:
                    remaining = len(issues) - self.max_examples
                    print(f"     ... and {remaining} more")
                    break
                print(f"     - {item}")
        else:
            print(f"   {success_message}")
        print()

    def _generate_summary(self) -> Dict:
        """Generate and print summary statistics."""
        print("SUMMARY STATISTICS")
        print("=" * 60)

        summary = {
            'total_programs': self.total_programs,
            'total_coverage_files': self.total_coverage_files,
            'programs_with_coverage': self.programs_with_coverage,
            'programs_without_coverage': len(self.programs_without_coverage),
            'empty_program_files': len(self.empty_program_files),
            'programs_with_empty_coverage': len(self.programs_with_empty_coverage),
            'programs_with_single_empty_coverage': len(self.programs_with_single_empty_coverage)
        }

        # Calculate percentages
        if self.total_programs > 0:
            coverage_percentage = (self.programs_with_coverage / self.total_programs) * 100
            empty_program_percentage = (len(self.empty_program_files) / self.total_programs) * 100
            empty_coverage_percentage = (len(self.programs_with_empty_coverage) / self.total_programs) * 100
            single_empty_percentage = (len(self.programs_with_single_empty_coverage) / self.total_programs) * 100
        else:
            coverage_percentage = empty_program_percentage = empty_coverage_percentage = single_empty_percentage = 0

        # Print statistics
        stats = [
            f"Total Programs: {self.total_programs}",
            f"Total Coverage Files: {self.total_coverage_files}",
            f"Programs with Coverage: {self.programs_with_coverage} ({coverage_percentage:.1f}%)",
            f"Programs without Coverage: {len(self.programs_without_coverage)} ({100-coverage_percentage:.1f}%)",
            f"Empty Program Files: {len(self.empty_program_files)} ({empty_program_percentage:.1f}%)",
            f"Programs with Empty Coverage: {len(self.programs_with_empty_coverage)} ({empty_coverage_percentage:.1f}%)",
            f"  - Single Empty Coverage (sig-0): {len(self.programs_with_single_empty_coverage)} ({single_empty_percentage:.1f}%)"
        ]

        for stat in stats:
            print(stat)
        print()

        # Overall health assessment
        issues = sum([
            bool(self.programs_without_coverage),
            bool(self.empty_program_files),
            bool(self.programs_with_empty_coverage)
        ])

        status_messages = {
            0: ("DUMP FUNCTIONALITY STATUS: HEALTHY ✓", "All programs have coverage, no empty files found."),
            1: ("DUMP FUNCTIONALITY STATUS: MINOR ISSUES", "Some issues detected but overall functionality appears working."),
            2: ("DUMP FUNCTIONALITY STATUS: PROBLEMS DETECTED", "Multiple issues found. Please investigate the dump functionality."),
            3: ("DUMP FUNCTIONALITY STATUS: CRITICAL ISSUES", "Multiple critical issues found. Dump functionality needs immediate attention.")
        }

        status, message = status_messages.get(issues, status_messages[3])
        print(status)
        print(f"   {message}")

        return summary


def main():
    """Main function to run the dump analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze syzkaller dump directory to check if dump functionality is working properly.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 analyze_dump.py /path/to/dump_dir
  python3 analyze_dump.py /path/to/dump_dir --max-examples 10
  python3 analyze_dump.py /path/to/dump_dir --max-examples 3
        """
    )

    parser.add_argument(
        "dump_dir",
        help="Path to the dump directory containing programs/ and coverage/ subdirectories"
    )

    parser.add_argument(
        "--max-examples",
        type=int,
        default=5,
        help="Maximum number of examples to show for long lists (default: 5)"
    )

    args = parser.parse_args()

    try:
        analyzer = DumpAnalyzer(args.dump_dir, args.max_examples)
        summary = analyzer.analyze()

        if not summary:
            sys.exit(1)

    except Exception as e:
        print(f"Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
