#!/usr/bin/env python3
"""
Batch callgraph extraction using KallGraph.

This script extracts callgraph.csv from KallGraph for multiple kernel cases.
It reads a CSV configuration file and processes each case to generate call graphs.

Usage:
    python extract_cg.py \
        --config /artifact/assets/compile_case_1-60.csv \
        --workdir /artifact/assets/cases_irgen \
        --output-dir /opt/syzpilot/analyzer/KallGraph-full/output \
        --opt O0 \
        --thread-num 80
"""

import argparse
import csv
import glob
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple, Optional


class ExtractionResult:
    """Result of a single callgraph extraction."""

    def __init__(self, case_name: str, success: bool, callgraph_path: str = "", message: str = ""):
        self.case_name = case_name
        self.success = success
        self.callgraph_path = callgraph_path
        self.message = message

    def __repr__(self):
        status = "SUCCESS" if self.success else "FAILED"
        return f"<{case_name}: {status}>"


def find_bc_list(workdir: Path, case_name: str, opt_level: str) -> Optional[Path]:
    """Find bc.list file for a given case."""
    bc_list = workdir / case_name / f"bc.list.{opt_level}"
    if bc_list.exists():
        return bc_list
    return None


def find_kallgraph_binary() -> Optional[str]:
    """Find the KallGraph binary."""
    # Check common locations
    candidates = [
        "/opt/syzpilot/analyzer/KallGraph-full/build/bin/KallGraph",
        "/opt/syzpilot/analyzer/KallGraph-fork/build/bin/KallGraph",
    ]
    for candidate in candidates:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    # Try PATH
    result = subprocess.run(["which", "KallGraph"], capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout.strip()
    return None


def extract_single_callgraph(
    case_name: str,
    bc_list: Path,
    output_dir: Path,
    kallgraph_bin: str,
    thread_num: int = 80
) -> ExtractionResult:
    """
    Extract callgraph for a single kernel case.

    Args:
        case_name: Name of the kernel case
        bc_list: Path to bc.list file
        output_dir: Output directory for callgraph
        kallgraph_bin: Path to KallGraph binary
        thread_num: Number of threads for KallGraph

    Returns:
        ExtractionResult object
    """
    case_output_dir = output_dir / case_name
    case_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{case_name}] Running KallGraph with {thread_num} threads...")
    print(f"[{case_name}] bc.list: {bc_list}")
    print(f"[{case_name}] output: {case_output_dir}")

    cmd = [
        kallgraph_bin,
        f"@{bc_list}",
        f"-OutputDir={case_output_dir}",
        f"-ThreadNum={thread_num}"
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=14400  # 4 hour timeout per case (better not set timeout here, or the task will not finish if the cpu/job resource is not enough)
        )

        if result.returncode != 0:
            error_msg = f"KallGraph failed (exit code: {result.returncode})"
            print(f"[{case_name}] ERROR: {error_msg}")
            if result.stderr:
                # Print last few lines of stderr
                stderr_lines = result.stderr.strip().split('\n')
                for line in stderr_lines[-5:]:
                    print(f"[{case_name}]   {line}")
            return ExtractionResult(case_name, False, message=error_msg)

        # Find the generated callgraph.csv
        callgraph_files = sorted(case_output_dir.glob("*/callgraph.csv"))
        if not callgraph_files:
            error_msg = "No callgraph.csv found in output"
            print(f"[{case_name}] ERROR: {error_msg}")
            return ExtractionResult(case_name, False, message=error_msg)

        # Use the most recent one
        callgraph_path = callgraph_files[-1]
        print(f"[{case_name}] SUCCESS: {callgraph_path}")
        return ExtractionResult(case_name, True, str(callgraph_path))

    except subprocess.TimeoutExpired:
        error_msg = "KallGraph timed out (1 hour limit)"
        print(f"[{case_name}] ERROR: {error_msg}")
        return ExtractionResult(case_name, False, message=error_msg)
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(f"[{case_name}] ERROR: {error_msg}")
        return ExtractionResult(case_name, False, message=error_msg)


def extract_callgraphs_batch(
    config_file: str,
    workdir: str,
    output_dir: str,
    opt_level: str = "O0",
    thread_num: int = 80,
    kallgraph_bin: str = None
) -> List[ExtractionResult]:
    """
    Batch extract callgraphs for multiple kernel cases (Serial).

    Args:
        config_file: Path to CSV configuration file
        workdir: Working directory containing case directories with bc.list files
        output_dir: Output directory for callgraph results
        opt_level: Optimization level (O0 or O1)
        thread_num: Number of threads for KallGraph
        kallgraph_bin: Path to KallGraph binary (auto-detected if None)

    Returns:
        List of ExtractionResult objects
    """
    workdir_path = Path(workdir).resolve()
    output_dir_path = Path(output_dir).resolve()
    config_file_path = Path(config_file).resolve()

    # Find KallGraph binary
    if kallgraph_bin is None:
        kallgraph_bin = find_kallgraph_binary()
        if kallgraph_bin is None:
            print("ERROR: KallGraph binary not found")
            return []
    print(f"KallGraph binary: {kallgraph_bin}")

    # Validate inputs
    if not config_file_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    if not workdir_path.exists():
        raise FileNotFoundError(f"Workdir not found: {workdir}")

    # Create output directory
    output_dir_path.mkdir(parents=True, exist_ok=True)

    print(f"Configuration file: {config_file_path}")
    print(f"Working directory: {workdir_path}")
    print(f"Output directory: {output_dir_path}")
    print(f"Optimization level: {opt_level}")
    print(f"Thread count: {thread_num}")
    print("=" * 80)

    # Parse CSV configuration file
    targets = []
    with open(config_file_path, 'r') as f:
        reader = csv.reader(f)
        for row_num, row in enumerate(reader, start=1):
            if not row or all(cell.strip() == '' for cell in row):
                continue
            if row[0].strip().startswith('#'):
                continue
            if len(row) < 1:
                continue
            case_name = row[0].strip()
            targets.append(case_name)

    if not targets:
        print("ERROR: No valid targets found in configuration file")
        return []

    print(f"\nFound {len(targets)} target(s) to process:")
    for case_name in targets:
        bc_list = find_bc_list(workdir_path, case_name, opt_level)
        status = "bc.list found" if bc_list else "bc.list MISSING"
        print(f"  - {case_name}: {status}")
    print("=" * 80 + "\n")

    # Process each case
    results = []
    for idx, case_name in enumerate(targets, start=1):
        print(f"\n[{idx}/{len(targets)}] Processing {case_name}...")

        # Find bc.list
        bc_list = find_bc_list(workdir_path, case_name, opt_level)
        if bc_list is None:
            error_msg = f"bc.list.{opt_level} not found"
            print(f"[{case_name}] SKIP: {error_msg}")
            results.append(ExtractionResult(case_name, False, message=error_msg))
            continue

        result = extract_single_callgraph(
            case_name=case_name,
            bc_list=bc_list,
            output_dir=output_dir_path,
            kallgraph_bin=kallgraph_bin,
            thread_num=thread_num
        )
        results.append(result)

    # Print summary
    print("\n" + "=" * 80)
    print("EXTRACTION SUMMARY")
    print("=" * 80)

    success_count = sum(1 for r in results if r.success)
    failed_count = len(results) - success_count

    print(f"Total: {len(results)}")
    print(f"Success: {success_count}")
    print(f"Failed: {failed_count}")

    if success_count > 0:
        print(f"\nSuccessful extractions:")
        for result in results:
            if result.success:
                print(f"  - {result.case_name}: {result.callgraph_path}")

    if failed_count > 0:
        print(f"\nFailed cases:")
        for result in results:
            if not result.success:
                print(f"  - {result.case_name}: {result.message}")

    return results


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Batch extract callgraphs using KallGraph.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract callgraphs for all cases with O0 optimization
  python extract_cg.py \\
      --config /artifact/assets/compile_case_1-60.csv \\
      --workdir /artifact/assets/cases_irgen \\
      --output-dir /opt/syzpilot/analyzer/KallGraph-full/output \\
      --opt O0

  # Extract with O1 optimization and custom thread count
  python extract_cg.py \\
      --config compile_case_1-60.csv \\
      --workdir cases_irgen \\
      --output-dir output \\
      --opt O1 \\
      --thread-num 40
        """
    )

    parser.add_argument(
        '--config',
        required=True,
        help='CSV configuration file with case names'
    )

    parser.add_argument(
        '--workdir',
        required=True,
        help='Working directory containing case directories with bc.list files'
    )

    parser.add_argument(
        '--output-dir',
        required=True,
        help='Output directory for callgraph results'
    )

    parser.add_argument(
        '--opt',
        choices=['O0', 'O1'],
        default='O0',
        help='Optimization level (default: O0)'
    )

    parser.add_argument(
        '--thread-num',
        type=int,
        default=80,
        help='Number of threads for KallGraph (default: 80)'
    )

    parser.add_argument(
        '--kallgraph-bin',
        default=None,
        help='Path to KallGraph binary (auto-detected if not specified)'
    )

    args = parser.parse_args()

    try:
        results = extract_callgraphs_batch(
            config_file=args.config,
            workdir=args.workdir,
            output_dir=args.output_dir,
            opt_level=args.opt,
            thread_num=args.thread_num,
            kallgraph_bin=args.kallgraph_bin
        )

        failed_count = sum(1 for r in results if not r.success)
        sys.exit(1 if failed_count > 0 else 0)

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
