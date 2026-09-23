#!/usr/bin/env python3
"""
Batch kernel compilation script.

This script compiles multiple Linux kernel versions based on a CSV configuration file.
It can be used as a standalone command-line tool or imported as a module.
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple


class KernelCompilationResult:
    """Represents the result of a kernel compilation."""

    def __init__(self, case_name: str, success: bool, message: str = ""):
        self.case_name = case_name
        self.success = success
        self.message = message

    def __repr__(self):
        status = "SUCCESS" if self.success else "FAILED"
        return f"<{self.case_name}: {status}>"


def run_command(
    cmd: List[str],
    cwd: str = None,
    capture_output: bool = False,
) -> Tuple[int, str, str]:
    """
    Run a shell command and return exit code, stdout, stderr.

    Args:
        cmd: Command and arguments as a list
        cwd: Working directory
        capture_output: Whether to capture stdout/stderr

    Returns:
        Tuple of (exit_code, stdout, stderr)
    """
    try:
        if capture_output:
            result = subprocess.run(
                cmd,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            return result.returncode, result.stdout, result.stderr
        else:
            result = subprocess.run(cmd, cwd=cwd, check=False)
            return result.returncode, "", ""
    except Exception as e:
        return 1, "", str(e)


def git_commit_exists(repo_path: Path, commit_hash: str) -> bool:
    """Return True if commit_hash resolves to a commit inside repo_path."""
    returncode, _, _ = run_command(
        ["git", "rev-parse", "--verify", f"{commit_hash}^{{commit}}"],
        cwd=str(repo_path),
        capture_output=True,
    )
    return returncode == 0


def select_source_repo(
    commit_hash: str,
    linux_git_master: Path,
    linux_stable_git_master: Optional[Path] = None,
) -> Tuple[Path, str]:
    """
    Select the source repository that contains commit_hash.

    The primary repository is tried first. If the commit is not found there and a
    stable repository is available, fall back to the stable repository.
    """
    if git_commit_exists(linux_git_master, commit_hash):
        return linux_git_master, "linux-git-master"

    if linux_stable_git_master and git_commit_exists(linux_stable_git_master, commit_hash):
        return linux_stable_git_master, "linux-stable-git-master"

    searched = [str(linux_git_master)]
    if linux_stable_git_master:
        searched.append(str(linux_stable_git_master))
    raise ValueError(
        f"Commit {commit_hash} was not found in any source repository: {', '.join(searched)}"
    )


def compile_single_kernel(
    case_name: str,
    commit_hash: str,
    config_path: str,
    workdir: Path,
    linux_git_master: Path,
    jobs: int,
    dry_run: bool = False,
    rebuild: bool = False,
    linux_stable_git_master: Optional[Path] = None,
) -> KernelCompilationResult:
    """
    Compile a single kernel version.

    Args:
        case_name: Name of the kernel case (e.g., "case_1")
        commit_hash: Git commit hash to checkout
        config_path: Path to the kernel .config file
        workdir: Working directory where kernels will be compiled
        linux_git_master: Path to the linux-git-master repository
        linux_stable_git_master: Optional path to linux-stable-git-master.
            If provided and commit_hash is not found in linux_git_master,
            the stable repository will be used as a fallback source.
        jobs: Number of parallel jobs for make
        dry_run: If True, skip actual compilation (for testing)
        rebuild: If True, rebuild existing case directory (make clean + rebuild)

    Returns:
        KernelCompilationResult object
    """
    print(f"[{case_name}] Starting compilation...")

    # Prepare paths
    case_dir = workdir / case_name
    log_file = workdir / f"{case_name}_compilation.log"

    try:
        if rebuild and case_dir.exists():
            # Rebuild existing directory: just run make clean
            print(f"[{case_name}] Reusing existing directory, running make clean...")
            returncode, stdout, stderr = run_command(
                ["make", "clean"],
                cwd=str(case_dir),
                capture_output=True
            )
            if returncode != 0:
                print(f"[{case_name}] WARNING: make clean returned non-zero: {stderr}")
        else:
            try:
                source_repo, source_repo_name = select_source_repo(
                    commit_hash,
                    linux_git_master,
                    linux_stable_git_master,
                )
            except ValueError as e:
                error_msg = str(e)
                print(f"[{case_name}] ERROR: {error_msg}")
                with open(log_file, 'w') as f:
                    f.write(error_msg + "\n")
                return KernelCompilationResult(case_name, False, error_msg)

            print(f"[{case_name}] Using source repository: {source_repo_name} ({source_repo})")
            print(f"[{case_name}] Copying repository to {case_dir}...")
            if case_dir.exists():
                print(f"[{case_name}] Removing existing directory...")
                shutil.rmtree(case_dir)

            shutil.copytree(source_repo, case_dir, symlinks=True)

            # Step 2: Clean working directory and checkout the specified commit
            print(f"[{case_name}] Cleaning git working directory...")
            returncode, stdout, stderr = run_command(
                ["git", "reset", "--hard"],
                cwd=str(case_dir),
                capture_output=True
            )

            if returncode != 0:
                error_msg = f"Git reset failed: {stderr}"
                print(f"[{case_name}] WARNING: {error_msg}")

            # Clean untracked files
            run_command(
                ["git", "clean", "-fdx"],
                cwd=str(case_dir),
                capture_output=True
            )

            print(f"[{case_name}] Checking out commit {commit_hash}...")
            returncode, stdout, stderr = run_command(
                ["git", "checkout", "-f", commit_hash],
                cwd=str(case_dir),
                capture_output=True
            )

            if returncode != 0:
                error_msg = f"Git checkout failed: {stderr}"
                print(f"[{case_name}] ERROR: {error_msg}")
                with open(log_file, 'w') as f:
                    f.write(f"Git checkout failed for commit {commit_hash}\n")
                    f.write(f"stdout: {stdout}\n")
                    f.write(f"stderr: {stderr}\n")
                return KernelCompilationResult(case_name, False, error_msg)

            # Step 3: Remove .git directory to save space
            print(f"[{case_name}] Removing .git directory to save space...")
            git_dir = case_dir / ".git"
            if git_dir.exists():
                shutil.rmtree(git_dir)

        # Step 4: Copy .config file
        print(f"[{case_name}] Copying .config file from {config_path}...")
        config_src = Path(config_path)
        if not config_src.exists():
            error_msg = f"Config file not found: {config_path}"
            print(f"[{case_name}] ERROR: {error_msg}")
            with open(log_file, 'w') as f:
                f.write(f"{error_msg}\n")
            return KernelCompilationResult(case_name, False, error_msg)

        config_dst = case_dir / ".config"
        shutil.copy2(config_src, config_dst)

        # Step 4.5: Comment some CONFIGs in .config
        # E.g., if CONFIG_DEBUG_INFO_BTF=y exists, it should be commented
        cmd = ["sed", "-i", "s/^CONFIG_DEBUG_INFO_BTF=y/# CONFIG_DEBUG_INFO_BTF is not set/", str(config_dst)]
        returncode, stdout, stderr = run_command(cmd, capture_output=True)
        if returncode != 0:
            error_msg = f"sed command failed: {stderr}"
            print(f"[{case_name}] WARNING: {error_msg}")
            with open(log_file, 'w') as f:
                f.write(f"sed command failed: {stderr}\n")
            # no need to return, continue

        # Step 5: Run make olddefconfig
        print(f"[{case_name}] Running make olddefconfig...")
        returncode, stdout, stderr = run_command(
            ["make", "CC=gcc", "olddefconfig"],
            cwd=str(case_dir),
            capture_output=True,
        )

        if returncode != 0:
            error_msg = f"make olddefconfig failed"
            print(f"[{case_name}] ERROR: {error_msg}")
            with open(log_file, 'w') as f:
                f.write(f"make olddefconfig failed\n")
                f.write(f"stdout: {stdout}\n")
                f.write(f"stderr: {stderr}\n")
            return KernelCompilationResult(case_name, False, error_msg)

        # Step 6: Compile the kernel
        if dry_run:
            print(f"[{case_name}] DRY RUN: Skipping actual compilation (make CC=gcc -j{jobs})")
            return KernelCompilationResult(case_name, True, "Dry run completed")

        print(f"[{case_name}] Compiling kernel with -j{jobs}...")
        with open(log_file, 'w') as f:
            f.write(f"Compilation log for {case_name} (commit: {commit_hash})\n")
            f.write("=" * 80 + "\n\n")
            f.flush()

            # Run make and redirect output to log file
            result = subprocess.run(
                ["make", "CC=gcc", f"-j{jobs}"],
                cwd=str(case_dir),
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                check=False
            )

        if result.returncode != 0:
            error_msg = f"Kernel compilation failed (exit code: {result.returncode})"
            print(f"[{case_name}] ERROR: {error_msg}")
            print(f"[{case_name}] See log file: {log_file}")
            return KernelCompilationResult(case_name, False, error_msg)

        # Success - remove the log file as it's not needed
        if log_file.exists():
            log_file.unlink()

        print(f"[{case_name}] Compilation completed successfully!")
        return KernelCompilationResult(case_name, True, "Compilation successful")

    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(f"[{case_name}] ERROR: {error_msg}")
        try:
            with open(log_file, 'a') as f:
                f.write(f"\nUnexpected error: {str(e)}\n")
        except:
            pass
        return KernelCompilationResult(case_name, False, error_msg)


def compile_single_kernel_irgen(
    case_name: str,
    commit_hash: str,
    config_path: str,
    workdir: Path,
    linux_git_master: Path,
    irgen_script: str,
    opt_level: str = "O0",
    jobs: int = 80,
    dry_run: bool = False,
    rebuild: bool = False,
    linux_stable_git_master: Optional[Path] = None,
) -> KernelCompilationResult:
    """
    Compile a single kernel with IRDumper for IR generation.

    This function:
    1. Copies the kernel source and checks out the specified commit
    2. Copies the .config file
    3. Calls kallgraph_irgen.sh which handles the full build (IRDumper injection + make)
    4. Generates bc.list from the resulting .bc files

    Args:
        case_name: Name of the kernel case (e.g., "case_1")
        commit_hash: Git commit hash to checkout
        config_path: Path to the kernel .config file
        workdir: Working directory where kernels will be compiled
        linux_git_master: Path to the linux-git-master repository
        linux_stable_git_master: Optional path to linux-stable-git-master.
            If provided and commit_hash is not found in linux_git_master,
            the stable repository will be used as a fallback source.
        irgen_script: Path to kallgraph_irgen.sh
        opt_level: Optimization level (O0 or O1)
        jobs: Number of parallel jobs for make
        dry_run: If True, skip actual compilation (for testing)
        rebuild: If True, rebuild existing case directory

    Returns:
        KernelCompilationResult object
    """
    print(f"[{case_name}] Starting IR generation (opt={opt_level})...")

    case_dir = workdir / case_name
    log_file = workdir / f"{case_name}_irgen.log"

    try:
        if rebuild and case_dir.exists():
            print(f"[{case_name}] Reusing existing directory, cleaning...")
            run_command(["make", "clean"], cwd=str(case_dir), capture_output=True)
        else:
            try:
                source_repo, source_repo_name = select_source_repo(
                    commit_hash,
                    linux_git_master,
                    linux_stable_git_master,
                )
            except ValueError as e:
                error_msg = str(e)
                print(f"[{case_name}] ERROR: {error_msg}")
                return KernelCompilationResult(case_name, False, error_msg)

            # Steps 1-4: Same as normal compilation
            print(f"[{case_name}] Using source repository: {source_repo_name} ({source_repo})")
            print(f"[{case_name}] Copying repository to {case_dir}...")
            if case_dir.exists():
                shutil.rmtree(case_dir)
            shutil.copytree(source_repo, case_dir, symlinks=True)

            # Git checkout
            print(f"[{case_name}] Checking out commit {commit_hash}...")
            run_command(["git", "reset", "--hard"], cwd=str(case_dir), capture_output=True)
            run_command(["git", "clean", "-fdx"], cwd=str(case_dir), capture_output=True)
            returncode, stdout, stderr = run_command(
                ["git", "checkout", "-f", commit_hash],
                cwd=str(case_dir), capture_output=True
            )
            if returncode != 0:
                error_msg = f"Git checkout failed: {stderr}"
                print(f"[{case_name}] ERROR: {error_msg}")
                return KernelCompilationResult(case_name, False, error_msg)

            # Remove .git
            git_dir = case_dir / ".git"
            if git_dir.exists():
                shutil.rmtree(git_dir)

        # Copy .config
        config_src = Path(config_path)
        if not config_src.exists():
            error_msg = f"Config file not found: {config_path}"
            print(f"[{case_name}] ERROR: {error_msg}")
            return KernelCompilationResult(case_name, False, error_msg)

        config_dst = case_dir / ".config"
        shutil.copy2(config_src, config_dst)

        # Patch CONFIG_DEBUG_INFO_BTF
        run_command(
            ["sed", "-i", "s/^CONFIG_DEBUG_INFO_BTF=y/# CONFIG_DEBUG_INFO_BTF is not set/",
             str(config_dst)],
            capture_output=True
        )

        if dry_run:
            print(f"[{case_name}] DRY RUN: Skipping IR generation")
            return KernelCompilationResult(case_name, True, "Dry run completed")

        # Step 5: Call kallgraph_irgen.sh
        print(f"[{case_name}] Running kallgraph_irgen.sh (opt={opt_level}, jobs={jobs})...")
        irgen_cmd = [
            irgen_script,
            "--kernel-src", str(case_dir),
            "--opt", opt_level,
            "--jobs", str(jobs),
            "--config", "olddefconfig"
        ]

        with open(log_file, 'w') as f:
            f.write(f"IR generation log for {case_name} (commit: {commit_hash}, opt: {opt_level})\n")
            f.write("=" * 80 + "\n\n")
            f.flush()

            result = subprocess.run(
                irgen_cmd,
                cwd=str(case_dir),
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                check=False
            )

        if result.returncode != 0:
            error_msg = f"IR generation failed (exit code: {result.returncode})"
            print(f"[{case_name}] ERROR: {error_msg}")
            print(f"[{case_name}] See log file: {log_file}")
            return KernelCompilationResult(case_name, False, error_msg)

        # Step 6: Generate bc.list
        bc_list_file = case_dir / f"bc.list.{opt_level}"
        print(f"[{case_name}] Generating {bc_list_file}...")
        find_result = subprocess.run(
            ["find", str(case_dir), "-name", "*.bc", "!", "-name", "*timeconst.bc"],
            capture_output=True, text=True, check=False
        )
        bc_files = sorted(find_result.stdout.strip().split('\n'))
        bc_files = [f for f in bc_files if f]  # Remove empty strings

        with open(bc_list_file, 'w') as f:
            for bc in bc_files:
                f.write(bc + '\n')

        print(f"[{case_name}] Found {len(bc_files)} .bc files")

        # Success - remove log file
        if log_file.exists():
            log_file.unlink()

        print(f"[{case_name}] IR generation completed successfully!")
        return KernelCompilationResult(case_name, True, f"Generated {len(bc_files)} .bc files")

    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(f"[{case_name}] ERROR: {error_msg}")
        try:
            with open(log_file, 'a') as f:
                f.write(f"\nUnexpected error: {str(e)}\n")
        except:
            pass
        return KernelCompilationResult(case_name, False, error_msg)


def compile_kernels_batch(
    config_file: str,
    workdir: str,
    linux_git_master: str,
    jobs: int,
    dry_run: bool = False,
    rebuild: bool = False,
    irgen: bool = False,
    irgen_script: str = None,
    irgen_opt: str = "O0",
    irgen_jobs: int = 80,
    linux_stable_git_master: Optional[str] = None,
) -> List[KernelCompilationResult]:
    """
    Batch compile multiple kernels based on a CSV configuration file.

    CSV format: case_name,commit_hash,config_path[,optional_columns...]
    Only the first three columns are parsed.

    Args:
        config_file: Path to CSV configuration file
        workdir: Working directory for compilation
        linux_git_master: Path to linux-git-master repository
        linux_stable_git_master: Optional path to linux-stable-git-master.
            If omitted, the script will try a sibling directory named
            linux-stable-git-master when a commit is not found in
            linux-git-master.
        jobs: Number of parallel jobs for make
        dry_run: If True, skip actual compilation (for testing)
        rebuild: If True, rebuild existing case directories (make clean + rebuild)

    Returns:
        List of KernelCompilationResult objects
    """
    # Convert paths to Path objects
    workdir_path = Path(workdir).resolve()
    linux_git_path = Path(linux_git_master).resolve()
    linux_stable_git_path = None
    config_file_path = Path(config_file).resolve()

    # Validate inputs
    if not config_file_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    if not linux_git_path.exists():
        raise FileNotFoundError(f"Linux git repository not found: {linux_git_master}")

    if not linux_git_path.is_dir():
        raise NotADirectoryError(f"Linux git path is not a directory: {linux_git_master}")

    if linux_stable_git_master:
        candidate_path = Path(linux_stable_git_master).resolve()
    else:
        candidate_path = linux_git_path.parent / "linux-stable-git-master"

    if candidate_path.exists():
        if not candidate_path.is_dir():
            raise NotADirectoryError(
                f"Linux stable git path is not a directory: {candidate_path}"
            )
        linux_stable_git_path = candidate_path

    # Create workdir if it doesn't exist
    workdir_path.mkdir(parents=True, exist_ok=True)

    print(f"Configuration file: {config_file_path}")
    print(f"Working directory: {workdir_path}")
    print(f"Linux git master: {linux_git_path}")
    if linux_stable_git_path:
        print(f"Linux stable git master: {linux_stable_git_path}")
    else:
        print("Linux stable git master: <not available>")
    print(f"Jobs: {jobs}")
    print(f"Dry run: {dry_run}")
    print(f"Rebuild: {rebuild}")
    print("=" * 80)

    # Parse CSV configuration file
    targets = []
    with open(config_file_path, 'r') as f:
        reader = csv.reader(f)
        for row_num, row in enumerate(reader, start=1):
            # Skip empty lines
            if not row or all(cell.strip() == '' for cell in row):
                continue

            # Skip comment lines (starting with #)
            if row[0].strip().startswith('#'):
                continue

            # Parse only the first three columns
            if len(row) < 3:
                print(f"WARNING: Line {row_num} has less than 3 columns, skipping: {row}")
                continue

            case_name = row[0].strip()
            commit_hash = row[1].strip()
            config_path = row[2].strip()

            # Resolve config_path relative to config file's directory if it's relative
            if not Path(config_path).is_absolute():
                config_path = str(config_file_path.parent / config_path)

            targets.append((case_name, commit_hash, config_path))

    if not targets:
        print("ERROR: No valid targets found in configuration file")
        return []

    print(f"\nFound {len(targets)} target(s) to compile:")
    for case_name, commit_hash, config_path in targets:
        print(f"  - {case_name}: {commit_hash} ({config_path})")
    print("=" * 80 + "\n")

    # Compile each kernel
    results = []
    for idx, (case_name, commit_hash, config_path) in enumerate(targets, start=1):
        print(f"\n[{idx}/{len(targets)}] Processing {case_name}...")
        if irgen:
            result = compile_single_kernel_irgen(
                case_name=case_name,
                commit_hash=commit_hash,
                config_path=config_path,
                workdir=workdir_path,
                linux_git_master=linux_git_path,
                linux_stable_git_master=linux_stable_git_path,
                irgen_script=irgen_script,
                opt_level=irgen_opt,
                jobs=irgen_jobs,
                dry_run=dry_run,
                rebuild=rebuild
            )
        else:
            result = compile_single_kernel(
                case_name=case_name,
                commit_hash=commit_hash,
                config_path=config_path,
                workdir=workdir_path,
                linux_git_master=linux_git_path,
                linux_stable_git_master=linux_stable_git_path,
                jobs=jobs,
                dry_run=dry_run,
                rebuild=rebuild
            )
        results.append(result)
        print(f"[{idx}/{len(targets)}] {case_name}: {'SUCCESS' if result.success else 'FAILED'}")

    # Print summary
    print("\n" + "=" * 80)
    print("COMPILATION SUMMARY")
    print("=" * 80)

    success_count = sum(1 for r in results if r.success)
    failed_count = len(results) - success_count

    print(f"Total: {len(results)}")
    print(f"Success: {success_count}")
    print(f"Failed: {failed_count}")

    if failed_count > 0:
        print(f"\nFailed cases:")
        for result in results:
            if not result.success:
                print(f"  - {result.case_name}: {result.message}")

    return results


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Batch compile Linux kernels from a CSV configuration file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
CSV Configuration File Format:
  case_name,commit_hash,config_path[,optional_columns...]

  Example:
    case_1,761c6d7ec820,../benchmark/configs/case_1.config
    case_2,e8f71f89236e,../benchmark/configs/case_2.config

  - Only the first 3 columns are parsed
  - Lines starting with # are treated as comments
  - Empty lines are ignored
  - Relative paths in config_path are resolved relative to the CSV file location
        """
    )

    parser.add_argument(
        '--workdir',
        required=True,
        help='Working directory for kernel compilation'
    )

    parser.add_argument(
        '--linux-git-master',
        required=True,
        help='Path to linux-git-master repository'
    )

    parser.add_argument(
        '--linux-stable-git-master',
        default=None,
        help=(
            'Optional path to linux-stable-git-master repository. '
            'If omitted, the script will try a sibling directory named '
            'linux-stable-git-master when the commit is not found in '
            'linux-git-master.'
        )
    )

    parser.add_argument(
        '--config',
        required=True,
        help='CSV configuration file with targets (case_name,commit_hash,config_path,...)'
    )

    parser.add_argument(
        '-j', '--jobs',
        type=int,
        default=os.cpu_count() or 32,
        help='Number of parallel jobs for make (default: number of CPU cores)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Skip actual kernel compilation (for testing)'
    )

    parser.add_argument(
        '--rebuild',
        action='store_true',
        help='Rebuild existing case directories: run make clean + rebuild instead of copying from scratch'
    )

    # IR generation options
    parser.add_argument(
        '--irgen',
        action='store_true',
        help='Enable IR generation mode: compile with IRDumper to produce .bc files'
    )

    parser.add_argument(
        '--irgen-opt',
        choices=['O0', 'O1'],
        default='O0',
        help='Optimization level for IR generation (default: O0)'
    )

    parser.add_argument(
        '--irgen-script',
        default='/opt/syzpilot/analyzer/mlta/kallgraph_irgen.sh',
        help='Path to kallgraph_irgen.sh script'
    )

    parser.add_argument(
        '--irgen-jobs',
        type=int,
        default=80,
        help='Number of parallel jobs for IR generation (default: 80)'
    )

    args = parser.parse_args()

    try:
        results = compile_kernels_batch(
            config_file=args.config,
            workdir=args.workdir,
            linux_git_master=args.linux_git_master,
            linux_stable_git_master=args.linux_stable_git_master,
            jobs=args.jobs,
            dry_run=args.dry_run,
            rebuild=args.rebuild,
            irgen=args.irgen,
            irgen_script=args.irgen_script,
            irgen_opt=args.irgen_opt,
            irgen_jobs=args.irgen_jobs
        )

        # Exit with error code if any compilation failed
        failed_count = sum(1 for r in results if not r.success)
        sys.exit(1 if failed_count > 0 else 0)

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()


# python compile_kernel.py
#        --workdir /root/kernels/SyzPilot-experiments/
#        --linux-git-master /root/kernels/linux-git-master/
#        --config /path/to/compile_targets.csv
#        --jobs 32
