import os
import argparse
import shutil
import random
from time import time
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Set, Tuple
from multiprocessing import Pool, cpu_count, Manager
import functools


def read_cover(fn):
    cover = []
    with open(fn, 'r') as f:
        cover = [line.strip() for line in f.readlines()]
    return cover


def merge_covers(covers):
    merged_cover_set = set()
    for cover in covers:
        cover_set = set(cover)
        merged_cover_set |= cover_set
    return list(merged_cover_set)


class ProgDatasetBuilder:
    """src_dirs are supposed to be the programs dirs"""
    def __init__(self, src_dirs: List[str], dst_dir: str):
        self.src_dirs = [Path(src_dir) for src_dir in src_dirs]
        self.dst_dir = Path(dst_dir)

    def _get_progs(self, directory: Path, desc: str) -> List[Path]:
        """Get all prog files from a directory with optional progress bar."""
        prog_files = []
        items = list(directory.iterdir())

        for file in tqdm(items, desc=f"Scanning {desc} directory..."):
            if file.is_file() and file.stat().st_size > 0:
                prog_files.append(file)

        return prog_files

    def build_dataset(self, virtual_exec: bool = False, num_jobs: int = None):
        """Parallel build dataset"""
        if num_jobs is None:
            num_jobs = min(cpu_count(), 16)

        print(f"[build_dataset] merge programs from {len(self.src_dirs)} source directories to {self.dst_dir}")
        print(f"[build_dataset] using {num_jobs} processes for parallel processing")

        if not virtual_exec:
            self.dst_dir.mkdir(parents=True, exist_ok=True)

        all_prog_files = []
        for src_dir in self.src_dirs:
            prog_files = self._get_progs(src_dir, f"{src_dir}")
            all_prog_files.extend(prog_files)
            print(f"[build_dataset] found {len(prog_files)} programs in {src_dir}")

        print(f"[build_dataset] total {len(all_prog_files)} programs to process")

        if virtual_exec:
            print(f"[build_dataset] virtual execution - would copy {len(all_prog_files)} programs")
            return

        with Pool(processes=num_jobs) as pool:
            worker_func = functools.partial(
                _copy_program_worker,
                dst_dir=str(self.dst_dir)
            )

            # use imap_unordered for parallel processing
            successful_copies = 0
            with tqdm(total=len(all_prog_files), desc="Copying programs...") as pbar:
                for result in pool.imap_unordered(worker_func, all_prog_files, chunksize=100):
                    pbar.update(1)
                    if result['success']:
                        successful_copies += 1

        print(f"[build_dataset] successfully copied {successful_copies}/{len(all_prog_files)} programs")
        print(f"[build_dataset] {self.dst_dir} now has {len(list(self.dst_dir.iterdir()))} programs")

    def sample_dataset(self, dataset_size: int, virtual_exec: bool = False, num_jobs: int = None):
        """Parallel sample dataset"""
        if num_jobs is None:
            num_jobs = min(cpu_count(), 16)

        print(f"[build_dataset] sample {dataset_size} programs from merged dataset to {self.dst_dir}")
        print(f"[build_dataset] using {num_jobs} processes for parallel processing")

        # check the src_dir must be a built dataset directory
        if len(self.src_dirs) != 1:
            raise ValueError("sample_dataset requires exactly one source directory (merged dataset)")

        src_dataset_dir = self.src_dirs[0]
        prog_files = self._get_progs(src_dataset_dir, f"dataset {src_dataset_dir}")

        if len(prog_files) < dataset_size:
            print(f"[build_dataset] warning: requested {dataset_size} samples but only {len(prog_files)} available")
            dataset_size = len(prog_files)

        # random sample
        sampled_prog_files = random.sample(prog_files, dataset_size)
        print(f"[build_dataset] selected {len(sampled_prog_files)} programs for sampling")

        if virtual_exec:
            print(f"[build_dataset] virtual execution - would sample {len(sampled_prog_files)} programs")
            return

        self.dst_dir.mkdir(parents=True, exist_ok=True)

        # parallel copy sampled files
        with Pool(processes=num_jobs) as pool:
            worker_func = functools.partial(
                _copy_program_worker,
                dst_dir=str(self.dst_dir)
            )

            # Use imap_unordered for parallel processing
            successful_copies = 0
            with tqdm(total=len(sampled_prog_files), desc="Sampling programs...") as pbar:
                for result in pool.imap_unordered(worker_func, sampled_prog_files, chunksize=50):
                    pbar.update(1)
                    if result['success']:
                        successful_copies += 1

        print(f"[build_dataset] successfully sampled {successful_copies}/{len(sampled_prog_files)} programs")


def _copy_program_worker(prog_file: Path, dst_dir: str) -> Dict:
    """Worker function to copy a single program file"""
    try:
        dst_path = Path(dst_dir) / prog_file.name

        # copy when not exists
        if not dst_path.exists():
            shutil.copy(prog_file, dst_path)
        return {'success': True, 'prog_file': str(prog_file), 'action': 'copied'}

    except Exception as e:
        return {'success': False, 'prog_file': str(prog_file), 'error': str(e)}


class DatasetBuilder:
    def __init__(self, src_dir: str, dst_dir: str, compatibility_mode: bool = False):
        self.src_dir = Path(src_dir)
        self.dst_dir = Path(dst_dir)
        self.compatibility_mode = compatibility_mode
        self.src_programs_dir = self.src_dir / "programs"
        self.src_coverage_dir = self.src_dir / "coverages" if compatibility_mode else self.src_dir / "coverage"
        self.dst_programs_dir = self.dst_dir / "programs"
        self.dst_coverage_dir = self.dst_dir / "coverages" if compatibility_mode else self.dst_dir / "coverage"
        self.src_program_files = self._get_files(self.src_programs_dir, "programs")
        self.dst_programs_dir.mkdir(parents=True, exist_ok=True)
        self.dst_coverage_dir.mkdir(parents=True, exist_ok=True)

    def _get_files(self, directory: Path, desc: str) -> List[Path]:
        """Get all files from a directory with optional progress bar."""
        files = []
        items = list(directory.iterdir())

        for file_path in tqdm(items, desc=f"Scanning {desc} directory..."):
            if file_path.is_file():
                files.append(file_path)

        return files

    def _get_valid_coverage_files(self, program_file: Path) -> List[Path]:
        coverage_files = []
        program_name = program_file.name
        seq = 0
        while True:
            coverage_file = self.src_dir / f"{program_name}-{seq}"
            if not coverage_file.exists():
                break
            coverage_files.append(coverage_file)
            seq += 1
        valid_coverage_files = []
        for coverage_file in coverage_files:
            if coverage_file.stat().st_size > 0:
                valid_coverage_files.append(coverage_file)
        return valid_coverage_files

    def _get_merged_coverage(self, program_file: Path) -> Tuple[List[str], int]:
        """Return merged coverage and zero coverage count"""
        covers = []
        program_name = program_file.name
        zero_coverage_cnt = 0
        seq = 0
        while True:
            coverage_file = self.src_coverage_dir / f"{program_name}-{seq}"
            if not coverage_file.exists():
                break
            seq += 1
            if coverage_file.stat().st_size == 0:
                zero_coverage_cnt += 1
                continue
            covers.append(read_cover(coverage_file))
        return merge_covers(covers), zero_coverage_cnt

    def build_dataset(self, virtual_exec: bool = False, num_jobs: int = None):
        """Parallel build dataset"""
        if num_jobs is None:
            num_jobs = min(cpu_count(), 16)  # limit the number of processes to avoid overloading resources

        print(f"[build_dataset] copy valid programs and coverage from {self.src_dir} to {self.dst_dir}")
        print(f"[build_dataset] using {num_jobs} processes for parallel processing")

        # use Manager to share counters
        with Manager() as manager:
            shared_stats = manager.dict()
            shared_stats['no_coverage_cnt'] = 0
            shared_stats['zero_coverage_cnt'] = 0
            shared_stats['processed_cnt'] = 0

            # create process pool
            with Pool(processes=num_jobs) as pool:
                # create partial function for processing, pass necessary path information
                worker_func = functools.partial(
                    _process_program_for_build,
                    src_coverage_dir=str(self.src_coverage_dir),
                    dst_programs_dir=str(self.dst_programs_dir),
                    dst_coverage_dir=str(self.dst_coverage_dir),
                    virtual_exec=virtual_exec
                )

                # use imap_unordered for parallel processing, show progress bar
                with tqdm(total=len(self.src_program_files), desc="Building dataset...") as pbar:
                    for result in pool.imap_unordered(worker_func, self.src_program_files, chunksize=100):
                        pbar.update(1)

                        # update shared statistics
                        if result['has_coverage']:
                            shared_stats['processed_cnt'] += 1
                        else:
                            shared_stats['no_coverage_cnt'] += 1
                        shared_stats['zero_coverage_cnt'] += result['zero_coverage_cnt']

            print(f"[build_dataset] valid dataset size: {shared_stats['processed_cnt']}, no coverage: {shared_stats['no_coverage_cnt']}, zero coverage: {shared_stats['zero_coverage_cnt']}")

    def sample_dataset(self, dataset_size: int, virtual_exec: bool = False, num_jobs: int = None):
        """Parallel sample dataset"""
        if num_jobs is None:
            num_jobs = min(cpu_count(), 16)

        # check the src_dir must be a built dataset directory, and the coverage file name must not contain '-'
        src_coverage_files = self._get_files(self.src_coverage_dir, "coverage")
        assert len(src_coverage_files) == len(self.src_program_files), "coverage and program files are not matched"
        assert '-' not in src_coverage_files[0].name, "coverage file name must not contain '-'"

        # start sampling, sample size is the minimum of dataset_size and the number of programs
        sample_size = min(dataset_size, len(self.src_program_files))
        print(f"[build_dataset] sample {sample_size} programs and coverage from {self.src_dir} to {self.dst_dir}")
        print(f"[build_dataset] using {num_jobs} processes for parallel processing")

        sampled_program_files = random.sample(self.src_program_files, sample_size)

        if virtual_exec:
            print(f"[build_dataset] virtual execution - would sample {len(sampled_program_files)} programs")
            return

        # create process pool for parallel copying
        with Pool(processes=num_jobs) as pool:
            worker_func = functools.partial(
                _process_program_for_sample,
                src_coverage_dir=str(self.src_coverage_dir),
                dst_programs_dir=str(self.dst_programs_dir),
                dst_coverage_dir=str(self.dst_coverage_dir)
            )

            # use imap_unordered for parallel processing
            list(tqdm(
                pool.imap_unordered(worker_func, sampled_program_files, chunksize=50),
                total=len(sampled_program_files),
                desc="Sampling dataset..."
            ))


def _process_program_for_build(program_file: Path, src_coverage_dir: str, dst_programs_dir: str, dst_coverage_dir: str, virtual_exec: bool) -> Dict:
    """Process single program file for building dataset"""
    try:
        # get merged coverage
        merged_coverage, zero_coverage_cnt = _get_merged_coverage_worker(program_file, src_coverage_dir)

        result = {
            'has_coverage': len(merged_coverage) > 0,
            'zero_coverage_cnt': zero_coverage_cnt,
            'program_file': str(program_file)
        }

        if len(merged_coverage) > 0 and not virtual_exec:
            dst_program_path = Path(dst_programs_dir) / program_file.name
            if not dst_program_path.exists():
                shutil.copy(program_file, dst_program_path)

            # write merged coverage file
            dst_coverage_path = Path(dst_coverage_dir) / program_file.name
            with open(dst_coverage_path, "w") as f:
                for line in merged_coverage:
                    f.write(line + "\n")

        return result

    except Exception as e:
        return {
            'has_coverage': False,
            'zero_coverage_cnt': 0,
            'program_file': str(program_file),
            'error': str(e)
        }


def _process_program_for_sample(program_file: Path, src_coverage_dir: str, dst_programs_dir: str, dst_coverage_dir: str):
    """Process single program file for sampling dataset"""
    try:
        # copy program file
        dst_program_path = Path(dst_programs_dir) / program_file.name
        shutil.copy(program_file, dst_program_path)

        # copy coverage file
        src_coverage_path = Path(src_coverage_dir) / program_file.name
        dst_coverage_path = Path(dst_coverage_dir) / program_file.name
        shutil.copy(src_coverage_path, dst_coverage_path)

        return str(program_file)

    except Exception as e:
        return f"Error processing {program_file}: {str(e)}"


def _get_merged_coverage_worker(program_file: Path, src_coverage_dir: str) -> Tuple[List[str], int]:
    """Get merged coverage in worker process"""
    covers = []
    program_name = program_file.name
    zero_coverage_cnt = 0
    seq = 0
    src_coverage_dir_path = Path(src_coverage_dir)

    while True:
        coverage_file = src_coverage_dir_path / f"{program_name}-{seq}"
        if not coverage_file.exists():
            break
        seq += 1
        if coverage_file.stat().st_size == 0:
            zero_coverage_cnt += 1
            continue
        covers.append(read_cover(coverage_file))

    return merge_covers(covers), zero_coverage_cnt


if __name__ == '__main__':
    t0 = time()
    parser = argparse.ArgumentParser(description='Dataset Preprocessor')
    parser.add_argument('-d', '--rawdump_dir', type=str, help='rawdump dir, used with -o, conflict with -D')
    parser.add_argument('-D', '--rawdataset_dir', type=str, help='rawdataset dir, used with -o and -s, conflict with -d')
    parser.add_argument('-p', '--rawprog_dir', type=str, nargs='+', help='rawdump prog dirs, used with -o, conflict with -P')
    parser.add_argument('-P', '--progdataset_dir', type=str, nargs='+', help='raw prog dataset dirs, used with -o and -s, conflict with -p')
    parser.add_argument('-o', '--out_dir', type=str, required=True, help='out dir, used with -d or -D')
    parser.add_argument('-s', '--dataset_size', type=int, help='size of the dataset to be sampled, used with -D, -P and -o')
    parser.add_argument('-V', '--virtual_exec', action='store_true', help='virtual execution, print the stats but not write to out_dir')
    parser.add_argument('-C', '--compatibility_mode', action='store_true', help='compatibility mode, used with -d or -D')
    parser.add_argument('-j', '--jobs', type=int, help='number of processes to use (default: min(cpu_count(), 16))')
    args = parser.parse_args()

    if args.virtual_exec:
        print('[build_dataset] virtual execution, print the stats but not write to out_dir')

    if args.rawdump_dir:
        builder = DatasetBuilder(args.rawdump_dir, args.out_dir, args.compatibility_mode)
        builder.build_dataset(args.virtual_exec, args.jobs)
    elif args.rawdataset_dir:
        builder = DatasetBuilder(args.rawdataset_dir, args.out_dir, args.compatibility_mode)
        builder.sample_dataset(args.dataset_size, args.virtual_exec, args.jobs)
    elif args.rawprog_dir:
        builder = ProgDatasetBuilder(args.rawprog_dir, args.out_dir)
        builder.build_dataset(args.virtual_exec, args.jobs)
    elif args.progdataset_dir:
        builder = ProgDatasetBuilder(args.progdataset_dir, args.out_dir)
        builder.sample_dataset(args.dataset_size, args.virtual_exec, args.jobs)
    else:
        print('[build_dataset] -d or -D or -p or -P must be provided')
        exit(1)
    print(f'[build_dataset] time cost: {time() - t0:.2f}s')