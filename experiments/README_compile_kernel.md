# Kernel Compilation Script

## Overview

`compile_kernel.py` is a Python script for batch compilation of Linux kernels based on a CSV configuration file. It supports both command-line usage and programmatic import as a module.

## Features

- Batch compilation of multiple kernel versions
- CSV-based configuration (easy to edit and extend)
- Automatic error handling and logging
- Summary statistics after compilation
- Support for parallel compilation jobs
- Support rebuild on existing kernel directories
- Can be imported as a module or used as CLI tool
- Dry-run mode for testing without actual compilation
- Automatic fallback to `linux-stable-git-master` when a commit is not present in `linux-git-master`

## Usage

### Command Line

```bash
python3 compile_kernel.py \
    --workdir /path/to/workdir \
    --linux-git-master /path/to/linux-git-master \
    [--linux-stable-git-master /path/to/linux-stable-git-master] \
    --config config.csv \
    -j 8 \
    [--dry-run]
```

**Arguments:**
- `--workdir`: Working directory where kernels will be compiled
- `--linux-git-master`: Path to the linux-git-master repository
- `--linux-stable-git-master`: Optional path to the linux-stable-git-master repository
- `--config`: CSV configuration file with compilation targets
- `-j, --jobs`: Number of parallel jobs for make (default: CPU count)
- `--dry-run`: Skip actual compilation (useful for testing)
- `--rebuild`: Rebuild on top of existing kernel directories

If `--linux-stable-git-master` is not provided, the script automatically tries a sibling
directory named `linux-stable-git-master` when a target commit cannot be found in
`linux-git-master`.

### CSV Configuration Format

```csv
# Format: case_name,commit_hash,config_path[,optional_columns...]
# Lines starting with # are comments

case_1,761c6d7ec820,case_1.config
case_2,e8f71f89236e,case_2.config
case_3,02d5e016800d,case_3.config
```

**Notes:**
- Only the first 3 columns are parsed: `case_name`, `commit_hash`, `config_path`
- You can add additional columns for your own reference (they will be ignored)
- Relative paths in `config_path` are resolved relative to the CSV file location
- Lines starting with `#` are treated as comments
- Empty lines are ignored

### Example

```bash
# Compile kernels for cases 1-5
python3 experiments/compile_kernel.py \
    --workdir ~/kernels/test \
    --linux-git-master ~/kernels/linux-git-master \
    --linux-stable-git-master ~/kernels/linux-stable-git-master \
    --config benchmark/compile_targets.csv \
    -j 8
```

### Dry Run (Testing)

```bash
# Test the script without actual compilation
python3 experiments/compile_kernel.py \
    --workdir ~/kernels/test \
    --linux-git-master ~/kernels/linux-git-master \
    --linux-stable-git-master ~/kernels/linux-stable-git-master \
    --config experiments/test_compile.csv \
    -j 4 \
    --dry-run
```

## Programmatic Usage

You can import and use the module in your own Python scripts:

```python
from compile_kernel import compile_kernels_batch, compile_single_kernel
from pathlib import Path

# Batch compilation
results = compile_kernels_batch(
    config_file="benchmark/compile_targets.csv",
    workdir="/home/user/kernels/test",
    linux_git_master="/home/user/kernels/linux-git-master",
    jobs=8,
    dry_run=False,
    linux_stable_git_master="/home/user/kernels/linux-stable-git-master",
)

# Check results
for result in results:
    if result.success:
        print(f"{result.case_name}: SUCCESS")
    else:
        print(f"{result.case_name}: FAILED - {result.message}")

# Single kernel compilation
result = compile_single_kernel(
    case_name="test_case",
    commit_hash="761c6d7ec820",
    config_path="/path/to/kernel.config",
    workdir=Path("/home/user/kernels/test"),
    linux_git_master=Path("/home/user/kernels/linux-git-master"),
    jobs=8,
    dry_run=False,
    linux_stable_git_master=Path("/home/user/kernels/linux-stable-git-master"),
)
```

See `example_import_compile.py` for more examples.

## Compilation Process

For each target kernel, the script performs the following steps:

1. **Select source repository**: Try `linux-git-master`; if the commit is absent there,
   fall back to `linux-stable-git-master`
2. **Copy repository**: Copy the selected repository to `workdir/case_N`
3. **Clean & checkout**: Clean the working directory and checkout the specified commit
4. **Remove .git**: Delete the `.git` directory to save disk space
5. **Copy config**: Copy the `.config` file to the kernel directory
6. **Configure**: Run `make CC=gcc olddefconfig`
7. **Compile**: Run `make CC=gcc -jX` (where X is the number of jobs)

## Error Handling

- Failed compilations are logged to `workdir/case_N_fail.log`
- Compilation failures do not stop the main process
- A summary of all compilation results is printed at the end
- The script exits with code 1 if any compilation failed, 0 if all succeeded

## Output

### Success

When all kernels compile successfully:
- Compiled kernels are in `workdir/case_N/`
- No error logs are generated
- Summary shows all successes

### Failure

When a compilation fails:
- Error log is saved to `workdir/case_N_fail.log`
- Failed case is listed in the summary
- Other cases continue to compile
- Script exits with code 1

## Example Output

```
Configuration file: /home/user/experiments/test_compile.csv
Working directory: /home/user/kernels/test
Linux git master: /home/user/kernels/linux-git-master
Jobs: 8
Dry run: False
================================================================================

Found 3 target(s) to compile:
  - case_1: 761c6d7ec820 (/home/user/benchmark/configs/case_1.config)
  - case_2: e8f71f89236e (/home/user/benchmark/configs/case_2.config)
  - case_3: 02d5e016800d (/home/user/benchmark/configs/case_3.config)
================================================================================

[1/3] Processing case_1...
[case_1] Starting compilation...
[case_1] Copying repository...
[case_1] Checking out commit 761c6d7ec820...
[case_1] Compilation completed successfully!
[1/3] case_1: SUCCESS

[2/3] Processing case_2...
...

================================================================================
COMPILATION SUMMARY
================================================================================
Total: 3
Success: 2
Failed: 1

Failed cases:
  - case_2: Kernel compilation failed (exit code: 2)
```

## Tips

1. **Disk Space**: Each kernel compilation requires ~5-10 GB of disk space
2. **Time**: Compilation can take 30-60 minutes per kernel (depending on hardware)
3. **Parallel Jobs**: Use `-j` equal to your CPU core count for best performance
4. **Testing**: Always use `--dry-run` first to verify your configuration
5. **Logs**: Check `case_N_fail.log` files for detailed error information

## Files

- `compile_kernel.py` - Main script
- `example_import_compile.py` - Example of importing as module
- `test_compile.csv` - Example test configuration
- `benchmark/compile_targets.csv` - Sample batch configuration
