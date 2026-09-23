# run_syzkaller.py - Batch Syzkaller Fuzzing

A script for running batch directed kernel fuzzing experiments with Syzkaller.
It generates one syzkaller config per target/round, starts the corresponding
`syz-manager` processes, and can pin each fuzzer instance to a fixed CPU set.

## Quick Start

```bash
cd /root/syzkaller-exp

# Dry run: generate configs and print CPU/port assignment without fuzzing.
python3 run_syzkaller.py \
  --targets batch/batch_2.csv \
  --template syzkaller_base.cfg \
  --image-dir image-1 \
  --workdir exp \
  --start-port 12630 \
  --timeout 48h \
  --rounds 3 \
  --cpu-pool 0-89 \
  --cpus-per-instance 2 \
  --cfg-overrides cfg_overrides.json \
  --dry-run

# Production run.
python3 run_syzkaller.py \
  --targets batch/batch_2.csv \
  --template syzkaller_base.cfg \
  --image-dir image-1 \
  --workdir exp \
  --start-port 12630 \
  --timeout 48h \
  --rounds 3 \
  --cpu-pool 0-89 \
  --cpus-per-instance 2 \
  --cfg-overrides cfg_overrides.json
```

Do not wrap the whole Python command with an outer `taskset` when using
`--cpu-pool`. The script applies `taskset -c <cpu-set>` to each individual
`syz-manager` instance.

## Target CSV

The target CSV has no header. Lines starting with `#` are ignored.

```csv
# case_name,kernel_dir,task_name,target_pc
case_13,/root/kernels/SyzPilot-experiments/cases/case_13,kernel BUG in foo,0xffffffff81000000
```

Columns:

- `case_name`: case identifier used under the output directory
- `kernel_dir`: absolute path to the compiled kernel object directory
- `task_name`: bug description or patch commit id
- `target_pc`: target program counter address

## Template Config

`syzkaller_base.cfg` provides the base syzkaller manager config. The script
rewrites these fields per run:

- `image`
- `sshkey`
- `syzkaller`
- `kernel_obj`
- `workdir`
- `http`
- `vm.kernel`
- `SyzPilot.dump_dir`
- `SyzPilot.task_name`
- `SyzPilot.target_pcs`

The per-instance default CPU demand is `vm.count * vm.cpu`. With the current
template, `vm.count=1` and `vm.cpu=2`, so each `syz-manager` instance defaults
to 2 CPUs when `--cpu-pool` is set.

## Arguments

Required:

- `--targets`: path to the targets CSV file
- `--template`: path to the syzkaller config template
- `--image-dir`: directory containing `bullseye.img` and `bullseye.id_rsa`
- `--workdir`: root work directory for generated configs and syzkaller output

Optional:

- `--start-port`: first HTTP port to try, default `12630`
- `--timeout`: timeout per `syz-manager`, for example `24h` or `48h`
- `--rounds`: new runs to prepare per target, default `3`
- `--max-workers`: maximum parallel Python workers, default all prepared tasks
- `--dry-run`: generate configs and print assignments without starting fuzzers
- `--rerun`: delete existing output for the target cases before preparing new
  runs; this is intended for failed/invalid cases that should restart at run 1
- `--cpu-pool`: CPU ids available to fuzzers, for example `0-89` or `0,4,7-11`
- `--cpus-per-instance`: CPU ids assigned to each `syz-manager`; defaults to
  `vm.count * vm.cpu` when `--cpu-pool` is set
- `--cfg-overrides`: optional per-case generated-config overrides JSON

## CPU Pinning

When `--cpu-pool` is provided, the script parses it as a taskset-style CPU list
and allocates fixed CPU sets in task order.

Example:

```bash
python3 run_syzkaller.py \
  --targets batch/batch_2.csv \
  --template syzkaller_base.cfg \
  --image-dir image-1 \
  --workdir exp \
  --rounds 3 \
  --cpu-pool 0-89 \
  --cpus-per-instance 2
```

For 15 targets and 3 rounds, this prepares 45 instances. With 2 CPUs per
instance, the script requires 90 CPUs and assigns:

```text
case_13/run1 -> 0-1
case_13/run2 -> 2-3
...
case_27/run3 -> 88-89
```

Formula:

```text
instances = number_of_targets * rounds
required_cpus = instances * cpus_per_instance
```

If `required_cpus` exceeds the size of `--cpu-pool`, the script exits before
starting fuzzing.

Current behavior allocates a unique CPU set for every prepared task. This means
`--cpu-pool` must cover all `targets * rounds` tasks even if `--max-workers` is
smaller than the total task count.

## Config Overrides

Use `--cfg-overrides` for case-specific changes to the generated syzkaller
config. The JSON top level maps `case_name` to a partial config object whose
structure matches the final fuzzer config.

Example `cfg_overrides.json`:

```json
{
  "case_22": {
    "vm": {
      "cmdline": "systemd.unified_cgroup_hierarchy=0"
    }
  }
}
```

Run with:

```bash
python3 run_syzkaller.py \
  --targets batch/batch_2.csv \
  --template syzkaller_base.cfg \
  --image-dir image-1 \
  --workdir exp \
  --rounds 3 \
  --cpu-pool 0-89 \
  --cpus-per-instance 2 \
  --cfg-overrides cfg_overrides.json
```

Override semantics are intentionally uniform and simple:

- If a key does not exist in the generated config, it is created.
- If a key exists and both values are JSON objects, the merge recurses.
- Otherwise the override value replaces the generated value.
- Strings, lists, numbers, booleans, and null are not appended or merged.

This is deliberate. The script does not implement field-specific append logic,
because that creates hidden semantics and makes config provenance harder to
reason about. If you need to preserve an existing scalar, string, or list value,
put the full desired value in `cfg_overrides.json`.

For example, if a template already contains:

```json
{
  "vm": {
    "cmdline": "debug earlyprintk=serial"
  }
}
```

and you want to add `systemd.unified_cgroup_hierarchy=0`, write the complete
final value:

```json
{
  "case_22": {
    "vm": {
      "cmdline": "debug earlyprintk=serial systemd.unified_cgroup_hierarchy=0"
    }
  }
}
```

Avoid overriding fields that are generated per run, such as `http`, `workdir`,
`vm.kernel`, or `SyzPilot.dump_dir`, unless you intentionally want to replace
the script-generated value.

## run.sh Logging

Use `tee` when you want the terminal to keep showing logs while also writing the
same output to `run.log`.

```bash
#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

{
  echo "[$(date -Is)] Starting run_syzkaller"
  python3 -u ~/syzkaller-exp/run_syzkaller.py \
    --targets ~/syzkaller-exp/batch/batch_2.csv \
    --template ~/syzkaller-exp/syzkaller_base.cfg \
    --image-dir ~/syzkaller-exp/image-1 \
    --workdir ~/syzkaller-exp/exp \
    --start-port 12630 \
    --timeout 48h \
    --rounds 3 \
    --cpu-pool 0-89 \
    --cpus-per-instance 2 \
    --cfg-overrides ~/syzkaller-exp/cfg_overrides.json
  echo "[$(date -Is)] run_syzkaller finished"
} 2>&1 | tee -a run.log
```

`-u` for unbuffered real-time refresh with pipe command. `2>&1` sends stderr to stdout. `tee -a run.log` appends that combined stream to
`run.log` and still prints it to the console. `set -o pipefail` keeps the script
exit code non-zero if `run_syzkaller.py` fails.

## Output Directory Structure

For `--workdir /root/syzkaller-exp/exp`, output is written under:

```text
/root/syzkaller-exp/exp/syzkaller/
├── case_13/
│   ├── 1/
│   ├── 2/
│   ├── 3/
│   ├── 1.cfg
│   ├── 1.console.log
│   ├── 1.bench.log
│   └── ...
└── ...
```

Per-run files:

- `<run_id>.cfg`: generated syzkaller config
- `<run_id>.console.log`: `syz-manager` stdout/stderr
- `<run_id>.bench.log`: syzkaller bench metrics
- `<run_id>/`: syzkaller workdir with corpus, crashes, and dumps

## Run Ids And Reruns

By default, the script preserves existing case output and starts from the next
available run id. It detects both numeric run directories and per-run files such
as `N.cfg`, `N.console.log`, and `N.bench.log`.

- first `--rounds 3`: creates `1`, `2`, `3`
- later `--rounds 2`: creates `4`, `5`

Use this default behavior when a case already has healthy runs and only needs
extra repetitions.

For failed or invalid cases, pass `--rerun`. The script removes the existing
output directory for each case listed in `--targets`, recreates it, and starts
numbering from `1`.

Example:

```bash
python3 run_syzkaller.py \
  --targets batch/rerun_cases.csv \
  --template syzkaller_base.cfg \
  --image-dir image-1 \
  --workdir exp \
  --timeout 48h \
  --rounds 3 \
  --cpu-pool 0-5 \
  --cpus-per-instance 2 \
  --cfg-overrides cfg_overrides.json \
  --rerun
```

`--rerun` is intentionally not allowed with `--dry-run`, because rerun mode is
destructive.

## Port Allocation

HTTP ports are assigned from `--start-port`. The script checks availability and
skips ports already in use.

## Monitoring

Check running managers:

```bash
pgrep -af syz-manager
```

Check CPU affinity for managers:

```bash
pgrep -f syz-manager | while read -r pid; do taskset -pc "$pid"; done
```

Access a syzkaller web UI at the `http` port shown in the generated `.cfg` file.
For example:

```bash
grep '"http"' /root/syzkaller-exp/exp/syzkaller/case_13/1.cfg
```

## Notes

- Ensure `arch/x86/boot/bzImage` exists under each target kernel directory.
- Ensure `bullseye.img` and `bullseye.id_rsa` exist under `--image-dir`.
- Use `--dry-run` before a production run to verify port and CPU assignment.
- Exposing syzkaller HTTP ports on `0.0.0.0` can be risky on shared networks;
  restrict access with firewall rules or SSH tunneling when needed.
