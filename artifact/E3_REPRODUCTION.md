# E3: Bounded Result Reproduction

E3 is separate from the E2 functional test. E2 checks that the components work
together and that the online model lifecycle can complete. E3 compares directed
outcomes against a Syzkaller-like control under a frozen, paired protocol.

This experiment is a scaled-down reproduction. It does not claim to rerun the
paper's complete 70-case, 10-repetition, 24-hour evaluation or the full
SyzDirect/Snowplow comparison. Its reachability claim is limited to case 36:
whether Full SyzPilot reaches and repeatedly executes the final target basic
block more often than the control under the same four-hour budget. Case 21
provides a separate cross-case check that the online model can be activated.
Exact target-bug crashes remain
an independently reported outcome and are claimed only when the retained
evidence supports them.

## Frozen design

- Cases: `case_21` and `case_36`, selected before running E3. Both are paper
  ablation cases and exercise different kernel subsystems.
- Arms:
  - `Blank`: Brain registration, static/cold-start guidance, directed corpus,
    Predictor, IG, and sequence/property guidance are disabled. Compact target
    observation remains enabled only to measure target hits.
  - `Full`: report-derived cold start, directed corpus, online three-stage
    Predictor, sequence guidance, and IG guidance are enabled. No target PoC,
    reproducer, syscall sequence, or argument information is used.
- Repetitions: three independent paired repetitions per case and arm.
- Duration: four hours per run. The four runs in one repetition execute
  concurrently; the three repetitions execute sequentially.
- Resources per run: one 2-vCPU, 4-GiB VM, `procs=8`, and
  `reproduce=false`.
- Isolation: each run has its own workdir, corpus, manager port, bench log, and
  Brain task name. CPU pairs do not overlap within a repetition.
- GPUs: the Brain receives two physical GPUs, mapped to container devices 0
  (training) and 1 (serving/IG). Formal repetitions require exclusive GPUs and
  record process telemetry once per minute.

The image OCI revision, container Git HEAD, Fuzzer patch, built manager and
executor, kernel build IDs, target PCs, runner, harvester, and this protocol
are recorded in `manifest.json`. The three E3 tools are also copied into the
result directory with SHA-256 hashes so that the rules cannot change silently
between repetitions.

## Hardware and expected time

- Linux x86-64 with Docker and KVM;
- two exclusive CUDA-capable GPUs (approximately 24 GiB free VRAM for training
  and 12 GiB for serving/IG);
- at least 16 free CPU cores and 32 GiB free RAM;
- the already built `case_21` and `case_36` kernels; and
- approximately 30 minutes for setup, 30 minutes for the mandatory pilot, and
  12 hours for the formal fuzzing repetitions.

Run vulnerable kernels only inside isolated VMs.

## Build one candidate revision

Run E3 from one clean candidate revision. Build a local tag without replacing
the Docker Hub release tag. `SYZPILOT_REVISION` must be fetchable from
`SYZPILOT_REPOSITORY`; the default repository works for an already published
revision:

```bash
git rev-parse HEAD
docker build --network host \
  -f docker/Dockerfile \
  --build-arg SYZPILOT_REVISION="$(git rev-parse HEAD)" \
  -t syzpilot:e3-candidate .
```

For a not-yet-published author candidate, first expose that exact commit through
a temporary read-only Git remote (for example, a local `git daemon` used with
`--network host`) and pass its URL explicitly:

```bash
docker build --network host \
  -f docker/Dockerfile \
  --build-arg SYZPILOT_REPOSITORY=git://127.0.0.1:9419/SyzPilot \
  --build-arg SYZPILOT_REVISION="$(git rev-parse HEAD)" \
  -t syzpilot:e3-candidate .
```

The setup command independently checks that the runner, harvester, protocol,
container source, and OCI revision all belong to this same commit. A dirty E3
tool is rejected even if unrelated files are present in the worktree.

If the environment requires an HTTP(S) proxy during the build, pass the
corresponding `http_proxy`, `https_proxy`, and `no_proxy` build arguments.
During setup, the runner also forwards any standard proxy variables already
set on the host to the Fuzzer container while it prepares Syzkaller.

## Setup and mandatory pilot

Choose an empty result directory, the two physical GPUs, and four nonoverlapping
two-core CPU pairs. Replace the result and kernel placeholders with persistent
local paths. The source-only mini-benchmark and kernel-build workflow are
documented in [`../mini-benchmark/README.md`](../mini-benchmark/README.md).

```bash
REVISION="$(git rev-parse HEAD)"
RESULT_ROOT=/path/to/e3-results
CPU_PAIRS='0-1;2-3;4-5;6-7'  # Replace these if the cores are busy.

python scripts/run_e3_reproduction.py setup \
  --image syzpilot:e3-candidate \
  --expected-revision "$REVISION" \
  --output-root "$RESULT_ROOT" \
  --kernel-root /path/to/SyzPilot-experiments/cases \
  --gpu-devices 0,1

python scripts/run_e3_reproduction.py pilot \
  --output-root "$RESULT_ROOT" \
  --cpu-pairs "$CPU_PAIRS"
```

The 30-minute pilot is not a formal repetition. It must finish normally and
show nonzero executions in both arms, successful Full registration, readable
Brain task state, and at least one Receiver batch. A failed pilot must be fixed
and rerun from a fresh setup rather than waived.

## Formal run and collection

Before starting, `nvidia-smi` must show no external compute process on either
selected GPU. The runner rejects an occupied GPU and invalidates a repetition
if an external process appears later.

```bash
python scripts/run_e3_reproduction.py formal \
  --output-root "$RESULT_ROOT" \
  --cpu-pairs "$CPU_PAIRS"

python scripts/harvest_e3_reproduction.py --root "$RESULT_ROOT"
```

An interrupted or invalid repetition is retained and rerun as a new attempt;
the selected attempt is recorded explicitly. The harvester never selects runs
by outcome.

## Measurements and claim boundary

For every run, E3 reports final-PC hit success, interval-censored first-hit
time, the raw number of final-PC-hitting executions (`N_h` in the paper),
final hits per execution, exact target crash title, executions per
second, VM restarts, model-ready time, model versions, and maximum curriculum
stage. All paired raw repetitions remain visible.

The result is eligible for an appendix reproduction claim only if:

1. all 12 frozen runs complete with matching identities, full duration, and
   exit status 0;
2. no external GPU process appears in formal telemetry;
3. at least four Full runs, including one per case, show a successful live
   Predictor inference after model deployment;
4. on `case_36`, Full reaches the final target in more repetitions than Blank
   and has a larger raw number of final-PC-hitting executions in at least two
   of three paired repetitions. The normalized hit rate is reported only to
   help interpret throughput differences; and
5. a bug-reproduction claim is made only if Full produces more exact target
   crashes than Blank across the two cases.

If checks 1--4 fail, E3 is not used to request Results Reproduced. If only
check 5 fails, the evidence can support at most the narrower target-reachability
direction, subject to evaluator acceptance. No two-case result is extrapolated
to the paper's full benchmark effect size.

## Cleanup

After preserving `manifest.json`, `results.json`, bench logs, crash reports,
Receiver/model events, and GPU telemetry:

```bash
python scripts/run_e3_reproduction.py cleanup --output-root "$RESULT_ROOT"
```

Cleanup removes only the two E3 containers. It does not delete the result
directory.
