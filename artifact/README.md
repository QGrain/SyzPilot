# SyzPilot NDSS 2027 functional artifact

This artifact accompanies *SyzPilot: Steering Directed Kernel Fuzzing from
Reachability Prediction to Attribution-Guided Scheduling*. The evaluation
primarily targets the **Available** and **Functional** badges through E1/E2.
The optional E3 protocol provides bounded evidence for considering Results
Reproduced without claiming to repeat the paper's full multi-target,
multi-run performance evaluation.

The normal functional workflow accepts a public bug report and matching kernel
build. It never uses the target PoC or reproducer as a fuzzing seed or guidance
input. Reaching a target PC and reproducing the corresponding crash are
reported separately.

This is the maintained evaluator-facing guide. The commands below use the
single helper [`scripts/artifact_eval.sh`](../scripts/artifact_eval.sh) for
repeatable in-container steps; each section also states the expected output.

## Evaluation budget

| Activity | Approximate wall-clock time |
| --- | ---: |
| Pull or build `qgrain/syzpilot:ndss27-ae` | 10--30 min |
| Patch and build SyzPilot-Fuzzer | 5--20 min |
| First case-36 kernel build | 1--3 h |
| E1 waypoint extraction | 5--15 min |
| E2-A Fuzzer-side functional test | 10 min |
| E2-B online training and inference test | up to 4 h |
| E3 bounded result reproduction (optional) | about 13 h |
| Inspect evidence and clean up | 5--15 min |

The kernel build is reusable when its source, config, compiler, `vmlinux`, and
`bzImage` remain unchanged. Network and hardware speed can substantially alter
the estimates.

## Requirements

- Linux x86-64, Docker, and at least 8 CPU cores.
- A Fuzzer host with KVM. The smoke configuration uses one 2-vCPU, 4-GiB guest
  and `procs=8`.
- E2-A needs only the KVM-capable Fuzzer host; it does not start the Brain or
  require an NVIDIA GPU.
- E2-B additionally needs a Brain host with the NVIDIA container runtime and two
  visible NVIDIA GPUs:
  approximately 24 GiB free VRAM for training and 12 GiB for serving/IG.
- For E2-B only, private two-way connectivity between the Brain and Fuzzer.
  The Fuzzer must reach Brain TCP 48000, 31001--31999, and 37030--37034. The
  Brain must reach the configured manager callback port.
- Approximately 30 GiB of free workspace beyond the container image.

Run vulnerable kernels only inside isolated VMs. Do not expose Controller,
Receiver, TorchServe, or manager callback ports to the public Internet.

Evaluators without NVIDIA GPUs can complete A1, A2, E1, and E2-A in the
Fuzzer container. E2-A validates the patched Fuzzer's VM, execution, KCOV, and
target-observation path; it does not claim to validate online training,
deployment, inference, or Brain-to-Fuzzer communication. E2-B covers those
GPU-dependent functions.

## A0: Create the two roles

The same image is used on both hosts. E2-A needs only the Fuzzer container;
E2-B uses both containers. The image deliberately starts no service by
default. Its environment selects direct-only Brain networking and supplies
the Linux/amd64 Syzlang manifest and descriptions matched to the pinned
Fuzzer patch.

```bash
docker pull qgrain/syzpilot:ndss27-ae
docker image inspect qgrain/syzpilot:ndss27-ae \
  --format '{{index .Config.Labels "org.opencontainers.image.revision"}}'
mkdir -p /HOST/SYZPILOT-BRAIN-STATE/{receiver_data,logs,model_store}

# GPU host
docker run -d --init --name syzpilot-brain --network host \
  --gpus '"device=0,1"' \
  -v /HOST/SYZPILOT-RUNS:/root/syzpilot-runs \
  -v /HOST/SYZPILOT-BRAIN-STATE/receiver_data:/root/SyzPilot/brain/receiver_data \
  -v /HOST/SYZPILOT-BRAIN-STATE/logs:/root/SyzPilot/brain/logs \
  -v /HOST/SYZPILOT-BRAIN-STATE/model_store:/root/SyzPilot/brain/model_store \
  qgrain/syzpilot:ndss27-ae sleep infinity

# CPU/KVM host
docker run -d --init --name syzpilot-fuzzer --network host --device /dev/kvm \
  -v /HOST/KERNELS:/root/kernels \
  -v /HOST/SYZPILOT-RUNS:/root/syzpilot-runs \
  qgrain/syzpilot:ndss27-ae sleep infinity
```

The image starts no hidden service. The Fuzzer container is deliberately not
created with `--cpuset-cpus`; CPU affinity is applied only to the measured
`syz-manager` process in E2. `--device /dev/kvm` exposes only the device QEMU
needs and is preferred to the much broader `--privileged` mode.

Confirm that the image contains one matched source/model/toolchain snapshot:

```bash
docker exec syzpilot-fuzzer bash -lc \
  'cd /root/SyzPilot && scripts/artifact_eval.sh image-info'
```

Expected output includes a 40-character SyzPilot revision, `gcc (GCC) 11.4.0`,
and `SyzEncoder: available`.

If the prebuilt tag is unavailable, build `docker/Dockerfile` as described in
the root README and [`docker/README.md`](../docker/README.md). For evaluation,
keep the image's source checkout, Fuzzer patch, and compiled Syzlang assets at
their matched revision. Rebuild the image to adopt a newer public revision.

## A1: Patch and build the Fuzzer

Run from the Fuzzer host; the command enters the named container explicitly:

```bash
docker exec syzpilot-fuzzer bash -lc '
  cd /root/SyzPilot
  scripts/artifact_eval.sh prepare-fuzzer
'
```

The patch script obtains exact upstream Syzkaller commit
`6e83b42dcfcd13c3b8e0d5c803cdcc424c0fbff9` over HTTPS and applies
`fuzzer/SyzPilot-fuzzer.diff`. Success ends with
`Fuzzer setup is ready with KVM access.`

## A2: Build case 36

From the Fuzzer host, clone Linux under the mounted kernel workspace and build
only into a new case directory. `compile_kernel.py` removes an existing case
output before a fresh build, so both explicit guards are important.

```bash
docker exec syzpilot-fuzzer bash -lc '
  cd /root/SyzPilot
  scripts/artifact_eval.sh build-case36
'
```

Success ends with `case_36 kernel build is ready` and leaves nonempty
`vmlinux` and `arch/x86/boot/bzImage` files. The helper refuses to overwrite an
existing `/root/kernels/case_36`; move or preserve an earlier build explicitly.

The unified image provides GCC 11.4 at `SYZPILOT_KERNEL_CC`, matching the
retained local benchmark builds. The build helper applies it to both `CC` and
`HOSTCC`; GCC 12 and newer can otherwise fail while building this Linux
5.11-rc2 tree's objtool. This does not replace the image-wide default compiler.
The compiler fields in the checked-in input config are historical benchmark
metadata; `make olddefconfig` refreshes them for the selected compiler before
the build.

For a current source checkout outside the rebuilt image, the equivalent
toolchain setup on Ubuntu 24.04 is:

```bash
sudo apt-get update
sudo apt-get install \
  gcc-11=11.4.0-9ubuntu1 gcc-11-base=11.4.0-9ubuntu1 \
  cpp-11=11.4.0-9ubuntu1 libgcc-11-dev=11.4.0-9ubuntu1 \
  libasan6=11.4.0-9ubuntu1 libtsan0=11.4.0-9ubuntu1
python experiments/compile_kernel.py ... \
  --compiler /usr/bin/gcc-11
```

If a direct `cvm install gcc 11.4.0` fails because the host defaults to GCC 14,
use LLVM only to bootstrap the GCC 11.4 toolchain:

```bash
command -v clang clang++
(
  export CC="$(command -v clang)"
  export CXX="$(command -v clang++)"
  cvm install gcc 11.4.0 --jobs 8
)

python experiments/compile_kernel.py ... \
  --compiler "$HOME/.cvm/toolchains/gcc/11.4.0/bin/gcc"
```

GCC 14 rejects a legacy source incompatibility while building GCC 11.4, whereas
Clang accepts it with a warning. LLVM is used only for this bootstrap step; the
result is a normal GCC 11.4 installation, and GCC 11.4 still compiles the target
kernel. The released image already includes the apt-provided GCC 11.4 and does
not require this workaround.

The unified image inherits a disposable Trixie guest template at
`/root/images/image-template/{disk.img,disk.id_rsa}`. The generated manager
configuration uses these paths. The benchmark kernel is needed only on the
Fuzzer host; the Brain consumes the report, labels, and programs over the
documented interfaces.

## E1: Report-derived waypoints

Run from the Fuzzer host, where the freshly built kernel is already present:

```bash
docker exec syzpilot-fuzzer bash -lc '
  cd /root/SyzPilot
  scripts/artifact_eval.sh extract-case36
'
```

Success is an ordered, nonempty waypoint list followed by a
`[For SyzPilot-fuzzer:]` JSON array. PCs must always be regenerated from the
exact `vmlinux` used by the VM.

For a source installation outside Docker, install
`requirements/requirements-brain.txt` first. It includes the pinned
`tree-sitter` and `tree-sitter-c` packages used by `get_targets.py`.

The two E2 profiles below generate their own manager configurations from this
same waypoint output.

## E2: Directed-fuzzing functional checks

### E2-A: Fuzzer-side functional test

Generate a standalone manager configuration from the PCs produced by E1. This
profile disables Brain registration and ML guidance, but keeps target-PC
observation enabled:

```bash
docker exec syzpilot-fuzzer bash -lc '
  cd /root/SyzPilot
  scripts/artifact_eval.sh prepare-fuzzer-only-config
'

docker exec -d syzpilot-fuzzer bash -lc '
  cd /root/SyzPilot
  scripts/artifact_eval.sh start-fuzzer-only
'
```

During or after the ten-minute run, inspect the Fuzzer log with:

```bash
docker exec syzpilot-fuzzer bash -lc \
  'cd /root/SyzPilot && scripts/artifact_eval.sh fuzzer-status'
```

The log should show a booted guest and increasing execution counts without an
immediate `FATAL` exit. The command also prints the latest bench values for
`exec total`, `coverage`, `executor target observations`, and `executor target
hits`. E2-A passes when the first three values are greater than zero; target
hits may remain zero because this bounded functional check does not require
reaching the final PC. This check requires KVM, but no Brain container or GPU.

### E2-B: Online training and inference test

Create the distributed manager configuration. Replace both example addresses
with mutually reachable private IPs:

```bash
export BRAIN_HOST=10.0.0.10
export FUZZER_HOST=10.0.0.20
docker exec -e BRAIN_HOST="$BRAIN_HOST" -e FUZZER_HOST="$FUZZER_HOST" \
  syzpilot-fuzzer bash -lc '
    cd /root/SyzPilot
    scripts/artifact_eval.sh prepare-config
  '
```

Verify and start the Brain explicitly:

```bash
docker exec syzpilot-brain bash -lc \
  'cd /root/SyzPilot && scripts/artifact_eval.sh verify-brain'

docker exec -d syzpilot-brain bash -lc '
  cd /root/SyzPilot
  scripts/artifact_eval.sh start-brain
'
```

On the Fuzzer host, start the bounded four-hour E2-B path. The helper waits
up to 60 seconds for the Controller, which avoids a race with its background
startup:

```bash
docker exec -d -e BRAIN_HOST="$BRAIN_HOST" -e SYZPILOT_DURATION=4h \
  syzpilot-fuzzer bash -lc \
  'cd /root/SyzPilot && scripts/artifact_eval.sh start-fuzzer'
```

Set `SYZPILOT_CPUSET` on `docker exec` when cores 0--1 are not available. This
changes only the fuzzer process affinity, not the reusable container:

```bash
docker exec -d -e BRAIN_HOST="$BRAIN_HOST" -e SYZPILOT_CPUSET=8-9 \
  syzpilot-fuzzer bash -lc \
  'cd /root/SyzPilot && scripts/artifact_eval.sh start-fuzzer'
```

Only one manager may use a run root at a time. A later invocation with the
default run root deliberately resumes its existing corpus with new log and
bench filenames. For an independent fresh run, pass a new
`SYZPILOT_RUN_ROOT` consistently to `extract-case36`, `prepare-config`, and
`start-fuzzer`.

Inspect progress:

```bash
curl -fsS "http://${BRAIN_HOST}:48000/list_tasks" | python -m json.tool
docker exec -e BRAIN_HOST="$BRAIN_HOST" syzpilot-brain bash -lc \
  'cd /root/SyzPilot && scripts/artifact_eval.sh brain-status'
docker exec syzpilot-fuzzer bash -lc \
  'cd /root/SyzPilot && scripts/artifact_eval.sh fuzzer-status'
```

E2-B observations are progressive:

1. registration, a running QEMU guest, and increasing execution counts;
2. report-derived guidance and labeled Receiver batches; and
3. Stage-1 training, model promotion, TorchServe notification, and successful
   prediction during the four-hour online training and inference test.

| Check | Expected observation |
| --- | --- |
| Brain health | The `/health` request succeeds and `list_tasks` becomes nonempty after registration. |
| Fuzzer runtime | The latest fuzzer log shows a running VM and increasing executions rather than an immediate `FATAL` exit. |
| Static guidance | The task view reports a nonzero guidance version after report-derived guidance is accepted. |
| Training data | New Receiver batch files appear under the mounted Brain state directory. |
| Online ML | Before the four-hour timeout, Brain logs show a promoted model version; the Fuzzer receives model-ready notification and prediction succeeds. |

Stage 1 needs at least 1,000 samples including 100 positive examples. A
complete E2-B pass requires the Online ML observation before the four-hour hard
timeout; otherwise preserve the evidence and report E2-B as incomplete rather
than waiting without a bound. E2-A remains an independently useful GPU-free
Fuzzer check, but it does not substitute for E2-B's Brain and online-ML
observations. A target crash is not required for either functional check.

E2 does not carry a Results Reproduced claim. The optional E3 protocol uses a
separate paired Blank/Full experiment and predeclared evidence gates; see
[`E3_REPRODUCTION.md`](E3_REPRODUCTION.md).

## Evidence and cleanup

For E2-A, retain the manager bench log, Fuzzer log, target observations, and
any crash reports, then stop only `syzpilot-fuzzer`. For E2-B, additionally
retain the Brain log, Receiver/model state, and final task view. The bind mounts
above persist these files after container removal.

E2-B only: capture the final task view before stopping the two named
containers:

```bash
curl -fsS "http://${BRAIN_HOST}:48000/list_tasks" \
  > /HOST/SYZPILOT-RUNS/controller_tasks.json
# Run each pair on the host that owns the named container.
docker stop syzpilot-brain && docker rm syzpilot-brain
docker stop syzpilot-fuzzer && docker rm syzpilot-fuzzer
```

For E2-A, run only the final Fuzzer stop/remove command.

Use a new bench filename on every restart because `syz-manager` refuses to
overwrite an existing one. Do not delete shared kernels, guest templates, or
earlier experiment results during cleanup.
