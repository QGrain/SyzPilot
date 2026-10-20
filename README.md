# SyzPilot: Steering Directed Kernel Fuzzing from Reachability Prediction to Attribution-Guided Scheduling

<div align="center">
<a href="https://doi.org/10.5281/zenodo.22874328"><img alt="Artifact Zenodo" src="https://zenodo.org/badge/DOI/10.5281/zenodo.22874328.svg"/></a>
<a href="https://huggingface.co/zzra1n/SyzEncoder"><img alt="SyzEncoder" src="https://img.shields.io/badge/%F0%9F%A4%97%20Model-SyzEncoder-ffc107"/></a>
<a href="https://huggingface.co/datasets/zzra1n/SyzPilot-dataset"><img alt="SyzPilot Dataset" src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-SyzPilot--dataset-ffc107"/></a>
<a href="https://deepwiki.com/QGrain/SyzPilot"><img alt="DeepWiki" src="https://deepwiki.com/badge.svg"/></a>
<a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache--2.0-blue.svg"/></a>
</div>

> [!NOTE]
> **For artifact evaluation reviewers:** this is the maintained source
> repository for the SyzPilot artifact. We may continue improving
> documentation and portability during the evaluation period, while the
> submitted functional workflow, versioned source package, and observable
> success criteria remain stable and available. The current command-by-command
> procedure is the [Functional artifact guide](artifact/README.md).

> [!NOTE]
> The name **SyzPilot** combines Syzkaller's conventional `Syz` prefix with
> *Pilot*, reflecting the paper's central idea of steering and controlling
> directed kernel fuzzing. This repository specifically implements ML-guided
> directed kernel fuzzing from bug reports to target-aware scheduling.

SyzPilot is a distributed directed kernel fuzzing framework that combines
report-derived static guidance, online reachability prediction, and
attribution-guided mutation. Given a target bug report and a compiled kernel,
SyzPilot extracts waypoint targets, collects labeled syz-program executions,
trains a reachability classifier online, and feeds model and attribution
guidance back to a modified Syzkaller instance.

This repository implements **"SyzPilot: Steering Directed Kernel Fuzzing from
Reachability Prediction to Attribution-Guided Scheduling,"** accepted at
NDSS 2027.

## 📰 News

| Date | Update |
| --- | --- |
| **Sep 2026** | Released the first public source version for NDSS 2027 artifact evaluation, together with the Zenodo source package, SyzEncoder, and the SyzPilot dataset. |

---

**Quick Glance.** SyzPilot separates GPU-intensive learning from CPU-intensive
kernel fuzzing. The Brain coordinates target analysis, training, deployment,
and guidance; one or more Fuzzers execute programs in isolated kernel VMs and
stream reachability observations back to the Brain. The normal functional
workflow derives guidance from the bug report, kernel, online executions, and
model attribution. It does **not** use the target PoC or reproducer as fuzzing
input.

An auto-generated overview is available on
[DeepWiki](https://deepwiki.com/QGrain/SyzPilot) while its index is being
completed.

```text
┌───────────────────────────────────────────────────────────┐
│                     SyzPilot-Brain (GPU)                  │
│  Controller ── Receiver ── Trainer ── TorchServe          │
│       ├── Path/Static Analyzer                            │
│       └── Attribution + Sequence Guidance                 │
└────────────────────────────┬──────────────────────────────┘
                             │ HTTP + gRPC
┌────────────────────────────┴──────────────────────────────┐
│                   SyzPilot-Fuzzer (CPU/KVM)               │
│  syz-manager ── Collector/Predictor ── guided scheduling  │
│       └── native corpus + Full-profile directed corpus    │
└───────────────────────────────────────────────────────────┘
```

**Project Structure**

```text
.
├── analyzer/         # Report/waypoint extraction and static guidance
├── artifact/         # Functional artifact guide and portable example
├── benchmark/        # Public non-PoC benchmark metadata and reports
├── brain/            # Controller, Receiver, TorchServe, and lifecycle logic
├── common/           # Shared curriculum and label contracts
├── docker/           # Unified Brain/Fuzzer image recipe
├── experiments/      # Kernel builds, experiment launch, and result analysis
├── filter/           # SyzEncoder, classifier training, and attribution
├── fuzzer/           # SyzPilot-Fuzzer and archived baseline source patches
├── mini-benchmark/   # Three source-only functional benchmark cases
├── requirements/     # Brain and Fuzzer dependencies
├── scripts/          # Dataset and experiment utilities
└── tests/            # CPU-side regression and contract tests
```

## 1 Setup

The following wall-clock estimates describe a first-time functional
evaluation. Hardware, network speed, and the target's positive-sample rate can
change them substantially.

| Step | Expected time | Required hardware | Functional observation |
| --- | ---: | --- | --- |
| Pull/build the unified image | 10--30 min | Docker, 30+ GiB disk | Python, Go, Java, and SyzEncoder available |
| Patch and build SyzPilot-Fuzzer | 5--20 min | 8+ CPU cores | Manager, executor, and execprog binaries |
| Build the case-36 kernel | 1--3 h, one time | 8+ CPU cores, 30 GiB disk | Matching `vmlinux` and `bzImage` |
| E1: report-derived waypoint extraction | 5--15 min | CPU only after setup | Ordered waypoints and nonempty target PCs |
| E2-A: Fuzzer-side functional test | 10 min + 5 min inspection | KVM, no GPU | Guest boot, program execution, KCOV, and target-PC observation |
| E2-B: online training and inference test | up to 4 h + 5--15 min inspection | KVM and 2 GPUs | Brain registration, labeled batches, guidance, model promotion, and inference |
| E3: bounded result reproduction | about 13 h | KVM, 2 exclusive GPUs, 16 CPU cores | Paired Blank/Full evidence on cases 21 and 36 |
| Preserve evidence and clean up | 5--10 min | None | Bench/log evidence retained; no orphan containers |

E1/E2 validate functionality rather than the paper's long-running performance
claims. A target-PC hit and an exact target crash are separate observations;
neither is required by either functional profile.

E3 is a separate, optional result-reproduction protocol with frozen controls,
three paired repetitions, and explicit claim gates. See
[`artifact/E3_REPRODUCTION.md`](artifact/E3_REPRODUCTION.md).

### 1.1 Hardware and Software Requirements

For the CPU-only waypoint extraction exercise:

- Linux x86-64 with Docker;
- 8 or more CPU cores, 16 GiB RAM, and approximately 30 GiB free disk space;
- a local Linux checkout and the ability to compile a target kernel.

For E2-A:

- KVM access and QEMU; run vulnerable kernels only inside isolated VMs;
- no NVIDIA GPU or Brain container is required.

For E2-B:

- KVM access and QEMU; run vulnerable kernels only inside isolated VMs;
- two CUDA-capable NVIDIA GPUs so training and serving can be isolated;
- approximately 24 GiB free VRAM on the training GPU and 12 GiB on the
  serving/attribution GPU;
- Java 17, Python 3.11, Docker, and the ordinary Syzkaller build toolchain;
- free Controller, Receiver, manager, and TorchServe ports.

The supplied functional configuration uses one 2-vCPU, 4-GiB guest and
`procs=8`. Longer directed-fuzzing experiments can scale to multiple guests,
but should use explicit non-overlapping CPU affinity.

### 1.2 Artifact Setup with Docker (Recommended)

The release uses one image for both the GPU Brain and CPU/KVM Fuzzer roles.
The image is based on `qgrain/kernel-fuzz:2404_v1`, contains the public
repository, a `syzpilot` Conda environment, the pinned SyzEncoder, and matching
Linux/amd64 Syzlang metadata. It defaults the Controller to direct-only
networking, but has no default service process.

```bash
docker pull qgrain/syzpilot:ndss27-ae

# Equivalent local build:
git clone https://github.com/QGrain/SyzPilot.git
cd SyzPilot
REVISION="$(git rev-parse HEAD)"
docker build -f docker/Dockerfile \
  --build-arg SYZPILOT_REVISION="$REVISION" \
  -t qgrain/syzpilot:ndss27-ae .
```

The complete artifact procedure is documented in
[`artifact/README.md`](artifact/README.md). It covers:

1. pulling or rebuilding the unified image;
2. compiling the pinned example kernel;
3. creating Brain/Fuzzer containers from the unified image;
4. extracting report-derived waypoint PCs;
5. running either the Fuzzer-side test or the online training and inference test; and
6. checking observable outputs and safely cleaning owned containers.

The image does not embed target kernels, target PoCs, or experiment logs. The
base image supplies a disposable guest template; benchmark kernels remain
external and must be built from their pinned commits.

Create the roles explicitly. Replace host paths, GPU IDs, and CPU IDs with
resources assigned to the evaluation:

```bash
# Create these host directories before replacing the /HOST placeholders below.
mkdir -p /HOST/SYZPILOT-BRAIN-STATE/{receiver_data,logs,model_store}

# GPU server
docker run -d --init --name syzpilot-brain --network host \
  --gpus '"device=0,1"' \
  -v /HOST/SYZPILOT-RUNS:/root/syzpilot-runs \
  -v /HOST/SYZPILOT-BRAIN-STATE/receiver_data:/root/SyzPilot/brain/receiver_data \
  -v /HOST/SYZPILOT-BRAIN-STATE/logs:/root/SyzPilot/brain/logs \
  -v /HOST/SYZPILOT-BRAIN-STATE/model_store:/root/SyzPilot/brain/model_store \
  qgrain/syzpilot:ndss27-ae sleep infinity

# CPU/KVM server
docker run -d --init --name syzpilot-fuzzer --network host --device /dev/kvm \
  -v /HOST/KERNELS:/root/kernels \
  -v /HOST/SYZPILOT-RUNS:/root/syzpilot-runs \
  qgrain/syzpilot:ndss27-ae sleep infinity
```

The Fuzzer container has no Docker CPU-set restriction. The functional command
applies `taskset` only to the measured manager process. `--device /dev/kvm`
exposes the device QEMU needs without the much broader `--privileged` mode.

For persistent development experiments outside Docker, set
`SYZPILOT_DATA_ROOT` to an experiment-specific directory, for example
`$HOME/datasets/SyzPilot/receiver_runs/<RUN_ID>/brain/receiver_data`. The
default remains `brain/receiver_data`.

The source checkout, Fuzzer patch, and compiled Syzlang assets in the image
form one tested snapshot. Use it unchanged for artifact evaluation:

```bash
docker exec syzpilot-brain bash -lc \
  'cd /root/SyzPilot && scripts/artifact_eval.sh image-info'

docker exec syzpilot-fuzzer bash -lc '
  cd /root/SyzPilot
  scripts/artifact_eval.sh prepare-fuzzer
'
```

To use a newer public revision, rebuild the unified image so the source,
patch, and static assets advance together; do not update only one container.

The explicit `sleep infinity` is a user-selected keepalive. The image itself
does not start sshd, a Controller, TorchServe, QEMU, or a fuzzer.

### 1.3 Setup from Source

```bash
conda create -n syzpilot python=3.11
conda activate syzpilot
pip install -r requirements/requirements-brain.txt \
  -r requirements/requirements-fuzzer.txt \
  --extra-index-url https://download.pytorch.org/whl/cu121
bash requirements/required_packages.sh
make -C brain
```

`analyzer/get_targets.py` uses `tree-sitter` and `tree-sitter-c` for
`function@file:line` resolution. Both are installed by
`requirements/requirements-brain.txt`; install that file before running the
analyzer outside Docker.

Build the Fuzzer from the exact upstream Syzkaller revision and reviewed patch:

```bash
cd /path/to/SyzPilot
export SYZPILOT_FUZZER_DIR="$(pwd -P)/../SyzPilot-fuzzer"
scripts/patch_fuzzers.sh SyzPilot
cd "$SYZPILOT_FUZZER_DIR"
make -j"$(nproc)"
cd /path/to/SyzPilot
scripts/verify_setup.sh fuzzer
```

Build the optional KallGraph component using
[`analyzer/README.md`](analyzer/README.md).

### 1.4 Run the CPU Regression Suite

Generate the local protobuf bindings before running the tests. The generated
files are intentionally excluded from version control.

```bash
make -C brain
env -u CUDA_VISIBLE_DEVICES PYTHONPATH=. \
  python -m unittest discover -s tests -p 'test_*.py'
```

## 2 Usage

### 2.1 Extract Report-Derived Waypoints (CPU Only)

```bash
python analyzer/waypoints_extractor.py \
  -k /path/to/compiled/case_N \
  -t benchmark/configs/case_N.title \
  -r benchmark/configs/case_N.report
```

Success is an ordered, nonempty waypoint list followed by a
`[For SyzPilot-fuzzer:]` JSON array of target PCs. PC values must always be
resolved from the exact `vmlinux` used by the corresponding fuzzing VM.

### 2.2 Choose an E2 Profile

E2-A is the ten-minute Fuzzer-side functional test. It does not start the
Brain or require a GPU. E2-B tests online training and inference and requires
the two-GPU Brain. Both use the same case-36 kernel and freshly extracted target
PCs; the commands are shown below.

### 2.3 Run the Complete Functional Mini-Benchmark

[`mini-benchmark/README.md`](mini-benchmark/README.md) provides source-only
metadata for cases 21, 25, and 36. It deliberately excludes compiled kernels,
VM images, target PoCs, and fixed PC addresses. Case 36 is the canonical
Functional workflow below; cases 21 and 25 are additional target profiles for
later evaluation.

First run the centralized helper on the Fuzzer host. It builds with GCC 11.4
and extracts PCs from the exact `vmlinux`:

```bash
docker exec syzpilot-fuzzer bash -lc '
  cd /root/SyzPilot
  scripts/artifact_eval.sh build-case36
  scripts/artifact_eval.sh extract-case36
'
```

For E2-A, generate and run the standalone Fuzzer configuration:

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

For E2-B, start the Brain and then generate and run the distributed
configuration. Replace `BRAIN_HOST` and `FUZZER_HOST` with mutually reachable
private addresses:

```bash
docker exec syzpilot-brain bash -lc \
  'cd /root/SyzPilot && scripts/artifact_eval.sh verify-brain'
docker exec -d syzpilot-brain bash -lc '
  cd /root/SyzPilot
  scripts/artifact_eval.sh start-brain
'

export BRAIN_HOST=10.0.0.10
export FUZZER_HOST=10.0.0.20
docker exec -e BRAIN_HOST="$BRAIN_HOST" -e FUZZER_HOST="$FUZZER_HOST" \
  syzpilot-fuzzer bash -lc \
  'cd /root/SyzPilot && scripts/artifact_eval.sh prepare-config'

docker exec -d -e BRAIN_HOST="$BRAIN_HOST" -e SYZPILOT_DURATION=4h \
  syzpilot-fuzzer bash -lc \
  'cd /root/SyzPilot && scripts/artifact_eval.sh start-fuzzer'
```

`compile_kernel.py` still defaults to the system `gcc` for general use; pass
`--compiler` to select an explicit toolchain outside the unified image. The
selected compiler is used for both the kernel and host tools such as objtool.

The base image supplies the disposable guest paths used by the generated
configuration. See [`artifact/README.md`](artifact/README.md) for kernel
checkout commands, distributed file placement, status checks, expected
observations, and safe cleanup.

Functional observations are separated into three levels:

1. **Fuzzer execution flow (E2-A):** VM boot, program execution, KCOV, target-PC
   observation, and clean shutdown without a GPU.
2. **Online training and inference (E2-B):** registration, report-derived guidance,
   labeled Receiver batches, Stage-1 training, model promotion, TorchServe
   notification, and a successful inference request before the four-hour hard
   timeout.
3. **Directed outcome:** first target-PC hit, number of target-reaching
   executions, and exact target crash title. Neither functional check is
   expected to guarantee this outcome.

Always use a new `-bench` filename for each manager launch, retain useful
evidence, and remove only resources created by that run.

### 2.4 Train and Inspect the Model

The current implementation uses the paper's fixed three-stage online
curriculum:

- Stage 1: Unreachable versus Reachable;
- Stage 2: Unreachable versus Shallow versus Deep;
- Stage 3: exact deepest-waypoint classes.

The classifier keeps a common output structure across stages and records a
training manifest for checkpoint compatibility and promotion decisions. Dense
Stage 3 is the stable default. The optional sparse-adaptive profile maps
temporarily unsupported exact classes to `Reach_Other`; it remains an explicit
engineering profile rather than a replacement for Stage 2. See
[`filter/train_v2.py`](filter/train_v2.py) and
[`docs/pretrain_syzencoder.md`](docs/pretrain_syzencoder.md).

Released resources:

- [SyzEncoder](https://huggingface.co/zzra1n/SyzEncoder): the continued-pretrained
  syz-program encoder used by the online classifier;
- [SyzPilot-dataset](https://huggingface.co/datasets/zzra1n/SyzPilot-dataset):
  the released dataset for model and artifact research;
- [Zenodo artifact](https://doi.org/10.5281/zenodo.22874328): the versioned
  source-only evaluation package.

## 3 Method Components

- **Report-derived waypoints:** sanitizes crash traces, separates stages, and
  resolves instrumented PCs from the exact target kernel.
- **Cold-start syscall guidance:** combines lightweight report/path analysis
  with an authenticated generic `syz-imagegen` seed catalog for seed-only
  `NoGenerate` syz calls.
- **Online reachability learning:** streams exactly-one-hot reachability labels,
  trains a stage-aware classifier, and promotes only compatible checkpoints.
- **Attribution-guided mutation:** converts token attribution and reaching
  program patterns into syscall weights and sequence templates.
- **Additive directed corpus:** optionally retains bounded target-relevant
  programs as ordinary mutation sources without replacing Syzkaller's native
  signal-based corpus admission or persistence.

## 4 Reproducibility and Evaluation Boundaries

- The normal workflow may use the target bug report and report-derived
  waypoints, but never the target PoC or reproducer as fuzzing input.
- Target-PC coverage and an exact target crash are reported separately.
- A functional smoke test establishes component interoperability; it does not
  establish a directed-fuzzing speedup.
- Performance comparisons require matched kernels, VM resources, CPU affinity,
  runtime, repetitions, and measurement instrumentation.
- Model, tokenizer, kernel, waypoint, and label-schema compatibility must be
  checked before replaying a historical checkpoint.

## 5 Citation

The citation entry will be added when the paper is publicly available. Until
then, please cite the repository and the versioned Zenodo artifact DOI.

## 6 License

SyzPilot project code is released under the Apache License 2.0. Third-party
components, upstream patches, models, datasets, and benchmark inputs retain
their respective licenses; consult their source pages before redistribution.
