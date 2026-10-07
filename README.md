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
> success criteria remain stable and available.

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

This repository implements the paper **"SyzPilot: Steering Directed Kernel
Fuzzing from Reachability Prediction to Attribution-Guided Scheduling"**
(under review). The paper PDF will be added after the review process permits
public release.

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

| Step | Expected time | Functional observation |
| --- | ---: | --- |
| Pull/build the unified image | 10--30 min | Python, Go, Java, and SyzEncoder available |
| Patch and build SyzPilot-Fuzzer | 5--20 min | Manager, executor, and execprog binaries |
| Build the case-36 kernel | 1--3 h, one time | Matching `vmlinux` and `bzImage` |
| E1: report-derived waypoint extraction | 5--15 min | Ordered waypoints and nonempty target PCs |
| E2: directed-fuzzing smoke run | 90 min + 5--15 min inspection | Registration, executions, labeled batches, guidance, and model service if data-ready |
| Preserve evidence and clean up | 5--10 min | Bench/log evidence retained; no orphan containers |

E1/E2 validate functionality rather than the paper's long-running performance
claims. A target-PC hit and an exact target crash are separate observations;
neither is required during the 90-minute smoke run.

### 1.1 Hardware and Software Requirements

For the CPU-only waypoint extraction exercise:

- Linux x86-64 with Docker;
- 8 or more CPU cores, 16 GiB RAM, and approximately 30 GiB free disk space;
- a local Linux checkout and the ability to compile a target kernel.

For the complete online fuzzing and ML workflow:

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
  --build-arg OCI_REVISION="$REVISION" \
  -t qgrain/syzpilot:ndss27-ae .
```

The complete artifact procedure is documented in
[`artifact/README.md`](artifact/README.md). It covers:

1. pulling or rebuilding the unified image;
2. compiling the pinned example kernel;
3. creating Brain/Fuzzer containers from the unified image;
4. extracting report-derived waypoint PCs;
5. running the component-level directed-fuzzing smoke test; and
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
docker run -d --name syzpilot-brain --network host \
  --gpus 'device=0,1' \
  -v /HOST/KERNELS:/root/kernels \
  -v /HOST/SYZPILOT-RUNS:/root/syzpilot-runs \
  -v /HOST/SYZPILOT-BRAIN-STATE/receiver_data:/root/SyzPilot/brain/receiver_data \
  -v /HOST/SYZPILOT-BRAIN-STATE/logs:/root/SyzPilot/brain/logs \
  -v /HOST/SYZPILOT-BRAIN-STATE/model_store:/root/SyzPilot/brain/model_store \
  qgrain/syzpilot:ndss27-ae sleep infinity

# CPU/KVM server
docker run -d --name syzpilot-fuzzer --network host --device /dev/kvm \
  --cpuset-cpus=0-1 \
  -v /HOST/KERNELS:/root/kernels \
  -v /HOST/SYZPILOT-RUNS:/root/syzpilot-runs \
  qgrain/syzpilot:ndss27-ae sleep infinity
```

For persistent development experiments outside Docker, set
`SYZPILOT_DATA_ROOT` to an experiment-specific directory, for example
`$HOME/datasets/SyzPilot/receiver_runs/<RUN_ID>/brain/receiver_data`. The
default remains `brain/receiver_data`.

The source checkout, Fuzzer patch, and compiled Syzlang assets in the image
form one tested snapshot. Use it unchanged for artifact evaluation:

```bash
docker exec syzpilot-brain bash -lc \
  'cd /root/SyzPilot && make -C brain'
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

### 2.2 Start the Brain

```bash
docker exec syzpilot-brain bash -lc \
  'cd /root/SyzPilot && scripts/verify_setup.sh brain'

docker exec -d syzpilot-brain bash -lc '
  cd /root/SyzPilot
  export SYZPILOT_DIRECT_ONLY=true
  export SYZPILOT_TRAINING_GPU_IDS=0
  export SYZPILOT_INFERENCE_GPU_ID=1
  export SYZPILOT_ATTRIBUTION_GPU_ID=1
  exec python brain/controller.py --host 0.0.0.0 --port 48000 \
    > /root/syzpilot-runs/brain.log 2>&1
'

curl -fsS http://BRAIN_HOST:48000/health
curl -fsS http://BRAIN_HOST:48000/list_tasks
```

Replace `BRAIN_HOST` with the GPU server's reachable private address. The
unified image defaults the model and tokenizer paths to its pinned
`/root/models/SyzEncoder` snapshot.

### 2.3 Run the Functional Mini-Benchmark

[`mini-benchmark/README.md`](mini-benchmark/README.md) provides source-only
metadata for cases 25, 21, and 36. It deliberately excludes compiled kernels,
VM images, target PoCs, and fixed PC addresses. Start with case 25 for a short
component exercise; use cases 21 or 36 for longer online-training or deeper
target-reach checks.

For the appendix's case-36 workflow, run the following from the Fuzzer host to
compile the pinned kernel only into a new output directory:

```bash
docker exec syzpilot-fuzzer bash -lc '
  cd /root/SyzPilot
  test ! -e /root/kernels/case_36
  python experiments/compile_kernel.py \
    --workdir /root/kernels \
    --linux-git-master /root/kernels/linux-git-master \
    --config mini-benchmark/compile_case_36.csv -j 8
'

docker exec syzpilot-brain bash -lc '
  mkdir -p /root/syzpilot-runs/case_36
  cd /root/SyzPilot
  python analyzer/waypoints_extractor.py \
    -k /root/kernels/case_36 \
    -t benchmark/configs/case_36.title \
    -r benchmark/configs/case_36.report \
    | tee /root/syzpilot-runs/case_36/waypoints.txt
'
```

Transfer `waypoints.txt` to the Fuzzer host when the roles do not share
storage, then generate the manager config from those fresh PCs. The two host
values must be mutually reachable private addresses.

```bash
export BRAIN_HOST=10.0.0.10
export FUZZER_HOST=10.0.0.20
docker exec -e BRAIN_HOST="$BRAIN_HOST" -e FUZZER_HOST="$FUZZER_HOST" \
  syzpilot-fuzzer bash -lc '
    cd /root/SyzPilot
    python scripts/prepare_functional_config.py \
      --case 36 \
      --waypoints-output /root/syzpilot-runs/case_36/waypoints.txt \
      --output /root/syzpilot-runs/case_36/manager.cfg \
      --brain-host "$BRAIN_HOST" --fuzzer-host "$FUZZER_HOST"
  '

curl -fsS "http://${BRAIN_HOST}:48000/health"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
docker exec syzpilot-fuzzer bash -lc "
  taskset -c 0-1 /root/fuzzers/SyzPilot-fuzzer/bin/syz-manager \
    -config /root/syzpilot-runs/case_36/manager.cfg \
    -timeout 90m \
    -bench /root/syzpilot-runs/case_36/case_36_${RUN_ID}.bench.log \
    2>&1 | tee /root/syzpilot-runs/case_36/fuzzer_${RUN_ID}.log
"
```

The base image supplies the disposable guest paths used by the generated
configuration. See [`artifact/README.md`](artifact/README.md) for kernel
checkout commands, distributed file placement, status checks, expected
observations, and safe cleanup.

Functional observations are separated into three levels:

1. **Component flow:** registration, VM execution, report-derived guidance,
   labeled Receiver batches, and clean shutdown.
2. **Online ML flow:** Stage-1 training, model promotion, TorchServe
   notification, and a successful inference request when sufficient positive
   data and GPU resources are available.
3. **Directed outcome:** first target-PC hit, number of target-reaching
   executions, and exact target crash title. A short smoke test is not
   expected to guarantee the third level.

Always use a new `-bench` filename for each manager launch, retain useful
evidence, and remove only resources created by that run.

### 2.4 Train and Inspect the Model

The current implementation uses a two-stage online curriculum:

- Stage 1: binary classification of unreachable versus any reached waypoint;
- Stage 2: multi-class prediction of the deepest reached waypoint.

The classifier keeps a common output structure across stages and records a
training manifest for checkpoint compatibility and promotion decisions. See
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
