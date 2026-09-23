# SyzPilot: Steering Directed Kernel Fuzzing Beyond Reachability Prediction with Attribution Guidance

<div align="center">
<a href="https://doi.org/10.5281/zenodo.22874328"><img alt="Artifact Zenodo" src="https://zenodo.org/badge/DOI/10.5281/zenodo.22874328.svg"/></a>
<a href="https://huggingface.co/zzra1n/SyzEncoder"><img alt="SyzEncoder" src="https://img.shields.io/badge/%F0%9F%A4%97%20Model-SyzEncoder-ffc107"/></a>
<a href="https://huggingface.co/datasets/zzra1n/SyzPilot-dataset"><img alt="SyzPilot Dataset" src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-SyzPilot--dataset-ffc107"/></a>
<a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache--2.0-blue.svg"/></a>
</div>

> [!NOTE]
> **For artifact evaluation reviewers:** this is the maintained source
> repository for the SyzPilot artifact. We may continue improving
> documentation and portability during the evaluation period, while the
> submitted functional workflow, versioned source package, and observable
> success criteria remain stable and available.

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

**Quick glance.** SyzPilot separates GPU-intensive learning from CPU-intensive
kernel fuzzing. The Brain coordinates target analysis, training, deployment,
and guidance; one or more Fuzzers execute programs in isolated kernel VMs and
stream reachability observations back to the Brain. The normal functional
workflow derives guidance from the bug report, kernel, online executions, and
model attribution. It does **not** use the target PoC or reproducer as fuzzing
input.

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
│       └── native corpus + opt-in directed runtime corpus  │
└───────────────────────────────────────────────────────────┘
```

## Project Structure

```text
.
├── analyzer/         # Report/waypoint extraction and static guidance
├── artifact/         # Functional artifact guide and portable example
├── benchmark/        # Public non-PoC benchmark metadata and reports
├── brain/            # Controller, Receiver, TorchServe, and lifecycle logic
├── common/           # Shared curriculum and label contracts
├── docker/           # Reproducible Brain and Fuzzer image recipes
├── experiments/      # Kernel builds, experiment launch, and result analysis
├── filter/           # SyzEncoder, classifier training, and attribution
├── fuzzer/           # SyzPilot-Fuzzer and archived baseline source patches
├── mini-benchmark/   # Three source-only functional benchmark cases
├── requirements/     # Brain, Fuzzer, and optional agent dependencies
├── scripts/          # Dataset and experiment utilities
└── tests/            # CPU-side regression and contract tests
```

## 1 Setup

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

The complete artifact procedure is documented in
[`artifact/README.md`](artifact/README.md). It covers:

1. obtaining the small runtime assets from Zenodo;
2. compiling the pinned example kernel and creating a local guest image;
3. building the Brain and patched Fuzzer images;
4. extracting report-derived waypoint PCs;
5. running the component-level directed-fuzzing smoke test;
6. checking observable outputs and safely cleaning owned containers.

```bash
git clone https://github.com/QGrain/SyzPilot.git
cd SyzPilot

docker build -f docker/Dockerfile.brain -t syzpilot-brain:artifact .
docker build -f docker/Dockerfile.fuzzer -t syzpilot-fuzzer:artifact .
```

The Docker recipes do not embed target kernels, VM disks, private keys, PoCs,
or experiment logs. The optional Brain model build downloads SyzEncoder from
Hugging Face at a caller-selected revision.

### 1.3 Setup from Source

```bash
conda create -n syzpilot python=3.11
conda activate syzpilot
pip install -r requirements/requirements-brain.txt \
  --extra-index-url https://download.pytorch.org/whl/cu121
bash requirements/required_packages.sh
make -C brain
```

Build the Fuzzer from the exact upstream Syzkaller revision and reviewed
patch as described in [`fuzzer/README.md`](fuzzer/README.md). Build the optional
KallGraph component using [`analyzer/README.md`](analyzer/README.md).

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

For the resumable multi-case extraction and evaluation workflow, see
[`docs/WAYPOINTS_TESTING.md`](docs/WAYPOINTS_TESTING.md).

### 2.2 Start the Brain

```bash
export SYZPILOT_BASE_MODEL_PATH=/path/to/SyzEncoder
export SYZPILOT_TOKENIZER_PATH=/path/to/SyzTokenizer
export SYZPILOT_SYZKALLER_SYSLINUX=/path/to/syzkaller/sys/linux
python brain/controller.py --host 127.0.0.1 --port 48000

curl -fsS http://127.0.0.1:48000/health
curl -fsS http://127.0.0.1:48000/list_tasks
```

Use explicit environment variables for local paths and GPU assignments. The
Docker workflow supplies portable in-container defaults and is recommended
for artifact evaluation.

### 2.3 Run the Functional Mini-Benchmark

[`mini-benchmark/README.md`](mini-benchmark/README.md) provides source-only
metadata for cases 25, 21, and 36. It deliberately excludes compiled kernels,
VM images, target PoCs, and fixed PC addresses. Start with case 25 for a short
component exercise; use cases 21 or 36 for longer online-training or deeper
target-reach checks.

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

## 5 Related Repositories

- [QGrain/SyzPilot-fuzzer](https://github.com/QGrain/SyzPilot-fuzzer): the
  development repository for the modified Syzkaller runtime;
- [google/syzkaller](https://github.com/google/syzkaller): upstream kernel
  fuzzer, pinned by the artifact patch;
- [QGrain/kernel-fuzz-docker-images](https://github.com/QGrain/kernel-fuzz-docker-images):
  base container build recipes used by our kernel-fuzzing projects.

## 6 Citation

The citation entry will be added when the paper is publicly available. Until
then, please cite the repository and the versioned Zenodo artifact DOI.

## 7 License

SyzPilot project code is released under the Apache License 2.0. Third-party
components, upstream patches, models, datasets, and benchmark inputs retain
their respective licenses; consult their source pages before redistribution.
