# Fuzzer source patches

## SyzPilot-Fuzzer

`SyzPilot-fuzzer.diff` contains the SyzPilot changes against the **exact**
upstream Syzkaller commit
`6e83b42dcfcd13c3b8e0d5c803cdcc424c0fbff9`. Use a clean clone and
apply from the Syzkaller root. Do not apply it to an arbitrary newer
Syzkaller revision, since its Go APIs and program mutation semantics change.
The current patch is based on SyzPilot-Fuzzer `4ed993a` (`v0.0.13`)
and includes the authenticated generic seed catalog, additive cold-start seed
injection, and the opt-in directed corpus used by the case-25 mini-benchmark.
The v0.0.13 follow-up changes only public developer-tool hygiene; it does not
change the v0.0.12 fuzzing runtime semantics. The public artifact copy only
normalizes the main repository name in documentation embedded by the patch.

```bash
git clone https://github.com/google/syzkaller.git SyzPilot-fuzzer
cd SyzPilot-fuzzer
git checkout --detach 6e83b42dcfcd13c3b8e0d5c803cdcc424c0fbff9
git apply --check /path/to/SyzPilot/fuzzer/SyzPilot-fuzzer.diff
git apply /path/to/SyzPilot/fuzzer/SyzPilot-fuzzer.diff
make syzpilot
make -j4 TARGETOS=linux TARGETARCH=amd64
```

The build needs Go 1.24.8, `protoc`, the pinned protobuf Go generators,
network access for the pinned googleapis checkout, and the ordinary Syzkaller
Linux build dependencies. [`docker/Dockerfile.fuzzer`](../docker/Dockerfile.fuzzer)
implements these steps against a published base image. The SyzPilot-Fuzzer
repository's initial import (`872788c`) omitted several upstream CI and test
fixture files, including `pkg/mgrconfig/testdata/*.cfg`; this patch is meant
for **the official upstream checkout**, not that incomplete import. The
functional workflow uses report-derived guidance only; target PoC injection
is reserved for offline oracle comparisons and is not part of the artifact
exercise.

## External baseline patches

`SyzDirect.diff` and `MOCK.diff` archive the baseline-specific changes used by
our evaluation. They are independent of `SyzPilot-fuzzer.diff`: apply each one
only to its own source tree, never to Syzkaller or SyzPilot-Fuzzer.

## SyzDirect

The original [`seclab-fudan/SyzDirect`](https://github.com/seclab-fudan/SyzDirect)
artifact did not run successfully in our evaluation environment without
additional fixes. `SyzDirect.diff` targets the exact SyzDirect commit
`02c9a6504a757e6cec0f10202624d175aa474d94` and preserves the compatibility
and measurement changes used by our baseline runs. In particular, it:

- makes the kernel/KCOV preparation less brittle and replaces the interactive
  `oldconfig` invocation with `olddefconfig`;
- guards several LLVM-IR extractors against missing initializers, unexpected
  struct layouts, non-constant operands, and out-of-range fields;
- reduces analyzer build pressure by disabling release debug information and
  limits the embedded Syzkaller target registrations to Linux;
- fixes the multi-case run counter/HTTP-port allocation and supplies the
  Python dependencies required by the runner; and
- records target-hit input and execution counts needed by our evaluation.

Apply the patch from a clean checkout of that revision:

```bash
git clone https://github.com/seclab-fudan/SyzDirect.git
cd SyzDirect
git checkout --detach 02c9a6504a757e6cec0f10202624d175aa474d94

git apply --check /path/to/SyzPilot/fuzzer/SyzDirect.diff
git apply /path/to/SyzPilot/fuzzer/SyzDirect.diff

python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r source/syzdirect/requirements.txt
```

Then configure the machine-specific paths and benchmark entries described by
SyzDirect's `source/README.md`, build its function-model and kernel-analysis
components, prepare each target kernel, and launch the runner from the patched
tree. SyzDirect performs substantial per-kernel and per-target static analysis;
do not treat patch application alone as a completed build or functional test.
The patch has been checked with `git apply --check` against the revision above.

## MOCK

`MOCK.diff` archives the adjustments used with the Healer-based
[`m0ck1ng/mock`](https://github.com/m0ck1ng/mock) baseline in our experiments.
It expects the same pre-patch experimental MOCK checkout from which the diff
was produced; that exact source revision is not distributed in this repository,
and the patch is not intended for an arbitrary MOCK or Healer revision.

From a compatible checkout, validate before applying and then follow MOCK's
normal build and launch procedure:

```bash
cd /path/to/compatible-mock-checkout
git apply --check /path/to/SyzPilot/fuzzer/MOCK.diff
git apply /path/to/SyzPilot/fuzzer/MOCK.diff
cargo build --release

python3 -m pip install 'numpy<1.24' django torch torchvision torchaudio
cd tools/model_manager
python3 manage.py runserver 127.0.0.1:8000
```

Start the built `healer` binary in another terminal with the guest image,
kernel image, and SSH key required by the target experiment. If
`git apply --check` fails, stop and recover the matching pre-patch source tree;
do not force this archival patch with `--reject` or `--3way`.
