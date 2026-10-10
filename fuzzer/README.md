# Fuzzer source patches

## SyzPilot-Fuzzer

`SyzPilot-fuzzer.diff` contains the SyzPilot changes against the **exact**
upstream Syzkaller commit
`6e83b42dcfcd13c3b8e0d5c803cdcc424c0fbff9`. Use a clean clone and
apply from the Syzkaller root. Do not apply it to an arbitrary newer
Syzkaller revision, since its Go APIs and program mutation semantics change.
The current patch corresponds to the reviewed SyzPilot-Fuzzer `v0.2.0`
source release (`212601e`).

```bash
git clone https://github.com/google/syzkaller.git SyzPilot-fuzzer
cd SyzPilot-fuzzer
git checkout --detach 6e83b42dcfcd13c3b8e0d5c803cdcc424c0fbff9
git apply --check /path/to/SyzPilot/fuzzer/SyzPilot-fuzzer.diff
git apply /path/to/SyzPilot/fuzzer/SyzPilot-fuzzer.diff
make syzpilot
make -j4 TARGETOS=linux TARGETARCH=amd64
```

The build needs Go 1.24.8, `protoc`, protobuf development headers, the pinned
protobuf Go generators, network access for the pinned googleapis checkout,
and the ordinary Syzkaller Linux build dependencies.
[`docker/Dockerfile`](../docker/Dockerfile) provides
the shared Brain/Fuzzer environment; `scripts/patch_fuzzers.sh SyzPilot`
implements the pinned clone-and-apply step. The SyzPilot-Fuzzer
repository's initial import (`872788c`) omitted several upstream CI and test
fixture files, including `pkg/mgrconfig/testdata/*.cfg`; this patch is meant
for **the official upstream checkout**, not that incomplete import. The
functional workflow uses report-derived guidance only; target PoC injection
is reserved for offline oracle comparisons and is not part of the artifact
exercise.

The performance profile keeps the Collector training queue in memory and
leaves `SyzPilot.durable_training_wal` disabled by default. Enable it only for
an explicit manager-process recovery experiment. The predictor-only ablation
sets `SyzPilot.enable_online_guidance=false`: it retains cold-start guidance,
online training, TorchServe model hot replacement, and reach filtering, while
disabling only post-model sequence/attribution guidance refreshes. For component
ablations, keep online guidance enabled and set `enable_sequence_guidance` or
`enable_attribution_guidance` independently. Use these fields with the current
public SyzPilot release and upgrade the Brain and Fuzzer together. All three
guidance switches default to the normal full-pipeline behavior when omitted.

The reachability filter accepts the fixed three-stage Brain protocol: binary
Unreachable/Reachable predictions, ternary Unreachable/Shallow/Deep
predictions, and exact-waypoint predictions. Sparse-adaptive Stage 3 may also
return `Reach_Other` for reached samples whose exact waypoint is inactive. The
filter treats that label as reached but does not grant an exact-waypoint smash
bonus. Brain and Fuzzer releases must therefore be upgraded together.

## External baseline patches

The remaining patches archive compatibility changes used to reproduce
independent directed-fuzzing baselines. They are separate from the
SyzPilot-Fuzzer patch above.

## SyzDirect

`SyzDirect.diff` targets the upstream
[`seclab-fudan/SyzDirect`](https://github.com/seclab-fudan/SyzDirect) repository
at commit `02c9a6504a757e6cec0f10202624d175aa474d94`. It contains build fixes,
runtime configuration adjustments, and experiment instrumentation used by the
SyzPilot evaluation.

Apply it from the SyzDirect repository root:

```bash
git checkout 02c9a6504a757e6cec0f10202624d175aa474d94
git apply /path/to/SyzPilot/fuzzer/SyzDirect.diff
```

The patch passes `git apply --check` against that revision.

## MOCK

`MOCK.diff` targets the Healer-based MOCK baseline used in the evaluation. The
original source checkout and exact base revision are not distributed in this
repository, so this file is retained as an archival patch rather than a
standalone reproducible package. Apply and validate it against the original
experiment checkout before use.
