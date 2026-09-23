# SyzPilot NDSS 2027 functional artifact

This package accompanies *SyzPilot: Steering Directed Kernel Fuzzing from
Reachability Prediction to Attribution-Guided Scheduling* (NDSS 2027
submission). It is a functional, scaled-down artifact, not the repeated
long-duration experiment behind the paper's performance tables. The submitted
code currently implements a two-stage online curriculum; the intermediate
third stage described in the paper is not included.

The source-only Zenodo package contains the Brain source,
`fuzzer/SyzPilot-fuzzer.diff`, `analyzer/KallGraph.diff` (optional), two
Dockerfiles, benchmark configs, and small runtime assets (SyzTokenizer and
Syzlang definitions). No compiled kernel, Linux source tree, guest disk,
SSH key, target PoC, or Git history is included. Its
`assets/case_36/configs` has the public report, title, kernel config, and
pinned commit. A GitHub source checkout does not include `assets/`: obtain
the tokenizer and Syzlang assets from the
[source-only Zenodo record](https://doi.org/10.5281/zenodo.22874328) before
running E2. **SyzEncoder weights are not in the Zenodo archive**. The model is hosted at
[zzra1n/SyzEncoder](https://huggingface.co/zzra1n/SyzEncoder) and downloaded
when the final Brain image is built with `INCLUDE_SYZENCODER=true`. The image
digest records the resulting immutable artifact; record the Hugging Face
commit revision used for the build. It derives from gated BigCode StarEncoder;
reviewers must observe
the [upstream model license](https://huggingface.co/bigcode/starencoder).
Other project code is under the root `LICENSE`.

Hardware: Linux x86-64 with KVM, Docker, at least 8 CPU cores and 16 GB host
RAM for the CPU-only E1 exercise. The one-guest fuzzer configuration uses
`vm.count=1`, `vm.cpu=2`, `vm.mem=4096`, and `procs=8`. Training and model
deployment additionally require **two visible NVIDIA GPUs** (GPU 0 for
training, GPU 1 for TorchServe/attribution); the first training GPU should
have at least 24 GiB free, plus the separately published SyzEncoder weights
in the Brain image. Without these GPUs, only E1 is documented as executable;
the E2 recipe below requires two GPUs. Check that TCP ports 48000,
37030--37034, 39836, and the
Receiver port range are free before launching. Do not run the vulnerable
kernel on the host. Building the kernel and guest requires network access,
root privileges for guest-image creation, and at least 30 GB of additional
free disk space; this preparation can take hours.

## Build the example kernel and guest (required for E1/E2)

For a GitHub checkout, unpack only the runtime assets from the source-only
Zenodo archive into this checkout, or copy them from an already unpacked
copy; [`mini-benchmark/README.md`](../mini-benchmark/README.md) gives an
exact `tar --zstd` example. Verify that `assets/models/SyzTokenizer_224w/tokenizer.json`,
`assets/syzlang/linux-amd64.json`, and `assets/syzlang/sys/linux` exist.
The Zenodo archive itself already contains `assets/case_36/configs`; in a
GitHub checkout, populate that directory from `benchmark/configs`:

```bash
mkdir -p assets/case_36/configs assets/kernels assets/guest run
cp benchmark/configs/case_36.{config,commit,title,report} assets/case_36/configs/
test -s assets/models/SyzTokenizer_224w/tokenizer.json
test -s assets/syzlang/linux-amd64.json
test -d assets/syzlang/sys/linux
git clone --filter=blob:none https://github.com/torvalds/linux.git run/linux-git-master
git -C run/linux-git-master cat-file -e \
  6207214a70bfaec7b41f39502353fd3ca89df68c^{commit}
test ! -e assets/kernels/case_36 && python3 experiments/compile_kernel.py \
  --workdir "$PWD/assets/kernels" \
  --linux-git-master "$PWD/run/linux-git-master" \
  --config artifact/case_36.compile.csv -j 8
test -s assets/kernels/case_36/vmlinux
test -s assets/kernels/case_36/arch/x86/boot/bzImage
```

`compile_kernel.py` removes an existing output case directory before a
fresh build. Never rerun it on an existing build containing valuable data.
See `experiments/README_compile_kernel.md` for prerequisites. Generate a
local Debian Bullseye guest using pinned Syzkaller `tools/create-image.sh`
on an isolated Linux host. Review the downloaded script before executing it:

```bash
mkdir -p run/guest-build
(
  set -e
  test ! -e run/guest-build/create-image.sh
  test ! -e run/guest-build/bullseye
  test ! -e run/guest-build/bullseye.img
  test ! -e assets/guest/bullseye.img
  test ! -e assets/guest/bullseye.id_rsa
  curl -fsSLo run/guest-build/create-image.sh \
    https://raw.githubusercontent.com/google/syzkaller/6e83b42dcfcd13c3b8e0d5c803cdcc424c0fbff9/tools/create-image.sh
  (cd run/guest-build && bash create-image.sh --distribution bullseye --seek 4096)
  cp run/guest-build/bullseye.img assets/guest/bullseye.img
  cp run/guest-build/bullseye.id_rsa assets/guest/bullseye.id_rsa
  chmod 600 assets/guest/bullseye.id_rsa
)
```

This requires `sudo`, `debootstrap`, `e2fsprogs`, `openssh-client`, and
QEMU/KVM. The script removes its own `bullseye/` working directory: use
only a dedicated empty build directory. The guest disk and private key
remain local and must not be committed or uploaded.

## E1: report-derived waypoints and target PCs (CPU only)

Build the Brain image, then run extraction against the locally built kernel.
The kernel mount is writable because the extractor may refresh local caches.

```bash
docker build -f docker/Dockerfile.brain -t syzpilot-brain:artifact .
mkdir -p run/case_36
docker run --rm --entrypoint python \
  -v "$PWD/assets:/artifact/assets" syzpilot-brain:artifact \
  analyzer/waypoints_extractor.py \
  -k /artifact/assets/kernels/case_36 \
  -t /artifact/assets/case_36/configs/case_36.title \
  -r /artifact/assets/case_36/configs/case_36.report \
  | tee run/case_36/waypoints.txt
python3 artifact/prepare_case36_config.py \
  --waypoints-output run/case_36/waypoints.txt \
  --template artifact/case_36.manager.cfg \
  --output run/case_36/manager.cfg
```

Success is an ordered, nonempty waypoint list and a final `[For
SyzPilot-fuzzer:]` JSON array. The helper copies these newly resolved PCs
into the manager config. Earlier PCs cannot be reused across independently
built kernels; keep each build's `vmlinux`/`bzImage` pair together. This
exercise uses only a bug report and compiled kernel, never the target PoC.

## E2: single-guest directed-fuzzing smoke test

Build the Fuzzer image from the patch and the self-contained Brain image with
the separately published SyzEncoder. Complete the kernel, guest, and E1
steps first. Use a host-network Brain/Fuzzer
pair on a dedicated KVM-capable machine. The commands below use paths inside
the images; no author-specific host path is required. Choose unused ports
before starting. Adjust the example CPU/GPU IDs to your available devices.
Keep the host firewalled: the Controller listens on all interfaces and its
default dashboard token is for testing only. Set a fresh `DASHBOARD_TOKEN`
if the dashboard is exposed. This is an illustrative one-run recipe, not a concurrency
or speed benchmark.

```bash
docker build -f docker/Dockerfile.fuzzer -t syzpilot-fuzzer:artifact .
docker build -f docker/Dockerfile.brain --build-arg INCLUDE_SYZENCODER=true \
  --build-arg SYZENCODER_REVISION=6140b0b46c81bb6428458fde5fd800a0e4a0687d \
  -t syzpilot-brain:artifact .
mkdir -p run/brain_receiver run/case_36
test -s run/case_36/manager.cfg
test -s assets/guest/bullseye.img
test -s assets/guest/bullseye.id_rsa
test ! -e run/case_36/case_36_ae.bench.log

docker run -d --name syzpilot-ae-brain --network host --gpus 'device=0,1' \
  -e SYZPILOT_TRAINING_GPU_IDS=0 \
  -e SYZPILOT_ATTRIBUTION_GPU_ID=1 -e SYZPILOT_INFERENCE_GPU_ID=1 \
  -e SYZPILOT_BASE_MODEL_PATH=/opt/syzpilot/models/SyzEncoder_224w_full/best_model \
  -e SYZPILOT_TOKENIZER_PATH=/artifact/assets/models/SyzTokenizer_224w \
  -e TOKENIZER_PATH=/artifact/assets/models/SyzTokenizer_224w \
  -e SYZPILOT_SYZKALLER_SYSLINUX=/artifact/assets/syzlang/sys/linux \
  -e SYZPILOT_SYZLANG_MANIFEST=/artifact/assets/syzlang/linux-amd64.json \
  -e SYZPILOT_GUIDANCE_REPORT_ROOTS=/artifact/assets/case_36/configs \
  -e SYZPILOT_GUIDANCE_KALLGRAPH_ROOTS=/artifact/assets/kallgraph \
  -v "$PWD/assets:/artifact/assets:ro" \
  -v "$PWD/run/brain_receiver:/opt/syzpilot/brain/receiver_data" \
  syzpilot-brain:artifact

for attempt in $(seq 1 30); do
  curl -fsS http://127.0.0.1:48000/health && break
  sleep 1
done
curl -fsS http://127.0.0.1:48000/health

test ! -e run/case_36/case_36_ae.bench.log && \
docker run -d --name syzpilot-ae-fuzzer --network host --device /dev/kvm \
  --cpuset-cpus=0-1 --entrypoint /root/SyzPilot-fuzzer/bin/syz-manager \
  -v "$PWD/assets:/artifact/assets:ro" \
  -v "$PWD/run/case_36:/artifact_runs/case_36" \
  syzpilot-fuzzer:artifact \
  -config /artifact_runs/case_36/manager.cfg -timeout 90m \
  -bench /artifact_runs/case_36/case_36_ae.bench.log
```

The manager config sets `network_mode=direct` and `callback_ip=127.0.0.1`,
which require both containers to use the host network namespace. Its
`report_path` is a **Brain-local** path under the configured trusted root.
`TOKENIZER_PATH` is additionally required by the TorchServe worker; the
controller uses `SYZPILOT_TOKENIZER_PATH` for training.
If `/dev/kvm` is inaccessible, provide a dedicated host with KVM permissions;
do not silently switch to a non-equivalent VM backend. With fewer than two
GPUs, follow only the documented E1 procedure; E2 requires separate training
and inference devices in this release.

Check the Brain's `/list_tasks`, the fuzzer's manager log and bench log, and
new `run/brain_receiver` batch files. Successful registration, advancing
execution counts, accepted guidance and labeled batches establish the
CPU-side data flow. Only an actual model-ready notification followed by a
successful inference request establishes online ML service functionality.
Training is data-triggered (stage 1 needs at least 1000 samples including
100 positive examples); a 90-minute smoke run may never reach that trigger.
Absence of a target hit or crash in this smoke test is not itself a component
failure. Keep the target crash title distinct from mere target-PC coverage.

```bash
curl -fsS http://127.0.0.1:48000/list_tasks
docker logs --tail 60 syzpilot-ae-fuzzer
docker logs --tail 60 syzpilot-ae-brain
```

Stop only the two containers launched above with `docker stop
syzpilot-ae-fuzzer syzpilot-ae-brain`. Retain `run/` for evaluation evidence;
it is ignored by Git. The stopped containers retain `docker logs`; after
exporting them, remove only these containers with `docker rm
syzpilot-ae-fuzzer syzpilot-ae-brain`. Use a fresh bench filename on every rerun because
`syz-manager` refuses to overwrite an existing bench log. The example does
not inject a target PoC, crash reproducer or PoC-derived mutation template.
PoC analysis utilities elsewhere in the repository are offline diagnostics
or oracle baselines, not inputs to E1/E2.

The two Dockerfiles have build-from-scratch instructions in
[`docker/README.md`](../docker/README.md). Prebuilt SyzPilot image tags should
be used only after their exact digests and availability have been verified;
the published `qgrain/kernel-fuzz:2204_v3` is a **base image**, not the
completed SyzPilot-Fuzzer image. To apply the source patches without Docker,
see [`fuzzer/README.md`](../fuzzer/README.md) and
[`analyzer/README.md`](../analyzer/README.md).
