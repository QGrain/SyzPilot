# SyzPilot functional mini-benchmark

This source-only subset offers three report-grounded Linux targets for short
integration checks. It contains the case configs, reports, titles, commit IDs,
and hashes, but **no target PoCs, compiled kernels, guest disks, or fixed PC
addresses**. A target-PC hit and a matching target crash are different outcomes;
neither is guaranteed by a short run. Do not use a target PoC or reproducer as
a fuzzing seed or guidance input.

| Case | Purpose | Historical observation, not an acceptance threshold |
| --- | --- | --- |
| `case_25` (F2FS) | First choice for fast registration, report guidance, labeled data, and Stage-1 training | Guided data reached the Stage-1 threshold within minutes; one prior two-VM run deployed a Stage-1 model after about 15.5 minutes. |
| `case_21` (cfg80211) | Full online training/deployment and rfilter regression | One prior four-VM run organically trained and deployed Stage 1 in about 101 minutes. |
| `case_36` (RDS) | Deeper target-PC reach and guided scheduling check | One prior guided run reached the final PC after about 31 minutes; no exact target crash was seen. VM instability and GPU contention have affected other runs. |

The evidence above is recorded in local, Git-ignored `agent_analysis/` run
summaries and is not independently verifiable from this source checkout. It
does not establish a statistical speedup or an AEC pass threshold. Start with
`case_25`; use the other
two only when their longer runtime or deeper-reach behavior is relevant. The
mini-benchmark was added after the initial Zenodo v0.1.0 snapshot and does not
retroactively change files already deposited there.

## Prepare one case

Run from the SyzPilot repository root. A GitHub checkout excludes the ignored
`assets/` directory: first obtain `assets/models/SyzTokenizer_224w` and
`assets/syzlang` from the
[source-only Zenodo record](https://doi.org/10.5281/zenodo.22874328).
For example, extract those two directories from the downloaded archive into
an otherwise empty scratch directory, then copy them into this checkout:

```bash
mkdir -p run/zenodo-assets assets
(
  set -e
  test ! -e assets/models
  test ! -e assets/syzlang
  test ! -e run/zenodo-assets/SyzPilot-artifact/assets/models
  test ! -e run/zenodo-assets/SyzPilot-artifact/assets/syzlang
  tar --zstd -xf /path/to/SyzPilot-source-only.tar.zst \
    -C run/zenodo-assets \
    SyzPilot-artifact/assets/models SyzPilot-artifact/assets/syzlang
  cp -a run/zenodo-assets/SyzPilot-artifact/assets/models assets/
  cp -a run/zenodo-assets/SyzPilot-artifact/assets/syzlang assets/
  test -s assets/models/SyzTokenizer_224w/tokenizer.json
  test -s assets/syzlang/linux-amd64.json
  test -d assets/syzlang/sys/linux
)
```

Use a Linux Git checkout that contains the selected commit and follow
`experiments/README_compile_kernel.md` for build dependencies. The one-row
`compile_case_XX.csv` files prevent the batch
compiler from unnecessarily building all three cases. **The compiler deletes
an existing output case directory before a fresh build**; choose a new output
directory or check that the case directory does not exist before invoking it.

```bash
CASE=25
mkdir -p assets/kernels assets/guest "assets/case_${CASE}/configs" "run/case_${CASE}"
test ! -e "assets/kernels/case_${CASE}" && python3 experiments/compile_kernel.py \
  --workdir "$PWD/assets/kernels" \
  --linux-git-master /path/to/linux-git-master \
  --config "mini-benchmark/compile_case_${CASE}.csv" -j 8
cp "mini-benchmark/configs/case_${CASE}.report" \
   "mini-benchmark/configs/case_${CASE}.title" \
   "assets/case_${CASE}/configs/"
```

Create a local Bullseye guest image and SSH key under `assets/guest/` using
the pinned upstream `tools/create-image.sh` procedure in
`artifact/README.md`. The example manager expects `bullseye.img` and
`bullseye.id_rsa`; no guest credential is included in this repository.

Resolve waypoint PCs **from the newly compiled `vmlinux`** using the Brain
image, then generate the manager config with host Python 3 (standard library
only). The parser rejects empty, duplicate, zero, or malformed PCs. The
template contains no historical PCs.

```bash
docker build -f docker/Dockerfile.brain -t syzpilot-brain:artifact .
docker run --rm --entrypoint python \
  -v "$PWD/assets:/artifact/assets" syzpilot-brain:artifact \
  analyzer/waypoints_extractor.py \
  -k "/artifact/assets/kernels/case_${CASE}" \
  -t "/artifact/assets/case_${CASE}/configs/case_${CASE}.title" \
  -r "/artifact/assets/case_${CASE}/configs/case_${CASE}.report" \
  | tee "run/case_${CASE}/waypoints.txt"
python3 mini-benchmark/prepare_manager.py \
  --case "$CASE" \
  --waypoints-output "run/case_${CASE}/waypoints.txt" \
  --output "run/case_${CASE}/manager.cfg"
```

The generated config uses one QEMU guest (`vm.cpu=2`, `vm.mem=4096`,
`procs=8`), direct Brain registration on `127.0.0.1:48000`, and a separate
manager HTTP port per case. Run **one case at a time** with this recipe;
check that port 48000, the selected manager port, and TorchServe ports are
free first. Change the example host CPU/GPU assignments to available devices.
For case 25 only, the generated config enables the Fuzzer's versioned generic
`NoGenerate` seed catalog and bounded automatic resource closure. The catalog
is generated from the pinned upstream `syz-imagegen` assets rather than naming
F2FS in manager code; the matching guidance selects its 16 authenticated
`syz_mount_image$f2fs` programs. The manager leaves Syzkaller's native candidate
flow unchanged and appends these programs once; confirm the native, injected,
reused, and rejected counts in the manager log. The assets are not derived from
the target PoC, reproducer, or crash-specific values. Keep both catalog/seed
paths and the closure setting identical across all arms of a component-level
A/B, unless this cold-start mechanism is explicitly the tested treatment.

Case 25 also opts into the bounded, runtime-only directed corpus. It retains
authenticated cold-start seeds and successfully completed programs that reach
a configured waypoint, then sources 20% of ordinary mutations from this pool
through the same mutator used by the native corpus. A selected directed
mutation preserves at least one existing guidance anchor with 95% probability;
the remaining 5% deliberately permits the unmodified mutator to delete every
anchor. This pool neither changes native corpus admission nor persists into
`corpus.db`. Cases 21 and 36 remove this opt-in field to preserve their current
behavior.

The E1 waypoint extraction above needs no GPU; the E2 launch below requires
two GPUs even if online training has not yet triggered. Keep the host-network
test machine firewalled and set a new `DASHBOARD_TOKEN` if exposing the
Controller dashboard.

```bash
docker build -f docker/Dockerfile.fuzzer -t syzpilot-fuzzer:artifact .
docker build -f docker/Dockerfile.brain --build-arg INCLUDE_SYZENCODER=true \
  --build-arg SYZENCODER_REVISION=6140b0b46c81bb6428458fde5fd800a0e4a0687d \
  -t syzpilot-brain:artifact .
mkdir -p run/brain_receiver "run/case_${CASE}"
test -s "run/case_${CASE}/manager.cfg"
test -s assets/guest/bullseye.img
test -s assets/guest/bullseye.id_rsa
test ! -e "run/case_${CASE}/case_${CASE}.bench.log"

docker run -d --name syzpilot-mini-brain --network host \
  --gpus 'device=0,1' \
  -e SYZPILOT_TRAINING_GPU_IDS=0 \
  -e SYZPILOT_ATTRIBUTION_GPU_ID=1 -e SYZPILOT_INFERENCE_GPU_ID=1 \
  -e SYZPILOT_BASE_MODEL_PATH=/opt/syzpilot/models/SyzEncoder_224w_full/best_model \
  -e SYZPILOT_TOKENIZER_PATH=/artifact/assets/models/SyzTokenizer_224w \
  -e TOKENIZER_PATH=/artifact/assets/models/SyzTokenizer_224w \
  -e SYZPILOT_SYZKALLER_SYSLINUX=/artifact/assets/syzlang/sys/linux \
  -e SYZPILOT_SYZLANG_MANIFEST=/artifact/assets/syzlang/linux-amd64.json \
  -e "SYZPILOT_GUIDANCE_REPORT_ROOTS=/artifact/assets/case_${CASE}/configs" \
  -v "$PWD/assets:/artifact/assets:ro" \
  -v "$PWD/run/brain_receiver:/opt/syzpilot/brain/receiver_data" \
  syzpilot-brain:artifact
curl -fsS http://127.0.0.1:48000/health

test ! -e "run/case_${CASE}/case_${CASE}.bench.log" && \
docker run -d --name syzpilot-mini-fuzzer --network host --device /dev/kvm \
  --cpuset-cpus=0-1 --entrypoint /root/SyzPilot-fuzzer/bin/syz-manager \
  -v "$PWD/assets:/artifact/assets:ro" \
  -v "$PWD/run/case_${CASE}:/artifact_runs/case_${CASE}" \
  syzpilot-fuzzer:artifact \
  -config "/artifact_runs/case_${CASE}/manager.cfg" -timeout 90m \
  -bench "/artifact_runs/case_${CASE}/case_${CASE}.bench.log"
```

The 90-minute timeout may be too short for online model training on some
cases; it is a smoke test, not a speed benchmark. Check `/list_tasks`, both
container logs, the bench log, and `run/brain_receiver` for progress. Keep
logs/results worth analyzing. On completion, stop only these named containers:
`docker stop syzpilot-mini-fuzzer syzpilot-mini-brain`. They remain as stopped
containers so `docker logs` is available. After exporting the logs, remove
only these containers with `docker rm syzpilot-mini-fuzzer syzpilot-mini-brain`.
Use a new unique
`-bench` filename on every manager restart; syz-manager refuses to overwrite
an existing one. No PoC is injected into the run.

## Functional observations

For a CPU-side smoke check, confirm Controller registration, a running QEMU
guest, advancing execution counts, accepted report-derived syscall guidance,
and labeled batches delivered to the Brain Receiver. With the separately
distributed SyzEncoder and two available GPUs, additionally check the
data-triggered Stage-1 training, model deployment notification, and at least
one successful inference request. Stage-1 activation is conditional on enough
positive samples and may not occur in a short run. Record target-PC coverage
and exact crash title separately; no crash is required for the component
functional check. Retain useful bench, Receiver, and crash logs and remove
only the work directories created for the test.
