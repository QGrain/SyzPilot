# SyzPilot NDSS 2027 functional artifact

This artifact accompanies *SyzPilot: Steering Directed Kernel Fuzzing from
Reachability Prediction to Attribution-Guided Scheduling*. The evaluation
targets the **Available** and **Functional** badges. It does not attempt to
repeat the paper's multi-target, multi-run performance evaluation.

The normal functional workflow accepts a public bug report and matching kernel
build. It never uses the target PoC or reproducer as a fuzzing seed or guidance
input. Reaching a target PC and reproducing the corresponding crash are
reported separately.

## Evaluation budget

| Activity | Approximate wall-clock time |
| --- | ---: |
| Pull or build `qgrain/syzpilot:ndss27-ae` | 10--30 min |
| Patch and build SyzPilot-Fuzzer | 5--20 min |
| First case-36 kernel build | 1--3 h |
| E1 waypoint extraction | 5--15 min |
| E2 directed-fuzzing smoke run | 90 min |
| Inspect evidence and clean up | 5--15 min |

The kernel build is reusable when its source, config, compiler, `vmlinux`, and
`bzImage` remain unchanged. Network and hardware speed can substantially alter
the estimates.

## Requirements

- Linux x86-64, Docker, and at least 8 CPU cores.
- A Fuzzer host with KVM. The smoke configuration uses one 2-vCPU, 4-GiB guest
  and `procs=8`.
- A Brain host with two visible NVIDIA GPUs for complete ML validation:
  approximately 24 GiB free VRAM for training and 12 GiB for serving/IG.
- Private two-way connectivity between the Brain and Fuzzer. The Fuzzer must
  reach Brain TCP 48000, 31001--31999, and 37030--37034. The Brain must reach
  the configured manager callback port.
- Approximately 30 GiB of free workspace beyond the container image.

Run vulnerable kernels only inside isolated VMs. Do not expose Controller,
Receiver, TorchServe, or manager callback ports to the public Internet.

## A0: Create the two roles

The same image is used on both hosts. It deliberately starts no service by
default. Its environment selects direct-only Brain networking and supplies
the Linux/amd64 Syzlang manifest and descriptions matched to the pinned
Fuzzer patch.

```bash
docker pull qgrain/syzpilot:ndss27-ae
mkdir -p /HOST/SYZPILOT-BRAIN-STATE/{receiver_data,logs,model_store}

# GPU host
docker run -d --name syzpilot-brain --network host \
  --gpus 'device=0,1' \
  -v /HOST/KERNELS:/root/kernels \
  -v /HOST/SYZPILOT-RUNS:/root/syzpilot-runs \
  -v /HOST/SYZPILOT-BRAIN-STATE/receiver_data:/root/SyzPilot/brain/receiver_data \
  -v /HOST/SYZPILOT-BRAIN-STATE/logs:/root/SyzPilot/brain/logs \
  -v /HOST/SYZPILOT-BRAIN-STATE/model_store:/root/SyzPilot/brain/model_store \
  qgrain/syzpilot:ndss27-ae sleep infinity

# CPU/KVM host
docker run -d --name syzpilot-fuzzer --network host --device /dev/kvm \
  --cpuset-cpus=0-1 \
  -v /HOST/KERNELS:/root/kernels \
  -v /HOST/SYZPILOT-RUNS:/root/syzpilot-runs \
  qgrain/syzpilot:ndss27-ae sleep infinity
```

If the prebuilt tag is unavailable, build `docker/Dockerfile` as described in
the root README and [`docker/README.md`](../docker/README.md). For evaluation,
keep the image's source checkout, Fuzzer patch, and compiled Syzlang assets at
their matched revision. Rebuild the image to adopt a newer public revision.

## A1: Patch and build the Fuzzer

Run from the Fuzzer host; the command enters the named container explicitly:

```bash
docker exec syzpilot-fuzzer bash -lc '
  cd /root/SyzPilot
  scripts/patch_fuzzers.sh SyzPilot
  cd /root/fuzzers/SyzPilot-fuzzer
  make -j"$(nproc)"
  /root/SyzPilot/scripts/verify_setup.sh fuzzer
'
```

The patch script obtains exact upstream Syzkaller commit
`6e83b42dcfcd13c3b8e0d5c803cdcc424c0fbff9` over HTTPS and applies
`fuzzer/SyzPilot-fuzzer.diff`.

## A2: Build case 36

From the Fuzzer host, clone Linux under the mounted kernel workspace and build
only into a new case directory. `compile_kernel.py` removes an existing case
output before a fresh build, so both explicit guards are important.

```bash
docker exec syzpilot-fuzzer bash -lc '
  test ! -e /root/kernels/linux-git-master
  git clone --filter=blob:none https://github.com/torvalds/linux.git \
    /root/kernels/linux-git-master
  git -C /root/kernels/linux-git-master fetch origin \
    6207214a70bfaec7b41f39502353fd3ca89df68c

  cd /root/SyzPilot
  test ! -e /root/kernels/case_36
  python experiments/compile_kernel.py \
    --workdir /root/kernels \
    --linux-git-master /root/kernels/linux-git-master \
    --config mini-benchmark/compile_case_36.csv -j 8
  test -s /root/kernels/case_36/vmlinux
  test -s /root/kernels/case_36/arch/x86/boot/bzImage
'
```

The unified image inherits a disposable Trixie guest template at
`/root/images/image-template/{disk.img,disk.id_rsa}`. The generated manager
configuration uses these paths. If the two roles are on different machines,
place the exact same case-36 build at `/root/kernels/case_36` on both hosts.

## E1: Report-derived waypoints

Run from the Brain host:

```bash
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

Success is an ordered, nonempty waypoint list followed by a
`[For SyzPilot-fuzzer:]` JSON array. PCs must always be regenerated from the
exact `vmlinux` used by the VM. Transfer `waypoints.txt` to the Fuzzer host if
the two roles do not share the run directory.

Create the Fuzzer-local manager config. Replace both example addresses with
mutually reachable private IPs:

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
```

## E2: Directed-fuzzing smoke test

Verify and start the Brain explicitly:

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
```

On the Fuzzer host, verify connectivity and launch one 90-minute run:

```bash
curl -fsS "http://${BRAIN_HOST}:48000/health"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
docker exec -d syzpilot-fuzzer bash -lc "
  taskset -c 0-1 /root/fuzzers/SyzPilot-fuzzer/bin/syz-manager \
    -config /root/syzpilot-runs/case_36/manager.cfg \
    -timeout 90m \
    -bench /root/syzpilot-runs/case_36/case_36_${RUN_ID}.bench.log \
    > /root/syzpilot-runs/case_36/fuzzer_${RUN_ID}.log 2>&1
"
```

Inspect progress:

```bash
curl -fsS "http://${BRAIN_HOST}:48000/list_tasks" | python -m json.tool
docker exec syzpilot-fuzzer bash -lc \
  'tail -n 60 /root/syzpilot-runs/case_36/fuzzer_*.log'
docker exec syzpilot-brain bash -lc \
  'tail -n 60 /root/syzpilot-runs/brain.log'
```

Functional observations are progressive:

1. registration, a running QEMU guest, and increasing execution counts;
2. report-derived guidance and labeled Receiver batches; and
3. when sufficient positive samples arrive, Stage-1 training, model promotion,
   TorchServe notification, and successful prediction.

Stage 1 needs at least 1,000 samples including 100 positive examples. A short
run may validate the CPU-side flow without satisfying this data-dependent
trigger. No target crash in the smoke test is not a functional failure.

## Evidence and cleanup

Retain the manager bench log, fuzzer log, Brain log, Receiver data, target-hit
evidence, and any crash reports needed for evaluation. The bind mounts above
persist run logs under `/HOST/SYZPILOT-RUNS` and Receiver/model state under
`/HOST/SYZPILOT-BRAIN-STATE`, so removing a container does not discard them.
Capture the final task view, then stop only the two named containers:

```bash
curl -fsS "http://${BRAIN_HOST}:48000/list_tasks" \
  > /HOST/SYZPILOT-RUNS/controller_tasks.json
# Run each pair on the host that owns the named container.
docker stop syzpilot-brain && docker rm syzpilot-brain
docker stop syzpilot-fuzzer && docker rm syzpilot-fuzzer
```

Use a new bench filename on every restart because `syz-manager` refuses to
overwrite an existing one. Do not delete shared kernels, guest templates, or
earlier experiment results during cleanup.
