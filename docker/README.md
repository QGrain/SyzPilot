# Unified SyzPilot image

The release uses one image for both SyzPilot roles. It is based on the
published `qgrain/kernel-fuzz:2404_v1` image and adds:

- the public `QGrain/SyzPilot` checkout at `/root/SyzPilot`;
- a Python 3.11 Conda environment named `syzpilot`;
- the Brain and Fuzzer-side Python dependencies;
- Java 17 runtime support for TorchServe; and
- the pinned public SyzEncoder and tokenizer at `/root/models/SyzEncoder`; and
- compiled Linux/amd64 Syzlang metadata plus matching descriptions under
  `/root/syzpilot-assets/syzlang` for cold-start guidance.

The image does not contain a target kernel, target PoC, experiment output, or
patched Syzkaller checkout. It also has no active entrypoint: creating a
container does not implicitly start sshd, initialize a guest, run the Brain,
or launch a fuzzer.

The container environment defaults to `SYZPILOT_DIRECT_ONLY=true`. Functional
containers therefore use explicit private-IP connectivity and do not require
the optional SSH tunnel service or its privileged host helpers.

## Build

Build from the repository root. `SYZPILOT_REVISION` and `OCI_REVISION` are
required to be the same exact public commit. Record that revision and the
resulting image digest for an evaluation.

```bash
REVISION="$(git rev-parse HEAD)"
docker build -f docker/Dockerfile \
  --build-arg SYZPILOT_REVISION="$REVISION" \
  --build-arg OCI_REVISION="$REVISION" \
  -t qgrain/syzpilot:ndss27-ae .

docker image inspect qgrain/syzpilot:ndss27-ae \
  --format '{{json .Config.Entrypoint}} {{json .Config.Cmd}}'
docker run --rm qgrain/syzpilot:ndss27-ae \
  bash -lc 'python --version; go version; java -version; test -s /root/models/SyzEncoder/model.safetensors'
```

The default base is pinned to the multi-platform digest of
`qgrain/kernel-fuzz:2404_v1`. Override `KERNEL_FUZZ_IMAGE` only with a tested,
digest-pinned compatible image.

The optional `requirements/requirements-agent.txt` is not installed in the
functional image. It requires a newer Pydantic release than the pinned Brain
environment and belongs in a separate environment when evaluating the
experimental agentic waypoint tools.

## Create a Brain container

Run the Brain role on a GPU host. The explicit `sleep infinity` below is a
user-selected keepalive, not an image default. Replace the GPU IDs and host
mounts for the evaluation machine.

```bash
mkdir -p /HOST/SYZPILOT-BRAIN-STATE/{receiver_data,logs,model_store}

docker run -d --name syzpilot-brain --network host \
  --gpus 'device=0,1' \
  -v /HOST/KERNELS:/root/kernels \
  -v /HOST/SYZPILOT-RUNS:/root/syzpilot-runs \
  -v /HOST/SYZPILOT-BRAIN-STATE/receiver_data:/root/SyzPilot/brain/receiver_data \
  -v /HOST/SYZPILOT-BRAIN-STATE/logs:/root/SyzPilot/brain/logs \
  -v /HOST/SYZPILOT-BRAIN-STATE/model_store:/root/SyzPilot/brain/model_store \
  qgrain/syzpilot:ndss27-ae sleep infinity

docker exec syzpilot-brain bash -lc '
  cd /root/SyzPilot
  make -C brain
  scripts/verify_setup.sh brain
'
```

Start the Controller explicitly only after paths, GPUs, ports, and network
routing are configured:

```bash
docker exec -it syzpilot-brain bash -lc '
  cd /root/SyzPilot
  export SYZPILOT_DIRECT_ONLY=true
  export SYZPILOT_TRAINING_GPU_IDS=0
  export SYZPILOT_INFERENCE_GPU_ID=1
  export SYZPILOT_ATTRIBUTION_GPU_ID=1
  python brain/controller.py --host 0.0.0.0 --port 48000
'
```

The Brain host must allow the Fuzzer to reach TCP 48000, the dynamically
allocated Receiver range 31001--31999, and the configured TorchServe ports.
Restrict these ports to the evaluation hosts rather than exposing them to the
public Internet.

## Create a Fuzzer container

Run the Fuzzer role on a KVM-capable CPU host. Mount kernels and run outputs
explicitly. The inherited base image supplies the disposable guest template
used by the default functional configuration.

```bash
docker run -d --name syzpilot-fuzzer --network host --device /dev/kvm \
  --cpuset-cpus=0-1 \
  -v /HOST/KERNELS:/root/kernels \
  -v /HOST/SYZPILOT-RUNS:/root/syzpilot-runs \
  qgrain/syzpilot:ndss27-ae sleep infinity

docker exec syzpilot-fuzzer bash -lc '
  cd /root/SyzPilot
  scripts/patch_fuzzers.sh SyzPilot
  cd /root/fuzzers/SyzPilot-fuzzer
  make -j$(nproc)
  /root/SyzPilot/scripts/verify_setup.sh fuzzer
'
```

`patch_fuzzers.sh` fetches exact upstream Syzkaller commit
`6e83b42dcfcd13c3b8e0d5c803cdcc424c0fbff9` over HTTPS and applies the
reviewed `fuzzer/SyzPilot-fuzzer.diff`. It refuses to overwrite an unrelated
destination. MOCK and SyzDirect patch automation are intentionally not part of
this release.

The image's source checkout, Fuzzer patch, and compiled Syzlang metadata are a
matched snapshot. Rebuild the image to adopt a newer public revision instead
of running `git pull` in only one role container.

For distributed deployment, replace loopback Controller/callback addresses in
the manager configuration with the two hosts' mutually reachable private
addresses. Keep HTTP, Receiver, and manager callback ports firewalled to those
hosts.

## Publishing

Pushing a rebuilt image to an existing Docker Hub tag atomically replaces the
tag reference; deleting the old tag first is unnecessary and creates an
avoidable availability gap. Record the new digest after pushing:

```bash
docker push qgrain/syzpilot:ndss27-ae
docker buildx imagetools inspect qgrain/syzpilot:ndss27-ae
```
