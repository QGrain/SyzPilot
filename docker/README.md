# Artifact container builds

Run these commands from the root of this repository. The default Docker builds
do not embed pretrained model weights; the optional Brain model build downloads
and embeds the public SyzEncoder release. The Brain image includes the public
benchmark metadata and bug reports tracked by this repository. Neither build
embeds compiled kernel cases, VM images, runtime credentials, or experiment
logs.

## Fuzzer

`Dockerfile.fuzzer` applies `fuzzer/SyzPilot-fuzzer.diff` to upstream syzkaller
commit `6e83b42dcfcd13c3b8e0d5c803cdcc424c0fbff9` and builds the
minimal functional runtime: manager, executor, execprog, and a compiled
Syzlang manifest. It does not claim that every optional `make all` utility
builds. The default base is the published
`qgrain/kernel-fuzz:2204_v2` image, pinned to its Docker Hub manifest digest.

```bash
docker build -f docker/Dockerfile.fuzzer -t syzpilot-fuzzer:artifact .
docker run --rm --entrypoint /bin/bash syzpilot-fuzzer:artifact -lc \
  'git rev-parse HEAD; test -x bin/syz-manager; test -x bin/linux_amd64/syz-executor; test -x bin/linux_amd64/syz-execprog; test -s bin/linux-amd64-syzlang-manifest.json'
```

If `qgrain/kernel-fuzz:2204_v3` has been independently verified for the
required upstream revision and toolchain, it can be selected explicitly:

```bash
docker build -f docker/Dockerfile.fuzzer \
  --build-arg KERNEL_FUZZ_IMAGE=qgrain/kernel-fuzz:2204_v3 \
  -t syzpilot-fuzzer:artifact-v3 .
```

The default pinned `2204_v2` image remains the tested build base. Do not
substitute a mutable tag in a reproducibility claim without recording its
resolved digest.

For a real fuzzing run, provide a KVM-capable host, compiled kernel case,
guest disk image and SSH key through mounts. Use unique manager work and bench
paths and a suitable CPU affinity for each instance; see the project README
for manager configuration. Do not mount a host source tree over
`/root/SyzPilot-fuzzer` when testing the image-built binary, as doing so would
hide the built code.

## Brain

The Brain recipe uses the published PyTorch 2.2.1 CUDA 12.1 runtime image,
installs the pinned Python requirements and Java 17 for TorchServe, and
generates the receiver's protobuf bindings. The default build has no model
weights and supports the CPU-only preparation exercise. The final model image
downloads the separately hosted
[`zzra1n/SyzEncoder`](https://huggingface.co/zzra1n/SyzEncoder) at build time;
do not pass access tokens as Docker build arguments.
The container defaults to `SYZPILOT_DIRECT_ONLY=true`: it skips the
host-specific SSH tunnel helper and refuses isolated-mode registrations.
Use the `direct` fuzzer configuration in `artifact/README.md`.

```bash
docker build -f docker/Dockerfile.brain -t syzpilot-brain:artifact .
docker run --rm --entrypoint /bin/bash syzpilot-brain:artifact -lc \
  'python -c "import torch, grpc, fastapi; print(torch.__version__)"; command -v torchserve'

# Pin the public model to the validated immutable Hugging Face commit.
docker build -f docker/Dockerfile.brain \
  --build-arg INCLUDE_SYZENCODER=true \
  --build-arg SYZENCODER_REVISION=6140b0b46c81bb6428458fde5fd800a0e4a0687d \
  -t syzpilot-brain:artifact-model .
```

Building or importing the Brain is **not** an end-to-end functional test.
Its container-oriented defaults can be overridden with
`SYZPILOT_BASE_MODEL_PATH`, `SYZPILOT_TOKENIZER_PATH`,
`SYZPILOT_SYZKALLER_SYSLINUX`, `SYZPILOT_SYZLANG_MANIFEST`,
`SYZPILOT_GUIDANCE_REPORT_ROOTS`, and
`SYZPILOT_GUIDANCE_KALLGRAPH_ROOTS`. The two roots variables use the
platform path separator and restrict Brain-local paths supplied in task
registration. When either roots variable is set, the corresponding
author-machine benchmark-title fallback map is disabled: pass an explicit
`report_path` and/or `kallgraph_dir` in the fuzzer task configuration.
See `artifact/README.md` for a concrete example that builds the kernel and
guest locally from the packaged configuration and pinned public sources.
Online training and inference need two suitable NVIDIA GPUs, reachable
controller/receiver/TorchServe ports, and configured network routing.
The optional self-contained build adds only the public SyzEncoder release to
the default source image. A runtime-selected target report, compiled kernel,
VM image, credentials, and generated results remain external and are supplied
through configuration or mounts.

The image has not been published by this recipe. Before distributing a Brain
image, verify the exact image tag and run an integration test with the
artifact's actual model and kernel-case mounts. Pass secrets only at runtime,
never as Docker build arguments or committed files.
