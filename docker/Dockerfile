# syntax=docker/dockerfile:1.7
ARG KERNEL_FUZZ_IMAGE=qgrain/kernel-fuzz:2404_v1@sha256:946b5620a9e31690c5b67458a63410a75afbff7d9ea4223f2acb93812018afa0
FROM ${KERNEL_FUZZ_IMAGE}

ARG SYZPILOT_REPOSITORY=https://github.com/QGrain/SyzPilot.git
ARG SYZPILOT_REVISION
ARG SYZENCODER_REPOSITORY=zzra1n/SyzEncoder
ARG SYZENCODER_REVISION=6140b0b46c81bb6428458fde5fd800a0e4a0687d
ARG OCI_CREATED=unknown
ARG OCI_REVISION

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    PATH=/opt/miniforge/envs/syzpilot/bin:/opt/miniforge/envs/kernel-fuzz/bin:${PATH} \
    SYZPILOT_DIRECT_ONLY=true \
    SYZPILOT_BASE_MODEL_PATH=/root/models/SyzEncoder \
    SYZPILOT_TOKENIZER_PATH=/root/models/SyzEncoder \
    TOKENIZER_PATH=/root/models/SyzEncoder \
    SYZPILOT_GUIDANCE_REPORT_ROOTS=/root/SyzPilot/benchmark/configs \
    SYZPILOT_SYZKALLER_SYSLINUX=/root/syzpilot-assets/syzlang/sys/linux \
    SYZPILOT_SYZLANG_MANIFEST=/root/syzpilot-assets/syzlang/linux-amd64.json

RUN apt-get update && apt-get install -y --no-install-recommends \
    autossh libaio-dev libprotobuf-dev openjdk-17-jre-headless && \
    rm -rf /var/lib/apt/lists/*

RUN /opt/miniforge/bin/conda create -y -n syzpilot python=3.11 pip && \
    /opt/miniforge/bin/conda clean -afy && \
    sed -i 's/conda activate kernel-fuzz/conda activate syzpilot/' /root/.bash_env

RUN test -n "${SYZPILOT_REVISION}" && \
    test -n "${OCI_REVISION}" && \
    test "${SYZPILOT_REVISION}" = "${OCI_REVISION}" && \
    git init -q /root/SyzPilot && \
    git -C /root/SyzPilot remote add origin "${SYZPILOT_REPOSITORY}" && \
    git -C /root/SyzPilot fetch -q --depth 1 origin "${SYZPILOT_REVISION}" && \
    git -C /root/SyzPilot checkout -q --detach FETCH_HEAD && \
    python -m pip install --no-cache-dir \
      -r /root/SyzPilot/requirements/requirements-brain.txt \
      -r /root/SyzPilot/requirements/requirements-fuzzer.txt \
      --extra-index-url https://download.pytorch.org/whl/cu121 && \
    make -C /root/SyzPilot/brain && \
    python -m compileall -q \
      /root/SyzPilot/brain /root/SyzPilot/filter \
      /root/SyzPilot/analyzer /root/SyzPilot/common && \
    python -m pip check

RUN test -s \
      /opt/miniforge/envs/syzpilot/lib/python3.11/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12 && \
    ln -s \
      /opt/miniforge/envs/syzpilot/lib/python3.11/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12 \
      /usr/local/lib/libcudart.so.12 && \
    ldconfig && \
    python -c 'import ctypes.util; assert ctypes.util.find_library("cudart") == "libcudart.so.12"'

RUN SYZPILOT_FUZZER_DIR=/tmp/SyzPilot-fuzzer \
      /root/SyzPilot/scripts/patch_fuzzers.sh SyzPilot && \
    make -C /tmp/SyzPilot-fuzzer -j"$(nproc)" syzpilot-manifest && \
    mkdir -p /root/syzpilot-assets/syzlang/sys/linux && \
    cp /tmp/SyzPilot-fuzzer/sys/linux/*.txt \
      /root/syzpilot-assets/syzlang/sys/linux/ && \
    /tmp/SyzPilot-fuzzer/bin/syzpilot-manifest \
      -os linux -arch amd64 \
      -output /root/syzpilot-assets/syzlang/linux-amd64.json && \
    rm -rf /tmp/SyzPilot-fuzzer

RUN SYZENCODER_REPOSITORY="${SYZENCODER_REPOSITORY}" \
    SYZENCODER_REVISION="${SYZENCODER_REVISION}" \
    python - <<'PY'
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download

target = Path("/root/models/SyzEncoder")
snapshot_download(
    repo_id=os.environ["SYZENCODER_REPOSITORY"],
    revision=os.environ["SYZENCODER_REVISION"],
    local_dir=target,
)
for name in ("config.json", "model.safetensors", "tokenizer.json"):
    if not (target / name).is_file():
        raise RuntimeError(f"SyzEncoder snapshot is missing {name}")
config_path = target / "config.json"
config = json.loads(config_path.read_text())
config["_name_or_path"] = str(target)
config_path.write_text(json.dumps(config, indent=2) + "\n")
PY

LABEL org.opencontainers.image.title="SyzPilot" \
    org.opencontainers.image.description="Unified environment for the SyzPilot Brain and Fuzzer" \
    org.opencontainers.image.source="https://github.com/QGrain/SyzPilot" \
    org.opencontainers.image.revision="${OCI_REVISION}" \
    org.opencontainers.image.created="${OCI_CREATED}" \
    org.opencontainers.image.licenses="Apache-2.0" \
    org.opencontainers.image.base.name="qgrain/kernel-fuzz:2404_v1"

EXPOSE 48000 37030 37031 37032 37033 37034
WORKDIR /root

# Intentionally override the base image entrypoint. The image does not start
# sshd, initialize guests, run the Controller, or launch a fuzzer implicitly.
ENTRYPOINT []
CMD ["/bin/bash"]
