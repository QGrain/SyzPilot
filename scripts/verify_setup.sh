#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage: verify_setup.sh brain|fuzzer

Run this inside a container created from the unified SyzPilot image.
The brain check expects two visible NVIDIA GPUs and the bundled SyzEncoder.
The fuzzer check expects /dev/kvm and an already built SyzPilot-Fuzzer tree.
EOF
}

if [[ $# -ne 1 ]]; then
    usage >&2
    exit 2
fi

readonly ROLE="$1"
readonly PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
readonly FUZZER_DIR="${SYZPILOT_FUZZER_DIR:-/root/fuzzers/SyzPilot-fuzzer}"
readonly MODEL_DIR="${SYZPILOT_BASE_MODEL_PATH:-/root/models/SyzEncoder}"
readonly SYZLANG_MANIFEST="${SYZPILOT_SYZLANG_MANIFEST:-/root/syzpilot-assets/syzlang/linux-amd64.json}"
readonly SYSLINUX_DIR="${SYZPILOT_SYZKALLER_SYSLINUX:-/root/syzpilot-assets/syzlang/sys/linux}"

for command in git go clang java conda; do
    command -v "$command" >/dev/null || {
        echo "Missing required command: $command" >&2
        exit 1
    }
done

[[ -d "$PROJECT_ROOT/.git" ]] || {
    echo "SyzPilot checkout not found at $PROJECT_ROOT" >&2
    exit 1
}

conda run -n syzpilot python -c \
    'import fastapi, grpc, torch, transformers; print(f"torch={torch.__version__}")'

case "$ROLE" in
    brain)
        for file in config.json model.safetensors tokenizer.json; do
            [[ -s "$MODEL_DIR/$file" ]] || {
                echo "Missing model file: $MODEL_DIR/$file" >&2
                exit 1
            }
        done
        [[ -s "$SYZLANG_MANIFEST" ]] || {
            echo "Missing compiled Syzlang manifest: $SYZLANG_MANIFEST" >&2
            exit 1
        }
        compgen -G "$SYSLINUX_DIR/*.txt" >/dev/null || {
            echo "Missing Syzlang descriptions under: $SYSLINUX_DIR" >&2
            exit 1
        }
        case "${SYZPILOT_DIRECT_ONLY:-}" in
            1|true|TRUE|yes|YES) ;;
            *)
                echo "Functional containers require SYZPILOT_DIRECT_ONLY=true." >&2
                exit 1
                ;;
        esac
        command -v nvidia-smi >/dev/null || {
            echo "nvidia-smi is unavailable; start the container with NVIDIA GPU access." >&2
            exit 1
        }
        gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
        if (( gpu_count < 2 )); then
            echo "Brain functional evaluation requires two visible GPUs; found $gpu_count." >&2
            exit 1
        fi
        PYTHONPATH="$PROJECT_ROOT/brain" conda run -n syzpilot python -c '
from config import ControllerConfig
from gpu_admission import validate_physical_gpu_namespace

config = ControllerConfig()
required = {
    *config.training_gpu_ids,
    *config.training_fallback_gpu_ids,
    config.attribution_gpu_id,
    config.inference_gpu_id,
}
validate_physical_gpu_namespace(required)
print("CUDA/NVML physical GPU mapping is valid")
'
        make -C "$PROJECT_ROOT/brain" >/dev/null
        echo "Brain setup is ready with $gpu_count visible GPUs."
        ;;
    fuzzer)
        [[ -r /dev/kvm && -w /dev/kvm ]] || {
            echo "/dev/kvm is not accessible; start the container with KVM access." >&2
            exit 1
        }
        for file in \
            bin/syz-manager \
            bin/linux_amd64/syz-executor \
            bin/linux_amd64/syz-execprog; do
            [[ -x "$FUZZER_DIR/$file" ]] || {
                echo "Missing Fuzzer binary: $FUZZER_DIR/$file" >&2
                exit 1
            }
        done
        echo "Fuzzer setup is ready with KVM access."
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
