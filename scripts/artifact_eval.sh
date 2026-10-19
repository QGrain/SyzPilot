#!/usr/bin/env bash
set -Eeuo pipefail

readonly PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
readonly KERNEL_ROOT="${SYZPILOT_KERNEL_ROOT:-/root/kernels}"
readonly RUN_ROOT="${SYZPILOT_RUN_ROOT:-/root/syzpilot-runs/case_36}"
readonly FUZZER_DIR="${SYZPILOT_FUZZER_DIR:-/root/fuzzers/SyzPilot-fuzzer}"
readonly LINUX_REPO="${SYZPILOT_LINUX_REPO:-${KERNEL_ROOT}/linux-git-master}"
readonly CASE_DIR="${KERNEL_ROOT}/case_36"
readonly CASE_COMMIT="6207214a70bfaec7b41f39502353fd3ca89df68c"

usage() {
    cat <<'EOF'
Usage: scripts/artifact_eval.sh COMMAND

Run inside a container created from qgrain/syzpilot:ndss27-ae.

Commands:
  image-info       Print the bundled source revision and key dependencies.
  prepare-fuzzer   Patch, build, and verify SyzPilot-Fuzzer.
  build-case36     Fetch and build the case-36 kernel with GCC 11.4.
  extract-case36   Run report-derived waypoint extraction for case 36.
  prepare-config   Generate the case-36 manager config. Requires
                   BRAIN_HOST and FUZZER_HOST.
  verify-brain     Verify the Brain environment and two visible GPUs.
  start-brain      Start the Controller in the foreground and log to RUN_ROOT.
  wait-brain       Wait up to 60 seconds for the Controller health endpoint.
  start-fuzzer     Run the 90-minute fuzzer smoke test in the foreground.
  brain-status     Show Controller health, tasks, and the latest Brain log.
  fuzzer-status    Show the latest fuzzer log.
  help             Show this message.

Useful overrides:
  SYZPILOT_BUILD_JOBS=8
  SYZPILOT_KERNEL_CC=/usr/bin/gcc-11
  SYZPILOT_KERNEL_ROOT=/root/kernels
  SYZPILOT_RUN_ROOT=/root/syzpilot-runs/case_36
  SYZPILOT_FUZZER_DIR=/root/fuzzers/SyzPilot-fuzzer
  SYZPILOT_TRAINING_GPU_IDS=0
  SYZPILOT_INFERENCE_GPU_ID=1
  SYZPILOT_ATTRIBUTION_GPU_ID=1
  SYZPILOT_CPUSET=0-1
  SYZPILOT_DURATION=90m
EOF
}

require_file() {
    [[ -s "$1" ]] || {
        echo "Missing required file: $1" >&2
        exit 1
    }
}

image_info() {
    printf 'SyzPilot revision: '
    git -C "$PROJECT_ROOT" rev-parse HEAD
    printf 'Kernel compiler: '
    "${SYZPILOT_KERNEL_CC:-gcc}" --version | head -n 1
    require_file "${SYZPILOT_BASE_MODEL_PATH:-/root/models/SyzEncoder}/model.safetensors"
    echo "SyzEncoder: available"
}

prepare_fuzzer() {
    export SYZPILOT_FUZZER_DIR="$FUZZER_DIR"
    "$PROJECT_ROOT/scripts/patch_fuzzers.sh" SyzPilot
    make -C "$FUZZER_DIR" -j"${SYZPILOT_BUILD_JOBS:-8}"
    "$PROJECT_ROOT/scripts/verify_setup.sh" fuzzer
}

build_case36() {
    local compiler="${SYZPILOT_KERNEL_CC:-}"
    [[ -n "$compiler" && -x "$compiler" ]] || {
        echo "SYZPILOT_KERNEL_CC does not name an executable compiler." >&2
        exit 1
    }
    [[ ! -e "$CASE_DIR" ]] || {
        echo "$CASE_DIR already exists; preserve or move it before a fresh build." >&2
        exit 1
    }
    mkdir -p "$KERNEL_ROOT"
    if [[ ! -d "$LINUX_REPO/.git" ]]; then
        git clone --filter=blob:none https://github.com/torvalds/linux.git "$LINUX_REPO"
    fi
    git -C "$LINUX_REPO" fetch origin "$CASE_COMMIT"
    python "$PROJECT_ROOT/experiments/compile_kernel.py" \
        --workdir "$KERNEL_ROOT" \
        --linux-git-master "$LINUX_REPO" \
        --config "$PROJECT_ROOT/mini-benchmark/compile_case_36.csv" \
        --compiler "$compiler" \
        -j "${SYZPILOT_BUILD_JOBS:-8}"
    require_file "$CASE_DIR/vmlinux"
    require_file "$CASE_DIR/arch/x86/boot/bzImage"
    echo "case_36 kernel build is ready under $CASE_DIR"
}

extract_case36() {
    require_file "$CASE_DIR/vmlinux"
    mkdir -p "$RUN_ROOT"
    python "$PROJECT_ROOT/analyzer/waypoints_extractor.py" \
        -k "$CASE_DIR" \
        -t "$PROJECT_ROOT/benchmark/configs/case_36.title" \
        -r "$PROJECT_ROOT/benchmark/configs/case_36.report" \
        | tee "$RUN_ROOT/waypoints.txt"
    grep -q '\[For SyzPilot-fuzzer:\]' "$RUN_ROOT/waypoints.txt"
    echo "Waypoint output: $RUN_ROOT/waypoints.txt"
}

prepare_config() {
    : "${BRAIN_HOST:?Set BRAIN_HOST to the Brain host private address.}"
    : "${FUZZER_HOST:?Set FUZZER_HOST to the Fuzzer host private address.}"
    require_file "$RUN_ROOT/waypoints.txt"
    python "$PROJECT_ROOT/scripts/prepare_functional_config.py" \
        --case 36 \
        --waypoints-output "$RUN_ROOT/waypoints.txt" \
        --output "$RUN_ROOT/manager.cfg" \
        --run-dir "$RUN_ROOT" \
        --kernel-dir "$CASE_DIR" \
        --syzkaller "$FUZZER_DIR" \
        --brain-host "$BRAIN_HOST" \
        --fuzzer-host "$FUZZER_HOST"
}

verify_brain() {
    "$PROJECT_ROOT/scripts/verify_setup.sh" brain
}

start_brain() {
    verify_brain
    mkdir -p "$RUN_ROOT"
    export SYZPILOT_DIRECT_ONLY=true
    export SYZPILOT_TRAINING_GPU_IDS="${SYZPILOT_TRAINING_GPU_IDS:-0}"
    export SYZPILOT_INFERENCE_GPU_ID="${SYZPILOT_INFERENCE_GPU_ID:-1}"
    export SYZPILOT_ATTRIBUTION_GPU_ID="${SYZPILOT_ATTRIBUTION_GPU_ID:-1}"
    if curl -fsS http://127.0.0.1:48000/health >/dev/null 2>&1; then
        echo "A Controller is already listening on port 48000." >&2
        exit 1
    fi
    cd "$PROJECT_ROOT"
    exec python brain/controller.py --host 0.0.0.0 --port 48000 \
        >> "$RUN_ROOT/brain.log" 2>&1
}

wait_brain() {
    : "${BRAIN_HOST:?Set BRAIN_HOST to the Brain host private address.}"
    for _ in {1..30}; do
        if curl -fsS "http://${BRAIN_HOST}:48000/health" >/dev/null 2>&1; then
            echo "Brain Controller is ready at ${BRAIN_HOST}:48000"
            return
        fi
        sleep 2
    done
    echo "Brain Controller did not become ready within 60 seconds." >&2
    exit 1
}

start_fuzzer() {
    : "${BRAIN_HOST:?Set BRAIN_HOST to the Brain host private address.}"
    "$PROJECT_ROOT/scripts/verify_setup.sh" fuzzer
    require_file "$RUN_ROOT/manager.cfg"
    wait_brain
    mkdir -p "$RUN_ROOT"
    exec 9>"$RUN_ROOT/syz-manager.lock"
    flock -n 9 || {
        echo "Another syz-manager is already using $RUN_ROOT." >&2
        exit 1
    }

    local run_id="${SYZPILOT_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
    local bench="$RUN_ROOT/case_36_${run_id}.bench.log"
    local log="$RUN_ROOT/fuzzer_${run_id}.log"
    [[ ! -e "$bench" && ! -e "$log" ]] || {
        echo "Run files already exist for ID $run_id; choose a new SYZPILOT_RUN_ID." >&2
        exit 1
    }
    printf '%s\n' "$run_id" > "$RUN_ROOT/current_run_id"
    echo "Fuzzer log: $log"
    echo "Bench log: $bench"
    exec taskset -c "${SYZPILOT_CPUSET:-0-1}" \
        "$FUZZER_DIR/bin/syz-manager" \
        -config "$RUN_ROOT/manager.cfg" \
        -timeout "${SYZPILOT_DURATION:-90m}" \
        -bench "$bench" \
        > "$log" 2>&1
}

brain_status() {
    local host="${BRAIN_HOST:-127.0.0.1}"
    curl -fsS "http://${host}:48000/health"
    echo
    curl -fsS "http://${host}:48000/list_tasks" | python -m json.tool
    [[ ! -f "$RUN_ROOT/brain.log" ]] || tail -n 40 "$RUN_ROOT/brain.log"
}

fuzzer_status() {
    require_file "$RUN_ROOT/current_run_id"
    local run_id
    run_id="$(<"$RUN_ROOT/current_run_id")"
    require_file "$RUN_ROOT/fuzzer_${run_id}.log"
    tail -n 60 "$RUN_ROOT/fuzzer_${run_id}.log"
}

case "${1:-help}" in
    image-info) image_info ;;
    prepare-fuzzer) prepare_fuzzer ;;
    build-case36) build_case36 ;;
    extract-case36) extract_case36 ;;
    prepare-config) prepare_config ;;
    verify-brain) verify_brain ;;
    start-brain) start_brain ;;
    wait-brain) wait_brain ;;
    start-fuzzer) start_fuzzer ;;
    brain-status) brain_status ;;
    fuzzer-status) fuzzer_status ;;
    help|-h|--help) usage ;;
    *)
        echo "Unknown command: $1" >&2
        usage >&2
        exit 2
        ;;
esac
