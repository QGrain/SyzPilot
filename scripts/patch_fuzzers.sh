#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage: patch_fuzzers.sh SyzPilot

Clone the pinned upstream Syzkaller revision and apply the reviewed
SyzPilot-Fuzzer patch. The default output directory is
/root/fuzzers/SyzPilot-fuzzer. Set SYZPILOT_FUZZER_DIR to use another path.
EOF
}

if [[ $# -ne 1 ]]; then
    usage >&2
    exit 2
fi

case "$1" in
    SyzPilot) ;;
    MOCK|SyzDirect)
        echo "$1 patching is not implemented by this script yet." >&2
        exit 2
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac

readonly UPSTREAM_REPOSITORY=https://github.com/google/syzkaller.git
readonly UPSTREAM_COMMIT=6e83b42dcfcd13c3b8e0d5c803cdcc424c0fbff9
readonly PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
readonly PATCH_FILE="$PROJECT_ROOT/fuzzer/SyzPilot-fuzzer.diff"
readonly FUZZER_DIR="${SYZPILOT_FUZZER_DIR:-/root/fuzzers/SyzPilot-fuzzer}"

if [[ ! -s "$PATCH_FILE" ]]; then
    echo "Missing SyzPilot-Fuzzer patch: $PATCH_FILE" >&2
    exit 1
fi

if [[ ! -e "$FUZZER_DIR" ]]; then
    mkdir -p "$(dirname -- "$FUZZER_DIR")"
    git init -q "$FUZZER_DIR"
    git -C "$FUZZER_DIR" remote add origin "$UPSTREAM_REPOSITORY"
    git -C "$FUZZER_DIR" fetch -q --depth 1 origin "$UPSTREAM_COMMIT"
    git -C "$FUZZER_DIR" checkout -q --detach FETCH_HEAD
elif [[ ! -d "$FUZZER_DIR/.git" ]]; then
    echo "Destination exists but is not a Git checkout: $FUZZER_DIR" >&2
    exit 1
fi

if ! git -C "$FUZZER_DIR" rev-parse --verify HEAD >/dev/null 2>&1; then
    origin="$(git -C "$FUZZER_DIR" remote get-url origin 2>/dev/null || true)"
    if [[ "$origin" != "$UPSTREAM_REPOSITORY" ]] ||
        find "$FUZZER_DIR" -mindepth 1 -maxdepth 1 ! -name .git -print -quit |
            grep -q .; then
        echo "Refusing to recover an unrecognized empty checkout: $FUZZER_DIR" >&2
        exit 1
    fi
    git -C "$FUZZER_DIR" fetch -q --depth 1 origin "$UPSTREAM_COMMIT"
    git -C "$FUZZER_DIR" checkout -q --detach FETCH_HEAD
fi

actual_commit="$(git -C "$FUZZER_DIR" rev-parse HEAD)"
if [[ "$actual_commit" != "$UPSTREAM_COMMIT" ]]; then
    echo "Expected upstream $UPSTREAM_COMMIT, found $actual_commit in $FUZZER_DIR" >&2
    exit 1
fi

if git -C "$FUZZER_DIR" apply --reverse --check "$PATCH_FILE" \
    >/dev/null 2>&1; then
    echo "SyzPilot-Fuzzer patch is already applied in $FUZZER_DIR"
elif git -C "$FUZZER_DIR" apply --check "$PATCH_FILE" >/dev/null 2>&1; then
    git -C "$FUZZER_DIR" apply "$PATCH_FILE"
    echo "Applied SyzPilot-Fuzzer patch in $FUZZER_DIR"
else
    echo "The working tree is incompatible with the reviewed patch." >&2
    echo "Use a fresh destination or inspect existing local changes." >&2
    exit 1
fi

git -C "$FUZZER_DIR" diff --check
echo "Upstream revision: $UPSTREAM_COMMIT"
echo "Next: cd $FUZZER_DIR && make -j\$(nproc)"
