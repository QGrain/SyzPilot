#!/usr/bin/env bash
set -euo pipefail

: "${SYZPILOT_CODEX_REAL_BIN:?missing pinned Codex binary path}"
: "${CODEX_HOME:?missing CODEX_HOME}"
: "${SYZPILOT_MASKED_REPO:?missing repository path to mask}"
: "${SYZPILOT_ALLOWED_ROOTS:?missing evidence roots}"
: "${HOME:?missing source home}"

source_codex_home="$(readlink -f "${CODEX_HOME}")"
source_user_home="$(readlink -f "${HOME}")"
masked_repo="$(readlink -f "${SYZPILOT_MASKED_REPO}")"
auth_file="${source_codex_home}/auth.json"
code_mode_host="$(dirname "${SYZPILOT_CODEX_REAL_BIN}")/codex-code-mode-host"
if [[ ! -f "${auth_file}" ]]; then
    echo "Codex authentication file not found: ${auth_file}" >&2
    exit 2
fi
if [[ ! -x "${code_mode_host}" ]]; then
    echo "Codex code-mode host not found: ${code_mode_host}" >&2
    exit 2
fi
if [[ ! -d "${masked_repo}" ]]; then
    echo "Repository path to mask is not a directory: ${masked_repo}" >&2
    exit 2
fi

declare -a mount_args=()
declare -a mask_args=()
declare -A created_dirs=()
declare -A masked_analysis_dirs=()

add_directory_chain() {
    local directory="$1"
    local current="${directory}"
    local -a pending=()
    while [[ "${current}" == "${source_user_home}"/* ]]; do
        pending=("${current}" "${pending[@]}")
        current="$(dirname "${current}")"
    done
    for current in "${pending[@]}"; do
        if [[ -z "${created_dirs[${current}]:-}" ]]; then
            mount_args+=(--dir "${current}")
            created_dirs["${current}"]=1
        fi
    done
}

IFS=':' read -r -a allowed_roots <<< "${SYZPILOT_ALLOWED_ROOTS}"
for root in "${allowed_roots[@]}"; do
    [[ -n "${root}" ]] || continue
    root="$(readlink -f "${root}")"
    if [[ ! -e "${root}" ]]; then
        echo "Allowed evidence root not found: ${root}" >&2
        exit 2
    fi
    if [[
        "${root}" == "${masked_repo}" ||
        "${root}" == "${masked_repo}"/* ||
        "${masked_repo}" == "${root}"/*
    ]]; then
        echo "Evidence root overlaps masked repository: ${root}" >&2
        exit 2
    fi
    if [[ "${root}" == "${source_user_home}"/* ]]; then
        add_directory_chain "$(dirname "${root}")"
        mount_args+=(--ro-bind "${root}" "${root}")
    fi
    for analysis_dir in \
        "${root}/SyzPilot-analysis" \
        "${root}"/case_*/SyzPilot-analysis; do
        [[ -d "${analysis_dir}" ]] || continue
        analysis_dir="$(readlink -f "${analysis_dir}")"
        if [[ -z "${masked_analysis_dirs[${analysis_dir}]:-}" ]]; then
            mask_args+=(--tmpfs "${analysis_dir}")
            masked_analysis_dirs["${analysis_dir}"]=1
        fi
    done
done

if [[ "${masked_repo}" == "${source_user_home}"/* ]]; then
    add_directory_chain "${masked_repo}"
fi
mask_args+=(--tmpfs "${masked_repo}")

exec bwrap \
    --die-with-parent \
    --new-session \
    --unshare-pid \
    --unshare-ipc \
    --unshare-uts \
    --ro-bind / / \
    --dev-bind /dev /dev \
    --proc /proc \
    --tmpfs /tmp \
    --tmpfs "${source_user_home}" \
    "${mount_args[@]}" \
    "${mask_args[@]}" \
    --ro-bind "${auth_file}" /tmp/syzpilot-codex-auth.json \
    --ro-bind "${SYZPILOT_CODEX_REAL_BIN}" /tmp/syzpilot-codex-real \
    --ro-bind "${code_mode_host}" /tmp/codex-code-mode-host \
    --setenv HOME "${source_user_home}" \
    --setenv CODEX_HOME /tmp/syzpilot-codex-home \
    --chdir /tmp \
    /bin/bash -c '
        set -euo pipefail
        umask 077
        mkdir -p "${CODEX_HOME}"
        cp /tmp/syzpilot-codex-auth.json "${CODEX_HOME}/auth.json"
        ln -s /tmp/syzpilot-codex-real /tmp/codex-linux-sandbox
        ln -s /tmp/syzpilot-codex-real /tmp/codex-execve-wrapper
        ln -s /tmp/syzpilot-codex-real /tmp/apply_patch
        ln -s /tmp/syzpilot-codex-real /tmp/applypatch
        export PATH="/tmp:${PATH}"
        exec /tmp/syzpilot-codex-real "$@"
    ' syzpilot-codex "$@"
