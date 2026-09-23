#!/bin/bash
# this is a script to set up the environment for the syzkaller tunnel service.
# it must be run with root privileges (via sudo).
# 1. create a low-privilege user named fuzzer-tunnel.
# 2. set up the .ssh directory and authorized_keys file for the user, and ensure the permissions are correct.
# 3. create a special helper script for appending public keys (supports --check/--clear/--read mode).
# 4. configure sudoers, allow the user running this script to execute the helper script without a password.

set -e

# --- configure variables ---
TUNNEL_USER="fuzzer-tunnel"
HELPER_SCRIPT_PATH="/usr/local/bin/add-fuzzer-key"

# --- check if the script is running with root privileges ---
if [[ $EUID -ne 0 ]]; then
   echo "error: this script must be run with root privileges (via sudo)."
   exit 1
fi

# SUDO_USER variable is set by the sudo command, pointing to the username of the user who initially called sudo.
# if the user directly logs in as root, the variable will be empty. we require that the script must be run via sudo.
if [ -z "$SUDO_USER" ]; then
    echo "error: please do not log in as root directly, please use 'sudo bash setup.sh'"
    exit 1
fi
APP_USER="$SUDO_USER"
echo "[+] configure no-password sudo privileges for user '${APP_USER}' to manage tunnel keys."


# --- 1. create the tunnel user ---
if id -u "${TUNNEL_USER}" >/dev/null 2>&1; then
    echo "[√] user '${TUNNEL_USER}' already exists, skip creation."
else
    echo "[+] creating user '${TUNNEL_USER}'..."
    useradd --system --shell /sbin/nologin --create-home "${TUNNEL_USER}"
    echo "[√] user '${TUNNEL_USER}' created successfully."
fi

# --- 2. create and set the SSH file permissions ---
TUNNEL_USER_HOME=$(eval echo ~${TUNNEL_USER})
SSH_DIR="${TUNNEL_USER_HOME}/.ssh"
AUTH_KEYS_FILE="${SSH_DIR}/authorized_keys"

echo "[+] setting up SSH directory and file permissions..."
mkdir -p -m 700 "${SSH_DIR}"
touch "${AUTH_KEYS_FILE}"
chmod 600 "${AUTH_KEYS_FILE}"
chown -R "${TUNNEL_USER}:${TUNNEL_USER}" "${SSH_DIR}"
echo "[√] SSH directory and file permissions set up."

# --- 3. create the helper script ---
echo "[+] creating safe key appending helper script..."
cat > "${HELPER_SCRIPT_PATH}" << EOF
#!/bin/bash
# add-fuzzer-key is a safe helper script, which is used to append public key to the authorized_keys file of the user.
# 1. normal mode: add-fuzzer-key <user> <public_key_string>
#    validate the public key format and append it to the authorized_keys file of the user.
# 2. check mode: add-fuzzer-key <user> --check
#    execute a harmless write test, to verify if the permission is correctly configured.
#    it will append a temporary line with a unique marker, verify the write, and then ensure to clean it up.
# 3. read mode: add-fuzzer-key <user> --read
#    read the authorized_keys file of the user.
# 4. remove mode: add-fuzzer-key <user> --remove <exact_entry>
#    atomically remove one exact authorized_keys entry.
# if any command fails, the script will exit immediately.

set -e

TARGET_USER=\$1

# check if the third parameter is --check
if [ "\$2" == "--check" ]; then
    # validate the number of parameters
    if [ \$# -ne 2 ]; then
        echo "usage: \$0 <user> --check"
        exit 1
    fi

    TARGET_AUTH_FILE=$(eval echo ~\${TARGET_USER})/.ssh/authorized_keys

    # construct a unique check marker
    CHECK_MARKER="fuzzer-tunnel-permission-check-\$(date +%s)"

    # define a cleanup function, which is used to delete the check marker
    # this function will be called by the trap command when the script exits.
    cleanup() {
        sed -i "/\${CHECK_MARKER}/d" "\${TARGET_AUTH_FILE}"
    }

    # set trap: whether the script exits normally, exits with an error, or is interrupted by a signal,
    # will execute the cleanup function. this is the key to ensure the cleanup.
    trap cleanup EXIT

    # 1. try to append a line with a unique marker
    echo "\${CHECK_MARKER}" >> "\${TARGET_AUTH_FILE}"

    # 2. verify if the write is successful
    if ! grep -qF -- "\${CHECK_MARKER}" "\${TARGET_AUTH_FILE}"; then
        echo "error: write permission check failed, cannot find the check marker in the file."
        exit 1
    fi

    echo "write permission check successful."
    exit 0
elif [ "\$2" == "--clear" ]; then
    # validate the number of parameters
    if [ \$# -ne 2 ]; then
        echo "usage: \$0 <user> --clear"
        exit 1
    fi

    TARGET_AUTH_FILE=$(eval echo ~\${TARGET_USER})/.ssh/authorized_keys

    # clear the authorized_keys file
    cat /dev/null > "\${TARGET_AUTH_FILE}"
    echo "authorized_keys file cleared."
    exit 0
elif [ "\$2" == "--read" ]; then
    # validate the number of parameters
    if [ \$# -ne 2 ]; then
        echo "usage: \$0 <user> --read"
        exit 1
    fi

    TARGET_AUTH_FILE=$(eval echo ~\${TARGET_USER})/.ssh/authorized_keys

    # read the authorized_keys file
    cat "\${TARGET_AUTH_FILE}"
    exit 0
elif [ "\$2" == "--remove" ]; then
    if [ \$# -ne 3 ]; then
        echo "usage: \$0 <user> --remove <exact_entry>"
        exit 1
    fi

    TARGET_AUTH_FILE=$(eval echo ~\${TARGET_USER})/.ssh/authorized_keys
    ENTRY_TO_REMOVE=\$3
    TEMP_AUTH_FILE="\${TARGET_AUTH_FILE}.tmp.\$\$"
    trap 'rm -f "\${TEMP_AUTH_FILE}"' EXIT
    while IFS= read -r entry || [ -n "\${entry}" ]; do
        if [ "\${entry}" != "\${ENTRY_TO_REMOVE}" ]; then
            printf '%s\n' "\${entry}" >> "\${TEMP_AUTH_FILE}"
        fi
    done < "\${TARGET_AUTH_FILE}"
    chmod 600 "\${TEMP_AUTH_FILE}"
    mv "\${TEMP_AUTH_FILE}" "\${TARGET_AUTH_FILE}"
    trap - EXIT
    echo "authorized_keys entry removed."
    exit 0
else
    # validate the number of parameters
    if [ \$# -ne 2 ]; then
        echo "usage: \$0 <user> <public_key_string>"
        exit 1
    fi

    PUB_KEY=\$2
    TARGET_AUTH_FILE=$(eval echo ~\${TARGET_USER})/.ssh/authorized_keys

    # safe check, ensure the key string looks like a valid SSH public key format
    # (allow prefix with "command=...")
    if ! [[ "\${PUB_KEY}" =~ ssh-(rsa|ed25519) ]]; then
        echo "error: invalid public key format. no operation is performed."
        echo "provided public key: '\${PUB_KEY}'"
        exit 1
    fi

    # append the key
    echo "\${PUB_KEY}" >> "\${TARGET_AUTH_FILE}"
    echo "publickey append successful."
    exit 0
fi
EOF

chmod +x "${HELPER_SCRIPT_PATH}"
echo "[√] helper script created at ${HELPER_SCRIPT_PATH}"

# --- 4. configure sudoers ---
SUDOERS_FILE="/etc/sudoers.d/99-fuzzer-tunnel"
SUDOERS_RULE="${APP_USER} ALL=(${TUNNEL_USER}) NOPASSWD: ${HELPER_SCRIPT_PATH} ${TUNNEL_USER} *"

echo "[+] configuring sudoers rules..."
# write the rules to a new file in the /etc/sudoers.d/ directory, this is the safest way.
# `*` allows passing any public key as an argument
echo "${SUDOERS_RULE}" > "${SUDOERS_FILE}"
chmod 440 "${SUDOERS_FILE}"
echo "[√] sudoers rules written to ${SUDOERS_FILE}"

echo ""
echo "✅ environment setup successfully!"
