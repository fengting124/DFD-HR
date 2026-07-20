#!/usr/bin/env bash

set -euo pipefail
umask 077

: "${DFDHR_PYTHON:?Set DFDHR_PYTHON to the existing DFD-HR Python executable}"
: "${DFDHR_RUNTIME_ROOT:?Set DFDHR_RUNTIME_ROOT to a user-owned scratch directory}"

if [[ "$DFDHR_PYTHON" != /* || ! -x "$DFDHR_PYTHON" ]]; then
    echo "ERROR: DFDHR_PYTHON must be an absolute executable path" >&2
    exit 1
fi

if [[ "$DFDHR_RUNTIME_ROOT" != /* ]]; then
    echo "ERROR: DFDHR_RUNTIME_ROOT must be an absolute path" >&2
    exit 1
fi

if ! "$DFDHR_PYTHON" -c 'import jupyterlab' >/dev/null 2>&1; then
    echo "ERROR: JupyterLab must already exist in DFDHR_PYTHON" >&2
    exit 1
fi

jupyter_port="${DFDHR_JUPYTER_PORT:-8888}"
if [[ ! "$jupyter_port" =~ ^[0-9]+$ ]] || ((jupyter_port < 1 || jupyter_port > 65535)); then
    echo "ERROR: DFDHR_JUPYTER_PORT must be an integer from 1 to 65535" >&2
    exit 1
fi

for protected_root in "${DFDHR_REPO_ROOT:-}" "${DFDHR_DATA_ROOT:-}"; do
    if [[ -n "$protected_root" ]]; then
        case "$DFDHR_RUNTIME_ROOT/" in
            "$protected_root/"*)
                echo "ERROR: DFDHR_RUNTIME_ROOT cannot be inside the repository or data root" >&2
                exit 1
                ;;
        esac
    fi
done

export JUPYTER_RUNTIME_DIR="$DFDHR_RUNTIME_ROOT/jupyter/runtime"
export JUPYTER_CONFIG_DIR="$DFDHR_RUNTIME_ROOT/jupyter/config"
mkdir -p "$JUPYTER_RUNTIME_DIR" "$JUPYTER_CONFIG_DIR"

exec "$DFDHR_PYTHON" -m jupyter lab \
    --no-browser \
    --ServerApp.ip=127.0.0.1 \
    --ServerApp.port="$jupyter_port" \
    --ServerApp.port_retries=0
