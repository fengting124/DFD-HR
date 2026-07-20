#!/usr/bin/env bash

set -euo pipefail
: "${DFDHR_PYTHON:?Set DFDHR_PYTHON}"
exec "$DFDHR_PYTHON" "$(dirname "$0")/experiment_lifecycle.py" capture "$@"
