#!/usr/bin/env bash

set -euo pipefail

: "${DFDHR_PYTHON:?Set DFDHR_PYTHON to the existing DFD-HR Python executable}"

if [[ "$DFDHR_PYTHON" != /* ]]; then
    echo "ERROR: DFDHR_PYTHON must be an absolute path" >&2
    exit 1
fi

if [[ ! -x "$DFDHR_PYTHON" ]]; then
    echo "ERROR: DFDHR_PYTHON is not executable" >&2
    exit 1
fi

if ! "$DFDHR_PYTHON" -c 'import ipykernel, jupyter_core' >/dev/null 2>&1; then
    echo "ERROR: ipykernel and Jupyter must already exist in DFDHR_PYTHON" >&2
    exit 1
fi

"$DFDHR_PYTHON" -m ipykernel install \
    --user \
    --name dfd-hr \
    --display-name "Python (DFD-HR)"

kernel_dir="$({
    "$DFDHR_PYTHON" -m jupyter kernelspec list --json
} | "$DFDHR_PYTHON" -c '
import json
import sys

specs = json.load(sys.stdin)["kernelspecs"]
if "dfd-hr" not in specs:
    raise SystemExit("ERROR: dfd-hr kernelspec was not registered")
print(specs["dfd-hr"]["resource_dir"])
')"

kernel_python="$("$DFDHR_PYTHON" -c '
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    print(json.load(handle)["argv"][0])
' "$kernel_dir/kernel.json")"

if [[ "$kernel_python" != "$DFDHR_PYTHON" ]]; then
    echo "ERROR: dfd-hr kernel does not use DFDHR_PYTHON" >&2
    exit 1
fi

echo "Registered dfd-hr kernelspec: $kernel_dir"
echo "Kernel interpreter: $kernel_python"
