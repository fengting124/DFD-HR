# DFD-HR Notebooks

Source notebooks are small, output-free diagnostics. Executed copies, runtime files, URLs, tokens, and logs stay outside Git under `${DFDHR_RUNTIME_ROOT}`.

## Local setup

Use the existing DFD-HR Python environment and user-owned runtime storage:

```bash
export DFDHR_REPO_ROOT=/path/to/DFD-HR
export DFDHR_PYTHON=/path/to/dfd-hr/bin/python
export DFDHR_DATA_ROOT=/path/to/read-only/data
export DFDHR_RUNTIME_ROOT=/path/to/user/runtime

./scripts/register_jupyter_kernel.sh
./scripts/start_jupyter_local.sh
```

The server binds only to `127.0.0.1`. Access from another machine must use an SSH tunnel that follows local infrastructure policy. Do not paste Jupyter URLs or tokens into tracked files.

## Execute a source notebook

Create a timestamped validation directory outside Git and execute with the registered kernel:

```bash
run_dir="${DFDHR_RUNTIME_ROOT}/jupyter-validation/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$run_dir"

"$DFDHR_PYTHON" -m jupyter nbconvert \
  --to notebook \
  --execute \
  --ExecutePreprocessor.kernel_name=dfd-hr \
  --ExecutePreprocessor.timeout=300 \
  notebooks/00_environment_and_paths.ipynb \
  --output "$run_dir/00_environment_and_paths.executed.ipynb"
```

Before committing a source notebook, clear its outputs:

```bash
"$DFDHR_PYTHON" -m jupyter nbconvert \
  --clear-output \
  --inplace \
  notebooks/00_environment_and_paths.ipynb
```

Notebook responsibilities remain separate. Do not add model loading, dataset traversal, training, or evaluation to the environment and path audit.
