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

## Official checkpoint notebooks

The strict-load and bounded-evaluation notebooks also require:

```bash
export DFDHR_OFFICIAL_CHECKPOINT=/path/to/official-checkpoint.pth
```

`01_checkpoint_strict_load.ipynb` constructs the published vision architecture offline and verifies a complete `strict=True` load. `04_official_weight_eval.ipynb` defaults to an eight-sample external-dataset calibration and writes its report under `${DFDHR_RUNTIME_ROOT}/jupyter-validation/`.

Select a different dataset or an intentional complete evaluation through runtime-only environment variables:

```bash
export DFDHR_EVAL_DATASET=e4s_ff
export DFDHR_EVAL_MAX_SAMPLES=0
```

The value `0` means no sample cap. These variables and executed notebooks are local run metadata and must not be committed.

## Dataset protocol audit

`02_dataset_protocol_audit.ipynb` verifies explicit FaceForensics++ c23 train/validation/test metadata, split disjointness, current loader behavior, JSON hashes, and a bounded set of referenced files. It defaults to `Celeb-DF-v2` for the external test-role check. Override that role without editing the source notebook:

```bash
export DFDHR_AUDIT_EXTERNAL_DATASET=DFDC
```

The audit parses registry JSON but never walks the dataset directory tree or decodes images.

## Single-GPU smoke test

`03_single_gpu_memory_smoke.ipynb` invokes the bounded smoke harness once for FP32 and once for AMP. Each process reads exactly two class-balanced FaceForensics++ samples with micro-batch 1 and no augmentation, then verifies unscaled gradients and a complete checkpoint round-trip. Reports, logs, and checkpoints remain under `${DFDHR_RUNTIME_ROOT}/jupyter-validation/`.

This notebook performs optimizer steps, but never enters the epoch training loop. It is a correctness and capacity check, not a research experiment or training result.
