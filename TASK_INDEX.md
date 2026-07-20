# TASK_INDEX.md

This file is the active task-level handoff for DFD-HR. Update it whenever a task changes state or new evidence is produced.

## Current milestone

**Milestone:** establish a new controller on `vipsl7`, inventory `vipsl11` and `vipsl12`, then complete the bounded DFD-HR preflight and smoke-test sequence.

**Current branch:** `infra/vipsl7-controller-vipsl11-12-audit`

**Branch base:** historical audit head `ca684d3`.

**Do not merge this branch into `main` until the public-document privacy review is complete.**

## Status legend

- `TODO`: not started.
- `ACTIVE`: currently being performed.
- `BLOCKED`: cannot proceed; record the blocking evidence.
- `DONE`: completed with reproducible evidence.
- `SUPERSEDED`: replaced by a newer task or report.

## P0 — Controller migration and new-node inventory

### T0.1 Establish `vipsl7` as controller

**Status:** TODO

Required checks:

- [ ] SSH login to `vipsl7` is stable.
- [ ] Project path `/home/fengting/Experiments/DDF/DFD-HR` is visible.
- [ ] `git fetch --all --prune` succeeds.
- [ ] Branch `infra/vipsl7-controller-vipsl11-12-audit` can be checked out.
- [ ] Git worktree is clean.
- [ ] Existing audit scripts under `/scratch/fengting/server-resource-audit/scripts/` are present or restored from an approved source.
- [ ] The DFD-HR Conda Python path exists and imports the required stack.
- [ ] Unit tests pass on `vipsl7`.
- [ ] vipsl7 can resolve and attempt non-interactive SSH to the target aliases.
- [ ] The controller writes audit artifacts only under the owner's directory.

Evidence to record:

```text
hostname
Git branch and commit
Python executable
PyTorch/CUDA versions
pytest summary
controller scratch free space
```

### T0.2 Recheck `vipsl6` failure

**Status:** BLOCKED

Current observation: remote SSH connections time out while other nodes remain reachable. Treat `vipsl6` as unavailable.

Allowed follow-up:

- [ ] From `vipsl7`, perform one bounded reachability check to `vipsl6`.
- [ ] Record DNS resolution, TCP result, and timestamp.
- [ ] Do not repeatedly retry or change network/SSH configuration.

Exit condition: mark reachable only after a successful fresh connection. Do not restore it as archive or training target based on historical data.

### T0.3 Expand audit targets to `vipsl1` through `vipsl12`

**Status:** TODO

Modify the target list on a dedicated infrastructure branch or in the local audit configuration. Do not mix this with model changes.

For every target, collect only bounded, public system state:

- [ ] DNS/alias resolution and SSH reachability.
- [ ] Canonical hostname.
- [ ] CPU model, sockets, physical cores, and threads.
- [ ] Total and available RAM.
- [ ] GPU count, model, VRAM, driver, and CUDA compatibility.
- [ ] Current GPU utilization and whether a new exclusive job is safe.
- [ ] `/scratch` filesystem type, mount, total, free, percentage used, and inode use.
- [ ] Project shared-home visibility.
- [ ] Owner's local working/output directory availability.
- [ ] Scheduler/monitoring capability, without installing anything.

Do not collect another user's command arguments, environment, shell history, project contents, dataset contents, keys, or credentials.

### T0.4 Inventory `vipsl11`

**Status:** TODO

Hardware checklist:

- [ ] CPU/RAM inventory.
- [ ] GPU model/count/VRAM.
- [ ] NVIDIA driver and reported CUDA compatibility.
- [ ] Current GPU state.
- [ ] `/scratch` capacity and free space.
- [ ] Root filesystem free space.
- [ ] Network reachability from `vipsl7`.

DFD-HR readiness checklist:

- [ ] `/home/fengting/Experiments/DDF/DFD-HR` is visible.
- [ ] `/scratch/datasets/deepfake` exists.
- [ ] Required dataset directories exist.
- [ ] Dataset JSON registry is present.
- [ ] Sampled JSON paths resolve locally.
- [ ] DFD-HR Conda environment exists, or absence is recorded without installing/copying it.
- [ ] Official checkpoint exists and its hash matches the approved reference.
- [ ] Unit tests pass if the environment already exists.
- [ ] Output space is sufficient for the intended smoke test or training role.

### T0.5 Inventory `vipsl12`

**Status:** TODO

Use the same checklist and safety boundary as T0.4. Keep results separate; do not infer parity between the two new nodes.

### T0.6 Dataset parity and path validation

**Status:** TODO

Minimum required datasets for the first reproduction phase:

- [ ] FaceForensics++ c23 train/validation/test paths.
- [ ] Celeb-DF-v2.
- [ ] DFD/DeepFakeDetection if referenced by the current test registry.
- [ ] DFDC.
- [ ] DFDCP.
- [ ] DeeperForensics/test_DFR if referenced.
- [ ] WildDeepfake/test_WDF if referenced.
- [ ] FFIW/test_FFIW if referenced.
- [ ] Required DF40 method subsets.

Validation method:

1. Compare directory presence and bounded counts.
2. Hash the project's JSON registry files.
3. Sample paths from train, validation, and test entries.
4. Confirm sampled paths resolve on each node.
5. Record missing datasets and broken paths explicitly.

Do not copy datasets during the audit. Dataset transfer requires a separate size estimate, source/target approval, bandwidth limit, resumable copy, and post-copy verification.

### T0.7 Produce sanitized node report

**Status:** TODO

Planned public-safe report:

```text
docs/controller_migration_and_node_audit_results.md
```

The report may contain:

- Timestamped reachability.
- Generic hardware configuration.
- Owner-visible free capacity.
- Dataset readiness status.
- DFD-HR environment/test readiness.
- Scheduling recommendation.

It must not contain:

- Internal IPs or credentials.
- Other users' PIDs, command lines, personal directory sizes, or file contents.
- Raw SSH configuration or private host-key material.
- Secrets, tokens, or environment variables.

## P1 — Reproduction correctness blockers

These tasks are required before a full self-trained checkpoint is considered valid.

### T1.1 Explicit validation protocol

**Status:** TODO

- [ ] Configure FaceForensics++ validation explicitly.
- [ ] Reject a missing validation split instead of silently falling back to test.
- [ ] Add tests covering valid and invalid split behavior.

### T1.2 Final evaluation return/save behavior

**Status:** TODO

- [ ] Return current final-test metrics when `save_best=False`.
- [ ] Save dataset-level and average metrics as stable JSON.
- [ ] Add a regression test.

### T1.3 AMP and gradient accumulation

**Status:** TODO

- [ ] Add configuration-driven AMP.
- [ ] Add gradient accumulation.
- [ ] Log per-GPU micro-batch, world size, accumulation steps, and effective batch size.
- [ ] Verify finite loss and gradients.

### T1.4 Resumable checkpoints

**Status:** TODO

- [ ] Save and restore model, optimizer, scheduler, epoch, best metrics, scaler, and RNG state.
- [ ] Use atomic local writes.
- [ ] Keep only required `best` and `last` checkpoints by default.
- [ ] Add checkpoint round-trip tests.

### T1.5 DDP validation synchronization

**Status:** TODO

- [ ] Prevent rank desynchronization during validation.
- [ ] Decide and document distributed evaluation or rank-0 evaluation with barriers.
- [ ] Run a two-process bounded smoke test.

### T1.6 Official-checkpoint evaluation calibration

**Status:** TODO

- [ ] Strict-load official weight.
- [ ] Run a small deterministic evaluation subset.
- [ ] Run at least one complete external dataset evaluation.
- [ ] Record checkpoint hash, config hash, dataset JSON hashes, and metrics.

## P2 — Jupyter and experiment maintainability

### T2.1 Register the existing DFD-HR environment as a kernel

**Status:** TODO

Target kernel Python:

```text
/scratch/fengting/miniconda3/envs/dfd-hr/bin/python
```

- [ ] Verify `ipykernel` availability.
- [ ] Register `Python (DFD-HR)` without replacing the environment.
- [ ] Store runtime/config files under the owner's scratch area.
- [ ] Bind the server to localhost and use an SSH tunnel.

### T2.2 Add standard notebooks

**Status:** TODO

- [ ] `00_environment_and_paths.ipynb`
- [ ] `01_checkpoint_strict_load.ipynb`
- [ ] `02_dataset_protocol_audit.ipynb`
- [ ] `03_single_gpu_memory_smoke.ipynb`
- [ ] `04_official_weight_eval.ipynb`
- [ ] `05_training_monitor.ipynb`

Source notebooks belong in Git. Executed notebooks and large outputs belong in `/scratch/fengting/runs/...`.

### T2.3 Experiment registry and run template

**Status:** TODO

Each run must record:

- Unique run ID.
- Objective and single changed variable.
- Branch and commit.
- Frozen resolved config and hash.
- Environment and GPU metadata.
- Dataset manifest/JSON hashes.
- Command and output directory.
- Metrics log.
- Checkpoint hashes.
- Final summary and failure reason if applicable.

## P3 — Smoke tests and full reproduction

### T3.1 Single-GPU memory smoke

**Status:** BLOCKED by P0 and P1 preflight

Run on the best verified node, beginning with micro-batch 1. Record FP32 and AMP peak memory, step time, finite loss, and gradient presence.

### T3.2 Two-GPU DDP smoke

**Status:** BLOCKED by T3.1 and T1.5

Run a bounded number of steps. Verify synchronization, logging, checkpoint save/load, and clean termination.

### T3.3 Mini reproduction

**Status:** BLOCKED by T3.2

Use a fixed subset for one to three epochs. This is a pipeline validation result, not a paper result.

### T3.4 Full training

**Status:** BLOCKED by T3.3

Train from CLIP initialization without loading the released DFD-HR checkpoint. Use validation-selected `best` and a resumable `last` checkpoint.

### T3.5 Cross-dataset evaluation

**Status:** BLOCKED by T3.4

Freeze the selected checkpoint and evaluate all approved target datasets once. Store frame-level and video-level metrics with exact data and code provenance.

## Node selection rules

Do not select a node from historical tables alone. Immediately before launch, compare:

1. Per-card VRAM and measured DFD-HR peak memory.
2. Number of available GPUs.
3. Current exclusive availability.
4. Local dataset readiness.
5. Local output capacity and safety margin.
6. Runtime environment readiness.
7. Archive destination reachability.

Current provisional roles:

| Node | Provisional role | Reason | Required recheck |
| --- | --- | --- | --- |
| vipsl7 | controller and fallback smoke node | reachable candidate with prepared DFD-HR assets in the historical audit | full controller preflight |
| vipsl6 | unavailable | current SSH timeout | one bounded reachability check |
| vipsl9 | unavailable for new work | continuously occupied | live GPU check and coordination |
| vipsl10 | memory-oriented smoke/inference | 24 GiB GPUs, but historically low scratch capacity | GPU state and at least 50 GiB normal-user scratch free |
| vipsl11 | unknown | newly added | complete hardware/data/environment audit |
| vipsl12 | unknown | newly added | complete hardware/data/environment audit |

## New-machine Codex handoff

A new Codex session should execute this sequence:

```bash
cd /home/fengting/Experiments/DDF/DFD-HR
git fetch --all --prune
git switch infra/vipsl7-controller-vipsl11-12-audit
git status --short
git log --oneline --decorate -12
```

Then read:

```text
AGENTS.md
TASK_INDEX.md
docs/controller_migration_and_node_audit_plan.md
docs/server_resource_audit_vipsl_20260720.md
```

The first action is T0.1. Do not begin model training and do not make system-level changes during the node audit.

## Definition of done for this branch

- [ ] `vipsl7` is verified as controller.
- [ ] `vipsl6` current status is timestamped.
- [ ] `vipsl11` and `vipsl12` inventories are complete.
- [ ] Dataset and environment readiness are recorded separately for both nodes.
- [ ] A sanitized scheduling recommendation exists.
- [ ] `AGENTS.md` and this task index match the actual repository state.
- [ ] No secrets or sensitive user/process details were committed.
- [ ] The next smoke-test node is selected using fresh evidence.
