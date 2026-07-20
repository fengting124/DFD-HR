# AGENTS.md

This file is the repository entry point for Codex and other coding agents. Read it before changing code, running experiments, or auditing laboratory nodes.

## Project objective

DFD-HR is being reproduced and evaluated as a generalizable deepfake detector. The immediate objective is to establish a protocol-correct, resumable, storage-bounded training and evaluation workflow before launching a full reproduction run.

## Required startup sequence

On every new machine or after losing chat context:

1. Run `git status --short`, `git branch --show-current`, and `git log --oneline --decorate -12`.
2. Read this file, `TASK_INDEX.md`, `README.md`, and the latest relevant document under `docs/`.
3. Confirm the current Git worktree is clean before creating a task branch.
4. Confirm the active Python executable, Conda environment, CUDA availability, dataset paths, checkpoint hashes, and output path before running code.
5. Run the existing unit tests before modifying training or evaluation behavior.
6. Do not start a full training run until the preflight, single-GPU smoke test, checkpoint round trip, and DDP smoke test are complete.

## Current infrastructure state

The previous resource audit was controlled from `vipsl6` and covered `vipsl1` through `vipsl10`. That report is historical evidence, not a live scheduler.

Current operational assumptions that must be revalidated:

- `vipsl6` is currently unavailable and must not be treated as controller, archive target, or training candidate until connectivity is restored.
- `vipsl7` is the selected replacement controller candidate.
- `vipsl11` and `vipsl12` are newly added nodes. Their CPU, RAM, GPU, driver, CUDA compatibility, storage, datasets, environment, and current utilization are unknown.
- `vipsl9` has been continuously occupied and must not be scheduled without a fresh idle check and coordination.
- `vipsl10` has 24 GiB GPUs but historically had very limited normal-user scratch capacity. Recheck free space immediately before every run and keep outputs tightly bounded.
- `/home/fengting` may be shared while each node's `/scratch` is local. Never infer dataset or environment parity from identical path names.

See `TASK_INDEX.md` and `docs/controller_migration_and_node_audit_plan.md` for the active plan.

## Safety and privacy boundary

Resource discovery begins as read-only inspection.

Allowed without additional approval:

- SSH reachability and hostname resolution checks.
- Public system inventory: CPU model/count, RAM, mounted filesystems, free capacity, inode use, GPU model/count, driver, CUDA compatibility, and current GPU utilization.
- Verification that the owner's project, Conda environment, dataset root, JSON registry, and official weights exist.
- Sampling paths referenced by the project's own JSON files.
- Running project unit tests and bounded smoke tests in the owner's directories.

Require explicit approval before execution:

- Installing packages or services system-wide.
- Changing SSH, firewall, scheduler, crontab, systemd, CUDA, driver, mount, or monitoring configuration.
- Copying large datasets or environments to a new node.
- Deleting files, checkpoints, caches, or another user's data.
- Preempting, terminating, or sharing another user's GPU process.
- Starting a long training run.

Never commit credentials, private keys, tokens, internal IP addresses, raw SSH configuration, other users' file contents, command lines, shell history, or sensitive process details. This repository is public. Public documents should contain only the minimum operational metadata needed for reproducibility.

## Git workflow

Do not push directly to `main`.

Use focused branches:

- `fix/*` for correctness bugs.
- `feat/*` for reusable training or evaluation capabilities.
- `infra/*` for Jupyter, run management, node audit, and artifact synchronization.
- `exp/*` for a single experimental hypothesis.
- `docs/*` for documentation-only work.

Each commit should contain one logical change. Do not mix node audit changes with model, dataset, or training changes.

Before committing:

```bash
git status --short
git diff
python -m pytest -q
```

Do not use `git add .` without reviewing every untracked file. Model weights, datasets, logs, notebook execution outputs, caches, and server snapshots must remain outside Git.

## Experiment rules

- Every experiment receives a unique run ID, frozen configuration, Git commit, environment record, dataset manifest/checksum, command, metrics log, and summary.
- Jupyter is for environment checks, dataset audits, one-batch debugging, memory measurement, visualization, and monitoring.
- Formal training is launched independently through `tmux` and `torchrun`; it must not depend on a browser or notebook kernel remaining alive.
- Store active outputs under the current node's `/scratch/fengting` area, not in the repository or shared home.
- Keep only required checkpoints such as `best` and `last`; disable feature dumps unless the experiment explicitly requires them.
- Copy artifacts to a verified reachable archive node only after local atomic writes, then compare hashes before deleting any local copy.

## Protocol correctness requirements

Before a full reproduction, verify and, where needed, fix:

- An explicit FaceForensics++ validation split is used for model selection.
- Validation never silently falls back to the test split.
- Final test metrics return and save the metrics from the current evaluation.
- AMP and gradient accumulation preserve the intended effective batch size.
- Checkpoints restore model, optimizer, scheduler, epoch, best metrics, and RNG state.
- DDP validation does not desynchronize ranks.
- The official checkpoint loads strictly and the evaluation pipeline is calibrated before self-training.

## Handoff discipline

At the end of each task:

1. Update `TASK_INDEX.md` status and evidence.
2. Record the exact branch and commit.
3. Add links or paths to the relevant report, configuration, test output, and run summary.
4. State what remains unknown and the next bounded action.
5. Leave the worktree clean or explicitly document uncommitted local work.

Historical resource measurements must be labelled with their timestamp. Never describe a historical GPU state as current without rerunning the live check.
