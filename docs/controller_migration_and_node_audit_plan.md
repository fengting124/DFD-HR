# Controller Migration to vipsl7 and vipsl11-vipsl12 Audit Plan

## Purpose

The earlier DFD-HR server audit was controlled from `vipsl6` and covered `vipsl1` through `vipsl10`. `vipsl6` is now unavailable by SSH timeout, so the operational controller must move to `vipsl7`. Two new nodes, `vipsl11` and `vipsl12`, must be inventoried before they are used for datasets, Jupyter, smoke tests, training, evaluation, or artifact storage.

This document is a plan. It does not claim that the new nodes are ready.

## Source of truth

Read in this order:

1. `AGENTS.md`
2. `TASK_INDEX.md`
3. This document
4. `docs/server_resource_audit_vipsl_20260720.md` as historical evidence
5. Current live audit artifacts generated from `vipsl7`

Historical resource values must not be used as current scheduling evidence.

## Branch and change isolation

Perform controller and node-audit work on:

```text
infra/vipsl7-controller-vipsl11-12-audit
```

Do not add model changes, dataset protocol changes, AMP, checkpoint logic, or DDP changes to this branch. Those belong in separate `fix/*` or `feat/*` branches after the infrastructure inventory is complete.

## Phase A — Bootstrap vipsl7

### A1. Confirm controller identity and repository

From `vipsl7`:

```bash
hostname
cd /home/fengting/Experiments/DDF/DFD-HR
git fetch --all --prune
git switch infra/vipsl7-controller-vipsl11-12-audit
git status --short
git log --oneline --decorate -12
```

Expected result:

- Canonical host is `vipsl7`.
- The expected project path is visible.
- The task branch exists.
- The worktree is clean.

Stop if the shared project path resolves to unexpected content or if the worktree contains unexplained changes.

### A2. Confirm DFD-HR runtime

Use the existing environment only:

```bash
DFDHR_PY=/scratch/fengting/miniconda3/envs/dfd-hr/bin/python

"${DFDHR_PY}" - <<'PY'
import sys
import torch
print("python", sys.executable)
print("torch", torch.__version__)
print("cuda_runtime", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("gpu_count", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY
```

Then run the project unit tests:

```bash
"${DFDHR_PY}" -m pytest -q
```

Do not install or upgrade packages during this phase. Record a missing or broken environment as a finding.

### A3. Confirm owner-controlled audit workspace

Expected local workspace:

```text
/scratch/fengting/server-resource-audit/
```

Required subdirectories should include, or be recreated only within the owner's area:

```text
scripts/
snapshots/
logs/
reports/
```

Do not write audit artifacts into the Git repository, shared project tree, another user's directory, or a system directory.

### A4. Confirm vipsl7 can act as SSH controller

Use one bounded non-interactive connection attempt per target:

```bash
ssh -o BatchMode=yes \
    -o ConnectTimeout=8 \
    -o StrictHostKeyChecking=yes \
    TARGET_HOST true
```

Do not disable host-key checking and do not repeatedly retry failed hosts.

## Phase B — Expand the target set

The new target set is:

```text
vipsl1 vipsl2 vipsl3 vipsl4 vipsl5 vipsl6
vipsl7 vipsl8 vipsl9 vipsl10 vipsl11 vipsl12
```

The audit script must treat each host independently. A failed connection must not abort all remaining targets.

Record for each target:

- Timestamp.
- Resolution result.
- SSH reachability.
- High-level failure class: timeout, authentication failure, host-key failure, or reachable.

Do not include internal IP addresses in the public report.

## Phase C — Hardware and storage inventory

For reachable nodes, collect bounded public system information.

### CPU and memory

Recommended commands:

```bash
lscpu
free -h
uptime
```

Extract:

- CPU model.
- Sockets.
- Physical cores.
- Logical threads.
- Total and available RAM.
- Current load.

### GPU

Recommended command:

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.free,utilization.gpu,utilization.memory,temperature.gpu,driver_version --format=csv,noheader,nounits
```

Record:

- GPU count and model.
- Per-card VRAM.
- Driver version.
- CUDA compatibility reported by `nvidia-smi`.
- Current utilization and free memory.

A point-in-time idle result is not a reservation. Recheck immediately before any experiment.

### Storage

Recommended commands:

```bash
df -hT /scratch /
df -ih /scratch /
findmnt -T /scratch
```

Record:

- Filesystem and mount.
- Total, used, and normal-user available capacity.
- Percentage used.
- Inode use.

Do not enumerate or size other users' directories in the public audit. The purpose is to determine whether the owner's planned workload can run safely.

## Phase D — DFD-HR data and environment readiness

Treat each node's `/scratch` as local even when path names match.

### D1. Shared project visibility

Verify:

```bash
test -d /home/fengting/Experiments/DDF/DFD-HR
```

Record the checked-out commit only after entering the project directory.

### D2. Dataset root

Expected root:

```text
/scratch/datasets/deepfake
```

Check directory presence for required first-phase datasets. Do not read arbitrary dataset media files merely to estimate readiness.

Suggested first-phase inventory:

```text
FaceForensics++
Celeb-DF-v2
DeepFakeDetection or the local DFD equivalent
DFDC
DFDCP
DeeperForensics or the local test_DFR equivalent
WildDeepfake or the local test_WDF equivalent
FFIW or the local test_FFIW equivalent
required DF40 method subsets
```

Local names may differ. The project's JSON registry is authoritative for paths actually consumed by the code.

### D3. JSON registry and sampled paths

For each required JSON file:

1. Record SHA256.
2. Count train/validation/test video entries where applicable.
3. Select a small deterministic sample from each required split.
4. Confirm sampled frame paths exist locally.
5. Record broken-path counts from the bounded sample.

Do not silently reinterpret test as validation.

### D4. Conda environment

Expected Python:

```text
/scratch/fengting/miniconda3/envs/dfd-hr/bin/python
```

If absent, record `environment_missing`. Do not copy or install the environment until the size, source, target space, transfer method, and approval are confirmed.

### D5. Official checkpoint

Verify:

- Expected file exists in the owner's area.
- File size is plausible.
- SHA256 matches the approved reference.
- Strict model load is deferred to the bounded model preflight notebook or test script.

Do not commit the checkpoint or its private storage location.

## Phase E — Specific checks for vipsl11 and vipsl12

Run the complete checklist separately for each node.

Minimum result table:

| Field | vipsl11 | vipsl12 |
| --- | --- | --- |
| Reachable | unknown | unknown |
| CPU/threads | unknown | unknown |
| RAM | unknown | unknown |
| GPU model/count | unknown | unknown |
| VRAM per card | unknown | unknown |
| Driver/CUDA compatibility | unknown | unknown |
| Current GPU state | unknown | unknown |
| Scratch total/free | unknown | unknown |
| Project visible | unknown | unknown |
| Required datasets | unknown | unknown |
| JSON sampled paths | unknown | unknown |
| DFD-HR environment | unknown | unknown |
| Unit tests | unknown | unknown |
| Official weight | unknown | unknown |
| Provisional role | unknown | unknown |

Do not assign either node a role until all material unknowns are resolved.

Possible roles after evidence:

- Controller only.
- Jupyter and debugging.
- Single-GPU smoke test.
- Multi-GPU training.
- Evaluation worker.
- Artifact archive.
- Not ready.

## Phase F — vipsl6 bounded recheck

From `vipsl7`, perform one current reachability check to `vipsl6` and record the timestamp and failure class.

Do not:

- Change SSH or VPN configuration.
- Add insecure host-key exceptions.
- Loop repeated retries.
- Treat historical hardware and disk data as current readiness.

If it remains unreachable, remove it from current scheduling and archive plans.

## Phase G — Select current roles

Use fresh evidence and select:

1. Controller node.
2. Jupyter/debugging node.
3. Single-GPU memory-smoke node.
4. Two-GPU or four-GPU DDP-smoke node.
5. Full-training candidate.
6. Artifact archive target.

Selection criteria, in order:

- Reachability and exclusive GPU availability.
- Per-card VRAM versus measured DFD-HR peak memory.
- Local dataset readiness.
- Runtime readiness.
- Scratch free-space safety margin.
- GPU count and expected throughput.
- Archive-path reliability.

Do not select a full-training node before the single-GPU smoke test measures actual peak memory.

## Phase H — Required outputs

Keep raw artifacts locally under the owner's scratch area:

```text
/scratch/fengting/server-resource-audit/
```

Commit only sanitized outputs:

```text
TASK_INDEX.md
docs/controller_migration_and_node_audit_results.md
```

The sanitized result document should include:

- Audit timestamp and controller.
- Reachability matrix.
- Generic hardware and owner-visible storage.
- Dataset/environment readiness.
- Current scheduling recommendation.
- Unknowns and blocked items.
- Exact Git branch and commit used for the audit scripts.

## Completion gate

This infrastructure task is complete only when:

- `vipsl7` is verified as controller.
- `vipsl11` and `vipsl12` have separate complete inventories.
- Required dataset JSON hashes and sampled-path checks are recorded.
- Environment and official-weight readiness are known for both new nodes.
- `vipsl6` has a current timestamped status.
- A sanitized node-role decision is documented.
- `TASK_INDEX.md` is updated with evidence and next actions.
- No system changes, large transfers, or long training jobs were performed without approval.

## Next bounded action

On `vipsl7`, complete Phase A only. If Phase A passes, run the read-only target reachability and inventory audit for `vipsl1` through `vipsl12`. Do not start DFD-HR training as part of the audit task.
