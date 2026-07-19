# Laboratory Server Resource Audit

> Superseded: this run used the incorrect aliases `ssh1..ssh10`. Use `server_resource_audit_vipsl_20260720.md` for scheduling decisions.

Audit run: `20260720_003845`

Controller: `vipsl6` (`ssh6`)

Snapshot times: `2026-07-20T00:39:02+08:00` and `2026-07-20T00:40:13+08:00`

Task scope: analysis only; no training code, configuration, dependency, dataset, process, GPU, SSH, or server setting was changed.

## Scope and method

The audit tested `ssh1` through `ssh10` once with non-interactive SSH, `BatchMode=yes`, an 8-second connection timeout, and host-key verification left enabled. Reachable nodes were sampled twice. The second sample was 71 seconds after the first. Process attribution was restricted to username, PID, executable name, elapsed time, start time, GPU UUID, and GPU memory. No full command line, environment, shell history, project file, dataset, or other user's directory was read.

`ssh6` did not resolve as a DNS name from the current environment. The controller's canonical local hostname `vipsl6` resolved and accepted the same non-interactive SSH authentication, so the report records ssh6 as reachable through `vipsl6`. The other aliases had no resolvable hostname in `ssh -G` beyond the alias itself.

## Node connectivity

| Host | Reachable | Result |
| --- | --- | --- |
| ssh1 | No | Hostname resolution failed; no retry performed |
| ssh2 | No | Hostname resolution failed; no retry performed |
| ssh3 | No | Hostname resolution failed; no retry performed |
| ssh4 | No | Hostname resolution failed; no retry performed |
| ssh5 | No | Hostname resolution failed; no retry performed |
| ssh6 | Yes | Connected non-interactively through controller hostname `vipsl6` |
| ssh7 | No | Hostname resolution failed; no retry performed |
| ssh8 | No | Hostname resolution failed; no retry performed |
| ssh9 | No | Hostname resolution failed; no retry performed |
| ssh10 | No | Hostname resolution failed; no retry performed |

These failures do not prove that ssh1 to ssh5 or ssh7 to ssh10 are powered off. They establish that the current ssh6 environment could not resolve their configured aliases during this audit.

## Current GPU summary

| Host | GPU configuration | Free now | Occupied | Current users |
| --- | --- | ---: | ---: | --- |
| ssh1-ssh5 | Unknown; unreachable | 0 confirmed | 0 confirmed | Unknown |
| ssh6 | 4 × NVIDIA GeForce RTX 2080 Ti, 11264 MiB each | 4 | 0 | None |
| ssh7-ssh10 | Unknown; unreachable | 0 confirmed | 0 confirmed | Unknown |

All four ssh6 GPUs had no compute process in either snapshot, 1 MiB reported memory use, 0% GPU utilization, and 0% memory utilization. They satisfy the audit's `available_now` rule. This is point-in-time evidence and must be checked again immediately before training.

| GPU | UUID | Free/total MiB | Utilization | Temperature | Power | Status |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 0 | `GPU-215691e0-79ad-39de-e8e4-25173376ce8d` | 10822/11264 | 0% | 59 C | 22.19 W | `available_now` |
| 1 | `GPU-e5c52e32-3356-c8cd-91d4-9e57a4ff9e4f` | 10822/11264 | 0% | 57 C | 3.86 W | `available_now` |
| 2 | `GPU-cb333ddd-04c3-9c86-670a-3232910e4fea` | 10822/11264 | 0% | 56 C | 13.38 W | `available_now` |
| 3 | `GPU-06d47343-ccff-95c6-e9aa-1ef190cffc70` | 10822/11264 | 0% | 56 C | 21.78 W | `available_now` |

ssh6 uses NVIDIA driver `570.195.03`; `nvidia-smi` reports CUDA compatibility `12.8`. `nvcc` was unavailable in the login environment, and its default `python3` did not contain PyTorch. These environment results do not limit the dedicated DFD-HR Conda environment.

## CPU, memory, load, and storage

| Host | CPU | RAM | Load 1m | Scratch free |
| --- | --- | --- | ---: | ---: |
| ssh1-ssh5 | Unknown | Unknown | Unknown | Unknown |
| ssh6 | 2 × Intel Xeon E5-2620 v4; 16 physical cores, 32 threads | 62 GiB total, 56 GiB available | 0.09 | 2.0 TiB of 3.6 TiB |
| ssh7-ssh10 | Unknown | Unknown | Unknown | Unknown |

At the second snapshot, `/scratch` was an ext4 filesystem on `/dev/sdb1`, 42% used. Root had 321 GiB available.

## Historical visibility

| Host | Slurm | Monitoring | User-level history | Conclusion |
| --- | --- | --- | --- | --- |
| ssh1-ssh5 | Unknown | Unknown | Unknown | `unreachable` |
| ssh6 | `sinfo`, `squeue`, and `sacct` absent | DCGM, DCGM exporter, Prometheus, Grafana, and node exporter not detected | `lastcomm` and `accton` absent; `psacct` and `acct` inactive | `no_history_available` |
| ssh7-ssh10 | Unknown | Unknown | Unknown | `unreachable` |

Historical GPU users cannot be identified in the current environment. No reliable 7-day or 30-day allocation source was found on ssh6, so the generated user-summary CSV files contain headers only. Existing files named `eval_matrix_gpu*.csv` under the owner's DDF log directory are evaluation logs, not authoritative GPU accounting evidence.

`nvidia-smi` has no long-term history database. Current process visibility does not establish historical allocation. Login history cannot prove GPU use. Generic process accounting cannot map a historical process to a specific GPU without additional instrumentation.

## Training candidates

| Priority | Host | GPU | Free VRAM | Current status | Risk |
| ---: | --- | --- | ---: | --- | --- |
| 1 | ssh6 | 0-3, RTX 2080 Ti | 10822 MiB each | Two snapshots satisfy `available_now` | Status may change after the snapshot; run the live check before launch |

No conclusion is available for the other nine hosts. GPU model and memory alone are insufficient to select a DFD-HR batch size. Confirm batch size with the approved smoke test.

## Long-term monitoring proposal

The audit directory contains a one-shot append-only collector and a deployment proposal. After approval, the collector can run every five minutes, retain at least 90 days, and record GPU telemetry plus username, PID, executable name, and elapsed time. It does not record command arguments.

Personal-account sampling misses short-lived jobs and may not see every process. Group-level attribution should use administrator-managed Slurm accounting. DCGM exporter plus Prometheus provides stronger telemetry history, while reliable per-user GPU hours require scheduler allocation records or an approved process-attribution design.

No crontab, systemd timer, monitoring service, Slurm setting, or remote package was installed or changed.

## Audit artifacts

The full timestamped artifacts remain outside the repository at `/scratch/fengting/server-resource-audit/`:

- `snapshots/20260720_003845/`: sanitized SSH metadata and two raw system snapshots.
- `logs/audit_20260720_003845.log`: tmux execution log.
- `reports/server_inventory_20260720_003845.csv` and `.md`.
- `reports/gpu_current_status_20260720_003845.csv`.
- `reports/gpu_history_capability_20260720_003845.md`.
- `reports/gpu_user_summary_7d_20260720_003845.csv` and `gpu_user_summary_30d_20260720_003845.csv`.
- `reports/training_scheduler_20260720_003845.md`.
- `scripts/audit_servers.sh`, `build_reports.py`, `check_gpu_now.sh`, and `watch_gpu_candidates.sh`.
- `proposals/gpu_monitoring_plan.md`, `gpu_snapshot_collector.py`, and `gpu_snapshot_collector.sh`.

## Approval boundary

Deployment of the collector, changes to crontab or systemd, installation or activation of DCGM and Prometheus, and Slurm accounting changes require explicit researcher and administrator approval.

## Next controlled step

Use the current free-GPU table to select one unoccupied ssh6 GPU for a DFD-HR smoke test, then run `/scratch/fengting/server-resource-audit/scripts/check_gpu_now.sh` immediately before launch.
