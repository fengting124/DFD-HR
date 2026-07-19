# Laboratory Server Resource Audit: vipsl1 to vipsl10

Audit run: `20260720_005246`

Controller: `vipsl6`

Target correction: `vipsl1` through `vipsl10`

Task scope: controlled read-only analysis. No process, GPU state, SSH setting, server configuration, dependency, dataset, training code, or experiment configuration was changed.

## Method and safety boundary

Each target received one non-interactive SSH connection attempt with `BatchMode=yes`, an 8-second connection timeout, and host-key verification enabled. Reachable nodes were sampled twice. Per-node snapshot intervals were 69 to 74 seconds. Collection was restricted to public system state and GPU process username, PID, executable name, elapsed time, start time, GPU UUID, and GPU memory. No command arguments, environment variables, shell history, project files, datasets, private keys, tokens, or other users' directories were read.

## Connectivity

| Host | Reachable | Result |
| --- | --- | --- |
| vipsl1 | No | TCP connection to port 22 timed out; no retry |
| vipsl2 | Yes | Connected and sampled twice |
| vipsl3 | Yes | Connected and sampled twice |
| vipsl4 | Yes | Connected and sampled twice |
| vipsl5 | Yes | Connected and sampled twice |
| vipsl6 | Yes | Connected and sampled twice |
| vipsl7 | Yes | Connected and sampled twice |
| vipsl8 | No | Public-key authentication failed; no retry |
| vipsl9 | Yes | Connected and sampled twice |
| vipsl10 | Yes | Connected and sampled twice |

The vipsl1 timeout does not prove that the server is powered off. vipsl8 may be operational, but the current account cannot authenticate.

## Server inventory

| Host | GPUs | Current state | Current users | CPU threads | RAM | Load 1m | Scratch free |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| vipsl1 | Unknown | Unreachable | Unknown | Unknown | Unknown | Unknown | Unknown |
| vipsl2 | 2 × RTX 3090, 24 GiB | 2 occupied | `hengji` | 24 | 62 GiB | 2.45 | 1.6 TiB |
| vipsl3 | 2 × RTX 2080 Ti, 11 GiB | 2 available | None | 8 | 62 GiB | 0.06 | 558 GiB |
| vipsl4 | 2 × RTX 2080 Ti, 11 GiB | 2 available | None | 12 | 62 GiB | 0.23 | 628 GiB |
| vipsl5 | 4 × RTX 2080 Ti, 11 GiB | 4 available | None | 32 | 62 GiB | 0.07 | 655 GiB |
| vipsl6 | 4 × RTX 2080 Ti, 11 GiB | 4 available | None | 32 | 62 GiB | 0.34 | 2.0 TiB |
| vipsl7 | 2 × RTX 2080 Ti, 11 GiB | 2 available | None | 8 | 62 GiB | 0.15 | 1013 GiB |
| vipsl8 | Unknown | Authentication failed | Unknown | Unknown | Unknown | Unknown | Unknown |
| vipsl9 | 2 × RTX 3090, 24 GiB | 2 occupied | `hengji` | 32 | 46 GiB | 3.15 | `/scratch` unavailable |
| vipsl10 | 2 × RTX 3090, 24 GiB | 2 available | None | 20 | 62 GiB | 0.00 | 89 GiB |

## GPU and CUDA configuration

| Host | GPU model | Driver | CUDA compatibility | GPU count | Available now |
| --- | --- | --- | --- | ---: | ---: |
| vipsl2 | RTX 3090 | 570.195.03 | 12.8 | 2 | 0 |
| vipsl3 | RTX 2080 Ti | 570.172.08 | 12.8 | 2 | 2 |
| vipsl4 | RTX 2080 Ti | 570.172.08 | 12.8 | 2 | 2 |
| vipsl5 | RTX 2080 Ti | 570.195.03 | 12.8 | 4 | 4 |
| vipsl6 | RTX 2080 Ti | 570.195.03 | 12.8 | 4 | 4 |
| vipsl7 | RTX 2080 Ti | 570.195.03 | 12.8 | 2 | 2 |
| vipsl9 | RTX 3090 | 570.211.01 | 12.8 | 2 | 0 |
| vipsl10 | RTX 3090 | 580.159.03 | 13.0 | 2 | 2 |

`nvcc` was unavailable and the default `python3` environment had no PyTorch on all eight reachable nodes. This describes the login environment only. It does not invalidate dedicated Conda environments.

## Current GPU processes

| Host | GPU | User | PID | Program | Elapsed | Start time | GPU memory MiB | Status |
| --- | ---: | --- | ---: | --- | --- | --- | ---: | --- |
| vipsl2 | 0 | `hengji` | 3325120 | `python3.11` | 04:19:10 | 2026-07-19 20:35 | 20464 | occupied |
| vipsl2 | 1 | `hengji` | 3325121 | `python3.11` | 04:19:10 | 2026-07-19 20:35 | 19908 | occupied |
| vipsl9 | 0 | `hengji` | 435862 | `python3.11` | 1-20:36:07 | 2026-07-18 04:18 | 20460 | occupied |
| vipsl9 | 1 | `hengji` | 435863 | `python3.11` | 1-20:36:07 | 2026-07-18 04:18 | 19896 | occupied |

The table contains no process command arguments. All usernames were resolved with `ps`. No compute process was found on vipsl3 through vipsl7 or vipsl10 in either snapshot.

## Training scheduling

| Priority | Host | GPU | Available cards | Free VRAM per card | Current status | Risk |
| ---: | --- | --- | ---: | ---: | --- | --- |
| 1 | vipsl10 | RTX 3090 | 2 | 24111 to 24124 MiB | `available_now` | Only 89 GiB `/scratch` free; confirm output and dataset storage requirements |
| 2 | vipsl6 | RTX 2080 Ti | 4 | 10822 MiB | `available_now` | Recheck immediately before launch |
| 3 | vipsl5 | RTX 2080 Ti | 4 | 10822 MiB | `available_now` | 655 GiB `/scratch` free |
| 4 | vipsl7 | RTX 2080 Ti | 2 | 10822 MiB | `available_now` | Recheck immediately before launch |
| 5 | vipsl4 | RTX 2080 Ti | 2 | 10821 to 10822 MiB | `available_now` | Recheck immediately before launch |
| 6 | vipsl3 | RTX 2080 Ti | 2 | 10822 MiB | `available_now` | Recheck immediately before launch |

vipsl2 and vipsl9 are currently unsuitable for a new exclusive training job. Their GPUs are actively used by `hengji`, with 98% to 100% GPU utilization in the second snapshot. Do not share or preempt those GPUs without coordination. vipsl1 and vipsl8 cannot be scheduled from the current account because their resource state is unknown.

The ranking favors vipsl10 for memory-intensive DFD-HR smoke tests, then the four-GPU vipsl6 and vipsl5 nodes. It does not estimate batch size. Confirm batch size and memory requirements with the approved smoke test. Check whether vipsl10's limited `/scratch` capacity is adequate before writing outputs.

## Historical visibility

| Hosts | Slurm | GPU monitoring | Process accounting | Conclusion |
| --- | --- | --- | --- | --- |
| vipsl2-vipsl7, vipsl9-vipsl10 | `sinfo`, `squeue`, and `sacct` absent | DCGM and exporter inactive; no Prometheus, Grafana, or node exporter process detected | `lastcomm` and `accton` absent; `psacct` and `acct` inactive | `no_history_available` |
| vipsl1 | Unknown | Unknown | Unknown | `unreachable` |
| vipsl8 | Unknown | Unknown | Unknown | `unreachable` under current credentials |

Historical GPU users cannot be identified from the current environment. The 7-day and 30-day summary CSV files therefore contain headers only. `nvidia-smi` has no long-term database. Current processes, login history, and generic process history cannot establish historical GPU allocation.

## Long-term monitoring proposal

The audit directory contains a one-shot append-only collector and deployment proposal. After approval, it can sample every five minutes and retain at least 90 days. It records GPU telemetry plus username, PID, executable name, and elapsed time, without command arguments.

Personal-account sampling can miss short jobs and cannot guarantee group-wide visibility. Reliable user-level GPU-hour accounting requires administrator-managed Slurm accounting or an approved equivalent. DCGM exporter plus Prometheus provides telemetry history but does not inherently provide reliable user attribution.

No scheduler, monitoring agent, crontab, systemd timer, or package was installed or modified.

## Audit artifacts

Full timestamped artifacts are stored at `/scratch/fengting/server-resource-audit/`:

- `snapshots/20260720_005246/`
- `logs/audit_20260720_005246.log`
- `reports/server_inventory_20260720_005246.csv` and `.md`
- `reports/gpu_current_status_20260720_005246.csv`
- `reports/gpu_history_capability_20260720_005246.md`
- `reports/gpu_user_summary_7d_20260720_005246.csv`
- `reports/gpu_user_summary_30d_20260720_005246.csv`
- `reports/training_scheduler_20260720_005246.md`
- `scripts/audit_servers.sh`, `build_reports.py`, `check_gpu_now.sh`, and `watch_gpu_candidates.sh`
- `proposals/gpu_monitoring_plan.md`, `gpu_snapshot_collector.py`, and `gpu_snapshot_collector.sh`

## Approval boundary

Collector deployment, crontab or systemd changes, DCGM or Prometheus installation, and Slurm accounting changes require explicit researcher and administrator approval.

## Next controlled step

Select one currently unoccupied GPU for the DFD-HR smoke test and run `/scratch/fengting/server-resource-audit/scripts/check_gpu_now.sh` immediately before launch.
