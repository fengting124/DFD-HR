# Laboratory Server Resource Audit: vipsl1 to vipsl10

Audit run: `20260720_005246`

Controller: `vipsl6`

Target correction: `vipsl1` through `vipsl10`

Task scope: controlled read-only analysis. No process, GPU state, SSH setting, server configuration, dependency, dataset, training code, or experiment configuration was changed.

## Latest DFD-HR readiness update

Update time: `2026-07-20 03:22 +0800`

The shared DFD-HR code is under `/home/fengting/Experiments/DDF/DFD-HR`, on branch `agent/server-resource-audit-20260720`, commit `4520fdf`. The `/home/fengting` tree is an NFS mount shared across the reachable vipsl nodes. Each node's `/scratch` is local storage, so dataset names matching across nodes do not imply a shared filesystem.

`ready` in the scheduling context means all of the following are true for DFD-HR:

- the expected local dataset root exists under `/scratch/datasets/deepfake`
- sampled JSON paths resolve on that node
- `/scratch/fengting/miniconda3/envs/dfd-hr` exists and imports the required runtime stack
- DFD-HR unit tests pass with that environment
- the local output area has enough free scratch space for the intended run

The DFD-HR environment, dataset JSON registry, official weight, and project symlinks were staged and verified on `vipsl3`, `vipsl5`, `vipsl7`, `vipsl9`, and `vipsl10` using resumable `rsync --partial --append-verify` with a 50 MiB/s bandwidth limit. Dataset directories were not copied because the required local replicas were already present.

| Host | GPUs | Current GPU state | DFD-HR runtime | `/scratch` free | Scheduling conclusion |
| --- | --- | --- | --- | ---: | --- |
| vipsl3 | 2 x RTX 2080 Ti | idle | ready | 551 GiB | usable for 11 GiB-card smoke tests and moderate runs |
| vipsl5 | 4 x RTX 2080 Ti | idle | ready | 648 GiB | best 2080 Ti choice for multi-GPU jobs |
| vipsl6 | 4 x RTX 2080 Ti | idle | ready | 2022 GiB | best local 2080 Ti choice when free |
| vipsl7 | 2 x RTX 2080 Ti | idle | ready | 1006 GiB | good secondary 2080 Ti target |
| vipsl9 | 2 x RTX 3090 | occupied | ready | 2777 GiB | prepared, but do not schedule while current jobs run |
| vipsl10 | 2 x RTX 3090 | idle | ready | 81 GiB | usable for smoke tests, debugging, inference, or tightly bounded small runs only |

vipsl10 has the strongest per-card memory among idle nodes, but `/scratch` is 98 percent used. It should not be used for long training runs that write large checkpoints, logs, caches, frame dumps, or repeated experiment outputs unless the run writes to another node or a quota is explicitly enforced. Keep new vipsl10 output below roughly 30 GiB and maintain at least 30 to 50 GiB free scratch as a failure margin.

### vipsl10 scratch storage detail

vipsl10 `/scratch` is local ext4 storage on `/dev/sda`, not the shared NFS home mount. Current filesystem state:

| Metric | Value |
| --- | ---: |
| Total size | 3.6 TiB |
| Used | 3.4 TiB |
| Available to user jobs | 81 GiB |
| Use percentage | 98 percent |
| Raw free blocks | 267 GiB |
| Reserved blocks not available to normal users | 186 GiB |
| Inode use | 22M / 233M, 10 percent |

The inode count is healthy. The constraint is block capacity. Because ext4 reserves blocks for privileged use, normal jobs see only 81 GiB available even though raw free blocks are larger.

The bounded top-level accounting result is:

| Path | Size bytes | Approx size |
| --- | ---: | ---: |
| `/scratch/shengrong` | 2,427,615,830,016 | 2.21 TiB |
| `/scratch/hengji` | 634,093,793,280 | 590 GiB |
| `/scratch/datasets` | 236,320,665,600 | 220 GiB |
| `/scratch/linjing` | 210,462,986,240 | 196 GiB |
| `/scratch/jinyong` | 21,534,650,368 | 20 GiB |
| `/scratch/fengting` | 10,738,233,344 | 10 GiB |
| `/scratch/qifei` | 2,643,435,520 | 2.5 GiB |
| `/scratch/yuanqiao` | not fully measured | timeout and permission limits |

This shows the main known vipsl10 scratch pressure is `/scratch/shengrong`, not the deepfake datasets and not `/scratch/fengting`. `/scratch/yuanqiao` could not be fully measured within the 20-second bounded pass because some subdirectories denied read access and the command timed out. No other users' file contents were read.

The local deepfake dataset replica under `/scratch/datasets/deepfake` accounts for 236,320,665,600 bytes in total:

| Dataset path | Size bytes | Approx size |
| --- | ---: | ---: |
| `WildDeepfake` | 74,350,268,416 | 69 GiB |
| `HydraFake` | 61,449,695,232 | 57 GiB |
| `LAV-DF` | 25,887,793,152 | 24 GiB |
| `FaceForensics++` | 24,968,605,696 | 23 GiB |
| `Celeb-DF-v2` | 13,614,120,960 | 13 GiB |
| `DFDCP` | 10,422,145,024 | 9.7 GiB |
| `DFDC` | 8,429,289,472 | 7.9 GiB |
| `VDDL` | 8,432,566,272 | 7.9 GiB |
| `DF40` | 7,027,781,632 | 6.5 GiB |
| `simswap` | 1,738,391,552 | 1.6 GiB |

`/scratch/fengting` on vipsl10 is small relative to the disk pressure. Its measured top-level entries are mostly the staged DFD-HR runtime assets: `/scratch/fengting/miniconda3` is 6.1 GB, `/scratch/fengting/crucial_results` is 2.3 GB, `/scratch/fengting/DFD-HR` is 1.6 GB, and `/scratch/fengting/dfd_hr_repro_20260718` is 88 MB.

The original audit sections below are retained as historical snapshots from `20260720_005246`. Prefer the latest readiness table above for DFD-HR scheduling decisions.

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
