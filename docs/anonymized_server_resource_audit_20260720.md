# Anonymized Server Resource Audit

Date: 2026-07-20

Scope: public-safe summary for DFD-HR experiment scheduling. The complete laboratory server audit contains internal hostnames, usernames, directory paths, storage accounting, and live GPU ownership data, and must remain outside the public repository.

## Redaction policy

The public repository must not contain:

- internal hostnames or SSH aliases
- laboratory usernames or account-to-person mappings
- per-user directory names or storage usage
- internal absolute paths
- private dataset names, local dataset sizes, or replica topology
- live GPU ownership or process attribution

Use the following placeholders in public documentation:

| Sensitive class | Public placeholder |
| --- | --- |
| physical server | `node-a`, `node-b`, `node-c` |
| laboratory account | `user-01`, `user-02` |
| dataset location | `local-dataset-root` |
| environment location | `local-env-root` |
| output location | `local-output-root` |
| private model weight location | `local-weight-root` |

## Public scheduling summary

The audited cluster contains two useful GPU classes for DFD-HR work:

| Public node class | GPU memory class | Recommended use |
| --- | --- | --- |
| `node-a` class | 24 GiB per GPU | smoke tests, debugging, inference, and memory-heavy short runs |
| `node-b` class | 11 GiB per GPU | moderate training runs and multi-GPU experiments with tuned batch size |

One high-memory node was found to have very limited local scratch headroom. It should be treated as suitable only for bounded smoke tests or short jobs unless output is redirected to a larger approved storage target.

## Public readiness definition

For public scheduling notes, a node is `ready` only when:

- `local-dataset-root` exists and sampled dataset references resolve
- `local-env-root` imports the required Python and PyTorch runtime stack
- project tests pass with that runtime
- `local-weight-root` contains the required model weights
- `local-output-root` has enough free space for the planned run
- the node is idle at launch time

## Operational guidance

- Keep code, configuration, and documentation in the shared project repository.
- Keep large datasets, environments, checkpoints, caches, logs, and run outputs on approved local storage.
- Recheck GPU and local scratch availability immediately before launching a job.
- Do not publish full server inventories, user mappings, per-user storage accounting, or live process ownership in this public repository.

## Private audit handling

The complete internal audit should be stored only in a private repository or access-controlled internal storage. If a public branch ever receives the full report, deleting the file in a later commit is insufficient because the sensitive content remains in Git history. The branch must be removed or rewritten after a private backup is created, and any downstream clones or caches must be treated as potentially exposed.
