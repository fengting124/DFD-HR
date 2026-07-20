#!/usr/bin/env python3

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import yaml


RUN_ID_PATTERN = re.compile(r'^dfdhr_[a-z0-9+-]+_[a-z0-9-]+_[0-9]{8}_[0-9]{3}$')
STATUSES = {
    'planned',
    'preflight',
    'smoke_passed',
    'running',
    'completed',
    'failed',
    'aborted',
    'archived',
}
REQUIRED_ENVIRONMENT = (
    'DFDHR_REPO_ROOT',
    'DFDHR_PYTHON',
    'DFDHR_DATA_ROOT',
    'DFDHR_RUNTIME_ROOT',
    'DFDHR_ARCHIVE_ROOT',
    'DFDHR_CACHE_ROOT',
)
RUN_FILES = (
    'manifest.yaml',
    'config.resolved.yaml',
    'command.sh',
    'environment.txt',
    'git.txt',
    'metrics.jsonl',
    'training.log',
    'summary.md',
    'checksums.sha256',
)
REGISTRY_FIELDS = (
    'experiment_id',
    'date',
    'status',
    'objective',
    'branch',
    'commit',
    'config_path',
    'node_role',
    'gpu',
    'dataset',
    'effective_batch',
    'best_val_auc',
    'test_video_auc',
    'run_dir',
    'archive_dir',
    'summary_path',
)


def atomic_write_text(path, text, mode=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8', newline='') as file:
            file.write(text)
        if mode is not None:
            os.chmod(temporary_path, mode)
        os.replace(temporary_path, path)
    except Exception:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
        raise


def validate_run_id(run_id):
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(
            'RUN_ID must match dfdhr_<train-set>_<core-variable>_<YYYYMMDD>_<sequence>.'
        )
    return run_id


def validate_public_text(value, field_name):
    forbidden = re.compile(
        r'(/home/|/scratch/|(?:[0-9]{1,3}\.){3}[0-9]{1,3}|ssh|token|password|private[._ -]?key)',
        re.IGNORECASE,
    )
    if forbidden.search(value):
        raise ValueError(f'{field_name} contains private infrastructure text.')
    return value


def is_within(path, root):
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def require_environment(environment=None, executable=None):
    environment = os.environ if environment is None else environment
    missing = [name for name in REQUIRED_ENVIRONMENT if not environment.get(name)]
    if missing:
        raise ValueError(f'Missing required environment variables: {missing}')
    paths = {name: Path(environment[name]).expanduser().resolve() for name in REQUIRED_ENVIRONMENT}
    for name in ('DFDHR_REPO_ROOT', 'DFDHR_DATA_ROOT'):
        if not paths[name].is_dir():
            raise ValueError(f'{name} must be an existing directory.')
    if not paths['DFDHR_PYTHON'].is_file() or not os.access(paths['DFDHR_PYTHON'], os.X_OK):
        raise ValueError('DFDHR_PYTHON must be an executable file.')
    expected_executable = Path(executable or sys.executable).resolve()
    if paths['DFDHR_PYTHON'] != expected_executable:
        raise ValueError('Run lifecycle tools with DFDHR_PYTHON, not another interpreter.')

    repo_root = Path(subprocess.check_output(
        ['git', 'rev-parse', '--show-toplevel'],
        cwd=paths['DFDHR_REPO_ROOT'],
        text=True,
    ).strip()).resolve()
    if repo_root != paths['DFDHR_REPO_ROOT']:
        raise ValueError('DFDHR_REPO_ROOT does not match the active Git repository.')

    for name in ('DFDHR_RUNTIME_ROOT', 'DFDHR_ARCHIVE_ROOT', 'DFDHR_CACHE_ROOT'):
        path = paths[name]
        if is_within(path, paths['DFDHR_REPO_ROOT']) or is_within(path, paths['DFDHR_DATA_ROOT']):
            raise ValueError(f'{name} must be outside the repository and data root.')
        path.mkdir(parents=True, exist_ok=True)
    return paths


def git_output(repo_root, *args):
    return subprocess.check_output(['git', *args], cwd=repo_root, text=True).strip()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_config(source_path, base_config_path, destination_path, run_dir):
    with open(source_path, encoding='utf-8') as file:
        config = yaml.safe_load(file) or {}
    with open(base_config_path, encoding='utf-8') as file:
        base_config = yaml.safe_load(file) or {}
    config.update(base_config)
    run_dir = Path(run_dir).resolve()
    output_dir = run_dir / 'training'
    config.update({
        'run_id': run_dir.name,
        'log_dir': str(output_dir),
        'logdir': str(output_dir),
        'metrics_jsonl': str(run_dir / 'metrics.jsonl'),
        'save_feat': False,
        'save_ckpt': True,
    })
    payload = yaml.safe_dump(config, sort_keys=False, allow_unicode=False)
    atomic_write_text(destination_path, payload)
    return config, sha256_file(destination_path)


def assert_resolved(value, location='manifest'):
    if isinstance(value, dict):
        for key, item in value.items():
            assert_resolved(item, f'{location}.{key}')
    elif isinstance(value, list):
        for index, item in enumerate(value):
            assert_resolved(item, f'{location}[{index}]')
    elif isinstance(value, str) and 'REPLACE_ME' in value:
        raise ValueError(f'Unresolved manifest placeholder: {location}')


def capture_metadata(run_dir, paths):
    repo_root = paths['DFDHR_REPO_ROOT']
    environment_lines = [
        f'python={sys.executable}',
        f'python_version={sys.version.split()[0]}',
    ]
    probe = subprocess.check_output(
        [sys.executable, '-c', (
            'import cv2, torch, transformers; '
            'print(f"torch={torch.__version__}"); '
            'print(f"cuda={torch.version.cuda}"); '
            'print(f"cuda_available={torch.cuda.is_available()}"); '
            'print(f"gpu_count={torch.cuda.device_count()}"); '
            'print(f"transformers={transformers.__version__}"); '
            'print(f"opencv={cv2.__version__}")'
        )],
        text=True,
    ).strip()
    environment_lines.extend(probe.splitlines())
    for name in REQUIRED_ENVIRONMENT:
        environment_lines.append(f'{name}={paths[name]}')
    atomic_write_text(Path(run_dir) / 'environment.txt', '\n'.join(environment_lines) + '\n')

    status = git_output(repo_root, 'status', '--short')
    git_lines = [
        f'branch={git_output(repo_root, "branch", "--show-current")}',
        f'commit={git_output(repo_root, "rev-parse", "HEAD")}',
        f'dirty={str(bool(status)).lower()}',
    ]
    if status:
        git_lines.append('status:')
        git_lines.extend(status.splitlines())
    atomic_write_text(Path(run_dir) / 'git.txt', '\n'.join(git_lines) + '\n')
    return not bool(status)


def checksum_entries(run_dir):
    run_dir = Path(run_dir).resolve()
    entries = []
    for path in sorted(run_dir.rglob('*')):
        if not path.is_file() or path.name == 'checksums.sha256':
            continue
        entries.append((path.relative_to(run_dir).as_posix(), sha256_file(path)))
    return entries


def write_checksums(run_dir):
    entries = checksum_entries(run_dir)
    payload = ''.join(f'{digest}  {relative_path}\n' for relative_path, digest in entries)
    atomic_write_text(Path(run_dir) / 'checksums.sha256', payload)
    return entries


def verify_checksums(run_dir):
    checksum_path = Path(run_dir) / 'checksums.sha256'
    expected = {}
    with checksum_path.open(encoding='utf-8') as file:
        for line in file:
            digest, relative_path = line.rstrip('\n').split('  ', 1)
            expected[relative_path] = digest
    actual = dict(checksum_entries(run_dir))
    return expected == actual


def initialize_run(args):
    validate_run_id(args.run_id)
    for name in (
        'epochs',
        'gpu_count',
        'per_gpu_batch',
        'gradient_accumulation_steps',
        'maximum_local_size_gib',
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f'{name} must be positive.')
    if args.workers < 0:
        raise ValueError('workers cannot be negative.')
    paths = require_environment()
    repo_root = paths['DFDHR_REPO_ROOT']
    if git_output(repo_root, 'status', '--short'):
        raise ValueError('Git worktree must be clean before run initialization.')

    registry_path = repo_root / 'registry/experiments.csv'
    with registry_path.open(encoding='utf-8', newline='') as file:
        if any(row['experiment_id'] == args.run_id for row in csv.DictReader(file)):
            raise ValueError('RUN_ID already exists in the public registry.')

    run_dir = paths['DFDHR_RUNTIME_ROOT'] / 'runs' / args.run_id
    archive_dir = paths['DFDHR_ARCHIVE_ROOT'] / 'runs' / args.run_id
    if run_dir.exists() or archive_dir.exists():
        raise ValueError('RUN_ID has already been used in runtime or archive storage.')
    run_dir.mkdir(parents=True)
    (run_dir / 'checkpoints').mkdir()
    (run_dir / 'notebooks_executed').mkdir()

    config, config_sha256 = freeze_config(
        args.config,
        args.base_config,
        run_dir / 'config.resolved.yaml',
        run_dir,
    )
    now = datetime.now(timezone.utc).isoformat()
    with (repo_root / 'templates/experiment_manifest.yaml').open(encoding='utf-8') as file:
        manifest = yaml.safe_load(file)
    manifest.update({
        'experiment_id': args.run_id,
        'status': 'planned',
        'created_at': now,
        'updated_at': now,
        'objective': args.objective,
        'hypothesis': args.hypothesis,
        'single_changed_variable': args.single_changed_variable,
        'success_criteria': args.success_criterion,
    })
    manifest['protocol'].update({
        'train_dataset': config['train_dataset'],
        'validation_dataset': config['validation_dataset'],
        'test_datasets': config['test_dataset'],
        'compression': config['compression'],
        'train_frames_per_video': config['frame_num']['train'],
        'validation_frames_per_video': config['frame_num']['val'],
        'test_frames_per_video': config['frame_num']['test'],
    })
    manifest['model'].update({
        'name': config['model_name'],
        'backbone': config['backbone_name'],
        'resolution': config['resolution'],
        'capacity': config['backbone_config']['capacity'],
        'routed_layers': list(range(
            config['backbone_config']['remain_layer'],
            config['backbone_config']['layer'],
        )),
    })
    manifest['model']['moe'].update({
        'num_experts': config['backbone_config']['moe']['num_experts'],
        'top_k': config['backbone_config']['moe']['top_k'],
        'noise': config['backbone_config']['moe']['noise'],
    })
    manifest['training'].update({
        'epochs': args.epochs,
        'precision': args.precision,
        'gpu_count': args.gpu_count,
        'per_gpu_batch': args.per_gpu_batch,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'effective_batch_size': (
            args.gpu_count * args.per_gpu_batch * args.gradient_accumulation_steps
        ),
        'workers_per_process': args.workers,
        'optimizer': config['optimizer']['type'],
        'learning_rate': config['optimizer'][config['optimizer']['type']]['lr'],
        'seed': config['manualSeed'],
    })
    manifest['version'].update({
        'git_branch': git_output(repo_root, 'branch', '--show-current'),
        'git_commit': git_output(repo_root, 'rev-parse', 'HEAD'),
        'git_dirty': False,
        'config_sha256': config_sha256,
        'dataset_json_sha256': sha256_file(args.dataset_json),
        'initial_weight_sha256': (
            sha256_file(args.initial_weight) if args.initial_weight else None
        ),
    })
    import torch

    manifest['runtime'].update({
        'node_role': args.node_role,
        'python_executable': str(paths['DFDHR_PYTHON']),
        'torch_version': str(torch.__version__),
        'cuda_version': torch.version.cuda,
        'gpu_model': (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        'gpu_indices': args.gpu_indices,
    })
    manifest['storage'].update({
        'runtime_root': str(paths['DFDHR_RUNTIME_ROOT']),
        'archive_root': str(paths['DFDHR_ARCHIVE_ROOT']),
        'maximum_local_size_gib': args.maximum_local_size_gib,
    })
    manifest['validation'].update({
        'preflight_passed': args.preflight_passed,
        'single_gpu_smoke_passed': args.single_gpu_smoke_passed,
        'checkpoint_roundtrip_passed': args.checkpoint_roundtrip_passed,
        'ddp_smoke_passed': args.ddp_smoke_passed,
    })
    manifest['initialization'].update({
        'dfd_hr_checkpoint': str(Path(args.initial_weight).resolve()) if args.initial_weight else None,
        'independent_reproduction': not bool(args.initial_weight),
    })
    assert_resolved(manifest)
    atomic_write_text(
        run_dir / 'manifest.yaml',
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=False),
    )
    command = '#!/usr/bin/env bash\nset -euo pipefail\n\n' + args.command.rstrip() + '\n'
    atomic_write_text(run_dir / 'command.sh', command, mode=0o700)
    shutil.copyfile(repo_root / 'templates/experiment_summary.md', run_dir / 'summary.md')
    for name in ('metrics.jsonl', 'training.log'):
        atomic_write_text(run_dir / name, '')
    capture_metadata(run_dir, paths)
    write_checksums(run_dir)
    print('run_role=DFDHR_RUNTIME_ROOT/runs/<RUN_ID>')


def freeze_run_config(args):
    validate_run_id(args.run_id)
    paths = require_environment()
    run_dir = paths['DFDHR_RUNTIME_ROOT'] / 'runs' / args.run_id
    manifest_path = run_dir / 'manifest.yaml'
    with manifest_path.open(encoding='utf-8') as file:
        manifest = yaml.safe_load(file)
    _, config_sha256 = freeze_config(
        args.config,
        args.base_config,
        run_dir / 'config.resolved.yaml',
        run_dir,
    )
    manifest['version']['config_sha256'] = config_sha256
    manifest['updated_at'] = datetime.now(timezone.utc).isoformat()
    assert_resolved(manifest)
    atomic_write_text(
        manifest_path,
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=False),
    )
    write_checksums(run_dir)


def verify_run(args):
    validate_run_id(args.run_id)
    paths = require_environment()
    run_dir = paths['DFDHR_RUNTIME_ROOT'] / 'runs' / args.run_id
    missing = [name for name in RUN_FILES if not (run_dir / name).is_file()]
    if missing:
        raise ValueError(f'Run directory is missing required files: {missing}')
    if any(path.is_symlink() for path in run_dir.rglob('*')):
        raise ValueError('Run directories cannot contain symbolic links.')
    if not os.access(run_dir / 'command.sh', os.X_OK):
        raise ValueError('command.sh must be executable.')

    with (run_dir / 'manifest.yaml').open(encoding='utf-8') as file:
        manifest = yaml.safe_load(file)
    assert_resolved(manifest)
    if manifest['experiment_id'] != args.run_id:
        raise ValueError('Manifest RUN_ID does not match its directory.')
    with (run_dir / 'config.resolved.yaml').open(encoding='utf-8') as file:
        config = yaml.safe_load(file)
    if config.get('run_id') != args.run_id:
        raise ValueError('Resolved config RUN_ID does not match its directory.')
    for key in ('log_dir', 'logdir', 'metrics_jsonl'):
        if not is_within(config[key], run_dir):
            raise ValueError(f'Resolved config {key} must stay inside the run directory.')
    if config.get('save_feat', True) or not config.get('save_ckpt', False):
        raise ValueError('Resolved config must disable features and retain checkpoints.')
    if not verify_checksums(run_dir):
        raise ValueError('Run checksums are stale or invalid.')
    total_bytes = sum(path.stat().st_size for path in run_dir.rglob('*') if path.is_file())
    budget_bytes = float(manifest['storage']['maximum_local_size_gib']) * 1024**3
    if total_bytes > budget_bytes:
        raise ValueError('Run directory exceeds its manifest storage budget.')
    print('run_valid=true paths_protected=true checksums_valid=true budget_valid=true')


def register_run(args):
    validate_run_id(args.run_id)
    paths = require_environment()
    repo_root = paths['DFDHR_REPO_ROOT']
    run_dir = paths['DFDHR_RUNTIME_ROOT'] / 'runs' / args.run_id
    with (run_dir / 'manifest.yaml').open(encoding='utf-8') as file:
        manifest = yaml.safe_load(file)
    status = args.status
    if status not in STATUSES:
        raise ValueError(f'Unsupported experiment status: {status}')

    objective = validate_public_text(args.objective or str(manifest['objective']).strip(), 'objective')
    node_role = validate_public_text(str(manifest['runtime']['node_role']), 'node_role')
    gpu = validate_public_text(args.gpu, 'gpu')
    row = {
        'experiment_id': args.run_id,
        'date': str(manifest['created_at'])[:10],
        'status': status,
        'objective': objective,
        'branch': manifest['version']['git_branch'],
        'commit': manifest['version']['git_commit'],
        'config_path': '${DFDHR_RUNTIME_ROOT}/runs/' + args.run_id + '/config.resolved.yaml',
        'node_role': node_role,
        'gpu': gpu,
        'dataset': ','.join(manifest['protocol']['train_dataset']),
        'effective_batch': manifest['training']['effective_batch_size'],
        'best_val_auc': args.best_val_auc or '',
        'test_video_auc': args.test_video_auc or '',
        'run_dir': '${DFDHR_RUNTIME_ROOT}/runs/' + args.run_id,
        'archive_dir': (
            '${DFDHR_ARCHIVE_ROOT}/runs/' + args.run_id
            if status == 'archived' else ''
        ),
        'summary_path': '${DFDHR_RUNTIME_ROOT}/runs/' + args.run_id + '/summary.md',
    }
    registry_path = repo_root / 'registry/experiments.csv'
    with registry_path.open(encoding='utf-8', newline='') as file:
        rows = list(csv.DictReader(file))
    if any(existing['experiment_id'] == args.run_id for existing in rows):
        raise ValueError('Registry already contains this RUN_ID; edit through a reviewed change.')
    rows.append({field: row[field] for field in REGISTRY_FIELDS})
    buffer = tempfile.SpooledTemporaryFile(mode='w+', newline='', encoding='utf-8')
    writer = csv.DictWriter(buffer, fieldnames=REGISTRY_FIELDS, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    buffer.seek(0)
    atomic_write_text(registry_path, buffer.read())
    buffer.close()
    manifest['status'] = status
    manifest['updated_at'] = datetime.now(timezone.utc).isoformat()
    atomic_write_text(
        run_dir / 'manifest.yaml',
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=False),
    )
    write_checksums(run_dir)


def archive_run(args):
    validate_run_id(args.run_id)
    paths = require_environment()
    source = paths['DFDHR_RUNTIME_ROOT'] / 'runs' / args.run_id
    destination = paths['DFDHR_ARCHIVE_ROOT'] / 'runs' / args.run_id
    if not verify_checksums(source):
        raise ValueError('Source checksums are stale or invalid.')
    if destination.exists():
        raise ValueError('Archive destination already exists.')
    if not args.execute:
        print('archive_plan=copy runtime role to archive role; source will be retained')
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)
    if dict(checksum_entries(source)) != dict(checksum_entries(destination)):
        raise RuntimeError('Archive checksum comparison failed; source was retained.')
    print('archive_verified=true source_retained=true')


def build_parser():
    parser = argparse.ArgumentParser(description='Manage DFD-HR experiment lifecycle metadata.')
    subparsers = parser.add_subparsers(dest='command_name', required=True)

    init_parser = subparsers.add_parser('init')
    init_parser.add_argument('--run-id', required=True)
    init_parser.add_argument('--config', required=True)
    init_parser.add_argument('--base-config', required=True)
    init_parser.add_argument('--dataset-json', required=True)
    init_parser.add_argument('--initial-weight')
    init_parser.add_argument('--objective', required=True)
    init_parser.add_argument('--hypothesis', required=True)
    init_parser.add_argument('--success-criterion', action='append', required=True)
    init_parser.add_argument('--single-changed-variable', required=True)
    init_parser.add_argument('--node-role', required=True)
    init_parser.add_argument('--command', required=True)
    init_parser.add_argument('--epochs', type=int, required=True)
    init_parser.add_argument('--precision', choices=('fp32', 'amp'), required=True)
    init_parser.add_argument('--gpu-count', type=int, required=True)
    init_parser.add_argument('--gpu-indices', required=True)
    init_parser.add_argument('--per-gpu-batch', type=int, required=True)
    init_parser.add_argument('--gradient-accumulation-steps', type=int, required=True)
    init_parser.add_argument('--workers', type=int, required=True)
    init_parser.add_argument('--maximum-local-size-gib', type=float, required=True)
    init_parser.add_argument('--preflight-passed', action='store_true')
    init_parser.add_argument('--single-gpu-smoke-passed', action='store_true')
    init_parser.add_argument('--checkpoint-roundtrip-passed', action='store_true')
    init_parser.add_argument('--ddp-smoke-passed', action='store_true')
    init_parser.set_defaults(handler=initialize_run)

    capture_parser = subparsers.add_parser('capture')
    capture_parser.add_argument('--run-id', required=True)
    capture_parser.set_defaults(handler=lambda args: capture_metadata(
        require_environment()['DFDHR_RUNTIME_ROOT'] / 'runs' / validate_run_id(args.run_id),
        require_environment(),
    ))

    freeze_parser = subparsers.add_parser('freeze')
    freeze_parser.add_argument('--run-id', required=True)
    freeze_parser.add_argument('--config', required=True)
    freeze_parser.add_argument('--base-config', required=True)
    freeze_parser.set_defaults(handler=freeze_run_config)

    checksum_parser = subparsers.add_parser('checksums')
    checksum_parser.add_argument('--run-id', required=True)
    checksum_parser.set_defaults(handler=lambda args: write_checksums(
        require_environment()['DFDHR_RUNTIME_ROOT'] / 'runs' / validate_run_id(args.run_id)
    ))

    verify_parser = subparsers.add_parser('verify')
    verify_parser.add_argument('--run-id', required=True)
    verify_parser.set_defaults(handler=verify_run)

    register_parser = subparsers.add_parser('register')
    register_parser.add_argument('--run-id', required=True)
    register_parser.add_argument('--status', required=True)
    register_parser.add_argument('--objective')
    register_parser.add_argument('--gpu', required=True)
    register_parser.add_argument('--best-val-auc')
    register_parser.add_argument('--test-video-auc')
    register_parser.set_defaults(handler=register_run)

    archive_parser = subparsers.add_parser('archive')
    archive_parser.add_argument('--run-id', required=True)
    archive_parser.add_argument('--execute', action='store_true')
    archive_parser.set_defaults(handler=archive_run)
    return parser


def main():
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == '__main__':
    main()
