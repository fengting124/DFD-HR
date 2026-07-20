#!/usr/bin/env python3

import argparse
from pathlib import Path

import yaml


def is_within(path, root):
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def build_formal_config(
    detector_path,
    base_config_path,
    dataset_json_folder,
    clip_model_path,
    workers=4,
    reproducibility_mode='deterministic',
    gpu_count=2,
    train_batch_size=1,
    gradient_accumulation_steps=8,
    validation_batch_size=1,
):
    with open(base_config_path, encoding='utf-8') as file:
        config = yaml.safe_load(file) or {}
    with open(detector_path, encoding='utf-8') as file:
        config.update(yaml.safe_load(file) or {})

    clip_model_path = Path(clip_model_path).resolve()
    if not (clip_model_path / 'model.safetensors').is_file():
        raise ValueError('clip_model_path must contain model.safetensors.')
    if workers < 0:
        raise ValueError('workers cannot be negative.')
    for name, value in (
        ('gpu_count', gpu_count),
        ('train_batch_size', train_batch_size),
        ('gradient_accumulation_steps', gradient_accumulation_steps),
        ('validation_batch_size', validation_batch_size),
    ):
        if value <= 0:
            raise ValueError(f'{name} must be positive.')
    effective_batch_size = (
        gpu_count * train_batch_size * gradient_accumulation_steps
    )
    if effective_batch_size != 16:
        raise ValueError(
            'Formal training requires effective batch size 16; '
            f'got {gpu_count} x {train_batch_size} x '
            f'{gradient_accumulation_steps} = {effective_batch_size}.'
        )
    if reproducibility_mode not in {'deterministic', 'seeded_best_effort'}:
        raise ValueError('Unsupported reproducibility mode.')
    deterministic = reproducibility_mode == 'deterministic'

    config.update({
        'protocol_mode': 'paper_spec',
        'paper_spec_basis': {
            'moe_routing': 'paper_equations_13_14',
            'epochs': 'official_repository_default',
            'amp_and_batch_adaptation': 'hardware_adaptation_effective_batch_16',
        },
        'dataset_json_folder': str(Path(dataset_json_folder).resolve()),
        'backbone_name': 'ViT-L/14_proj',
        'backbone_pretrained': True,
        'backbone_pretrained_path': str(clip_model_path),
        'backbone_local_files_only': True,
        'initialization_mode': 'pinned_clip_pretrained',
        'train_dataset': ['FaceForensics++'],
        'validation_dataset': ['FaceForensics++'],
        'test_dataset': [],
        'compression': 'c23',
        'frame_num': {'train': 8, 'val': 32, 'test': 32},
        'train_batchSize': train_batch_size,
        'test_batchSize': validation_batch_size,
        'workers': workers,
        'amp': True,
        'amp_initial_scale': 1024,
        'gradient_accumulation_steps': gradient_accumulation_steps,
        'ddp_timeout_minutes': 180,
        'nEpochs': 20,
        'start_epoch': 0,
        'save_ckpt': True,
        'save_feat': False,
        'save_avg': True,
        'manualSeed': 1024,
        'validation_checks_per_epoch': {
            'first_epoch': 1,
            'later_epochs': 2,
        },
        'run_final_test_after_training': False,
        'reproducibility_mode': reproducibility_mode,
        'cudnn_benchmark': False,
        'cudnn_deterministic': True,
        'deterministic_algorithms': deterministic,
        'cublas_workspace_config': ':4096:8',
        'dry_run': False,
    })
    config['optimizer']['type'] = 'adam'
    config['optimizer']['adam']['lr'] = 0.0001
    moe_config = config.setdefault('backbone_config', {}).setdefault('moe', {})
    moe_config.update({
        'num_experts': 4,
        'top_k': 4,
        'noise': True,
        'load_balancing_weight': 0.0,
    })
    return config


def parse_args():
    parser = argparse.ArgumentParser(description='Build a frozen DFD-HR formal training config.')
    parser.add_argument('--detector-path', required=True)
    parser.add_argument('--base-config-path', required=True)
    parser.add_argument('--dataset-json-folder', required=True)
    parser.add_argument('--clip-model-path', required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--gpu-count', type=int, default=2)
    parser.add_argument('--train-batch-size', type=int, default=1)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=8)
    parser.add_argument('--validation-batch-size', type=int, default=1)
    parser.add_argument(
        '--reproducibility-mode',
        choices=('deterministic', 'seeded_best_effort'),
        default='deterministic',
    )
    parser.add_argument('--output', required=True)
    parser.add_argument('--repo-root', required=True)
    parser.add_argument('--data-root', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    output = Path(args.output).expanduser().resolve()
    for protected_root in (args.repo_root, args.data_root):
        if is_within(output, protected_root):
            raise ValueError('Formal config output must be outside the repository and data root.')
    config = build_formal_config(
        args.detector_path,
        args.base_config_path,
        args.dataset_json_folder,
        args.clip_model_path,
        workers=args.workers,
        reproducibility_mode=args.reproducibility_mode,
        gpu_count=args.gpu_count,
        train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        validation_batch_size=args.validation_batch_size,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=False),
        encoding='utf-8',
    )
    print('formal_config_role=DFDHR_RUNTIME_ROOT/preflight/<config>.yaml')


if __name__ == '__main__':
    main()
