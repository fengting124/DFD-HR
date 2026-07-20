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
    if reproducibility_mode not in {'deterministic', 'seeded_best_effort'}:
        raise ValueError('Unsupported reproducibility mode.')
    deterministic = reproducibility_mode == 'deterministic'

    config.update({
        'protocol_mode': 'paper_aligned',
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
        'train_batchSize': 1,
        'test_batchSize': 1,
        'workers': workers,
        'amp': True,
        'amp_initial_scale': 1024,
        'gradient_accumulation_steps': 8,
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
    return config


def parse_args():
    parser = argparse.ArgumentParser(description='Build a frozen DFD-HR formal training config.')
    parser.add_argument('--detector-path', required=True)
    parser.add_argument('--base-config-path', required=True)
    parser.add_argument('--dataset-json-folder', required=True)
    parser.add_argument('--clip-model-path', required=True)
    parser.add_argument('--workers', type=int, default=4)
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
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=False),
        encoding='utf-8',
    )
    print('formal_config_role=DFDHR_RUNTIME_ROOT/preflight/<config>.yaml')


if __name__ == '__main__':
    main()
