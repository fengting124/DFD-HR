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


def build_mini_config(detector_path, base_config_path, dataset_json_folder):
    with open(base_config_path, encoding='utf-8') as file:
        config = yaml.safe_load(file) or {}
    with open(detector_path, encoding='utf-8') as file:
        config.update(yaml.safe_load(file) or {})
    config.update({
        'backbone_pretrained': False,
        'initialization_mode': 'architecture_only_random',
        'dataset_json_folder': str(Path(dataset_json_folder).resolve()),
        'train_dataset': ['FaceForensics++'],
        'validation_dataset': ['FaceForensics++'],
        'test_dataset': ['FaceForensics++'],
        'frame_num': {'train': 1, 'val': 1, 'test': 1},
        'train_max_samples': 16,
        'validation_max_samples': 8,
        'test_max_samples': 8,
        'train_batchSize': 1,
        'test_batchSize': 1,
        'workers': 0,
        'use_data_augmentation': False,
        'amp': True,
        'amp_initial_scale': 1024,
        'gradient_accumulation_steps': 16,
        'nEpochs': 1,
        'start_epoch': 0,
        'metrics_interval': 1,
        'save_ckpt': True,
        'save_feat': False,
        'save_avg': True,
        'dry_run': False,
    })
    return config


def parse_args():
    parser = argparse.ArgumentParser(description='Build a bounded DFD-HR Mini Run config.')
    parser.add_argument('--detector-path', required=True)
    parser.add_argument('--base-config-path', required=True)
    parser.add_argument('--dataset-json-folder', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--repo-root', required=True)
    parser.add_argument('--data-root', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    output = Path(args.output).expanduser().resolve()
    for protected_root in (args.repo_root, args.data_root):
        if is_within(output, protected_root):
            raise ValueError('Mini Run config output must be outside the repository and data root.')
    config = build_mini_config(
        args.detector_path,
        args.base_config_path,
        args.dataset_json_folder,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=False),
        encoding='utf-8',
    )
    print('mini_config_role=DFDHR_RUNTIME_ROOT/preflight/<config>.yaml')


if __name__ == '__main__':
    main()
