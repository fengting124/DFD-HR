import argparse
import logging
import os
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from detectors import DETECTOR
from evaluation_utils import (
    atomic_write_json,
    load_checkpoint_strict,
    select_fixed_subset,
    sha256_file,
)
from train import choose_optimizer
from trainer.trainer import Trainer


GRADIENT_CATEGORIES = {
    'adapter': ('adapters_attn.', 'adapters_mlp.'),
    'router': ('token_router.', 'layer_router.'),
    'head': ('head.',),
    'query': ('query_token', 'query_attn.'),
}


def parse_args():
    parser = argparse.ArgumentParser(description='Run exactly two bounded single-GPU training steps.')
    parser.add_argument('--detector_path', required=True)
    parser.add_argument('--test_config_path', required=True)
    parser.add_argument('--dataset_json_folder', required=True)
    parser.add_argument('--data_root', required=True)
    initialization = parser.add_mutually_exclusive_group(required=True)
    initialization.add_argument('--weights_path')
    initialization.add_argument('--clip_model_path')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--precision', choices=('fp32', 'amp'), required=True)
    parser.add_argument('--dataset', default='FaceForensics++')
    return parser.parse_args()


def gradient_category(parameter_name):
    parameter_name = parameter_name.removeprefix('module.')
    for category, prefixes in GRADIENT_CATEGORIES.items():
        if parameter_name.startswith(prefixes):
            return category
    return 'other'


def validate_output_boundary(output_dir, repo_root, data_root):
    output_dir = Path(output_dir).resolve()
    for protected_root in (Path(repo_root).resolve(), Path(data_root).resolve()):
        if output_dir == protected_root or protected_root in output_dir.parents:
            raise ValueError('Smoke output must be outside the repository and data root.')
    return output_dir


def apply_initialization_config(config, weights_path=None, clip_model_path=None):
    if bool(weights_path) == bool(clip_model_path):
        raise ValueError('Specify exactly one of weights_path or clip_model_path.')
    if weights_path:
        config['backbone_pretrained'] = False
        return config

    clip_model_path = Path(clip_model_path).resolve()
    if not clip_model_path.is_dir():
        raise ValueError('clip_model_path must be an existing local directory.')
    if not (clip_model_path / 'model.safetensors').is_file():
        raise ValueError('clip_model_path must contain model.safetensors.')
    config.update({
        'backbone_pretrained': True,
        'backbone_pretrained_path': str(clip_model_path),
        'backbone_local_files_only': True,
    })
    return config


def describe_initialization(args, checkpoint_info=None):
    if args.weights_path:
        return {
            'type': 'dfd_checkpoint',
            'source_sha256': sha256_file(args.weights_path),
            'source_tensor_count': checkpoint_info['tensor_count'],
        }
    model_path = Path(args.clip_model_path).resolve() / 'model.safetensors'
    return {
        'type': 'clip_pretrained',
        'model_sha256': sha256_file(model_path),
        'model_size_bytes': model_path.stat().st_size,
        'local_files_only': True,
        'dfd_checkpoint_loaded': False,
    }


def build_config(args, output_dir):
    with open(args.detector_path, encoding='utf-8') as file:
        config = yaml.safe_load(file)
    with open(args.test_config_path, encoding='utf-8') as file:
        config.update(yaml.safe_load(file))
    config.update({
        'dataset_json_folder': str(Path(args.dataset_json_folder).resolve()),
        'train_dataset': [args.dataset],
        'train_batchSize': 1,
        'workers': 0,
        'use_data_augmentation': False,
        'amp': args.precision == 'amp',
        'amp_initial_scale': 1024,
        'gradient_accumulation_steps': 1,
        'device': 'cuda',
        'ddp': False,
        'local_rank': 0,
        'log_dir': str(output_dir),
        'save_ckpt': True,
        'save_feat': False,
    })
    return apply_initialization_config(
        config,
        weights_path=args.weights_path,
        clip_model_path=args.clip_model_path,
    )


def observe_gradients(model):
    observations = defaultdict(lambda: {'tensors': 0, 'finite': True, 'max_abs': 0.0})
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or parameter.grad is None:
            continue
        category = gradient_category(name)
        gradient = parameter.grad.detach()
        entry = observations[category]
        entry['tensors'] += 1
        finite = bool(torch.isfinite(gradient).all().item())
        entry['finite'] = entry['finite'] and finite
        if finite:
            entry['max_abs'] = max(entry['max_abs'], float(gradient.abs().max().item()))
    return dict(observations)


def run_smoke(args):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the single-GPU smoke test.')
    repo_root = Path(subprocess.check_output(
        ['git', 'rev-parse', '--show-toplevel'], text=True
    ).strip()).resolve()
    output_dir = validate_output_boundary(args.output_dir, repo_root, args.data_root)
    output_dir.mkdir(parents=True, exist_ok=False)
    config = build_config(args, output_dir)

    torch.manual_seed(config['manualSeed'])
    torch.cuda.manual_seed_all(config['manualSeed'])
    dataset = DeepfakeAbstractBaseDataset(config=config, mode='train')
    select_fixed_subset(dataset, 2)
    if len(dataset) != 2 or len(set(int(label) for label in dataset.label_list)) != 2:
        raise RuntimeError('Smoke subset must contain exactly one sample per binary class.')
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=dataset.collate_fn,
        drop_last=False,
    )

    model = DETECTOR[config['model_name']](config)
    checkpoint_info = None
    if args.weights_path:
        checkpoint_info = load_checkpoint_strict(model, args.weights_path)
    frozen_backbone = {
        name: parameter
        for name, parameter in model.named_parameters()
        if name.startswith(('backbone.', 'visual_projection.'))
    }
    if not frozen_backbone or any(parameter.requires_grad for parameter in frozen_backbone.values()):
        raise RuntimeError('Backbone and visual projection must remain frozen.')

    optimizer = choose_optimizer(model, config)
    logger = logging.getLogger(f'single_gpu_smoke.{args.precision}')
    logger.handlers[:] = [logging.NullHandler()]
    trainer = Trainer(
        config=config,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        logger=logger,
        metric_scoring=config['metric_scoring'],
        time_now=args.precision,
    )
    gradient_steps = []
    tracked_parameter = next(
        parameter for parameter in trainer.model.parameters() if parameter.requires_grad
    )
    initial_parameter = tracked_parameter.detach().cpu().clone()

    torch.cuda.reset_peak_memory_stats()
    losses = []
    step_seconds = []
    completed_batches = 0
    for data_dict in loader:
        trainer.setTrain()
        data_dict = trainer.move_data_dict_to_device(data_dict)
        torch.cuda.synchronize()
        start = time.perf_counter()
        batch_losses, _ = trainer.train_step(
            data_dict,
            gradient_observer=lambda model: gradient_steps.append(observe_gradients(model)),
        )
        torch.cuda.synchronize()
        step_seconds.append(time.perf_counter() - start)
        loss = float(batch_losses['overall'].detach().cpu().item())
        if not torch.isfinite(torch.tensor(loss)):
            raise FloatingPointError('Smoke loss is not finite.')
        losses.append(loss)
        completed_batches += 1

    required_categories = {'adapter', 'router', 'head', 'query'}
    if len(gradient_steps) != 2:
        raise RuntimeError('Expected one unscaled gradient observation per batch.')
    final_gradients = gradient_steps[-1]
    missing_categories = sorted(
        category for category in required_categories
        if final_gradients.get(category, {}).get('tensors', 0) == 0
    )
    if missing_categories:
        raise RuntimeError(f'Missing gradient categories: {missing_categories}')
    if any(not final_gradients[category]['finite'] for category in required_categories):
        raise FloatingPointError('A required trainable category has non-finite gradients.')
    if any(parameter.grad is not None for parameter in frozen_backbone.values()):
        raise RuntimeError('Frozen backbone unexpectedly received gradients.')
    optimizer_updated = not torch.equal(initial_parameter, tracked_parameter.detach().cpu())
    if not optimizer_updated:
        raise RuntimeError('Optimizer did not update the tracked trainable parameter.')

    model_to_check = trainer.model
    parameter_name, parameter = next(
        (name, parameter)
        for name, parameter in model_to_check.named_parameters()
        if parameter.requires_grad
    )
    expected_parameter = parameter.detach().cpu().clone()
    checkpoint_path = trainer.save_last_ckpt(epoch=0)
    with torch.no_grad():
        parameter.add_(1)
    next_epoch = trainer.resume_from_checkpoint(checkpoint_path)
    restored_parameter = dict(trainer.model.named_parameters())[parameter_name].detach().cpu()
    roundtrip_ok = next_epoch == 1 and torch.equal(expected_parameter, restored_parameter)
    if not roundtrip_ok:
        raise RuntimeError('Full checkpoint round-trip did not restore model state.')

    checkpoint_report = {
        'roundtrip': roundtrip_ok,
        'saved_sha256': sha256_file(checkpoint_path),
        'saved_size_bytes': os.path.getsize(checkpoint_path),
        'temporary_file_absent': not os.path.exists(checkpoint_path + '.tmp'),
    }
    if checkpoint_info is not None:
        checkpoint_report.update({
            'source_sha256': sha256_file(args.weights_path),
            'source_tensor_count': checkpoint_info['tensor_count'],
        })

    return {
        'schema_version': 1,
        'status': 'ok',
        'code': {
            'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip()),
        },
        'mode': args.precision,
        'amp_initial_scale': config['amp_initial_scale'],
        'amp_final_scale': trainer.scaler.get_scale(),
        'dataset_role': args.dataset,
        'batches': completed_batches,
        'micro_batch_size': 1,
        'resolution': config['resolution'],
        'data_augmentation': config['use_data_augmentation'],
        'losses': losses,
        'step_seconds': step_seconds,
        'peak_memory_bytes': {
            'allocated': torch.cuda.max_memory_allocated(),
            'reserved': torch.cuda.max_memory_reserved(),
        },
        'gradient_steps': gradient_steps,
        'optimizer_updated': optimizer_updated,
        'frozen_backbone_parameters': len(frozen_backbone),
        'checkpoint': checkpoint_report,
        'initialization': describe_initialization(args, checkpoint_info),
        'config_sha256': sha256_file(args.detector_path),
        'dataset_json_sha256': sha256_file(
            os.path.join(args.dataset_json_folder, f'{args.dataset}.json')
        ),
    }


def main():
    args = parse_args()
    report_path = Path(args.output_dir).resolve().parent / f'{args.precision}.json'
    try:
        report = run_smoke(args)
    except RuntimeError as error:
        if 'out of memory' not in str(error).lower():
            raise
        torch.cuda.empty_cache()
        report = {
            'schema_version': 1,
            'status': 'oom',
            'mode': args.precision,
            'error': 'CUDA out of memory',
            'code': {
                'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                'dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip()),
            },
        }
    atomic_write_json(report, report_path)
    print(f"status={report['status']} mode={args.precision}")
    print('report_role=output_parent/<precision>.json')


if __name__ == '__main__':
    main()
