import argparse
import logging
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from detectors import DETECTOR
from evaluation_utils import (
    atomic_write_json,
    load_checkpoint_strict,
    select_fixed_subset,
    sha256_file,
)
from smoke import (
    apply_initialization_config,
    describe_initialization,
    observe_gradients,
    validate_output_boundary,
)
from train import choose_optimizer
from trainer.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description='Run a bounded two-GPU DDP smoke test.')
    parser.add_argument('--detector_path', required=True)
    parser.add_argument('--test_config_path', required=True)
    parser.add_argument('--dataset_json_folder', required=True)
    parser.add_argument('--data_root', required=True)
    initialization = parser.add_mutually_exclusive_group(required=True)
    initialization.add_argument('--weights_path')
    initialization.add_argument('--clip_model_path')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--dataset', default='FaceForensics++')
    parser.add_argument('--steps', type=int, default=20)
    return parser.parse_args()


def validate_ddp_contract(world_size, steps):
    if world_size != 2:
        raise ValueError(f'DDP smoke requires exactly two ranks, got {world_size}.')
    if steps != 20:
        raise ValueError(f'DDP smoke is fixed at 20 steps, got {steps}.')


def build_config(args, output_dir, local_rank, world_size):
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
        'amp': True,
        'amp_initial_scale': 1024,
        'gradient_accumulation_steps': 1,
        'device': f'cuda:{local_rank}',
        'ddp': True,
        'local_rank': local_rank,
        'log_dir': str(output_dir),
        'save_ckpt': True,
        'save_feat': False,
        'world_size': world_size,
    })
    return apply_initialization_config(
        config,
        weights_path=args.weights_path,
        clip_model_path=args.clip_model_path,
    )


def values_match(left, right):
    return (
        left['python'] == right['python']
        and left['numpy'] == right['numpy']
        and torch.equal(left['torch'], right['torch'])
        and torch.equal(left['cuda'], right['cuda'])
    )


def next_rng_values(device):
    return {
        'python': random.random(),
        'numpy': float(np.random.random()),
        'torch': torch.rand(4),
        'cuda': torch.rand(4, device=device).cpu(),
    }


def balance_strided_rank_labels(dataset):
    indices_by_label = {}
    for label in sorted(set(int(value) for value in dataset.label_list)):
        indices_by_label[label] = [
            index for index, value in enumerate(dataset.label_list)
            if int(value) == label
        ]
    if set(indices_by_label) != {0, 1} or len(indices_by_label[0]) != len(indices_by_label[1]):
        raise ValueError('DDP smoke requires equal binary label groups.')

    order = []
    for offset in range(0, len(indices_by_label[0]), 2):
        for label in (0, 1):
            order.extend(indices_by_label[label][offset:offset + 2])
    dataset.image_list = [dataset.image_list[index] for index in order]
    dataset.label_list = [dataset.label_list[index] for index in order]
    dataset.data_dict = {
        key: [values[index] for index in order]
        for key, values in dataset.data_dict.items()
    }


def run_smoke(args, rank, local_rank, world_size):
    validate_ddp_contract(world_size, args.steps)
    repo_root = Path(subprocess.check_output(
        ['git', 'rev-parse', '--show-toplevel'], text=True
    ).strip()).resolve()
    output_dir = validate_output_boundary(args.output_dir, repo_root, args.data_root)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=False)
    dist.barrier()

    config = build_config(args, output_dir, local_rank, world_size)
    seed = int(config['manualSeed']) + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    dataset = DeepfakeAbstractBaseDataset(config=config, mode='train')
    select_fixed_subset(dataset, args.steps * world_size)
    balance_strided_rank_labels(dataset)
    if len(dataset) != args.steps * world_size:
        raise RuntimeError('DDP smoke subset has an unexpected size.')
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=True,
    )
    sampler.set_epoch(0)
    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=0,
        collate_fn=dataset.collate_fn,
        drop_last=True,
    )
    if len(loader) != args.steps:
        raise RuntimeError('Each DDP rank must receive exactly 20 batches.')

    model = DETECTOR[config['model_name']](config)
    checkpoint_info = None
    if args.weights_path:
        checkpoint_info = load_checkpoint_strict(model, args.weights_path)
    optimizer = choose_optimizer(model, config)
    logger = logging.getLogger(f'ddp_smoke.rank{rank}')
    logger.handlers[:] = [logging.NullHandler()]
    trainer = Trainer(
        config=config,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        logger=logger,
        metric_scoring=config['metric_scoring'],
        time_now='ddp',
    )
    if trainer.config['effective_batch_size'] != 2:
        raise RuntimeError('DDP effective batch size must be two.')

    model_without_ddp = trainer.model.module
    frozen_backbone = {
        name: parameter
        for name, parameter in model_without_ddp.named_parameters()
        if name.startswith(('backbone.', 'visual_projection.'))
    }
    tracked_name, tracked_parameter = next(
        (name, parameter)
        for name, parameter in model_without_ddp.named_parameters()
        if parameter.requires_grad
    )
    initial_parameter = tracked_parameter.detach().cpu().clone()

    losses = []
    labels_seen = []
    step_seconds = []
    gradient_steps = []
    torch.cuda.reset_peak_memory_stats(local_rank)
    trainer.optimizer.zero_grad(set_to_none=True)
    for data_dict in loader:
        labels_seen.extend(int(label) for label in data_dict['label'])
        trainer.setTrain()
        data_dict = trainer.move_data_dict_to_device(data_dict)
        torch.cuda.synchronize(local_rank)
        start = time.perf_counter()
        batch_losses, _ = trainer.train_step(
            data_dict,
            gradient_observer=lambda model: gradient_steps.append(observe_gradients(model)),
        )
        torch.cuda.synchronize(local_rank)
        step_seconds.append(time.perf_counter() - start)
        loss = float(batch_losses['overall'].detach().cpu().item())
        if not np.isfinite(loss):
            raise FloatingPointError('DDP smoke loss is not finite.')
        losses.append(loss)

    required_categories = {'adapter', 'router', 'head', 'query'}
    final_gradients = gradient_steps[-1]
    local_checks_ok = (
        len(losses) == args.steps
        and required_categories.issubset(final_gradients)
        and all(final_gradients[name]['finite'] for name in required_categories)
        and all(final_gradients[name]['tensors'] > 0 for name in required_categories)
        and not torch.equal(initial_parameter, tracked_parameter.detach().cpu())
        and not any(parameter.grad is not None for parameter in frozen_backbone.values())
        and labels_seen.count(0) == args.steps // 2
        and labels_seen.count(1) == args.steps // 2
    )
    check_tensor = torch.tensor(int(local_checks_ok), device=local_rank)
    dist.all_reduce(check_tensor, op=dist.ReduceOp.MIN)
    if check_tensor.item() != 1:
        raise RuntimeError('At least one rank failed loss, gradient, optimizer, or freeze checks.')

    parameter_probe = tracked_parameter.detach().float().sum()
    gathered_probes = [torch.zeros_like(parameter_probe) for _ in range(world_size)]
    dist.all_gather(gathered_probes, parameter_probe)
    if not all(torch.equal(gathered_probes[0], probe) for probe in gathered_probes[1:]):
        raise RuntimeError('DDP trainable parameters diverged across ranks.')

    synchronized_result = trainer._run_rank_zero_synchronized(
        lambda: {'completed_steps': args.steps, 'world_size': world_size}
    )
    if synchronized_result != {'completed_steps': args.steps, 'world_size': world_size}:
        raise RuntimeError('Rank-zero result broadcast failed.')

    checkpoint_path = trainer.save_last_ckpt_synchronized(epoch=0)
    expected_rng = next_rng_values(trainer.device)
    expected_parameter = tracked_parameter.detach().cpu().clone()
    with torch.no_grad():
        tracked_parameter.add_(1)
    random.seed(9000 + rank)
    np.random.seed(9000 + rank)
    torch.manual_seed(9000 + rank)
    torch.cuda.manual_seed_all(9000 + rank)
    next_epoch = trainer.resume_from_checkpoint(checkpoint_path)
    restored_parameter = dict(model_without_ddp.named_parameters())[tracked_name].detach().cpu()
    restored_rng = next_rng_values(trainer.device)
    roundtrip_ok = (
        next_epoch == 1
        and torch.equal(expected_parameter, restored_parameter)
        and values_match(expected_rng, restored_rng)
    )
    roundtrip_tensor = torch.tensor(int(roundtrip_ok), device=local_rank)
    dist.all_reduce(roundtrip_tensor, op=dist.ReduceOp.MIN)
    if roundtrip_tensor.item() != 1:
        raise RuntimeError('At least one rank failed checkpoint or RNG round-trip.')

    rng_probe = torch.tensor(
        [expected_rng['python'], expected_rng['numpy']],
        device=local_rank,
        dtype=torch.float64,
    )
    rng_probes = [torch.zeros_like(rng_probe) for _ in range(world_size)]
    dist.all_gather(rng_probes, rng_probe)
    if torch.equal(rng_probes[0], rng_probes[1]):
        raise RuntimeError('Per-rank RNG streams unexpectedly collapsed to one state.')

    rank_summary = {
        'rank': rank,
        'steps': len(losses),
        'label_counts': {'0': labels_seen.count(0), '1': labels_seen.count(1)},
        'loss_first': losses[0],
        'loss_last': losses[-1],
        'mean_step_seconds': sum(step_seconds) / len(step_seconds),
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(local_rank),
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(local_rank),
        'amp_final_scale': trainer.scaler.get_scale(),
        'final_gradients': final_gradients,
        'roundtrip': roundtrip_ok,
    }
    rank_summaries = [None] * world_size
    dist.all_gather_object(rank_summaries, rank_summary)

    if rank != 0:
        return None
    checkpoint_report = {
        'roundtrip_all_ranks': True,
        'saved_sha256': sha256_file(checkpoint_path),
        'saved_size_bytes': os.path.getsize(checkpoint_path),
        'temporary_file_absent': not os.path.exists(checkpoint_path + '.tmp'),
        'rng_rank_count': world_size,
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
        'backend': dist.get_backend(),
        'world_size': world_size,
        'steps_per_rank': args.steps,
        'micro_batch_size': 1,
        'effective_batch_size': trainer.config['effective_batch_size'],
        'amp_initial_scale': config['amp_initial_scale'],
        'dataset_role': args.dataset,
        'dataset_samples': len(dataset),
        'rank_zero_broadcast': synchronized_result,
        'parameter_synchronized': True,
        'per_rank_rng_restored': True,
        'rank_summaries': rank_summaries,
        'checkpoint': checkpoint_report,
        'initialization': describe_initialization(args, checkpoint_info),
        'config_sha256': sha256_file(args.detector_path),
        'dataset_json_sha256': sha256_file(
            os.path.join(args.dataset_json_folder, f'{args.dataset}.json')
        ),
    }


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the DDP smoke test.')
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    report = None
    try:
        report = run_smoke(args, rank, local_rank, world_size)
        dist.barrier()
    finally:
        dist.destroy_process_group()

    if rank == 0:
        report['process_group_destroyed'] = not dist.is_initialized()
        report_path = Path(args.output_dir).resolve().parent / 'ddp.json'
        atomic_write_json(report, report_path)
        print('status=ok mode=ddp world_size=2 steps_per_rank=20')
        print('report_role=output_parent/ddp.json')


if __name__ == '__main__':
    main()
