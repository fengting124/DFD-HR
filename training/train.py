# author: Jiamu Sun
# email: genisun@tencent.com
# date: 2026-05-28
# description: training code.

import os
import argparse
from os.path import join
import cv2
import random
import datetime
import time
import yaml
from tqdm import tqdm
import numpy as np
from datetime import timedelta
from copy import deepcopy
from PIL import Image as pil_image

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.optim as optim
from torch.utils.data import Sampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from optimizor.SAM import SAM
from optimizor.LinearLR import LinearDecayLR

from trainer.trainer import Trainer
from detectors import DETECTOR
from metrics.utils import parse_metric_for_print
from evaluation_utils import select_fixed_subset
from logger import create_logger, RankFilter

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset

parser = argparse.ArgumentParser(description='Process some paths.')
parser.add_argument('--detector_path', type=str,
                    default='./training/config/detector/dfd_hr.yaml',
                    help='path to detector YAML file')
parser.add_argument("--train_dataset", nargs="+")
parser.add_argument("--validation_dataset", nargs="+")
parser.add_argument("--test_dataset", nargs="+")
parser.add_argument("--resume", type=str)
parser.add_argument('--no-save_ckpt', dest='save_ckpt', action='store_false', default=True)
parser.add_argument('--no-save_feat', dest='save_feat', action='store_false', default=True)
parser.add_argument("--ddp", action='store_true', default=False)
parser.add_argument('--local_rank', '--local-rank', type=int)


def parse_args():
    return parser.parse_args()


def resolve_local_rank(cli_local_rank, environment=None):
    if cli_local_rank is not None:
        local_rank = cli_local_rank
    else:
        environment = os.environ if environment is None else environment
        try:
            local_rank = int(environment.get('LOCAL_RANK', 0))
        except (TypeError, ValueError) as error:
            raise ValueError('LOCAL_RANK must be a non-negative integer') from error
    if local_rank < 0:
        raise ValueError('LOCAL_RANK must be a non-negative integer')
    return local_rank


def configure_cublas_workspace(config, environment=None):
    mode = config.get('reproducibility_mode', 'seeded_best_effort')
    if 'cublas_workspace_config' not in config and mode != 'deterministic':
        return None
    environment = os.environ if environment is None else environment
    workspace_config = config.get('cublas_workspace_config', ':4096:8')
    if workspace_config not in {':4096:8', ':16:8'}:
        raise ValueError('cublas_workspace_config must be :4096:8 or :16:8')
    existing = environment.get('CUBLAS_WORKSPACE_CONFIG')
    if existing is not None and existing != workspace_config:
        raise ValueError('CUBLAS_WORKSPACE_CONFIG conflicts with the deterministic config')
    environment['CUBLAS_WORKSPACE_CONFIG'] = workspace_config
    return workspace_config


def resolve_runtime_device(ddp, local_rank, cuda_enabled):
    if cuda_enabled and torch.cuda.is_available():
        if ddp:
            torch.cuda.set_device(local_rank)
        return torch.device(f'cuda:{local_rank}' if ddp else 'cuda')
    return torch.device('cpu')


def resolve_ddp_timeout(config):
    timeout_minutes = config.get('ddp_timeout_minutes', 30)
    if isinstance(timeout_minutes, bool) or not isinstance(timeout_minutes, int):
        raise ValueError('ddp_timeout_minutes must be a positive integer')
    if timeout_minutes <= 0:
        raise ValueError('ddp_timeout_minutes must be a positive integer')
    return timedelta(minutes=timeout_minutes)


def build_epoch_range(config):
    return range(config['start_epoch'], config['nEpochs'])


def resolve_eval_loader_names(config):
    validation_dataset = config.get('validation_dataset')
    if not validation_dataset:
        raise ValueError(
            'validation_dataset must explicitly name at least one dataset; '
            'test_dataset cannot be used for checkpoint selection.'
        )
    return validation_dataset


def should_run_final_test(config):
    enabled = config.get('run_final_test_after_training', True)
    if not isinstance(enabled, bool):
        raise ValueError('run_final_test_after_training must be a boolean')
    return enabled and bool(config.get('test_dataset'))


def load_training_config(detector_path, base_config_path='./training/config/train_config.yaml'):
    with open(base_config_path, encoding='utf-8') as file:
        config = yaml.safe_load(file) or {}
    with open(detector_path, encoding='utf-8') as file:
        config.update(yaml.safe_load(file) or {})
    return config


def apply_fixed_subset(dataset, max_samples, role):
    if max_samples is None:
        return len(dataset)
    selected = select_fixed_subset(dataset, max_samples)
    labels = {int(label) for label in dataset.label_list}
    if selected != max_samples:
        raise ValueError(f'{role} fixed subset requested {max_samples} samples but found {selected}.')
    if labels != {0, 1}:
        raise ValueError(f'{role} fixed subset must contain both binary classes.')
    return selected


def configure_reproducibility(config, rank=0):
    mode = config.get('reproducibility_mode', 'seeded_best_effort')
    if mode not in {'deterministic', 'seeded_best_effort'}:
        raise ValueError('reproducibility_mode must be deterministic or seeded_best_effort')
    if config.get('manualSeed') is None:
        config['manualSeed'] = random.SystemRandom().randint(1, 10000)

    seed = int(config['manualSeed']) + int(rank)
    config['runtime_seed'] = seed
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if config.get('cuda') and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    deterministic = mode == 'deterministic'
    benchmark = config.get(
        'cudnn_benchmark',
        bool(config.get('cudnn', False)) and not deterministic,
    )
    cudnn_deterministic = config.get('cudnn_deterministic', deterministic)
    if not isinstance(benchmark, bool) or not isinstance(cudnn_deterministic, bool):
        raise ValueError('cudnn_benchmark and cudnn_deterministic must be booleans')
    if deterministic and (benchmark or not cudnn_deterministic):
        raise ValueError(
            'deterministic mode requires cudnn_benchmark=false and cudnn_deterministic=true'
        )
    cudnn.benchmark = benchmark
    cudnn.deterministic = cudnn_deterministic
    deterministic_algorithms = config.get('deterministic_algorithms', deterministic)
    if not isinstance(deterministic_algorithms, bool):
        raise ValueError('deterministic_algorithms must be a boolean')
    if deterministic and not deterministic_algorithms:
        raise ValueError('deterministic mode requires deterministic_algorithms=true')
    torch.use_deterministic_algorithms(deterministic_algorithms)
    return seed


def seed_data_loader_worker(worker_id):
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def build_data_loader_generator(config, role):
    role_offsets = {'train': 0, 'val': 10_000, 'test': 20_000}
    if role not in role_offsets:
        raise ValueError(f'Unsupported data loader role: {role}')
    generator = torch.Generator()
    generator.manual_seed(int(config['runtime_seed']) + role_offsets[role])
    return generator


class DistributedEvalSampler(Sampler):
    """Partition evaluation data exactly, without padding or duplication."""

    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            if not dist.is_available() or not dist.is_initialized():
                raise RuntimeError('Distributed evaluation requires an initialized process group')
            num_replicas = dist.get_world_size()
        if rank is None:
            rank = dist.get_rank()
        if num_replicas < 1 or rank < 0 or rank >= num_replicas:
            raise ValueError('Invalid distributed evaluation rank or world size')
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.start = len(dataset) * rank // num_replicas
        self.end = len(dataset) * (rank + 1) // num_replicas

    def __iter__(self):
        return iter(range(self.start, self.end))

    def __len__(self):
        return self.end - self.start


def build_eval_dataset(config, mode):
    role_offsets = {'val': 10_000, 'test': 20_000}
    if mode not in role_offsets:
        raise ValueError(f'Unsupported evaluation mode: {mode}')
    seed = int(config['manualSeed']) + role_offsets[mode]
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    try:
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        return DeepfakeAbstractBaseDataset(config=config, mode=mode)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)


def prepare_training_data(config):

    train_set = DeepfakeAbstractBaseDataset(
                config=config,
                mode='train',
            )
    apply_fixed_subset(train_set, config.get('train_max_samples'), 'train')

    if config['ddp']:
        sampler = DistributedSampler(train_set, seed=int(config['manualSeed']))
        train_data_loader = \
            torch.utils.data.DataLoader(
                dataset=train_set,
                batch_size=config['train_batchSize'],
                num_workers=int(config['workers']),
                collate_fn=train_set.collate_fn,
                sampler=sampler,
                worker_init_fn=seed_data_loader_worker,
                generator=build_data_loader_generator(config, 'train'),
            )
    else:
        train_data_loader = \
            torch.utils.data.DataLoader(
                dataset=train_set,
                batch_size=config['train_batchSize'],
                shuffle=True,
                num_workers=int(config['workers']),
                collate_fn=train_set.collate_fn,
                worker_init_fn=seed_data_loader_worker,
                generator=build_data_loader_generator(config, 'train'),
                )
    return train_data_loader


def prepare_eval_data(config, mode, dataset_names):
    def get_eval_data_loader(config, dataset_name):
        # update the config dictionary with the specific testing dataset
        config = config.copy()  # create a copy of config to avoid altering the original one
        if mode == 'val':
            config['validation_dataset'] = dataset_name
        else:
            config['test_dataset'] = dataset_name
        test_set = build_eval_dataset(config, mode)
        limit_key = 'validation_max_samples' if mode == 'val' else 'test_max_samples'
        apply_fixed_subset(test_set, config.get(limit_key), mode)

        distributed_evaluation = bool(
            config.get('distributed_validation', False)
            and config.get('ddp', False)
            and dist.is_available()
            and dist.is_initialized()
        )
        sampler = DistributedEvalSampler(test_set) if distributed_evaluation else None

        test_data_loader = \
            torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=config['test_batchSize'],
                shuffle=False,
                num_workers=int(config['workers']),
                collate_fn=test_set.collate_fn,
                sampler=sampler,
                drop_last=(dataset_name == 'DeepFakeDetection' and not distributed_evaluation),
                worker_init_fn=seed_data_loader_worker,
                generator=build_data_loader_generator(config, mode),
            )

        return test_data_loader

    data_loaders = {}
    for dataset_name in dataset_names:
        data_loaders[dataset_name] = get_eval_data_loader(config, dataset_name)
    return data_loaders


def choose_optimizer(model, config):
    opt_name = config['optimizer']['type']
    if opt_name == 'sgd':
        optimizer = optim.SGD(
            params=model.parameters(),
            lr=config['optimizer'][opt_name]['lr'],
            momentum=config['optimizer'][opt_name]['momentum'],
            weight_decay=config['optimizer'][opt_name]['weight_decay']
        )
        return optimizer
    elif opt_name == 'adam':
        optimizer = optim.Adam(
            params=model.parameters(),
            lr=config['optimizer'][opt_name]['lr'],
            weight_decay=config['optimizer'][opt_name]['weight_decay'],
            betas=(config['optimizer'][opt_name]['beta1'], config['optimizer'][opt_name]['beta2']),
            eps=config['optimizer'][opt_name]['eps'],
            amsgrad=config['optimizer'][opt_name]['amsgrad'],
        )
        return optimizer
    elif opt_name == 'sam':
        optimizer = SAM(
            model.parameters(),
            optim.SGD,
            lr=config['optimizer'][opt_name]['lr'],
            momentum=config['optimizer'][opt_name]['momentum'],
        )
    else:
        raise NotImplementedError('Optimizer {} is not implemented'.format(config['optimizer']))
    return optimizer


def choose_scheduler(config, optimizer):
    if config['lr_scheduler'] is None:
        return None
    elif config['lr_scheduler'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['lr_step'],
            gamma=config['lr_gamma'],
        )
        return scheduler
    elif config['lr_scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['lr_T_max'],
            eta_min=config['lr_eta_min'],
        )
        return scheduler
    elif config['lr_scheduler'] == 'linear':
        scheduler = LinearDecayLR(
            optimizer,
            config['nEpochs'],
            int(config['nEpochs']/4),
        )
    else:
        raise NotImplementedError('Scheduler {} is not implemented'.format(config['lr_scheduler']))


def choose_metric(config):
    metric_scoring = config['metric_scoring']
    if metric_scoring not in ['eer', 'auc', 'acc', 'ap']:
        raise NotImplementedError('metric {} is not implemented'.format(metric_scoring))
    return metric_scoring


def main():
    args = parse_args()
    local_rank = resolve_local_rank(args.local_rank)
    # parse options and load config
    config = load_training_config(args.detector_path)
    config['local_rank']=local_rank
    if config['dry_run']:
        config['nEpochs'] = 0
        config['save_feat']=False
    # If arguments are provided, they will overwrite the yaml settings
    if args.train_dataset:
        config['train_dataset'] = args.train_dataset
    if args.validation_dataset:
        config['validation_dataset'] = args.validation_dataset
    if args.test_dataset:
        config['test_dataset'] = args.test_dataset
    config['save_ckpt'] = args.save_ckpt
    config['save_feat'] = args.save_feat
    configure_cublas_workspace(config)
    config['device'] = str(resolve_runtime_device(args.ddp, local_rank, config['cuda']))
    if config['lmdb']:
        config['dataset_json_folder'] = 'preprocessing/dataset_json_v3'
    # create logger
    config['ddp']= args.ddp
    logger_path = config['log_dir']
    os.makedirs(logger_path, exist_ok=True)
    logger = create_logger(os.path.join(logger_path, 'training.log'))
    logger.info('Save log to {}'.format(logger_path))
    # print configuration
    logger.info("--------------- Configuration ---------------")
    params_string = "Parameters: \n"
    for key, value in config.items():
        params_string += "{}: {}".format(key, value) + "\n"
    logger.info(params_string)

    if config['ddp']:
        dist.init_process_group(
            backend='nccl' if config['cuda'] and torch.cuda.is_available() else 'gloo',
            timeout=resolve_ddp_timeout(config),
        )
        logger.addFilter(RankFilter(0))
    rank = dist.get_rank() if config['ddp'] else 0
    configure_reproducibility(config, rank=rank)
    # prepare the training data loader
    train_data_loader = prepare_training_data(config)

    # prepare the validation and testing data loaders
    validation_dataset_names = resolve_eval_loader_names(config)
    validation_data_loaders = prepare_eval_data(config, 'val', validation_dataset_names) if validation_dataset_names else None
    test_data_loaders = (
        prepare_eval_data(config, 'test', config['test_dataset'])
        if should_run_final_test(config)
        else None
    )

    # prepare the model (detector)
    model_class = DETECTOR[config['model_name']]
    model = model_class(config)
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # num_total_param = sum(p.numel() for p in model.parameters())
    # print('Number of total parameters: {}, tunable parameters: {}'.format(num_total_param, num_param))
    # from pdb import set_trace as st
    # st()

    # prepare the optimizer
    optimizer = choose_optimizer(model, config)

    # prepare the scheduler
    scheduler = choose_scheduler(config, optimizer)

    # prepare the metric
    metric_scoring = choose_metric(config)

    # prepare the trainer
    trainer_kwargs = {}
    if config.get('run_id'):
        trainer_kwargs['time_now'] = config['run_id']
    trainer = Trainer(
        config,
        model,
        optimizer,
        scheduler,
        logger,
        metric_scoring,
        **trainer_kwargs,
    )
    if args.resume:
        config['start_epoch'] = trainer.resume_from_checkpoint(args.resume)

    # start training
    best_metric = None
    for epoch in build_epoch_range(config):
        if config['ddp'] and isinstance(train_data_loader.sampler, DistributedSampler):
            train_data_loader.sampler.set_epoch(epoch)
        trainer.model.epoch = epoch
        best_metric = trainer.train_epoch(
                    epoch=epoch,
                    train_data_loader=train_data_loader,
                    eval_data_loaders=validation_data_loaders,
                    eval_phase='val',
                )
        if scheduler is not None and not config.get('SWA', False):
            scheduler.step()
        if best_metric is not None:
            logger.info(f"===> Epoch[{epoch}] end with validation {metric_scoring}: {parse_metric_for_print(best_metric)}!")
        if config['save_ckpt']:
            trainer.save_last_ckpt_synchronized(epoch)
    logger.info("Stop Training on best validation metric {}".format(parse_metric_for_print(best_metric)))

    if validation_data_loaders is not None and test_data_loaders is not None and should_run_final_test(config):
        def run_final_test():
            best_validation_ckpt = os.path.join(
                trainer.log_dir, 'val', 'avg', 'ckpt_best.pth'
            )
            if not os.path.isfile(best_validation_ckpt):
                return None
            trainer.load_ckpt(best_validation_ckpt)
            return trainer.test_epoch(
                epoch=config['nEpochs'],
                iteration=0,
                test_data_loaders=test_data_loaders,
                step=config['nEpochs'] * max(len(train_data_loader), 1),
                phase='test',
                save_best=False,
            )

        if trainer._distributed_validation_enabled():
            final_test_metrics = run_final_test()
        else:
            final_test_metrics = trainer._run_rank_zero_synchronized(run_final_test)
        if final_test_metrics is not None:
            logger.info("Final test metrics after validation-selected training: {}".format(parse_metric_for_print(final_test_metrics)))

    # close the tensorboard writers
    for writer in trainer.writers.values():
        writer.close()



if __name__ == '__main__':
    # python3 -m torch.distributed.launch --nproc_per_node=4 training/train.py --detector_path ./training/config/detector/dfd_hr.yaml --train_dataset FaceForensics++ --validation_dataset FaceForensics++ --test_dataset Celeb-DF-v2 --ddp
    main()
