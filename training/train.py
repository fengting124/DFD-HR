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
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from optimizor.SAM import SAM
from optimizor.LinearLR import LinearDecayLR

from trainer.trainer import Trainer
from detectors import DETECTOR
from metrics.utils import parse_metric_for_print
from logger import create_logger, RankFilter

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset

parser = argparse.ArgumentParser(description='Process some paths.')
parser.add_argument('--detector_path', type=str,
                    default='./training/config/detector/dfd_hr.yaml',
                    help='path to detector YAML file')
parser.add_argument("--train_dataset", nargs="+")
parser.add_argument("--test_dataset", nargs="+")
parser.add_argument('--no-save_ckpt', dest='save_ckpt', action='store_false', default=True)
parser.add_argument('--no-save_feat', dest='save_feat', action='store_false', default=True)
parser.add_argument("--ddp", action='store_true', default=False)
parser.add_argument('--local_rank', '--local-rank', type=int, default=0)


def parse_args():
    return parser.parse_args()


def resolve_runtime_device(ddp, local_rank, cuda_enabled):
    if cuda_enabled and torch.cuda.is_available():
        if ddp:
            torch.cuda.set_device(local_rank)
        return torch.device(f'cuda:{local_rank}' if ddp else 'cuda')
    return torch.device('cpu')


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


def init_seed(config):
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    if config['cuda']:
        torch.manual_seed(config['manualSeed'])
        torch.cuda.manual_seed_all(config['manualSeed'])


def prepare_training_data(config):

    train_set = DeepfakeAbstractBaseDataset(
                config=config,
                mode='train',
            )

    if config['ddp']:
        sampler = DistributedSampler(train_set)
        train_data_loader = \
            torch.utils.data.DataLoader(
                dataset=train_set,
                batch_size=config['train_batchSize'],
                num_workers=int(config['workers']),
                collate_fn=train_set.collate_fn,
                sampler=sampler
            )
    else:
        train_data_loader = \
            torch.utils.data.DataLoader(
                dataset=train_set,
                batch_size=config['train_batchSize'],
                shuffle=True,
                num_workers=int(config['workers']),
                collate_fn=train_set.collate_fn,
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
        test_set = DeepfakeAbstractBaseDataset(
                    config=config,
                    mode=mode,
            )

        test_data_loader = \
            torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=config['test_batchSize'],
                shuffle=False,
                num_workers=int(config['workers']),
                collate_fn=test_set.collate_fn,
                drop_last=(dataset_name == 'DeepFakeDetection'),
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
    # parse options and load config
    with open(args.detector_path, 'r') as f:
        config = yaml.safe_load(f)
    with open('./training/config/train_config.yaml', 'r') as f:
        config2 = yaml.safe_load(f)
    config.update(config2)
    config['local_rank']=args.local_rank
    if config['dry_run']:
        config['nEpochs'] = 0
        config['save_feat']=False
    # If arguments are provided, they will overwrite the yaml settings
    if args.train_dataset:
        config['train_dataset'] = args.train_dataset
    if args.test_dataset:
        config['test_dataset'] = args.test_dataset
    config['save_ckpt'] = args.save_ckpt
    config['save_feat'] = args.save_feat
    config['device'] = str(resolve_runtime_device(args.ddp, args.local_rank, config['cuda']))
    if config['lmdb']:
        config['dataset_json_folder'] = 'preprocessing/dataset_json_v3'
    # create logger
    logger_path = config['log_dir']
    os.makedirs(logger_path, exist_ok=True)
    logger = create_logger(os.path.join(logger_path, 'training.log'))
    logger.info('Save log to {}'.format(logger_path))
    config['ddp']= args.ddp
    # print configuration
    logger.info("--------------- Configuration ---------------")
    params_string = "Parameters: \n"
    for key, value in config.items():
        params_string += "{}: {}".format(key, value) + "\n"
    logger.info(params_string)

    # init seed
    init_seed(config)

    # set cudnn benchmark if needed
    if config['cudnn']:
        cudnn.benchmark = True
    if config['ddp']:
        dist.init_process_group(
            backend='nccl' if config['cuda'] and torch.cuda.is_available() else 'gloo',
            timeout=timedelta(minutes=30)
        )
        logger.addFilter(RankFilter(0))
    # prepare the training data loader
    train_data_loader = prepare_training_data(config)

    # prepare the validation and testing data loaders
    validation_dataset_names = resolve_eval_loader_names(config)
    validation_data_loaders = prepare_eval_data(config, 'val', validation_dataset_names) if validation_dataset_names else None
    test_data_loaders = prepare_eval_data(config, 'test', config['test_dataset']) if config.get('test_dataset') else None

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
    trainer = Trainer(config, model, optimizer, scheduler, logger, metric_scoring)

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
    logger.info("Stop Training on best validation metric {}".format(parse_metric_for_print(best_metric)))

    best_validation_ckpt = os.path.join(trainer.log_dir, 'val', 'avg', 'ckpt_best.pth')
    if validation_data_loaders is not None and test_data_loaders is not None and os.path.isfile(best_validation_ckpt):
        trainer.load_ckpt(best_validation_ckpt)
        final_test_metrics = trainer.test_epoch(
            epoch=config['nEpochs'],
            iteration=0,
            test_data_loaders=test_data_loaders,
            step=config['nEpochs'] * max(len(train_data_loader), 1),
            phase='test',
            save_best=False,
        )
        logger.info("Final test metrics after validation-selected training: {}".format(parse_metric_for_print(final_test_metrics)))

    # close the tensorboard writers
    for writer in trainer.writers.values():
        writer.close()



if __name__ == '__main__':
    # python3 -m torch.distributed.launch --nproc_per_node=4 training/train.py --detector_path ./training/config/detector/dfd_hr.yaml --train_dataset FaceForensics++  --test_dataset Celeb-DF-v2 --ddp
    main()
