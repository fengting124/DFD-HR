"""
eval pretained model.
"""
import os
import numpy as np
from os.path import join
import cv2
import random
import datetime
import time
import yaml
import pickle
import subprocess
from tqdm import tqdm
from copy import deepcopy
from PIL import Image as pil_image
from metrics.utils import get_test_metrics
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset

from trainer.trainer import Trainer
from detectors import DETECTOR
from metrics.base_metrics_class import Recorder
from collections import defaultdict

import argparse
from logger import create_logger
from evaluation_utils import (
    atomic_write_json,
    load_checkpoint_strict,
    select_fixed_subset,
    sha256_file,
    summarize_metrics,
)

parser = argparse.ArgumentParser(description='Process some paths.')
parser.add_argument('--detector_path', type=str, 
                    default='./training/config/detector/dfd_hr.yaml',
                    help='path to detector YAML file')
parser.add_argument("--test_dataset", nargs="+")
parser.add_argument('--weights_path', type=str, 
                    default='./logs/dfd_hr/ckpt_best.pth')
parser.add_argument('--architecture_only', action='store_true',
                    help='construct the backbone without downloading pretrained weights')
parser.add_argument('--trusted_checkpoint', action='store_true',
                    help='allow full deserialization of a verified project checkpoint')
parser.add_argument('--max_samples_per_dataset', type=int)
parser.add_argument('--test_batch_size', type=int)
parser.add_argument('--workers', type=int)
parser.add_argument('--output_path', type=str)
#parser.add_argument("--lmdb", action='store_true', default=False)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

on_2060 = torch.cuda.is_available() and "2060" in torch.cuda.get_device_name()
def init_seed(config):
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    torch.manual_seed(config['manualSeed'])
    if config['cuda'] and torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['manualSeed'])


def prepare_testing_data(config):
    def get_test_data_loader(config, test_name):
        # update the config dictionary with the specific testing dataset
        config = config.copy()  # create a copy of config to avoid altering the original one
        config['test_dataset'] = test_name  # specify the current test dataset
        test_set = DeepfakeAbstractBaseDataset(
                config=config,
                mode='test', 
            )
        select_fixed_subset(test_set, config.get('max_samples_per_dataset'))
        test_data_loader = \
            torch.utils.data.DataLoader(
                dataset=test_set, 
                batch_size=config['test_batchSize'],
                shuffle=False, 
                num_workers=int(config['workers']),
                collate_fn=test_set.collate_fn,
                drop_last=False
            )
        return test_data_loader

    test_data_loaders = {}
    for one_test_name in config['test_dataset']:
        test_data_loaders[one_test_name] = get_test_data_loader(config, one_test_name)
    return test_data_loaders


def choose_metric(config):
    metric_scoring = config['metric_scoring']
    if metric_scoring not in ['eer', 'auc', 'acc', 'ap']:
        raise NotImplementedError('metric {} is not implemented'.format(metric_scoring))
    return metric_scoring


def test_one_dataset(model, data_loader):
    prediction_lists = []
    label_lists = []
    for i, data_dict in tqdm(enumerate(data_loader), total=len(data_loader)):
        # get data
        data, label, mask, landmark = \
        data_dict['image'], data_dict['label'], data_dict['mask'], data_dict['landmark']
        label = torch.where(data_dict['label'] != 0, 1, 0)
        # move data to GPU
        data_dict['image'], data_dict['label'] = data.to(device), label.to(device)
        if mask is not None:
            data_dict['mask'] = mask.to(device)
        if landmark is not None:
            data_dict['landmark'] = landmark.to(device)

        # model forward without considering gradient computation
        predictions = inference(model, data_dict)
        label_lists += list(data_dict['label'].cpu().detach().numpy())
        prediction_lists += list(predictions['prob'].cpu().detach().numpy())
    
    return np.array(prediction_lists), np.array(label_lists)
    
def test_epoch(model, test_data_loaders):
    # set model to eval mode
    model.eval()

    # define test recorder
    metrics_all_datasets = {}

    # testing for all test data
    keys = test_data_loaders.keys()
    for key in keys:
        data_dict = test_data_loaders[key].dataset.data_dict
        # compute loss for each dataset
        predictions_nps, label_nps = test_one_dataset(model, test_data_loaders[key])
        
        # compute metric for each dataset
        metric_inputs = [
            {'image_path': image_path, 'video_id': video_id}
            for image_path, video_id in zip(data_dict['image'], data_dict.get('video_id', data_dict['image']))
        ]
        metric_one_dataset = get_test_metrics(y_pred=predictions_nps, y_true=label_nps,
                                              img_names=metric_inputs)
        metrics_all_datasets[key] = metric_one_dataset
        
        # info for each dataset
        tqdm.write(f"dataset: {key}")
        for k, v in metric_one_dataset.items():
            if k not in {'pred', 'label'}:
                tqdm.write(f"{k}: {v}")

    return metrics_all_datasets

@torch.no_grad()
def inference(model, data_dict):
    predictions = model(data_dict, inference=True)
    return predictions


def main():
    # parse options and load config
    with open(args.detector_path, 'r') as f:
        config = yaml.safe_load(f)
    with open('./training/config/test_config.yaml', 'r') as f:
        config2 = yaml.safe_load(f)
    config.update(config2)
    if on_2060:
        config['lmdb_dir'] = r'I:\transform_2_lmdb'
        config['train_batchSize'] = 10
        config['workers'] = 0
    weights_path = None
    # If arguments are provided, they will overwrite the yaml settings
    if args.test_dataset:
        config['test_dataset'] = args.test_dataset
    if args.weights_path:
        config['weights_path'] = args.weights_path
        weights_path = args.weights_path
    if args.architecture_only:
        config['backbone_pretrained'] = False
    if args.max_samples_per_dataset is not None:
        config['max_samples_per_dataset'] = args.max_samples_per_dataset
    if args.test_batch_size is not None:
        config['test_batchSize'] = args.test_batch_size
    if args.workers is not None:
        config['workers'] = args.workers
    
    # init seed
    init_seed(config)

    # set cudnn benchmark if needed
    if config['cudnn']:
        cudnn.benchmark = True

    # prepare the testing data loader
    test_data_loaders = prepare_testing_data(config)
    
    # prepare the model (detector)
    model_class = DETECTOR[config['model_name']]
    model = model_class(config)
    model.eval()
    epoch = 0
    if weights_path:
        try:
            epoch = int(weights_path.split('/')[-1].split('.')[0].split('_')[2])
        except:
            epoch = 0
        checkpoint_info = load_checkpoint_strict(
            model,
            weights_path,
            trusted=args.trusted_checkpoint,
        )
        model = model.to(device)
        print('===> Load checkpoint done!')
    else:
        print('Fail to load the pre-trained weights')
    
    # start testing
    best_metric = test_epoch(model, test_data_loaders)
    if args.output_path:
        dataset_hashes = {
            dataset_name: sha256_file(os.path.join(config['dataset_json_folder'], f'{dataset_name}.json'))
            for dataset_name in config['test_dataset']
        }
        atomic_write_json({
            'schema_version': 1,
            'code': {
                'git_commit': subprocess.check_output(
                    ['git', 'rev-parse', 'HEAD'], text=True
                ).strip(),
                'dirty': bool(subprocess.check_output(
                    ['git', 'status', '--porcelain'], text=True
                ).strip()),
            },
            'checkpoint': {
                'sha256': sha256_file(weights_path),
                **checkpoint_info,
            },
            'config': {
                'detector_sha256': sha256_file(args.detector_path),
                'test_sha256': sha256_file('./training/config/test_config.yaml'),
            },
            'dataset_json_sha256': dataset_hashes,
            'evaluation': {
                'datasets': list(config['test_dataset']),
                'max_samples_per_dataset': config.get('max_samples_per_dataset'),
                'batch_size': config['test_batchSize'],
                'protocol_mode': config.get('protocol_mode'),
                'sample_counts': {
                    name: len(loader.dataset)
                    for name, loader in test_data_loaders.items()
                },
                'video_counts': {
                    name: len(set(loader.dataset.data_dict['video_id']))
                    for name, loader in test_data_loaders.items()
                },
                'metrics': summarize_metrics(best_metric),
            },
        }, args.output_path)
        print('===> Wrote evaluation report')
    print('===> Test Done!')

if __name__ == '__main__':
    # CUDA_VISIBLE_DEVICES=0 python3 training/test.py --detector_path ./training/config/detector/dfd_hr.yaml --test_dataset Celeb-DF-v2 FaceShifter DeepFakeDetection DFDC DFDCP test_DFR test_WDF test_FFIW uniface_ff blendface_ff mobileswap_ff e4s_ff facedancer_ff fsgan_ff inswap_ff simswap_ff --weights_path logs/dfd-hr/dfd_hr_2025-10-24-01-19-06/test/avg/ckpt_best.pth
    main()
