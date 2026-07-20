# author: Jiamu Sun
# email: genisun@tencent.com
# date: 2026-05-28
# description: trainer
import os
import sys
import json
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
project_root_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(project_root_dir)

import pickle
import datetime
import logging
import numpy as np
import random
from contextlib import nullcontext
from copy import deepcopy
from collections import defaultdict
from tqdm import tqdm
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter
from metrics.base_metrics_class import Recorder
from torch.optim.swa_utils import AveragedModel, SWALR
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from metrics.utils import get_test_metrics

FFpp_pool=['FaceForensics++','FF-DF','FF-F2F','FF-FS','FF-NT']#


class Trainer(object):
    def __init__(
        self,
        config,
        model,
        optimizer,
        scheduler,
        logger,
        metric_scoring='auc',
        time_now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'),
        swa_model=None
        ):
        # check if all the necessary components are implemented
        if config is None or model is None or optimizer is None or logger is None:
            raise ValueError("config, model, optimizier, logger, and tensorboard writer must be implemented")

        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.swa_model = swa_model
        self.writers = {}  # dict to maintain different tensorboard writers for each dataset and metric
        self.logger = logger
        self.metric_scoring = metric_scoring
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        # maintain the best metric of all epochs
        self.best_metrics_all_time = defaultdict(
            lambda: defaultdict(lambda: float('-inf')
            if self.metric_scoring != 'eer' else float('inf'))
        )
        self.speed_up()  # move model to GPU
        self._configure_optimization_runtime()

        # get current time
        self.timenow = time_now
        # create directory path
        if 'task_target' not in config:
            self.log_dir = os.path.join(
                self.config['log_dir'],
                self.config['model_name'] + '_' + self.timenow
            )
        else:
            task_str = f"_{config['task_target']}" if config['task_target'] is not None else ""
            self.log_dir = os.path.join(
                self.config['log_dir'],
                self.config['model_name'] + task_str + '_' + self.timenow
            )
        os.makedirs(self.log_dir, exist_ok=True)

    def _configure_optimization_runtime(self):
        accumulation_steps = self.config.get('gradient_accumulation_steps', 1)
        if isinstance(accumulation_steps, bool) or not isinstance(accumulation_steps, int) or accumulation_steps < 1:
            raise ValueError('gradient_accumulation_steps must be a positive integer')

        amp_requested = bool(self.config.get('amp', False))
        optimizer_type = self.config.get('optimizer', {}).get('type')
        if optimizer_type == 'sam' and (amp_requested or accumulation_steps != 1):
            raise ValueError('SAM does not support AMP or gradient accumulation')

        self.gradient_accumulation_steps = accumulation_steps
        self.amp_enabled = amp_requested and self.device.type == 'cuda'
        amp_initial_scale = float(self.config.get('amp_initial_scale', 2**16))
        if not np.isfinite(amp_initial_scale) or amp_initial_scale <= 0:
            raise ValueError('amp_initial_scale must be a positive finite number')
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=self.amp_enabled,
            init_scale=amp_initial_scale,
        )
        world_size = (
            dist.get_world_size()
            if self.config.get('ddp', False) and dist.is_available() and dist.is_initialized()
            else 1
        )
        micro_batch_size = int(self.config['train_batchSize'])
        effective_batch_size = micro_batch_size * world_size * accumulation_steps
        self.config.update({
            'amp_enabled': self.amp_enabled,
            'world_size': world_size,
            'gradient_accumulation_steps': accumulation_steps,
            'effective_batch_size': effective_batch_size,
        })
        self.logger.info(
            'Batch semantics: micro_batch=%s world_size=%s accumulation_steps=%s '
            'effective_batch=%s amp_requested=%s amp_enabled=%s',
            micro_batch_size,
            world_size,
            accumulation_steps,
            effective_batch_size,
            amp_requested,
            self.amp_enabled,
        )

    @staticmethod
    def _accumulation_window_size(iteration, total_batches, steps):
        window_start = (iteration // steps) * steps
        return min(steps, total_batches - window_start)

    def _ensure_finite_gradients(self):
        for parameter in self.model.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                if self.amp_enabled:
                    self.logger.warning(
                        'Non-finite gradient detected; GradScaler will skip the optimizer step'
                    )
                    return False
                raise FloatingPointError('Non-finite gradient detected')
        return True

    def _run_rank_zero_synchronized(self, operation):
        distributed = (
            self.config.get('ddp', False)
            and dist.is_available()
            and dist.is_initialized()
        )
        if not distributed:
            return operation()

        dist.barrier()
        payload = [None]
        if dist.get_rank() == 0:
            try:
                payload[0] = {'ok': True, 'value': operation()}
            except Exception as error:
                payload[0] = {
                    'ok': False,
                    'error': f'{type(error).__name__}: {error}',
                }
        dist.broadcast_object_list(payload, src=0)
        dist.barrier()
        if not payload[0]['ok']:
            raise RuntimeError(f"Rank-0 operation failed: {payload[0]['error']}")
        return payload[0]['value']

    def get_writer(self, phase, dataset_key, metric_key):
        writer_key = f"{phase}-{dataset_key}-{metric_key}"
        if writer_key not in self.writers:
            # update directory path
            writer_path = os.path.join(
                self.log_dir,
                phase,
                dataset_key,
                metric_key,
                "metric_board"
            )
            os.makedirs(writer_path, exist_ok=True)
            # update writers dictionary
            self.writers[writer_key] = SummaryWriter(writer_path)
        return self.writers[writer_key]


    def speed_up(self):
        self.model.to(self.device)
        self.model.device = self.device
        if self.config['ddp'] == True:
            num_gpus = torch.cuda.device_count()
            print(f'avai gpus: {num_gpus}')
            if self.device.type == 'cuda':
                self.model = DDP(
                    self.model,
                    device_ids=[self.config['local_rank']],
                    find_unused_parameters=True,
                    output_device=self.config['local_rank'],
                )
            else:
                self.model = DDP(self.model, find_unused_parameters=True)

    def setTrain(self):
        self.model.train()
        self.train = True

    def setEval(self):
        self.model.eval()
        self.train = False

    def load_ckpt(self, model_path):
        if os.path.isfile(model_path):
            saved = torch.load(model_path, map_location=self.device)
            suffix = model_path.split('.')[-1]
            model_to_load = self.model.module if isinstance(self.model, DDP) else self.model
            if suffix == 'p':
                model_to_load.load_state_dict(saved.state_dict())
            elif isinstance(saved, dict) and 'state_dict' in saved:
                model_to_load.load_state_dict(saved['state_dict'])
            else:
                model_to_load.load_state_dict(saved)
            self.logger.info('Model found in {}'.format(model_path))
        else:
            raise NotImplementedError(
                "=> no model found at '{}'".format(model_path))

    def resume_from_checkpoint(self, checkpoint_path):
        # Keep RNG and metadata tensors on CPU; state_dict loaders move parameter
        # and optimizer tensors to their owning device without corrupting RNG state.
        saved = torch.load(checkpoint_path, map_location='cpu')
        required_keys = {
            'state_dict',
            'optimizer',
            'scheduler',
            'scaler',
            'epoch',
            'best_metrics_all_time',
            'rng_state',
        }
        if not isinstance(saved, dict) or not required_keys.issubset(saved):
            raise ValueError('resume requires a full training checkpoint')
        if saved['epoch'] is None:
            raise ValueError('full training checkpoint is missing a completed epoch')

        model_to_load = self.model.module if isinstance(self.model, DDP) else self.model
        model_to_load.load_state_dict(saved['state_dict'], strict=True)
        if self.optimizer is not None:
            if saved['optimizer'] is None:
                raise ValueError('full training checkpoint is missing optimizer state')
            self.optimizer.load_state_dict(saved['optimizer'])
        if self.scheduler is not None:
            if saved['scheduler'] is None:
                raise ValueError('full training checkpoint is missing scheduler state')
            self.scheduler.load_state_dict(saved['scheduler'])
        self.scaler.load_state_dict(saved['scaler'])

        restored_best = defaultdict(
            lambda: defaultdict(
                lambda: float('-inf') if self.metric_scoring != 'eer' else float('inf')
            )
        )
        for key, value in saved['best_metrics_all_time'].items():
            restored_best[key] = dict(value)
        self.best_metrics_all_time = restored_best

        rng_state = saved['rng_state']
        rng_states_by_rank = saved.get('rng_states_by_rank')
        if (
            rng_states_by_rank is not None
            and self.config.get('ddp', False)
            and dist.is_available()
            and dist.is_initialized()
        ):
            rank = dist.get_rank()
            if rank >= len(rng_states_by_rank):
                raise ValueError('full training checkpoint is missing RNG state for this rank')
            rng_state = rng_states_by_rank[rank]
        required_rng_keys = {'python', 'numpy', 'torch', 'cuda'}
        if not isinstance(rng_state, dict) or not required_rng_keys.issubset(rng_state):
            raise ValueError('full training checkpoint is missing RNG state')
        random.setstate(rng_state['python'])
        np.random.set_state(rng_state['numpy'])
        torch.set_rng_state(rng_state['torch'])
        if torch.cuda.is_available() and rng_state['cuda'] is not None:
            if rng_state.get('cuda_is_local', False):
                torch.cuda.set_rng_state(rng_state['cuda'], device=self.device)
            else:
                torch.cuda.set_rng_state_all(rng_state['cuda'])

        next_epoch = int(saved['epoch']) + 1
        self.config['start_epoch'] = next_epoch
        self.logger.info('Resumed full training checkpoint from %s at epoch %s', checkpoint_path, next_epoch)
        return next_epoch

    def _plain_best_metrics(self):
        return {
            key: dict(value) if isinstance(value, defaultdict) else value
            for key, value in self.best_metrics_all_time.items()
        }

    def _rng_state(self):
        distributed = (
            self.config.get('ddp', False)
            and dist.is_available()
            and dist.is_initialized()
        )
        return {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': (
                torch.cuda.get_rng_state(self.device)
                if torch.cuda.is_available() and distributed
                else torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
            'cuda_is_local': distributed,
        }

    def _checkpoint_state(self, epoch, rng_states_by_rank=None):
        model_to_save = self.model.module if isinstance(self.model, DDP) else self.model
        checkpoint = {
            'state_dict': model_to_save.state_dict(),
            'optimizer': self.optimizer.state_dict() if self.optimizer is not None else None,
            'scheduler': self.scheduler.state_dict() if self.scheduler is not None else None,
            'scaler': self.scaler.state_dict(),
            'epoch': epoch,
            'best_metrics_all_time': self._plain_best_metrics(),
            'config': self.config,
            'rng_state': self._rng_state(),
        }
        if rng_states_by_rank is not None:
            checkpoint['rng_states_by_rank'] = rng_states_by_rank
        if 'svdd' in self.config['model_name']:
            checkpoint['R'] = self.model.R
            checkpoint['c'] = self.model.c
        return checkpoint

    @staticmethod
    def _atomic_torch_save(checkpoint, save_path):
        temporary_path = save_path + '.tmp'
        try:
            torch.save(checkpoint, temporary_path)
            os.replace(temporary_path, save_path)
        finally:
            if os.path.exists(temporary_path):
                os.remove(temporary_path)

    def save_ckpt(self, phase, dataset_key, ckpt_info=None, epoch=None, checkpoint_name='best'):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        if checkpoint_name not in {'best', 'last'}:
            raise ValueError("checkpoint_name must be 'best' or 'last'")
        ckpt_name = f"ckpt_{checkpoint_name}.pth"
        save_path = os.path.join(save_dir, ckpt_name)
        self._atomic_torch_save(self._checkpoint_state(epoch), save_path)
        self.logger.info(f"Checkpoint saved to {save_path}, current ckpt is {ckpt_info}")
        return save_path

    def save_last_ckpt(self, epoch, rng_states_by_rank=None):
        save_dir = os.path.join(self.log_dir, 'checkpoints')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'ckpt_last.pth')
        self._atomic_torch_save(
            self._checkpoint_state(epoch, rng_states_by_rank=rng_states_by_rank),
            save_path,
        )
        self.logger.info('Last checkpoint saved to %s', save_path)
        return save_path

    def save_last_ckpt_synchronized(self, epoch):
        distributed = (
            self.config.get('ddp', False)
            and dist.is_available()
            and dist.is_initialized()
        )
        if not distributed:
            return self.save_last_ckpt(epoch)

        rng_states_by_rank = [None] * dist.get_world_size()
        dist.all_gather_object(rng_states_by_rank, self._rng_state())
        return self._run_rank_zero_synchronized(
            lambda: self.save_last_ckpt(
                epoch,
                rng_states_by_rank=rng_states_by_rank,
            )
        )

    def save_swa_ckpt(self):
        save_dir = self.log_dir
        os.makedirs(save_dir, exist_ok=True)
        ckpt_name = f"swa.pth"
        save_path = os.path.join(save_dir, ckpt_name)
        torch.save(self.swa_model.state_dict(), save_path)
        self.logger.info(f"SWA Checkpoint saved to {save_path}")


    def save_feat(self, phase, fea, dataset_key):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        features = fea
        feat_name = f"feat_best.npy"
        save_path = os.path.join(save_dir, feat_name)
        np.save(save_path, features)
        self.logger.info(f"Feature saved to {save_path}")

    def save_data_dict(self, phase, data_dict, dataset_key):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, f'data_dict_{phase}.pickle')
        with open(file_path, 'wb') as file:
            pickle.dump(data_dict, file)
        self.logger.info(f"data_dict saved to {file_path}")

    def save_metrics(self, phase, metric_one_dataset, dataset_key):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, 'metric_dict_best.pickle')
        with open(file_path, 'wb') as file:
            pickle.dump(metric_one_dataset, file)
        self.logger.info(f"Metrics saved to {file_path}")

    @staticmethod
    def _plain_metric_values(metric_values):
        plain = {}
        for key, value in metric_values.items():
            if key in {'pred', 'label'}:
                continue
            if isinstance(value, dict):
                plain[key] = Trainer._plain_metric_values(value)
            elif isinstance(value, np.generic):
                plain[key] = value.item()
            elif value is None or isinstance(value, (str, int, float, bool)):
                plain[key] = value
            else:
                raise TypeError(f"Metric {key!r} is not JSON serializable: {type(value).__name__}")
        return plain

    def save_metric_report(self, phase, current_metrics):
        save_dir = os.path.join(self.log_dir, phase)
        os.makedirs(save_dir, exist_ok=True)
        report_path = os.path.join(save_dir, 'metrics.json')
        temporary_path = report_path + '.tmp'
        payload = {
            'schema_version': 1,
            'phase': phase,
            'metric_scoring': self.metric_scoring,
            'datasets': {
                key: value for key, value in current_metrics.items() if key != 'avg'
            },
            'average': current_metrics.get('avg', {}),
        }
        with open(temporary_path, 'w', encoding='utf-8') as file:
            json.dump(payload, file, indent=2, sort_keys=True)
            file.write('\n')
        os.replace(temporary_path, report_path)
        self.logger.info(f"Current metrics saved to {report_path}")

    def train_step(
        self,
        data_dict,
        should_step=True,
        accumulation_divisor=1,
        gradient_observer=None,
    ):
        if self.config['optimizer']['type']=='sam':
            if not should_step or accumulation_divisor != 1:
                raise ValueError('SAM does not support gradient accumulation')
            for i in range(2):
                predictions = self.model(data_dict)
                losses = self.model.get_losses(data_dict, predictions)
                if not torch.isfinite(losses['overall']).all():
                    raise FloatingPointError('Non-finite loss detected')
                if i == 0:
                    pred_first = predictions
                    losses_first = losses
                self.optimizer.zero_grad()
                losses['overall'].backward()
                self._ensure_finite_gradients()
                if i == 0:
                    self.optimizer.first_step(zero_grad=True)
                else:
                    self.optimizer.second_step(zero_grad=True)
            return losses_first, pred_first
        else:
            with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                predictions = self.model(data_dict)
                if type(self.model) is DDP:
                    losses = self.model.module.get_losses(data_dict, predictions)
                else:
                    losses = self.model.get_losses(data_dict, predictions)
            if not torch.isfinite(losses['overall']).all():
                raise FloatingPointError('Non-finite loss detected')
            scaled_loss = losses['overall'] / accumulation_divisor
            self.scaler.scale(scaled_loss).backward()
            if should_step:
                self.scaler.unscale_(self.optimizer)
                if gradient_observer is not None:
                    gradient_observer(self.model)
                self._ensure_finite_gradients()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
            return losses,predictions

    def move_data_dict_to_device(self, data_dict):
        for key in data_dict.keys():
            if isinstance(data_dict[key], torch.Tensor):
                data_dict[key] = data_dict[key].to(self.device)
        return data_dict


    def train_epoch(
        self,
        epoch,
        train_data_loader,
        eval_data_loaders=None,
        eval_phase='test',
        ):

        self.logger.info("===> Epoch[{}] start!".format(epoch))
        if epoch>=1:
            times_per_epoch = 2
        else:
            times_per_epoch = 1


        test_step = max(1, len(train_data_loader) // times_per_epoch)
        step_cnt = epoch * len(train_data_loader)
        test_best_metric = None

        # save the training data_dict
        data_dict = train_data_loader.dataset.data_dict
        self.save_data_dict('train', data_dict, ','.join(self.config['train_dataset']))
        # define training recorder
        train_recorder_loss = defaultdict(Recorder)
        train_recorder_metric = defaultdict(Recorder)

        self.optimizer.zero_grad(set_to_none=True)
        total_batches = len(train_data_loader)
        for iteration, data_dict in tqdm(enumerate(train_data_loader),total=total_batches):
            self.setTrain()
            data_dict = self.move_data_dict_to_device(data_dict)

            should_step = (
                (iteration + 1) % self.gradient_accumulation_steps == 0
                or iteration + 1 == total_batches
            )
            accumulation_divisor = self._accumulation_window_size(
                iteration,
                total_batches=total_batches,
                steps=self.gradient_accumulation_steps,
            )
            sync_context = (
                self.model.no_sync()
                if isinstance(self.model, DDP) and not should_step
                else nullcontext()
            )
            with sync_context:
                losses,predictions = self.train_step(
                    data_dict,
                    should_step=should_step,
                    accumulation_divisor=accumulation_divisor,
                )

            # update learning rate

            if 'SWA' in self.config and self.config['SWA'] and epoch>self.config['swa_start']:
                self.swa_model.update_parameters(self.model)

            # compute training metric for each batch data
            if type(self.model) is DDP:
                batch_metrics = self.model.module.get_train_metrics(data_dict, predictions)
            else:
                batch_metrics = self.model.get_train_metrics(data_dict, predictions)

            # store data by recorder
            ## store metric
            for name, value in batch_metrics.items():
                train_recorder_metric[name].update(value)
            ## store loss
            for name, value in losses.items():
                train_recorder_loss[name].update(value)

            # run tensorboard to visualize the training process
            if iteration % 300 == 0 and self.config['local_rank']==0:
                if self.config['SWA'] and self.scheduler is not None and (epoch>self.config['swa_start'] or self.config['dry_run']):
                    self.scheduler.step()
                # info for loss
                loss_str = f"Iter: {step_cnt}    "
                for k, v in train_recorder_loss.items():
                    v_avg = v.average()
                    if v_avg == None:
                        loss_str += f"training-loss, {k}: not calculated"
                        continue
                    loss_str += f"training-loss, {k}: {v_avg}    "
                    # tensorboard-1. loss
                    writer = self.get_writer('train', ','.join(self.config['train_dataset']), k)
                    writer.add_scalar(f'train_loss/{k}', v_avg, global_step=step_cnt)
                self.logger.info(loss_str)
                # info for metric
                metric_str = f"Iter: {step_cnt}    "
                for k, v in train_recorder_metric.items():
                    v_avg = v.average()
                    if v_avg == None:
                        metric_str += f"training-metric, {k}: not calculated    "
                        continue
                    metric_str += f"training-metric, {k}: {v_avg}    "
                    # tensorboard-2. metric
                    writer = self.get_writer('train', ','.join(self.config['train_dataset']), k)
                    writer.add_scalar(f'train_metric/{k}', v_avg, global_step=step_cnt)
                self.logger.info(metric_str)


                # clear recorder.
                # Note we only consider the current 300 samples for computing batch-level loss/metric
                for name, recorder in train_recorder_loss.items():  # clear loss recorder
                    recorder.clear()
                for name, recorder in train_recorder_metric.items():  # clear metric recorder
                    recorder.clear()

            # run test
            if (step_cnt+1) % test_step == 0:
                if eval_data_loaders is not None:
                    def run_validation():
                        self.logger.info(f"===> {eval_phase.capitalize()} start!")
                        return self.test_epoch(
                            epoch,
                            iteration,
                            eval_data_loaders,
                            step_cnt,
                            phase=eval_phase,
                        )

                    test_best_metric = self._run_rank_zero_synchronized(run_validation)
                else:
                    test_best_metric = None

            step_cnt += 1
        return test_best_metric

    def get_respect_acc(self,prob,label):
        pred = np.where(prob > 0.5, 1, 0)
        judge = (pred == label)
        zero_num = len(label) - np.count_nonzero(label)
        acc_fake = np.count_nonzero(judge[zero_num:]) / len(judge[zero_num:])
        acc_real = np.count_nonzero(judge[:zero_num]) / len(judge[:zero_num])
        return acc_real,acc_fake

    def test_one_dataset(self, data_loader):
        # define test recorder
        test_recorder_loss = defaultdict(Recorder)
        prediction_lists = []
        feature_lists=[]
        label_lists = []
        for i, data_dict in tqdm(enumerate(data_loader),total=len(data_loader)):
            # get data
            if 'label_spe' in data_dict:
                data_dict.pop('label_spe')  # remove the specific label
            data_dict['label'] = torch.where(data_dict['label']!=0, 1, 0)  # fix the label to 0 and 1 only
            data_dict = self.move_data_dict_to_device(data_dict)
            # model forward without considering gradient computation
            predictions = self.inference(data_dict)
            label_lists += list(data_dict['label'].cpu().detach().numpy())
            prediction_lists += list(predictions['prob'].cpu().detach().numpy())
            feature_lists += list(predictions['feat'].cpu().detach().numpy())
            if type(self.model) is not AveragedModel:
                # compute all losses for each batch data
                if type(self.model) is DDP:
                    losses = self.model.module.get_losses(data_dict, predictions)
                else:
                    losses = self.model.get_losses(data_dict, predictions)

                # store data by recorder
                for name, value in losses.items():
                    test_recorder_loss[name].update(value)

        return test_recorder_loss, np.array(prediction_lists), np.array(label_lists),np.array(feature_lists)

    def save_best(self, epoch, iteration, step, losses_one_dataset_recorder, key, metric_one_dataset, phase):
        best_metric = self.best_metrics_all_time[key].get(self.metric_scoring,
                                                          float('-inf') if self.metric_scoring != 'eer' else float(
                                                              'inf'))
        # Check if the current score is an improvement
        improved = (metric_one_dataset[self.metric_scoring] > best_metric) if self.metric_scoring != 'eer' else (
                    metric_one_dataset[self.metric_scoring] < best_metric)
        if improved:
            # Update the best metric
            self.best_metrics_all_time[key][self.metric_scoring] = metric_one_dataset[self.metric_scoring]
            if key == 'avg':
                self.best_metrics_all_time[key]['dataset_dict'] = metric_one_dataset['dataset_dict']
            # Save checkpoint, feature, and metrics if specified in config
            if self.config['save_ckpt'] and key == 'avg':
                self.save_ckpt(phase, key, f"{epoch}+{iteration}", epoch=epoch)
            self.save_metrics(phase, metric_one_dataset, key)
        if losses_one_dataset_recorder is not None:
            # info for each dataset
            loss_str = f"dataset: {key}    step: {step}    "
            for k, v in losses_one_dataset_recorder.items():
                writer = self.get_writer(phase, key, k)
                v_avg = v.average()
                if v_avg == None:
                    print(f'{k} is not calculated')
                    continue
                # tensorboard-1. loss
                writer.add_scalar(f'{phase}_losses/{k}', v_avg, global_step=step)
                loss_str += f"{phase}-loss, {k}: {v_avg}    "
            self.logger.info(loss_str)
        # tqdm.write(loss_str)
        metric_str = f"dataset: {key}    step: {step}    "
        for k, v in metric_one_dataset.items():
            if k == 'pred' or k == 'label' or k=='dataset_dict':
                continue
            metric_str += f"{phase}-metric, {k}: {v}    "
            # tensorboard-2. metric
            writer = self.get_writer(phase, key, k)
            writer.add_scalar(f'{phase}_metrics/{k}', v, global_step=step)
        if 'pred' in metric_one_dataset:
            acc_real, acc_fake = self.get_respect_acc(metric_one_dataset['pred'], metric_one_dataset['label'])
            metric_str += f'{phase}-metric, acc_real:{acc_real}; acc_fake:{acc_fake}'
            writer.add_scalar(f'{phase}_metrics/acc_real', acc_real, global_step=step)
            writer.add_scalar(f'{phase}_metrics/acc_fake', acc_fake, global_step=step)
        self.logger.info(metric_str)
    def test_epoch(self, epoch, iteration, test_data_loaders, step, phase='test', save_best=True):
        # set model to eval mode
        self.setEval()

        # define test recorder
        losses_all_datasets = {}
        metrics_all_datasets = {}
        avg_metric = {'acc': 0, 'auc': 0, 'eer': 0, 'ap': 0,'video_auc': 0,'dataset_dict':{}}
        # testing for all test data
        keys = test_data_loaders.keys()
        for key in keys:
            # save the testing data_dict
            data_dict = test_data_loaders[key].dataset.data_dict
            self.save_data_dict(phase, data_dict, key)

            # compute loss for each dataset
            losses_one_dataset_recorder, predictions_nps, label_nps, feature_nps = self.test_one_dataset(test_data_loaders[key])
            # print(f'stack len:{predictions_nps.shape};{label_nps.shape};{len(data_dict["image"])}')
            losses_all_datasets[key] = losses_one_dataset_recorder
            metric_inputs = data_dict.get('metric_input')
            if metric_inputs is None:
                metric_inputs = [
                    {'image_path': image_path, 'video_id': video_id}
                    for image_path, video_id in zip(data_dict['image'], data_dict.get('video_id', data_dict['image']))
                ]
            metric_one_dataset=get_test_metrics(y_pred=predictions_nps,y_true=label_nps,img_names=metric_inputs)
            metrics_all_datasets[key] = self._plain_metric_values(metric_one_dataset)
            for metric_name, value in metric_one_dataset.items():
                if metric_name in avg_metric:
                    avg_metric[metric_name]+=value
            avg_metric['dataset_dict'][key] = metric_one_dataset[self.metric_scoring]
            if type(self.model) is AveragedModel:
                metric_str = f"Iter Final for SWA:    "
                for k, v in metric_one_dataset.items():
                    metric_str += f"testing-metric, {k}: {v}    "
                self.logger.info(metric_str)
                continue
            if save_best:
                self.save_best(epoch, iteration, step, losses_one_dataset_recorder, key, metric_one_dataset, phase)

        if len(keys)>0:
            # calculate avg value
            for key in avg_metric:
                if key != 'dataset_dict':
                    avg_metric[key] /= len(keys)
            metrics_all_datasets['avg'] = self._plain_metric_values(avg_metric)
            if save_best and self.config.get('save_avg',False):
                self.save_best(epoch, iteration, step, None, 'avg', avg_metric, phase)

        self.logger.info(f'===> {phase.capitalize()} Done!')
        if not save_best:
            self.save_metric_report(phase, metrics_all_datasets)
            return metrics_all_datasets
        return self._plain_best_metrics()  # return all types of mean metrics for determining the best ckpt

    @torch.no_grad()
    def inference(self, data_dict):
        model = self.model.module if isinstance(self.model, DDP) else self.model
        predictions = model(data_dict, inference=True)
        return predictions
