import json
import sys
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import Mock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = PROJECT_ROOT / "training"
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from trainer.trainer import Trainer
from train import DistributedEvalSampler


class TinyInferenceModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1, bias=False)
        torch.nn.init.constant_(self.linear.weight, 0.875)

    def forward(self, data_dict, inference=False):
        return {"prob": self.linear(data_dict["x"])}

    def get_losses(self, data_dict, predictions):
        return {'overall': (predictions['prob'].flatten() - data_dict['label'].float()).abs().mean()}


class IndexedDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 5

    def __getitem__(self, index):
        return {
            'x': torch.tensor([float(index)]),
            'label': torch.tensor(index % 2),
        }


def ddp_sync_worker(rank, world_size, rendezvous_path, output_dir):
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{rendezvous_path}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        trainer = Trainer.__new__(Trainer)
        trainer.config = {"ddp": True}
        trainer.logger = Mock()
        trainer.model = DDP(TinyInferenceModel())
        trainer.train = False

        def rank_zero_validation():
            predictions = trainer.inference({"x": torch.ones(1, 1)})
            (Path(output_dir) / "rank-zero-validation.txt").write_text(
                "called", encoding="utf-8"
            )
            return {"avg": {"auc": predictions["prob"].item()}}

        result = trainer._run_rank_zero_synchronized(rank_zero_validation)
        collective = torch.tensor(float(rank + 1))
        dist.all_reduce(collective)
        (Path(output_dir) / f"rank-{rank}.json").write_text(
            json.dumps({"result": result, "collective": collective.item()}),
            encoding="utf-8",
        )
    finally:
        dist.destroy_process_group()


def distributed_evaluation_worker(rank, world_size, rendezvous_path, output_dir):
    dist.init_process_group(
        backend='gloo',
        init_method=f'file://{rendezvous_path}',
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        dataset = IndexedDataset()
        sampler = DistributedEvalSampler(dataset, num_replicas=world_size, rank=rank)
        loader = DataLoader(dataset, batch_size=2, sampler=sampler)
        trainer = Trainer.__new__(Trainer)
        trainer.config = {'ddp': True, 'distributed_validation': True, 'save_feat': False}
        trainer.logger = Mock()
        trainer.model = DDP(TinyInferenceModel())
        trainer.device = torch.device('cpu')

        losses, predictions, labels, features = trainer.test_one_dataset(loader)
        collective = torch.tensor(float(rank + 1))
        dist.all_reduce(collective)
        payload = {
            'indices': list(sampler),
            'collective': collective.item(),
        }
        if rank == 0:
            payload.update({
                'predictions': predictions.flatten().tolist(),
                'labels': labels.tolist(),
                'features': features.tolist(),
                'loss_count': losses['overall'].num,
            })
        (Path(output_dir) / f'eval-rank-{rank}.json').write_text(
            json.dumps(payload), encoding='utf-8'
        )
    finally:
        dist.destroy_process_group()


class DdpValidationSyncTests(unittest.TestCase):
    def test_rank_zero_validation_keeps_two_processes_aligned(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            rendezvous_path = str(Path(temp_dir) / "rendezvous")
            mp.spawn(
                ddp_sync_worker,
                args=(2, rendezvous_path, temp_dir),
                nprocs=2,
                join=True,
            )

            self.assertEqual(
                (Path(temp_dir) / "rank-zero-validation.txt").read_text(
                    encoding="utf-8"
                ),
                "called",
            )
            for rank in range(2):
                payload = json.loads(
                    (Path(temp_dir) / f"rank-{rank}.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(payload["result"], {"avg": {"auc": 0.875}})
                self.assertEqual(payload["collective"], 3.0)

    def test_two_process_evaluation_has_exact_coverage_and_ordered_aggregation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            rendezvous_path = str(Path(temp_dir) / 'eval-rendezvous')
            mp.spawn(
                distributed_evaluation_worker,
                args=(2, rendezvous_path, temp_dir),
                nprocs=2,
                join=True,
            )

            rank_zero = json.loads(
                (Path(temp_dir) / 'eval-rank-0.json').read_text(encoding='utf-8')
            )
            rank_one = json.loads(
                (Path(temp_dir) / 'eval-rank-1.json').read_text(encoding='utf-8')
            )
            self.assertEqual(rank_zero['indices'], [0, 1])
            self.assertEqual(rank_one['indices'], [2, 3, 4])
            self.assertEqual(rank_zero['predictions'], [0.0, 0.875, 1.75, 2.625, 3.5])
            self.assertEqual(rank_zero['labels'], [0, 1, 0, 1, 0])
            self.assertEqual(rank_zero['features'], [])
            self.assertEqual(rank_zero['loss_count'], 3)
            self.assertEqual(rank_zero['collective'], 3.0)
            self.assertEqual(rank_one['collective'], 3.0)


if __name__ == "__main__":
    unittest.main()
