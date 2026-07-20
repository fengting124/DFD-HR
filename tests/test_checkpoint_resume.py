import copy
import random
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from unittest.mock import Mock
from unittest.mock import patch

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = PROJECT_ROOT / "training"
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from trainer.trainer import Trainer


class StatefulScaler:
    def __init__(self, scale):
        self.scale = scale

    def state_dict(self):
        return {"scale": self.scale}

    def load_state_dict(self, state_dict):
        self.scale = state_dict["scale"]


class CheckpointResumeTests(unittest.TestCase):
    def make_trainer(self, log_dir):
        trainer = Trainer.__new__(Trainer)
        trainer.config = {"model_name": "tiny", "ddp": False}
        trainer.model = torch.nn.Linear(2, 1)
        trainer.optimizer = torch.optim.SGD(
            trainer.model.parameters(), lr=0.1, momentum=0.9
        )
        trainer.scheduler = torch.optim.lr_scheduler.StepLR(
            trainer.optimizer, step_size=1, gamma=0.5
        )
        trainer.scaler = StatefulScaler(scale=128.0)
        trainer.best_metrics_all_time = defaultdict(dict)
        trainer.best_metrics_all_time["avg"] = {
            "auc": 0.75,
            "dataset_dict": {"dataset-a": 0.75},
        }
        trainer.log_dir = str(log_dir)
        trainer.logger = Mock()
        trainer.device = torch.device("cpu")
        return trainer

    def test_checkpoint_round_trip_restores_complete_training_state(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = self.make_trainer(temp_dir)
            random.seed(11)
            np.random.seed(12)
            torch.manual_seed(13)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(14)

            loss = trainer.model(torch.ones(1, 2)).sum()
            loss.backward()
            trainer.optimizer.step()
            trainer.optimizer.zero_grad(set_to_none=True)
            trainer.scheduler.step()

            expected_model = copy.deepcopy(trainer.model.state_dict())
            expected_optimizer = copy.deepcopy(trainer.optimizer.state_dict())
            expected_scheduler = copy.deepcopy(trainer.scheduler.state_dict())
            expected_best = trainer._plain_best_metrics()

            best_path = trainer.save_ckpt(
                "val", "avg", ckpt_info="3+9", epoch=3, checkpoint_name="best"
            )
            last_path = trainer.save_last_ckpt(epoch=3)
            expected_random = random.random()
            expected_numpy = np.random.random()
            expected_torch = torch.rand(1)
            expected_cuda = (
                torch.rand(1, device="cuda") if torch.cuda.is_available() else None
            )

            with torch.no_grad():
                for parameter in trainer.model.parameters():
                    parameter.add_(10)
            trainer.optimizer.param_groups[0]["lr"] = 9.0
            trainer.scheduler.last_epoch = 99
            trainer.scaler.scale = 1.0
            trainer.best_metrics_all_time.clear()
            random.seed(101)
            np.random.seed(102)
            torch.manual_seed(103)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(104)

            next_epoch = trainer.resume_from_checkpoint(last_path)

            self.assertEqual(next_epoch, 4)
            for key, value in expected_model.items():
                self.assertTrue(torch.equal(trainer.model.state_dict()[key], value))
            restored_optimizer = trainer.optimizer.state_dict()
            self.assertEqual(
                restored_optimizer["param_groups"], expected_optimizer["param_groups"]
            )
            self.assertEqual(
                set(restored_optimizer["state"]), set(expected_optimizer["state"])
            )
            for parameter_id, expected_state in expected_optimizer["state"].items():
                restored_state = restored_optimizer["state"][parameter_id]
                self.assertEqual(set(restored_state), set(expected_state))
                for key, value in expected_state.items():
                    if isinstance(value, torch.Tensor):
                        self.assertTrue(torch.equal(restored_state[key], value))
                    else:
                        self.assertEqual(restored_state[key], value)
            self.assertEqual(trainer.scheduler.state_dict(), expected_scheduler)
            self.assertEqual(trainer.scaler.scale, 128.0)
            self.assertEqual(trainer._plain_best_metrics(), expected_best)
            self.assertEqual(random.random(), expected_random)
            self.assertEqual(np.random.random(), expected_numpy)
            self.assertTrue(torch.equal(torch.rand(1), expected_torch))
            if expected_cuda is not None:
                self.assertTrue(torch.equal(torch.rand(1, device="cuda"), expected_cuda))

            checkpoint_files = sorted(Path(temp_dir).rglob("*.pth"))
            self.assertEqual(checkpoint_files, sorted([Path(best_path), Path(last_path)]))
            self.assertFalse(list(Path(temp_dir).rglob("*.tmp")))

    def test_resume_rejects_weight_only_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = self.make_trainer(temp_dir)
            weight_path = Path(temp_dir) / "weights.pth"
            torch.save(trainer.model.state_dict(), weight_path)

            with self.assertRaisesRegex(ValueError, "full training checkpoint"):
                trainer.resume_from_checkpoint(weight_path)

    def test_resume_loads_checkpoint_on_cpu(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = self.make_trainer(temp_dir)
            checkpoint_path = trainer.save_last_ckpt(epoch=0)
            real_load = torch.load

            with patch('trainer.trainer.torch.load') as mocked_load:
                mocked_load.side_effect = real_load
                trainer.resume_from_checkpoint(checkpoint_path)

            self.assertEqual(mocked_load.call_args.kwargs['map_location'], 'cpu')


if __name__ == "__main__":
    unittest.main()
