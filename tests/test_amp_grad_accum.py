import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = PROJECT_ROOT / "training"
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from trainer.trainer import Trainer


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1, bias=False)
        torch.nn.init.constant_(self.linear.weight, 1.0)

    def forward(self, data_dict):
        return {"score": self.linear(data_dict["x"])}

    def get_losses(self, data_dict, predictions):
        loss = torch.nn.functional.mse_loss(predictions["score"], data_dict["y"])
        return {"overall": loss}


class AmpGradAccumTests(unittest.TestCase):
    def make_trainer(self, *, amp=False, accumulation_steps=1, optimizer_type="adam"):
        trainer = Trainer.__new__(Trainer)
        trainer.config = {
            "amp": amp,
            "gradient_accumulation_steps": accumulation_steps,
            "train_batchSize": 1,
            "ddp": False,
            "optimizer": {"type": optimizer_type},
        }
        trainer.logger = Mock()
        trainer.device = torch.device("cpu")
        trainer.model = TinyModel()
        trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
        trainer._configure_optimization_runtime()
        return trainer

    def test_runtime_records_effective_batch_and_disables_amp_on_cpu(self):
        trainer = self.make_trainer(amp=True, accumulation_steps=4)

        self.assertFalse(trainer.amp_enabled)
        self.assertEqual(trainer.gradient_accumulation_steps, 4)
        self.assertEqual(trainer.config["world_size"], 1)
        self.assertEqual(trainer.config["effective_batch_size"], 4)

    def test_accumulation_steps_only_on_window_boundary(self):
        trainer = self.make_trainer(accumulation_steps=2)
        trainer.optimizer.zero_grad(set_to_none=True)
        data = {"x": torch.tensor([[1.0]]), "y": torch.tensor([[0.0]])}
        initial_weight = trainer.model.linear.weight.detach().clone()

        trainer.train_step(data, should_step=False, accumulation_divisor=2)
        self.assertTrue(torch.equal(trainer.model.linear.weight, initial_weight))

        trainer.train_step(data, should_step=True, accumulation_divisor=2)
        self.assertAlmostEqual(trainer.model.linear.weight.item(), 0.8, places=6)

    def test_final_partial_window_uses_its_actual_size(self):
        sizes = [
            Trainer._accumulation_window_size(index, total_batches=5, steps=4)
            for index in range(5)
        ]

        self.assertEqual(sizes, [4, 4, 4, 4, 1])

    def test_non_finite_loss_fails_before_optimizer_step(self):
        trainer = self.make_trainer()
        trainer.optimizer.zero_grad(set_to_none=True)
        data = {"x": torch.tensor([[1.0]]), "y": torch.tensor([[float("inf")]])}

        with self.assertRaisesRegex(FloatingPointError, "Non-finite loss"):
            trainer.train_step(data, should_step=True, accumulation_divisor=1)

    def test_non_finite_gradient_fails_before_optimizer_step(self):
        trainer = self.make_trainer()
        trainer.optimizer.zero_grad(set_to_none=True)
        trainer.model.linear.weight.register_hook(
            lambda gradient: torch.full_like(gradient, float("inf"))
        )
        data = {"x": torch.tensor([[1.0]]), "y": torch.tensor([[0.0]])}

        with self.assertRaisesRegex(FloatingPointError, "Non-finite gradient"):
            trainer.train_step(data, should_step=True, accumulation_divisor=1)

    def test_sam_rejects_unsupported_accumulation(self):
        with self.assertRaisesRegex(ValueError, "SAM"):
            self.make_trainer(accumulation_steps=2, optimizer_type="sam")

    def test_detector_configs_declare_safe_defaults(self):
        for relative_path in (
            "training/config/detector/dfd_hr.yaml",
            "training/config/detector/dfd_hr_paper_aligned.yaml",
        ):
            config = yaml.safe_load(
                (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
            )
            self.assertIs(config["amp"], False)
            self.assertEqual(config["gradient_accumulation_steps"], 1)


if __name__ == "__main__":
    unittest.main()
