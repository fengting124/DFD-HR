import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = PROJECT_ROOT / "training"
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

import train


class TrainHelpersTests(unittest.TestCase):
    def test_build_epoch_range_uses_exact_epoch_count(self):
        config = {"start_epoch": 0, "nEpochs": 20}

        epochs = list(train.build_epoch_range(config))

        self.assertEqual(epochs, list(range(20)))

    def test_resolve_runtime_device_stays_on_cpu_without_cuda(self):
        device = train.resolve_runtime_device(ddp=False, local_rank=0, cuda_enabled=False)

        self.assertEqual(device, torch.device("cpu"))

    def test_resolve_eval_loader_names_prefers_validation_over_test(self):
        config = {
            "validation_dataset": ["Celeb-DF-v2"],
            "test_dataset": ["DFDC"],
        }

        names = train.resolve_eval_loader_names(config)

        self.assertEqual(names, ["Celeb-DF-v2"])

    def test_resolve_eval_loader_names_rejects_missing_validation_dataset(self):
        with self.assertRaisesRegex(ValueError, "validation_dataset"):
            train.resolve_eval_loader_names({"validation_dataset": []})

    def test_parse_args_accepts_validation_dataset_override(self):
        with patch.object(
            sys,
            "argv",
            ["train.py", "--validation_dataset", "FaceForensics++"],
        ):
            args = train.parse_args()

        self.assertEqual(args.validation_dataset, ["FaceForensics++"])

    def test_parse_args_accepts_resume_checkpoint(self):
        with patch.object(sys, "argv", ["train.py", "--resume", "last.pth"]):
            args = train.parse_args()

        self.assertEqual(args.resume, "last.pth")


if __name__ == "__main__":
    unittest.main()
