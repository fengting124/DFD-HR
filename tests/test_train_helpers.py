import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = PROJECT_ROOT / "training"
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

import train
from trainer.trainer import resolve_validation_iterations


class TrainHelpersTests(unittest.TestCase):
    def test_build_epoch_range_uses_exact_epoch_count(self):
        config = {"start_epoch": 0, "nEpochs": 20}

        epochs = list(train.build_epoch_range(config))

        self.assertEqual(epochs, list(range(20)))

    def test_resolve_runtime_device_stays_on_cpu_without_cuda(self):
        device = train.resolve_runtime_device(ddp=False, local_rank=0, cuda_enabled=False)

        self.assertEqual(device, torch.device("cpu"))

    def test_local_rank_uses_torchrun_environment(self):
        self.assertEqual(train.resolve_local_rank(None, {'LOCAL_RANK': '1'}), 1)

    def test_explicit_local_rank_overrides_environment(self):
        self.assertEqual(train.resolve_local_rank(0, {'LOCAL_RANK': '1'}), 0)

    def test_local_rank_rejects_invalid_environment(self):
        with self.assertRaisesRegex(ValueError, 'LOCAL_RANK'):
            train.resolve_local_rank(None, {'LOCAL_RANK': 'invalid'})

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

    def test_formal_training_can_disable_automatic_final_test(self):
        config = {
            'run_final_test_after_training': False,
            'test_dataset': ['Celeb-DF-v2'],
        }

        self.assertFalse(train.should_run_final_test(config))

    def test_final_test_requires_an_explicit_dataset(self):
        self.assertFalse(train.should_run_final_test({'test_dataset': []}))

    def test_validation_schedule_matches_official_epoch_profile(self):
        schedule = {'first_epoch': 1, 'later_epochs': 2}

        self.assertEqual(resolve_validation_iterations(5, 0, schedule), {5})
        self.assertEqual(resolve_validation_iterations(5, 1, schedule), {3, 5})

    def test_validation_schedule_caps_checks_at_batch_count(self):
        self.assertEqual(resolve_validation_iterations(2, 4, 5), {1, 2})

    def test_validation_schedule_rejects_incomplete_mapping(self):
        with self.assertRaisesRegex(ValueError, 'first_epoch'):
            resolve_validation_iterations(5, 0, {'first_epoch': 1})

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

    def test_training_config_prefers_experiment_values_over_base_defaults(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            detector = root / 'detector.yaml'
            base = root / 'base.yaml'
            detector.write_text('workers: 0\nnEpochs: 1\n', encoding='utf-8')
            base.write_text('workers: 8\ndry_run: false\n', encoding='utf-8')

            config = train.load_training_config(detector, base)

            self.assertEqual(config['workers'], 0)
            self.assertEqual(config['nEpochs'], 1)
            self.assertFalse(config['dry_run'])

    def test_deterministic_reproducibility_uses_rank_seed_and_cudnn_flags(self):
        config = {
            'manualSeed': 1024,
            'cuda': False,
            'cudnn': True,
            'reproducibility_mode': 'deterministic',
            'cudnn_benchmark': False,
            'cudnn_deterministic': True,
        }
        with patch.object(train.torch, 'use_deterministic_algorithms') as deterministic:
            seed = train.configure_reproducibility(config, rank=2)

        self.assertEqual(seed, 1026)
        self.assertEqual(config['runtime_seed'], 1026)
        deterministic.assert_called_once_with(True)

    def test_data_loader_generator_is_seeded_by_role(self):
        config = {'runtime_seed': 1024}

        train_generator = train.build_data_loader_generator(config, 'train')
        validation_generator = train.build_data_loader_generator(config, 'val')

        self.assertEqual(train_generator.initial_seed(), 1024)
        self.assertEqual(validation_generator.initial_seed(), 11024)

    def test_deterministic_mode_cannot_disable_deterministic_algorithms(self):
        config = {
            'manualSeed': 1024,
            'cuda': False,
            'cudnn': True,
            'reproducibility_mode': 'deterministic',
            'cudnn_benchmark': False,
            'cudnn_deterministic': True,
            'deterministic_algorithms': False,
        }

        with self.assertRaisesRegex(ValueError, 'deterministic_algorithms=true'):
            train.configure_reproducibility(config)

    def test_fixed_subset_is_exact_balanced_and_deterministic(self):
        dataset = SimpleNamespace(
            image_list=['z1', 'a1', 'z0', 'a0', 'm1', 'm0'],
            label_list=[1, 1, 0, 0, 1, 0],
            data_dict={
                'image': ['z1', 'a1', 'z0', 'a0', 'm1', 'm0'],
                'label': [1, 1, 0, 0, 1, 0],
                'video_id': ['z1', 'a1', 'z0', 'a0', 'm1', 'm0'],
            },
        )

        selected = train.apply_fixed_subset(dataset, 4, 'train')

        self.assertEqual(selected, 4)
        self.assertEqual(dataset.image_list, ['a0', 'a1', 'm0', 'm1'])
        self.assertEqual(dataset.label_list, [0, 1, 0, 1])


if __name__ == "__main__":
    unittest.main()
