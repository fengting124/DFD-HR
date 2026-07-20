import sys
import tempfile
import unittest
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / 'scripts'
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from build_formal_training_config import build_formal_config


class FormalTrainingConfigTests(unittest.TestCase):
    def test_builder_freezes_two_gpu_paper_spec_protocol(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            detector = root / 'detector.yaml'
            base = root / 'base.yaml'
            json_folder = root / 'json'
            clip_folder = root / 'clip'
            json_folder.mkdir()
            clip_folder.mkdir()
            (clip_folder / 'model.safetensors').touch()
            detector.write_text(
                yaml.safe_dump({
                    'model_name': 'dfd_hr',
                    'optimizer': {
                        'type': 'adam',
                        'adam': {'lr': 0.5},
                    },
                }),
                encoding='utf-8',
            )
            base.write_text('{}\n', encoding='utf-8')

            config = build_formal_config(
                detector,
                base,
                json_folder,
                clip_folder,
            )

            self.assertEqual(config['protocol_mode'], 'paper_spec')
            self.assertEqual(
                config['paper_spec_basis']['moe_routing'],
                'paper_equations_13_14',
            )
            self.assertEqual(config['backbone_config']['moe']['num_experts'], 4)
            self.assertEqual(config['backbone_config']['moe']['top_k'], 4)
            self.assertTrue(config['backbone_config']['moe']['noise'])
            self.assertEqual(config['train_dataset'], ['FaceForensics++'])
            self.assertEqual(config['validation_dataset'], ['FaceForensics++'])
            self.assertEqual(config['test_dataset'], [])
            self.assertEqual(config['frame_num'], {'train': 8, 'val': 32, 'test': 32})
            self.assertEqual(config['train_batchSize'], 1)
            self.assertEqual(config['gradient_accumulation_steps'], 8)
            self.assertEqual(config['ddp_timeout_minutes'], 180)
            self.assertEqual(config['nEpochs'], 20)
            self.assertEqual(config['workers'], 4)
            self.assertTrue(config['amp'])
            self.assertEqual(config['optimizer']['adam']['lr'], 0.0001)

    def test_builder_pins_validation_reproducibility_and_initialization(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            detector = root / 'detector.yaml'
            base = root / 'base.yaml'
            json_folder = root / 'json'
            clip_folder = root / 'clip'
            json_folder.mkdir()
            clip_folder.mkdir()
            (clip_folder / 'model.safetensors').touch()
            detector.write_text(
                yaml.safe_dump({
                    'optimizer': {'type': 'adam', 'adam': {'lr': 0.1}},
                }),
                encoding='utf-8',
            )
            base.write_text('{}\n', encoding='utf-8')

            config = build_formal_config(
                detector,
                base,
                json_folder,
                clip_folder,
                workers=2,
            )

            self.assertEqual(
                config['validation_checks_per_epoch'],
                {'first_epoch': 1, 'later_epochs': 2},
            )
            self.assertFalse(config['run_final_test_after_training'])
            self.assertEqual(config['reproducibility_mode'], 'deterministic')
            self.assertFalse(config['cudnn_benchmark'])
            self.assertTrue(config['cudnn_deterministic'])
            self.assertTrue(config['deterministic_algorithms'])
            self.assertEqual(config['cublas_workspace_config'], ':4096:8')
            self.assertEqual(config['manualSeed'], 1024)
            self.assertEqual(config['initialization_mode'], 'pinned_clip_pretrained')
            self.assertTrue(config['backbone_local_files_only'])
            self.assertEqual(config['workers'], 2)

    def test_builder_records_seeded_best_effort_fallback(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            detector = root / 'detector.yaml'
            base = root / 'base.yaml'
            json_folder = root / 'json'
            clip_folder = root / 'clip'
            json_folder.mkdir()
            clip_folder.mkdir()
            (clip_folder / 'model.safetensors').touch()
            detector.write_text(
                yaml.safe_dump({
                    'optimizer': {'type': 'adam', 'adam': {'lr': 0.1}},
                }),
                encoding='utf-8',
            )
            base.write_text('{}\n', encoding='utf-8')

            config = build_formal_config(
                detector,
                base,
                json_folder,
                clip_folder,
                reproducibility_mode='seeded_best_effort',
            )

            self.assertEqual(config['reproducibility_mode'], 'seeded_best_effort')
            self.assertFalse(config['cudnn_benchmark'])
            self.assertTrue(config['cudnn_deterministic'])
            self.assertFalse(config['deterministic_algorithms'])
            self.assertEqual(config['cublas_workspace_config'], ':4096:8')

    def test_builder_accepts_validated_3090_batch_configuration(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            detector = root / 'detector.yaml'
            base = root / 'base.yaml'
            json_folder = root / 'json'
            clip_folder = root / 'clip'
            json_folder.mkdir()
            clip_folder.mkdir()
            (clip_folder / 'model.safetensors').touch()
            detector.write_text(
                yaml.safe_dump({
                    'optimizer': {'type': 'adam', 'adam': {'lr': 0.1}},
                }),
                encoding='utf-8',
            )
            base.write_text('{}\n', encoding='utf-8')

            config = build_formal_config(
                detector,
                base,
                json_folder,
                clip_folder,
                gpu_count=2,
                train_batch_size=8,
                gradient_accumulation_steps=1,
                validation_batch_size=32,
            )

            self.assertEqual(config['train_batchSize'], 8)
            self.assertEqual(config['gradient_accumulation_steps'], 1)
            self.assertEqual(config['test_batchSize'], 32)

    def test_builder_rejects_wrong_effective_batch_size(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            detector = root / 'detector.yaml'
            base = root / 'base.yaml'
            json_folder = root / 'json'
            clip_folder = root / 'clip'
            json_folder.mkdir()
            clip_folder.mkdir()
            (clip_folder / 'model.safetensors').touch()
            detector.write_text(
                yaml.safe_dump({
                    'optimizer': {'type': 'adam', 'adam': {'lr': 0.1}},
                }),
                encoding='utf-8',
            )
            base.write_text('{}\n', encoding='utf-8')

            with self.assertRaisesRegex(ValueError, 'effective batch size 16'):
                build_formal_config(
                    detector,
                    base,
                    json_folder,
                    clip_folder,
                    gpu_count=2,
                    train_batch_size=4,
                    gradient_accumulation_steps=1,
                )


if __name__ == '__main__':
    unittest.main()
