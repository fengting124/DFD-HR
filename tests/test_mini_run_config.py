import sys
import tempfile
import unittest
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / 'scripts'
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from build_mini_run_config import build_mini_config


class MiniRunConfigTests(unittest.TestCase):
    def test_builder_creates_one_epoch_balanced_bounded_protocol(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            detector = root / 'detector.yaml'
            base = root / 'base.yaml'
            json_folder = root / 'json'
            json_folder.mkdir()
            detector.write_text(
                yaml.safe_dump({'model_name': 'dfd_hr', 'workers': 9}),
                encoding='utf-8',
            )
            base.write_text(yaml.safe_dump({'dry_run': True}), encoding='utf-8')

            config = build_mini_config(detector, base, json_folder)

            self.assertEqual(config['nEpochs'], 1)
            self.assertEqual(config['train_max_samples'], 16)
            self.assertEqual(config['validation_max_samples'], 8)
            self.assertEqual(config['test_max_samples'], 8)
            self.assertEqual(config['train_batchSize'], 1)
            self.assertEqual(config['gradient_accumulation_steps'], 16)
            self.assertTrue(config['amp'])
            self.assertFalse(config['backbone_pretrained'])
            self.assertEqual(config['initialization_mode'], 'architecture_only_random')
            self.assertFalse(config['dry_run'])
            self.assertFalse(config['save_feat'])
            self.assertEqual(config['dataset_json_folder'], str(json_folder.resolve()))

    def test_builder_supports_pinned_local_clip_initialization(self):
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
                yaml.safe_dump({'model_name': 'dfd_hr'}),
                encoding='utf-8',
            )
            base.write_text('{}\n', encoding='utf-8')

            config = build_mini_config(
                detector,
                base,
                json_folder,
                clip_model_path=clip_folder,
            )

            self.assertTrue(config['backbone_pretrained'])
            self.assertTrue(config['backbone_local_files_only'])
            self.assertEqual(config['initialization_mode'], 'pinned_clip_pretrained')
            self.assertEqual(
                config['backbone_pretrained_path'], str(clip_folder.resolve())
            )


if __name__ == '__main__':
    unittest.main()
