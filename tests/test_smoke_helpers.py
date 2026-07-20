import sys
import tempfile
import unittest
from pathlib import Path


TRAINING_DIR = Path(__file__).resolve().parents[1] / 'training'
sys.path.insert(0, str(TRAINING_DIR))

from smoke import (  # noqa: E402
    apply_initialization_config,
    gradient_category,
    validate_output_boundary,
)


class SmokeHelperTests(unittest.TestCase):
    def test_gradient_categories_cover_required_trainable_modules(self):
        self.assertEqual(gradient_category('adapters_attn.0.experts.0.0.weight'), 'adapter')
        self.assertEqual(gradient_category('token_router.0.fc1.weight'), 'router')
        self.assertEqual(gradient_category('head.weight'), 'head')
        self.assertEqual(gradient_category('query_attn.in_proj_weight'), 'query')
        self.assertEqual(gradient_category('module.head.weight'), 'head')
        self.assertEqual(gradient_category('unclassified.weight'), 'other')

    def test_output_boundary_rejects_protected_roots(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repo = root / 'repo'
            data = root / 'data'
            repo.mkdir()
            data.mkdir()
            with self.assertRaises(ValueError):
                validate_output_boundary(repo / 'runs', repo, data)
            with self.assertRaises(ValueError):
                validate_output_boundary(data / 'runs', repo, data)
            output = root / 'runtime' / 'run'
            self.assertEqual(validate_output_boundary(output, repo, data), output.resolve())

    def test_initialization_requires_exactly_one_source(self):
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            apply_initialization_config({}, None, None)
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            apply_initialization_config({}, 'checkpoint.pth', 'clip')
        config = apply_initialization_config({}, 'checkpoint.pth', None)
        self.assertFalse(config['backbone_pretrained'])

    def test_local_clip_initialization_is_offline_and_requires_safetensors(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            clip_dir = Path(temp_dir) / 'clip'
            clip_dir.mkdir()
            with self.assertRaisesRegex(ValueError, 'model.safetensors'):
                apply_initialization_config({}, None, clip_dir)
            (clip_dir / 'model.safetensors').touch()
            config = apply_initialization_config({}, None, clip_dir)
            self.assertTrue(config['backbone_pretrained'])
            self.assertTrue(config['backbone_local_files_only'])
            self.assertEqual(config['backbone_pretrained_path'], str(clip_dir.resolve()))


if __name__ == '__main__':
    unittest.main()
