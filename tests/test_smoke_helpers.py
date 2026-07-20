import sys
import tempfile
import unittest
from pathlib import Path


TRAINING_DIR = Path(__file__).resolve().parents[1] / 'training'
sys.path.insert(0, str(TRAINING_DIR))

from smoke import gradient_category, validate_output_boundary  # noqa: E402


class SmokeHelperTests(unittest.TestCase):
    def test_gradient_categories_cover_required_trainable_modules(self):
        self.assertEqual(gradient_category('adapters_attn.0.experts.0.0.weight'), 'adapter')
        self.assertEqual(gradient_category('token_router.0.fc1.weight'), 'router')
        self.assertEqual(gradient_category('head.weight'), 'head')
        self.assertEqual(gradient_category('query_attn.in_proj_weight'), 'query')
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


if __name__ == '__main__':
    unittest.main()
