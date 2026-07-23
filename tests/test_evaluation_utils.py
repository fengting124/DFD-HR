import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


TRAINING_DIR = Path(__file__).resolve().parents[1] / 'training'
sys.path.insert(0, str(TRAINING_DIR))

from evaluation_utils import (  # noqa: E402
    atomic_write_json,
    load_checkpoint_strict,
    normalize_checkpoint_state_dict,
    select_fixed_subset,
    summarize_metrics,
)


class EvaluationUtilsTests(unittest.TestCase):
    def test_normalizes_known_checkpoint_prefixes(self):
        state = normalize_checkpoint_state_dict({
            'state_dict': {
                'module.base_model.weight': torch.ones(1),
                'module.classifier.bias': torch.zeros(1),
            },
        })
        self.assertEqual(set(state), {'backbone.weight', 'head.bias'})

    def test_rejects_normalization_collisions(self):
        with self.assertRaisesRegex(ValueError, 'collision'):
            normalize_checkpoint_state_dict({
                'module.head.weight': torch.ones(1),
                'head.weight': torch.zeros(1),
            })

    def test_strict_load_rejects_missing_keys(self):
        model = torch.nn.Linear(2, 1)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, 'checkpoint.pth')
            torch.save({'module.weight': torch.ones(1, 2)}, path)
            with self.assertRaises(RuntimeError):
                load_checkpoint_strict(model, path)

    def test_full_project_checkpoint_requires_explicit_trust(self):
        model = torch.nn.Linear(2, 1)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, 'checkpoint.pth')
            torch.save({
                'state_dict': model.state_dict(),
                'best_metric': np.float64(0.75),
            }, path)

            with self.assertRaisesRegex(ValueError, 'trusted=True'):
                load_checkpoint_strict(model, path)

            info = load_checkpoint_strict(model, path, trusted=True)
            self.assertEqual(info, {'tensor_count': 2})

    def test_fixed_subset_is_deterministic_and_balanced(self):
        dataset = SimpleNamespace(
            image_list=['z1', 'b0', 'a1', 'a0', 'z0'],
            label_list=[1, 0, 1, 0, 0],
            data_dict={
                'image': ['z1', 'b0', 'a1', 'a0', 'z0'],
                'label': [1, 0, 1, 0, 0],
                'video_id': ['v1', 'v2', 'v3', 'v4', 'v5'],
            },
        )
        self.assertEqual(select_fixed_subset(dataset, 4), 4)
        self.assertEqual(dataset.image_list, ['a0', 'a1', 'b0', 'z1'])
        self.assertEqual(dataset.label_list, [0, 1, 0, 1])
        self.assertEqual(dataset.data_dict['video_id'], ['v4', 'v3', 'v2', 'v1'])

    def test_atomic_json_converts_numpy_scalars(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, 'report.json')
            atomic_write_json({'auc': np.float64(0.75)}, path)
            with open(path, encoding='utf-8') as file:
                self.assertEqual(json.load(file), {'auc': 0.75})

    def test_metric_summary_excludes_per_sample_arrays(self):
        summary = summarize_metrics({
            'external': {
                'auc': np.float64(0.75),
                'pred': np.array([0.2, 0.8]),
                'label': np.array([0, 1]),
            },
        })
        self.assertEqual(summary, {'external': {'auc': 0.75}})


if __name__ == '__main__':
    unittest.main()
