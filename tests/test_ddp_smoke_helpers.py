import sys
import unittest
from pathlib import Path


TRAINING_DIR = Path(__file__).resolve().parents[1] / 'training'
sys.path.insert(0, str(TRAINING_DIR))

from ddp_smoke import (  # noqa: E402
    balance_strided_rank_labels,
    validate_ddp_contract,
    values_match,
)


class DdpSmokeHelperTests(unittest.TestCase):
    def test_contract_requires_two_ranks_and_twenty_steps(self):
        validate_ddp_contract(world_size=2, steps=20)
        with self.assertRaisesRegex(ValueError, 'two ranks'):
            validate_ddp_contract(world_size=1, steps=20)
        with self.assertRaisesRegex(ValueError, '20 steps'):
            validate_ddp_contract(world_size=2, steps=19)

    def test_rng_value_comparison_checks_every_generator(self):
        import torch

        values = {
            'python': 0.1,
            'numpy': 0.2,
            'torch': torch.tensor([0.3]),
            'cuda': torch.tensor([0.4]),
        }
        self.assertTrue(values_match(values, values))
        changed = dict(values, numpy=0.5)
        self.assertFalse(values_match(values, changed))

    def test_balanced_subset_stays_balanced_after_rank_stride(self):
        from types import SimpleNamespace

        dataset = SimpleNamespace(
            image_list=list('abcdefgh'),
            label_list=[0, 1, 0, 1, 0, 1, 0, 1],
            data_dict={
                'image': list('abcdefgh'),
                'label': [0, 1, 0, 1, 0, 1, 0, 1],
            },
        )
        balance_strided_rank_labels(dataset)

        self.assertEqual(dataset.label_list[0::2], [0, 1, 0, 1])
        self.assertEqual(dataset.label_list[1::2], [0, 1, 0, 1])


if __name__ == '__main__':
    unittest.main()
