import sys
import random
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = PROJECT_ROOT / "training"
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset


class DatasetSamplingTests(unittest.TestCase):
    def test_frame_sampling_spreads_across_full_video_length(self):
        frame_paths = [f"frame_{idx:03d}.png" for idx in range(10)]

        sampled = DeepfakeAbstractBaseDataset._sample_frame_paths(
            frame_paths=frame_paths,
            frame_num=4,
            video_level=False,
        )

        self.assertEqual(
            sampled,
            [
                "frame_000.png",
                "frame_003.png",
                "frame_006.png",
                "frame_009.png",
            ],
        )

    def test_validation_mode_rejects_missing_validation_split(self):
        split_dict = {
            "test": {
                "c23": {
                    "vid0": {
                        "label": "FF-real",
                        "frames": ["a.png", "b.png"],
                    }
                }
            }
        }

        with self.assertRaisesRegex(ValueError, "explicit 'val' split"):
            DeepfakeAbstractBaseDataset._resolve_mode_split(
                split_dict=split_dict,
                mode="val",
                compression="c23",
                dataset_name="FaceForensics++",
                cp=None,
            )

    def test_validation_mode_uses_explicit_validation_split(self):
        split_dict = {
            "val": {
                "c23": {
                    "vid0": {
                        "label": "FF-real",
                        "frames": ["a.png", "b.png"],
                    }
                }
            }
        }

        resolved = DeepfakeAbstractBaseDataset._resolve_mode_split(
            split_dict=split_dict,
            mode="val",
            compression="c23",
            dataset_name="FaceForensics++",
            cp=None,
        )

        self.assertIn("vid0", resolved)

    def test_seeded_augmentation_restores_outer_random_state(self):
        dataset = SimpleNamespace(
            transform=lambda **kwargs: {
                'image': (kwargs['image'], random.random(), np.random.random()),
            }
        )
        random.seed(77)
        np.random.seed(77)
        expected = (random.random(), np.random.random())
        random.seed(77)
        np.random.seed(77)

        DeepfakeAbstractBaseDataset.data_aug(
            dataset,
            img='frame',
            augmentation_seed=1024,
        )
        actual = (random.random(), np.random.random())

        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
