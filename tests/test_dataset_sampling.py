import sys
import unittest
from pathlib import Path


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

    def test_validation_mode_falls_back_to_test_split(self):
        split_dict = {
            "real": {
                "test": {
                    "c23": {
                        "vid0": {
                            "label": "FF-real",
                            "frames": ["a.png", "b.png"],
                        }
                    }
                }
            }
        }

        resolved = DeepfakeAbstractBaseDataset._resolve_mode_split(
            split_dict=split_dict["real"],
            mode="val",
            compression="c23",
            dataset_name="FaceForensics++",
            cp=None,
        )

        self.assertIn("vid0", resolved)


if __name__ == "__main__":
    unittest.main()
