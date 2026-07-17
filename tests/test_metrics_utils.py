import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = PROJECT_ROOT / "training"
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from metrics.utils import get_test_metrics


class MetricsUtilsTests(unittest.TestCase):
    def test_video_metrics_use_dataset_video_ids_when_provided(self):
        y_pred = [0.1, 0.2, 0.8, 0.9]
        y_true = [0, 0, 1, 1]
        img_names = [
            {"image_path": "shallow/a.png", "video_id": "video_real"},
            {"image_path": "shallow/b.png", "video_id": "video_real"},
            {"image_path": "shallow/c.png", "video_id": "video_fake"},
            {"image_path": "shallow/d.png", "video_id": "video_fake"},
        ]

        metrics = get_test_metrics(y_pred=y_pred, y_true=y_true, img_names=img_names)

        self.assertEqual(metrics["video_auc"], 1.0)
        self.assertEqual(metrics["video_acc"], 1.0)


if __name__ == "__main__":
    unittest.main()
