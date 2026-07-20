import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = PROJECT_ROOT / "training"
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from trainer.trainer import Trainer


class TrainerMetricsTests(unittest.TestCase):
    def test_final_test_returns_current_metrics_and_writes_stable_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = Trainer.__new__(Trainer)
            trainer.config = {"save_avg": True}
            trainer.metric_scoring = "auc"
            trainer.best_metrics_all_time = {"stale": {"auc": 0.25}}
            trainer.log_dir = temp_dir
            trainer.logger = Mock()
            trainer.model = object()
            trainer.setEval = Mock()
            trainer.save_data_dict = Mock()
            trainer.test_one_dataset = Mock(
                return_value=(
                    None,
                    np.array([0.1, 0.2, 0.8, 0.9]),
                    np.array([0, 0, 1, 1]),
                    np.empty((4, 0)),
                )
            )
            data_dict = {
                "image": ["a.png", "b.png", "c.png", "d.png"],
                "video_id": ["real", "real", "fake", "fake"],
            }
            data_loaders = {
                "dataset-a": SimpleNamespace(
                    dataset=SimpleNamespace(data_dict=data_dict)
                )
            }

            result = trainer.test_epoch(
                epoch=1,
                iteration=0,
                test_data_loaders=data_loaders,
                step=1,
                phase="test",
                save_best=False,
            )

            self.assertNotIn("stale", result)
            self.assertEqual(result["dataset-a"]["auc"], 1.0)
            self.assertEqual(result["avg"]["auc"], 1.0)
            self.assertNotIn("pred", result["dataset-a"])
            self.assertNotIn("label", result["dataset-a"])

            report_path = Path(temp_dir) / "test" / "metrics.json"
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], 1)
            self.assertEqual(payload["phase"], "test")
            self.assertEqual(payload["metric_scoring"], "auc")
            self.assertEqual(payload["datasets"], {"dataset-a": result["dataset-a"]})
            self.assertEqual(payload["average"], result["avg"])


if __name__ == "__main__":
    unittest.main()
