import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = PROJECT_ROOT / 'training'
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from structured_metrics import JSONLMetricsWriter, read_complete_jsonl
from trainer.trainer import Trainer


class StructuredMetricsTests(unittest.TestCase):
    def test_writer_appends_independent_json_lines(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'metrics.jsonl'
            writer = JSONLMetricsWriter(path, run_id='run-001', sync=False)

            writer.append({'event': 'train_batch', 'global_step': 1, 'loss': 0.5})
            writer.append({'event': 'validation', 'global_step': 1, 'auc': 0.75})

            lines = path.read_text(encoding='utf-8').splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual([json.loads(line)['event'] for line in lines], [
                'train_batch',
                'validation',
            ])
            self.assertTrue(all(json.loads(line)['run_id'] == 'run-001' for line in lines))

    def test_reader_keeps_complete_events_before_interrupted_tail(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'metrics.jsonl'
            path.write_text('{"event":"first"}\n{"event":', encoding='utf-8')

            self.assertEqual(read_complete_jsonl(path), [{'event': 'first'}])

    def test_writer_normalizes_nonfinite_observations_to_null(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'metrics.jsonl'
            writer = JSONLMetricsWriter(path, sync=False)

            writer.append({'event': 'train_batch', 'metrics': {'auc': math.nan}})

            self.assertIsNone(read_complete_jsonl(path)[0]['metrics']['auc'])

    def test_disabled_writer_does_not_create_rank_local_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'metrics.jsonl'
            writer = JSONLMetricsWriter(path, enabled=False)

            self.assertFalse(writer.append({'event': 'ignored'}))
            self.assertFalse(path.exists())

    def test_trainer_disables_writer_on_nonzero_distributed_rank(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = Trainer.__new__(Trainer)
            trainer.config = {
                'ddp': True,
                'metrics_jsonl': str(Path(temp_dir) / 'metrics.jsonl'),
                'rec_iter': 10,
            }
            trainer.log_dir = temp_dir
            with patch('trainer.trainer.dist.is_available', return_value=True), \
                    patch('trainer.trainer.dist.is_initialized', return_value=True), \
                    patch('trainer.trainer.dist.get_rank', return_value=1):
                trainer._configure_structured_metrics()

            self.assertFalse(trainer.metrics_writer.enabled)
            self.assertFalse(trainer.metrics_writer.path.exists())

    def test_train_event_contains_batch_timing_resource_and_loss_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'metrics.jsonl'
            trainer = Trainer.__new__(Trainer)
            trainer.config = {
                'world_size': 2,
                'train_batchSize': 4,
                'effective_batch_size': 16,
            }
            trainer.device = torch.device('cpu')
            trainer.optimizer = SimpleNamespace(param_groups=[{'lr': 0.001}])
            trainer.gradient_accumulation_steps = 2
            trainer.metrics_interval = 1
            trainer._metrics_total_batches = 3
            trainer.metrics_writer = JSONLMetricsWriter(path, run_id='run-001', sync=False)

            trainer._record_train_batch(
                epoch=2,
                iteration=0,
                global_step=7,
                losses={'overall': torch.tensor(0.25)},
                batch_metrics={'acc': 0.75},
                step_time=0.2,
                data_time=0.1,
            )

            event = read_complete_jsonl(path)[0]
            self.assertEqual(event['event'], 'train_batch')
            self.assertEqual(event['global_step'], 7)
            self.assertEqual(event['loss'], 0.25)
            self.assertEqual(event['learning_rate'], 0.001)
            self.assertEqual(event['effective_batch_size'], 16)
            self.assertIn('disk_free_bytes', event)
            self.assertIn('cuda_memory_allocated_bytes', event)


if __name__ == '__main__':
    unittest.main()
