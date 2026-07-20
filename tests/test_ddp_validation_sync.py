import json
import sys
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import Mock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = PROJECT_ROOT / "training"
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from trainer.trainer import Trainer


def ddp_sync_worker(rank, world_size, rendezvous_path, output_dir):
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{rendezvous_path}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        trainer = Trainer.__new__(Trainer)
        trainer.config = {"ddp": True}
        trainer.logger = Mock()

        def rank_zero_validation():
            (Path(output_dir) / "rank-zero-validation.txt").write_text(
                "called", encoding="utf-8"
            )
            return {"avg": {"auc": 0.875}}

        result = trainer._run_rank_zero_synchronized(rank_zero_validation)
        collective = torch.tensor(float(rank + 1))
        dist.all_reduce(collective)
        (Path(output_dir) / f"rank-{rank}.json").write_text(
            json.dumps({"result": result, "collective": collective.item()}),
            encoding="utf-8",
        )
    finally:
        dist.destroy_process_group()


class DdpValidationSyncTests(unittest.TestCase):
    def test_rank_zero_validation_keeps_two_processes_aligned(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            rendezvous_path = str(Path(temp_dir) / "rendezvous")
            mp.spawn(
                ddp_sync_worker,
                args=(2, rendezvous_path, temp_dir),
                nprocs=2,
                join=True,
            )

            self.assertEqual(
                (Path(temp_dir) / "rank-zero-validation.txt").read_text(
                    encoding="utf-8"
                ),
                "called",
            )
            for rank in range(2):
                payload = json.loads(
                    (Path(temp_dir) / f"rank-{rank}.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(payload["result"], {"avg": {"auc": 0.875}})
                self.assertEqual(payload["collective"], 3.0)


if __name__ == "__main__":
    unittest.main()
