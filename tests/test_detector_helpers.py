import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = PROJECT_ROOT / "training"
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from detectors.dfd_hr_detector import (
    build_moe_adapter_kwargs,
    compute_routed_layer_indices,
)
from detectors.utils.moe_adapter import MoEAdapter


class DetectorHelpersTests(unittest.TestCase):
    def test_compute_routed_layer_indices_includes_final_layer(self):
        routed = compute_routed_layer_indices(layer_count=24, remain_layer=20)

        self.assertEqual(routed, [20, 21, 22, 23])

    def test_build_moe_adapter_kwargs_defaults_to_legacy_dense_settings(self):
        config = {"backbone_config": {}}

        kwargs = build_moe_adapter_kwargs(config)

        self.assertEqual(kwargs["num_experts"], 4)
        self.assertEqual(kwargs["top_k"], 4)
        self.assertFalse(kwargs["noise"])

    def test_build_moe_adapter_kwargs_reads_paper_aligned_overrides(self):
        config = {
            "backbone_config": {
                "moe": {
                    "num_experts": 4,
                    "top_k": 2,
                    "noise": True,
                }
            }
        }

        kwargs = build_moe_adapter_kwargs(config)

        self.assertEqual(kwargs["top_k"], 2)
        self.assertTrue(kwargs["noise"])

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required for autocast coverage')
    def test_moe_adapter_supports_cuda_autocast_backward(self):
        adapter = MoEAdapter(D_features=8, num_experts=2, top_k=2).cuda()
        inputs = torch.randn(2, 4, 8, device='cuda', requires_grad=True)

        with torch.cuda.amp.autocast():
            output = adapter(inputs)
            loss = output.square().mean()
        loss.backward()

        self.assertTrue(torch.isfinite(output).all())
        self.assertIsNotNone(inputs.grad)
        self.assertTrue(torch.isfinite(inputs.grad).all())


if __name__ == "__main__":
    unittest.main()
