import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = PROJECT_ROOT / "training"
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from detectors.dfd_hr_detector import (
    build_moe_adapter_kwargs,
    compute_routed_layer_indices,
    get_clip_visual,
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

    def test_build_moe_adapter_kwargs_reads_paper_spec_overrides(self):
        config = {
            "backbone_config": {
                "moe": {
                    "num_experts": 4,
                    "top_k": 4,
                    "noise": True,
                }
            }
        }

        kwargs = build_moe_adapter_kwargs(config)

        self.assertEqual(kwargs["top_k"], 4)
        self.assertTrue(kwargs["noise"])

    def test_pretrained_clip_uses_explicit_offline_safetensors_source(self):
        model = MagicMock()
        model.named_parameters.return_value = []
        with (
            patch(
                'detectors.dfd_hr_detector.AutoProcessor.from_pretrained',
                return_value='processor',
            ) as processor_loader,
            patch(
                'detectors.dfd_hr_detector.CLIPModel.from_pretrained',
                return_value=model,
            ) as model_loader,
        ):
            processor, loaded_model = get_clip_visual(
                model_name='openai/clip-vit-large-patch14',
                pretrained=True,
                pretrained_path='/local/pinned-clip',
                local_files_only=True,
            )

        self.assertEqual(processor, 'processor')
        self.assertIs(loaded_model, model)
        processor_loader.assert_called_once_with(
            '/local/pinned-clip', local_files_only=True
        )
        model_loader.assert_called_once_with(
            '/local/pinned-clip',
            local_files_only=True,
            use_safetensors=True,
        )

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
