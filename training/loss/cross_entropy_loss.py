import torch.nn as nn
from .abstract_loss_func import AbstractLossClass
from metrics.registry import LOSSFUNC


@LOSSFUNC.register_module(module_name="cross_entropy")
class CrossEntropyLoss(AbstractLossClass):
    """Two-class softmax cross-entropy used by DFD-HR.

    ``inputs=[B,2]`` contains unnormalized real/fake logits and
    ``targets=[B]`` contains integer class indices (0 real, 1 fake). PyTorch
    applies log-softmax internally, so the model must not softmax logits before
    this loss. The separate ``prob`` output is only for metrics and inference.
    """

    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        return self.loss_fn(inputs, targets)
