import torch.nn as nn

class AbstractLossClass(nn.Module):
    """Detector loss contract: prediction tensors and labels -> scalar loss."""

    def __init__(self):
        super().__init__()

    def forward(self, pred, label):
        """Return one differentiable scalar for the current batch."""
        raise NotImplementedError('Each subclass should implement the forward method.')
