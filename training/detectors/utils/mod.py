# author: Jiamu Sun
# email: genisun@tencent.com
# date: 2026-05-28
# description: layer and token router.
import logging
import warnings

import torch
import torch.nn as nn
from typing import Optional, Tuple, Any

from transformers import PreTrainedModel, DynamicCache

warnings.filterwarnings('once',
                        message="Attention mask is not 2D, this is not intended to happen, please check your model.")


class TokenRouter(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.weight_predictor = nn.Linear(embed_dim, 1, bias=False)

    def forward(self, x):
        original_type = x.dtype
        self.weight_predictor.to(torch.float32)
        weights = self.weight_predictor(x.to(self.weight_predictor.weight.dtype)).squeeze(
            -1
        )  # [batch_size, seq_len]
        return weights.to(original_type)

    
class LayerRouter(nn.Module):
    def __init__(self, d_model, hidden_dim=64):
        super(LayerRouter, self).__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1, bias=False)
        
    def forward(self, x_cls):
        """
        x_cls: shape (B, d_model), CLS token feature of each sample in the batch
        
        returns:
          p: shape (B, 1), probability that each sample passes through this block
        """
        x = self.fc1(x_cls)       # (B, hidden_dim)
        x = self.relu(x)
        x = self.fc2(x)           # (B, 1)
        return x