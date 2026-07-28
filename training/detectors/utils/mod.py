import torch
import torch.nn as nn


class TokenRouter(nn.Module):
    """DFD-HR token judge: ``[B,L,D] -> [B,L]`` importance logits.

    The linear score itself is standard. DFD-HR trains its ranking against the
    global-view CLS similarity and uses TopK to decide which tokens enter a
    routed Transformer block.
    """

    def __init__(self, embed_dim):
        super().__init__()
        self.weight_predictor = nn.Linear(embed_dim, 1, bias=False)

    def forward(self, x):
        original_type = x.dtype
        self.weight_predictor.to(torch.float32)
        weights = self.weight_predictor(
            x.to(self.weight_predictor.weight.dtype)
        ).squeeze(-1)
        return weights.to(original_type)


class LayerRouter(nn.Module):
    """DFD-HR layer judge: ``[B,D] -> [B,1]`` continuation logits.

    Training converts these logits to hard forward decisions with a
    straight-through Gumbel-Sigmoid estimator; inference thresholds sigmoid at
    0.5. The router does not return a probability by itself.
    """

    def __init__(self, d_model, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x_cls):
        x = self.fc1(x_cls)  # [B, hidden_dim]
        x = self.relu(x)
        x = self.fc2(x)  # [B, 1]
        return x
