#  ------------------------------------------------------------------------------------------
#  Copyright (c) 2024 Baifeng Shi.
#  All rights reserved.
#
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import torch
from einops import rearrange


def split_chessboard(x, num_split):
    """Split BCHW into tiles ordered as ``(tile, batch)`` on dimension 0."""
    B, C, H, W = x.shape
    assert H % num_split == 0 and W % num_split == 0
    x_split = rearrange(x, 'b c (nh h) (nw w) -> (nh nw b) c h w', nh=num_split, nw=num_split)
    return x_split


def merge_chessboard(x, num_split):
    """Invert :func:`split_chessboard` and restore the full spatial map."""
    B, C, H, W = x.shape
    assert B % (num_split**2) == 0
    x_merge = rearrange(x, '(nh nw b) c h w -> b c (nh h) (nw w)', nh=num_split, nw=num_split)
    
    return x_merge


def batched_forward(model, x, batch_size=-1):
    if batch_size == -1:
        return model(x)
    else:
        x_batched = x.split(batch_size)
        outs = [model(x) for x in x_batched]
        return torch.cat(outs, dim=0)


def sample_gumbel(shape, device, eps=1e-20):
    U = torch.rand(shape).to(device)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_sigmoid_sample(logits, tau=1.0):
    """Differentiable continuation probability used before hard thresholding."""
    g = sample_gumbel(logits.shape, logits.device)
    y = torch.sigmoid((logits + g) / tau)
    return y


def soft_rank(x, tau=1e-2):
    """Approximate ranks with pairwise sigmoids so rank loss can backpropagate."""
    pairwise = (x.unsqueeze(-1) - x.unsqueeze(-2)) / tau
    return torch.sigmoid(pairwise).sum(dim=-1)


def spearman_corr(x, y, tau=1e-2):
    """Return differentiable per-sample rank correlation for ``[B,T]`` pairs."""
    x_rank = soft_rank(x, tau=tau)
    y_rank = soft_rank(y, tau=tau)

    x_rank = (x_rank - x_rank.mean(dim=1, keepdim=True)) / (x_rank.std(dim=1, keepdim=True) + 1e-8)
    y_rank = (y_rank - y_rank.mean(dim=1, keepdim=True)) / (y_rank.std(dim=1, keepdim=True) + 1e-8)

    corr = (x_rank * y_rank).mean(dim=1)
    return corr
