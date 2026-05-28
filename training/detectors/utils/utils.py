#  ------------------------------------------------------------------------------------------
#  Copyright (c) 2024 Baifeng Shi.
#  All rights reserved.
#
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import torch
from einops import rearrange


def split_chessboard(x, num_split):
    """
        x: b * c * h * w
        Deividing x into num_split**2 sub-squares, and concatenate all the sub-squares on the batch dimension
    """
    B, C, H, W = x.shape
    assert H % num_split == 0 and W % num_split == 0
    x_split = rearrange(x, 'b c (nh h) (nw w) -> (nh nw b) c h w', nh=num_split, nw=num_split)
    return x_split


def merge_chessboard(x, num_split):
    """
        x: b * c * h * w
        Assuming x contains num_split**2 sub-squares concatenated along batch dimension, merge the sub-squares back to the original whole square.
        (inverse of split_chessboard)
    """
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
    g = sample_gumbel(logits.shape, logits.device)
    y = torch.sigmoid((logits + g) / tau)
    return y


def spearman_corr(x, y):
    """
    x, y: (B, T)
    Returns: (B,) Spearman correlation coefficient for each sample pair.
    """
    # Ranks starting from 0 (0 is the smallest).
    x_rank = x.argsort(dim=1).argsort(dim=1).float()
    y_rank = y.argsort(dim=1).argsort(dim=1).float()

    x_rank = (x_rank - x_rank.mean(dim=1, keepdim=True)) / (x_rank.std(dim=1, keepdim=True) + 1e-8)
    y_rank = (y_rank - y_rank.mean(dim=1, keepdim=True)) / (y_rank.std(dim=1, keepdim=True) + 1e-8)

    corr = (x_rank * y_rank).mean(dim=1)
    return corr