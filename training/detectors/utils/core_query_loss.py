#  ------------------------------------------------------------------------------------------
#  Copyright (c) 2024 Baifeng Shi.
#  All rights reserved.
#
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .utils import split_chessboard, merge_chessboard, batched_forward


def forward_query_loss(model, query_token, query_attn, input, scales=None, img_sizes=None, max_split_size=None,
                       resize_output_to_idx=0, num_prefix_token=0, output_shape='bnc', split_forward=False):

    assert input.dim() == 4, "Input image must be in the shape of BxCxHxW."
    assert input.shape[2] == input.shape[3], "Currently only square images are supported."
    assert output_shape in ['bnc', 'bchw'], "Output shape should be either BxNxC (e.g., ViT) or BxCxHxW (e.g., ConvNet)."
    assert output_shape == 'bnc' or num_prefix_token == 0, "For ConvNet there shouldn't be any prefix token."

    b, c, input_size, _ = input.shape

    # image size for each scale
    assert scales is not None or img_sizes is not None, "Please assign either scales or img_sizes."
    img_sizes = img_sizes or [int(input_size * scale) for scale in scales]

    # prepare multiscale inputs
    max_split_size = max_split_size or input_size   # the maximum size of each split of image; defaults to input size
    num_splits = [math.ceil(size / max_split_size) for size in img_sizes]   # number of splits per scale
    input_multiscale = []
    for size, num_split in zip(img_sizes, num_splits):
        x = F.interpolate(input.to(torch.float32), size=size, mode='bicubic').to(input.dtype)
        x = split_chessboard(x, num_split=num_split)
        input_multiscale.append(x)

    # run feedforward on each scale: global branch first, then local branch with the global query token
    outs_multiscale = []
    feature_global, _ = model(input_multiscale[0], None)
    outs_multiscale.append(feature_global[0])
    feature_local, loss_spearman = model(input_multiscale[1], feature_global[1].repeat(4, 1, 1)[:, 0, :])
    outs_multiscale.append(feature_local[0])

    if num_prefix_token > 0:
        outs_prefix_multiscale = [out[:, :num_prefix_token] for out in outs_multiscale]
        outs_multiscale = [out[:, num_prefix_token:] for out in outs_multiscale]
    if output_shape == 'bnc':
        outs_multiscale = [rearrange(out, 'b (h w) c -> b c h w',
                                     h=int(out.shape[1] ** 0.5),
                                     w=int(out.shape[1] ** 0.5))
                           for out in outs_multiscale]

    # merge outputs of different splits for each scale separately
    outs_multiscale = [merge_chessboard(out, num_split=num_split)
                       for num_split, out in zip(num_splits, outs_multiscale)]

    # interpolate outputs from different scales and concat together
    output_size = outs_multiscale[resize_output_to_idx].shape[-2]
    out = torch.cat([F.interpolate(outs_multiscale[i].to(torch.float32), size=output_size,
                                   mode='area').to(outs_multiscale[i].dtype)
                     for i in range(len(outs_multiscale))], dim=1)
    if output_shape == 'bnc':
        out = rearrange(out, 'b c h w -> b (h w) c')
    if num_prefix_token > 0:
        # aggregate prefix tokens: scale-0 averaged across splits; scale-1 fused via cross-attention with query token
        outs_prefix_multiscale[0] = torch.stack(outs_prefix_multiscale[0].split(b, dim=0), dim=0).mean(dim=0)
        split_token_bs = torch.stack(outs_prefix_multiscale[1].split(b, dim=0), dim=0).permute(2, 1, 0, 3).squeeze(0)
        query_token_bs = query_token.repeat(b, 1, 1)
        global_token, attn_weight = query_attn(query_token_bs, split_token_bs, split_token_bs, need_weights=True)
        outs_prefix_multiscale[1] = global_token
        out_prefix_multiscale = torch.cat(outs_prefix_multiscale, dim=-1)
        out = torch.cat([out_prefix_multiscale, out], dim=1)

    return out, loss_spearman