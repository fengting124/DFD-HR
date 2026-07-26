#  ------------------------------------------------------------------------------------------
#  Copyright (c) 2024 Baifeng Shi.
#  All rights reserved.
#
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import math
import torch
import torch.nn.functional as F
from einops import rearrange
from .utils import split_chessboard, merge_chessboard


def forward_query_loss(
    model,
    query_token,
    query_attn,
    input,
    scales=None,
    img_sizes=None,
    max_split_size=None,
    resize_output_to_idx=0,
    num_prefix_token=0,
    output_shape='bnc',
    split_forward=False,
):
    """Fuse DFD-HR's global image view with its local chessboard views.

    In the configured path, ``input=[B,3,448,448]``. Scale 0.5 becomes one
    224 crop per image and scale 1.0 becomes four 224 crops per image. Each
    CLIP crop produces ``[N,257,768]`` projected tokens. Query attention fuses
    the four local CLS tokens, then concatenation forms a 1536-wide global/local
    representation without changing the 16 x 16 patch grid.

    This helper is inherited from the multi-scale implementation used by the
    paper and currently assumes exactly the configured global/local branches.
    """

    assert input.dim() == 4, "Input image must be in the shape of BxCxHxW."
    assert input.shape[2] == input.shape[3], "Currently only square images are supported."
    assert output_shape in ['bnc', 'bchw'], "Output shape should be either BxNxC (e.g., ViT) or BxCxHxW (e.g., ConvNet)."
    assert output_shape == 'bnc' or num_prefix_token == 0, "For ConvNet there shouldn't be any prefix token."

    b, c, input_size, _ = input.shape

    # DFD-HR Eq. 1: create one global view and a 2 x 2 local view.
    assert scales is not None or img_sizes is not None, "Please assign either scales or img_sizes."
    img_sizes = img_sizes or [int(input_size * scale) for scale in scales]

    max_split_size = max_split_size or input_size   # the maximum size of each split of image; defaults to input size
    num_splits = [math.ceil(size / max_split_size) for size in img_sizes]   # number of splits per scale
    input_multiscale = []
    for size, num_split in zip(img_sizes, num_splits):
        x = F.interpolate(input.to(torch.float32), size=size, mode='bicubic').to(input.dtype)
        x = split_chessboard(x, num_split=num_split)
        input_multiscale.append(x)

    # Global tokens become the semantic prior for local Token Selection.
    outs_multiscale = []
    feature_global, _ = model(input_multiscale[0], None)
    outs_multiscale.append(feature_global[0])
    # split_chessboard orders crops as (crop, batch), so repeating the complete
    # global batch four times aligns each local crop with its source image.
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

    # Reassemble local patch maps before bringing both scales to one grid.
    outs_multiscale = [merge_chessboard(out, num_split=num_split)
                       for num_split, out in zip(num_splits, outs_multiscale)]

    output_size = outs_multiscale[resize_output_to_idx].shape[-2]
    out = torch.cat([F.interpolate(outs_multiscale[i].to(torch.float32), size=output_size,
                                   mode='area').to(outs_multiscale[i].dtype)
                     for i in range(len(outs_multiscale))], dim=1)
    if output_shape == 'bnc':
        out = rearrange(out, 'b c h w -> b (h w) c')
    if num_prefix_token > 0:
        # DFD-HR Eq. 15: a learned query attends over four local CLS tokens.
        outs_prefix_multiscale[0] = torch.stack(outs_prefix_multiscale[0].split(b, dim=0), dim=0).mean(dim=0)
        split_token_bs = torch.stack(outs_prefix_multiscale[1].split(b, dim=0), dim=0).permute(2, 1, 0, 3).squeeze(0)
        query_token_bs = query_token.repeat(b, 1, 1)
        global_token, _ = query_attn(query_token_bs, split_token_bs, split_token_bs, need_weights=True)
        outs_prefix_multiscale[1] = global_token
        out_prefix_multiscale = torch.cat(outs_prefix_multiscale, dim=-1)
        out = torch.cat([out_prefix_multiscale, out], dim=1)

    return out, loss_spearman
