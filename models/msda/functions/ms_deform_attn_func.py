# ------------------------------------------------------------------------------------------------
# Sequential DDETR
# Copyright (c) 2021 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from DCN (https://github.com/XinyiYing/D3Dnet)
# Copyright (c) 2018 Microsoft
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable

import MultiScale3DDeformableAttention as MSDA


class MS3DDeformAttnFunction(Function):
    @staticmethod
    def forward(ctx, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step, n_frames):
        ctx.im2col_step = im2col_step
        output = MSDA.ms_deform_attn_forward(
            value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, ctx.im2col_step, n_frames)
        ctx.save_for_backward(value, value_spatial_shapes,
                              value_level_start_index, sampling_locations, attention_weights, torch.tensor(n_frames))

        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, n_frames = ctx.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = \
            MSDA.ms_deform_attn_backward(
                value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, grad_output, ctx.im2col_step, n_frames)
        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None, None


def ms_3d_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights, num_frames):
    """
    :param value                       (N, Length_{query}, n_heads, C)
    :param value_spatial_shapes        (n_levels, 2)
    :param sampling_locations          (N, Length_{query}, n_heads, n_levels, n_points, 3)
    :param attention_weights           (N, Length_{query}, n_heads, n_levels, n_points)
    """
    # for debug and test only,
    # need to use cuda version instead
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    f_Lq_ = Lq_ // num_frames
    value_list = value.split([num_frames * H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    # map sampling_locations [0, 1] to sampling_grids [-1, 1]
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        # N_, num_frames*H_*W_, M_, D_ -> N_, num_frames*H_*W_, M_*D_ -> N_, M_*D_, num_frames*H_*W_ -> N_*M_, D_, num_frames, H_, W_
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_*M_, D_, num_frames, H_, W_)
        # N_, Lq_, M_, P_, 3 -> N_, M_, Lq_, P_, 3 -> N_*M_, Lq_, P_, 3 -> N_*M_, num_frames, f_Lq_, P_, 3
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(
            1, 2).flatten(0, 1).reshape(N_*M_, num_frames, f_Lq_, P_, 3)
        # N_*M_, D_, num_frames, f_Lq_, P_
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
                                          mode='bilinear', padding_mode='zeros', align_corners=False)
        sampling_value_list.append(sampling_value_l_.reshape(N_*M_, D_, Lq_, P_))
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_*M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.transpose(1, 2).reshape(N_*M_, 1, Lq_, L_*P_)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(N_, M_*D_, Lq_)
    return output.transpose(1, 2).contiguous()
