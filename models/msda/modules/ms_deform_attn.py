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

import warnings
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_

from ..functions import MS3DDeformAttnFunction


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError(
            "invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n-1) == 0) and n != 0


class MS3DDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_frames=3, n_levels=4, n_heads=8, n_points=4):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_frames     number of frames
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        n_t_heads = (n_heads * n_points * 2)
        if d_model % n_t_heads != 0:
            raise ValueError(
                'd_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_t_heads))
        _d_per_head = d_model // n_t_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")

        self.im2col_step = 64

        self.d_model = d_model
        self.n_frames = n_frames
        self.n_levels = n_levels
        self.n_heads = n_heads
        # XXX: add n_t_heads for 3D dimension
        self.n_t_heads = int(n_heads * n_points * 2)
        self.n_points = n_points

        # XXX: exppend offsets to (x, y, t)
        self.sampling_offsets = nn.Linear(d_model, self.n_t_heads * n_levels * n_points * 3)
        self.attention_weights = nn.Linear(d_model, self.n_t_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        # (8,) [0, pi/4, 2*pi/4, ..., 7*pi/4]
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        # (8, 2)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        # (8, 4, 4, 2)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1

        # XXX: add time init
        grid_init = grid_init[None].repeat(self.n_points*2, 1, 1, 1, 1)
        time_init = self.n_points * torch.linspace(-1, 1, steps=self.n_points*2)
        time_init = time_init.view(self.n_points*2, 1, 1, 1, 1).repeat(1, self.n_heads, self.n_levels, self.n_points, 1)

        with torch.no_grad():
            init = torch.cat((grid_init, time_init), -1)
            self.sampling_offsets.bias = nn.Parameter(init.view(-1))

        constant_(self.attention_weights.weight.data, 0.)
        # XXX change normalization
        # constant_(self.attention_weights.bias.data, 0.)
        constant_(self.attention_weights.bias.data, 1. / (self.n_levels * self.n_points))
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
                                        or (N, Length_{query}, n_levels, 3), add additional (t) to represent n_frames
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert ((input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() * self.n_frames) == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))

        # XXX: expend offsets to (x, y, t)
        value = value.view(N, Len_in, self.n_t_heads, self.d_model // self.n_t_heads)
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_t_heads, self.n_levels, self.n_points, 3)
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_t_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q,
                                                                  self.n_t_heads, self.n_levels, self.n_points)

        # XXX: add new 3D offsets policy
        if reference_points.shape[-1] == 3:
            device = input_spatial_shapes.device
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1],
                                             input_spatial_shapes[..., 0],
                                             self.n_frames * torch.ones(self.n_levels, device=device)], -1)
 
            reference_points = reference_points[:, :, None, :, None, :]
            # sampling_locations's shape: (N, Len_q, n_heads, n_levels, n_points, 3)
            sampling_locations = reference_points \
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        else:
            raise ValueError(
                'Last dim of reference_points must be 3, but get {} instead.'.format(reference_points.shape[-1]))

        '''
        {Shape summary}
        value: [N, Length_{spatial}, n_t_heads, self.d_model // n_t_heads]
        input_spatial_shapes: [n_levels, 2]
        input_level_start_index: [n_levels, ]
        sampling_locations: [N, Length_{query}, n_t_heads, n_levels, n_points, 3], 3 = (x, y, t)
        attention_weights: [N, Length_{query}, n_t_heads, n_levels, n_points]
        '''
        output = MS3DDeformAttnFunction.apply(
            value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, self.im2col_step, self.n_frames)
        output = self.output_proj(output)
        return output
