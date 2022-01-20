# ------------------------------------------------------------------------------------------------
# Sequential DDETR
# Copyright (c) 2022 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import sys

import warnings
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_

from ..functions import MSDeformAttnFunction


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError(
            "invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n-1) == 0) and n != 0


class MSDeformAttn(nn.Module):
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
        if d_model % n_heads != 0:
            raise ValueError(
                'd_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")

        self.im2col_step = 64

        self.d_model = d_model
        self.n_frames = n_frames
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        # TODO: exppend offsets to (x, y, t)
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.time_offsets = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        constant_(self.time_offsets.weight.data, 0.)
        # (8,) [0, pi/4, 2*pi/4, ..., 7*pi/4]
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        # (8, 2)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        # (8, 4, 4, 2)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1

        # # XXX: add time init
        time_init = torch.linspace(0, 1, steps=self.n_points, dtype=torch.float32)
        time_init = time_init[None, None, :].repeat(self.n_heads, self.n_levels, 1)

        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
            self.time_offsets.bias = nn.Parameter(time_init.view(-1))

        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    @staticmethod
    def set_frame_index(sampling_locations, num_frames):
        '''
        sampling_locations: [N, Length_{query}, n_heads, n_levels, n_points, 3], 3 = (x, y, t)
        '''
        sampling_xy = sampling_locations[:, :, :, :, :, 0:2]
        sampling_t = torch.round(sampling_locations[:, :, :, :, :, -1].unsqueeze(-1) * (num_frames - 1))
        new_locations = torch.cat((sampling_xy, sampling_t), -1)
        return new_locations

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

        * Length_{query} = n_frames * h_1 * w_1 + n_frames * h_2 * w_2 + ... + n_frames * h_l * w_l
        """
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert ((input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() * self.n_frames) == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads,
                           self.d_model // self.n_heads)
        # TODO: expend offsets to (x, y, t)
        sampling_offsets = self.sampling_offsets(query).view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)

        # XXX: time offsets
        time_offsets = self.time_offsets(query).view(
            N, Len_q, self.n_heads, self.n_levels * self.n_points)
        time_offsets = F.softmax(
            time_offsets, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 1)

        attention_weights = self.attention_weights(query).view(
            N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(
            attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]

        # TODO: add new 3D offsets policy
        elif reference_points.shape[-1] == 3:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1],
                                             input_spatial_shapes[..., 0],
                                             torch.ones(self.n_levels, device=input_spatial_shapes.device)], -1)

            reference_points = reference_points[:, :, None, :, None, :]
            # FIXME: concat query's t coordinate to sampling_offsets
            sampling_offsets = torch.cat((sampling_offsets, time_offsets), -1)
            # sampling_locations's shape: (N, Len_q, n_heads, n_levels, n_points, 3)
            sampling_locations = reference_points \
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            sampling_locations = self.set_frame_index(sampling_locations, self.n_frames)
            # print(sampling_locations[0, 300, 1, 0, 0])

        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2, 3 or 4, but get {} instead.'.format(reference_points.shape[-1]))

        '''
        {Shape summary}
        value: [N, Length_{query}, n_heads, self.d_model // self.n_heads]
        input_spatial_shapes: [n_levels, 2]
        input_level_start_index: [n_levels, ]
        sampling_locations: [N, Length_{query}, n_heads, n_levels, n_points, 3], 3 = (x, y, t)
        attention_weights: [N, Length_{query}, n_heads, n_levels, n_points]
        '''
        output = MSDeformAttnFunction.apply(
            value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, self.im2col_step, self.n_frames)
        output = self.output_proj(output)
        return output
