/*!
**************************************************************************
* Sequential DDETR
* Copyright (c) 2022 SenseTime. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 [see LICENSE for details]
**************************************************************************
* Modified from DCN (https://github.com/XinyiYing/D3Dnet)
* Copyright (c) 2018 Microsoft
**************************************************************************
*/

#pragma once

#include "cpu/ms_deform_attn_cpu.h"

#ifdef WITH_CUDA
#include "cuda/ms_deform_attn_cuda.h"
#endif

at::Tensor ms_deform_attn_forward(const at::Tensor &value, const at::Tensor &spatial_shapes,
                                  const at::Tensor &level_start_index,
                                  const at::Tensor &sampling_loc, const at::Tensor &attn_weight,
                                  const int im2col_step, const int n_frames) {
    if (value.type().is_cuda()) {
#ifdef WITH_CUDA
        return ms_deform_attn_cuda_forward(value, spatial_shapes, level_start_index, sampling_loc,
                                           attn_weight, im2col_step, n_frames);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    AT_ERROR("Not implemented on the CPU");
}

std::vector<at::Tensor> ms_deform_attn_backward(
    const at::Tensor &value, const at::Tensor &spatial_shapes, const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc, const at::Tensor &attn_weight, const at::Tensor &grad_output,
    const int im2col_step, const int n_frames) {
    if (value.type().is_cuda()) {
#ifdef WITH_CUDA
        return ms_deform_attn_cuda_backward(value, spatial_shapes, level_start_index, sampling_loc,
                                            attn_weight, grad_output, im2col_step, n_frames);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    AT_ERROR("Not implemented on the CPU");
}
