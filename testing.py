from models.ops.functions import MSDeformAttnFunction
import torch
cuda0 = torch.device('cuda:0')
value = torch.ones((1, 7230, 8, 32)).to(cuda0)
input_spatial_shapes = torch.tensor([[68, 80],
                                    [34, 40],
                                    [17, 20],
                                    [9, 10]]).to(cuda0)
input_level_start_index = torch.tensor([0, 5440, 6800, 7140]).to(cuda0)
sampling_locations = torch.ones((1, 7230, 8, 4, 4, 2)).to(cuda0)
attention_weights = torch.ones((1, 7230, 8, 4, 4)).to(cuda0)

im2col_step = 64

output = MSDeformAttnFunction.apply(
    value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, im2col_step)

print(output.shape)
