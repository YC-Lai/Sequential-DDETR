"""
Spatial embbding modules.
"""
from util.misc import NestedTensor
from torch import nn
from typing import List
from .backbone import Backbone

class Spatial_Embbding_backbone(nn.Module):
    def __init__(self, args):
        super().__init__()
        train_backbone = args.lr_backbone > 0
        return_interm_layers = args.masks or (args.num_feature_levels > 1)
        self.backbone = Backbone(args.backbone, train_backbone,
                            return_interm_layers, args.dilation)

    def forward(self, tensor_list: NestedTensor):
        if len(tensor_list.tensors.size()) > 4:
            bs, f, c, h, w = tensor_list.tensors.shape
            tensor_list.tensors = tensor_list.tensors.view(-1, c, h, w)
            tensor_list.mask = tensor_list.mask.view(-1, h, w)
        spatial_feats, _ = self.backbone(tensor_list)
        out: List[NestedTensor] = []
        for name, x in sorted(spatial_feats.items()):
            out.append(x)
        return out


class Spatial_embbding(nn.Module):
    def __init__(self, backbone, num_feature_levels, hidden_dim):
        super().__init__()
        self.num_feature_levels = num_feature_levels
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone

    def forward(self, samples: NestedTensor):
        if len(samples.tensors.size()) > 4:
            bs, f, c, h, w = samples.tensors.shape
            samples.tensors = samples.tensors.view(-1, c, h, w)
            samples.mask = samples.mask.view(-1, h, w)

        pos_feats, _ = self.backbone(samples)

        pos_srcs = []
        for l, feat in enumerate(pos_feats):
            src, _ = feat.decompose()
            pos_srcs.append(self.input_proj[l](src))
        if self.num_feature_levels > len(pos_srcs):
            _len_srcs = len(pos_srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](pos_feats[-1].tensors)
                else:
                    src = self.input_proj[l](pos_srcs[-1])
                pos_srcs.append(src)

        lvl_spatial_embed_flatten = []
        for lvl, src in enumerate(pos_srcs):
            bs, c, h, w = src.shape
            bs = int(bs/f)

            # decouple
            pos_srcs = pos_srcs.view(bs, f, c, h ,w)
            pos_srcs = pos_srcs.flatten(3).transpose(2, 3)

            lvl_spatial_embed_flatten.append(pos_srcs)

        return lvl_spatial_embed_flatten


def build_statial_embbding(args):
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks or (args.num_feature_levels > 1)
    backbone = Backbone(args.backbone, train_backbone,
                        return_interm_layers, args.dilation)
    embbding = Spatial_embbding(backbone, args.num_feature_levels, args.hidden_dim)
    return embbding