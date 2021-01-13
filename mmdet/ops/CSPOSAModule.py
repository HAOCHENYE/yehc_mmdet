import torch.nn as nn
import torch
from mmdet.ops.build import build_op
from mmcv.cnn import ConvModule

class CSPOSAModule(nn.Module):
    def __init__(self,
                 in_ch,
                 layer_per_block,
                 conv_cfg,
                 kernel_size=3,
                 conv1x1=True
                ):
        super(CSPOSAModule, self).__init__()
        assert isinstance(conv_cfg, dict), "conv_cfg should be dict!"
        conv_type = conv_cfg["type"]
        conv_info = conv_cfg["info"]
        assert isinstance(conv_type, str), "conv_type should be string!"
        assert isinstance(conv_info, dict), "conv_info should be dict!"

        conv = build_op(conv_type)

        assert in_ch % 2 == 0, "in_channel should be slice into two part!"
        self.layers = nn.ModuleList()
        self.mid_ch = in_ch // 2

        concat_ch = in_ch
        stage_ch = in_ch // 2

        if "norm_cfg" in conv_info:
            norm_cfg = conv_info["norm_cfg"]
        else:
            norm_cfg = dict(type='BN', requires_grad=True)

        for i in range(layer_per_block):
            Block = nn.ModuleList()
            if conv1x1:
                Block.append(ConvModule(stage_ch, stage_ch, kernel_size=1, norm_cfg=norm_cfg))
            else:
                Block.append(conv(stage_ch, stage_ch, kernel_size=kernel_size, **conv_info))
            Block.append(conv(stage_ch, stage_ch, kernel_size=kernel_size, **conv_info))
            self.layers.append(Block)



        # feature aggregation
            self.preleft = ConvModule(in_ch, stage_ch, kernel_size=1, norm_cfg=norm_cfg)
            self.preright = ConvModule(in_ch, stage_ch, kernel_size=1, norm_cfg=norm_cfg)
            # feature aggregation
            self.leftconv = ConvModule(stage_ch, stage_ch, kernel_size=1, norm_cfg=norm_cfg)
            self.rightconv = ConvModule(stage_ch, stage_ch, kernel_size=1, norm_cfg=norm_cfg)
            self.concat = ConvModule(concat_ch, concat_ch, kernel_size=1, norm_cfg=norm_cfg)

    def forward(self, x):
        left = self.preleft(x)
        right = self.preright(x)

        left = self.leftconv(left)
        for layer in self.layers:
            feature_identity = right
            for conv in layer:
                right = conv(right)
            right = right + feature_identity

        right = self.rightconv(right)
        y = torch.cat([left, right], dim=1)
        y = self.concat(y)
        return y

