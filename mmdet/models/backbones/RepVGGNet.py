import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.cnn import constant_init, kaiming_init
from ..builder import BACKBONES
import os
from collections import OrderedDict
from mmdet.ops.build import build_op

CONV_TYPE = build_op("RepVGGBlock")

class RepVGGStage(nn.Module):
    def __init__(self, in_ch, stage_ch, num_block, kernel_size=3, group=1):
        super(RepVGGStage, self).__init__()
        LayerDict = OrderedDict()

        for num in range(num_block):
            if num == 0:
                LayerDict["Block{}".format(num)] = CONV_TYPE(in_ch, stage_ch, group=group, kernel_size=kernel_size, stride=2)
                continue
            LayerDict["Block{}".format(num)] = CONV_TYPE(stage_ch, stage_ch, group=group, kernel_size=kernel_size, stride=1)
        self.Block = nn.Sequential(LayerDict)

    def forward(self, x):
        return self.Block(x)



@BACKBONES.register_module()
class RepVGGNet(nn.Module):
    def __init__(self,
                 stem_channels,
                 stage_channels,
                 block_per_stage,
                 kernel_size=3,
                 num_out=5,
                 norm_cfg=dict(type='BN', requires_grad=True)
                 ):
        super(RepVGGNet, self).__init__()
        if isinstance(kernel_size, int):
            kernel_sizes = [kernel_size for _ in range(len(stage_channels))]
        if isinstance(kernel_size, list):
            assert len(kernel_size) == len(stage_channels), \
            "if kernel_size is list, len(kernel_size) should == len(stage_channels)"
            kernel_sizes = kernel_size

        assert num_out <= len(stage_channels), 'num output should be less than stage channels!'

        self.stage_nums = len(stage_channels)
        self.stem = ConvModule(3, stem_channels, kernel_size=3, stride=2, padding=1,
                               norm_cfg=norm_cfg)
        '''defult end_stage is the last stage'''
        self.start_stage = len(stage_channels)-num_out+1

        self.stages = nn.ModuleList()
        self.last_stage = len(stage_channels)
        in_channel = stem_channels
        for num_stages in range(self.stage_nums):
            stage = RepVGGStage(in_channel, stage_channels[num_stages],
                                            block_per_stage[num_stages],
                                            kernel_size=kernel_sizes[num_stages],
                                            group=1)
            in_channel = stage_channels[num_stages]
            self.stages.append(stage)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            import torch
            assert os.path.isfile(pretrained), "file {} not found.".format(pretrained)
            self.load_state_dict(torch.load(pretrained), strict=False)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')


    def forward(self, x):
        x = self.stem(x)
        for i in range(self.start_stage):
            x = self.stages[i](x)
        out = []
        for i in range(self.start_stage, len(self.stages)):
            out.append(x)
            x = self.stages[i](x)
        out.append(x)
        return out

