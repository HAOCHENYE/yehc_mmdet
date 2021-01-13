import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.cnn import constant_init, kaiming_init
from ..builder import BACKBONES
import os
from mmdet.ops.CSPOSAModule import CSPOSAModule

class ConvStride2(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, exp=1, norm_cfg=dict(type='BN', requires_grad=True)):
        super(ConvStride2, self).__init__()
        # self.conv1x1 = ConvModule(in_ch, out_ch, kernel_size=1, stride=2, padding=0,
        #                           norm_cfg=dict(type='BN', requires_grad=True))
        self.conv3x3 = ConvModule(in_ch, out_ch, kernel_size=3, stride=2, padding=1,
                                  norm_cfg=norm_cfg)

    def forward(self, x):
        # return self.conv1x1(x)+self.conv3x3(x)
        return self.conv3x3(x)


class CSPOSAStage(nn.Module):
    def __init__(self, in_ch, stage_ch, num_block, kernel_size,
                 conv_type=dict(type="NormalConv",
                                info=dict(norm_cfg=dict(type='BN', requires_grad=True))),
                 conv1x1=True):
        assert isinstance(conv_type, dict), "conv_type must be string"
        super(CSPOSAStage, self).__init__()
        self.Block = nn.Sequential(ConvStride2(in_ch, stage_ch, kernel_size=kernel_size),
                                   CSPOSAModule(stage_ch, num_block, conv_type, kernel_size=kernel_size, conv1x1=conv1x1))
    def forward(self, x):
        return self.Block(x)



@BACKBONES.register_module()
class CSPOSANet(nn.Module):
    def __init__(self,
                 stem_channels,
                 stage_channels,
                 block_per_stage,
                 conv_type=dict(type="NormalConv",
                                info=dict(norm_cfg=dict(type='BN', requires_grad=True))),
                 num_out=5,
                 kernel_size=3,
                 conv1x1=True
                 ):
        super(CSPOSANet, self).__init__()
        if isinstance(kernel_size, int):
            kernel_sizes = [kernel_size for _ in range(len(stage_channels))]
        if isinstance(kernel_size, list):
            assert len(kernel_size) == len(stage_channels), \
            "if kernel_size is list, len(kernel_size) should == len(stage_channels)"
            kernel_sizes = kernel_size
        else:
            raise TypeError("type of kernel size should be int or list")
        assert num_out <= len(stage_channels), 'num output should be less than stage channels!'

        conv_info = conv_type["info"]
        norm_cfg = conv_info["norm_cfg"]

        self.stage_nums = len(stage_channels)
        self.stem = ConvModule(3, stem_channels, kernel_size=3, stride=2, padding=1,
                               norm_cfg=norm_cfg)
        '''defult end_stage is the last stage'''
        self.start_stage = len(stage_channels)-num_out+1

        self.stages = nn.ModuleList()
        self.last_stage = len(stage_channels)
        in_channel = stem_channels
        for num_stages in range(self.stage_nums):
            stage = CSPOSAStage(in_channel, stage_channels[num_stages], block_per_stage[num_stages],
                                kernel_size=kernel_sizes[num_stages], conv_type=conv_type, conv1x1=conv1x1)
            in_channel = stage_channels[num_stages]
            # stage = OrderedDict()
            # for num_layers in range(block_per_stage[num_stages]):
            #     stage.update({'stage_{}_layer{}'.format(num_stages, num_layers):_OSA_stage(in_channel, stage_channels[num_stages],
            #                              concat_channels[num_stages], layer_per_block[num_stages])})
            #     in_channel = concat_channels[num_stages]
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
        tmp = x
        x = self.stem(x)
        for i in range(self.start_stage):
            x = self.stages[i](x)
        out = []
        for i in range(self.start_stage, len(self.stages)):
            out.append(x)
            x = self.stages[i](x)
        out.append(x)
        return out

