import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.cnn import constant_init, kaiming_init
import torch
from ..builder import BACKBONES
import os
from collections import OrderedDict


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


class _OSA_module(nn.Module):
    def __init__(self,
                 in_ch,
                 stage_ch,
                 concat_ch,
                 layer_per_block,
                 identity=False,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super(_OSA_module, self).__init__()

        self.identity = identity
        self.layers = nn.ModuleList()
        in_channel = in_ch
        for i in range(layer_per_block):
            self.layers.append(ConvModule(in_ch, stage_ch, kernel_size=3, padding=1,
                                          norm_cfg=norm_cfg))
            in_ch = stage_ch

        # feature aggregation
        in_channel = in_channel + layer_per_block * stage_ch
        self.concat = ConvModule(in_channel, concat_ch, kernel_size=3, padding=1,
                                 norm_cfg=dict(type='BN', requires_grad=True))

    def forward(self, x):
        identity_feat = x
        output = []
        output.append(x)
        for layer in self.layers:
            x = layer(x)
            output.append(x)

        x = torch.cat(output, dim=1)
        xt = self.concat(x)

        if self.identity:
            xt = xt + identity_feat
        return xt

class _OSA_stage(nn.Module):
    def __init__(self,
                 in_ch,
                 stage_ch,
                 concat_ch,
                 block_per_stage,
                 layer_per_block,
                 stage_num,
                 kernel_size=3,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super(_OSA_stage, self).__init__()

        self.ConvS2 = ConvStride2(in_ch, concat_ch)

        self.layers = OrderedDict()

        for i in range(block_per_stage):
            self.layers.update({'stage{}_layer{}'.format(stage_num, i): _OSA_module(concat_ch,
                                stage_ch,
                                concat_ch,
                                layer_per_block,
                                identity=True,
                                norm_cfg=norm_cfg
                                )})

        self.layers = nn.Sequential(self.layers)

    def forward(self, x):
        return self.layers(self.ConvS2(x))

@BACKBONES.register_module()
class YL_Vovnet(nn.Module):
    def __init__(self,
                 stem_channels,
                 stage_channels,
                 concat_channels,
                 block_per_stage,
                 layer_per_block,
                 num_out=5,
                 kernel_size=3,
                 norm_cfg=dict(type='BN', requires_grad=True)
                 ):
        super(YL_Vovnet, self).__init__()

        assert len(stage_channels) == len(concat_channels), \
            'length of stage_channels and concat_channels should be the same!'
        assert num_out < len(stage_channels), 'num output should be less than stage channels!'

        self.stage_nums = len(stage_channels)
        self.stem = ConvModule(3, stem_channels, kernel_size=kernel_size, stride=2, padding=kernel_size//2,
                               norm_cfg=norm_cfg)
        '''defult end_stage is the last stage'''
        self.start_stage = len(stage_channels)-num_out+1

        self.stages = nn.ModuleList()
        self.last_stage = len(stage_channels)
        in_channel = stem_channels
        for num_stages in range(self.stage_nums):
            stage = _OSA_stage(in_channel, stage_channels[num_stages],
                               concat_channels[num_stages], block_per_stage[num_stages],
                               layer_per_block[num_stages], num_stages, norm_cfg=norm_cfg)
            in_channel = concat_channels[num_stages]
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
        x = self.stem(x)
        for i in range(self.start_stage):
            x = self.stages[i](x)
        out = []
        for i in range(self.start_stage, len(self.stages)):
            out.append(x)
            x = self.stages[i](x)
        out.append(x)
        return out

