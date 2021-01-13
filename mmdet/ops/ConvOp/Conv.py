from mmcv.cnn import ConvModule
from ..build import CUSTOM_CONV_OP
from collections import OrderedDict
import torch.nn as nn

# class BottleNeckDWConv(nn.Module):
#     def __init__(self, conv_ch,
#                  kernel_size=3,
#                  group="default",
#                  exp=1,
#                  norm_cfg=dict(type='BN', requires_grad=True),
#                  act_cfg=dict(type='ReLU')):
#         super(BottleNeckDWConv, self).__init__()
#         self.Identity = nn.Identity()
#         mid_ch = int(conv_ch*exp)
#         if group == "default":
#             groups = mid_ch
#         else:
#             groups = group
#         conv_list = OrderedDict([("Conv_1x1", ConvModule(conv_ch, mid_ch,
#                                                          kernel_size=1,
#                                                          padding=0,
#                                                          norm_cfg=norm_cfg,
#                                                          act_cfg=act_cfg)),
#                                 ("Conv_kxk", ConvModule(mid_ch, mid_ch, kernel_size=kernel_size,
#                                                         padding=kernel_size//2,
#                                                         groups=groups,
#                                                         norm_cfg=norm_cfg,
#                                                         act_cfg=act_cfg)),
#                                  ("Conv_1x1", ConvModule(conv_ch, conv_ch,
#                                                          kernel_size=1,
#                                                          padding=0,
#                                                          norm_cfg=norm_cfg,
#                                                          act_cfg=None))])
#
#         self.convs = nn.Sequential(conv_list)
#
#     def forward(self, x):
#         y = self.convs(x)
#         return y+x
@CUSTOM_CONV_OP.register_module()
class NormalConv(ConvModule):
    def __init__(self, in_ch,
                       out_ch,
                       kernel_size,
                       dialtion=1,
                       norm_cfg=dict(type='BN', requires_grad=True),
                       act_cfg=dict(type='ReLU')):

        super(NormalConv, self).__init__(in_ch,
                                       out_ch,
                                       kernel_size,
                                       padding=kernel_size//2,
                                       dilation=dialtion,
                                       norm_cfg=norm_cfg,
                                       act_cfg=act_cfg)
    def forward(self, x, activate=True, norm=True):
        return super(NormalConv, self).forward(x, activate=True, norm=True)

@CUSTOM_CONV_OP.register_module()
class SepConv(ConvModule):
    def __init__(self, conv_ch,
                       kernel_size,
                       dialtion=1,
                       norm_cfg=dict(type='BN', requires_grad=True),
                       act_cfg=dict(type='ReLU')):

        super(SepConv, self).__init__(conv_ch,
                                      conv_ch,
                                      kernel_size,
                                      padding=kernel_size//2,
                                      groups=conv_ch,
                                      dilation=dialtion,
                                      norm_cfg=norm_cfg,
                                      act_cfg=act_cfg)
    def forward(self, x, activate=True, norm=True):
        return super(SepConv, self).forward(x, activate=True, norm=True)
