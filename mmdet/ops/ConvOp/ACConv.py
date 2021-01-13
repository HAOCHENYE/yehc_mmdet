import torch.nn as nn
from mmcv.cnn import fuse_conv_bn
from mmcv.cnn import ConvModule
from collections import OrderedDict
import copy
import torch
import torch.nn.functional as F
from ..build import CUSTOM_CONV_OP

@CUSTOM_CONV_OP.register_module()
class ACBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, norm_cfg=dict(type='BN', requires_grad=True)):
        super(ACBlock, self).__init__()
        self.kernel_size = kernel_size
        self.norm_cfg = norm_cfg
        self.conv_kxk = ConvModule(in_ch, out_ch, kernel_size=kernel_size, stride=1, padding=kernel_size//2,
                              norm_cfg=norm_cfg, act_cfg=None)
        self.conv_kx1 = ConvModule(in_ch, out_ch, kernel_size=(kernel_size, 1), stride=1, padding=(kernel_size // 2, 0),
                                   norm_cfg=norm_cfg, act_cfg=None)
        self.conv_1xk = ConvModule(in_ch, out_ch, kernel_size=(1, kernel_size), stride=1, padding=(0, kernel_size // 2),
                                       norm_cfg=norm_cfg, act_cfg=None)

        self.conv = nn.ModuleList([self.conv_kxk, self.conv_kx1, self.conv_1xk])


        #     self.conv = nn.ModuleList([self.conv_kxk, self.conv_kx1, self.conv_1xk])
        # self.conv = ConvModule(in_ch, out_ch, kernel_size=kernel_size, stride=1, padding=kernel_size//2,
        #                        norm_cfg=norm_cfg)
        self.init_weight()

    def init_weight(self):
        def ini(module):
            for name, child in module.named_children():
                if isinstance(child, (nn.modules.batchnorm._BatchNorm, nn.SyncBatchNorm)):
                    nn.init.constant_(child.running_var, 0.3)
                else:
                    if isinstance(child, nn.Module):
                        ini(child)
            return
        ini(self)
    def fuse_conv(self):
        self.conv_kxk = fuse_conv_bn(self.conv_kxk)
        self.conv_kx1 = fuse_conv_bn(self.conv_kx1)
        self.conv_1xk = fuse_conv_bn(self.conv_1xk)
        # self.conv = nn.ModuleList([self.conv_kxk_fuse, self.conv_kx1_fuse, self.conv_1xk_fuse])
        self.conv_kxk.conv.weight[:, :, self.kernel_size // 2:self.kernel_size // 2 + 1, :] += self.conv_1xk.conv.weight
        self.conv_kxk.conv.weight[:, :, :, self.kernel_size // 2:self.kernel_size // 2 + 1] += self.conv_kx1.conv.weight
        self.conv = nn.ModuleList([self.conv_kxk])

    def forward(self, x):
        out = []
        for module in self.conv:
            out.append(module(x))
        res = out[0]
        for i in range(1, len(out)):
            res += out[i]
        res = F.relu(res)
        return res
