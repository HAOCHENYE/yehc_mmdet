import torch.nn as nn
from mmcv.cnn import fuse_conv_bn
from mmcv.cnn import ConvModule
import torch.nn.functional as F
from ..build import CUSTOM_CONV_OP

@CUSTOM_CONV_OP.register_module()
class RepVGGBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, group=1, norm_cfg=dict(type='BN', requires_grad=True)):
        super(RepVGGBlock, self).__init__()
        self.kernel_size = kernel_size
        self.norm_cfg = norm_cfg
        self.conv_3x3 = ConvModule(in_ch, out_ch,
                                   kernel_size=kernel_size,
                                   stride=1,
                                   padding=kernel_size//2,
                                   norm_cfg=norm_cfg,
                                   groups=1,
                                   act_cfg=None)

        self.conv_1x1 = ConvModule(in_ch, out_ch, kernel_size=(1, kernel_size), stride=1, padding=(0, kernel_size // 2),
                                       norm_cfg=norm_cfg, act_cfg=None)
        self.stride = stride
        if self.stride == 1:
            self.Identity = nn.Identity()

        self.conv = nn.ModuleList([self.conv_3x3, self.conv_1x1, self.Identity])

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
        self.conv_3x3 = fuse_conv_bn(self.conv_3x3)
        self.conv_1x1 = fuse_conv_bn(self.conv_1x1)

        # self.conv = nn.ModuleList([self.conv_kxk_fuse, self.conv_kx1_fuse, self.conv_1xk_fuse])
        self.conv_3x3.conv.weight[:,
                                  :,
                                  self.kernel_size // 2:self.kernel_size // 2 + 1,
                                  self.kernel_size // 2:self.kernel_size // 2 + 1] += self.conv_1x1.conv.weight
        if self.stride == 1:
            self.conv_3x3.conv.weight[:,
                                      :,
                                      self.kernel_size // 2:self.kernel_size // 2 + 1,
                                      self.kernel_size // 2:self.kernel_size // 2 + 1] += 1

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
