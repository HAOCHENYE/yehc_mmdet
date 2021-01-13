import torch.nn as nn
import torch
from mmcv.cnn import fuse_conv_bn
from mmcv.cnn import ConvModule
from torch.nn import functional as F
class RepVGGBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, norm_cfg=dict(type='BN', requires_grad=True)):
        super(RepVGGBlock, self).__init__()
        self.kernel_size = kernel_size
        self.norm_cfg = norm_cfg
        self.conv_3x3 = ConvModule(in_ch, out_ch, kernel_size=kernel_size, stride=1, padding=kernel_size//2,
                              norm_cfg=norm_cfg, act_cfg=None)

        self.conv_1x1 = ConvModule(in_ch, out_ch, kernel_size=(1, kernel_size), stride=1, padding=(0, kernel_size // 2),
                                       norm_cfg=norm_cfg, act_cfg=None)

        self.conv = nn.ModuleList([self.conv_3x3, self.conv_1x1])


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
        self.conv_3x3 = fuse_conv_bn(self.conv_3x3)
        self.conv_1x1 = fuse_conv_bn(self.conv_1x1)

        # self.conv = nn.ModuleList([self.conv_kxk_fuse, self.conv_kx1_fuse, self.conv_1xk_fuse])
        self.conv_3x3.conv.weight[:, :, self.kernel_size // 2:self.kernel_size // 2 + 1, :] += self.conv_1x1.conv.weight
        self.conv = nn.ModuleList([self.conv_3x3])

    def forward(self, x):
        out = []
        for module in self.conv:
            out.append(module(x))
        res = out[0]
        for i in range(1, len(out)):
            res += out[i]
        res = F.relu(res)
        return res

x = torch.randn(1, 3, 256, 256)
RepVGGUnit = RepVGGBlock(3, 16)
RepVGGUnit.eval()
y_train = RepVGGUnit(x)
RepVGGUnit.fuse_conv()
y_test = RepVGGUnit(x)

print((y_train == y_test).all())
