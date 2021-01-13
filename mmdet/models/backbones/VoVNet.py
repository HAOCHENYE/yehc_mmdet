import os
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import constant_init, kaiming_init
from collections import OrderedDict
from mmcv.cnn import build_norm_layer

from ..builder import BACKBONES

def common_deconv2d(inplanes,
                  planes,
                  norm_cfg=dict(type='BN'),
                  relu=True):
    cell = OrderedDict()
    cell['deconv'] = nn.ConvTranspose2d(inplanes, planes, kernel_size=2, stride=2, bias=False)
    if norm_cfg:
        norm_name, norm = build_norm_layer(norm_cfg, planes)
        cell[norm_name] = norm
    if relu:
        cell['ReLU'] = nn.ReLU()
    cell = nn.Sequential(cell)
    return cell


def common_conv2d(inplanes,
                  planes,
                  kernel,
                  padding,
                  stride,
                  group=1,
                  norm_cfg=dict(type='BN'),
                  relu=False):
    cell = OrderedDict()
    cell['conv'] = nn.Conv2d(inplanes, planes, kernel_size=kernel,
                             stride=stride, padding=padding, bias=False,groups=group)
    if norm_cfg:
        norm_name, norm = build_norm_layer(norm_cfg, planes)
        cell[norm_name] = norm
    if relu:
        cell['ReLU'] = nn.ReLU()
    cell = nn.Sequential(cell)
    return cell


class BottleNeck(nn.Module):
    """Darknet Basic Block. Which is a 1x1 reduce conv followed by 3x3 conv."""

    def __init__(self,
                 inplanes,
                 planes,
                 kernel,
                 stride,
                 exp,
                 group=1,
                 norm_cfg=dict(type='BN')):
        super(BottleNeck, self).__init__()
        self.stride = stride
        if stride == 1 and inplanes==planes:
            self.body = nn.Sequential(
                common_conv2d(inplanes=inplanes,     planes=int(planes*exp),   kernel=1,      padding=0,
                              stride=1,              norm_cfg=norm_cfg, relu=True),
                common_conv2d(inplanes=int(planes*exp),   planes=int(planes*exp), kernel=kernel, padding=kernel//2,
                              stride=stride,         norm_cfg=norm_cfg, relu=True),
                common_conv2d(inplanes=int(planes*exp), planes=planes,       kernel=1,      padding=0,
                              stride=1,              norm_cfg=norm_cfg, relu=True)
            )
            self.res = None
        else:
            self.body = nn.Sequential(
                common_conv2d(inplanes=inplanes,     planes=int(planes*exp),   kernel=1,      padding=0,
                              stride=1,              norm_cfg=norm_cfg,relu=True),

                common_conv2d(inplanes=int(planes*exp), planes=int(planes*exp), kernel=kernel, padding=kernel//2,
                              stride=stride,         norm_cfg=norm_cfg,   relu=True),

                common_conv2d(inplanes=int(planes*exp), planes=int(planes*exp), kernel=kernel, padding=kernel // 2,
                              stride=1, norm_cfg=norm_cfg,relu=True),
                common_conv2d(inplanes=int(planes*exp), planes=planes,       kernel=1,      padding=0,
                              stride=1,              norm_cfg=norm_cfg, relu=True)
            )

            self.res = common_conv2d(inplanes=inplanes, planes=planes,    kernel=kernel, padding=kernel//2,
                                     stride=stride,     norm_cfg=norm_cfg)

    def forward(self, x):
        if self.res != None:
            return self.res(x)+self.body(x)
        else:
            return x + self.body(x)


class OSAModule(nn.Module):
    def __init__(self, channel, kernel,exp, layer_num):
        super(OSAModule, self).__init__()
        self.layer_num = layer_num
        self.layers = nn.ModuleList()
        for i in range(layer_num):
            self.layers.append(BottleNeck(channel, channel, kernel=kernel, stride=1, exp=exp))

    def forward(self, x):
        res = x
        for i in range(self.layer_num):
            x = self.layers[i](x)
            res = res+x
        return res


class Upsample_fuse(nn.Module):
    def __init__(self,
                 layers_conv,
                 layers_deconv):
        super(Upsample_fuse, self).__init__()
        self.input     = input
        self.convs     = layers_conv
        self.deconv    = layers_deconv
        self.layer_num = len(layers_conv)

    def forward(self, x):
        tmp = []
        for i in range(self.layer_num):
            tmp.append(self.convs[i](x[i]))

        cur = tmp[-1]

        for i in range(self.layer_num-2,-1,-1):
            cur = self.deconv[i](cur)+tmp[i]

        return cur

@BACKBONES.register_module()
class VoVNet(nn.Module):
    def __init__(self,
                 channels=[3, 16, 32, 48, 96, 192, 192, 256],
                 frozen_stages=-1,
                 norm_eval=True):
        super(VoVNet, self).__init__()
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        self.upchannels = 32
        self.stem   = common_conv2d(channels[0], channels[1], kernel=3, padding=1, stride=2, relu=True)
        self.stage1 = nn.Sequential(BottleNeck(channels[1], channels[2], kernel=3, stride=1, exp=1),
                                    BottleNeck(channels[2], channels[2], kernel=3, stride=1, exp=1),
                                    )

        self.stage2 = nn.Sequential(BottleNeck(channels[2], channels[3], kernel=3, stride=2, exp=1),
                                    BottleNeck(channels[3], channels[3], kernel=3, stride=1, exp=1),
                                    )

        self.stage3 = nn.Sequential(BottleNeck(channels[3], channels[4], kernel=3, stride=2, exp=1),
                                    OSAModule(channels[4], kernel=3, exp=0.5, layer_num=3)
                                    )

        self.stage4 = nn.Sequential(BottleNeck(channels[4], channels[5], kernel=3, stride=2, exp=1),
                                    OSAModule(channels[5], kernel=3,  exp=0.5, layer_num=3)
                                    )

        self.stage5 = nn.Sequential(BottleNeck(channels[5], channels[6], kernel=3, stride=2, exp=0.5),
                                    OSAModule(channels[6], kernel=3, exp=0.5, layer_num=3)
                                    )

        self.stage6 = nn.Sequential(BottleNeck(channels[6], channels[7], kernel=3, stride=2, exp=0.5),
                                    OSAModule(channels[7], kernel=3, exp=0.5, layer_num=3)
                                    )

        self.upconv_stage2 = common_conv2d(inplanes=channels[3], planes=self.upchannels, kernel=3, padding=1, stride=1)
        self.upconv_stage3 = common_conv2d(inplanes=channels[4], planes=self.upchannels, kernel=3, padding=1, stride=1)
        self.upconv_stage4 = common_conv2d(inplanes=channels[5], planes=self.upchannels, kernel=3, padding=1, stride=1)
        self.upconv_stage5 = common_conv2d(inplanes=channels[6], planes=self.upchannels, kernel=3, padding=1, stride=1)
        self.upconv_stage6 = common_conv2d(inplanes=channels[7], planes=self.upchannels, kernel=3, padding=1, stride=1)

        self.deconv_stage3 = common_deconv2d(inplanes=self.upchannels, planes=self.upchannels)
        self.deconv_stage4 = common_deconv2d(inplanes=self.upchannels, planes=self.upchannels)
        self.deconv_stage5 = common_deconv2d(inplanes=self.upchannels, planes=self.upchannels)
        self.deconv_stage6 = common_deconv2d(inplanes=self.upchannels, planes=self.upchannels)

        self.Upsample = Upsample_fuse([self.upconv_stage2,
                                       self.upconv_stage3,
                                       self.upconv_stage4,
                                       self.upconv_stage5,
                                       self.upconv_stage6],
                                      [self.deconv_stage6,
                                       self.deconv_stage5,
                                       self.deconv_stage4,
                                       self.deconv_stage3
                                      ])
        # self._freeze_stages()

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

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for param in self.stem.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, 'layer{}'.format(i))
            for param in m.parameters():
                param.requires_grad = False

    def forward(self, x):
        x0 = self.stem(x)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        x5 = self.stage5(x4)
        x6 = self.stage6(x5)

        x_up = [x2, x3, x4, x5, x6]
        outs = self.Upsample(x_up)
        return [outs]


