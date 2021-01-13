import torch
import math
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict

def Conv_3x3(in_channels, out_channels, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


def Conv_1x1(in_channels, out_channels, relu=True):
    modules = [nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
               nn.BatchNorm2d(out_channels)]
    if relu:
        modules.append(nn.ReLU(inplace=True))

    return nn.Sequential(*modules)


def SepConv_3x3(in_channels, out_channels, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),

        nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
        nn.BatchNorm2d(out_channels)
    )


class MBConv_3x3(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand):
        super(MBConv_3x3, self).__init__()
        mid_channels = int(expand * in_channels)

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_channels, mid_channels, 3, stride, 1, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.use_skip_connect = (1 == stride and in_channels == out_channels)

    def forward(self, x):
        if self.use_skip_connect:
            return self.block(x) + x
        else:
            return self.block(x)


class MBConv_5x5(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand):
        super(MBConv_5x5, self).__init__()
        mid_channels = int(expand * in_channels)

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_channels, mid_channels, 5, stride, 2, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.use_skip_connect = (1 == stride and in_channels == out_channels)

    def forward(self, x):
        if self.use_skip_connect:
            return self.block(x) + x
        else:
            return self.block(x)


class BiFPNBlock(nn.Module):
    def __init__(self, num_channels):
        super(BiFPNBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, 3, 1, 1, groups=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU())

    def forward(self, input):
        return self.conv(input)


class Upsample(nn.Module):
    def __init__(self):
        super(Upsample, self).__init__()

    def forward(self, x, y):
        _, _, H, W = y.size()

        return F.interpolate(x, size=[int(H), int(W)], mode='nearest')
        # return F.interpolate(x, size=[H, W], mode='nearest')


class BiFPN(nn.Module):
    def __init__(self, num_channels, epsilon=1e-4):
        super(BiFPN, self).__init__()
        self.epsilon = epsilon
        # Conv layers
        self.conv6_up = BiFPNBlock(num_channels)
        self.conv5_up = BiFPNBlock(num_channels)
        self.conv4_up = BiFPNBlock(num_channels)
        self.conv3_up = BiFPNBlock(num_channels)
        self.conv4_down = BiFPNBlock(num_channels)
        self.conv5_down = BiFPNBlock(num_channels)
        self.conv6_down = BiFPNBlock(num_channels)
        self.conv7_down = BiFPNBlock(num_channels)

        # Feature scaling layers
        self.p6_upsample = Upsample()
        self.p5_upsample = Upsample()
        self.p4_upsample = Upsample()
        self.p3_upsample = Upsample()

        self.p4_downsample = Conv_3x3(num_channels, num_channels, stride=2)
        self.p5_downsample = Conv_3x3(num_channels, num_channels, stride=2)
        self.p6_downsample = Conv_3x3(num_channels, num_channels, stride=2)
        self.p7_downsample = Conv_3x3(num_channels, num_channels, stride=2)

        # Weight
        self.relu = nn.ReLU()

    def forward(self, p3_in, p4_in, p5_in, p6_in, p7_in):
        """
            P7_0 -------------------------- P7_2 -------->
            P6_0 ---------- P6_1 ---------- P6_2 -------->
            P5_0 ---------- P5_1 ---------- P5_2 -------->
            P4_0 ---------- P4_1 ---------- P4_2 -------->
            P3_0 -------------------------- P3_2 -------->
        """

        # P7_0 to P7_2
        # Weights for P6_0 and P7_0 to P6_1
        p6_up = self.conv6_up(p6_in + self.p6_upsample(p7_in, p6_in))

        # Connections for P5_0 and P6_0 to P5_1 respectively
        p5_up = self.conv5_up(p5_in + self.p5_upsample(p6_up, p5_in))

        # Connections for P4_0 and P5_0 to P4_1 respectively
        p4_up = self.conv4_up(p4_in + self.p4_upsample(p5_up, p4_in))

        # Connections for P3_0 and P4_1 to P3_2 respectively
        p3_out = self.conv3_up(p3_in + self.p3_upsample(p4_up, p3_in))

        # Connections for P4_0, P4_1 and P3_2 to P4_2 respectively
        p4_out = self.conv4_down(p4_in + p4_up + self.p4_downsample(p3_out))

        # Connections for P5_0, P5_1 and P4_2 to P5_2 respectively
        p5_out = self.conv5_down(p5_in + p5_up + self.p5_downsample(p4_out))

        # Connections for P6_0, P6_1 and P5_2 to P6_2 respectively
        p6_out = self.conv6_down(p6_in + p6_up + self.p6_downsample(p5_out))

        # Connections for P7_0 and P6_2 to P7_2
        p7_out = self.conv7_down(p7_in + self.p7_downsample(p6_out))

        return p3_out, p4_out, p5_out, p6_out, p7_out

        # self.feature1 = p5_out
        # self.feature2 = p6_out
        # self.feature3 = p7_out

        # return p3_out, p4_out, p5_out, p6_out, p7_out


class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, expand):
        super(MBConv, self).__init__()
        mid_channels = int(in_channels / expand)

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel, stride=stride, padding=kernel // 2, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.use_skip_connect = (1 == stride and in_channels == out_channels)

    def forward(self, x):
        if self.use_skip_connect:
            return F.relu(self.block(x) + x)
        else:
            return F.relu(self.block(x))

def conv3x3(in_channels, out_channels, module_name, postfix,
            stride=1, groups=1, kernel_size=3, padding=1):
    """3x3 convolution with padding"""
    return [
        ('{}_{}/conv'.format(module_name, postfix),
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      groups=groups,
                      bias=False)),
        ('{}_{}/norm'.format(module_name, postfix),
            nn.BatchNorm2d(out_channels)),
        ('{}_{}/relu'.format(module_name, postfix),
            nn.ReLU(inplace=True)),
    ]


def conv1x1(in_channels, out_channels, module_name, postfix,
            stride=1, groups=1, kernel_size=1, padding=0):
    """1x1 convolution"""
    return [
        ('{}_{}/conv'.format(module_name, postfix),
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      groups=groups,
                      bias=False)),
        ('{}_{}/norm'.format(module_name, postfix),
            nn.BatchNorm2d(out_channels)),
        ('{}_{}/relu'.format(module_name, postfix),
            nn.ReLU(inplace=True)),
    ]


class _OSA_module(nn.Module):
    def __init__(self,
                 in_ch,
                 stage_ch,
                 concat_ch,
                 layer_per_block,
                 module_name,
                 identity=False):
        super(_OSA_module, self).__init__()

        self.identity = identity
        self.layers = nn.ModuleList()
        in_channel = in_ch
        for i in range(layer_per_block):
            self.layers.append(nn.Sequential(
                OrderedDict(conv3x3(in_channel, stage_ch, module_name, i))))
            in_channel = stage_ch

        # feature aggregation
        in_channel = in_ch + layer_per_block * stage_ch
        self.concat = nn.Sequential(
            OrderedDict(conv1x1(in_channel, concat_ch, module_name, 'concat')))

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




class BaseNet(nn.Module):
    def __init__(self, base_num):
        super(BaseNet, self).__init__()
        stage_channels = [0, 16, 16, 32, 64, 128, 128]
        channels =       [3, 16, 32, 64, 128, 256, 384, 512]
        # stage 1: stride 2 feature map
        self.layer1 = Conv_3x3(channels[0], channels[1], 2)

        # stage 2: stride 4 feature map
        self.layer2 = nn.Sequential(Conv_3x3(channels[1], stage_channels[1], 2),
            _OSA_module(stage_channels[1], stage_channels[1], channels[2], 2, 'layer2'))

        # stage 3: stride 8 feature map
        self.layer3 = nn.Sequential(
            Conv_3x3(channels[2], stage_channels[2], 2),
            _OSA_module(stage_channels[2], stage_channels[2], channels[3], 3, 'layer3')
        )
        # stage 4: stride 16 feature map
        self.layer4 = nn.Sequential(
            Conv_3x3(channels[3], stage_channels[3], 2),
            _OSA_module(stage_channels[3], stage_channels[3], channels[4], 3, 'layer4')
        )
        # stage 5: stride 32 feature map
        self.layer5 = nn.Sequential(
            Conv_3x3(channels[4], stage_channels[4], 2),
            _OSA_module(stage_channels[4], stage_channels[4], channels[5], 4, 'layer5')
        )
        # stage 6: stride 64 feature map
        self.layer6 = nn.Sequential(
            Conv_3x3(channels[5], stage_channels[5], 2),
            _OSA_module(stage_channels[5], stage_channels[5], channels[6], 5, 'layer6')
        )

        self.layer7 = nn.Sequential(
            Conv_3x3(channels[6], stage_channels[6], 2),
            _OSA_module(stage_channels[6], stage_channels[6], channels[7], 3, 'layer7')
        )

        self.p7_lateral = Conv_1x1(channels[7], 32, relu=False)
        self.p6_lateral = Conv_1x1(channels[6], 32, relu=False)
        self.p5_lateral = Conv_1x1(channels[5], 32, relu=False)
        self.p4_lateral = Conv_1x1(channels[4], 32, relu=False)
        self.p3_lateral = Conv_1x1(channels[3], 32, relu=False)

        self.fpn = BiFPN(32)

        self.out_channels = 32

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n) ** 0.5)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):

        _p3 = self.layer3(self.layer2(self.layer1(x)))
        _p4 = self.layer4(_p3)
        _p5 = self.layer5(_p4)
        _p6 = self.layer6(_p5)
        _p7 = self.layer7(_p6)

        p3 = self.p3_lateral(_p3)
        p4 = self.p4_lateral(_p4)
        p5 = self.p5_lateral(_p5)
        p6 = self.p6_lateral(_p6)
        p7 = self.p7_lateral(_p7)

        P3, P4, P5, P6, P7 = self.fpn(p3, p4, p5, p6, p7)

        return [P3, P4, P5, P6, P7]


