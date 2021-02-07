import torch.nn as nn
from mmdet.ops.build import build_op
import torch.nn.functional as F
from mmcv.cnn import Scale
from ..builder import NECKS
from mmcv.cnn import ConvModule, xavier_init

@NECKS.register_module()
class YeFPNv2(nn.Module):
    """
    modified by Zylo117
    """

    def __init__(self, in_channels, out_channels, conv_cfg, num_stack=1):
        """
        Args:
            num_channels:
            conv_channels:
            first_time: whether the input comes directly from the efficientnet,
                        if True, downchannel it first, and downsample P5 to generate P6 then P7
            epsilon: epsilon of fast weighted attention sum of BiFPN, not the BN's epsilon
            onnx_export: if True, use Swish instead of MemoryEfficientSwish
        """

        super(YeFPNv2, self).__init__()
        assert isinstance(conv_cfg, dict), "conv_cfg should be dict!"
        conv_type = conv_cfg["type"]
        conv_info = conv_cfg["info"]
        assert isinstance(conv_type, str), "conv_type should be string!"
        # assert isinstance(conv_info, dict), "conv_info should be dict!"

        assert "norm_cfg" in conv_info
        norm_cfg = conv_info["norm_cfg"]


        Conv = build_op(conv_type)


        self.extra_conv6 = ConvModule(in_channels[3], out_channels, kernel_size=3, stride=2, padding=1, norm_cfg=norm_cfg)
        self.extra_conv7 = ConvModule(out_channels, out_channels, kernel_size=3, stride=2, padding=1, norm_cfg=norm_cfg)

        # Conv layers
        self.lateral_P2 = ConvModule(in_channels[0], out_channels, kernel_size=1, norm_cfg=norm_cfg)
        self.lateral_P3 = ConvModule(in_channels[1], out_channels, kernel_size=1, norm_cfg=norm_cfg)
        self.lateral_P4 = ConvModule(in_channels[2], out_channels, kernel_size=1, norm_cfg=norm_cfg)
        self.lateral_P5 = ConvModule(in_channels[3], out_channels, kernel_size=1, norm_cfg=norm_cfg)
        self.lateral_P6 = ConvModule(out_channels, out_channels, kernel_size=1, norm_cfg=norm_cfg)
        self.lateral_P7 = ConvModule(out_channels, out_channels, kernel_size=1, norm_cfg=norm_cfg)

        self.conv7_1 = Conv(out_channels, out_channels, kernel_size=3, **conv_info)
        self.conv6_1 = Conv(out_channels, out_channels, kernel_size=3, **conv_info)
        self.conv5_1 = Conv(out_channels, out_channels, kernel_size=3, **conv_info)
        self.conv4_1 = Conv(out_channels, out_channels, kernel_size=3, **conv_info)
        self.conv3_1 = Conv(out_channels, out_channels, kernel_size=3, **conv_info)
        self.conv2_1 = Conv(out_channels, out_channels, kernel_size=3, **conv_info)

        self.conv7_2 = Conv(out_channels, out_channels, kernel_size=3, **conv_info)
        self.conv6_2 = Conv(out_channels, out_channels, kernel_size=3, **conv_info)
        self.conv5_2 = Conv(out_channels, out_channels, kernel_size=3, **conv_info)
        self.conv4_2 = Conv(out_channels, out_channels, kernel_size=3, **conv_info)
        self.conv3_2 = Conv(out_channels, out_channels, kernel_size=3, **conv_info)
        self.conv2_2 = Conv(out_channels, out_channels, kernel_size=3, **conv_info)

        self.dowm_con6_7 = ConvModule(out_channels, out_channels, kernel_size=3, stride=2, padding=1, norm_cfg=norm_cfg)
        self.dowm_con3_4 = ConvModule(out_channels, out_channels, kernel_size=3, stride=2, padding=1, norm_cfg=norm_cfg)
        self.dowm_con2_3 = ConvModule(out_channels, out_channels, kernel_size=3, stride=2, padding=1, norm_cfg=norm_cfg)
        self.dowm_con5_6 = ConvModule(out_channels, out_channels, kernel_size=3, stride=2, padding=1, norm_cfg=norm_cfg)
        self.dowm_con4_5 = ConvModule(out_channels, out_channels, kernel_size=3, stride=2, padding=1, norm_cfg=norm_cfg)

        self.upsample7_6 = F.interpolate
        self.upsample6_5 = F.interpolate
        self.upsample5_4 = F.interpolate
        self.upsample4_3 = F.interpolate
        self.upsample3_2 = F.interpolate


        self.scale7_6 = Scale()
        self.scale5_4 = Scale()
        self.scale4_3 = Scale()
        self.scale3_2 = Scale()
        self.scale6_5 = Scale()
        self.scale4_5 = Scale()

        self.scale2_3 = Scale()


        self.shorcut_scale5_6 = Scale()
        self.shorcut_scale5_4 = Scale()
        self.shorcut_scale3_4 = Scale()

    def forward(self, x):
        '''
        :param x:
        :return:
        '''

        P2, P3, P4, P5 = x
        P6 = self.extra_conv6(P5)
        P7 = self.extra_conv7(P6)

        P2_shape = P2.shape[2:]
        P3_shape = P3.shape[2:]
        P4_shape = P4.shape[2:]
        P5_shape = P5.shape[2:]
        P6_shape = P6.shape[2:]
        P7_shape = P7.shape[2:]

        P2 = self.lateral_P2(P2)
        P3 = self.lateral_P3(P3)
        P4 = self.lateral_P4(P4)
        P5 = self.lateral_P5(P5)
        P6 = self.lateral_P6(P6)
        P7 = self.lateral_P7(P7)

        P2_1 = self.conv2_1(P2)
        P4_1 = self.conv4_1(P4)
        P3_1 = self.conv3_1(P3) + self.scale2_3(self.dowm_con2_3(P2_1)) + self.scale4_3(self.upsample4_3(P4_1, P3_shape, mode="nearest"))
        P4_1 = self.conv4_1(P4)
        P7_1 = self.conv7_1(P7)
        P6_1 = self.conv6_1(P6) + self.scale7_6(self.upsample7_6(P7_1, P6_shape, mode="nearest"))
        P5_1 = self.conv5_1(P5) + self.scale4_5(self.dowm_con4_5(P4)) + self.scale6_5(self.upsample6_5(P6_1, P5_shape, mode="nearest"))



        P3_2 = self.conv3_2(P3_1)
        P2_2 = self.conv2_2(P2_1) + self.scale3_2(self.upsample3_2(P3, P2_shape, mode="nearest"))
        P4_2 = self.shorcut_scale5_4(self.upsample5_4(P5, P4_shape, mode="nearest")) + self.shorcut_scale3_4(self.dowm_con3_4(P3)) + self.conv4_2(P4_1)
        P5_2 = self.conv5_2(P5_1)
        P6_2 = self.conv6_2(P6_1) + self.shorcut_scale5_6(self.dowm_con5_6(P5))
        P7_2 = self.conv7_2(P7_1) + self.dowm_con6_7(P6_2)



        return tuple([P2_2, P3_2, P4_2, P5_2, P6_2, P7_2])

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
