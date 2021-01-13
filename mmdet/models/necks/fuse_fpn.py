import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmdet.ops.build import build_op
from mmcv.runner import auto_fp16
from ..builder import NECKS
from .fpn import FPN
from mmcv.cnn import ConvModule, xavier_init

@NECKS.register_module()
class FuseFPN(nn.Module):
    """Path Aggregation Network for Instance Segmentation.

    This is an implementation of the `PAFPN in Path Aggregation Network
    <https://arxiv.org/abs/1803.01534>`_.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool): Whether to add conv layers on top of the
            original feature maps. Default: False.
        extra_convs_on_inputs (bool): Whether to apply extra conv on
            the original feature from the backbone. Default: False.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 stack_conv = 2,
                 conv_cfg=None):
        super(FuseFPN, self).__init__()
        # add extra bottom up pathway
        num_in = len(in_channels)
        self.in_channels = in_channels
        self.upsample_conv = nn.ModuleList()
        self.lateral_conv = nn.ModuleList()

        assert isinstance(conv_cfg, dict), "conv_cfg should be dict!"
        conv_type = conv_cfg["type"]
        conv_info = conv_cfg["info"]
        assert isinstance(conv_type, str), "conv_type should be string!"
        # assert isinstance(conv_info, dict), "conv_info should be dict!"

        assert "norm_cfg" in conv_info
        norm_cfg = conv_info["norm_cfg"]
        Conv = build_op(conv_type)

        for i in range(num_in):
            lateral_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                padding=0,
                norm_cfg=norm_cfg,
                inplace=True)
            if i < num_in - 1:
                upsample_conv = nn.Sequential()
                for index in range(stack_conv):
                    upsample_conv.add_module(name="upsample_conv3x3_{}".format(index),
                                             module=Conv(
                                                    out_channels,
                                                    out_channels,
                                                    3,
                                                    **conv_info))
                    upsample_conv.add_module(name="upsample_conv1x1_{}".format(index),
                                             module=ConvModule(
                                                    out_channels,
                                                    out_channels,
                                                    1,
                                                    stride=1,
                                                    padding=0,
                                                    norm_cfg=norm_cfg,
                                                    inplace=True))
                self.upsample_conv.append(upsample_conv)
            self.lateral_conv.append(lateral_conv)

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)
        # build laterals

        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_conv)
        ]

        # build top-down path
        for i in range(len(inputs)-1, 0, -1):
            laterals[i-1] = self.upsample_conv[i-1](F.interpolate(laterals[i], laterals[i-1].shape[2:]) + laterals[i-1])


        # build outputs
        # part 1: from original levels

        return tuple([laterals[0]])

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
