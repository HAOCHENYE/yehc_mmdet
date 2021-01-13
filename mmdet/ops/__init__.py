# This file is added for back-compatibility. Thus, downstream codebase
# could still use and import mmdet.ops.

# yapf: disable
from mmcv.ops import (Conv2d, ConvTranspose2d, CornerPool, Linear, MaskedConv2d, MaxPool2d,
                      RoIAlign, RoIPool, SAConv2d, SigmoidFocalLoss, SimpleRoIAlign, batched_nms,
                      deform_conv, get_compiler_version, get_compiling_cuda_version, modulated_deform_conv, nms,
                      nms_match, point_sample, rel_roi_point_to_rel_img_point, roi_align, roi_pool,
                      sigmoid_focal_loss, soft_nms)
from .ConvOp import NormalConv, SepConv, ACBlock

# yapf: enable

__all__ = [
    'nms', 'soft_nms', 'RoIAlign', 'roi_align', 'RoIPool', 'roi_pool',
    'deform_conv', 'modulated_deform_conv', 'SigmoidFocalLoss', 'sigmoid_focal_loss',
    'MaskedConv2d', 'get_compiler_version', 'get_compiling_cuda_version',
    'batched_nms', 'Conv2d', 'ConvTranspose2d', 'MaxPool2d', 'Linear', 'nms_match', 'CornerPool',
    'point_sample', 'rel_roi_point_to_rel_img_point', 'SimpleRoIAlign',
    'SAConv2d', 'NormalConv', 'SepConv', 'ACBlock'
]
