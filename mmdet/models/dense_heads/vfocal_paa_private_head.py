import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, Scale, bias_init_with_prob, normal_init
from mmcv.ops import DeformConv2d
from mmcv.runner import force_fp32
import torch.nn.functional as F
from mmdet.core import (bbox2distance, bbox_overlaps, build_anchor_generator,
                        build_assigner, build_sampler, distance2bbox,
                        multi_apply, multiclass_nms, reduce_mean)
from ..builder import HEADS, build_loss
from .atss_head import ATSSHead
from .fcos_head import FCOSHead
from .paa_atss_head import PAA_ATSSHead
try:
    import sklearn.mixture as skm
except ImportError:
    skm = None

INF = 1e8

def levels_to_images(mlvl_tensor):
    """Concat multi-level feature maps by image.
    [feature_level0, feature_level1...] -> [feature_image0, feature_image1...]
    Convert the shape of each element in mlvl_tensor from (N, C, H, W) to
    (N, H*W , C), then split the element to N elements with shape (H*W, C), and
    concat elements in same image of all level along first dimension.
    Args:
        mlvl_tensor (list[torch.Tensor]): list of Tensor which collect from
            corresponding level. Each element is of shape (N, C, H, W)
    Returns:
        list[torch.Tensor]: A list that contains N tensors and each tensor is
            of shape (num_elements, C)
    """
    batch_size = mlvl_tensor[0].size(0)
    batch_list = [[] for _ in range(batch_size)]
    channels = mlvl_tensor[0].size(1)
    for t in mlvl_tensor:
        t = t.permute(0, 2, 3, 1)
        t = t.view(batch_size, -1, channels).contiguous()
        for img in range(batch_size):
            batch_list[img].append(t[img])
    return [torch.cat(item, 0) for item in batch_list]

@HEADS.register_module()
class VFocalPAAHead(PAA_ATSSHead, ATSSHead, FCOSHead):
    """Head of `VarifocalNet (VFNet): An IoU-aware Dense Object
    Detector.<https://arxiv.org/abs/2008.13367>`_.
    The VFNet predicts IoU-aware classification scores which mix the
    object presence confidence and object localization accuracy as the
    detection score. It is built on the FCOS architecture and uses ATSS
    for defining positive/negative training examples. The VFNet is trained
    with Varifocal Loss and empolys star-shaped deformable convolution to
    extract features for a bbox.
    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        regress_ranges (tuple[tuple[int, int]]): Regress range of multiple
            level points.
        center_sampling (bool): If true, use center sampling. Default: False.
        center_sample_radius (float): Radius of center sampling. Default: 1.5.
        sync_num_pos (bool): If true, synchronize the number of positive
            examples across GPUs. Default: True
        gradient_mul (float): The multiplier to gradients from bbox refinement
            and recognition. Default: 0.1.
        bbox_norm_type (str): The bbox normalization type, 'reg_denom' or
            'stride'. Default: reg_denom
        loss_cls_fl (dict): Config of focal loss.
        use_vfl (bool): If true, use varifocal loss for training.
            Default: True.
        loss_cls (dict): Config of varifocal loss.
        loss_bbox (dict): Config of localization loss, GIoU Loss.
        loss_bbox (dict): Config of localization refinement loss, GIoU Loss.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: norm_cfg=dict(type='GN', num_groups=32,
            requires_grad=True).
        use_atss (bool): If true, use ATSS to define positive/negative
            examples. Default: True.
        anchor_generator (dict): Config of anchor generator for ATSS.
    Example:
        >>> self = VFNetHead(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_score, bbox_pred, bbox_pred_refine= self.forward(feats)
        >>> assert len(cls_score) == len(self.scales)
    """  # noqa: E501

    def __init__(self,
                 num_classes,
                 in_channels,
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 center_sampling=False,
                 center_sample_radius=1.5,
                 sync_num_pos=True,
                 gradient_mul=0.1,
                 topk=9,
                 bbox_norm_type='reg_denom',
                 loss_cls_fl=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 use_vfl=True,
                 loss_cls=dict(
                     type='VarifocalLoss',
                     use_sigmoid=True,
                     alpha=0.75,
                     gamma=2.0,
                     iou_weighted=True,
                     loss_weight=1.0),
                 loss_bbox=dict(type='GIoULoss', loss_weight=1.5),
                 loss_bbox_refine=dict(type='GIoULoss', loss_weight=2.0),
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 assign_type="paa",
                 anchor_generator=dict(
                     type='AnchorGenerator',
                     ratios=[1.0],
                     octave_base_scale=8,
                     scales_per_octave=1,
                     center_offset=0.0,
                     strides=[8, 16, 32, 64, 128]),
                 **kwargs):
        # dcn base offsets, adapted from reppoints_head.py
        self.topk=topk
        self.num_out = len(anchor_generator["strides"])
        self.anchor_generator = build_anchor_generator(anchor_generator)
        self.num_anchors = self.anchor_generator.num_base_anchors[0]
        super(FCOSHead, self).__init__(
            num_classes, in_channels, norm_cfg=norm_cfg, **kwargs)

        self.regress_ranges = regress_ranges
        self.reg_denoms = [
            regress_range[-1] for regress_range in regress_ranges
        ]
        self.reg_denoms[-1] = self.reg_denoms[-2] * 2
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.sync_num_pos = sync_num_pos
        self.bbox_norm_type = bbox_norm_type
        self.gradient_mul = gradient_mul
        self.use_vfl = use_vfl
        if self.use_vfl:
            self.loss_cls = build_loss(loss_cls)
        else:
            self.loss_cls = build_loss(loss_cls_fl)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_bbox_refine = build_loss(loss_bbox_refine)

        # for getting ATSS targets
        self.assign_type = assign_type
        if self.assign_type == 'paa':
            # paa_assign_cls_loss = dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0)
            paa_assign_cls_loss = loss_cls_fl
            self.paa_assign_cls_loss = build_loss(paa_assign_cls_loss)

        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        self.anchor_center_offset = anchor_generator['center_offset']
        self.num_anchors = self.anchor_generator.num_base_anchors[0]
        self.sampling = False
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.mlvl_cls_convs = nn.ModuleList()
        self.mlvl_reg_convs = nn.ModuleList()
        self.mlvl_vfl_cls_convs = nn.ModuleList()
        self.mlvl_vfl_reg_convs = nn.ModuleList()
        self.mlvl_vfl_reg = nn.ModuleList()
        self.mlvl_vfl_refine_convs = nn.ModuleList()
        self.mlvl_scale = nn.ModuleList()
        self.mlvl_refine_scale = nn.ModuleList()


        for level in range(self.num_out):
            cls_convs = nn.ModuleList()
            reg_convs = nn.ModuleList()
            for i in range(self.stacked_convs):
                chn = self.in_channels if i == 0 else self.feat_channels
                cls_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
                reg_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))

            vfl_cls_convs = nn.Conv2d(self.feat_channels, self.cls_out_channels, 1, padding=0)
            vfl_reg_convs = ConvModule(self.feat_channels, self.feat_channels, 3,
                                       stride=1,
                                       padding=1,
                                       conv_cfg=self.conv_cfg,
                                       norm_cfg=self.norm_cfg,
                                       bias=self.conv_bias)

            vfl_refine_convs = nn.Conv2d(self.feat_channels, 4, 1, padding=0)
            vfl_reg = nn.Conv2d(self.feat_channels, 4, 1, padding=0)

            scale = Scale(1.0)
            scale_refine = Scale(1.0)

            self.mlvl_cls_convs.append(cls_convs)
            self.mlvl_reg_convs.append(reg_convs)
            self.mlvl_vfl_cls_convs.append(vfl_cls_convs)
            self.mlvl_vfl_reg_convs.append(vfl_reg_convs)
            self.mlvl_vfl_refine_convs.append(vfl_refine_convs)
            self.mlvl_vfl_reg.append(vfl_reg)

            self.mlvl_scale.append(scale)
            self.mlvl_refine_scale.append(scale_refine)

    def init_weights(self):
        """Initialize weights of the head."""
        for level in range(self.num_out):
            for m in self.mlvl_cls_convs[level]:
                if isinstance(m.conv, nn.Conv2d):
                    normal_init(m.conv, std=0.01)
            for m in self.mlvl_reg_convs[level]:
                if isinstance(m.conv, nn.Conv2d):
                    normal_init(m.conv, std=0.01)
            normal_init(self.mlvl_vfl_reg[level], std=0.01)
            # normal_init(self.mlvl_vfl_cls_convs[level], std=0.01)
            normal_init(self.mlvl_vfl_reg_convs[level].conv, std=0.01)
            normal_init(self.mlvl_vfl_refine_convs[level], std=0.01)
            bias_cls = bias_init_with_prob(0.01)
            normal_init(self.mlvl_vfl_cls_convs[level], std=0.01, bias=bias_cls)

    def forward(self, feats, onnx=False):
        """Forward features from the upstream network.
        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            tuple:
                cls_scores (list[Tensor]): Box iou-aware scores for each scale
                    level, each is a 4D-tensor, the channel number is
                    num_points * num_classes.
                bbox_preds (list[Tensor]): Box offsets for each
                    scale level, each is a 4D-tensor, the channel number is
                    num_points * 4.
                bbox_preds_refine (list[Tensor]): Refined Box offsets for
                    each scale level, each is a 4D-tensor, the channel
                    number is num_points * 4.
        """
        return multi_apply(self.forward_single, feats, self.mlvl_scale,
                           self.mlvl_refine_scale, self.strides, self.reg_denoms,
                           [level for level in range(self.num_out)],
                           [onnx for _ in range(len(feats))])

    def forward_single(self, x, scale, scale_refine, stride, reg_denom, level, onnx=False):
        """Forward features of a single scale level.
        Args:
            x (Tensor): FPN feature maps of the specified stride.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.
            scale_refine (:obj: `mmcv.cnn.Scale`): Learnable scale module to
                resize the refined bbox prediction.
            stride (int): The corresponding stride for feature maps,
                used to normalize the bbox prediction when
                bbox_norm_type = 'stride'.
            reg_denom (int): The corresponding regression range for feature
                maps, only used to normalize the bbox prediction when
                bbox_norm_type = 'reg_denom'.
        Returns:
            tuple: iou-aware cls scores for each box, bbox predictions and
                refined bbox predictions of input feature maps.
        """
        cls_feat = x
        reg_feat = x

        for cls_layer in self.mlvl_cls_convs[level]:
            cls_feat = cls_layer(cls_feat)

        for reg_layer in self.mlvl_reg_convs[level]:
            reg_feat = reg_layer(reg_feat)

        # predict the bbox_pred of different level
        reg_feat_init = self.mlvl_vfl_reg_convs[level](reg_feat)
        if onnx:
            bbox_pred = self.mlvl_vfl_reg[level](reg_feat_init)
            scale = reg_denom*scale.scale

            bbox_pred = scale*bbox_pred
            bbox_pred_refine = scale_refine(self.mlvl_vfl_refine_convs[level](reg_feat))
            bbox_pred_refine = bbox_pred_refine * bbox_pred.detach()

            cls_score = F.sigmoid(self.mlvl_vfl_cls_convs[level](cls_feat))
            return cls_score, bbox_pred_refine
        elif self.bbox_norm_type == 'reg_denom':
            bbox_pred = scale(
                # self.mlvl_vfl_reg[level](reg_feat_init)).exp() * reg_denom
                self.mlvl_vfl_reg[level](reg_feat_init)) * reg_denom
        elif self.bbox_norm_type == 'stride':
            bbox_pred = scale(
                self.mlvl_vfl_reg[level](reg_feat_init)) * stride
                # self.mlvl_vfl_reg[level](reg_feat_init)).exp() * stride
        else:
            raise NotImplementedError

        # compute star deformable convolution offsets
        # converting dcn_offset to reg_feat.dtype thus VFNet can be
        # trained with FP16

        # refine the bbox_pred

        bbox_pred_refine = scale_refine(
            self.mlvl_vfl_refine_convs[level](reg_feat))
            # self.mlvl_vfl_refine_convs[level](reg_feat)).float().exp()
        bbox_pred_refine = bbox_pred_refine * bbox_pred.detach()

        # predict the iou-aware cls score
        cls_score = self.mlvl_vfl_cls_convs[level](cls_feat)

        return cls_score, bbox_pred, bbox_pred_refine
    # import pysnooper
    # @pysnooper.snoop(depth=2)
    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'bbox_preds_refine'))
    def loss(self,
             cls_scores,
             bbox_preds,
             bbox_preds_refine,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute loss of the head.
        Args:
            cls_scores (list[Tensor]): Box iou-aware scores for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * num_classes.
            bbox_preds (list[Tensor]): Box offsets for each
                scale level, each is a 4D-tensor, the channel number is
                num_points * 4.
            bbox_preds_refine (list[Tensor]): Refined Box offsets for
                each scale level, each is a 4D-tensor, the channel
                number is num_points * 4.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
                Default: None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert len(cls_scores) == len(bbox_preds) == len(bbox_preds_refine)
        device = cls_scores[0].device
        num_imgs = len(img_metas)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                           bbox_preds[0].device)

        anchor_list, valid_flag_list = PAA_ATSSHead.get_anchors(self,
            featmap_sizes, img_metas, device=device)

        if self.assign_type == 'paa':
            labels, label_weights, \
            bbox_targets, bbox_weights, \
            pos_inds_image, pos_gt_index_image = self.get_targets(
                cls_scores, all_level_points, gt_bboxes, gt_labels, img_metas,
                gt_bboxes_ignore)
        else:
            labels, label_weights, bbox_targets, bbox_weights = self.get_targets(
                cls_scores, all_level_points, gt_bboxes, gt_labels, img_metas,
                gt_bboxes_ignore)

        cls_scores = levels_to_images(cls_scores)
        cls_scores = [item.reshape(-1, self.cls_out_channels) for item in cls_scores]

        bbox_preds = levels_to_images(bbox_preds)
        bbox_preds = [item.reshape(-1, 4) for item in bbox_preds]

        bbox_preds_refine = levels_to_images(bbox_preds_refine)
        bbox_preds_refine = [item.reshape(-1, 4) for item in bbox_preds_refine]

        if self.assign_type == 'paa':
            all_level_points_cat = [torch.cat(all_level_points) for _ in range(num_imgs)]
        #     # print("paa get_pos_loss start")
            pos_losses_list, = multi_apply(self.get_pos_loss,
                                           cls_scores, bbox_preds, bbox_preds_refine,
                                           labels, label_weights,
                                           bbox_targets, bbox_weights,
                                           all_level_points_cat,
                                           pos_inds_image)
            # print("paa get_pos_loss end")
            with torch.no_grad():
                # print("paa_reassign start")
                labels, label_weights, bbox_weights, num_pos = multi_apply(
                    self.paa_reassign,
                    pos_losses_list,
                    labels,
                    label_weights,
                    bbox_weights,
                    pos_inds_image,
                    pos_gt_index_image,
                    anchor_list,
                )
                # print("paa_reassign end")

        bg_class_ind = self.num_classes
        # flatten cls_scores, bbox_preds and bbox_preds_refine
        if self.assign_type == 'paa':
            # print("assign_type start")
            #num_batch is first dim
            flatten_cls_scores = torch.cat(cls_scores, 0).view(-1, cls_scores[0].size(-1))
            flatten_bbox_preds = torch.cat(bbox_preds, 0).view(-1, bbox_preds[0].size(-1))
            flatten_bbox_preds_refine = torch.cat(bbox_preds_refine, 0).view(-1, bbox_preds_refine[0].size(-1))

            flatten_points = torch.cat(all_level_points_cat)
            flatten_labels = torch.cat(labels, 0).view(-1)
            flatten_bbox_targets = torch.cat(bbox_targets,
                                      0).view(-1, bbox_targets[0].size(-1))

            pos_inds = ((flatten_labels >= 0) & (flatten_labels < self.num_classes)).nonzero().reshape(-1)
            # print("assign_type end")
        else:
            # num_level is first dim
            flatten_cls_scores = [
                cls_score.permute(0, 2, 3,
                                  1).reshape(-1,
                                             self.cls_out_channels).contiguous()
                for cls_score in cls_scores
            ]
            flatten_bbox_preds = [
                bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4).contiguous()
                for bbox_pred in bbox_preds
            ]
            flatten_bbox_preds_refine = [
                bbox_pred_refine.permute(0, 2, 3, 1).reshape(-1, 4).contiguous()
                for bbox_pred_refine in bbox_preds_refine
            ]
            flatten_cls_scores = torch.cat(flatten_cls_scores)
            flatten_bbox_preds = torch.cat(flatten_bbox_preds)
            flatten_bbox_preds_refine = torch.cat(flatten_bbox_preds_refine)

            flatten_labels = torch.cat(labels)
            flatten_bbox_targets = torch.cat(bbox_targets)

            flatten_points = torch.cat([points.repeat(num_imgs, 1) for points in all_level_points])

            pos_inds = torch.where(
                ((flatten_labels >= 0) & (flatten_labels < bg_class_ind)) > 0)[0]

        num_pos = len(pos_inds)

        # FG cat_id: [0, num_classes - 1], BG cat_id: num_classes


        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_bbox_preds_refine = flatten_bbox_preds_refine[pos_inds]
        pos_labels = flatten_labels[pos_inds]

        # sync num_pos across all gpus
        if self.sync_num_pos:
            num_pos_avg_per_gpu = reduce_mean(
                pos_inds.new_tensor(num_pos).float()).item()
            num_pos_avg_per_gpu = max(num_pos_avg_per_gpu, 1.0)
        else:
            num_pos_avg_per_gpu = num_pos



        if num_pos > 0:
            pos_bbox_targets = flatten_bbox_targets[pos_inds]
            pos_points = flatten_points[pos_inds]

            pos_decoded_bbox_preds = distance2bbox(pos_points, pos_bbox_preds)
            pos_decoded_target_preds = distance2bbox(pos_points,
                                                     pos_bbox_targets)
            iou_targets_ini = bbox_overlaps(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds.detach(),
                is_aligned=True).clamp(min=1e-6)
            bbox_weights_ini = iou_targets_ini.clone().detach()
            iou_targets_ini_avg_per_gpu = reduce_mean(
                bbox_weights_ini.sum()).item()

            bbox_avg_factor_ini = max(iou_targets_ini_avg_per_gpu, 1.0)
            # print("loss_bbox start")
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds.detach(),
                weight=bbox_weights_ini,
                avg_factor=bbox_avg_factor_ini)
            # print("loss_bbox end")
            pos_decoded_bbox_preds_refine = \
                distance2bbox(pos_points, pos_bbox_preds_refine)
            iou_targets_rf = bbox_overlaps(
                pos_decoded_bbox_preds_refine,
                pos_decoded_target_preds.detach(),
                is_aligned=True).clamp(min=1e-6)
            bbox_weights_rf = iou_targets_rf.clone().detach()
            iou_targets_rf_avg_per_gpu = reduce_mean(
                bbox_weights_rf.sum()).item()
            bbox_avg_factor_rf = max(iou_targets_rf_avg_per_gpu, 1.0)
            # print("loss_bbox_refine start")
            loss_bbox_refine = self.loss_bbox_refine(
                pos_decoded_bbox_preds_refine,
                pos_decoded_target_preds.detach(),
                weight=bbox_weights_rf,
                avg_factor=bbox_avg_factor_rf)
            # print("loss_bbox_refine end")
            # build IoU-aware cls_score targets
            if self.use_vfl:
                pos_ious = iou_targets_rf.clone().detach()
                cls_iou_targets = torch.zeros_like(flatten_cls_scores)
                cls_iou_targets[pos_inds, pos_labels] = pos_ious
        else:
            loss_bbox = pos_bbox_preds.sum() * 0
            loss_bbox_refine = pos_bbox_preds_refine.sum() * 0
            if self.use_vfl:
                cls_iou_targets = torch.zeros_like(flatten_cls_scores)

        if self.use_vfl:
            loss_cls = self.loss_cls(
                flatten_cls_scores,
                cls_iou_targets,
                avg_factor=num_pos_avg_per_gpu)
        else:
            try:
                label_weights = torch.cat(label_weights)
            except:
                pass
            # print("cls loss start")
            loss_cls = self.loss_cls(
                flatten_cls_scores,
                flatten_labels,
                weight=label_weights,
                avg_factor=num_pos_avg_per_gpu)
            # print("cls loss end")

        label_weights = torch.cat(label_weights)
        loss_focal = self.paa_assign_cls_loss(
                flatten_cls_scores,
                flatten_labels,
                weight=label_weights,
                avg_factor=num_pos_avg_per_gpu)


        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_bbox_rf=loss_bbox_refine,
            loss_focal=loss_focal)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'bbox_preds_refine'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   bbox_preds_refine,
                   img_metas,
                   cfg=None,
                   rescale=None,
                   with_nms=True):
        """Transform network outputs for a batch into bbox predictions.
        Args:
            cls_scores (list[Tensor]): Box iou-aware scores for each scale
                level with shape (N, num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box offsets for each scale
                level with shape (N, num_points * 4, H, W).
            bbox_preds_refine (list[Tensor]): Refined Box offsets for
                each scale level with shape (N, num_points * 4, H, W).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used. Default: None.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before returning boxes.
                Default: True.
        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of
                the corresponding box.
        """
        assert len(cls_scores) == len(bbox_preds) == len(bbox_preds_refine)
        num_levels = len(cls_scores)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                      bbox_preds[0].device)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds_refine[i][img_id].detach()
                for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            det_bboxes = self._get_bboxes_single(cls_score_list,
                                                 bbox_pred_list, mlvl_points,
                                                 img_shape, scale_factor, cfg,
                                                 rescale, with_nms)
            result_list.append(det_bboxes)
        return result_list

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           mlvl_points,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False,
                           with_nms=True):
        """Transform outputs for a single batch item into bbox predictions.
        Args:
            cls_scores (list[Tensor]): Box iou-aware scores for a single scale
                level with shape (num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box offsets for a single scale
                level with shape (num_points * 4, H, W).
            mlvl_points (list[Tensor]): Box reference for a single scale level
                with shape (num_total_points, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arrange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before returning boxes.
                Default: True.
        Returns:
            tuple(Tensor):
                det_bboxes (Tensor): BBox predictions in shape (n, 5), where
                    the first 4 columns are bounding box positions
                    (tl_x, tl_y, br_x, br_y) and the 5-th column is a score
                    between 0 and 1.
                det_labels (Tensor): A (n,) tensor where each item is the
                    predicted class label of the corresponding box.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, points in zip(cls_scores, bbox_preds,
                                                mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).contiguous().sigmoid()
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4).contiguous()

            nms_pre = cfg.get('nms_pre', -1)
            if 0 < nms_pre < scores.shape[0]:
                max_scores, _ = scores.max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
            bboxes = distance2bbox(points, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        if with_nms:
            det_bboxes, det_labels = multiclass_nms(mlvl_bboxes, mlvl_scores,
                                                    cfg.score_thr, cfg.nms,
                                                    cfg.max_per_img)
            return det_bboxes, det_labels
        else:
            return mlvl_bboxes, mlvl_scores

    def _get_points_single(self,
                           featmap_size,
                           stride,
                           dtype,
                           device,
                           flatten=False):
        """Get points according to feature map sizes."""
        h, w = featmap_size
        x_range = torch.arange(
            0, w * stride, stride, dtype=dtype, device=device)
        y_range = torch.arange(
            0, h * stride, stride, dtype=dtype, device=device)
        y, x = torch.meshgrid(y_range, x_range)
        # to be compatible with anchor points in ATSS
        if self.assign_type == 'atss' or self.assign_type == 'paa':
            points = torch.stack(
                (x.reshape(-1), y.reshape(-1)), dim=-1) + \
                     stride * self.anchor_center_offset
        else:
            points = torch.stack(
                (x.reshape(-1), y.reshape(-1)), dim=-1) + stride // 2
        return points

    def get_targets(self, cls_scores, mlvl_points, gt_bboxes, gt_labels,
                    img_metas, gt_bboxes_ignore):
        """A wrapper for computing ATSS and FCOS targets for points in multiple
        images.
        Args:
            cls_scores (list[Tensor]): Box iou-aware scores for each scale
                level with shape (N, num_points * num_classes, H, W).
            mlvl_points (list[Tensor]): Points of each fpn level, each has
                shape (num_points, 2).
            gt_bboxes (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
        Returns:
            tuple:
                labels_list (list[Tensor]): Labels of each level.
                label_weights (Tensor/None): Label weights of all levels.
                bbox_targets_list (list[Tensor]): Regression targets of each
                    level, (l, t, r, b).
                bbox_weights (Tensor/None): Bbox weights of all levels.
        """
        if self.assign_type == "atss":
            return self.get_atss_targets(cls_scores, mlvl_points, gt_bboxes,
                                         gt_labels, img_metas,
                                         gt_bboxes_ignore)
        elif self.assign_type == "paa":
            return self.get_paa_targets(cls_scores, mlvl_points, gt_bboxes,
                                         gt_labels, img_metas,
                                         gt_bboxes_ignore)
        else:
            self.norm_on_bbox = False
            return self.get_fcos_targets(mlvl_points, gt_bboxes, gt_labels)

    def _get_target_single(self, *args, **kwargs):
        """Avoid ambiguity in multiple inheritance."""
        if self.assign_type == "atss":
            return ATSSHead._get_target_single(self, *args, **kwargs)
        if self.assign_type == "paa":
            return PAA_ATSSHead._get_target_single(self, *args, **kwargs)
        else:
            return FCOSHead._get_target_single(self, *args, **kwargs)

    def get_fcos_targets(self, points, gt_bboxes_list, gt_labels_list):
        """Compute FCOS regression and classification targets for points in
        multiple images.
        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels_list (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).
        Returns:
            tuple:
                labels (list[Tensor]): Labels of each level.
                label_weights: None, to be compatible with ATSS targets.
                bbox_targets (list[Tensor]): BBox targets of each level.
                bbox_weights: None, to be compatible with ATSS targets.
        """
        labels, bbox_targets = FCOSHead.get_targets(self, points,
                                                    gt_bboxes_list,
                                                    gt_labels_list)
        label_weights = None
        bbox_weights = None
        return labels, label_weights, bbox_targets, bbox_weights

    def get_atss_targets(self,
                         cls_scores,
                         mlvl_points,
                         gt_bboxes,
                         gt_labels,
                         img_metas,
                         gt_bboxes_ignore=None):
        """A wrapper for computing ATSS targets for points in multiple images.
        Args:
            cls_scores (list[Tensor]): Box iou-aware scores for each scale
                level with shape (N, num_points * num_classes, H, W).
            mlvl_points (list[Tensor]): Points of each fpn level, each has
                shape (num_points, 2).
            gt_bboxes (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4). Default: None.
        Returns:
            tuple:
                labels_list (list[Tensor]): Labels of each level.
                label_weights (Tensor): Label weights of all levels.
                bbox_targets_list (list[Tensor]): Regression targets of each
                    level, (l, t, r, b).
                bbox_weights (Tensor): Bbox weights of all levels.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1

        cls_reg_targets = ATSSHead.get_targets(
            self,
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            unmap_outputs=True)
        if cls_reg_targets is None:
            return None

        (anchor_list, labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, num_total_pos, num_total_neg) = cls_reg_targets

        bbox_targets_list = [
            bbox_targets.reshape(-1, 4) for bbox_targets in bbox_targets_list
        ]

        num_imgs = len(img_metas)
        # transform bbox_targets (x1, y1, x2, y2) into (l, t, r, b) format
        bbox_targets_list = self.transform_bbox_targets(
            bbox_targets_list, mlvl_points, num_imgs)

        labels_list = [labels.reshape(-1) for labels in labels_list]
        label_weights_list = [
            label_weights.reshape(-1) for label_weights in label_weights_list
        ]
        bbox_weights_list = [
            bbox_weights.reshape(-1) for bbox_weights in bbox_weights_list
        ]
        label_weights = torch.cat(label_weights_list)
        bbox_weights = torch.cat(bbox_weights_list)
        return labels_list, label_weights, bbox_targets_list, bbox_weights

    def get_paa_targets(self,
                         cls_scores,
                         mlvl_points,
                         gt_bboxes,
                         gt_labels,
                         img_metas,
                         gt_bboxes_ignore=None):
        """A wrapper for computing ATSS targets for points in multiple images.
        Args:
            cls_scores (list[Tensor]): Box iou-aware scores for each scale
                level with shape (N, num_points * num_classes, H, W).
            mlvl_points (list[Tensor]): Points of each fpn level, each has
                shape (num_points, 2).
            gt_bboxes (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4). Default: None.
        Returns:
            tuple:
                labels_list (list[Tensor]): Labels of each level.
                label_weights (Tensor): Label weights of all levels.
                bbox_targets_list (list[Tensor]): Regression targets of each
                    level, (l, t, r, b).
                bbox_weights (Tensor): Bbox weights of all levels.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1

        cls_reg_targets = PAA_ATSSHead.get_targets(
            self,
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            unmap_outputs=True)
        if cls_reg_targets is None:
            return None

        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         pos_inds, pos_gt_index) = cls_reg_targets

        bbox_targets_list = [
            bbox_targets.reshape(-1, 4) for bbox_targets in bbox_targets_list
        ]

        num_imgs = len(img_metas)
        # transform bbox_targets (x1, y1, x2, y2) into (l, t, r, b) format
        mlvl_points = torch.cat(mlvl_points)
        bbox_targets_list = self.transform_paa_bbox_targets(bbox_targets_list, mlvl_points)

        labels_list = [labels.reshape(-1) for labels in labels_list]
        label_weights_list = [
            label_weights.reshape(-1) for label_weights in label_weights_list
        ]
        bbox_weights_list = [
            bbox_weights.reshape(-1) for bbox_weights in bbox_weights_list
        ]
        # label_weights = torch.cat(label_weights_list)
        # bbox_weights = torch.cat(bbox_weights_list)
        return labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, pos_inds, pos_gt_index

    def transform_paa_bbox_targets(self, decoded_bboxes, mlvl_points):
        """Transform bbox_targets (x1, y1, x2, y2) into (l, t, r, b) format.
        Args:
            decoded_bboxes (list[Tensor]): Regression targets of each level,
                in the form of (x1, y1, x2, y2).
            mlvl_points (list[Tensor]): Points of each fpn level, each has
                shape (num_points, 2).
            num_imgs (int): the number of images in a batch.
        Returns:
            bbox_targets (list[Tensor]): Regression targets of each level in
                the form of (l, t, r, b).
        """
        # TODO: Re-implemented in Class PointCoder
        num_images = len(decoded_bboxes)
        bbox_targets = []
        assert len(decoded_bboxes[0]) == len(mlvl_points)

        for i in range(num_images):
            bbox_target = bbox2distance(mlvl_points, decoded_bboxes[i])
            bbox_targets.append(bbox_target)

        return bbox_targets

    def transform_bbox_targets(self, decoded_bboxes, mlvl_points, num_imgs):
        """Transform bbox_targets (x1, y1, x2, y2) into (l, t, r, b) format.
        Args:
            decoded_bboxes (list[Tensor]): Regression targets of each level,
                in the form of (x1, y1, x2, y2).
            mlvl_points (list[Tensor]): Points of each fpn level, each has
                shape (num_points, 2).
            num_imgs (int): the number of images in a batch.
        Returns:
            bbox_targets (list[Tensor]): Regression targets of each level in
                the form of (l, t, r, b).
        """
        # TODO: Re-implemented in Class PointCoder
        assert len(decoded_bboxes) == len(mlvl_points)
        num_levels = len(decoded_bboxes)
        mlvl_points = [points.repeat(num_imgs, 1) for points in mlvl_points]
        bbox_targets = []
        for i in range(num_levels):
            bbox_target = bbox2distance(mlvl_points[i], decoded_bboxes[i])
            bbox_targets.append(bbox_target)

        return bbox_targets

    def get_pos_loss(self,
                     cls_scores, bbox_preds, bbox_preds_refine,
                     labels, label_weights,
                     bbox_targets, bbox_weights,
                     all_level_points,
                     pos_inds):
        """Calculate loss of all potential positive samples obtained from first
        match process.
        Args:
            anchors (list[Tensor]): Anchors of each scale.
            cls_score (Tensor): Box scores of single image with shape
                (num_anchors, num_classes)
            bbox_pred (Tensor): Box energies / deltas of single image
                with shape (num_anchors, 4)
            label (Tensor): classification target of each anchor with
                shape (num_anchors,)
            label_weight (Tensor): Classification loss weight of each
                anchor with shape (num_anchors).
            bbox_target (dict): Regression target of each anchor with
                shape (num_anchors, 4).
            bbox_weight (Tensor): Bbox weight of each anchor with shape
                (num_anchors, 4).
            pos_inds (Tensor): Index of all positive samples got from
                first assign process.
        Returns:
            Tensor: Losses of all positive samples in single image.
        """
        num_pos = len(pos_inds)

        if self.sync_num_pos:
            num_pos_avg_per_gpu = reduce_mean(
                pos_inds.new_tensor(num_pos).float()).item()
            num_pos_avg_per_gpu = max(num_pos_avg_per_gpu, 1.0)
        else:
            num_pos_avg_per_gpu = num_pos

        # flatten_cls_scores = cls_scores.detach()
        # flatten_bbox_preds = bbox_preds.detach()
        # flatten_bbox_preds_refine = bbox_preds_refine.detach()
        #
        # flatten_labels = labels.detach()
        # flatten_labels_weights = label_weights.detach()
        # flatten_bbox_targets = bbox_targets.detach()
        # flatten_points = all_level_points.detach()

        pos_bbox_preds = bbox_preds[pos_inds]
        pos_bbox_preds_refine = bbox_preds_refine[pos_inds]

        pos_labels = labels[pos_inds]
        pos_bbox_targets = bbox_targets[pos_inds]
        pos_points = all_level_points[pos_inds]
        pos_scores = cls_scores[pos_inds]
        pos_labels_weight = label_weights[pos_inds]

        if not num_pos:
            return cls_scores.new([]),

        pos_decoded_bbox_preds = distance2bbox(pos_points, pos_bbox_preds)
        pos_decoded_target_preds = distance2bbox(pos_points, pos_bbox_targets)
        iou_targets_ini = bbox_overlaps(
            pos_decoded_bbox_preds,
            pos_decoded_target_preds.detach(),
            is_aligned=True).clamp(min=1e-6)
        bbox_weights_ini = iou_targets_ini.clone().detach()
        # iou_targets_ini_avg_per_gpu = reduce_mean(
        #     bbox_weights_ini.sum()).item()
        # bbox_avg_factor_ini = max(iou_targets_ini_avg_per_gpu, 1.0)
        bbox_avg_factor_ini = max(bbox_weights_ini.sum().item(), 1.0)

        loss_bbox = self.loss_bbox(
            pos_decoded_bbox_preds,
            pos_decoded_target_preds.detach(),
            weight=bbox_weights_ini,
            avg_factor=bbox_avg_factor_ini,
            reduction_override='none')

        pos_decoded_bbox_preds_refine = \
            distance2bbox(pos_points, pos_bbox_preds_refine)
        iou_targets_rf = bbox_overlaps(
            pos_decoded_bbox_preds_refine,
            pos_decoded_target_preds.detach(),
            is_aligned=True).clamp(min=1e-6)
        bbox_weights_rf = iou_targets_rf.clone().detach()
        # iou_targets_rf_avg_per_gpu = reduce_mean(
        #     bbox_weights_rf.sum()).item()
        iou_targets_rf_avg_per_gpu = bbox_weights_rf.sum().item()
        bbox_avg_factor_rf = max(iou_targets_rf_avg_per_gpu, 1.0)

        loss_bbox_refine = self.loss_bbox_refine(
            pos_decoded_bbox_preds_refine,
            pos_decoded_target_preds.detach(),
            weight=bbox_weights_rf,
            avg_factor=bbox_avg_factor_rf,
            reduction_override='none')

        # build IoU-aware cls_score targets
        if self.use_vfl:
            pos_ious = iou_targets_rf.clone().detach()
            cls_iou_targets = torch.zeros_like(cls_scores)
            cls_iou_targets[pos_inds, pos_labels] = pos_ious

        # if self.use_vfl:
        #     target_iou = torch.zeros_like(pos_scores)
        #     target_iou[:, pos_labels] = pos_ious
        #     loss_cls = self.loss_cls(
        #         pos_scores,
        #         target_iou,
        #         avg_factor=num_pos_avg_per_gpu,
        #         reduction_override='none')
    # else:
        loss_cls = self.paa_assign_cls_loss(
            pos_scores,
            pos_labels,
            weight=pos_labels_weight,
            avg_factor=num_pos_avg_per_gpu,
            reduction_override='none')

    # loss_cls = loss_cls.sum(-1)
        pos_loss = loss_bbox + loss_bbox_refine + loss_cls.sum(-1)
        return pos_loss,
        # return 0,

    def paa_reassign(self, pos_losses, label, label_weight, bbox_weight,
                     pos_inds, pos_gt_inds, anchors):
        """Fit loss to GMM distribution and separate positive, ignore, negative
        samples again with GMM model.
        Args:
            pos_losses (Tensor): Losses of all positive samples in
                single image.
            label (Tensor): classification target of each anchor with
                shape (num_anchors,)
            label_weight (Tensor): Classification loss weight of each
                anchor with shape (num_anchors).
            bbox_weight (Tensor): Bbox weight of each anchor with shape
                (num_anchors, 4).
            pos_inds (Tensor): Index of all positive samples got from
                first assign process.
            pos_gt_inds (Tensor): Gt_index of all positive samples got
                from first assign process.
            anchors (list[Tensor]): Anchors of each scale.
        Returns:
            tuple: Usually returns a tuple containing learning targets.
                - label (Tensor): classification target of each anchor after
                  paa assign, with shape (num_anchors,)
                - label_weight (Tensor): Classification loss weight of each
                  anchor after paa assign, with shape (num_anchors).
                - bbox_weight (Tensor): Bbox weight of each anchor with shape
                  (num_anchors, 4).
                - num_pos (int): The number of positive samples after paa
                  assign.
        """
        if not len(pos_inds):
            return label, label_weight, bbox_weight, 0

        num_gt = pos_gt_inds.max() + 1
        num_level = len(anchors)
        num_anchors_each_level = [item.size(0) for item in anchors]
        num_anchors_each_level.insert(0, 0)
        inds_level_interval = np.cumsum(num_anchors_each_level)
        pos_level_mask = []
        for i in range(num_level):
            mask = (pos_inds >= inds_level_interval[i]) & (
                pos_inds < inds_level_interval[i + 1]) #返回有效的pos_inds，并且将其拆解，分配到每个level上(维度不变，通true和false来表达当前level分配i情况)
            pos_level_mask.append(mask)
        pos_inds_after_paa = [label.new_tensor([])]  #用于存储重新分配后的pos_inds
        ignore_inds_after_paa = [label.new_tensor([])]
        for gt_ind in range(num_gt):
            pos_inds_gmm = []
            pos_loss_gmm = []
            gt_mask = pos_gt_inds == gt_ind  #逐个寻找符合要求的gt的mask
            for level in range(num_level):
                level_mask = pos_level_mask[level]
                level_gt_mask = level_mask & gt_mask #找到当前level，当前gt所对应的位置
                value, topk_inds = pos_losses[level_gt_mask].topk(
                    min(level_gt_mask.sum(), self.topk), largest=False)
                pos_inds_gmm.append(pos_inds[level_gt_mask][topk_inds]) #记录造成最大loss的几个pos_inds
                pos_loss_gmm.append(value)
            pos_inds_gmm = torch.cat(pos_inds_gmm)
            pos_loss_gmm = torch.cat(pos_loss_gmm)
            # fix gmm need at least two sample
            if len(pos_inds_gmm) < 2:
                continue
            device = pos_inds_gmm.device
            pos_loss_gmm, sort_inds = pos_loss_gmm.sort()
            pos_inds_gmm = pos_inds_gmm[sort_inds]
            pos_loss_gmm = pos_loss_gmm.view(-1, 1).cpu().numpy()
            min_loss, max_loss = pos_loss_gmm.min(), pos_loss_gmm.max()
            means_init = np.array([min_loss, max_loss]).reshape(2, 1)
            weights_init = np.array([0.5, 0.5])
            precisions_init = np.array([1.0, 1.0]).reshape(2, 1, 1)  # full
            # if self.covariance_type == 'spherical':
            #     precisions_init = precisions_init.reshape(2)
            # elif self.covariance_type == 'diag':
            #     precisions_init = precisions_init.reshape(2, 1)
            # elif self.covariance_type == 'tied':
            #     precisions_init = np.array([[1.0]])
            # if skm is None:
            #     raise ImportError('Please run "pip install sklearn" '
            #                       'to install sklearn first.')
            # gmm = skm.GaussianMixture(
            #     2,
            #     weights_init=weights_init,
            #     means_init=means_init,
            #     precisions_init=precisions_init,
            #     covariance_type=self.covariance_type)
            gmm = skm.GaussianMixture(
                2,
                weights_init=weights_init,
                means_init=means_init,
                precisions_init=precisions_init,
                covariance_type='full')
            gmm.fit(pos_loss_gmm)
            gmm_assignment = gmm.predict(pos_loss_gmm)
            scores = gmm.score_samples(pos_loss_gmm)
            gmm_assignment = torch.from_numpy(gmm_assignment).to(device)
            scores = torch.from_numpy(scores).to(device)
            pos_inds_temp, ignore_inds_temp = self.gmm_separation_scheme(
                gmm_assignment, scores, pos_inds_gmm) #ignore_inds_temp永远只返回空的？ gmm中0值为正样本，这一边相当于筛选正样本
            pos_inds_after_paa.append(pos_inds_temp)
            ignore_inds_after_paa.append(ignore_inds_temp)
        pos_inds_after_paa = torch.cat(pos_inds_after_paa)
        ignore_inds_after_paa = torch.cat(ignore_inds_after_paa)
        reassign_mask = (pos_inds.unsqueeze(1) != pos_inds_after_paa).all(1)
        reassign_ids = pos_inds[reassign_mask]        #需要被重新分配的id
        label[reassign_ids] = self.num_classes
        label_weight[ignore_inds_after_paa] = 0
        bbox_weight[reassign_ids] = 0
        num_pos = len(pos_inds_after_paa)
        return label, label_weight, bbox_weight, num_pos

    def gmm_separation_scheme(self, gmm_assignment, scores, pos_inds_gmm):
        """A general separation scheme for gmm model.
        It separates a GMM distribution of candidate samples into three
        parts, 0 1 and uncertain areas, and you can implement other
        separation schemes by rewriting this function.
        Args:
            gmm_assignment (Tensor): The prediction of GMM which is of shape
                (num_samples,). The 0/1 value indicates the distribution
                that each sample comes from.
            scores (Tensor): The probability of sample coming from the
                fit GMM distribution. The tensor is of shape (num_samples,).
            pos_inds_gmm (Tensor): All the indexes of samples which are used
                to fit GMM model. The tensor is of shape (num_samples,)
        Returns:
            tuple[Tensor]: The indices of positive and ignored samples.
                - pos_inds_temp (Tensor): Indices of positive samples.
                - ignore_inds_temp (Tensor): Indices of ignore samples.
        """
        # The implementation is (c) in Fig.3 in origin paper intead of (b).
        # You can refer to issues such as
        # https://github.com/kkhoot/PAA/issues/8 and
        # https://github.com/kkhoot/PAA/issues/9.
        fgs = gmm_assignment == 0
        pos_inds_temp = fgs.new_tensor([], dtype=torch.long)
        ignore_inds_temp = fgs.new_tensor([], dtype=torch.long)
        if fgs.nonzero().numel():
            _, pos_thr_ind = scores[fgs].topk(1)
            pos_inds_temp = pos_inds_gmm[fgs][:pos_thr_ind + 1]
            ignore_inds_temp = pos_inds_gmm.new_tensor([])
        return pos_inds_temp, ignore_inds_temp



    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """Override the method in the parent class to avoid changing para's
        name."""
        pass