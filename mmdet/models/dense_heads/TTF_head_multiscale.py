import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init, kaiming_init
import numpy as np
from mmcv.cnn import Scale as Scale

from mmcv.ops import ModulatedDeformConv2dPack as ModulatedDeformConvPack
from mmdet.core import multi_apply, calc_region
from mmcv.runner import force_fp32
from mmdet.models.losses import ct_focal_loss, giou_loss_ct
from mmcv.cnn import (build_norm_layer, bias_init_with_prob, ConvModule)


from .anchor_head import AnchorHead
from ..builder import HEADS, build_loss


@HEADS.register_module()
class TTFHead_multi(AnchorHead):

    def __init__(self,
                 inplanes=(64, 128, 256, 512),
                 planes=(256, 128, 64),
                 use_dla=False,
                 base_down_ratio=32,
                 head_conv=256,
                 wh_conv=64,
                 hm_head_conv_num=2,
                 wh_head_conv_num=2,
                 num_classes=81,
                 shortcut_kernel=3,
                 norm_cfg=dict(type='BN'),
                 shortcut_cfg=(1, 2, 3),
                 wh_offset_base=16.,
                 wh_area_process='log',
                 wh_agnostic=True,
                 wh_gaussian=True,
                 alpha=0.54,
                 beta=0.54,
                 hm_weight=1.,
                 wh_weight=5.,
                 max_objs=128):
        super(AnchorHead, self).__init__()
        assert len(planes) in [2, 3, 4]
        assert wh_area_process in [None, 'norm', 'log', 'sqrt']

        self.planes = planes
        self.use_dla = use_dla
        self.head_conv = head_conv
        self.num_classes = num_classes
        self.wh_offset_base = wh_offset_base
        self.wh_area_process = wh_area_process
        self.wh_agnostic = wh_agnostic
        self.wh_gaussian = wh_gaussian
        self.alpha = alpha
        self.beta = beta
        self.hm_weight = hm_weight
        self.wh_weight = wh_weight
        self.max_objs = max_objs
        self.fp16_enabled = False


        self.down_ratio = base_down_ratio // 2 ** len(planes)
        self.num_fg = num_classes
        self.wh_planes = 4 if wh_agnostic else 4 * self.num_fg
        self.base_loc = None
        self.dynamic_wh_weight1 = nn.parameter.Parameter(torch.ones(1)*(16))
        self.dynamic_wh_weight2 = nn.parameter.Parameter(torch.ones(1)*(32))
        # repeat upsampling n times. 32x to 4x by default.
        self.threshsize = 128/8
        if not self.use_dla:
            shortcut_num = min(len(inplanes) - 1, len(planes))
            assert shortcut_num == len(shortcut_cfg)
            self.deconv_layers = nn.ModuleList([
                self.build_upsample(inplanes[-1], planes[0], norm_cfg=norm_cfg),
                self.build_upsample(planes[0], planes[1], norm_cfg=norm_cfg)
            ])
            for i in range(2, len(planes)):
                self.deconv_layers.append(
                    self.build_upsample(planes[i - 1], planes[i], norm_cfg=norm_cfg))

            padding = (shortcut_kernel - 1) // 2
            self.shortcut_layers = self.build_shortcut(
                inplanes[:-1][::-1][:shortcut_num], planes[:shortcut_num], shortcut_cfg,
                kernel_size=shortcut_kernel, padding=padding)

        # heads
        self.wh_large = self.build_head(self.wh_planes, 2, wh_conv)
        self.wh_little = self.build_head(self.wh_planes, 1, wh_conv)
        self.hm_large = self.build_head(self.num_fg, 2, hm_head_conv_num)
        self.hm_little = self.build_head(self.num_fg, 1, hm_head_conv_num)
    def build_shortcut(self,
                       inplanes,
                       planes,
                       shortcut_cfg,
                       kernel_size=3,
                       padding=1):
        assert len(inplanes) == len(planes) == len(shortcut_cfg)

        shortcut_layers = nn.ModuleList()
        for (inp, outp, layer_num) in zip(
                inplanes, planes, shortcut_cfg):
            assert layer_num > 0
            layer = ShortcutConv2d(
                inp, outp, [kernel_size] * layer_num, [padding] * layer_num)
            shortcut_layers.append(layer)
        return shortcut_layers

    def build_upsample(self, inplanes, planes, norm_cfg=None):
        # mdcn = ModulatedDeformConvPack(inplanes, planes, 3, stride=1,
        #                                padding=1, dilation=1, deformable_groups=1)
        mdcn = nn.Conv2d(inplanes, planes, 3, stride=1,
                                       padding=1, dilation=1)
        up = nn.Upsample(scale_factor=2)

        layers = []
        layers.append(mdcn)
        if norm_cfg:
            layers.append(build_norm_layer(norm_cfg, planes)[1])
        layers.append(nn.ReLU(inplace=True))
        layers.append(up)

        return nn.Sequential(*layers)

    def build_head(self, out_channel, conv_num=1, head_conv_plane=None):  #head_conv_plane=64 conv_num=2 out_channel=4
        inp = self.planes[-1]
        head_convs = []
        head_conv_plane = self.head_conv if not head_conv_plane else head_conv_plane
        head_convs.append(ConvModule(inp, head_conv_plane, 1, padding=0))
        for i in range(conv_num):
            head_convs.append(ConvModule(head_conv_plane, head_conv_plane, 3, groups=head_conv_plane, padding=1))
        head_convs.append(nn.Conv2d(head_conv_plane, out_channel, 3, padding=1, groups=1))
        return nn.Sequential(*head_convs)

    def init_weights(self):
        if not self.use_dla:
            for _, m in self.shortcut_layers.named_modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)

        if not self.use_dla:
            for _, m in self.deconv_layers.named_modules():
                if isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        for _, m in self.hm_little.named_modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        for _, m in self.hm_large.named_modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)

        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.hm_large[-1], std=0.01, bias=bias_cls)
        normal_init(self.hm_little[-1], std=0.01, bias=bias_cls)
        for _, m in self.wh_large.named_modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)

        for _, m in self.wh_little.named_modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)

    def forward(self, feats):
        """

        Args:
            feats: list(tensor).

        Returns:
            hm: tensor, (batch, 80, h, w).
            wh: tensor, (batch, 4, h, w) or (batch, 80 * 4, h, w).
        """
        x = feats[-1]
        if not self.use_dla:
            for i, upsample_layer in enumerate(self.deconv_layers):
                x = upsample_layer(x)
                if i < len(self.shortcut_layers):
                    shortcut = self.shortcut_layers[i](feats[-i - 2])
                    x = x + shortcut


        wh_large = self.dynamic_wh_weight1*F.relu(self.wh_large(x))
        wh_little = self.dynamic_wh_weight2*F.relu(self.wh_little(x))

        hm_large = self.hm_large(x)
        hm_litte = self.hm_little(x)
        # wh = F.relu(self.wh(x))


        return hm_large, hm_litte, wh_large, wh_little

    @force_fp32(apply_to=('pred_heatmap', 'pred_wh'))
    def get_bboxes(self,
                   pred_heatmap_large,
                   pred_heatmap_little,
                   pred_wh_large,
                   pred_wh_little,
                   img_metas,
                   cfg=None,
                   rescale=False):
        score_thr = getattr(cfg, 'score_thr', 0.01)
        result_list = []
        bboxes_large, labels_large = self.get_result(pred_heatmap_large, pred_wh_large, score_thr, img_metas, rescale)
        bboxes_little, labels_little = self.get_result(pred_heatmap_little, pred_wh_little, score_thr, img_metas, rescale)
        bboxes = torch.cat((bboxes_large, bboxes_little), dim=0)
        labels = torch.cat((labels_large, labels_little), dim=0)
        result_list.append((bboxes, labels))
        return result_list

    @force_fp32(apply_to=('pred_heatmap', 'pred_wh'))
    def loss(self,
             pred_heatmap_large,
             pred_heatmap_little,
             pred_wh_large,
             pred_wh_little,
             gt_bboxes,
             gt_labels,
             img_metas,
             cfg=None,
             gt_bboxes_ignore=None):
        all_targets = self.target_generator(gt_bboxes, gt_labels, img_metas)
        hm_loss, wh_loss = self.loss_calc(pred_heatmap_large,pred_heatmap_little, pred_wh_large,pred_wh_little, *all_targets)
        return {'losses/ttfnet_loss_heatmap': hm_loss, 'losses/ttfnet_loss_wh': wh_loss}

    def _topk(self, scores, topk):
        batch, cat, height, width = scores.size()

        # both are (batch, 80, topk)
        topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), topk)

        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds / width).int().float()
        topk_xs = (topk_inds % width).int().float()

        # both are (batch, topk). select topk from 80*topk
        topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), topk)
        topk_clses = (topk_ind / topk).int()
        topk_ind = topk_ind.unsqueeze(2)
        topk_inds = topk_inds.view(batch, -1, 1).gather(1, topk_ind).view(batch, topk)
        topk_ys = topk_ys.view(batch, -1, 1).gather(1, topk_ind).view(batch, topk)
        topk_xs = topk_xs.view(batch, -1, 1).gather(1, topk_ind).view(batch, topk)

        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

    def gaussian_2d(self, shape, sigma_x=1, sigma_y=1):
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]

        h = np.exp(-(x * x / (2 * sigma_x * sigma_x) + y * y / (2 * sigma_y * sigma_y)))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return h

    def draw_truncate_gaussian(self, heatmap, center, h_radius, w_radius, k=1):
        h, w = 2 * h_radius + 1, 2 * w_radius + 1
        sigma_x = w / 6
        sigma_y = h / 6
        gaussian = self.gaussian_2d((h, w), sigma_x=sigma_x, sigma_y=sigma_y)
        gaussian = heatmap.new_tensor(gaussian)

        x, y = int(center[0]), int(center[1])

        height, width = heatmap.shape[0:2]

        left, right = min(x, w_radius), min(width - x, w_radius + 1)
        top, bottom = min(y, h_radius), min(height - y, h_radius + 1)

        masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[h_radius - top:h_radius + bottom,
                          w_radius - left:w_radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
            torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
        return heatmap

    def bbox_areas(self, bboxes, keep_axis=False):
        x_min, y_min, x_max, y_max = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
        areas = (y_max - y_min + 1) * (x_max - x_min + 1)
        if keep_axis:
            return areas[:, None]
        return areas

    def target_single_image(self, gt_boxes, gt_labels, feat_shape):
        """

        Args:
            gt_boxes: tensor, tensor <=> img, (num_gt, 4).
            gt_labels: tensor, tensor <=> img, (num_gt,).
            feat_shape: tuple.

        Returns:
            heatmap: tensor, tensor <=> img, (80, h, w).
            box_target: tensor, tensor <=> img, (4, h, w) or (80 * 4, h, w).
            reg_weight: tensor, same as box_target
        """
        output_h, output_w = feat_shape
        heatmap_channel = self.num_fg

        heatmap_large = gt_boxes.new_zeros((heatmap_channel, output_h, output_w))
        heatmap_little = gt_boxes.new_zeros((heatmap_channel, output_h, output_w))
        fake_heatmap = gt_boxes.new_zeros((output_h, output_w))
        box_target_large = gt_boxes.new_ones((self.wh_planes, output_h, output_w)) * -1
        box_target_little = gt_boxes.new_ones((self.wh_planes, output_h, output_w)) * -1
        reg_weight_little = gt_boxes.new_zeros((self.wh_planes // 4, output_h, output_w))
        reg_weight_large = gt_boxes.new_zeros((self.wh_planes // 4, output_h, output_w))

        if self.wh_area_process == 'log':
            boxes_areas_log = self.bbox_areas(gt_boxes).log()
        elif self.wh_area_process == 'sqrt':
            boxes_areas_log = self.bbox_areas(gt_boxes).sqrt()
        else:
            boxes_areas_log = self.bbox_areas(gt_boxes)
        boxes_area_topk_log, boxes_ind = torch.topk(boxes_areas_log, boxes_areas_log.size(0))

        if self.wh_area_process == 'norm':
            boxes_area_topk_log[:] = 1.

        gt_boxes = gt_boxes[boxes_ind]
        gt_labels = gt_labels[boxes_ind]

        feat_gt_boxes = gt_boxes / self.down_ratio
        feat_gt_boxes[:, [0, 2]] = torch.clamp(feat_gt_boxes[:, [0, 2]], min=0,
                                               max=output_w - 1)
        feat_gt_boxes[:, [1, 3]] = torch.clamp(feat_gt_boxes[:, [1, 3]], min=0,
                                               max=output_h - 1)
        feat_hs, feat_ws = (feat_gt_boxes[:, 3] - feat_gt_boxes[:, 1],
                            feat_gt_boxes[:, 2] - feat_gt_boxes[:, 0])

        # we calc the center and ignore area based on the gt-boxes of the origin scale
        # no peak will fall between pixels
        thresh_size = torch.min(feat_hs,feat_ws)
        mask_large = thresh_size>=(self.threshsize)
        mask_little = thresh_size<(self.threshsize)

        # print("large size is {}".format(thresh_size[mask_large].shape[0]))
        # print("little size is {}".format(thresh_size[mask_little].shape[0]))
        # we calc the center and ignore area based on the gt-boxes of the origin scale
        # no peak will fall between pixels

        ct_ints_large = (torch.stack([(torch.masked_select(gt_boxes[:, 0], mask_large) +
                                       torch.masked_select(gt_boxes[:, 2], mask_large)) / 2,
                                      (torch.masked_select(gt_boxes[:, 1], mask_large) +
                                       torch.masked_select(gt_boxes[:, 3], mask_large)) / 2
                                      ],
                               dim=1) / self.down_ratio).to(torch.int)

        ct_ints_little = (torch.stack([(torch.masked_select(gt_boxes[:, 0], mask_little) +
                                       torch.masked_select(gt_boxes[:, 2], mask_little)) / 2,
                                      (torch.masked_select(gt_boxes[:, 1], mask_little) +
                                       torch.masked_select(gt_boxes[:, 3], mask_little)) / 2
                                      ],
                                      dim=1) / self.down_ratio).to(torch.int)

        mask_large_gt = torch.stack([mask_large  for _ in range(4)], dim=1)
        mask_little_gt = torch.stack([mask_little for _ in range(4)], dim=1)
        gt_boxes_large  = torch.masked_select(gt_boxes, mask_large_gt).view(-1, 4)
        gt_boxes_little = torch.masked_select(gt_boxes, mask_little_gt).view(-1, 4)

        h_radiuses_beta_large =  (torch.masked_select(feat_hs, mask_large) / 2. * self.beta).int()
        h_radiuses_beta_little = (torch.masked_select(feat_hs, mask_little) / 2. * self.beta).int()
        w_radiuses_beta_large =  (torch.masked_select(feat_ws, mask_large) / 2. * self.beta).int()
        w_radiuses_beta_little = (torch.masked_select(feat_ws, mask_little) / 2. * self.beta).int()


        for k in range(len(ct_ints_large)):
            cls_id = gt_labels[k] - 1
            fake_heatmap = fake_heatmap.zero_()
            self.draw_truncate_gaussian(fake_heatmap, ct_ints_large[k],
                                        h_radiuses_beta_large[k].item(), w_radiuses_beta_large[k].item())
            heatmap_large[cls_id] = torch.max(heatmap_large[cls_id], fake_heatmap)
            # larger boxes have lower priority than small boxes.
            if self.wh_gaussian:
                box_target_inds_large = fake_heatmap.gt(0)

            if self.wh_agnostic:
                box_target_large[:, box_target_inds_large] = gt_boxes_large[k][:, None]
                cls_id = 0
            #得分越高，损失越大
            if self.wh_gaussian:
                local_heatmap = fake_heatmap[box_target_inds_large]
                ct_div = local_heatmap.sum()
                local_heatmap *= boxes_area_topk_log[k]
                reg_weight_large[cls_id, box_target_inds_large] = local_heatmap / ct_div

        for k in range(len(ct_ints_little)):
            cls_id = gt_labels[k] - 1
            fake_heatmap = fake_heatmap.zero_()
            self.draw_truncate_gaussian(fake_heatmap, ct_ints_little[k],
                                        h_radiuses_beta_little[k].item(), w_radiuses_beta_little[k].item())
            heatmap_little[cls_id] = torch.max(heatmap_little[cls_id], fake_heatmap)
            # larger boxes have lower priority than small boxes.
            if self.wh_gaussian:
                box_target_inds_little = fake_heatmap.gt(0)

            if self.wh_agnostic:
                box_target_little[:, box_target_inds_little] = gt_boxes_little[k][:, None]
                cls_id = 0

            if self.wh_gaussian:
                local_heatmap = fake_heatmap[box_target_inds_little]
                ct_div = local_heatmap.sum()
                local_heatmap *= boxes_area_topk_log[k]
                reg_weight_little[cls_id, box_target_inds_little] = local_heatmap / ct_div

        return heatmap_large, heatmap_little, box_target_large, box_target_little, reg_weight_large, reg_weight_little

        # return heatmap, box_target, reg_weight

    def simple_nms(self, heat, kernel=3, out_heat=None):
        pad = (kernel - 1) // 2
        hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
        keep = (hmax == heat).float()
        out_heat = heat if out_heat is None else out_heat
        return out_heat * keep

    def target_generator(self, gt_boxes, gt_labels, img_metas):
        """

        Args:
            gt_boxes: list(tensor). tensor <=> image, (gt_num, 4).
            gt_labels: list(tensor). tensor <=> image, (gt_num,).
            img_metas: list(dict).

        Returns:
            heatmap: tensor, (batch, 80, h, w).
            box_target: tensor, (batch, 4, h, w) or (batch, 80 * 4, h, w).
            reg_weight: tensor, same as box_target.
        """
        with torch.no_grad():
            feat_shape = (img_metas[0]['pad_shape'][0] // self.down_ratio,
                          img_metas[0]['pad_shape'][1] // self.down_ratio)
            heatmap_large, heatmap_little, box_target_large, box_target_little, reg_weight_large, reg_weight_little = multi_apply(
                self.target_single_image,
                gt_boxes,
                gt_labels,
                feat_shape=feat_shape
            )

            heatmap_large, heatmap_little, box_target_large, box_target_little \
                = [torch.stack(t, dim=0).detach() for t in [heatmap_large, heatmap_little, box_target_large, box_target_little]]
            reg_weight_large = torch.stack(reg_weight_large, dim=0).detach()
            reg_weight_little = torch.stack(reg_weight_little, dim=0).detach()

            return heatmap_large, heatmap_little, box_target_large, box_target_little, reg_weight_large, reg_weight_little

    def loss_calc(self,
                  pred_hm_large,
                  pred_hm_little,
                  pred_wh_large,
                  pred_wh_little,
                  heatmap_large,
                  heatmap_little,
                  box_target_large,
                  box_target_little,
                  wh_weight_large,
                  wh_weight_little):
        """

        Args:
            pred_hm: tensor, (batch, 80, h, w).
            pred_wh: tensor, (batch, 4, h, w) or (batch, 80 * 4, h, w).
            heatmap: tensor, same as pred_hm.
            box_target: tensor, same as pred_wh.
            wh_weight: tensor, same as pred_wh.

        Returns:
            hm_loss
            wh_loss
        """
        H, W = pred_hm_little.shape[2:]
        pred_hm_large = torch.clamp(pred_hm_large.sigmoid_(), min=1e-4, max=1 - 1e-4)
        pred_hm_little = torch.clamp(pred_hm_little.sigmoid_(), min=1e-4, max=1 - 1e-4)
        hm_loss = 1*(ct_focal_loss(pred_hm_large, heatmap_large) * self.hm_weight+ct_focal_loss(pred_hm_little, heatmap_little)* self.hm_weight)

        mask_large = wh_weight_large.view(-1, H, W)
        mask_little = wh_weight_little.view(-1, H, W)
        avg_factor_large = mask_large.sum() + 1e-4
        avg_factor_little = mask_little.sum() +1e-4

        if self.base_loc is None or H != self.base_loc.shape[1] or W != self.base_loc.shape[2]:
            base_step = self.down_ratio
            shifts_x = torch.arange(0, (W - 1) * base_step + 1, base_step,
                                    dtype=torch.float32, device=heatmap_little.device)
            shifts_y = torch.arange(0, (H - 1) * base_step + 1, base_step,
                                    dtype=torch.float32, device=heatmap_little.device)
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            self.base_loc = torch.stack((shift_x, shift_y), dim=0)  # (2, h, w)

        # (batch, h, w, 4)

        pred_boxes_large = torch.cat((self.base_loc - pred_wh_large[:, [0, 1]],
                                self.base_loc + pred_wh_large[:, [2, 3]]), dim=1).permute(0, 2, 3, 1)

        pred_boxes_little = torch.cat((self.base_loc - pred_wh_little[:, [0, 1]],
                                self.base_loc + pred_wh_little[:, [2, 3]]), dim=1).permute(0, 2, 3, 1)

        # (batch, h, w, 4)
        boxes_large = box_target_large.permute(0, 2, 3, 1)
        boxes_little = box_target_little.permute(0, 2, 3, 1)
        wh_loss = giou_loss_ct(pred_boxes_large, boxes_large, mask_large, avg_factor=avg_factor_large) * self.wh_weight+ \
                  giou_loss_ct(pred_boxes_little, boxes_little, mask_little, avg_factor=avg_factor_little) * self.wh_weight

        return hm_loss, wh_loss

    def get_result(self, pred_heatmap, pred_wh, score_thr, img_metas, rescale):
        batch, cat, height, width = pred_heatmap.size()
        pred_heatmap = pred_heatmap.detach().sigmoid_()
        wh = pred_wh.detach()


        # perform nms on heatmaps
        heat = self.simple_nms(pred_heatmap)  # used maxpool to filter the max score

        topk = 100
        # (batch, topk)
        scores, inds, clses, ys, xs = self._topk(heat, topk=topk)
        xs = xs.view(batch, topk, 1) * self.down_ratio
        ys = ys.view(batch, topk, 1) * self.down_ratio

        wh = wh.permute(0, 2, 3, 1).contiguous()
        wh = wh.view(wh.size(0), -1, wh.size(3))


        inds = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), wh.size(2))
        wh = wh.gather(1, inds)

        if not self.wh_agnostic:
            wh = wh.view(-1, topk, self.num_fg, 4)
            wh = torch.gather(wh, 2, clses[..., None, None].expand(
                clses.size(0), clses.size(1), 1, 4).long())

        wh = wh.view(batch, topk, 4)
        clses = clses.view(batch, topk, 1).float()
        scores = scores.view(batch, topk, 1)

        bboxes = torch.cat([xs - wh[..., [0]], ys - wh[..., [1]],
                            xs + wh[..., [2]], ys + wh[..., [3]]], dim=2)
        for batch_i in range(bboxes.shape[0]):
            scores_per_img = scores[batch_i]
            scores_keep = (scores_per_img > score_thr).squeeze(-1)

            scores_per_img = scores_per_img[scores_keep]
            bboxes_per_img = bboxes[batch_i][scores_keep]
            labels_per_img = clses[batch_i][scores_keep]
            img_shape = img_metas[batch_i]['pad_shape']
            bboxes_per_img[:, 0::2] = bboxes_per_img[:, 0::2].clamp(min=0, max=img_shape[1] - 1)
            bboxes_per_img[:, 1::2] = bboxes_per_img[:, 1::2].clamp(min=0, max=img_shape[0] - 1)

            if rescale:
                scale_factor = img_metas[batch_i]['scale_factor']
                bboxes_per_img /= bboxes_per_img.new_tensor(scale_factor)

            bboxes_per_img = torch.cat([bboxes_per_img, scores_per_img], dim=1)
            labels_per_img = labels_per_img.squeeze(-1)
        return bboxes_per_img, labels_per_img

class ShortcutConv2d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_sizes,
                 paddings,
                 activation_last=False):
        super(ShortcutConv2d, self).__init__()
        assert len(kernel_sizes) == len(paddings)

        layers = []
        for i, (kernel_size, padding) in enumerate(zip(kernel_sizes, paddings)):
            inc = in_channels if i == 0 else out_channels
            layers.append(nn.Conv2d(inc, out_channels, kernel_size, padding=padding))
            if i < len(kernel_sizes) - 1 or activation_last:
                layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        y = self.layers(x)
        return y
