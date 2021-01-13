import torch

SCORE_THRESH = 0.25
STRIDE_SCALE = 6
IOU_THRESH = 0.6

def IouCal(Box1, Box2):
    inner_x1 = torch.max(Box1[0], Box2[0])
    inner_y1 = torch.max(Box1[1], Box2[1])
    inner_x2 = torch.min(Box1[2], Box2[2])
    inner_y2 = torch.min(Box1[3], Box2[3])
    area_inner = (inner_x2 - inner_x1) * (inner_y2 - inner_y1)
    area = (Box2[2] - Box2[0]) * (Box2[3] - Box2[1]) + \
           (Box1[2] - Box1[0]) * (Box1[3] - Box1[1]) - \
           area_inner
    return torch.max(torch.tensor(0.), area_inner / area)

def nms(Bboxes):
    Bboxes = sorted(Bboxes, key=lambda x:x[4], reverse=True)
    record_dict = set()
    res = []
    for i in range(len(Bboxes)):
        if i not in record_dict:
            record_dict.add(i)
            res.append(Bboxes[i])
        else:
            continue
        for j in range(i + 1, len(Bboxes)):
            Iou = IouCal(Bboxes[i], Bboxes[j])
            if Iou > IOU_THRESH:
                record_dict.add(j)
                continue
    return res

def atss_post_process(output, extra_info):
    ml_scores, ml_bboxes, ml_centerness = output
    scale_factor = extra_info["scale_factor"]
    levels = 5
    total_bboxes = []
    for level in range(levels):
        stride = 2**(level)*8
        '''默认输出顺序为 小stride->大stride'''
        AnchorSize = stride*STRIDE_SCALE
        feat_h, feat_w = ml_scores[level].shape[2:]
        scores = ml_scores[level].permute(0, 2, 3, 1).view(feat_h*feat_w, 1).sigmoid()
        bboxes = ml_bboxes[level].permute(0, 2, 3, 1).view(feat_h*feat_w, 4)
        centerness = ml_centerness[level].permute(0, 2, 3, 1).view(feat_h*feat_w, 1).sigmoid()

        for i in range(len(scores)):
            if scores[i] > SCORE_THRESH:
                x = i % int(feat_w) * stride
                y = i // int(feat_w) * stride
                dx = bboxes[i][0] * AnchorSize / 10
                dy = bboxes[i][1] * AnchorSize / 10
                w = torch.exp(bboxes[i][2] / 5) * AnchorSize
                h = torch.exp(bboxes[i][3] / 5) * AnchorSize
                x1 = x + dx - w / 2
                y1 = y + dy - h / 2
                x2 = x + dx + w / 2
                y2 = y + dy + h / 2
                score_loc = centerness[i]*scores[i]
                box = torch.stack([x1, y1, x2, y2], dim=0)/torch.tensor(scale_factor)
                total_bboxes.append(torch.cat([box, score_loc], dim=0))
    nmsBoxes = nms(total_bboxes)
    return nmsBoxes
