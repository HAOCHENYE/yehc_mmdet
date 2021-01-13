import torch.nn as nn
import torch.nn.functional as F
import torch

SCORE_THRESH = 0.3
STRIDE_SCALE = 8
IOU_THRESH = 0.6

class Integral(nn.Module):
    """A fixed layer for calculating integral result from distribution.

    This layer calculates the target location by :math: `sum{P(y_i) * y_i}`,
    P(y_i) denotes the softmax vector that represents the discrete distribution
    y_i denotes the discrete set, usually {0, 1, 2, ..., reg_max}

    Args:
        reg_max (int): The maximal value of the discrete set. Default: 16. You
            may want to reset it according to your new dataset or related
            settings.
    """

    def __init__(self, reg_max=16):
        super(Integral, self).__init__()
        self.reg_max = reg_max
        self.register_buffer('project',
                             torch.linspace(0, self.reg_max, self.reg_max + 1))

    def forward(self, x):
        """Forward feature from the regression head to get integral result of
        bounding box location.

        Args:
            x (Tensor): Features of the regression head, shape (N, 4*(n+1)),
                n is self.reg_max.

        Returns:
            x (Tensor): Integral result of box locations, i.e., distance
                offsets from the box center in four directions, shape (N, 4).
        """
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        x = F.linear(x, self.project.type_as(x)).reshape(-1, 4)
        return x



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


def gfl_post_process(output, extra_info):
    integral = Integral(16)
    ml_scores, ml_bboxes = output
    scale_factor = extra_info["scale_factor"]
    levels = 5
    total_bboxes = []

    for level in range(levels):
        stride = 2**(level)*8
        '''默认输出顺序为 小stride->大stride'''
        feat_h, feat_w = ml_scores[level].shape[2:]
        scores = ml_scores[level].permute(0, 2, 3, 1).view(feat_h*feat_w, 1).sigmoid()
        bboxes = integral(ml_bboxes[level].permute(0, 2, 3, 1))*stride

        for i in range(len(scores)):
            if scores[i] > SCORE_THRESH:
                x = i % int(feat_w) * stride
                y = i // int(feat_w) * stride
                x1 = x - bboxes[i][0]
                y1 = y - bboxes[i][1]
                x2 = x + bboxes[i][2]
                y2 = y + bboxes[i][3]
                score_loc = scores[i]
                box = torch.stack([x1, y1, x2, y2], dim=0)/torch.tensor(scale_factor)
                total_bboxes.append(torch.cat([box, score_loc], dim=0))
    nmsBoxes = nms(total_bboxes)
    return nmsBoxes
