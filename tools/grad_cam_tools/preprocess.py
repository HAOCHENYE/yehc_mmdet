import torch
from mmdet.datasets.pipelines import Compose
import cv2

def pre_process(image, cfg):
    data = dict(img=image)
    cfg = cfg.copy()
    # set loading pipeline type
    cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'
    test_pipeline = Compose(cfg.data.test.pipeline)
    data = test_pipeline(data)

    input_image = torch.stack(data['img'], dim=0)
    img_metas = data['img_metas'][0].data
    scale_factor = img_metas['scale_factor']
    ori_shape = (img_metas['ori_shape'][1], img_metas['ori_shape'][0])

    return input_image, {"scale_factor":scale_factor, "ori_shape":ori_shape}


