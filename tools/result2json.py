from argparse import ArgumentParser
import cv2
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import os
import torch
import numpy as np
import json

image_mean = np.array([127.5, 127.5, 127.5])
image_std = 128.0
fpn_strides = [8, 16, 32, 64, 128]
thresholds = 0.5

CONF_THRESH = [0.52, 0.54, 0.56, 0.60, 0.62]
# CONF_THRESH = [0.25, 0.25, 0.25, 0.25, 0.25]
SCORE_THRESH = 0.26
ANCHNOR_COF = 6
NMS_THRESH = 0.3
NUM_CLASS = 3
CLASS_CAT = {0: "person", 1: "botlle", 2: 'chair', 3:'potted plant'}
COCO_ID = {"person":1, "botlle":44, 'chair':62, 'potted plant':64}
def nms(Bboxes):
    Bboxes = sorted(Bboxes, key=lambda x:x[4], reverse=True)
    record_dict = set()
    res = []
    for i in range(len(Bboxes)):
        if Bboxes[i][4] < SCORE_THRESH:
            continue
        if i not in record_dict:
            Area1 = (Bboxes[i][2] - Bboxes[i][0]) * (Bboxes[i][3] - Bboxes[i][1])
            record_dict.add(i)
            res.append(Bboxes[i])
        else:
            continue
        for j in range(i + 1, len(Bboxes)):
            Area2 = (Bboxes[j][2] - Bboxes[j][0]) * (Bboxes[j][3] - Bboxes[j][1])
            inner_x1 = max(Bboxes[i][0], Bboxes[j][0])
            inner_y1 = max(Bboxes[i][1], Bboxes[j][1])
            inner_x2 = min(Bboxes[i][2], Bboxes[j][2])
            inner_y2 = min(Bboxes[i][3], Bboxes[j][3])

            inner_w = inner_x2 - inner_x1
            inner_h = inner_y2 - inner_y1

            if inner_h <= 0 or inner_w <= 0:
                area_inner = 0
            else:
                area_inner = (inner_x2 - inner_x1) * (inner_y2 - inner_y1)

            area = Area1 + Area2 - area_inner
            Iou = area_inner / area
            Iof = area_inner / min(Area1, Area2)
            if Iou > NMS_THRESH:
                record_dict.add(j)
            if abs(Bboxes[i][1] - Bboxes[j][1]) < Area1 ** 0.5 * 0.05 and Area2 < Area1 and Iof > 0.95:
                record_dict.add(j)

    return res

def DumpJson(results):
    pass



def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--out_file', default='result.jpg', help='Test image')
    parser.add_argument(
        '--score-thr', type=float, default=0.25, help='bbox score threshold')
    args = parser.parse_args()

    model = init_detector(args.config, args.checkpoint, device=args.device)

    ImageList = os.listdir(args.img)

    ImageID = 0
    annID = 0
    ann_dict = {}
    images = []
    annotations = []

    ann_dict['categories'] = [{'supercategory': 'furniture', 'id': 64, 'name': 'potted plant'},
                              {'supercategory': 'kitchen', 'id': 44, 'name': 'bottle'},
                              {'supercategory': 'furniture', 'id': 62, 'name': 'chair'},
                              {'supercategory': 'person', 'id': 1, 'name': 'person'}]
    for ImageName in ImageList:
        image = {}
        print("Start solving {} images".format(ImageID))
        ImageID += 1

        ImagePath = os.path.join(args.img, ImageName)
        Image = cv2.imread(ImagePath)
        # build the model from a config file and a checkpoint file
        # test a single image
        results = inference_detector(model, Image)
        # show the results
        nms_result = [np.array(nms(result)).reshape(-1, 5) for result in results]

        image['id'] = ImageID
        image['width'] = Image.shape[1]
        image['height'] = Image.shape[0]
        image['file_name'] = ImageName
        images.append(image)

        for cls in range(len(nms_result)):
            for bbox in nms_result[cls]:
                ann = {}
                bbox = bbox.astype(int)
                annID += 1
                ann['id'] = annID
                ann['image_id'] = ImageID
                ann['segmentation'] = []
                ann['category_id'] = 1
                ann['iscrowd'] = 0
                ann['category_id'] = COCO_ID[CLASS_CAT[cls]]
                ann['bbox'] = [int(bbox[0]), int(bbox[1]), int(bbox[2]-bbox[0]), int(bbox[3]-bbox[1])]
                ann['area'] = int(bbox[2] - bbox[0]) * int(bbox[3] - bbox[1])
                annotations.append(ann)

    ann_dict['images'] = images
    json_file = os.path.join(args.out_file, "CrowdPoseFakeLabel.json")
    ann_dict['annotations'] = annotations
    with open(json_file, 'w', encoding='utf8') as outfile:
        json.dump(ann_dict, outfile, indent=4, sort_keys=True)
        # show_result_pyplot(model, Image, nms_result, score_thr=args.score_thr, out_file=os.path.join(args.out_file,ImageName))




if __name__ == '__main__':
    main()
