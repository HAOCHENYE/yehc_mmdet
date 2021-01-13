from argparse import ArgumentParser
import cv2
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import os


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
    for ImageName in ImageList:
        ImagePath = os.path.join(args.img, ImageName)
        Image = cv2.imread(ImagePath)
        # build the model from a config file and a checkpoint file
        # test a single image
        result = inference_detector(model, Image)
        # show the results
        show_result_pyplot(model, Image, result, score_thr=args.score_thr, out_file=os.path.join(args.out_file,ImageName))


if __name__ == '__main__':
    main()
