import numpy as np
import cv2
import mmcv
from mmcv.runner import load_checkpoint
from functools import partial
import argparse
import torch
import torch.nn as nn
import os
from mmdet.models import build_detector
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
from mmdet.datasets.pipelines import Compose
import warnings
from grad_cam_tools import *




def SimplifyModel(model, img_metas):
    model.cpu().eval()
    model.forward = partial(
        model.forward, img_metas=img_metas, return_loss="grad_cam")


class GuidedBackPropagation(object):

    def __init__(self, net):
        self.net = net
        for (name, module) in self.net.named_modules():
            if isinstance(module, nn.ReLU):
                module.register_backward_hook(self.backward_hook)
        self.net.eval()

    @classmethod
    def backward_hook(cls, module, grad_in, grad_out):
        """
        :param module:
        :param grad_in: tuple,长度为1
        :param grad_out: tuple,长度为1
        :return: tuple(new_grad_in,)
        """
        return torch.clamp(grad_in[0], min=0.0),

    def __call__(self, inputs, index=0):
        """
        :param inputs: {"image": [C,H,W], "height": height, "width": width}
        :param index: 第几个边框
        :return:
        """
        self.net.zero_grad()
        output = self.net.inference([inputs])
        score = output[0]['instances'].scores[index]
        score.backward()

        return inputs['image'].grad  # [3,H,W]

class GradCAM(object):
    """
    1: 网络不更新梯度,输入需要梯度更新
    2: 使用目标类别的得分做反向传播
    """

    def __init__(self, net, layer_name, output_file, image, pre_process,
                 post_process):
        self.layer_name = layer_name
        self.image = image
        '''for mmdet'''
        self.cfg = net.cfg
        self.data = ImagePreprocess(image, self.cfg)
        SimplifyModel(net, self.data['img_metas'])
        '''for mmdet'''
        self.post_process_func = post_process
        self.pre_process_func = pre_process
        self.output_file = output_file
        self.net = net
        self.feature = {}
        self.gradient = {}
        self.net.eval()
        self.handlers = []
        self._register_hook()

    def _get_features_hook(self, module, input, output, layer_name=None):

        try:
            self.feature[layer_name] = output  # 不同层级特征
            print("Layer:{} feature shape:{}".format(layer_name, output.size()))
        except:
            print("Layer:{} feature is a list !".format(layer_name))
            self.layer_name.remove(layer_name)

    def _get_grads_hook(self, module, input_grad, output_grad, layer_name=None):
        """
        :param input_grad: tuple, input_grad[0]: None
                                   input_grad[1]: weight
                                   input_grad[2]: bias
        :param output_grad:tuple,长度为1
        :return:
        """

        try:
            self.gradient[layer_name] = output_grad[0]  # 梯度的顺序反的
            print("Layer:{} gradient shape:{}".format(layer_name, output_grad[0].size()))
        except:
            print("Layer:{} feature is a list !".format(layer_name))
            self.layer_name.remove(layer_name)

    def _register_hook(self):
        for (name, module) in self.net.named_modules():
            if name in self.layer_name:
                func_forward = partial(self._get_features_hook, layer_name=name)
                func_backward = partial(self._get_grads_hook, layer_name=name)
                self.handlers.append(module.register_forward_hook(func_forward))
                self.handlers.append(module.register_backward_hook(func_backward))

    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()

    def save_heat_map(self, heatmap, nmsBboxes, layerName):
        heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        show_image = heatmap + self.image / 255
        show_image = show_image / show_image.max()
        drawBboxes(nmsBboxes, show_image)
        cv2.imwrite(os.path.join(self.output_file, "{}.jpg".format(layerName)), show_image * 255)

    def pre_process(self, *args, **kwargs):
        self.data, self.extra_info = self.pre_process_func(*args, **kwargs)
        assert "ori_shape" in self.extra_info.keys(), "ori_shape ,must in extra info !"
        setattr(self, "ori_shape", self.extra_info["ori_shape"])

    def inference(self, data):
        output = self.net([data])
        return output

    def single_cam(self, gradient, feature, nmsBboxes, layer_name):
        '''

        :gradient shape：[C,H W]:
        :param feature： [C H W]:
        :param nmsBboxes [N 5](x1, y1, x2, y2):
        :param layer_name: str
        :return:
        '''
        weight = np.mean(gradient, axis=(1, 2))  # H W
        cam = np.zeros_like(feature[1]).astype(np.float32)
        for i, w in enumerate(weight):
            cam += w * feature[i]
        cam = np.maximum(cam, 0)  # ReLU
        # 数值归一化
        cam -= np.min(cam)
        cam /= np.max(cam)
        # 缩放到输入图像尺寸
        cam = cv2.resize(cam, self.ori_shape)
        self.save_heat_map(cam, nmsBboxes, layer_name)

    def single_cam_plus(self, cost, gradient, feature, nmsBboxes, layer_name):
        gradient = np.exp(cost)*np.power(gradient, 1)
        second_gradient = np.exp(cost)*np.power(gradient, 2)
        third_gradient = np.exp(cost)*np.power(gradient, 3)



        alpha_num = second_gradient
        alpha_denom = 2*second_gradient + feature*third_gradient.sum(axis=(1, 2), keepdims=True)
        alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, 1)
        alphas = alpha_num/(alpha_denom+1e-7)   #C H W

        weights = np.maximum(gradient, 0)
        alpha_normalization_constant = np.sum(alphas, axis=(1, 2))
        alphas /= alpha_normalization_constant.reshape((-1, 1, 1))
        alphas_thresholding = np.where(weights, alphas, 0.0) #C H W

        alpha_normalization_constant = np.sum(alphas_thresholding, axis=(1, 2))
        alpha_normalization_constant_processed = np.where(alpha_normalization_constant != 0.0,
                                                          alpha_normalization_constant,
                                                          np.ones(alpha_normalization_constant.shape))


        alphas /= alpha_normalization_constant_processed.reshape((-1, 1, 1))

        cam = np.zeros_like(feature[1]).astype(np.float32)
        weight = np.sum(weights * alphas, axis = (1,2))
        for i, w in enumerate(weight):
            cam += w * feature[i]
        cam = np.maximum(cam, 0)  # ReLU
        # 数值归一化
        cam -= np.min(cam)
        cam /= np.max(cam)
        # 缩放到输入图像尺寸
        cam = cv2.resize(cam, self.ori_shape)
        self.save_heat_map(cam, nmsBboxes, layer_name)


    def __call__(self):
        """
        :param inputs: {"image": [C,H,W], "height": height, "width": width}
        :param index: 第几个边框
        :return:
        """
        self.net.zero_grad()
        output = self.inference(self.data)
        nmsBoxes = self.post_process_func(output, self.extra_info)

        valid_score = torch.stack(nmsBoxes, dim=0)[:, 4].sum()
        valid_score.backward()

        for LayerName in self.gradient:
            try:
                if not LayerName:
                    continue
                gradient = self.gradient[LayerName].cpu().data.numpy().squeeze(axis=0)
                feature = self.feature[LayerName].cpu().data.numpy().squeeze(axis=0)  # [C,H,W]
                if gradient.shape != feature.shape:
                    # raise TypeError("{} is a multi-scale shared head".format(self.layer_name))
                    warnings.warn("{} is a multi-scale shared head".format(self.layer_name))
                    continue

                # self.single_cam(gradient, feature, nmsBoxes, LayerName)
                self.single_cam_plus(1, gradient, feature, nmsBoxes, LayerName)
            except:
                print(LayerName)




class GradCamPlusPlus(GradCAM):
    def __init__(self, net, layer_name):
        super(GradCamPlusPlus, self).__init__(net, layer_name)

    def __call__(self, inputs, index=0):
        """
        :param inputs: {"image": [C,H,W], "height": height, "width": width}
        :param index: 第几个边框
        :return:
        """
        self.net.zero_grad()
        output = self.net.predict([inputs])
        print(output)
        score = output[0]['instances'].scores[index]
        feature_level = output[0]['instances'].feature_levels[index]  # box来自第几层feature map
        score.backward()

        gradient = self.gradient[feature_level][0].cpu().data.numpy()  # [C,H,W]
        gradient = np.maximum(gradient, 0.)  # ReLU
        indicate = np.where(gradient > 0, 1., 0.)  # 示性函数
        norm_factor = np.sum(gradient, axis=(1, 2))  # [C]归一化
        for i in range(len(norm_factor)):
            norm_factor[i] = 1. / norm_factor[i] if norm_factor[i] > 0. else 0.  # 避免除零
        alpha = indicate * norm_factor[:, np.newaxis, np.newaxis]  # [C,H,W]

        weight = np.sum(gradient * alpha, axis=(1, 2))  # [C]  alpha*ReLU(gradient)

        feature = self.feature[feature_level][0].cpu().data.numpy()  # [C,H,W]

        cam = feature * weight[:, np.newaxis, np.newaxis]  # [C,H,W]
        cam = np.sum(cam, axis=0)  # [H,W]
        # cam = np.maximum(cam, 0)  # ReLU

        # 数值归一化
        cam -= np.min(cam)
        cam /= np.max(cam)
        # 缩放到输入图像尺寸
        h, w = inputs['height'], inputs['width']
        cam = cv2.resize(cam, (w, h))

        return cam




def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert MMDetection models to ONNX')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--input-img', type=str, help='Images for input')
    parser.add_argument('--show', action='store_true', help='show onnx graph')
    parser.add_argument('--output-file', type=str, default='tmp.onnx')
    parser.add_argument('--opset-version', type=int, default=9)
    parser.add_argument(
        '--verify',
        action='store_true',
        help='verify the onnx model output against pytorch output')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[512, 512],
        help='input image size')
    parser.add_argument(
        '--mean',
        type=int,
        nargs='+',
        default=[128, 128, 128],
        help='mean value used for preprocess input data')
    parser.add_argument(
        '--std',
        type=int,
        nargs='+',
        default=[128, 128, 128],
        help='variance value used for preprocess input data')
    args = parser.parse_args()
    return args

def ImagePreprocess(image, cfg):
    data = dict(img=image)
    cfg = cfg.copy()
    # set loading pipeline type
    cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'
    test_pipeline = Compose(cfg.data.test.pipeline)
    data = test_pipeline(data)
    data['img'] = torch.stack(data['img'], dim=0)
    return data

def main(args):
    model = init_detector(args.config, args.checkpoint, device="cuda:1")
    cfg = model.cfg
    image = cv2.imread(args.input_img)

    layer_names = set()
    for (name, module) in model.named_modules():
        layer_names.add(name)
    # name.add('backbone.stages.4')
    # grad_cam = GradCAM(model, layer_names, args.output_file, image,
    #                    pre_process, atss_post_process)

    grad_cam = GradCAM(model, layer_names, args.output_file, image,
                       pre_process, atss_post_process)
    grad_cam.pre_process(image, cfg)

    grad_cam()
    # drawBboxes(mask, image)
    # cv2.imwrite('recImage.jpg', image)


    # image_cam, image_dict['heatmap'] = gen_cam(img[y1:y2, x1:x2], mask)

    # x = torch.zeros(1, 3, 512, 512)
    # model([x])





if __name__ == '__main__':
    args = parse_args()
    main(args)







