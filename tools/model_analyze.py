import argparse
import torch.nn as nn
from collections import OrderedDict
from functools import partial
import warnings
import pandas as pd
import numpy as np
import os.path as osp
import mmcv
from mmdet.models import build_detector
import torch
import sys

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


class ModelAnalyzer(object):
    class Logger(object):
        def __init__(self, filename="Default.log"):
            self.terminal = sys.stdout
            self.log = open(filename, "w")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            pass

    def __init__(self, model, skip_bn=True, skip_relu=True, layer_name=None, output_file=None):
        '''
        current anylyzer only conceron mutily operation
        :param model:
        :param skip_bn:
        :param skip_relu:
        :param layer_name:
        '''
        assert isinstance(model, nn.Module), \
            "model must be type nn.Module"
        if layer_name:
            assert isinstance(layer_name, list)
        self.output_file = output_file
        if output_file:
            self.logger = self.Logger(output_file)
        self.skip_bn = skip_bn
        self.ski_relu = skip_relu
        self.model = model
        self.layer_name = layer_name


        self.flops_dict = OrderedDict()
        self.macs_dict = OrderedDict()
        self.actsize_dict = OrderedDict()
        self.feat_dict = OrderedDict()
        self.input_shape = OrderedDict()
        self.out_put_shape = OrderedDict()
        self.kernel_size = OrderedDict()

        self._index_register_hook()
        self._feat_rigister_hook()


    def index_cal(self, module, input, output, layer_name):
        if module._modules:
            return
        if isinstance(input, tuple):
            input = input[0]
        output_size = output.shape[2] * output.shape[3]
        input_size = input.shape[2] * input.shape[3]
        input_ch = input.shape[1]
        output_ch = output.shape[1]
        act_size = output_size * output_ch * 4

        self.actsize_dict[layer_name] = act_size
        self.input_shape[layer_name] = input.shape
        self.out_put_shape[layer_name] = output.shape


        if isinstance(module, nn.Conv2d):
            kernel_size = module.weight.shape[3] * module.weight.shape[2]
            group = module.groups
            conv_flops = input_size * input_ch * output_ch / (input_size / output_size) * kernel_size / group
            conv_macs = output_size * output_ch*4 + kernel_size * input_ch * output_ch * 4


            self.flops_dict[layer_name] = conv_flops
            self.macs_dict[layer_name] = conv_macs
            self.kernel_size[layer_name] = module.weight.shape
            return

        elif isinstance(module, nn.BatchNorm2d):
            if not self.skip_bn:
                bn_flops = output_size * output_ch * 2
                bn_macs = output_size * output_ch * 4

                self.flops_dict[layer_name] = bn_flops
                self.macs_dict[layer_name] = bn_macs
                self.kernel_size[layer_name] = []
                return
            else:
                self.flops_dict[layer_name] = 0
                self.macs_dict[layer_name] = 0
                self.actsize_dict[layer_name] = 0
                self.kernel_size[layer_name] = []
                return

        elif isinstance(module, nn.ReLU):
            if not self.ski_relu:
                act_flops = output_size * output_ch
                act_macs = output_size * output_ch * 4

                self.flops_dict[layer_name] = act_flops
                self.macs_dict[layer_name] = act_macs
                self.kernel_size[layer_name] = []
                return
            else:
                self.flops_dict[layer_name] = 0
                self.macs_dict[layer_name] = 0
                self.actsize_dict[layer_name] = 0
                self.kernel_size[layer_name] = []
                return

        else:
            self.flops_dict[layer_name] = 0
            self.macs_dict[layer_name] = 0
            self.kernel_size[layer_name] = []
            warnings.warn("Operator {} cannnot be evaluated".format(module._get_name()))
            return
    
    def get_target_feature(self, module, input, output, layer_name):
        self.feat_dict[layer_name] = output


    def _index_register_hook(self):
        for (name, module) in self.model.named_modules():
            func_forward = partial(self.index_cal, layer_name=name)
            module.register_forward_hook(func_forward)


    def _feat_rigister_hook(self):
        if not self.layer_name:
            return
        for (name, module) in self.model.named_modules():
            if name in self.layer_name:
                func_forward = partial(self.get_target_feature, layer_name=name)
                module.register_forward_hook(func_forward)


    def analyze(self, input):
        self.model(input)
        assert len(self.flops_dict) == len(self.macs_dict) == len(self.actsize_dict), \
        "length of flops macs acts should be equal!"
        data = {}

        total_flops = 0
        total_macs = 0
        total_acts = 0

        data["flops"] = []
        data["macs"] = []
        data["acts"] = []
        data["intensity"] = []
        data["input_shape"] = []
        data["output_shape"] = []
        data["kernel_size"] = []
        row_index = []
        col_index = ["kernel_size", "input_shape", "output_shape", "flops", "macs", "acts", "intensity"]

        for layer_name in self.flops_dict.keys():
            # if self.macs_dict[layer_name] / (self.actsize_dict[layer_name]+1e-10) > 1.2:
            #     print(layer_name)
            row_index.append(layer_name)
            total_flops += self.flops_dict[layer_name]
            total_macs += self.macs_dict[layer_name]
            total_acts += self.actsize_dict[layer_name]



        for layer_name in self.flops_dict.keys():
            cur_flops = self.flops_dict[layer_name]
            cur_macs = self.macs_dict[layer_name]
            cur_acts = self.actsize_dict[layer_name]
            data["input_shape"].append(self.input_shape[layer_name])
            data["output_shape"].append(self.input_shape[layer_name])
            data["flops"].append("{}/{:.3f}%".format(int(cur_flops), cur_flops/total_flops * 100))
            data["macs"].append("{:.3f}M/{:.3f}%".format(cur_macs/1e6 , cur_macs/total_macs * 100))
            data["acts"].append("{:.3f}M/{:.3f}%".format(cur_acts/1e6 , cur_acts/total_acts * 100))
            data["intensity"].append("{:.3f}".format(cur_flops/(cur_macs+1e-10)))
            data["kernel_size"].append(self.kernel_size[layer_name])

        data["kernel_size"].append([])
        data["input_shape"].append([])
        data["output_shape"].append([])
        data["flops"].append("{}/{:.3f}%".format(int(total_flops), total_flops / total_flops * 100))
        data["macs"].append("{:.3f}M/{:.3f}%".format(total_macs/1e6, total_macs / total_macs * 100))
        data["acts"].append("{:.3f}M/{:.3f}%".format(total_acts/1e6, total_acts / total_acts * 100))
        data["intensity"].append("{:.3f}".format(total_flops / (total_macs+1e-10)))


        row_index.append("Total count")

        data_frame = pd.DataFrame(data, index=row_index, columns=col_index)
        self.logger.write(repr(data_frame))
        return data_frame

    def get_geat(self):
        return self.feat_dict

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert MMDetection models to ONNX')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--input-img', type=str, help='Images for input')
    parser.add_argument('--output-file', type=str, default='tmp.onnx')
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

def main(args):
    if not args.input_img:
        args.input_img = osp.join(
            osp.dirname(__file__), '../tests/data/color.jpg')

    if len(args.shape) == 1:
        input_shape = (1, 3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (1, 3) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    assert len(args.mean) == 3
    assert len(args.std) == 3

    normalize_cfg = {
        'mean': np.array(args.mean, dtype=np.float32),
        'std': np.array(args.std, dtype=np.float32)
    }

    cfg = mmcv.Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # build the model
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    input_img = args.input_img
    model.cpu().eval()
    # read image
    one_img = mmcv.imread(input_img)
    if normalize_cfg:
        one_img = mmcv.imnormalize(one_img, normalize_cfg['mean'],
                                   normalize_cfg['std'])
    one_img = mmcv.imresize(one_img, input_shape[2:]).transpose(2, 0, 1)
    one_img = torch.from_numpy(one_img).unsqueeze(0).float()
    (_, C, H, W) = input_shape
    one_meta = {
        'img_shape': (H, W, C),
        'ori_shape': (H, W, C),
        'pad_shape': (H, W, C),
        'filename': '<demo>.png',
        'scale_factor': 1.0,
        'flip': False
    }
    # onnx.export does not support kwargs

    model.forward = partial(
        model.forward, img_metas=[[one_meta]], return_loss="onnx")

    Analyzer = ModelAnalyzer(model, output_file=args.output_file)
    DataFrame = Analyzer.analyze((one_img))
    # print(DataFrame)

if __name__ == '__main__':
    args = parse_args()
    main(args)






