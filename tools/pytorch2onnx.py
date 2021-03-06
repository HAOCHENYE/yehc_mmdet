import argparse
import os.path as osp
from functools import partial
import tensorwatch as tw
import mmcv
import numpy as np
import onnx
import onnxruntime as rt
import torch
from mmcv.runner import load_checkpoint
from mmcv.cnn import fuse_conv_bn
from mmdet.models import build_detector

try:
    from mmcv.onnx.symbolic import register_extra_symbolics
except ModuleNotFoundError:
    raise NotImplementedError('please update mmcv to version>=v1.0.4')

def load(module, prefix=''):
    for name, child in module._modules.items():
        if not hasattr(child, 'fuse_conv'):
            load(child, prefix + name + '.')
        else:
            child.fuse_conv()

def pytorch2onnx(model,
                 input_img,
                 input_shape,
                 opset_version=11,
                 show=False,
                 output_file='tmp.onnx',
                 verify=False,
                 normalize_cfg=None):
    # model=fuse_conv_bn(model)
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
    origin_forward = model.forward
    model.forward = partial(
        model.forward, img_metas=[[one_meta]], return_loss="onnx")
    load(model)
    # pytorch has some bug in pytorch1.3, we have to fix it
    # by replacing these existing op
    register_extra_symbolics(opset_version)
    print(tw.model_stats(model, (1, 3, 512, 512)))
    output_names = ["P3_logits", "P4_logits", "P5_logits", "P6_logits","P7_logits",
                    "P3_bbox_reg", "P4_bbox_reg", "P5_bbox_reg", "P6_bbox_reg","P7_bbox_reg"]
    # output_names = ["hm", "wh"]
    torch.onnx.export(
        model, (one_img),
        output_file,
        export_params=True,
        keep_initializers_as_inputs=True,
        verbose=show,
        opset_version=11,
        input_names=['data'],
        output_names=output_names,
        do_constant_folding=True,

    )
    #
    # torch.onnx.export(
    #     model, [(one_img)],
    #     output_file,
    #     export_params=True,
    #     keep_initializers_as_inputs=True,
    #     verbose=show,
    #     opset_version=opset_version)
    #
    model.forward = origin_forward
    print(f'Successfully exported ONNX model: {output_file}')
    if verify:
        # check by onnx
        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)

        # check the numerical value
        # get pytorch output
        pytorch_result = model([one_img], [[one_meta]], return_loss=False)

        # get onnx output
        input_all = [node.name for node in onnx_model.graph.input]
        input_initializer = [
            node.name for node in onnx_model.graph.initializer
        ]
        net_feed_input = list(set(input_all) - set(input_initializer))
        assert (len(net_feed_input) == 1)
        sess = rt.InferenceSession(output_file)
        from mmdet.core import bbox2result
        det_bboxes, det_labels = sess.run(
            None, {net_feed_input[0]: one_img.detach().numpy()})
        # only compare a part of result
        bbox_results = bbox2result(det_bboxes, det_labels, 1)
        onnx_results = bbox_results[0]
        assert np.allclose(
            pytorch_result[0][:, 4], onnx_results[:, 4]
        ), 'The outputs are different between Pytorch and ONNX'
        print('The numerical values are same between Pytorch and ONNX')


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


if __name__ == '__main__':
    args = parse_args()

    assert args.opset_version == 11, 'MMDet only support opset 11 now'

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
    try:
        checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    except:
        Warning("Path of checkpoint is not correct")
    # conver model to onnx file
    pytorch2onnx(
        model,
        args.input_img,
        input_shape,
        opset_version=args.opset_version,
        show=args.show,
        output_file=args.output_file,
        verify=args.verify,
        normalize_cfg=normalize_cfg)
