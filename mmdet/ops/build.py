from mmcv.utils import Registry

CUSTOM_CONV_OP = Registry("custom_conv_op")


def build_op(conv_type):
    '''
     :param conv_type:
     conv must have input:
     in_ch
     out_ch
     kernel_size
    :return:
    cls func
    '''
    assert conv_type in CUSTOM_CONV_OP._module_dict.keys(), \
        "conv_type has not been registered!"
    obj_cls = CUSTOM_CONV_OP.module_dict[conv_type]
    return obj_cls
