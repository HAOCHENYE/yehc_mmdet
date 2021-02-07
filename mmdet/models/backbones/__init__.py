from .detectors_resnet import DetectoRS_ResNet
from .detectors_resnext import DetectoRS_ResNeXt
from .hourglass import HourglassNet
from .hrnet import HRNet
from .regnet import RegNet
from .res2net import Res2Net
from .resnet import ResNet, ResNetV1d
from .resnext import ResNeXt
from .ssd_vgg import SSDVGG
from .darknet import DarknetV3
from .VoVNet import VoVNet
from .VoVnet_lite import VoVNet_lite
from .VoVnet_s8 import VoVNet_lite_s8
from .resnet_lite import ResNetLite
from .vovnet_lite_normconv import VoVNet_lite_normconv
from .YL_vovnet import YL_Vovnet
from .CSPOSANet import CSPOSANet
from .RepVGGNet import RepVGGNet
from .resnextDy import ResNeXtDy
__all__ = [
    'RegNet', 'ResNet', 'ResNetV1d', 'ResNeXt', 'SSDVGG', 'HRNet', 'Res2Net',
    'HourglassNet', 'DetectoRS_ResNet', 'DetectoRS_ResNeXt', 'DarknetV3',
    'VoVNet', 'VoVNet_lite', 'VoVNet_lite_s8', 'VoVNet_lite_normconv', 'YL_Vovnet', 'CSPOSANet',
    'RepVGGNet', 'ResNeXtDy'
]
