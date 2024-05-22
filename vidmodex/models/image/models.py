import torch
import torchvision

from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision.models import vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn
from torchvision.models import alexnet
from torchvision.models import efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7
from torchvision.models import mobilenet_v2
from torchvision.models import shufflenet_v2_x0_5, shufflenet_v2_x1_0, shufflenet_v2_x1_5, shufflenet_v2_x2_0
from torchvision.models import inception_v3
from torchvision.models import densenet121, densenet169, densenet201, densenet161
from torchvision.models import googlenet
from torchvision.models import mnasnet0_5, mnasnet0_75, mnasnet1_0, mnasnet1_3
from torchvision.models import resnext50_32x4d, resnext101_32x8d
from torchvision.models import wide_resnet50_2, wide_resnet101_2
from torchvision.models import squeezenet1_0, squeezenet1_1

from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights
from torchvision.models import VGG11_Weights, VGG11_BN_Weights, VGG13_Weights, VGG13_BN_Weights, VGG16_Weights, VGG16_BN_Weights, VGG19_Weights, VGG19_BN_Weights
from torchvision.models import AlexNet_Weights
from torchvision.models import EfficientNet_B0_Weights, EfficientNet_B1_Weights, EfficientNet_B2_Weights, EfficientNet_B3_Weights, EfficientNet_B4_Weights, EfficientNet_B5_Weights, EfficientNet_B6_Weights, EfficientNet_B7_Weights
from torchvision.models import MobileNet_V2_Weights
from torchvision.models import ShuffleNet_V2_X0_5_Weights, ShuffleNet_V2_X1_0_Weights, ShuffleNet_V2_X1_5_Weights, ShuffleNet_V2_X2_0_Weights
from torchvision.models import Inception_V3_Weights
from torchvision.models import DenseNet121_Weights, DenseNet169_Weights, DenseNet201_Weights, DenseNet161_Weights
from torchvision.models import GoogLeNet_Weights
from torchvision.models import MNASNet0_5_Weights, MNASNet0_75_Weights, MNASNet1_0_Weights, MNASNet1_3_Weights
from torchvision.models import ResNeXt50_32X4D_Weights, ResNeXt101_32X8D_Weights
from torchvision.models import Wide_ResNet50_2_Weights, Wide_ResNet101_2_Weights
from torchvision.models import SqueezeNet1_0_Weights, SqueezeNet1_1_Weights

DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)
MODIFIED_MEAN = (0., 0., 0.)
MODIFIED_STD = (1., 1., 1.)
# transforms
gan2resnet18_transform = ResNet18_Weights.DEFAULT.transforms(mean=MODIFIED_MEAN, std=MODIFIED_STD)
gan2resnet34_transform = ResNet34_Weights.DEFAULT.transforms(mean=MODIFIED_MEAN, std=MODIFIED_STD)
gan2resnet50_transform = ResNet50_Weights.DEFAULT.transforms(mean=MODIFIED_MEAN, std=MODIFIED_STD)
gan2resnet101_transform = ResNet101_Weights.DEFAULT.transforms(mean=MODIFIED_MEAN, std=MODIFIED_STD)
gan2resnet152_transform = ResNet152_Weights.DEFAULT.transforms(mean=MODIFIED_MEAN, std=MODIFIED_STD)
gan2vgg11_transform = VGG11_Weights.DEFAULT.transforms(mean=MODIFIED_MEAN, std=MODIFIED_STD)
gan2vgg11_bn_transform = VGG11_BN_Weights.DEFAULT.transforms(mean=MODIFIED_MEAN, std=MODIFIED_STD)
gan2vgg13_transform = VGG13_Weights.DEFAULT.transforms(mean=MODIFIED_MEAN, std=MODIFIED_STD)
gan2vgg13_bn_transform = VGG13_BN_Weights.DEFAULT.transforms(mean=MODIFIED_MEAN, std=MODIFIED_STD)
gan2vgg16_transform = VGG16_Weights.DEFAULT.transforms(mean=MODIFIED_MEAN, std=MODIFIED_STD)
gan2vgg16_bn_transform = VGG16_BN_Weights.DEFAULT.transforms(mean=MODIFIED_MEAN, std=MODIFIED_STD)
gan2vgg19_transform = VGG19_Weights.DEFAULT.transforms(mean=MODIFIED_MEAN, std=MODIFIED_STD)
gan2vgg19_bn_transform = VGG19_BN_Weights.DEFAULT.transforms(mean=MODIFIED_MEAN, std=MODIFIED_STD)
gan2alexnet_transform = AlexNet_Weights.DEFAULT.transforms(mean=MODIFIED_MEAN, std=MODIFIED_STD)
gan2efficientnet_b0_transform = EfficientNet_B0_Weights.DEFAULT.transforms(mean=MODIFIED_MEAN, std=MODIFIED_STD)
gan2efficientnet_b1_transform = EfficientNet_B1_Weights.DEFAULT.transforms(mean=MODIFIED_MEAN, std=MODIFIED_STD)
gan2efficientnet_b2_transform = EfficientNet_B2_Weights.DEFAULT.transforms(mean=MODIFIED_MEAN, std=MODIFIED_STD)
gan2efficientnet_b3_transform = EfficientNet_B3_Weights.DEFAULT.transforms(mean=MODIFIED_MEAN, std=MODIFIED_STD)
gan2efficientnet_b4_transform = EfficientNet_B4_Weights.DEFAULT.transforms(mean=MODIFIED_MEAN, std=MODIFIED_STD)
gan2efficientnet_b5_transform = EfficientNet_B5_Weights.DEFAULT.transforms(mean=MODIFIED_MEAN, std=MODIFIED_STD)
gan2efficientnet_b6_transform = EfficientNet_B6_Weights.DEFAULT.transforms(mean=MODIFIED_MEAN, std=MODIFIED_STD)
gan2efficientnet_b7_transform = EfficientNet_B7_Weights.DEFAULT.transforms(mean=MODIFIED_MEAN, std=MODIFIED_STD)
gan2mobilenet_v2_transform = MobileNet_V2_Weights.DEFAULT.transforms(mean=MODIFIED_MEAN, std=MODIFIED_STD)
gan2shufflenet_v2_x0_5_transform = ShuffleNet_V2_X0_5_Weights.DEFAULT.transforms(mean=MODIFIED_MEAN, std=MODIFIED_STD)
gan2shufflenet_v2_x1_0_transform = ShuffleNet_V2_X1_0_Weights.DEFAULT.transforms(mean=MODIFIED_MEAN, std=MODIFIED_STD)
gan2shufflenet_v2_x1_5_transform = ShuffleNet_V2_X1_5_Weights.DEFAULT.transforms(mean=MODIFIED_MEAN, std=MODIFIED_STD)
gan2shufflenet_v2_x2_0_transform = ShuffleNet_V2_X2_0_Weights.DEFAULT.transforms(mean=MODIFIED_MEAN, std=MODIFIED_STD)
gan2inception_v3_transform = Inception_V3_Weights.DEFAULT.transforms(mean=MODIFIED_MEAN, std=MODIFIED_STD)
gan2densenet121_transform = DenseNet121_Weights.DEFAULT.transforms(mean=MODIFIED_MEAN, std=MODIFIED_STD)
gan2densenet169_transform = DenseNet169_Weights.DEFAULT.transforms(mean=MODIFIED_MEAN, std=MODIFIED_STD)
gan2densenet201_transform = DenseNet201_Weights.DEFAULT.transforms(mean=MODIFIED_MEAN, std=MODIFIED_STD)
gan2densenet161_transform = DenseNet161_Weights.DEFAULT.transforms(mean=MODIFIED_MEAN, std=MODIFIED_STD)
gan2googlenet_transform = GoogLeNet_Weights.DEFAULT.transforms(mean=MODIFIED_MEAN, std=MODIFIED_STD)
gan2mnasnet0_5_transform = MNASNet0_5_Weights.DEFAULT.transforms(mean=MODIFIED_MEAN, std=MODIFIED_STD)
gan2mnasnet0_75_transform = MNASNet0_75_Weights.DEFAULT.transforms(mean=MODIFIED_MEAN, std=MODIFIED_STD)
gan2mnasnet1_0_transform = MNASNet1_0_Weights.DEFAULT.transforms(mean=MODIFIED_MEAN, std=MODIFIED_STD)
gan2mnasnet1_3_transform = MNASNet1_3_Weights.DEFAULT.transforms(mean=MODIFIED_MEAN, std=MODIFIED_STD)
gan2resnext50_32x4d_transform = ResNeXt50_32X4D_Weights.DEFAULT.transforms(mean=MODIFIED_MEAN, std=MODIFIED_STD)
gan2resnext101_32x8d_transform = ResNeXt101_32X8D_Weights.DEFAULT.transforms(mean=MODIFIED_MEAN, std=MODIFIED_STD)
gan2wide_resnet50_2_transform = Wide_ResNet50_2_Weights.DEFAULT.transforms(mean=MODIFIED_MEAN, std=MODIFIED_STD)
gan2wide_resnet101_2_transform = Wide_ResNet101_2_Weights.DEFAULT.transforms(mean=MODIFIED_MEAN, std=MODIFIED_STD)
gan2squeezenet1_0_transform = SqueezeNet1_0_Weights.DEFAULT.transforms(mean=MODIFIED_MEAN, std=MODIFIED_STD)
gan2squeezenet1_1_transform = SqueezeNet1_1_Weights.DEFAULT.transforms(mean=MODIFIED_MEAN, std=MODIFIED_STD)

img2resnet18_transform = ResNet18_Weights.DEFAULT.transforms(mean=DEFAULT_MEAN, std=DEFAULT_STD)
img2resnet34_transform = ResNet34_Weights.DEFAULT.transforms(mean=DEFAULT_MEAN, std=DEFAULT_STD)
img2resnet50_transform = ResNet50_Weights.DEFAULT.transforms(mean=DEFAULT_MEAN, std=DEFAULT_STD)
img2resnet101_transform = ResNet101_Weights.DEFAULT.transforms(mean=DEFAULT_MEAN, std=DEFAULT_STD)
img2resnet152_transform = ResNet152_Weights.DEFAULT.transforms(mean=DEFAULT_MEAN, std=DEFAULT_STD)
img2vgg11_transform = VGG11_Weights.DEFAULT.transforms(mean=DEFAULT_MEAN, std=DEFAULT_STD)
img2vgg11_bn_transform = VGG11_BN_Weights.DEFAULT.transforms(mean=DEFAULT_MEAN, std=DEFAULT_STD)
img2vgg13_transform = VGG13_Weights.DEFAULT.transforms(mean=DEFAULT_MEAN, std=DEFAULT_STD)
img2vgg13_bn_transform = VGG13_BN_Weights.DEFAULT.transforms(mean=DEFAULT_MEAN, std=DEFAULT_STD)
img2vgg16_transform = VGG16_Weights.DEFAULT.transforms(mean=DEFAULT_MEAN, std=DEFAULT_STD)
img2vgg16_bn_transform = VGG16_BN_Weights.DEFAULT.transforms(mean=DEFAULT_MEAN, std=DEFAULT_STD)
img2vgg19_transform = VGG19_Weights.DEFAULT.transforms(mean=DEFAULT_MEAN, std=DEFAULT_STD)
img2vgg19_bn_transform = VGG19_BN_Weights.DEFAULT.transforms(mean=DEFAULT_MEAN, std=DEFAULT_STD)
img2alexnet_transform = AlexNet_Weights.DEFAULT.transforms(mean=DEFAULT_MEAN, std=DEFAULT_STD)
img2efficientnet_b0_transform = EfficientNet_B0_Weights.DEFAULT.transforms(mean=DEFAULT_MEAN, std=DEFAULT_STD)
img2efficientnet_b1_transform = EfficientNet_B1_Weights.DEFAULT.transforms(mean=DEFAULT_MEAN, std=DEFAULT_STD)
img2efficientnet_b2_transform = EfficientNet_B2_Weights.DEFAULT.transforms(mean=DEFAULT_MEAN, std=DEFAULT_STD)
img2efficientnet_b3_transform = EfficientNet_B3_Weights.DEFAULT.transforms(mean=DEFAULT_MEAN, std=DEFAULT_STD)
img2efficientnet_b4_transform = EfficientNet_B4_Weights.DEFAULT.transforms(mean=DEFAULT_MEAN, std=DEFAULT_STD)
img2efficientnet_b5_transform = EfficientNet_B5_Weights.DEFAULT.transforms(mean=DEFAULT_MEAN, std=DEFAULT_STD)
img2efficientnet_b6_transform = EfficientNet_B6_Weights.DEFAULT.transforms(mean=DEFAULT_MEAN, std=DEFAULT_STD)
img2efficientnet_b7_transform = EfficientNet_B7_Weights.DEFAULT.transforms(mean=DEFAULT_MEAN, std=DEFAULT_STD)
img2mobilenet_v2_transform = MobileNet_V2_Weights.DEFAULT.transforms(mean=DEFAULT_MEAN, std=DEFAULT_STD)
img2shufflenet_v2_x0_5_transform = ShuffleNet_V2_X0_5_Weights.DEFAULT.transforms(mean=DEFAULT_MEAN, std=DEFAULT_STD)
img2shufflenet_v2_x1_0_transform = ShuffleNet_V2_X1_0_Weights.DEFAULT.transforms(mean=DEFAULT_MEAN, std=DEFAULT_STD)
img2shufflenet_v2_x1_5_transform = ShuffleNet_V2_X1_5_Weights.DEFAULT.transforms(mean=DEFAULT_MEAN, std=DEFAULT_STD)
img2shufflenet_v2_x2_0_transform = ShuffleNet_V2_X2_0_Weights.DEFAULT.transforms(mean=DEFAULT_MEAN, std=DEFAULT_STD)
img2inception_v3_transform = Inception_V3_Weights.DEFAULT.transforms(mean=DEFAULT_MEAN, std=DEFAULT_STD)
img2densenet121_transform = DenseNet121_Weights.DEFAULT.transforms(mean=DEFAULT_MEAN, std=DEFAULT_STD)
img2densenet169_transform = DenseNet169_Weights.DEFAULT.transforms(mean=DEFAULT_MEAN, std=DEFAULT_STD)
img2densenet201_transform = DenseNet201_Weights.DEFAULT.transforms(mean=DEFAULT_MEAN, std=DEFAULT_STD)
img2densenet161_transform = DenseNet161_Weights.DEFAULT.transforms(mean=DEFAULT_MEAN, std=DEFAULT_STD)
img2googlenet_transform = GoogLeNet_Weights.DEFAULT.transforms(mean=DEFAULT_MEAN, std=DEFAULT_STD)
img2mnasnet0_5_transform = MNASNet0_5_Weights.DEFAULT.transforms(mean=DEFAULT_MEAN, std=DEFAULT_STD)
img2mnasnet0_75_transform = MNASNet0_75_Weights.DEFAULT.transforms(mean=DEFAULT_MEAN, std=DEFAULT_STD)
img2mnasnet1_0_transform = MNASNet1_0_Weights.DEFAULT.transforms(mean=DEFAULT_MEAN, std=DEFAULT_STD)
img2mnasnet1_3_transform = MNASNet1_3_Weights.DEFAULT.transforms(mean=DEFAULT_MEAN, std=DEFAULT_STD)
img2resnext50_32x4d_transform = ResNeXt50_32X4D_Weights.DEFAULT.transforms(mean=DEFAULT_MEAN, std=DEFAULT_STD)
img2resnext101_32x8d_transform = ResNeXt101_32X8D_Weights.DEFAULT.transforms(mean=DEFAULT_MEAN, std=DEFAULT_STD)
img2wide_resnet50_2_transform = Wide_ResNet50_2_Weights.DEFAULT.transforms(mean=DEFAULT_MEAN, std=DEFAULT_STD)
img2wide_resnet101_2_transform = Wide_ResNet101_2_Weights.DEFAULT.transforms(mean=DEFAULT_MEAN, std=DEFAULT_STD)
img2squeezenet1_0_transform = SqueezeNet1_0_Weights.DEFAULT.transforms(mean=DEFAULT_MEAN, std=DEFAULT_STD)
img2squeezenet1_1_transform = SqueezeNet1_1_Weights.DEFAULT.transforms(mean=DEFAULT_MEAN, std=DEFAULT_STD)