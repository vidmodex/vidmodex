from configparser import Interpolation
import torchvideo.transforms as VT
from torchvision.transforms import Compose

tgan_transform = Compose([
    VT.ResizeVideo((64,64), interpolation = 2),
    VT.CollectFrames(),
    VT.PILVideoToTensor(rescale=True),
    VT.NormalizeVideo(
        mean = [0.45, 0.45, 0.45],
        std = [0.5, 0.5, 0.5],
        channel_dim = 0,
        inplace = True
    )
])

transform_movinetA5 = Compose([
    VT.NDArrayToPILVideo('thwc'),
    VT.ResizeVideo((320, 320), interpolation=4),
    VT.RandomCropVideo((320, 320)),
    VT.PILVideoToTensor(),
])


transform_vivit = Compose([
    VT.NDArrayToPILVideo('thwc'),
    VT.ResizeVideo(
        (256,256)),
    VT.CenterCropVideo((224,224)),
    VT.PILVideoToTensor(rescale=False, ordering='TCHW'),
    VT.NormalizeVideo(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        channel_dim=1,
        inplace=True,
    ),
])


transform_swint = Compose([
    VT.NDArrayToPILVideo('thwc'),
    VT.ResizeVideo(
        (256,256)),
    VT.CenterCropVideo((224,224)),
    VT.PILVideoToTensor(rescale=False, ordering='CTHW'),
    VT.NormalizeVideo(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        channel_dim=0,
        inplace=True,
    ),
])
# transform_gan2vivit = Compose([
#     VT.NDArrayToPILVideo('cthw'),
#     VT.ResizeVideo(
#         (256,256)),
#     VT.CenterCropVideo((224,224)),
#     VT.PILVideoToTensor(rescale=False, ordering='TCHW'),
#     VT.NormalizeVideo(
#         mean=[123.675, 116.28, 103.53],
#         std=[58.395, 57.12, 57.375],
#         channel_dim=0,
#         inplace=True,
#     ),
# ])


# transform_gan2swint = Compose([
#     VT.NDArrayToPILVideo('cthw'),
#     VT.ResizeVideo(
#         (256,256)),
#     VT.CenterCropVideo((224,224)),
#     VT.PILVideoToTensor(rescale=False, ordering='CTHW'),
#     VT.NormalizeVideo(
#         mean=[123.675, 116.28, 103.53],
#         std=[58.395, 57.12, 57.375],
#         channel_dim=0,
#         inplace=True,
#     ),
# ])