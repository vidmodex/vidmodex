from torchvision.transforms import Compose
from torchvision.transforms import Resize, CenterCrop, Normalize
from torchvision.transforms.functional import resize
import torchvideo.transforms as VT
from torchvision.transforms.functional import InterpolationMode
import torch

class TensorTransformFactory:
    _transform_classes = {}

    @classmethod
    def get(cls, transform_type:str):
        try:
            return cls._transform_classes[transform_type]
        except KeyError:
            raise ValueError(f"unknown product type : {transform_type}")

    @classmethod
    def register(cls, transform_type:str):
        def inner_wrapper(wrapped_class):
            cls._transform_classes[transform_type] = wrapped_class
            return wrapped_class
        return inner_wrapper

def transform_gan2vivit(vids):
    # vid = 255.0 * (vids + 1.0) / 2.0
    old_shape = vids.shape
    vids = vids.reshape(-1, *old_shape[2:])
    vids = resize(vids,(224,224),interpolation=InterpolationMode.BICUBIC)
    vids = vids.reshape(*old_shape[:3], 224,224)

    vids = vids.transpose(1,2) # NCTHW -> NTCHW
    # Normalization not required if resize without 0-255
    return vids

def transform_gan2swint(vids):
    # vid = 255.0 * (vids + 1.0) / 2.0
    old_shape = vids.shape
    vids = vids.reshape(-1, *old_shape[2:])
    vids = resize(vids,(224,224),interpolation=InterpolationMode.BICUBIC)
    vids = vids.reshape(*old_shape[:3], 224,224)

    #vids = vids.transpose(1,2) # NCTHW -> NCTHW
    return vids

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