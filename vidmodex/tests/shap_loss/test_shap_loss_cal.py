
import torch
# Choose the `slowfast_r50` model 
model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r101', pretrained=True)


from typing import Dict
import json
import urllib
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    UniformCropVideo
)
from util_check import *

device = "cuda:1"
model = model.eval()
model = model.to(device)
model.share_memory()


side_size = 256
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
crop_size = 256
num_frames = 32
sampling_rate = 2
frames_per_second = 30
slowfast_alpha = 4
num_clips = 10
num_crops = 3

class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors. 
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // slowfast_alpha
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list

inverse_subsample = ApplyTransformToKey(
    key="video",
    transform = Compose(
        [
            NormalizeVideo(
                mean= [-m/s for m, s in zip(mean, std)],
                std= [1/s for s in std]
            ),
            Lambda(lambda x: x*255),
        ]
    ),
)
transform_subsample = ApplyTransformToKey(
    key="video",
    transform = Compose(
        [
            UniformTemporalSubsample(num_frames),
            Lambda(lambda x: x/255.0),
            NormalizeVideo(mean, std),
            ShortSideScale(
                size=side_size
            ),
            CenterCropVideo(crop_size),
        ]
    ),
)

transform =  ApplyTransformToKey(
    key="video",
    transform=Compose(
        [
            # UniformTemporalSubsample(num_frames),
            # Lambda(lambda x: x/255.0),
            # NormalizeVideo(mean, std),
            # ShortSideScale(
            #     size=side_size
            # ),
            # CenterCropVideo(crop_size),
            PackPathway()
        ]
    ),
)

# The duration of the input clip is also specific to the model.
clip_duration = (num_frames * sampling_rate)/frames_per_second


url_link = "https://dl.fbaipublicfiles.com/pytorchvideo/projects/archery.mp4"
video_path = 'archery.mp4'
try: urllib.URLopener().retrieve(url_link, video_path)
except: urllib.request.urlretrieve(url_link, video_path)


video_path = 'vidmodex_output_16.avi'
video_path='makeup.avi'
# video_path = 'archery.mp4'


start_sec = 0
end_sec = start_sec + clip_duration

# Initialize an EncodedVideo helper class and load the video
video = EncodedVideo.from_path(video_path)

# Load the desired clip
video_data_raw = video.get_clip(start_sec=start_sec, end_sec=end_sec)
video_data_raw = transform_subsample(video_data_raw)
# Apply a transform to normalize the video input
video_data = transform(video_data_raw)

# Move the inputs to the desired device
inputs = video_data["video"]
inputs = [i.to(device)[None, ...] for i in inputs]


video_data_raw = video.get_clip(start_sec=start_sec, end_sec=end_sec)
print(video_data_raw["video"].mean(), video_data_raw["video"].std(), video_data_raw["video"].min(), video_data_raw["video"].max())

video_data_raw = transform_subsample(video_data_raw)
print(video_data_raw["video"].mean(), video_data_raw["video"].std(), video_data_raw["video"].min(), video_data_raw["video"].max())
video_data_raw = inverse_subsample(video_data_raw)
print(video_data_raw["video"].mean(), video_data_raw["video"].std(), video_data_raw["video"].min(), video_data_raw["video"].max())
video_data_raw = transform_subsample(video_data_raw)
print(video_data_raw["video"].mean(), video_data_raw["video"].std(), video_data_raw["video"].min(), video_data_raw["video"].max())






if __name__ == "__main__":
    
    # import torch.multiprocessing as mp
    import numpy as np
    # mp.set_start_method("spawn")
    dummy_inp =  [np.zeros((5, 3, 32, 256, 256), dtype=np.float32) for _ in range(4)]
    # def g(x):
    #     return

    with mp.Pool(4) as p:
        res = p.map(fast_call_vid, [[di, model, transform, device] for di in dummy_inp])
    # res = fast_call_vid(dummy_inp[0], model, transform, device)
    print(res)
    exit()
# processes = []
# for rank in range(4):
#     p = mp.Process(target=call_vid, args=(dummy_inp[rank],))
#     p.start()
#     processes.append(p)
# for p in processes:
#     p.join()


# from vidmodex.loss import ShapLoss


# 
# import numpy as np
# from torch.nn import functional as F
# scale_factor = [0.5, 0.25, 0.25]
# inv_scale_factor = [1/s for s in scale_factor]
# one_downsample = lambda x: np.ones((*x.shape[:-4], int(x.shape[-3]*scale_factor[0]), int(x.shape[-2]*scale_factor[1]), int(x.shape[-1]*scale_factor[2])), dtype=x.dtype)

# def vid_downsample_fn(x):
#     scale_factor = [1., 0.5, 0.5]
#     if isinstance(x, torch.Tensor):
#         return F.interpolate(x.reshape(-1, *x.shape), scale_factor=scale_factor, mode='trilinear', align_corners=False)[0]
#     else:
#         return F.interpolate(torch.from_numpy(x.reshape(-1, *x.shape)), scale_factor=scale_factor, mode='trilinear', align_corners=False).detach().cpu().numpy()[0]

# def mask_upsample_fn(x):
#     scale_factor = [1., 0.5, 0.5]
#     inv_scale_factor = [1/s for s in scale_factor]
#     if isinstance(x, torch.Tensor):
#         return F.interpolate(x, scale_factor=inv_scale_factor, mode='nearest-exact')
#     else:
#         return  F.interpolate(torch.from_numpy(x), scale_factor=inv_scale_factor, mode='nearest-exact').detach().cpu().numpy()
# #     mask = mask
# #     base_input = base_input.detach().cpu().numpy()
# #     return base_input[~mask]
# # mask_upsample_fn = lambda x, base_input: np.ones((*x.shape[:-4], int(x.shape[-3]/scale_factor[0]), int(x.shape[-2]/scale_factor[1]), int(x.shape[-1]/scale_factor[2])), dtype=x.dtype)

# 
# def dummy_call(vid):
#     out = torch.randn(vid.shape[0], 400)
#     return F.softmax(out, dim=-1)

# 
# loss = ShapLoss(lambda x: x, video_data_raw["video"], num_classes=400, max_evals=128, inp_transform=vid_downsample_fn, inp_inverse_transform=mask_upsample_fn)
# loss = loss.to(device)

# 
# loss2 = ShapLoss(lambda x: x, video_data_raw["video"], num_classes=400, max_evals=128) #, inp_transform=vid_downsample_fn, inp_inverse_transform=mask_upsample_fn)
# loss2 = loss2.to(device)

# 
# loss3 = ShapLoss(lambda x: x, video_data_raw["video"], num_classes=400, max_evals=128, inp_transform=vid_downsample_fn, inp_inverse_transform=mask_upsample_fn)
# loss3 = loss3.to(device)

# 
# import shap
# loss.explainer.model = shap.models.Model(call_vid)
# loss2.explainer.model = shap.models.Model(call_vid)
# loss3.explainer.model = shap.models.Model(call_vid)

# 


# 
# max_evals = 64
# loss.set_max_evals(max_evals)
# loss2.set_max_evals(2*max_evals)
# loss3.set_max_evals(2*max_evals)

# 
# X = video_data_raw["video"]

# 
# from torch import nn
# import sys
# sys.path.append("/home/aiscuser")
# from discriminator_models import ConditionalShapDiscriminator
# sys.path.pop(-1)
# shap_prob_model = ConditionalShapDiscriminator(n_channels=3, out_channels=1, n_classes=400, width_multiplier=1, trilinear=True, prob_out=True, use_ds_conv=False) #.to('cuda:0')

# checkpoint = torch.load("/home/aiscuser/discriminator/discriminator/unet2_prob/version_46/checkpoints/epoch=99-step=2100.ckpt")
# for key in list(checkpoint["state_dict"].keys()):
#     if "model." in key:
#         checkpoint["state_dict"][key.replace("model.","")] = checkpoint["state_dict"].pop(key)
# shap_prob_model.load_state_dict(checkpoint["state_dict"])


# 
# cls_id = call_vid(X.reshape(-1, *X.shape).numpy()).argsort(-1).flip(-1)[...,0].cpu()

# 
# cls_id

# 
# mu, logvar = shap_prob_model(X.reshape(-1, *X.shape), cls_id)
# sigma = torch.exp(0.5 * logvar)

# 
# torch.cuda.empty_cache()

# 
# loss.batch_size = 128
# loss2.batch_size = 128
# loss3.batch_size = 128

# 
# loss_item, (shap_gt, shap_gt_abs_max_val, target_outputs) = loss(X.reshape(-1, *X.shape), mu, sigma, optimize_prob=True, optimize_energy=False)

# 
# loss_item2, (shap_gt2, shap_gt_abs_max_val2, target_outputs2) = loss2(X.reshape(-1, *X.shape), mu, sigma, optimize_prob=True, optimize_energy=False)

# 
# loss_item3, (shap_gt3, shap_gt_abs_max_val3, target_outputs3) = loss3(X.reshape(-1, *X.shape), mu, sigma, optimize_prob=True, optimize_energy=False)

# 
# target_outputs

# 
# inp = X.reshape(-1, *X.shape).repeat_interleave(5,dim=0)
# inp  = inp + torch.randn_like(inp)*0.2
# loss_item, (shap_gt, shap_gt_abs_max_val, target_cls) = loss(inp, mu.repeat_interleave(5,dim=0), sigma.repeat_interleave(5,dim=0), class_idx=shap.Explanation.argsort.flip[[0]], optimize_prob=True, optimize_energy=False, shap_gt=None)

# 
# loss_item2, (shap_gt2, shap_gt_abs_max_val2, target_cls2) = loss2(inp, mu.repeat_interleave(5,dim=0), sigma.repeat_interleave(5,dim=0), class_idx=shap.Explanation.argsort.flip[[0]], optimize_prob=True, optimize_energy=False, shap_gt=None)

# 
# import numpy as np
# if isinstance(target_cls, list):
#     target_cls = (target_cls[0]*np.ones((inp.shape[0],1))).astype(np.int64)

# 
# target_cls

# 
# import seaborn as sns

# sns.heatmap(shap_gt[0,0,0].cpu().detach().numpy(), cmap="coolwarm", center=0)

# 
# sns.heatmap(shap_gt2[0,0,0].cpu().detach().numpy(), cmap="coolwarm", center=0)

# 
# sns.heatmap(shap_gt3[0,0,0].cpu().detach().numpy(), cmap="coolwarm", center=0)

# 
# sns.heatmap(mu[0,0,0].cpu().detach().numpy(), cmap="coolwarm", center=0)

# 
# prob_val = loss.compute_norm_prob(mu, 0.2, shap_gt)
# loss_energy = mu * prob_val

# 
# prob_val2 = loss2.compute_norm_prob(mu, 0.2, shap_gt2)
# loss_energy2 = mu * prob_val2

# 
# prob_val3 = loss3.compute_norm_prob(mu, 0.2, shap_gt3)
# loss_energy3 = mu * prob_val3

# 
# sigma.mean(), sigma.std()

# 
# sns.heatmap(loss_energy[0,0,0].cpu().detach().numpy(), cmap="coolwarm")

# 
# sns.heatmap(loss_energy2[0,0,0].cpu().detach().numpy(), cmap="coolwarm")

# 
# sns.heatmap(loss_energy3[0,0,0].cpu().detach().numpy(), cmap="coolwarm")

# 
# sns.heatmap(prob_val[0,0,14].cpu().detach().numpy(), cmap="coolwarm")

# 
# sns.heatmap(prob_val2[0,0,14].cpu().detach().numpy(), cmap="coolwarm")

# 
# sns.heatmap(prob_val3[0,0,14].cpu().detach().numpy(), cmap="coolwarm")

# 



