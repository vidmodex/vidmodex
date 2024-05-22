import math
from cv2 import log, reduce
import shap
from shap.maskers import Image, Video
import torch
from torch import inverse, nn
from torch.nn import functional as F
import numpy as np
from contextlib import contextmanager

# @contextmanager
# def apply_transform(explainer, inp, transform, inverse_transform):
#     inp_t = inp
#     if transform:
#         inp_t = transform(inp.detach().cpu().numpy())
#     if inverse_transform:
#         fm = explainer.model.inner_model
#         new_fm = lambda x: fm(inverse_transform(x, base_input=inp.detach().cpu().numpy()))  # type: ignore
#         explainer.model.inner_model = new_fm
#     try:
#         yield explainer, inp_t
#     finally:
#         if inverse_transform:
#             explainer.model.inner_model = fm
#             inp = inverse_transform(inp_t)
        
@contextmanager
def apply_transform(explainer, inp, transform, inverse_transform):
    
    try:
        yield explainer, inp
    finally:
        pass
        


class ShapLoss(nn.Module):
    def __init__(self, model_call, input_sample, num_classes=None, model_type="video", explainer_batch_size=32, max_evals=128, inp_transform=None, inp_inverse_transform=None):
        super(ShapLoss, self).__init__()
        
        self.inp_transform = inp_transform
        self.inp_inverse_transform = inp_inverse_transform

        if "video" in model_type.lower():
            masker = Video("blur(3, 32, 32)", input_sample.shape, reduction_transform=self.inp_transform, inv_reduction_transform=self.inp_inverse_transform)
        elif "image" in model_type.lower():
            masker = Image("blur(32, 32)", input_sample.shape) #, reduction_transform=self.inp_transform, inv_reduction_transform=self.inp_inverse_transform) not required as size is not an issue
        else:
            raise ValueError("Model type not supported yet.")
        if num_classes:
            output_names = list(range(num_classes))
        else:
            output_names = None
        self.explainer = shap.PartitionExplainer(model_call, masker, output_names=output_names)
        self.max_evals = max_evals
        self.batch_size = explainer_batch_size
        self.model_call = model_call
        self.masker = masker
        self.num_classes = num_classes
        

    def set_max_evals(self, max_evals):
        self.max_evals = max_evals

    def compute_norm_prob(self, mu, sigma, value):
        log_prob = -1 * ( (value - mu)**2/ (2 * (sigma + 1e-6)**2) ) / ((sigma+1e-6) * math.sqrt(2 * math.pi))
                                                                
        return torch.exp(log_prob)
    
    def normalize_shap(self, shap_values):
        if isinstance(shap_values, torch.Tensor):
            shap_abs_max_val = torch.abs(shap_values).amax(dim=tuple(range(1,len(shap_values.shape))), keepdim=True)
            shap_abs_max_val[shap_abs_max_val < 1e-36] = 100.0
            shap_values_norm = shap_values / shap_abs_max_val
            # shap_values_norm = 2.0 * ( (shap_values - shap_values.amin(dim= tuple(range(1,len(shap_values.shape))) , keepdim=False)) / (shap_values.amax(dim=tuple(range(1,len(shap_values.shape))), keepdim=False) - shap_values.amin(dim=tuple(range(1,len(shap_values.shape))), keepdim=False))) - 1.0
            # shap_values_min = shap_values.amin(dim=tuple(range(1,len(shap_values.shape))), keepdim=False)
            # shap_values_max = shap_values.amax(dim=tuple(range(1,len(shap_values.shape))), keepdim=False)
        elif isinstance(shap_values, np.ndarray):
            shap_abs_max_val = np.abs(shap_values).max(axis=tuple(range(1,len(shap_values.shape))), keepdims=True)
            shap_abs_max_val[shap_abs_max_val < 1e-36] = 100.0
            shap_values_norm = shap_values / shap_abs_max_val
            # shap_values_norm = 2.0 * ( (shap_values - shap_values.min(axis=tuple(range(1,len(shap_values.shape))), keepdims=False)) / (shap_values.max(axis=tuple(range(1,len(shap_values.shape))), keepdims=False) - shap_values.min(axis=tuple(range(1,len(shap_values.shape))), keepdims=False))) - 1.0
            # shap_values_min = shap_values.min(axis=tuple(range(1,len(shap_values.shape))), keepdims=False)
            # shap_values_max = shap_values.max(axis=tuple(range(1,len(shap_values.shape))), keepdims=False)
        else:
            raise ValueError("shap_values should be a torch.Tensor or a numpy.ndarray.")
        
        return shap_values_norm, shap_abs_max_val
    
    # def denormalize_shap(self, shap_values_norm, shap_values_min, shap_values_max):
    #     shap_values = 0.5 * (shap_values_norm + 1.0) * (shap_values_max - shap_values_min) + shap_values_min
    #     return shap_values

    def denormalize_shap(self, shap_values_norm, shap_abs_max_val):
        #shap_abs_max_val[shap_abs_max_val == 100.0] = 0.0
        shap_values = shap_values_norm * shap_abs_max_val
        return shap_values
    
    def forward(self, inp, shap_out_mu, shap_out_sigma, class_idx=None, optimize_prob=False, optimize_energy=False, shap_gt=None, use_prob_energy=True):
        shap_gt_abs_max_val = None
        target_cls = None
        
        if optimize_energy and not use_prob_energy and shap_gt is None:
            shap_gt = torch.ones(shap_out_mu.shape, dtype=shap_out_mu.dtype, device=shap_out_mu.device)
            
        if shap_gt is None and self.max_evals > 0:
            if class_idx is None:
                target_output = shap.Explanation.argsort.flip[[0]] #,1,2,3]] # type: ignore
            elif isinstance(class_idx, int):
                target_output = [class_idx]
            elif isinstance(class_idx, list):
                target_output = class_idx
            elif isinstance(class_idx, shap.utils._general.OpChain):
                target_output = class_idx
            elif isinstance(class_idx, np.ndarray):
                target_output = class_idx
            elif isinstance(class_idx, torch.Tensor):
                target_output = class_idx.detach().cpu().numpy()
            else:
                raise ValueError("class_idx should be an integer or a list of integers or a shap.utils._general.OpChain object.") 

            with apply_transform(self.explainer, inp, self.inp_transform, self.inp_inverse_transform) as (explainer, inp_t):
                _explanation = explainer(
                    inp_t,
                    max_evals=self.max_evals,
                    batch_size=self.batch_size, # type: ignore
                    outputs=target_output,
                )
            shap_raw = _explanation.values[..., 0].sum(1, keepdims=True)
            if self.inp_inverse_transform:
                shap_raw = self.inp_inverse_transform(shap_raw)
            shap_gt, shap_gt_abs_max_val = self.normalize_shap(shap_raw)

            shap_gt = torch.tensor(shap_gt, dtype=shap_out_mu.dtype, device=shap_out_mu.device).sum(1, keepdim=True).squeeze(-1)

            target_cls = _explanation.output_names
        else:
            shap_gt = shap_out_mu
            target_cls = class_idx
            shap_gt_abs_max_val = None
            
        if (not optimize_prob) and (not optimize_energy):
            optimize_prob = True
        loss = None
        if optimize_prob:
            shap_pred = (shap_out_sigma * torch.randn_like(shap_out_mu)) + shap_out_mu

            loss = F.mse_loss(shap_pred, shap_gt)
        elif optimize_energy:
            if use_prob_energy:
                prob_value = self.compute_norm_prob(shap_out_mu, shap_out_sigma, shap_gt)
            else:
                prob_value = torch.ones(shap_out_mu.shape, dtype=shap_out_mu.dtype, device=shap_out_mu.device)
            masked_shap = prob_value * shap_out_mu
            loss = -1 * masked_shap.sum()
            
        return loss, (shap_gt, shap_gt_abs_max_val, target_cls)
