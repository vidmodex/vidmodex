import torch
from torch import nn
import torch.nn.functional as F
import gc
import numpy as np

class BlackBoxModel(nn.Module):
    def __init__(self, model, transform=None, topk=400, logits=False):
        super(BlackBoxModel, self).__init__()
        self.model = model
        self.transform = transform
        self.topk=topk
        self.logits = logits
        self.dummy_param = nn.Parameter(torch.empty(0))

    
    def fetch_topk_probs(self, preds, k=5):
        topk_probs, topk_idxs = preds.topk(k)
        mask = torch.zeros_like(preds, dtype=torch.bool)
        mask.scatter_(1, topk_idxs, True)
        rem_prob = (1. - topk_probs.sum(-1))/(preds.shape[-1]-k)
        out_probs = torch.einsum("j,jk->jk",rem_prob, torch.ones(preds.shape).to(device=preds.device))
        out_probs[mask.to(dtype=torch.bool)] = preds[mask.to(dtype=torch.bool)]
        return out_probs

    def fetch_topk_logits(self, preds, k=5):
        topk_probs, topk_idxs = preds.topk(k)
        mask = torch.zeros_like(preds, dtype=torch.bool)
        mask.scatter_(1, topk_idxs, True)
        rem_prob = 0.0*(1. - topk_probs.sum(-1))/(preds.shape[-1]-k)
        out_probs = torch.einsum("j,jk->jk",rem_prob, torch.ones(preds.shape).to(device=preds.device))
        out_probs[mask.to(dtype=torch.bool)] = 1.0
        return out_probs
        
    def apply_transform(self, inps, transform, video=False):
        processed_inps = []
        if video:
            for inp in inps:
                if isinstance(inp, np.ndarray):
                    inp = torch.from_numpy(inp)
                inp = transform(inp)
                if isinstance(inp, list):
                    if len(processed_inps)==0:
                        processed_inps = [[] for _ in range(len(inp))]
                    for i in range(len(inp)):
                        processed_inps[i].append(inp[i])    
                elif isinstance(inp, torch.Tensor):
                    processed_inps.append(inp)
            if isinstance(processed_inps[0], list):
                processed_inps = [torch.stack(processed_inp, dim=0).to(device=self.dummy_param.device) for processed_inp in processed_inps]
            elif isinstance(processed_inps[0], torch.Tensor):
                processed_inps = torch.stack(processed_inps, dim=0).to(device=self.dummy_param.device)
        else:
            processed_inps = transform(inps).to(device=self.dummy_param.device)
        return processed_inps
            
        
    def check_video(self, inp):
        is_video = False
        if isinstance(inp, list):
            test_inp = inp[0]
        elif isinstance(inp, torch.Tensor):
            test_inp = inp[0,...]
        elif isinstance(inp, np.ndarray):
            test_inp = torch.from_numpy(inp[0,...])
            
        if isinstance(test_inp, torch.Tensor) and len(test_inp.shape)==4:
            is_video = True if len(test_inp.shape)==4 else False

        return is_video
    
    def forward(self, inp, transform=None, logits=None, topk=None):
        logits = logits or self.logits
        topk = topk or self.topk
        if transform or self.transform:
            processed_inp = self.apply_transform(inp, transform or self.transform, video=self.check_video(inp))
        else:
            if isinstance(inp, np.ndarray):
                inp = torch.from_numpy(inp)
            processed_inp = torch.clone(inp).to(device=self.dummy_param.device)
        with torch.no_grad():
            pred = self.model(processed_inp)
            preds = F.softmax(pred, dim=1)
        
        del processed_inp
        
        torch.cuda.empty_cache()
        gc.collect()
        if logits:
            return self.fetch_topk_logits(preds,topk)
        else:
            return self.fetch_topk_probs(preds,topk)
        
        
