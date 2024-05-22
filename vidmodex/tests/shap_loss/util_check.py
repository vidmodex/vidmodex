import torch

import gc
import torch.multiprocessing as mp
mp.set_start_method("forkserver")

def fetch_topk_probs(preds, k=5):
    topk_probs, topk_idxs = preds.topk(k)
    mask = torch.zeros_like(preds, dtype=torch.bool)
    mask.scatter_(1, topk_idxs, True)
    # mask = torch.zeros_like(preds)
    # mask[torch.arange(mask.size(0)), topk_idxs] = 1.
    rem_prob = (1. - topk_probs.sum(-1))/(preds.shape[-1]-k)
    out_probs = torch.einsum("j,jk->jk",rem_prob, torch.ones(preds.shape).to(device=preds.device))
    out_probs[mask.to(dtype=torch.bool)] = preds[mask.to(dtype=torch.bool)]
    # out_probs.index_put_((topk_idxs[], topk_idxs[]), topk_probs)
    # print(torch.index_select(out_probs, 1, topk_idxs))
    return out_probs


def call_vid(videos, model, transform, device):
    input_slow = []
    input_fast = []
    print("batch_size", videos.shape[0])
    for i in range(videos.shape[0]):
        vid = {"video":torch.from_numpy(videos[i,...])}
        video_dict = transform(vid)
        inputs = video_dict["video"]
        input_slow.append(inputs[0].to(device))
        input_fast.append(inputs[1].to(device))
        
    input_slow = torch.stack(input_slow, dim=0).to(device)
    input_fast = torch.stack(input_fast, dim=0).to(device)
    with torch.no_grad():
        pred = model([input_slow, input_fast])
        post_act = torch.nn.Softmax(dim=1)
        preds = post_act(pred)
    
    torch.cuda.empty_cache()
    gc.collect()
    return fetch_topk_probs(preds,5).cpu().numpy()

def fetch_transformed_video(inputs):
    print(inputs["video"].shape)
    vid = {"video":torch.from_numpy(inputs["video"])}
    video_dict = inputs["transform"](vid)
    inputs = video_dict["video"]
    return [inputs[0].cpu().numpy(), inputs[1].cpu().numpy()]

def fast_call_vid(input_all):
    videos, model, transform, device = input_all
    with mp.Pool(min(64, videos.shape[0])) as p:
        results = p.map(fetch_transformed_video, [ {"video":videos[i,...], "transform":transform} for i in range(videos.shape[0])])
    print(results)
    input_slow = torch.stack([torch.from_numpy(r[0]) for r in results], dim=0).to(device)
    input_fast = torch.stack([torch.from_numpy(r[1]) for r in results], dim=0).to(device)
    with torch.no_grad():
        pred = model([input_slow, input_fast])
        post_act = torch.nn.Softmax(dim=1)
        preds = post_act(pred)
    
    torch.cuda.empty_cache()
    gc.collect()
    return fetch_topk_probs(preds,5).cpu().numpy()
    