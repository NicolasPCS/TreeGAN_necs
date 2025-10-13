import torch
import numpy as np
import argparse

import os, sys
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(REPO_ROOT)

from evaluation.FPD import calculate_fpd

def load_pcs_from_pth(path):

    obj = torch.load(path, map_location='cpu')
    pcs = None

    if isinstance(obj, torch.Tensor):
        pcs = obj
        if pcs.dim() != 3:
            raise ValueError("Expected a 3D tensor (B, N, 3) but got shape {}".format(pcs.shape))
        if pcs.size(-1) != 3 and pcs.size(1) == 3:
            pcs = pcs.transpose(1,2) # -> (B,N,3)
    
    else:
        raise ValueError("Unsupported data format: {}".format(type(obj)))
    
    pcs = pcs.float().contiguous()
    return pcs  # (B,N,3)

def normalize_sphere(pcs, eps=1e-8):
    center = pcs.mean(dim=1, keepdim=True)  # (B,1,3)
    pcs = pcs - center
    radius = pcs.norm(dim=-1).max(dim=1, keepdim=True)[0]  # (B,1)
    pcs = pcs / (radius.unsqueeze(-1) + eps)
    return pcs

def normalize(pcs, eps=1e-8):
    p_min = pcs.min(dim=1, keepdim=True)[0]  # (B,1,3)
    p_max = pcs.max(dim=1, keepdim=True)[0]  # (B,1,3)
    center = (p_min + p_max) / 2  # (B,1,3)
    pcs = pcs - center

    scale = (p_max - p_min).max(dim=-1, keepdim=True)[0]  # (B,1)

    pcs = pcs / (scale / 2 + eps)
    return pcs

def choose_bs(b, pref=50):
    pref = min(pref, b)
    if b % pref == 0:
        return pref
    for bs in range(pref, 0, -1):
        if b % bs == 0:
            return bs
    return 1

def compute_fpd_from_pth(gen_pth, g_class="airplane", dims=1808, prefer_batch=100, device=None):
    pcs = load_pcs_from_pth(gen_pth)
    pcs = normalize(pcs)
    b = pcs.shape[0]
    #bs = choose_bs(b, prefer_batch)

    if g_class == "airplane":
        pcs = pcs[:404, :, :]  # use first 404 samples for airplane
    elif g_class == "car":
        pcs = pcs[:346, :, :]  # use first 346 samples for car
    elif g_class == "chair":
        pcs = pcs[:637, :, :]  # use first 637 samples for chair

    fpd = calculate_fpd(pcs, pointclouds2=None, batch_size=prefer_batch, dims=dims, device=device)

    return fpd, prefer_batch, b

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-g', '--gen_pth', type=str, required=True)
    ap.add_argument("--device", default="cuda:0", type=str)
    ap.add_argument("--g_class", default="airplane", type=str)
    args = ap.parse_args()

    device = torch.device(args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu")

    fpd, bs, b = compute_fpd_from_pth(args.gen_pth, g_class=args.g_class, device=device)

    print(f"[EVAL] B={b} BS={bs} FPD={fpd:.6f}")

# python ComputeFPD.py --gen_pth /home/ncaytuir/TreeGAN_necs/MyScripts/chair/ckpt_1199/reference.pth --device cpu --g_class chair