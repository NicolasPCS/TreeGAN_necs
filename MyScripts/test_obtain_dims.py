import torch

import os, sys
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(REPO_ROOT)

from evaluation.pointnet import PointNetCls

# Load PointNet
model = PointNetCls(k=16)
model.load_state_dict(torch.load('/home/ncaytuir/data-local/TreeGAN/evaluation/cls_model_39.pth', map_location='cpu'))
model.eval()

x = torch.randn(2, 1024, 3)        # (B,N,3)
with torch.no_grad():
    out = model(x.transpose(1,2))  # PointNet suele esperar (B,3,N)
    _, _, actv = out               # en ese repo suelen devolver (logits, trans, actv)
    D = actv.view(2, -1).shape[1]
print("dims =", D)    