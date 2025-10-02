import numpy as np
import torch

data = torch.load("/home/ncaytuir/data-local/TreeGAN/MyScripts/ckpt_2899/samples.pth")

# Ver qué contiene
print(type(data))
#print(data.keys())
print(data.shape)

if isinstance(data, dict):
    for k, v in data.items():
        print(f"{k}: {type(v)}, shape: {getattr(v, 'shape', 'N/A')}")
elif isinstance(data, list):
    print(f"Es una lista con {len(data)} elementos")
    for i, v in enumerate(data[:5]):
        print(f"Elemento {i}: {type(v)}, shape: {getattr(v, 'shape', 'N/A')}")