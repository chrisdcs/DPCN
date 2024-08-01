import os
import numpy as np
import argparse
from pathlib import Path
import sys
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
#from torch.autograd import Variable
#from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from utils.datasets import conv_loader
import matplotlib.pyplot as plt
import numpy as np
from utils.general import make_patches
from utils.model import MM_Conv_Layer#, Correntropy


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd())) # relative path to current working directory

save_dir = ROOT / "runs" / "shrinkage" / "mario_conv" / "1"
if not save_dir.exists():
    save_dir.mkdir(parents=True, exist_ok=True)

torch.manual_seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    
# video data loader
loader = DataLoader(conv_loader('data/zca_mario.pt'), batch_size=32, shuffle=True, num_workers=4)

max_epochs = 10

n_ch = 16

layer1 = MM_Conv_Layer(in_ch=3, x_ch=n_ch, u_ch=n_ch, k_size=9, i_h=32, i_w=32, n_x=1, n_u=30, n_a=2, 
                       beta=0.05, gamma0=.5)
layer1 = layer1.to(device)

opt_C1 = torch.optim.SGD([{'params':layer1.C, 'lr':1e-3}])
opt_B1 = torch.optim.SGD([{'params':layer1.B, 'lr':1e-3}])

scheduler_c_1 = torch.optim.lr_scheduler.ExponentialLR(opt_C1, gamma=0.8)
scheduler_b_1 = torch.optim.lr_scheduler.ExponentialLR(opt_B1, gamma=0.8)

n_updates = 20

for epoch in range(max_epochs):
    
    for _, data in enumerate(loader):
        data = data.to(torch.float32).to(device)
        # padding the data manually
        inp = F.pad(data, (4, 4, 4, 4), mode='constant', value=0)
        # flatten the data to work with convolution toeplitz matrix
        inp = inp.flatten(1).unsqueeze(-1)
        x, u = layer1.inference(inp)
        layer1.reset()
        np.save(save_dir / "U.npy", u.detach().cpu().numpy())
        np.save(save_dir / "X.npy", x.detach().cpu().numpy())
        
        x, u = x.detach().clone().requires_grad_(True), u.detach().clone().requires_grad_(True)
        _, indices = layer1.pool(x.view(-1, n_ch, 32, 32))
        #xp = xp.flatten(1).unsqueeze(-1)
        #xp = xp.detach().clone().requires_grad_(True)
        # update dictionaries
        for _ in range(n_updates):
            recon, cause_state = layer1(x, u, indices)
            recon = recon.view(-1, 3, 40, 40)[:,:,4:-4,4:-4]
            loss_C = ((recon-data)**2).sum()
            opt_C1.zero_grad()
            loss_C.backward()
            opt_C1.step()
            layer1.normalize_weights()
            
            #_, cause_state = layer1(x, u, indices)
            
            loss_B = torch.sum(cause_state)
            opt_B1.zero_grad()
            loss_B.backward()
            opt_B1.step()
            layer1.normalize_weights()
            
        
        recon_cause = layer1.B @ u
        recon_cause = layer1.unpool(recon_cause.view(-1, n_ch, 24, 24)[:,:,4:-4,4:-4], indices)
        recon_cause = layer1.C @ recon_cause.flatten(1).unsqueeze(-1)
        recon_cause = recon_cause.view(-1, 3, 40, 40)[:,:,4:-4,4:-4]
        
        # save the reconstructions
        np.save(save_dir / "recon.npy", recon.detach().cpu().numpy())
        np.save(save_dir / "recon_cause.npy", recon_cause.detach().cpu().numpy())
        print(f"Epoch: {epoch+1}, Recon Loss: {loss_C.item()}, Cause Loss: {loss_B.item()}")
    scheduler_c_1.step()
    scheduler_b_1.step()