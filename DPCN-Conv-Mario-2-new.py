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

save_dir = ROOT / "runs" / "shrinkage" / "mario_conv" / "conj_grad"

if not save_dir.exists():
    save_dir.mkdir(parents=True, exist_ok=True)
    
torch.manual_seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 32
# video data loader
loader = DataLoader(conv_loader('data/zca_mario.pt'), batch_size=batch_size, shuffle=True, num_workers=4)

max_epochs = 200

n_ch = 16
k_size = 7
pad = k_size // 2
layer1 = MM_Conv_Layer(in_ch=3, x_ch=n_ch, u_ch=n_ch, k_size=k_size, i_h=32, i_w=32, n_x=1, n_u=100, n_a=1, 
                       alpha = .5, beta=0.5, gamma0=1.)
layer1 = layer1.to(device)

layer2 = MM_Conv_Layer(in_ch=n_ch, x_ch=n_ch, u_ch=n_ch, k_size=k_size, i_h=16, i_w=16, n_x=1, n_u=100, n_a=1,
                       alpha = .5, beta=0.5, gamma0=1.)
layer2 = layer2.to(device)

opt_C1 = torch.optim.SGD([{'params':layer1.C_conv, 'lr':1e-3}])
opt_B1 = torch.optim.SGD([{'params':layer1.B_conv, 'lr':1e-3}])

opt_C2 = torch.optim.SGD([{'params':layer2.C_conv, 'lr':1e-3}])
opt_B2 = torch.optim.SGD([{'params':layer2.B_conv, 'lr':1e-3}])

scheduler_c_1 = torch.optim.lr_scheduler.ExponentialLR(opt_C1, gamma=0.94)
scheduler_b_1 = torch.optim.lr_scheduler.ExponentialLR(opt_B1, gamma=0.94)
scheduler_c_2 = torch.optim.lr_scheduler.ExponentialLR(opt_C2, gamma=0.94)
scheduler_b_2 = torch.optim.lr_scheduler.ExponentialLR(opt_B2, gamma=0.94)

n_updates = 20

for epoch in range(1,max_epochs+1):
    
    for i, data in enumerate(loader):
        data = data.to(torch.float32).to(device)
        # padding the data manually
        #inp = F.pad(data, (pad, pad, pad, pad), mode='constant', value=0)
        # flatten the data to work with convolution toeplitz matrix
        #inp = inp.flatten(1).unsqueeze(-1)
        
        for t in range(2):
            if t == 0:
                Td_1, Td_2 = None, None
            else:
                Td_1 = layer2.C @ x2
                # Td_1 = Td_1.clone().detach().clip(0)
                Td_1 = Td_1.view(-1, n_ch, 16+pad*2, 16+pad*2)[:,:,pad:-pad,pad:-pad]#.flatten(1).unsqueeze(-1)
                
                Td_2 = u2.clone().detach()
            x1, u1 = layer1.inference(data, Td_1)
            # pad u1 for inputting to layer2
            #u1_to_next = F.pad(u1.view(-1, n_ch, 16, 16), (pad, pad, pad, pad), mode='constant', value=0)
            #u1_to_next = u1_to_next.flatten(1).unsqueeze(-1)
            u1_to_next = u1.clone()
            x2, u2 = layer2.inference(u1_to_next, Td_2)
        layer1.reset()
        layer2.reset()
        
        x1, u1 = x1.detach().clone().requires_grad_(True), u1.detach().clone().requires_grad_(True)
        x1 = x1.view(-1, n_ch, 32, 32)
        #u1 = u1.view(-1, n_ch, 16, 16)
        #_, indices1 = layer1.pool(x1.view(-1, n_ch, 32, 32))
        #_, indices1 = layer1.pool(x1)
        
        x2, u2 = x2.detach().clone().requires_grad_(True), u2.detach().clone().requires_grad_(True)
        x2 = x2.view(-1, n_ch, 16, 16)
        #u2 = u2.view(-1, n_ch, 8, 8)
        #_, indices2 = layer2.pool(x2.view(-1, n_ch, 16, 16))
        #_, indices2 = layer2.pool(x2)
        
        # update dictionaries
        for _ in range(n_updates):
            recon1, cause_state1 = layer1(x1, u1)
            # recon1 = recon1.view(-1, 3, 40, 40)[:,:,4:-4,4:-4]
            loss_C1 = ((recon1-data)**2).sum()
            opt_C1.zero_grad()
            loss_C1.backward()
            # layer1.zero_grad_dict_C()
            opt_C1.step()
            layer1.normalize_weights(skip_cause=True)
        for _ in range(n_updates*5):
            recon1, cause_state1 = layer1(x1, u1)
            loss_B1 = torch.sum(cause_state1)
            opt_B1.zero_grad()
            loss_B1.backward()
            # layer1.zero_grad_dict_B()
            opt_B1.step()
            layer1.normalize_weights(skip_state=True)
        for _ in range(n_updates):
            recon2, cause_state2 = layer2(x2, u2)
            # recon2 = recon2.view(-1, n_ch, 24, 24)[:,:,4:-4,4:-4]
            loss_C2 = ((recon2-u1)**2).sum()
            opt_C2.zero_grad()
            loss_C2.backward()
            #layer2.zero_grad_dict_C()
            opt_C2.step()
            layer2.normalize_weights(skip_cause=True)
        for _ in range(n_updates*5):
            recon2, cause_state2 = layer2(x2, u2)
            loss_B2 = torch.sum(cause_state2)
            opt_B2.zero_grad()
            loss_B2.backward()
            #layer2.zero_grad_dict_B()
            opt_B2.step()
            layer2.normalize_weights(skip_state=True)
            
            #layer1.normalize_weights()
            #layer2.normalize_weights()
        
        layer1.update_dict()
        layer2.update_dict()
        
        if data.shape[0] == batch_size and (i+1) % 5 == 0:
            with torch.no_grad():
                recon2 = F.conv_transpose2d(recon2, layer1.B_conv, padding=pad)
                recon2 = layer1.unpool(recon2, layer1.indices)
                recon2 = F.conv_transpose2d(recon2, layer1.C_conv, padding=pad)#.clip(0)
                
                recon_cause1 = F.conv_transpose2d(u1, layer1.B_conv, padding=pad)
                recon_cause1 = layer1.unpool(recon_cause1, layer1.indices)
                recon_cause1 = F.conv_transpose2d(recon_cause1, layer1.C_conv, padding=pad)#.clip(0)
                
                recon_cause2 = F.conv_transpose2d(u2, layer2.B_conv, padding=pad)
                recon_cause2 = layer2.unpool(recon_cause2, layer2.indices)
                recon_cause2 = F.conv_transpose2d(recon_cause2, layer2.C_conv, padding=pad)
                recon_cause2 = recon_cause2#.clip(0)
                recon_cause2 = F.conv_transpose2d(recon_cause2, layer1.B_conv, padding=pad)
                recon_cause2 = layer1.unpool(recon_cause2, layer1.indices)
                recon_cause2 = F.conv_transpose2d(recon_cause2, layer1.C_conv, padding=pad)#.clip(0)
                
                np.save(save_dir / "recon_x1.npy", recon1.detach().cpu().numpy())
                np.save(save_dir / "recon_x2.npy", recon2.detach().cpu().numpy())
                np.save(save_dir / "recon_cause1.npy", recon_cause1.detach().cpu().numpy())
                np.save(save_dir / "recon_cause2.npy", recon_cause2.detach().cpu().numpy())
                
                np.save(save_dir / "U1.npy", u1.detach().cpu().numpy())
                np.save(save_dir / "X1.npy", x1.detach().cpu().numpy())
                np.save(save_dir / "U2.npy", u2.detach().cpu().numpy())
                np.save(save_dir / "X2.npy", x2.detach().cpu().numpy())
                
                
                v1 = []
                v2 = []
                for show_i in range(n_ch):
                    x_show1 = x1.clone().detach()
                    maps = torch.zeros(x_show1.shape, device=device)
                    maps[:, show_i, :, :] = 1
                    x_show1 = x_show1 * maps
                    filter1 = F.conv_transpose2d(x_show1, layer1.C_conv, padding=pad)
                    v1.append(filter1)
                    
                    x_show2 = x2.clone().detach()
                    maps = torch.zeros(x_show2.shape, device=device)
                    maps[:, show_i, :, :] = 1
                    x_show2 = x_show2 * maps
                    filter2 = F.conv_transpose2d(x_show2, layer2.C_conv, padding=pad)
                    filter2 = F.conv_transpose2d(filter2, layer1.B_conv, padding=pad)
                    filter2 = layer1.unpool(filter2, layer1.indices)
                    filter2 = F.conv_transpose2d(filter2, layer1.C_conv, padding=pad)
                    v2.append(filter2)
                    
                torch.save(v1, save_dir / "v1.pt")
                torch.save(v2, save_dir / "v2.pt")
                torch.save(layer1.state_dict(), save_dir / "layer1.pth.tar")
                torch.save(layer2.state_dict(), save_dir / "layer2.pth.tar")
            
        print(f"Epoch {epoch}, Recon Loss 1: {loss_C1.item()}, Cause Loss1: {loss_B1.item()}, Recon Loss 1: {loss_C2.item()}, Cause Loss 2: {loss_B2.item()}")#, u1.max())