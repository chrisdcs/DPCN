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

from utils.datasets import video_loader, mario_loader
import matplotlib.pyplot as plt
import numpy as np
from utils.general import make_patches
from utils.model import MM_Layer


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd())) # relative path to current working directory

save_dir = ROOT / "runs" / "shrinkage" / "youtube"
if not save_dir.exists():
    save_dir.mkdir(parents=True, exist_ok=True)

torch.manual_seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    
# video data loader
loader = DataLoader(mario_loader('data/youtube_video_train.npy'), batch_size=1, shuffle=False, num_workers=1)

max_epochs = 10

layer1 = MM_Layer(n_ch=1, lam=1., gamma0=0.3, mu=0.0006, beta=0.3, n_u=50, multi_dict=True, state_size=1000,
                         cause_size=80, patch_size=16)
layer1 = layer1.to(device)
layer1.load_state_dict(torch.load(save_dir / "layer1.pth.tar"))
layer1.eval()
layer2 = MM_Layer(n_ch=1, lam=1., gamma0=0.3, mu=0.001, beta=0.3, n_u=20, multi_dict=False, state_size=200,
                         cause_size=20, input_size=80, l_idx=2)
layer2.to(device)

opt_A = torch.optim.SGD([{'params':layer2.A, 'lr':1e-4}])
opt_B = torch.optim.SGD([{'params':layer2.B, 'lr':1e-4}])
opt_C = torch.optim.SGD([{'params':layer2.C, 'lr':1e-4}])

for epoch in range(max_epochs):
    loss_list = []
    print("Epoch: ", epoch+1)
    for _, data in enumerate(loader):
        data = data.to(torch.float32).to(device)
        print('\t', "Layer 1 Inference: ")
        X, U = layer1.inference(data)
        print('\t', "Layer 2 Inference: ")
        X1, U1 = layer2.inference(U)
        np.save(save_dir / "U1.npy", U1.detach().cpu().numpy()[0])
        np.save(save_dir / "X1.npy", X1.detach().cpu().numpy()[0])
        X1, U1 = X1.detach().clone().requires_grad_(True), U1.detach().clone().requires_grad_(True)
        for _ in range(3):
            x_t_minus_1 = torch.randn_like(X1[:, :, :, :, :, 0])
            video_recon_1, X_hat_1, X_U_1 = layer2(X1, U1, x_t_minus_1)
            #np.save(save_dir / "recon.npy", video_recon_1.detach().cpu().numpy()[0])
            
            vid_recon_loss = torch.sum(torch.square(video_recon_1-U))#F.mse_loss(video_recon, data)
            opt_C.zero_grad()
            vid_recon_loss.backward()
            opt_C.step()
            
            X_pred_loss = torch.sum(torch.abs(X_hat_1-X1))
            opt_A.zero_grad()
            X_pred_loss.backward()
            opt_A.step()
            
            X_U_loss = torch.sum(X_U_1)#F.mse_loss(X_recon, X)
            opt_B.zero_grad()
            X_U_loss.backward()
            opt_B.step()
            
            
            total_loss = vid_recon_loss + X_pred_loss + X_U_loss
            loss_list.append(total_loss.item())
        layer1.normalize_weights()
        print('\ttotal loss:' , '{:.2f}'.format(total_loss.item()), 
              '\tvideo_recon_loss:', '{:.2f}'.format(vid_recon_loss.item()), 
              '\tstate_pred_loss:', '{:.2f}'.format(X_pred_loss.item()), 
              '\tE2:', '{:.2f}'.format(X_U_loss.item()))
    print('Epoch Loss: ', np.mean(loss_list))
torch.save(layer2.state_dict(), save_dir / "layer2.pth.tar")