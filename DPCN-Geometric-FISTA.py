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

from utils.datasets import video_loader, make_patches
import matplotlib.pyplot as plt
import numpy as np
from utils.model import FISTA_Layer


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd())) # relative path to current working directory

save_dir = ROOT / "runs" / "FISTA" / "geometric"
if not save_dir.exists():
    save_dir.mkdir()

torch.manual_seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    
# video data loader
loader = DataLoader(video_loader('data/Video_train.mat'), batch_size=5, shuffle=False, num_workers=1)

max_epochs = 10

layer1 = FISTA_Layer(n_ch=1, gamma0=1., mu=0.01/150, lam=0.5, beta=0.5, X_dim=300, U_dim=40, patch_size=16, 
                     isTopLayer=True, input_size=None, n_steps=20)
layer1 = layer1.to(device)
opt_A = torch.optim.SGD([{'params':layer1.A, 'lr':1e-4}])
opt_B = torch.optim.SGD([{'params':layer1.B, 'lr':1e-4}])
opt_C = torch.optim.SGD([{'params':layer1.C, 'lr':1e-4}])

for epoch in range(max_epochs):
    loss_list = []
    print("Epoch: ", epoch+1)
    for _, data in enumerate(loader):
        data = data.to(device)
        print('\t', "inference: ")
        # print(data.shape)
        X, U = layer1.inference(data)
        np.save(save_dir / "U.npy", U.detach().cpu().numpy()[0])
        np.save(save_dir / "X.npy", X.detach().cpu().numpy()[0])
        X, U = X.detach().clone().requires_grad_(True), U.detach().clone().requires_grad_(True)
        for _ in range(5):
            x_t_minus_1 = torch.zeros_like(X[:, :, :, :, :, 0])
            video_recon, X_hat, X_U = layer1(X, U, x_t_minus_1)
            np.save(save_dir / "recon.npy", video_recon.detach().cpu().numpy()[0])
            
            vid_recon_loss = torch.sum(torch.square(video_recon-data))#F.mse_loss(video_recon, data)
            opt_C.zero_grad()
            vid_recon_loss.backward()
            opt_C.step()
            
            X_pred_loss = torch.sum(torch.abs(X_hat-X))
            opt_A.zero_grad()
            X_pred_loss.backward()
            opt_A.step()
            
            X_U_loss = torch.sum(X_U)#F.mse_loss(X_recon, X)
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
torch.save(layer1.state_dict(), save_dir / "layer1.pth.tar")