import os
import numpy as np
import argparse
import scipy.io as scio
from pathlib import Path
import sys
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
# import matplotlib.pyplot as plt
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd())) # relative path to current working directory

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def parse_opt():
    parser = argparse.ArgumentParser(description='DPCN Demo')
    parser.add_argument('--max_epoch', type=int, default=50, help='max epoch')
    
    
    return parser

def Correntropy(y, sigma):
    y = y.flatten(start_dim=1)
    batch_size, dim = y.shape
    batch_correntropy = torch.sum(torch.exp(-torch.square(y) / sigma**2), axis=-1) / dim
    return torch.sum(1 - batch_correntropy) / batch_size

class layer(torch.nn.Module):
    def __init__(self, input_dim, state_dim, cause_dim, state_size, cause_size, ksize, kstride, padding, sigma=0.5, lr=1e-2):
        super(layer, self).__init__()
        self.A = nn.Sequential(
            nn.Conv2d(state_dim, state_dim, ksize['A'], kstride['A'], padding['A'], bias=False), 
            nn.ReLU(),
            nn.Conv2d(state_dim, state_dim, ksize['A'], kstride['A'], padding['A'], bias=False),
            nn.ReLU()
        )
        self.B = nn.ConvTranspose2d(cause_dim, state_dim, ksize['B'], kstride['B'], padding['B'], bias=False)
        self.C = nn.ConvTranspose2d(state_dim, input_dim, ksize['C'], kstride['C'], padding['C'], bias=False)

        
        
        torch.nn.init.uniform_(self.A[0].weight,-0.01,0.01)        # initialize A
        torch.nn.init.uniform_(self.A[2].weight,-0.01,0.01)        # initialize A
        torch.nn.init.uniform_(self.B.weight,-0.1,0.1)        # initialize B
        torch.nn.init.uniform_(self.C.weight,-0.1,0.1)        # initialize C
        
        self.beta = nn.Parameter(torch.FloatTensor([1.]), requires_grad=False)         # namely the beta in eq. 5
        self.gamma = nn.Parameter(torch.FloatTensor([1.]), requires_grad=False)         # namely the gamma 0 in eq. 5
        self.lam = nn.Parameter(torch.FloatTensor([0.3]), requires_grad=False)          # namely the lambda in eq. 5
        self.sigma_x = nn.Parameter(torch.FloatTensor([sigma]), requires_grad=False)  # kernel size for correntropy
        self.sigma_u = nn.Parameter(torch.FloatTensor([sigma]), requires_grad=False)  # kernel size for correntropy
        self.sigma_t = nn.Parameter(torch.FloatTensor([sigma / 10]), requires_grad=False)  # kernel size for correntropy
        self.lr = lr
        
        self.state_dim = state_dim
        self.cause_dim = cause_dim
        
        self.state_size = state_size
        self.cause_size = cause_size
        
        self.test = []
    
    def reduce_kernel_size(self):
        self.sigma_x *= 0.9  # kernel size for correntropy
        self.sigma_u *= 0.9  # kernel size for correntropy
    
    def batch_inference(self, video):
        batch_size, c, h, w, T = video.shape
        X = []
        U = []
        xt_1 = torch.zeros((batch_size, self.state_dim, self.state_size, self.state_size), device = device)
        for t in range(T):
            if (t+1) % 100 == 0:
                    print('\t', t+1, 'frame')
            yt = video[:, :, :, :, t]
            xt, ut = self.infer(yt, xt_1)
            X.append(xt)
            U.append(ut)
            xt_1 = xt.clone()
        
        X = torch.stack(X, dim=-1)
        U = torch.stack(U, dim=-1)
        return X, U
        
    
    def infer(self, img, xt_1):
        batch_size, c, h, w = img.shape
        
        # initialize cause
        ut = torch.zeros((batch_size, self.cause_dim, self.cause_size, self.cause_size),
                          requires_grad=True, device=device)
        
        xt = self.A(xt_1) + xt_1
        xt_hat = xt.detach().clone().requires_grad_(True)
        xt = xt.detach().clone().requires_grad_(True)
        opt_x = torch.optim.Adam([xt], lr=self.lr)
        #opt_u = torch.optim.Adam([ut], lr=self.lr)
        for _ in range(10):
            Cxt = self.C(xt)
            gamma = self.gamma * (1 + torch.exp(-self.B(ut.detach().clone()))) / 2
            # self.lam * F.huber_loss(xt, xt_hat, 'sum', delta=0.001) + \
            E1 = F.mse_loss(img, Cxt) + \
                 self.lam * Correntropy(xt-xt_hat, self.sigma_t) + \
                 Correntropy(gamma * xt, self.sigma_x)
            self.test.append(E1.item())
            opt_x.zero_grad()
            E1.backward()
            opt_x.step()
        
        # optimize cause
        
        opt_u = torch.optim.Adam([ut], lr=self.lr)
        for _ in range(10):
            gamma = self.gamma * torch.exp(-self.B(ut)) / 2
            E2 = Correntropy(gamma * xt, self.sigma_x) + self.beta * Correntropy(ut, self.sigma_u)
            opt_u.zero_grad()
            E2.backward()
            opt_u.step()
            
        return xt, ut
    
    def fista(self, frame, state, cause, lamda, max_iter):
        pass
    
    def forward(self, xt_1, xt, ut):
        predicted_frame = self.C(xt)
        predicted_state = self.A(xt_1) + xt_1
        gamma = self.gamma * (1 + torch.exp(-self.B(ut))) / 2
        cause_state = gamma * xt

        return predicted_frame, predicted_state, cause_state

class DPCN(torch.nn.Module):
    def __init__(self, L):
        super(DPCN, self).__init__()
        # initialize one layer and then add layers if needed
        self.n_layers = 1
        self.layers = nn.ModuleList([L])
    
    def add_layer(self, L):
        self.layers.append(L)
        self.n_layers += 1
    
    def multi_layer_inference(self, video):
        # generate causes as the input for the last layer
        U = video.clone()
        for idx, layer in enumerate(self.layers[:-1]):
            print(f"inference for laye {idx + 1}: ")
            layer.eval()
            _, U = layer.batch_inference(U)
            
        return U
    
    def forward(self, xt_1, xt, ut):
        layer = self.layers[-1]
        predicted_frame, predicted_state, cause_state = layer(xt_1, xt, ut)
        
        return predicted_frame, predicted_state, cause_state
        

class video_loader(Dataset):
    def __init__(self, file_path):
        self.video = scio.loadmat(file_path)['M']
        # N: number of video sequences
        # T: number of frames for each video sequence
        self.N, self.T = self.video.shape
        self.T //= 2
        h, _ = self.video[0][0].shape
        self.frame_size = h
        
    def __getitem__(self, index):
        video = self.video[index]
        batch_sequence = torch.zeros((self.frame_size, self.frame_size, self.T))
        for i in range(self.T):
            batch_sequence[:, :, i] = torch.FloatTensor(video[i])
        return batch_sequence.unsqueeze_(0)
    
    def __len__(self):
        return self.N
    
# video data loader 
loader = DataLoader(video_loader('Video_train.mat'), batch_size=5, shuffle=True, num_workers=8)

max_epochs = 30

input_dim = [1, 4]
state_dim = [10, 4]
cause_dim = [10, 2]
state_size = [16, 4]
cause_size = [8, 4]
ksize = [{'A': 3, 'B': 2, 'C': 2}, {'A': 3, 'B': 3, 'C': 2}]
kstride = [{'A': 1, 'B': 2, 'C': 2}, {'A': 1, 'B': 1, 'C': 2}]
padding = [{'A': 1, 'B': 0, 'C': 0}, {'A': 1, 'B': 1, 'C': 0}]

layer1 = layer(input_dim[0], state_dim[0], cause_dim[0], state_size[0], cause_size[0], ksize[0], kstride[0], padding[0])

# checkpoint = torch.load(ROOT / "runs" / "checkpoint.pth.tar")
# layer1.load_state_dict(checkpoint['layer1_state_dict'])
# layer1.reduce_kernel_size()
layer1 = layer1.to(device)

save_dir = ROOT / "runs"
if not save_dir.exists():
    save_dir.mkdir()
    

opt_A = torch.optim.Adam(layer1.A.parameters(), lr=1e-3, weight_decay=1e-3)
opt_B = torch.optim.Adam([layer1.B.weight], lr=1e-3)
opt_C = torch.optim.Adam(layer1.C.parameters(), lr=1e-3)

sched_A = torch.optim.lr_scheduler.ExponentialLR(opt_A, gamma=0.9)
sched_B = torch.optim.lr_scheduler.ExponentialLR(opt_B, gamma=0.9)
sched_C = torch.optim.lr_scheduler.ExponentialLR(opt_C, gamma=0.9)
for epoch in range(max_epochs):
    loss_list = []
    print('Epoch: ', epoch+1)
    for _, data in enumerate(loader):
        data = data.to(device)
        print('\t', "inference: ")
        X, U = layer1.batch_inference(data)
        np.save(save_dir / "U.npy", U.detach().cpu().numpy()[0])
        np.save(save_dir / "X.npy", X.detach().cpu().numpy()[0])
        xt_1 = torch.zeros((data.shape[0], state_dim[0], state_size[0], state_size[0]), device = device, requires_grad=True)
        old_A = layer1.A[0].weight.detach().clone().requires_grad_(True)
        for t in range(X.shape[-1]):
            xt = X[:, :, :, :, t].detach().clone().requires_grad_(True)
            ut = U[:, :, :, :, t].detach().clone().requires_grad_(True)
            frame = data[:, :, :, :, t].detach().clone().requires_grad_(True)
            predicted_frame, predicted_state, weighted_state = layer1(xt_1, xt, ut)
            video_recon_loss = F.mse_loss(predicted_frame, frame)
            state_recon_loss = Correntropy(predicted_state-xt, layer1.sigma_t)#F.huber_loss(xt, predicted_state, 'sum', delta=1e-2)#F.l1_loss(xt, predicted_state)#
            
            sparsity_loss = Correntropy(weighted_state, layer1.sigma_x)
            
            opt_A.zero_grad()
            state_recon_loss.backward()
            opt_A.step()
            
            opt_B.zero_grad()
            sparsity_loss.backward()
            opt_B.step()
            
            video_recon_loss.backward()
            opt_C.step()
            opt_C.zero_grad()

            total_loss = video_recon_loss + state_recon_loss + sparsity_loss
            loss_list.append(total_loss.item())
            
            xt_1 = xt.detach().clone().requires_grad_(True)
        print('\ttotal loss:' , total_loss.item(), '\tvideo_recon_loss:', video_recon_loss.item(), '\tstate_recon_loss:', state_recon_loss.item(), '\tsparsity_loss:', sparsity_loss.item())
    print('Epoch Loss: ', np.mean(loss_list))

    torch.save({'epoch': epoch,
                'layer1_state_dict': layer1.state_dict(),
                #'optimizer_A_state_dict': opt_A.state_dict(),
                #'optimizer_B_state_dict': opt_B.state_dict(),
                #'optimizer_C_state_dict': opt_C.state_dict(),
                }, save_dir / 'checkpoint.pth.tar')
    if (epoch + 1) % 10 == 0:
        layer1.reduce_kernel_size()
        
    #sched_A.step()
    #sched_B.step()
    #sched_C.step()

"""
layer2 = layer(input_dim[1], state_dim[1], cause_dim[1], state_size[1], cause_size[1], ksize[1], kstride[1], padding[1], 2., 1e-3)
layer2 = layer2.to(device)
model = DPCN(layer1)
model.add_layer(layer2)

opt_A = torch.optim.Adam([model.layers[-1].A.weight], lr=1e-3)
opt_B = torch.optim.Adam([model.layers[-1].B.weight], lr=1e-3)
opt_C = torch.optim.Adam([model.layers[-1].C.weight], lr=1e-3)

for epoch in range(max_epochs):
    loss_list = []
    print('Epoch: ', epoch+1)
    for _, data in enumerate(loader):
        data = data.to(device)
        print('\t', "inference: ")
        data = model.multi_layer_inference(data)
        np.save(save_dir / "test_U.npy", data.detach().cpu().numpy()[0])
        print("inference for layer 2: ")
        X, U = model.layers[-1].batch_inference(data)
        np.save(save_dir / "U2.npy", U.detach().cpu().numpy()[0])
        np.save(save_dir / "X2.npy", X.detach().cpu().numpy()[0])
        xt_1 = torch.zeros((data.shape[0], state_dim[1], state_size[1], state_size[1]), device = device, requires_grad=True)
        for t in range(data.shape[-1]):
            xt = X[:, :, :, :, t].detach().clone().requires_grad_(True)
            ut = U[:, :, :, :, t].detach().clone().requires_grad_(True)
            frame = data[:, :, :, :, t].detach().clone().requires_grad_(True)
            predicted_frame, predicted_state, weighted_state = model(xt_1, xt, ut)
            video_recon_loss = F.mse_loss(predicted_frame, frame)
            state_recon_loss = F.mse_loss(predicted_state, xt)
            sparsity_loss = Correntropy(weighted_state, layer1.sigma_x)
            
            opt_A.zero_grad()
            state_recon_loss.backward()
            opt_A.step()
            
            opt_B.zero_grad()
            sparsity_loss.backward()
            opt_B.step()
            
            video_recon_loss.backward()
            opt_C.step()
            opt_C.zero_grad()

            total_loss = video_recon_loss + state_recon_loss + sparsity_loss
            
            loss_list.append(total_loss.item())
            
            xt_1 = xt.detach().clone().requires_grad_(True)
        print('\ttotal loss:' , total_loss.item(), '\tvideo_recon_loss:', video_recon_loss.item(), '\tstate_recon_loss:', state_recon_loss.item(), '\tsparsity_loss:', sparsity_loss.item())
    print('Epoch Loss: ', np.mean(loss_list))
    """
