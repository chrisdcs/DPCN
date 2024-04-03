import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import grad

from torch.utils.data import Dataset, DataLoader


def soft_threshold(x, lambd):
    return torch.sign(x) * torch.max(torch.abs(x) - lambd, torch.zeros_like(x))

def hard_threshold(x, lambd):
    return x * (torch.abs(x) > lambd).float()


# MLCSC6 returns the reconstructed image from the latent space (n_batch, 1024, 2, 2)
class MLCSC6(nn.Module):
    def __init__(self, n_layers):
        super(MLCSC6, self).__init__()
        
        D1 = nn.ConvTranspose2d(32, 3, kernel_size=6, stride=2, padding=0)
        D2 = nn.ConvTranspose2d(128, 32, kernel_size=3, stride=1, padding=0)
        D3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=0)
        D4 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=0)
        D5 = nn.ConvTranspose2d(768, 512, kernel_size=3, stride=1, padding=0)
        D6 = nn.ConvTranspose2d(1024, 768, kernel_size=5, stride=1, padding=0)
        
        #self.layers = [D6, D5, D4, D3, D2, D1]
        self.layers = [D6, D5, D4, D3, D2, D1]
        self.n_layers = n_layers
        self.layers = nn.ModuleList(self.layers[6-n_layers:])
        self.strides = [1, 1, 1, 1, 1, 2][6-n_layers:]
        self.ksizes = [5, 3, 3, 3, 3, 6][6-n_layers:]
    
    def F(self, x, lambd = 0.1):
        return torch.div(torch.square(x).flatten(1).sum(1), 2) + lambd * torch.abs(x).flatten(1).sum(1)
    
    def initialize(self, x):
        
        for i in range(self.n_layers-1, -1, -1):
            x = F.conv2d(x, self.layers[i].weight, stride=self.strides[i], padding=0)
            #x = self.layers[i](x)
            #print(x.shape)
        #pass
        return x
    
    def FISTA(self, x, y, lambd = 0.1, n_iter = 100):
        batch_size = x.size(0)
        Dx = self.forward(x)
        
        gradient = Dx-y
        for i in range(self.n_layers, 0, -1):
            gradient = F.conv2d(gradient, self.layers[i-1].weight, stride=self.strides[i-1], padding=0)
        tk, tk_next = torch.tensor(1., device = x.device), torch.tensor(1., device = x.device)
        loss_list = []
        for _ in range(n_iter):
            z = x.clone()
            const = self.F(z, lambd).reshape(-1, 1, 1, 1)
            
            L = torch.ones((batch_size, 1, 1, 1), device = x.device)
            stop_line_search = torch.zeros((batch_size), device=x.device).bool()
            while torch.sum(stop_line_search) < batch_size:
                # line search
                # print(z.shape, gradient.shape, L.shape)
                prox_z = soft_threshold(z - torch.div(gradient, L), torch.div(lambd, L))
                
                # check descent condition
                temp1 = self.F(prox_z, lambd).reshape(-1, 1, 1, 1)
                temp2 = const + torch.mul(gradient, prox_z - z).flatten(1).sum(1).reshape(-1, 1, 1, 1) + \
                                torch.div(L, 2) * torch.square(prox_z - z).flatten(1).sum(1).reshape(-1, 1, 1, 1)
                stop_line_search = temp1 <= temp2
                L = torch.where(stop_line_search, L, 2 * L)
            
            tk_next = (1 + torch.sqrt(1 + 4 * tk**2)) / 2
            x = prox_z + torch.div(tk - 1, tk_next) * (prox_z - z)
            tk = tk_next
            loss_list.append(torch.mean(self.F(x, lambd)).item())
        
        return x, loss_list
    
    def IHT(self, lambds):
        # lambd2, lambd3 = 0.005, 0.01#0.01, 0.005, 0.01
        # self.layer1.weight = hard_threshold(self.layer1.weight, lambd1)
        # self.layer2.weight = nn.Parameter(hard_threshold(self.layer2.weight, lambd2))
        # self.layer3.weight = nn.Parameter(hard_threshold(self.layer3.weight, lambd3))
        for i in range(1, self.n_layers):
            self.layers[i].weight = nn.Parameter(hard_threshold(self.layers[i].weight, lambds[i-1]))
            #print(torch.sum(self.layers[i].weight.flatten()==0) / torch.numel(self.layers[i].weight))
    
    def forward(self, x):
        for i in range(self.n_layers):
            x = self.layers[i](x)
        
        return x
    
class dataset(Dataset):
    def __init__(self):
        self.data = np.load('zca_images.npy')
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_layers = 6
model = MLCSC6(n_layers)
model = model.to(device)

cifar10 = dataset()
cifar_loader = DataLoader(cifar10, batch_size=32, shuffle=True, num_workers=8)
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.001)

print(f"training for ML-CSC ({n_layers} layers)")

for epoch in range(1,11):
    loss_list = []
    for batch_id, data in enumerate(cifar_loader):
        data = data.cuda()
        # code = torch.randn(data.size(0), 1024, 10, 10).cuda()
        
        with torch.no_grad():
            code = model.initialize(data)
            code = torch.randn_like(code)
            x, _ = model.FISTA(code, data, lambd=0.03, n_iter=10)
        
        optimizer.zero_grad()
        x = x.clone().requires_grad_(True)
        output = model(x)
        loss = F.mse_loss(output, data)
        
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        # iterative hard thresholding step
        # model.IHT()
        # model.IHT([0.001, 0.005, 0.01, 0.02, 0.03])
        model.IHT([0.01, 0.015, 0.025, 0.04, 0.055][5-n_layers+1:])
        
        if batch_id % 200 == 0:
            l1_ratio = [(torch.sum(model.layers[i].weight.flatten()==0) / torch.numel(model.layers[i].weight)).item() for i in range(1, model.n_layers)]
            print("loss", "{:.4f}".format(loss.item()), "sparsity:", ["{0:0.2f}".format(i) for i in l1_ratio])
            # print([np.abs(model.layers[i].weight.cpu().detach().numpy().flatten()).max() for i in range(model.n_layers)])
    print(f'epoch {epoch}, batch {batch_id}, loss {np.mean(loss_list)}\n')
    
    
    # if epoch % 10 == 0:
torch.save(model.state_dict(), f'model_cifar_{n_layers}layers.pt')