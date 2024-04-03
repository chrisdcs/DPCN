import torch
import matplotlib.pyplot as plt
import numpy as np

dim = 10
x = torch.rand((dim,1))
A = torch.rand((dim,dim))
z = A @ x

loss_list = []

def L0(y, sigma):
    return 1 - Correntropy(y, sigma)

def Correntropy(y, sigma):
    d = y.shape[0]
    return torch.sum(torch.exp(-torch.square(y) / sigma**2)) / d



x_ori = (A.T @ A).inverse() @ A.T @ z
#x = A.inverse() @ z
#print('ground truth', z)
#print('prediction', A @ x)
#print('coefficients', x)
proj = A.T @ (A @ A.T).inverse() @ A
I = torch.eye(dim)

sig_max = 5
sig = sig_max
step = (0.001 / sig)**(1/10)
T = 10000
loss = float('inf')
loss_list = []
lr = 1e-3
for t in range(1,T+1):
    g = x / sig**2 * torch.exp(-torch.square(x) / sig**2 / 2)
    #print('step', t)
    if t % 100 == 0:
        sig = sig_max * np.exp(-10 * t/T) + 0.001
    g = (I-proj) @ g
    g = g / (torch.norm(g)+1e-12)
    x = x - lr * g
    loss = Correntropy(x, sig)
    loss_list.append(loss)
    # x -= 1e-3 * A.T @ (A @ x - z)

print('x')
print(x)
print('kernel size', sig)

print('predict')
print(A@x)
print('ground truth')
print(z)

_ = plt.hist(x.numpy().flatten(), bins=10)
plt.show()