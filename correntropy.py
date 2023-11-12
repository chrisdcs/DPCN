import torch
import matplotlib.pyplot as plt

x = torch.ones((3,), requires_grad=True, device="cuda")
#x = x.cuda()
A = torch.FloatTensor([1,5,5])
A = A.cuda()

loss_list = []

def L0(y, sigma):
    return 1 - Correntropy(y, sigma)

def Correntropy(y, sigma):
    d = y.shape[0]
    return torch.sum(torch.exp(-torch.square(y) / sigma**2)) / d

opt = torch.optim.Adam([x], lr=0.005)
for i in range(500):
    y = A @ x
    loss = L0(x, 0.5) + torch.norm(1 - y)
    loss_list.append(loss.item())
    loss.backward()
    opt.step()
    opt.zero_grad()
    
print(x)
print(A @ x)
print(min(loss_list))
plt.plot(loss_list)
plt.show()