import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from utils.general import make_patches
from utils.general import S, patch_to_image, soft_thresholding, toeplitz_mult_ch, conjugate_gradient

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MM_Layer(nn.Module):
    def __init__(self, *args, **kwargs):
        super(MM_Layer, self).__init__()
        
        self.n_ch = kwargs['n_ch']
        
        if "state_size" in kwargs: 
            self.state_size = kwargs['state_size']
        else:
            self.state_size = 300
            
        if "cause_size" in kwargs:
            self.cause_size = kwargs['cause_size']
        else:
            self.cause_size = 40
        
        if "patch_size" in kwargs:
            self.patch_size = kwargs['patch_size']
            self.n_patch = (32 // self.patch_size) ** 2
        else:
            self.patch_size = 16
            self.n_patch = 4
            
        if "multi_dict" in kwargs and kwargs['multi_dict']:
            self.n_dict = self.n_patch
        else:
            self.n_dict = 1

        
        C = torch.rand(1, self.n_ch, self.n_dict, self.patch_size**2, self.state_size)#.unsqueeze(0).unsqueeze(0)
        C = C / C.norm(dim=-2, keepdim=True)
        
        B = torch.rand(1, self.n_ch, 1, self.state_size, self.cause_size)#.unsqueeze(0).unsqueeze(0)
        B = B / B.norm(dim=-2, keepdim=True)
        
        A = torch.rand(1, self.n_ch, self.n_dict, self.state_size, self.state_size)#.unsqueeze(0).unsqueeze(0)
        A = A / A.norm(dim=-2, keepdim=True)
        
        self.A = nn.Parameter(A)
        self.B = nn.Parameter(B)
        self.C = nn.Parameter(C)
       
        I = torch.eye(C.shape[-2]).unsqueeze(0).unsqueeze(0)
        self.I_x = nn.Parameter(I, requires_grad=False)
        I = torch.eye(B.shape[-2]).unsqueeze(0).unsqueeze(0)
        self.I_u = nn.Parameter(I, requires_grad=False)
        
        self.n_x = kwargs['n_x']
        self.n_u = kwargs['n_u']
        
        self.use_A = kwargs['use_A']
        
        lam, gamma0, mu, beta =kwargs['lam'], kwargs['gamma0'], kwargs['mu'], kwargs['beta']
        # weight for the transition error term
        self.lam = nn.Parameter(torch.tensor(lam), requires_grad=False)
        # weight for the state-cause correlation term
        self.gamma0 = nn.Parameter(torch.tensor(gamma0), requires_grad=False)
        # parameter for the smoothed transition error
        self.mu = nn.Parameter(torch.tensor(mu), requires_grad=False)
        # weight for the sparsity of causes
        self.beta = nn.Parameter(torch.tensor(beta), requires_grad=False)
        
        self.device = torch.device("cpu") if device is None else device
        
    def normalize_weights(self):
        with torch.no_grad():
            self.A.data = self.A.data / self.A.data.norm(dim=-2, keepdim=True)
            self.B.data = self.B.data / self.B.data.norm(dim=-2, keepdim=True)
            self.C.data = self.C.data / self.C.data.norm(dim=-2, keepdim=True)

    def inference(self, data):
        batch_size, c, h, w, T = data.shape
        
        X = []
        U = []
        with torch.no_grad():
            for t in range(T):
                if (t+1) % 100 == 0:
                    print('\t', t+1, 'frame')
                    
                # loop over frames in the data
                y_t = data[:, :, :, :, t]
                
                # separate patches from an entire image
                patches_t = make_patches(y_t, self.patch_size)
                # patches_t: (batch_size, 4, 256, 1) if gray scale images
                # patches_t: (batch_size, 3, 4, 256, 1) if RGB images
                if t == 0 or not self.use_A:
                    x_prev = torch.randn_like(self.C.transpose(-1,-2) @ patches_t)
                x_t, u_t = self.AM(patches_t, x_prev)
                X.append(x_t)
                U.append(u_t)
                x_prev = x_t.clone()
            
        X = torch.stack(X, dim=-1)
        U = torch.stack(U, dim=-1)
        return X, U
    
    def AM(self, patches, x_t_minus_1):
        # alternating minimization algorithm
        
        x_hat = self.A @ x_t_minus_1# + x_t_minus_1
        x_t = x_hat.clone()
        u_t = torch.rand_like(self.B.transpose(-1,-2) @ torch.mean(x_t, dim=2, keepdim=True))
        u_t = u_t / u_t.norm(dim=-2, keepdim=True)
        Cy = self.C.transpose(-1,-2) @ patches
        beta = self.beta
        
        for _ in range(5):
            # emperically showed that shrinkage algorithm converges/produces good solution in 5 iterations
            x_t = self.update_state(x_t.clone(), x_hat.clone(), u_t.clone(), Cy)
            u_t, beta = self.update_cause(x_t.clone(), u_t.clone(), beta)
            #u_t = self.update_cause(x_t)
            #cause = self.update_cause_FISTA(x_t, cause)
            
        return x_t, u_t
    
    def update_state(self, x_k, x_hat, u, Cy):
        # x: previous state, shape (batch, n_patches, 300, 1)
        # y: current observation (image patch), shape (batch, n_patches, 16**2, 1)
        # initialize with x_hat
        gamma = torch.tile(torch.div(1 + torch.exp(-self.B @ u), 2), (1, 1, self.n_patch, 1, 1))
        
        for _ in range(self.n_x):
            alpha = (x_k - x_hat) / self.mu
            alpha = S(alpha)

            # shrinkage algorithm
            Cyla = Cy - self.lam * alpha
            W_inv = torch.div(torch.abs(x_k), gamma * self.gamma0)
            D = self.I_x+(self.C * W_inv.transpose(-1,-2)) @ self.C.transpose(-1,-2)
            v = (self.C * W_inv.transpose(-1,-2)) @ Cyla
            x_k = W_inv * Cyla - (W_inv.transpose(-1,-2) * self.C).transpose(-1,-2) @ \
                    torch.inverse(D) @ v#conjugate_gradient(D,v)
        # print(x_k.min(), x_k.max())
        # normalize & shresholding
        # x_index = x_k / torch.norm(x_k)
        #x_index = torch.abs(x_k) < 1e-8
        #x_k[x_index] = 0
        
        # recon = self.C @ x_k
        # test = recon[0,0,:,:].detach().cpu().squeeze().numpy().reshape(16,16)
        
        return x_k
    
    def update_cause(self, x_k, u_k, beta):
        # initialize u_k
        # x = torch.mean(x_k, dim=1, keepdim=True)
        # u_k = torch.randn_like(self.B.transpose(2,3) @ x)
        # u_k = self.B.transpose(2,3) @ x_k
        
        # abs(x)
        x_abs = torch.sum(torch.abs(x_k), dim=2, keepdim=True)
        beta = self.beta
        loss_list = []
        for _ in range(self.n_u):
            W_inv = torch.div(torch.abs(u_k), beta)
            u_k = W_inv * self.B.transpose(-1,-2) @ (self.gamma0 * x_abs * torch.exp(-self.B @ u_k))
            #loss_list.append(((self.gamma0 * (torch.div(1+torch.exp(-self.B @ u_k),2) * x_abs)).sum() + beta * torch.sum(torch.abs(u_k))).item())
            # beta = max(0.95 * beta, 1e-3)
        #u_index = u_k / torch.norm(u_k)
        #u_index = torch.abs(u_index) < 1e-5
        #u_k[u_index] = 0
        #print('cause loss:', loss_list)
        return u_k, beta
    
    def forward(self, X, U, x0):
        # X: [batch, n_patches, dim, 1, T]
        # U: [batch, n_patches, dim, 1, T]
        X_pred = []
        X_U = []
        vid_recon = []
        
        T = X.shape[-1]
        
        for t in range(T):
            x = X[:, :, :, :, :, t]#.detach().clone().requires_grad_(True)
            u = U[:, :, :, :, :, t]#.detach().clone().requires_grad_(True)
            
            x_hat = self.A @ x0# + x0
            X_pred.append(x_hat)
            
            recon = self.C @ x
            recon = patch_to_image(recon, self.patch_size)
            vid_recon.append(recon)
            
            x0 = x.clone()
            
            gamma = self.gamma0 * torch.div(1 + torch.exp(-self.B @ u), 2)
            cause_state = gamma * torch.sum(torch.abs(x), dim=2, keepdim=True)
            
            # x_recon = self.B @ u
            X_U.append(cause_state)
            
        return torch.stack(vid_recon, dim=-1), torch.stack(X_pred, dim=-1), torch.stack(X_U, dim=-1)


class MM_Conv_Layer(nn.Module):
    def __init__(self, *args, **kwargs):
        super (MM_Conv_Layer, self).__init__()
        
        self.in_ch = kwargs['in_ch']
        self.x_ch = kwargs['x_ch']
        self.u_ch = kwargs['u_ch']
        self.k_size = kwargs['k_size']
        
        # assume only stride 1 for now
        
        self.i_h = kwargs['i_h']
        self.i_w = kwargs['i_w']
        
        self.n_x = kwargs['n_x']
        self.n_u = kwargs['n_u']
        self.n_a = kwargs['n_a']
        
        # self.mu = kwargs['mu']
        self.alpha = kwargs['alpha']
        self.beta = kwargs['beta']
        self.gamma0 = kwargs['gamma0']
        
        self.device = torch.device("cpu") if device is None else device
        
        # input shape is the shape after padding
        self.inp_shape = (self.in_ch, self.i_h+self.k_size//2*2, self.i_w+self.k_size//2*2)
        self.state_shape = (self.x_ch, self.i_h//2+self.k_size//2*2, 
                            self.i_w//2+self.k_size//2*2)
        # initialize dictionaries
        k = np.random.rand(self.x_ch * self.in_ch * self.k_size**2) * 0.1
        k = k.reshape((self.x_ch, self.in_ch, self.k_size, self.k_size))
        
        C = toeplitz_mult_ch(k, self.inp_shape)
        C = torch.FloatTensor(C).unsqueeze(0).transpose(-1,-2)
        self.C = nn.Parameter(C, requires_grad=False)
        self.C_conv = nn.Parameter(torch.FloatTensor(k.copy()), requires_grad=True)
        
        # initialize cause-state matrix
        k = np.random.rand(self.u_ch * self.x_ch * self.k_size**2) * 0.1
        k = k.reshape((self.u_ch, self.x_ch, self.k_size, self.k_size))

        # assuming the kernel size is odd
        #B = toeplitz_mult_ch(k, self.state_shape)
        #B = torch.FloatTensor(B).unsqueeze(0).transpose(-1,-2)
        
        #self.B = nn.Parameter(B, requires_grad=False)
        self.B_conv = nn.Parameter(torch.FloatTensor(k.copy()), requires_grad=True)
        
        self.normalize_weights()
        self.update_dict()
        I = torch.eye(C.shape[-2]).unsqueeze(0)
        self.I_x = nn.Parameter(I, requires_grad=False)
        #I = torch.eye(B.shape[-1]).unsqueeze(0)
        #self.I_u = nn.Parameter(I, requires_grad=False)
        
        self.pool = nn.MaxPool2d(2, stride=2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2, stride=2)
        
        self.reset()
        
        
        
    
    def reset(self):
        self.x, self.u = None, None
    
    def normalize_weights(self, skip_state=None, skip_cause=None):
        with torch.no_grad():
            if not skip_state:
                C_data = self.C_conv.data
                C_data = C_data / (torch.sqrt(torch.sum(C_data**2, dim=(1,2,3), keepdim=True))+1e-12)
                self.C_conv.data = C_data
            if not skip_cause:
                B_data = self.B_conv.data
                B_data = B_data / (torch.sqrt(torch.sum(B_data**2, dim=(1,2,3), keepdim=True))+1e-12)
                self.B_conv.data = B_data
            
            # self.update_dict()
    
    def update_dict(self):
        with torch.no_grad():
            k_C = self.C_conv.detach().cpu().numpy()
            #k_B = self.B_conv.detach().cpu().numpy()
            
            C = toeplitz_mult_ch(k_C, self.inp_shape)
            C = torch.FloatTensor(C).unsqueeze(0).transpose(-1,-2).to(self.device)
            self.C.data = C
            
            #B = toeplitz_mult_ch(k_B, self.state_shape)
            #B = torch.FloatTensor(B).unsqueeze(0).transpose(-1,-2).to(self.device)
            #self.B.data = B
    
    def inference(self, data, td=None):
        with torch.no_grad():
            y = data.clone()
            x, u = self.AM(y, td)
        # self.x, self.u = x, u
        return x, u
    
    def AM(self, y, td=None):
        
        # alternating minimization
        # initialize x
        # Cy = self.C.transpose(-1,-2) @ y
        Cy = F.conv2d(y, self.C_conv, padding=self.k_size//2).flatten(1).unsqueeze(-1)
        if self.x is None:
            x_k = torch.rand_like(Cy.clone())
        else:
            x_k = self.x.clone()
        # x_k = x_k.flatten(1).unsqueeze(-1)
        # initialize u
        if self.u is None:
            u_k = torch.rand(y.shape[0], self.u_ch, self.i_h // 2, self.i_w // 2)
            u_k = u_k.flatten(1).unsqueeze(-1)
            u_k = u_k / torch.norm(u_k, dim=1, keepdim=True)
            u_k = u_k.view(y.shape[0], self.u_ch, self.i_h // 2, self.i_w // 2).to(device)
            # u_k = u_k / torch.norm(u_k, dim=(1,2,3), keepdim=True)
            #u_k = torch.rand(y.shape[0], self.B.shape[-1], 1).to(self.device)
            #u_k = torch.abs(u_k)
            #u_k = u_k / torch.norm(u_k, dim=1, keepdim=True)
            
        else:
            u_k = self.u.clone()
        
        _, self.indices = self.pool(x_k.view(-1, self.x_ch, self.i_h, self.i_w))
        
        for _ in range(self.n_a):
            
            # update state
            x_k = self.update_state(x_k, u_k, Cy)
            
            # Do max pooling, then padding, then flatten state back to a vector
            x_inp, self.indices = self.pool(x_k.view(y.shape[0], self.x_ch, self.i_h, self.i_w))
            #x_inp = F.pad(x_inp, (self.k_size//2, self.k_size//2, self.k_size//2, self.k_size//2))
            #x_inp = x_inp.flatten(1).unsqueeze(-1)
            x_inp = torch.abs(x_inp)
            # u_k = self.B.transpose(-1,-2) @ x_inp
            # update cause
            u_k = self.update_cause(x_inp, u_k, td)
            
        return x_k, u_k
    
    def update_state(self, x_k, u_k, Cy):
        #gamma = -self.B @ u_k
        # need to reshape u_k into 2D tensor then unpool
        #gamma = gamma.view(-1, self.u_ch, self.state_shape[-2], self.state_shape[-1])
        # we manually pad image, state etc.
        # remove the padded region
        #gamma = gamma[:, :, self.k_size//2:-(self.k_size//2), self.k_size//2:-(self.k_size//2)]
        
        #gamma = F.conv_transpose2d(u_k, self.B_conv, padding=self.k_size//2)
        # unpool to work with states
        #gamma = self.unpool(gamma, self.indices)
        #gamma = torch.div(1 + torch.exp(gamma), 2)
        #gamma = gamma.flatten(1).unsqueeze(-1)
        for _ in range(self.n_x):
            W_inv = torch.div(torch.abs(x_k), self.alpha)# * gamma)
            #W_inv = torch.div(torch.abs(x_k), self.gamma0)
            D = self.I_x + (self.C * W_inv.transpose(-1,-2)) @ self.C.transpose(-1,-2)
            v = (self.C * W_inv.transpose(-1,-2)) @ Cy
            x_k = W_inv * Cy - W_inv * F.conv2d(
                conjugate_gradient(D,v).view(-1, 
                                             self.in_ch, 
                                             self.i_h + 2 * (self.k_size//2), 
                                             self.i_w + 2 * (self.k_size//2)
                                             ), 
                                            self.C_conv).flatten(1).unsqueeze(-1)
            #x_k = W_inv * Cy - (W_inv.transpose(-1,-2) * self.C).transpose(-1,-2) @ \
            #        conjugate_gradient(D, v)#torch.inverse(D) @ v
        
        #x_index = x_k / torch.norm(x_k)
        #x_index = torch.abs(x_index) < 1e-5
        #x_k[x_index] = 0
        
        return x_k
    
    def update_cause(self, x_k, u_k, td):
        # assume x_k is already padded and taken the absolute value
        #Cy = self.B.transpose(-1,-2) @ x_k
        #u_k = Cy.clone()
        beta = self.beta
        prev_u = u_k.clone()
        for _ in range(self.n_u):
            
            #D = self.I_u + (self.B * W_inv.transpose(-1,-2)) @ self.B.transpose(-1,-2)
            #v = (self.B * W_inv.transpose(-1,-2)) @ Cy
            #u_k = W_inv * Cy - (W_inv.transpose(-1,-2) * self.B).transpose(-1,-2) @ conjugate_gradient(D, v)
            exp = torch.exp(-F.conv_transpose2d(u_k, self.B_conv, padding=self.k_size//2))
            exp = self.gamma0 * x_k * exp
            if td is None:
                W_inv = torch.div(torch.abs(u_k), beta)
                u_k = W_inv * F.conv2d(exp, self.B_conv, padding=self.k_size//2)
                #u_k = W_inv * (self.B.transpose(-1,-2) @ (self.gamma0 * x_k * torch.exp(-self.B @ u_k)))
            else:
                W_inv = torch.div(torch.abs(u_k), beta+torch.abs(u_k))
                u_k = W_inv * (F.conv2d(exp, self.B_conv, padding=self.k_size//2) + 0.01 * td)
                #u_k = W_inv * (self.B.transpose(-1,-2) @ (self.gamma0 * x_k * torch.exp(-self.B @ u_k))+0.01 * td)
                
            u_k = 0.5 * u_k + 0.5 * prev_u
            prev_u = u_k.clone()
            beta = max(0.985*beta,1e-3)
        #u_index = u_k / torch.norm(u_k)
        #u_index = torch.abs(u_index) < 1e-5
        #u_k[u_index] = 0
        
        return u_k
    
    def forward(self, x_k, u_k):
        #recon = self.C @ x_k
        recon = F.conv_transpose2d(x_k, self.C_conv, padding=self.k_size//2)
        #recon_x = self.B @ u_k
        gamma = torch.exp(-F.conv_transpose2d(u_k, self.B_conv, padding=self.k_size//2))
        # gamma = torch.exp(-self.B @ u_k).view(-1, self.u_ch, self.state_shape[-2], self.state_shape[-1])
        # gamma = gamma[:, :, self.k_size//2:-(self.k_size//2), self.k_size//2:-(self.k_size//2)]
        #gamma = self.unpool(gamma, self.indices)
        # gamma = gamma.flatten(1).unsqueeze(-1)
        cause_state = gamma * torch.abs(self.pool(x_k)[0])
        return recon, cause_state#recon_x
        
class FISTA_Layer(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(FISTA_Layer, self).__init__()
        # weight for the state-cause correlation term
        self.gamma0 = kwargs['gamma0']#nn.Parameter(torch.tensor(1), requires_grad=False)
        # parameter for the smoothed transition error
        self.mu = kwargs['mu']#0.01/150#nn.Parameter(torch.tensor(0.01/150), requires_grad=False)
        # weight for the transition error term
        self.lam = kwargs['lam']#0.5 # 0.5
        # weight for the sparsity of causes
        self.beta = kwargs['beta']#0.5 # 0.5
        
        self.n_ch = kwargs['n_ch']
        
        self.n_steps = kwargs['n_steps']
        
        # X_dim = 300, U_dim = 40
        self.X_dim = kwargs['X_dim']
        self.U_dim = kwargs['U_dim']
        
        self.patch_size = kwargs['patch_size']
        self.input_size = kwargs['input_size']
        self.isTopLayer = kwargs['isTopLayer']
        
        if "n_dict" in kwargs:
            self.n_dict = kwargs['n_dict']
        else:
            self.n_dict = 1
        
        if "use_A" in kwargs:
            self.use_A = kwargs['use_A']
        else:
            self.use_A = True
        
        input_size = self.patch_size ** 2 if self.isTopLayer else kwargs['input_size']
        
        C = torch.rand(1, self.n_ch, self.n_dict, input_size, self.X_dim)
        C = C / C.norm(dim=-2, keepdim=True)
        
        B = torch.rand(1, self.n_ch, self.n_dict, self.X_dim, self.U_dim)
        B = B / B.norm(dim=-2, keepdim=True)
        
        A = torch.rand(1, self.n_ch, self.n_dict, self.X_dim, self.X_dim)
        A = A / A.norm(dim=-2, keepdim=True)
        
        self.A = nn.Parameter(A)
        self.B = nn.Parameter(B)
        self.C = nn.Parameter(C)
        
        # self.patch_size = 16
        
        self.device = torch.device("cpu") if device is None else device
        
        
    def inference(self, data):
        X = []
        U = []
        
        with torch.no_grad():
            for t in range(data.shape[-1]):
                if (t+1) % 100 == 0:
                    print('\t', t+1, 'frame')
                    
                # loop over frames in the data
                y_t = data[:, :, :, :, t]
                
                # separate patches from an entire image
                if self.isTopLayer:
                    patches_t = make_patches(y_t, self.patch_size)
                if t == 0 or not self.use_A:
                    x_prev = torch.zeros_like(self.C.transpose(-1,-2) @ patches_t)
                x_t, u_t = self.AM_FISTA(patches_t, x_prev)
                X.append(x_t)
                U.append(u_t)
                if self.use_A:
                    x_prev = x_t.clone()
                
        X = torch.stack(X, dim=-1)
        U = torch.stack(U, dim=-1)
        
        return X, U
    
    def AM_FISTA(self, patches, x_t_minus_1):
        # alternating minimization algorithm
        # state prediction
        x_hat = self.A @ x_t_minus_1
        # initialize x_t with A x_{t-1}
        x_t = x_hat.clone()
        u_t = torch.zeros_like(self.B.transpose(-1,-2) @ torch.mean(x_t, dim=2, keepdim=True))
        
        Cy = self.C.transpose(-1,-2) @ patches
        CTC = self.C.transpose(-1,-2) @ self.C
        gamma = self.gamma0 * torch.div(1 + torch.exp(-self.B @ u_t), 2)
        '''
        states = [{
                    'L': torch.ones((x_t.shape[0], 1, 1, 1, 1), device=self.device),
                    'eta': 2,
                    't_k': 1.,
                    't_k_1': 1.,
                    'x_k': x_t[:,:,i:i+1,:,:].clone(),
                    'x_k_1': x_t[:,:,i:i+1,:,:].clone(), 
                    'x_hat': x_hat[:,:,i:i+1,:,:],
                    'z_k': x_t[:,:,i:i+1,:,:].clone(),
                    'Cy': Cy[:,:,i:i+1,:,:].clone(),
                    'CTC': CTC.clone(), 
                    'y': patches[:,:,i:i+1,:,:].clone(),
                    'gamma': gamma
                  } for i in range(x_t.shape[2])]
        '''
        
        state = {
                 'L': torch.ones((x_t.shape[0], 1, 1, 1, 1), device=self.device),
                 'eta': 2,
                 't_k': 1.,
                 't_k_1': 1.,
                 'x_k': x_t.clone(),
                 'x_k_1': x_t.clone(), 
                 'x_hat': x_hat.clone(),
                 'z_k': x_t.clone(),
                 'Cy': self.C.transpose(-1,-2) @ patches.clone(), 
                 'CTC': self.C.transpose(-1,-2) @ self.C, 
                 'y': patches.clone(),
                 'gamma': gamma.clone()
                 }
        
        
        cause = {
                 'L': torch.ones((u_t.shape[0], 1, 1, 1, 1), device=self.device),
                 'u_k': u_t.clone(),
                 'u_k_1': u_t.clone(), 
                 'z_k': u_t.clone(),
                 'eta': 1.5,
                 't_k': 1., 
                 't_k_1': 1.,
                 'x': None,
                 'beta': self.beta
                 }
        #stop = False
        #step = 0
        #x_sparsity = torch.abs(state['x_k'].clone()) > 0
        #while not stop and step < self.max_iter:
        for i in range(self.n_steps):
            #step += 1
            #x_sparsity_prev = x_sparsity.clone()
            state = self.update_state_FISTA(state)
            #for j in range(x_t.shape[2]):
            #    states[j] = self.update_state_FISTA(states[j])
            # state = torch.concatenate([states[j]['x_k'] for j in range(x_t.shape[2])], dim=2)
            cause['x'] = torch.sum(torch.abs(state['x_k'].clone()), dim=2, keepdim=True)
            #cause['x'] = torch.sum(torch.abs(state.clone()), dim=2, keepdim=True)
            cause = self.update_cause_FISTA(cause)
            gamma = self.gamma0 * torch.div(1 + torch.exp(-self.B @ cause['u_k'].clone()), 2)
            #for j in range(x_t.shape[2]):
            #    states[j]['gamma'] = gamma
            state['gamma'] = gamma
            
            #x_sparsity = torch.abs(state['x_k'].clone()) > 0
            
            #sparsity_change = torch.sum(x_sparsity != x_sparsity_prev)
            #change_ratio = sparsity_change / torch.sum(x_sparsity)
            
            #stop = change_ratio < 0.03
        
        return state['x_k'], cause['u_k']
    
    def update_state_FISTA(self, state):
        # unpack state
        L = state['L'].clone()
        eta = state['eta']
        t_k, t_k_1 = state['t_k'], state['t_k_1']
        x_k, x_k_1, x_hat = state['x_k'].clone(), state['x_k_1'].clone(), state['x_hat'].clone()
        z_k = state['z_k'].clone()
        Cy, CTC = state['Cy'].clone(), state['CTC'].clone()
        y = state['y'].clone()
        gamma = state['gamma'].clone()
        
        alpha = (z_k - x_hat) / self.mu
        alpha = S(alpha)
        
        # gradient of 1/2 * ||y_t - C x_t||_2^2 + lambda * ||x_t - Ax_t_1||_1 w.r.t. x_t
        gradient_zk = CTC @ z_k - Cy + self.lam * alpha
        
        # line search
        stop_line_search = torch.zeros((z_k.shape[0]), device=self.device)
        const = 1 / 2 * torch.square(y - self.C @ z_k).flatten(1).sum(1) + \
                self.lam * torch.abs(z_k - x_hat).flatten(1).sum(1)
        step = 0
        keep_going = True
        while keep_going:# and step <= 100:
            step += 1
            x_k = soft_thresholding(z_k - torch.div(gradient_zk, L), 
                                    torch.div(gamma, L))
            temp1 = 1 / 2 * torch.square(y - self.C @ x_k).flatten(1).sum(1) + \
                    self.lam * torch.abs(x_k - x_hat).flatten(1).sum(1)
            temp2 = const + ((x_k - z_k) * gradient_zk).flatten(1).sum(1) +\
                    L.squeeze() / 2 * torch.square(x_k - z_k).flatten(1).sum(1)
            if temp1.sum() <= temp2.sum(): keep_going = False
            stop_line_search = temp1 <= temp2
            L = torch.where(stop_line_search[:,None,None,None,None], L, eta * L)
            
        # acceleration step
        t_k_1 = (1 + (1 + 4 * t_k**2)**0.5) / 2
        z_k = x_k + (t_k - 1) / t_k_1 * (x_k - x_k_1)
        x_k_1 = x_k.clone()
        t_k = t_k_1
        
        state['z_k'], state['x_k'], state['x_k_1'], state['t_k'], state['t_k_1'], state['L'] = \
        z_k, x_k, x_k_1, t_k, t_k_1, L
        
        return state
        
    def update_cause_FISTA(self, cause):
        # unpack cause
        x = cause['x'].clone()
        z_k = cause['z_k'].clone()
        u_k, u_k_1 = cause['u_k'].clone(), cause['u_k_1'].clone()
        t_k, t_k_1 = cause['t_k'], cause['t_k_1']
        L = cause['L'].clone()
        eta = cause['eta']
        beta = cause['beta']
        batch_size = z_k.shape[0]
        
        # gradient of 1/2 * ||y_t - C x_t||_2^2 + lambda * ||x_t - Ax_t_1||_1 w.r.t. u_t
        exp_func = self.gamma0 * torch.exp(-self.B @ z_k) / 2
        gradient_zk = -self.B.transpose(-1,-2) @ (exp_func * x)
        # print("shape of gradient: ", gradient_xk.shape)
        
        stop_line_search = torch.zeros((batch_size), device=self.device)
        
        # line search
        const = (self.gamma0 * (x * torch.exp(-self.B @ z_k)) / 2).flatten(1).sum(1)
        step = 0
        keep_going = True
        while keep_going:# and step <= 100:
            step += 1
            # FISTA
            # update u_k
            u_k = soft_thresholding(z_k - torch.div(gradient_zk, L), beta / L)
            
            temp1 = (self.gamma0 * (x * torch.exp(-self.B @ u_k)) / 2).flatten(1).sum(1)
            temp2 = const + ((u_k - z_k) * gradient_zk).flatten(1).sum(1) +\
                    L.squeeze() / 2 * torch.square(u_k - z_k).flatten(1).sum(1)
            stop_line_search = temp1 <= temp2
            
            L = torch.where(stop_line_search[:,None,None,None,None], L, eta * L)
            if temp1.sum() <= temp2.sum(): keep_going = False
            
        
        beta = max(0.99 * beta, 1e-3)
        # acceleration step
        t_k_1 = (1 + (1 + 4 * t_k**2)**0.5) / 2
        z_k = u_k + (t_k - 1) / t_k_1 * (u_k - u_k_1)
        u_k_1 = u_k.clone()
        t_k = t_k_1
        
        cause['z_k'], cause['u_k'], cause['u_k_1'], cause['t_k'], cause['t_k_1'], cause['L'], cause['beta'] = \
            z_k, u_k, u_k_1, t_k, t_k_1, L, beta
        
        return cause
    
    def forward(self, X, U, x0):
        # X: [batch, n_patches, dim, 1, T]
        # U: [batch, n_patches, dim, 1, T]
        X_pred = []
        X_U = []
        vid_recon = []
        
        T = X.shape[-1]
        
        for t in range(T):
            x = X[:, :, :, :, :, t]#.detach().clone().requires_grad_(True)
            u = U[:, :, :, :, :, t]#.detach().clone().requires_grad_(True)
            
            x_hat = self.A @ x0
            X_pred.append(x_hat)
            
            recon = self.C @ x
            recon = patch_to_image(recon, self.patch_size)
            vid_recon.append(recon)
            
            x0 = x.clone()
            
            gamma = self.gamma0 * torch.div(1 + torch.exp(-self.B @ u), 2)
            cause_state = gamma * torch.sum(torch.abs(x), dim=2, keepdim=True)
            
            # x_recon = self.B @ u
            X_U.append(cause_state)
            
        return torch.stack(vid_recon, dim=-1), torch.stack(X_pred, dim=-1), torch.stack(X_U, dim=-1)
    
    def normalize_weights(self):
        with torch.no_grad():
            #self.A.data = self.A.data / self.A.data.norm(dim=-2, keepdim=True)
            self.B.data = self.B.data / self.B.data.norm(dim=-2, keepdim=True)
            self.C.data = self.C.data / self.C.data.norm(dim=-2, keepdim=True)