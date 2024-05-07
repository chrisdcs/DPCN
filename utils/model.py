import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.datasets import make_patches

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def S(x):
    x = torch.where(x > 1, torch.ones_like(x), x)
    x = torch.where(x < -1, -torch.ones_like(x), x)
    return x

def soft_thresholding(x, gamma):
    return torch.sign(x) * torch.max(torch.abs(x) - gamma, torch.zeros_like(x))

def patch_to_image(patches, patch_size):
    # patches: [batch, n_patches, 16**2, 1]
    # patch_size: 16
    batch_size, n_channel, n_patch, patch_dim, _ = patches.shape
    patch_size = int(patch_dim ** 0.5)
    grid_size = int(n_patch ** 0.5)
    image = patches.view(batch_size, n_channel, grid_size, grid_size, patch_size, patch_size)
    image = image.permute(0, 1, 2, 4, 3, 5).contiguous()
    image = image.view(batch_size, n_channel, grid_size*patch_size, grid_size*patch_size)
    return image

class Shrinkage_Layer(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Shrinkage_Layer, self).__init__()
        
        self.n_ch = kwargs['n_ch']
        
        C = torch.rand(1, self.n_ch, 1, 16**2, 300)#.unsqueeze(0).unsqueeze(0)
        C = C / C.norm(dim=-2, keepdim=True)
        
        B = torch.rand(1, self.n_ch, 1, 300, 40)#.unsqueeze(0).unsqueeze(0)
        B = B / B.norm(dim=-2, keepdim=True)
        
        A = torch.rand(1, self.n_ch, 1, 300, 300)#.unsqueeze(0).unsqueeze(0)
        A = A / A.norm(dim=-2, keepdim=True)
        
        self.A = nn.Parameter(A)
        self.B = nn.Parameter(B)
        self.C = nn.Parameter(C)
       
        I = torch.eye(C.shape[-2]).unsqueeze(0).unsqueeze(0)
        self.I_x = nn.Parameter(I, requires_grad=False)
        I = torch.eye(B.shape[-2]).unsqueeze(0).unsqueeze(0)
        self.I_u = nn.Parameter(I, requires_grad=False)
        
        self.patch_size = 16
        self.n_patch = 4

        self.n_u = kwargs['n_u']

        
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

    def inference(self, video):
        # raise NotImplementedError
        batch_size, c, h, w, T = video.shape
        
        X = []
        U = []
        with torch.no_grad():
            for t in range(T):
                if (t+1) % 100 == 0:
                    print('\t', t+1, 'frame')
                    
                # loop over frames in the video
                y_t = video[:, :, :, :, t]
                
                # separate patches from an entire image
                patches_t = make_patches(y_t, self.patch_size)
                # patches_t: (batch_size, 4, 256, 1) if gray scale images
                # patches_t: (batch_size, 3, 4, 256, 1) if RGB images
                if t == 0:
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
        
        
        for _ in range(1):
            # emperically showed that shrinkage algorithm converges/produces good solution in 5 iterations
            x_t = self.update_state(x_t, x_hat, u_t, Cy)
            u_t = self.update_cause(x_t, u_t)
            #u_t = self.update_cause(x_t)
            #cause = self.update_cause_FISTA(x_t, cause)
            
        return x_t, u_t
    
    def update_state(self, x_k, x_hat, u, Cy):
        # x: previous state, shape (batch, n_patches, 300, 1)
        # y: current observation (image patch), shape (batch, n_patches, 16**2, 1)
        # initialize with x_hat
        
        gamma = torch.tile(torch.div(1 + torch.exp(-self.B @ u), 2), (1, 1, self.n_patch, 1, 1))
        
        for _ in range(1):
            alpha = (x_k - x_hat) / self.mu
            alpha = S(alpha)

            # shrinkage algorithm
            Cyla = Cy - self.lam * alpha
            W_inv = torch.div(torch.abs(x_k), gamma * self.gamma0)
            x_k = W_inv * Cyla - (W_inv.transpose(-1,-2) * self.C).transpose(-1,-2) @ \
                    torch.inverse(self.I_x+(self.C * W_inv.transpose(-1,-2)) @ self.C.transpose(-1,-2)) @ \
                            ((self.C * W_inv.transpose(-1,-2)) @ Cyla)
        # print(x_k.min(), x_k.max())
        # normalize & shresholding
        x_index = x_k / torch.norm(x_k)
        x_index = torch.abs(x_index) < 1e-3
        x_k[x_index] = 0
        
        # recon = self.C @ x_k
        # test = recon[0,0,:,:].detach().cpu().squeeze().numpy().reshape(16,16)
        
        return x_k
    
    def update_cause(self, x_k, u_k):
        # initialize u_k
        # x = torch.mean(x_k, dim=1, keepdim=True)
        # u_k = torch.randn_like(self.B.transpose(2,3) @ x)
        # u_k = self.B.transpose(2,3) @ x_k
        
        # abs(x)
        x_abs = torch.sum(torch.abs(x_k), dim=2, keepdim=True)
        
        for _ in range(self.n_u):
            W_inv = torch.div(torch.abs(u_k), self.beta)
            u_k = W_inv * self.B.transpose(-1,-2) @ (self.gamma0 * x_abs * torch.exp(-self.B @ u_k))
                        
        u_index = u_k / torch.norm(u_k)
        u_index = torch.abs(u_index) < 1e-3
        u_k[u_index] = 0
        
        return u_k
    
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

class Layer_FISTA(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(Layer_FISTA, self).__init__()
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
        
        input_size = self.patch_size ** 2 if self.isTopLayer else kwargs['input_size']
        
        C = torch.rand(1, self.n_ch, 1, input_size, self.X_dim)
        C = C / C.norm(dim=-2, keepdim=True)
        
        B = torch.rand(1, self.n_ch, 1, self.X_dim, self.U_dim)
        B = B / B.norm(dim=-2, keepdim=True)
        
        A = torch.rand(1, self.n_ch, 1, self.X_dim, self.X_dim)
        A = A / A.norm(dim=-2, keepdim=True)
        
        self.A = nn.Parameter(A)
        self.B = nn.Parameter(B)
        self.C = nn.Parameter(C)
        
        # self.patch_size = 16
        
        self.device = torch.device("cpu") if device is None else device
        
        
    def inference(self, video):
        X = []
        U = []
        
        with torch.no_grad():
            for t in range(video.shape[-1]):
                if (t+1) % 100 == 0:
                    print('\t', t+1, 'frame')
                    
                # loop over frames in the video
                y_t = video[:, :, :, :, t]
                
                # separate patches from an entire image
                if self.isTopLayer:
                    patches_t = make_patches(y_t, self.patch_size)
                if t == 0:
                    x_prev = torch.zeros_like(self.C.transpose(-1,-2) @ patches_t)
                x_t, u_t = self.AM_FISTA(patches_t, x_prev)
                X.append(x_t)
                U.append(u_t)
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
                 'x_hat': x_hat,
                 'z_k': x_t.clone(),
                 'Cy': self.C.transpose(-1,-2) @ patches, 
                 'CTC': self.C.transpose(-1,-2) @ self.C, 
                 'y': patches,
                 'gamma': self.gamma0 * (1 + torch.exp(-self.B @ u_t)) / 2
                 }'''
        
        
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
            #state = self.update_state_FISTA(state)
            for j in range(x_t.shape[2]):
                states[j] = self.update_state_FISTA(states[j])
            state = torch.concatenate([states[j]['x_k'] for j in range(x_t.shape[2])], dim=2)
            #cause['x'] = torch.sum(torch.abs(state['x_k'].clone()), dim=2, keepdim=True)
            cause['x'] = torch.sum(torch.abs(state.clone()), dim=2, keepdim=True)
            cause = self.update_cause_FISTA(cause)
            gamma = self.gamma0 * torch.div(1 + torch.exp(-self.B @ cause['u_k'].clone()), 2)
            # states[j]['gamma'] = gamma
            for j in range(x_t.shape[2]):
                states[j]['gamma'] = gamma
            # states['gamma'] = self.gamma0 * (1 + torch.exp(-self.B @ cause['u_k'].clone())) / 2
            
            #x_sparsity = torch.abs(state['x_k'].clone()) > 0
            
            #sparsity_change = torch.sum(x_sparsity != x_sparsity_prev)
            #change_ratio = sparsity_change / torch.sum(x_sparsity)
            
            #stop = change_ratio < 0.03
        
        return state, cause['u_k']
    
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
        while torch.sum(stop_line_search) < x_k.shape[0] and step <= 100:
            step += 1
            x_k = soft_thresholding(z_k - torch.div(gradient_zk, L), 
                                    torch.div(gamma, L))
            temp1 = 1 / 2 * torch.square(y - self.C @ x_k).flatten(1).sum(1) + \
                    self.lam * torch.abs(x_k - x_hat).flatten(1).sum(1)
            temp2 = const + ((x_k - z_k) * gradient_zk).flatten(1).sum(1) +\
                    L.squeeze() / 2 * torch.square(x_k - z_k).flatten(1).sum(1)
                    
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
        exp_func = torch.exp(-self.B @ z_k) / 2
        gradient_zk = -self.B.transpose(-1,-2) @ (exp_func * x)
        # print("shape of gradient: ", gradient_xk.shape)
        
        stop_line_search = torch.zeros((batch_size), device=self.device)
        
        # line search
        const = (self.gamma0 * (x * torch.exp(-self.B @ z_k)) / 2).flatten(1).sum(1)
        step = 0
        while torch.sum(stop_line_search) < batch_size and step <= 100:
            step += 1
            # FISTA
            # update u_k
            u_k = soft_thresholding(z_k - torch.div(gradient_zk, L), 
                                    beta / L)
            
            temp1 = (self.gamma0 * (x * torch.exp(-self.B @ u_k)) / 2).flatten(1).sum(1)
            temp2 = const + ((u_k - z_k) * gradient_zk).flatten(1).sum(1) +\
                    L.squeeze() / 2 * torch.square(u_k - z_k).flatten(1).sum(1)
            stop_line_search = temp1 <= temp2
            L = torch.where(stop_line_search[:,None,None,None,None], L, eta * L)
        
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
            self.A.data = self.A.data / self.A.data.norm(dim=-2, keepdim=True)
            self.B.data = self.B.data / self.B.data.norm(dim=-2, keepdim=True)
            self.C.data = self.C.data / self.C.data.norm(dim=-2, keepdim=True)