import torch
import torch.nn as nn
import torch.nn.functional as F


class AE(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
         
        # Building an linear encoder with Linear
        # layer followed by Relu activation function
        # 784 ==> 9
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 36),
            torch.nn.ReLU(),
            torch.nn.Linear(36, 18),
            torch.nn.ReLU(),
            torch.nn.Linear(18, 9)
        )
         
        # Building an linear decoder with Linear
        # layer followed by Relu activation function
        # The Sigmoid activation function
        # outputs the value between 0 and 1
        # 9 ==> 784
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(9, 18),
            torch.nn.ReLU(),
            torch.nn.Linear(18, 36),
            torch.nn.ReLU(),
            torch.nn.Linear(36, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, input_size),
            torch.nn.Sigmoid()
        )
 
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    
class WTA_RNN_AE(nn.Module):
    def __init__(self, input_size, k):
        super(WTA_RNN_AE, self).__init__()
        self.hidden_size = 64
        self.k = k
        
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
        )
        
        self.RNN = nn.RNN(64, 64, num_layers=5, batch_first=True)
        
        self.decoder = torch.nn.Sequential(torch.nn.Linear(64, input_size))
        self.predictor = torch.nn.Sequential(torch.nn.Linear(64, input_size))
    
    def WTA(self, code):
        # keep the top k% of the activations
        # make it differentiable
        k = self.k
        topk, indices = torch.topk(code, k, dim=1)
        code = torch.zeros_like(code).scatter_(1, indices, topk)
        
        return code
    
    def forward(self, x, h):
        code = self.encoder(x)
        pred, h = self.RNN(code, h)
        recon = self.decoder(self.WTA(code))
        pred = self.predictor(self.WTA(pred))
        
        return recon, pred, h