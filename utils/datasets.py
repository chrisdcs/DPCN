import torch
from torch.utils.data import Dataset

import scipy.io as scio

class video_loader(Dataset):
    # load video data: shape (N, C, H, W, T)
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
    
    
def make_patches(img, patch_size):
    # outupt shape: (batch, n_patches, patch_size^2, 1)
    try:
        assert img.shape[2] % patch_size == 0
        patches = img.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches = patches.contiguous().view(patches.size(0), -1, patch_size, patch_size).flatten(-2).unsqueeze(-1)
        return patches
    except:
        raise ValueError('Image size must be divisible by patch size')