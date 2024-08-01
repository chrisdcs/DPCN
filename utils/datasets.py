import torch
from torch.utils.data import Dataset

import scipy.io as scio
import numpy as np

import kornia as K

class mario_loader(Dataset):
    
    def __init__(self, file_path):
        self.data = np.load(file_path)
        self.N = self.data.shape[0]
        self.frame_size = self.data.shape[-2]
        # self.zca = K.enhance.ZCAWhitening(eps=0.1, compute_inv=True)
    
    def __getitem__(self, index):
        video = torch.from_numpy(self.data[index])
        video = video.permute(3,1,2,0)
        # video = self.zca.fit(video)
        return video#.permute(0,3,1,2)
    
    def __len__(self):
        return self.N

class video_loader(Dataset):
    # load video data: shape (N, C, H, W, T)
    def __init__(self, file_path):
        self.video = scio.loadmat(file_path)['M']
        # N: number of video sequences
        # T: number of frames for each video sequence
        self.N, self.T = self.video.shape
        
        # use only half of the frames to speed up training
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


class image_loader(Dataset):
    # load image data: shape (N, C, H, W)
    def __init__(self, file_path):
        videos = np.load(file_path)
        # N: number of video sequences
        # T: number of frames for each video sequence
        self.N, self.T, h, w, c = videos.shape
        
        self.images = []
        
        for i in range(self.N):
            video = videos[i]
            for j in range(self.T):
                self.images.append(video[j])

        self.N = len(self.images)
        
    def __getitem__(self, index):
        image = self.images[index]
        return torch.FloatTensor(image).permute(2,0,1)
    
    def __len__(self):
        return self.N
    
class conv_loader(Dataset):
    def __init__(self, file_path):
        zca_images = torch.load(file_path).cpu().numpy()
        self.zca_images = zca_images
        
    
    def __getitem__(self, index):
        image = self.zca_images[index]
        # normalize the image
        #image = (image - image.min()) / (image.max() - image.min())
        return torch.FloatTensor(image)#.permute(2,0,1)
    
    def __len__(self):
        return self.zca_images.shape[0]