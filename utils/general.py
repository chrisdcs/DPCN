import torch
import yaml
import numpy as np
from scipy import linalg
from concurrent.futures import ThreadPoolExecutor

def yaml_load(file="data.yaml"):
    """Safely loads and returns the contents of a YAML file specified by `file` argument."""
    with open(file, errors="ignore") as f:
        return yaml.safe_load(f)
    
def conjugate_gradient(A, b, x=None, tol=1e-5, max_iterations=200):
    if x is None:
        x = torch.zeros_like(b)
    
    r = b - torch.matmul(A, x)
    p = r.clone()
    rs_old = r.transpose(-1,-2) @ r#torch.dot(r, r)
    
    for i in range(max_iterations):
        Ap = torch.matmul(A, p)
        alpha = rs_old / (p.transpose(-1,-2) @ Ap+1e-10)#torch.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        rs_new = r.transpose(-1,-2) @ r
        
        if (torch.sqrt(rs_new) < tol).all():
            break
        
        p = r + (rs_new / (rs_old+1e-10)) * p
        rs_old = rs_new
    # print(i, rs_new)
    return x

def toeplitz_1_ch(kernel, input_size):
    # shapes
    k_h, k_w = kernel.shape
    i_h, i_w = input_size
    o_h, o_w = i_h-k_h+1, i_w-k_w+1

    # construct 1d conv toeplitz matrices for each row of the kernel
    toeplitz = []
    for r in range(k_h):
        toeplitz.append(linalg.toeplitz(c=(kernel[r,0], *np.zeros(i_w-k_w)), r=(*kernel[r], *np.zeros(i_w-k_w))))

    # construct toeplitz matrix of toeplitz matrices (just for padding=0)
    h_blocks, w_blocks = o_h, i_h
    h_block, w_block = toeplitz[0].shape

    W_conv = np.zeros((h_blocks, h_block, w_blocks, w_block))

    for i, B in enumerate(toeplitz):
        for j in range(o_h):
            W_conv[j, :, i+j, :] = B

    W_conv.shape = (h_blocks*h_block, w_blocks*w_block)

    return W_conv

def toeplitz_mult_ch(kernel, input_size):
    """Compute toeplitz matrix for 2d conv with multiple in and out channels.
    Args:
        kernel: shape=(n_out, n_in, H_k, W_k)
        input_size: (n_in, H_i, W_i)"""

    kernel_size = kernel.shape
    output_size = (kernel_size[0], input_size[1] - (kernel_size[2] - 1), input_size[2] - (kernel_size[3] - 1))
    T = np.zeros((output_size[0], int(np.prod(output_size[1:])), input_size[0], int(np.prod(input_size[1:]))))

    def process_channel(args):
        i, ks, input_size = args
        for j, k in enumerate(ks):  # loop over input channel
            T_k = toeplitz_1_ch(k, input_size[1:])
            T[i, :, j, :] = T_k

    with ThreadPoolExecutor() as executor:
        executor.map(process_channel, [(i, ks, input_size) for i, ks in enumerate(kernel)])

    T.shape = (np.prod(output_size), np.prod(input_size))
    
    return T


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


def make_patches(img, patch_size):
    # outupt shape: (batch, n_patches, patch_size^2, 1)
    try:
        assert img.shape[2] % patch_size == 0
        patches = img.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        n_ch = patches.shape[1]
        patches = patches.contiguous().view(patches.size(0), n_ch, -1, patch_size, patch_size).flatten(-2).unsqueeze(-1)
        return patches
    except:
        raise ValueError('Image size must be divisible by patch size')