import torch
import time
import PIL.Image as Image
import numpy as np
from scipy import linalg
def conjugate_gradient(A, b, tol=1e-5, max_iter=1000):
    x = torch.zeros_like(b)
    r = b - torch.matmul(A, x)
    p = r.clone()
    rs_old = torch.matmul(r.t(), r)

    for _ in range(max_iter):
        Ap = torch.matmul(A, p)
        alpha = rs_old / (torch.matmul(p.t(), Ap) + 1e-10)
        x += alpha * p
        r -= alpha * Ap
        rs_new = torch.matmul(r.t(), r)
        if torch.sqrt(rs_new) < tol:
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new

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
    output_size = (kernel_size[0], input_size[1] - (kernel_size[2]-1), input_size[2] - (kernel_size[3]-1))
    T = np.zeros((output_size[0], int(np.prod(output_size[1:])), input_size[0], int(np.prod(input_size[1:]))))

    for i,ks in enumerate(kernel):  # loop over output channel
        for j,k in enumerate(ks):  # loop over input channel
            T_k = toeplitz_1_ch(k, input_size[1:])
            T[i, :, j, :] = T_k

    T.shape = (np.prod(output_size), np.prod(input_size))

    return T
# Assuming C is a dense matrix with 95% sparsity
# initialize kernel
out_ch = 32
k = np.random.randn(out_ch*3*9*9).reshape((out_ch,3*9*9))

# Normalize kernel
k_ = k / np.linalg.norm(k, axis=1, keepdims=True)#.reshape((128,3,9,9))
k_ = k_.reshape((out_ch,3,9,9))

# Normalize the toeplitz matrix
k = k.reshape((out_ch,3,9,9))
C = toeplitz_mult_ch(k, (3,40,40))
C = C / np.linalg.norm(C, axis=1, keepdims=True)
C = C.T
C_dense = torch.FloatTensor(C)
# C_dense[C_dense < 2] = 0  # Making the matrix sparse for the example

# Example initializations
I_x = torch.eye(4800)
Cy = torch.randn(32768, 1)
x_k = torch.randn(32768, 1)
mu = 1.
# Dense computation
start_time = time.time()
W_inv = torch.div(torch.abs(x_k), mu)
D = I_x+(C_dense * W_inv.transpose(-1,-2)) @ C_dense.transpose(-1,-2)
v = (C_dense * W_inv.transpose(-1,-2)) @ Cy
# print(D.shape, v.shape)
x_k_dense  = W_inv * Cy - (W_inv.transpose(-1,-2) * C_dense).transpose(-1,-2) @ conjugate_gradient(D, v)
dense_time = time.time() - start_time

# Convert C to a sparse matrix
indices = C_dense.nonzero().t()
values = C_dense[indices[0], indices[1]]
C_sparse = torch.sparse_coo_tensor(indices, values, C_dense.size())

# Sparse computation
start_time = time.time()
W_inv_sparse = torch.div(torch.abs(x_k), mu)
D_sparse = I_x + torch.sparse.mm(C_sparse * W_inv_sparse.transpose(-1, -2), C_sparse.transpose(-1, -2).to_dense())
v_sparse = torch.sparse.mm(C_sparse * W_inv_sparse.transpose(-1, -2), Cy)
x_k_sparse = W_inv_sparse * Cy - torch.sparse.mm((W_inv_sparse.transpose(-1, -2)*C_sparse).transpose(-1,-2), conjugate_gradient(D_sparse, v_sparse))
sparse_time = time.time() - start_time

# Comparing memory usage
dense_memory = C_dense.element_size() * C_dense.nelement()
sparse_memory = indices.element_size() * indices.nelement() + values.element_size() * values.nelement()

print(f"Dense matrix computation time: {dense_time:.6f} seconds")
print(f"Sparse matrix computation time: {sparse_time:.6f} seconds")
print(f"Memory usage of dense matrix: {dense_memory / 1024**2:.2f} MB")
print(f"Memory usage of sparse matrix: {sparse_memory / 1024**2:.2f} MB")

# Check if the results are similar
print(torch.allclose(x_k_dense, x_k_sparse, atol=1e-6))

# Placeholder for conjugate_gradient function

