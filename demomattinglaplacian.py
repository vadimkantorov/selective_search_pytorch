# https://github.com/vadimkantorov/fastcontextlocnet/blob/fcn/spectral.py

# https://github.com/MarcoForte/closed-form-matting

import torch

def matting_laplacian(x, sigma = 1):
    h, w = x.shape[:2]
    win_rad = int(sigma)
    img = x.view(h, w, -1)
    breakpoint()
    win_diam, win_size = win_rad * 2 + 1, (win_rad * 2 + 1) ** 2

    patches_inds = torch.arange(h * w).as_strided((h - win_diam + 1, w - win_diam + 1, win_diam, win_diam), (w, 1, w, 1)).reshape(h - 2 * win_rad, w - 2 * win_rad, win_size)
    patches = img.flatten(end_dim = 1)[patches_inds]
    centered = patches - patches.mean(dim = 2, keepdim = True)
    variance = centered.transpose(-1, -2).matmul(centered) / win_size
    inv = torch.inverse((variance + (eps / win_size) * torch.eye(variance.shape[-1], out = variance.new())).flatten(end_dim = -3)).view_as(variance)
    rowcol = torch.stack([patches_inds.unsqueeze(-1).expand(-1, -1, -1, win_size).flatten(), patches_inds.unsqueeze(-2).expand(-1, -1, win_size, -1).flatten()])
    vals = (1 + centered.matmul(inv).matmul(centered.transpose(-1, -2))) / win_size
    
    return torch.sparse_coo_tensor(rowcol, vals.flatten(), (h * w, h * w)).to_dense()

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    img = torch.as_tensor(plt.imread('astronaut.jpg').copy())
    L = matting_laplacian(img)
    print(L.shape)
