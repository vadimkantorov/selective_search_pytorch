# https://github.com/MarcoForte/closed-form-matting/blob/master/closed_form_matting/closed_form_matting.py
# https://github.com/vadimkantorov/fastcontextlocnet/blob/fcn/spectral.py

import torch
import scipy.sparse.linalg

def matting_laplacian(img, mask = None, eps = 1e-7, win_rad = 1):
    assert img.dtype == torch.float64 and img.ndim == 3 and img.shape[-1] == 3
    assert mask is None or (mask.ndim == 2 and mask.shape == img.shape[:2] and mask.dtype == torch.bool)
    win_diam, win_size = win_rad * 2 + 1, (win_rad * 2 + 1) ** 2
    h, w, d = img.shape[-3:]

    I = torch.arange(h * w).view(h, w).as_strided((h - win_diam + 1, w - win_diam + 1, win_diam, win_diam), stride = (w, 1, w, 1)).flatten(start_dim = -2)
    
    if mask is not None:
        #mask = cv2.dilate(mask.astype(np.uint8),np.ones((win_diam, win_diam), np.uint8)).astype(bool)
        win_mask = torch.sum(mask.ravel()[I], dim=2)
        I = I[win_mask > 0, :]
    else:
        I = I.flatten(end_dim = -2)

    winI = img.flatten(end_dim = -2)[I]
    win_mu = winI.mean(dim = -2, keepdim = True)
    win_var = torch.einsum('...ji,...jk ->...ik', winI, winI) / win_size - torch.einsum('...ji,...jk ->...ik', win_mu, win_mu)

    A = win_var + (eps/win_size)*torch.eye(win_var.shape[-1], dtype = win_var.dtype, device = win_var.device)
    B = (winI - win_mu).permute(0, 2, 1)
    X = torch.linalg.solve(A, B).permute(0, 2, 1)

    vals = torch.eye(win_size, dtype = X.dtype, device = X.device) - (1.0/win_size)*(1 + X @ B)

    nz_indsCol = torch.tile(I, (1, win_size)).flatten()
    nz_indsRow = torch.repeat_interleave(I, win_size).flatten()
    nz_indsVal = vals.flatten()

    L = torch.sparse_coo_tensor(torch.stack([nz_indsRow, nz_indsCol]), nz_indsVal, size=(h*w, h*w))
    #L = torch.sparse_csr_tensor(torch.arange(0, vals.numel() + 1, win_size), nz_indsCol, vals.flatten(), size = (h * w, h * w))
    #L_numpy = scipy.sparse.csr_matrix((vals.flatten().numpy(), nz_indsCol.numpy(), torch.arange(0, vals.numel() + 1, win_size).numpy()), shape = (h * w, h * w))
    return L

def closed_form_matting_with_prior(img, prior, prior_confidence, consts_map=None):
    #prior: matrix of same width and height as input img holding apriori alpha map.
    #prior_confidence: matrix of the same shape as prior hodling confidence of prior alpha.
    #consts_map: binary mask of pixels that aren't expected to change due to high prior confidence.

    laplacian = matting_laplacian(img, ~consts_map if consts_map is not None else None)
    confidence = torch.sparse.spdiags(prior_confidence.ravel(), offsets = torch.tensor([0]), shape = laplacian.shape, layout = laplacian.layout)
    laplacian_confidence = laplacian + confidence

    alpha = torch.as_tensor(scipy.sparse.linalg.spsolve(scipy.sparse.coo_array((laplacian_confidence._values(), laplacian_confidence._indices()), shape = laplacian_confidence.shape).tocsr(), (prior.ravel() * prior_confidence.ravel()).numpy()))
    
    return alpha.reshape_as(prior).clamp(0.0, 1.0)

if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', '-i', default = 'astronaut.jpg')
    parser.add_argument('--output-path', '-o', default = 'astronaut.jpg.png')
    parser.add_argument('--trimap')
    parser.add_argument('--scribbles')
    parser.add_argument('--confidence', type = float, default = 1.0)
    args = parser.parse_args()

    img = torch.as_tensor(plt.imread(args.input_path).copy()).flip(-1).to(torch.float64) / 255.0
    guidance = torch.as_tensor(plt.imread(args.scribbles or args.trimap).copy()).flip(-1).to(torch.float64) / 255.0
    assert img.shape[:2] == guidance.shape[:2]
    
    if args.scribbles:
        guidance = torch.sign(torch.sum(guidance - img, dim=2)) / 2 + 0.5
        consts_map = guidance != 0.5
    
    if args.trimap:
        guidance = guidance.mean(dim = -1)
        consts_map = (guidance < 0.1) | (guidance > 0.9)
    
    alpha = closed_form_matting_with_prior(img, guidance, args.confidence * consts_map, consts_map)

    plt.imsave(args.output_path, alpha * 255.0)
