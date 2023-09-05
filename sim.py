import torch
import torchvision

import matplotlib, matplotlib.pyplot as plt

import selectivesearchsegmentation

def spectral_clustering(Asym, k = 1):
    Dsqrt = Asym.sum(dim = -1).sqrt()
    normalized_symmetric_laplacian = torch.ones_like(Dsqrt).diag() - Asym / Dsqrt.unsqueeze(-1) / Dsqrt.unsqueeze(-2)
    eigval, eigvec = torch.linalg.eigh(normalized_symmetric_laplacian)
    return eigvec[:, k]

def spectral_order(Asym):
    unnormalized_symmetric_laplacian = Asym.sum(dim = -1).diag() - Asym
    Dsqrt = Asym.sum(dim = -1).sqrt()
    normalized_symmetric_laplacian = torch.ones_like(Dsqrt).diag() - Asym / Dsqrt.unsqueeze(-1) / Dsqrt.unsqueeze(-2)
    eigval, eigvec = torch.linalg.eigh(normalized_symmetric_laplacian)
    return eigvec[:, 1].argsort()

def segment_scatter(f, regions, reg_lab):
    max_num_segments = 1 + reg_lab.amax()
    y = torch.zeros(reg_lab.shape[:-2] + (max_num_segments, ), dtype = f.dtype)
    for i, reg in enumerate(regions):
        y[reg['plane_id'][:-1]][reg['id']] = f[i]
    return y.gather(-1, reg_lab.to(torch.int64).flatten(start_dim = -2)).unflatten(-1, reg_lab.shape[-2:])

selsearch = selectivesearchsegmentation.SelectiveSearch(preset = 'single', return_region_features = True, randomize_rank = False)
img_rgb3hw_255 = torchvision.io.read_image('examples/astronaut.jpg')
regions, reg_lab = selsearch(img_rgbb3hw_1 = img_rgb3hw_255.div(255).unsqueeze(0).contiguous())[-2:]
regions, reg_lab = sorted([dict(reg, plane_id = reg['plane_id'][1:]) for reg in regions[0]], key = lambda reg: reg['idx']), reg_lab[0]
assert all(regions[i]['idx'] == i for i in range(len(regions)))

simhist = (selectivesearchsegmentation.HandcraftedRegionFeatures.compute_region_affinity_pdist(img_size = img_rgb3hw_255.shape[-2] * img_rgb3hw_255.shape[-1], region_size = torch.tensor([reg['region_size'] for reg in regions], dtype = torch.int32), bbox_xywh = torch.tensor([reg['bbox_xywh'] for reg in regions], dtype = torch.int16), color_hist = torch.stack([reg['color_hist'] for reg in regions]), texture_hist = torch.stack([reg['texture_hist'] for reg in regions])) * torch.tensor(selsearch.strategies[0])).sum(-1)

l = torch.tensor([i for i, reg in enumerate(regions) if reg['level'] == 1], dtype = torch.int64)
u = torch.tensor([reg['idx'] for reg in regions], dtype = torch.int64)
v = torch.tensor([reg['parent_idx'] for reg in regions], dtype = torch.int64) # fixup tree root
adj = torch.zeros((len(regions), len(regions)), dtype = torch.bool)
adj[u, v] = True
adj[v, u] = True
dist = torch.full_like(adj, float('inf'), dtype = torch.float32).masked_fill_(adj, 1).fill_diagonal_(0)
for k in range(dist.shape[-1]):
    dist = torch.min(dist, dist[:, k, None] + dist[None, k, :])
A = (1 - dist / dist.amax()).clamp_(min = 0.01).fill_diagonal_(1)
#A = torch.where(dist <= 3, 1.0, 0.0).fill_diagonal_(1)

f = torch.zeros(len(regions), dtype = torch.int64)
k = 1
for i, regi in enumerate(regions):
    for j, regj in enumerate(regions):
        if i < j and regi['level'] == 1 and regj['level'] == 1 and regi['parent_idx'] == regj['parent_idx']:
            f[i] = f[j] = k
            k += 1
max_num_segments = 1 + int(f.amax())
f = segment_scatter(f, regions, reg_lab)[0, 0]
colormap = torch.as_tensor(matplotlib.colormaps['jet'].resampled(max_num_segments)(range(max_num_segments)))
colormap = colormap[torch.randperm(max_num_segments, generator = torch.Generator().manual_seed(42))]
f = colormap[f] 
plt.figure()
plt.imshow(f)
plt.colorbar()
plt.savefig('deg_.png')
plt.close()

#A = simhist
I = spectral_order(A)
plt.figure()
plt.imshow(A[I[l][:, None], I[l][None, :]])
plt.colorbar()
plt.savefig('sim.png')
plt.close()

f = segment_scatter(spectral_clustering(A, k = 2), regions, reg_lab)[0, 0]
plt.figure()
plt.imshow(f)
plt.colorbar()
plt.savefig('deg.png')
plt.close()
