import torch
import torchvision

import matplotlib.pyplot as plt

import selectivesearchsegmentation

spectral_order = lambda A: torch.linalg.eigh(A.sum(dim = -1, keepdim = True) - A).eigenvectors[:, 1].argsort()


selsearch = selectivesearchsegmentation.SelectiveSearch(preset = 'single', return_region_features = True)
img_rgb3hw_255 = torchvision.io.read_image('examples/astronaut.jpg')
regions = selsearch(img_rgbb3hw_1 = img_rgb3hw_255.div(255).unsqueeze(0).contiguous())[1][0]

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

A = simhist
I = spectral_order(A)
A = A[I[l][:, None], I[l][None, :]]
#A = A[I[:, None], I[None, :]]
#A = A[I][:, I]

plt.imshow(A)
plt.colorbar()
plt.savefig('sim.png')
