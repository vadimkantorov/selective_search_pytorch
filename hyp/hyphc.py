import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import selectivesearchsegmentation
from opencv_custom import selectivesearchsegmentation_opencv_custom

import hyplcapoincare
import hypadam

class MergedRegionsForestDataset(torch.utils.data.Dataset):
    def __init__(self, regions):
        dist = self.calc_tree_dist(regions)
        #self.sim_graph = (1 - dist / dist.amax()).clamp_(min = 0.01).fill_diagonal_(1)
        self.sim_graph = torch.where(dist < 5, 1.0, 0.01).fill_diagonal_(1)
        self.triples = self.generate_triples()

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        triple = self.triples[idx]
        s01 = self.sim_graph[triple[0], triple[1]]
        s02 = self.sim_graph[triple[0], triple[2]]
        s12 = self.sim_graph[triple[1], triple[2]]
        sim_graph = torch.tensor([s01, s02, s12])
        return triple, sim_graph
    
    def generate_triples(self):
        I = torch.arange(self.sim_graph.shape[0])
        return torch.stack(torch.meshgrid(I, I, I, indexing = 'ij'), dim = -1).flatten(end_dim = -2)

    @staticmethod
    def calc_tree_dist(regions):
        u = torch.tensor([reg['idx'] for reg in regions], dtype = torch.int64)
        v = torch.tensor([reg['parent_idx'] for reg in regions], dtype = torch.int64)
        # TODO: fixup tree root
        
        adj = torch.zeros((len(regions), len(regions)), dtype = torch.bool)
        adj[u, v] = True
        adj[v, u] = True

        dist = torch.full_like(adj, float('inf'), dtype = torch.float32).masked_fill_(adj, 1.0).fill_diagonal_(0)
        for k in range(dist.shape[-1]):
            dist = torch.min(dist, dist[:, k, None] + dist[None, k, :])

        return dist

class HypHCVisualEmbedding(nn.Embedding):
    # adapted from https://github.com/HazyResearch/HypHC
    def __init__(self, num_embeddings: int = 1, embedding_dim: int = 2, max_norm: float = 1. - 1e-3, init_size: float = 1e-3):
        super().__init__(num_embeddings, embedding_dim)
        self.init_size = init_size
        self.max_norm_ = max_norm
        self.scale = nn.Parameter(torch.tensor([init_size]))
        
        with torch.no_grad():
            #self.weight.copy_(hyplcapoincare.project(torch.nn.init.uniform_(self.weight, 0.0, 1.0).mul_(2.0).sub_(1.0).mul_(self.scale)))
            self.weight.copy_(torch.nn.init.uniform_(self.weight, 0.0, 1.0).mul_(2.0).sub_(1.0).mul_(self.scale))

    def forward(self, x : 'B3') -> 'B3D':
        return F.normalize(super().forward(x), p = 2, dim = -1) * self.scale.clamp(max = self.max_norm_, min = 1e-2) #self.init_size

    def loss(self, emb_pred : 'B3D', sim_gt : 'B3 # [s12, s13, s23]', temperature: float = 0.01):
        e0, e1, e2 = emb_pred.unbind(-2)
        d_01 = hyplcapoincare.hyp_lca(e0, e1, return_coord=False)
        d_02 = hyplcapoincare.hyp_lca(e0, e2, return_coord=False)
        d_12 = hyplcapoincare.hyp_lca(e1, e2, return_coord=False)
        weights = torch.softmax(torch.cat([d_01, d_02, d_12], dim=-1) / temperature, dim=-1)
        print(weights)
        #total = torch.sum(sim_gt, dim=-1, keepdim=True) - torch.sum(sim_gt * weights, dim=-1, keepdim=True)
        total = (-sim_gt * weights).sum(dim=-1)
        return torch.mean(total)


def main(args):
    torch.manual_seed(args.seed)

    selsearch = selectivesearchsegmentation.SelectiveSearch(preset = 'single')
    
    img = torchvision.io.read_image(args.input_path) / 255.0
    boxes_xywh, regions, reg_lab = selsearch(img_rgbb3hw_1 = img.unsqueeze(0).contiguous())
    regions = regions[0]

    dataset = MergedRegionsForestDataset(regions)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size = args.batch_size, num_workers = args.num_workers, shuffle = True, pin_memory = True)
    
    model = HypHCVisualEmbedding(dataset.sim_graph.shape[0], args.embedding_dim)
    #optimizer = hypadam.RiemannianAdam(model.parameters(), lr = args.lr)
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    for step, (triple_ids, sim_gt) in enumerate(data_loader):
        inp_ids, sim_gt = triple_ids.to(args.device), sim_gt.to(args.device)
        emb_pred = model(inp_ids)
        loss = model.loss(emb_pred, sim_gt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print(step, '/', len(data_loader), float(loss))

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', '-i', default = 'astronaut.jpg')
    parser.add_argument('--seed', type = int, default = 42)
    parser.add_argument('--epochs', type = int, default = 50)
    parser.add_argument('--batch-size', type = int, default = 256)
    parser.add_argument('--num-workers', type = int, default = 0)
    parser.add_argument('--lr', type = float, default = 1e-3)
    parser.add_argument('--embedding-dim', type = int, default = 2)
    parser.add_argument('--device', type = str, default = 'cpu')
    args = parser.parse_args()
    
    main(parser.parse_args())

# TODO: 
# - idea is to be able to retrieve pixels by selecting other pixels within a hyperbolic ball
# - need to be use the intermediate tree nodes somehow?
# - is tree distance a good metric to embed leaves?
