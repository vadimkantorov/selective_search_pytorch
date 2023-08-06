import argparse

import torch
import torch.nn as nn
import torchvision

import selectivesearchsegmentation
from opencv_custom import selectivesearchsegmentation_opencv_custom

import hyplcapoincare
import hypadam

class MergedRegionsForestDataset(torch.utils.data.Dataset):
    def __init__(self, regions):
        self.sim_graph = self.build_graph(regions)
        self.triples = self.generate_triples()

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        triple = self.triples[idx]
        s12 = self.sim_graph[triple[0], triple[1]]
        s13 = self.sim_graph[triple[0], triple[2]]
        s23 = self.sim_graph[triple[1], triple[2]]
        sim_graph = torch.tensor([s12, s13, s23])
        return triple, sim_graph
    
    def generate_triples(self):
        I = torch.arange(self.sim_graph.shape[0])
        return torch.stack(torch.meshgrid(I, I, I, indexing = 'ij'), dim = -1).flatten(end_dim = -2)

    @staticmethod
    def build_graph(regions):
        sim = -torch.ones((len(regions), len(regions)), dtype = torch.float32)
        
        for i, ri in enumerate(regions):
            for j, rj in enumerate(regions):
                if i < j and ri['level'] == rj['level'] == 0:
                    for k, rk in enumerate(regions):
                        if ri['id'] in rk['ids'] and rj['id'] in rk['ids']:
                            sim[i, j] = sim[j, i] = sum([rk['level'] - ri['level'], rk['level'] - rj['level'] ]) / 2.0
                            break

        breakpoint()
        sim = (1 - sim / sim.amax()).masked_fill_(sim < 0, 0.0).clamp_(min = 0.05)
        sim.diagonal().fill_(1)
        return sim

class HypHCVisualEmbedding(nn.Embedding):
    # adapted from https://github.com/HazyResearch/HypHC
    def __init__(self, num_embeddings: int = 1, embedding_dim: int = 2, max_norm: float = 1. - 1e-3, init_size: float = 1e-3):
        super().__init__(num_embeddings, embedding_dim)
        self.init_size = init_size
        self.max_norm_ = max_norm
        self.scale = nn.Parameter(torch.tensor([init_size]))
        
        with torch.no_grad():
            self.weight.copy_(hyplcapoincare.project(torch.nn.init.uniform_(self.weight, 0.0, 1.0).mul_(2.0).sub_(1.0).mul_(self.scale)))

    def forward(self, x : 'B3') -> 'B3D':
        return torch.nn.functional.normalize(super().forward(x), p = 2, dim = -1) * self.scale.clamp(max = self.max_norm_, min = 1e-2) #self.init_size

    def loss(self, emb_pred : 'B3D', sim_gt : 'B3 # [s12, s13, s23]', temperature: float = 0.01):
        e1, e2, e3 = emb_pred.unbind(-2)
        breakpoint()
        d_12 = hyplcapoincare.hyp_lca(e1, e2, return_coord=False)
        d_13 = hyplcapoincare.hyp_lca(e1, e3, return_coord=False)
        d_23 = hyplcapoincare.hyp_lca(e2, e3, return_coord=False)
        lca_norm = torch.cat([d_12, d_13, d_23], dim=-1)
        weights = torch.softmax(lca_norm / temperature, dim=-1)
        w_ord = torch.sum(sim_gt * weights, dim=-1, keepdim=True)
        total = torch.sum(sim_gt, dim=-1, keepdim=True) - w_ord
        return torch.mean(total)


def main(args):
    torch.manual_seed(args.seed)

    selsearch = selectivesearchsegmentation.SelectiveSearch(preset = 'single')
    
    img = torchvision.io.read_image(args.input_path) / 255.0
    boxes_xywh, regions, reg_lab = selsearch(img.unsqueeze(0))
    regions = regions[0]

    dataset = MergedRegionsForestDataset(regions)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size = args.batch_size, num_workers = args.num_workers, shuffle = True, pin_memory = True)
    
    model = HypHCVisualEmbedding(dataset.sim_graph.shape[0], args.embedding_dim)
    optimizer = hypadam.RiemannianAdam(model.parameters(), lr = args.lr)

    for step, (triple_ids, sim_gt) in enumerate(data_loader):
        inp_ids, sim_gt = triple_ids.to(args.device), sim_gt.to(args.device)
        emb_pred = model(inp_ids)
        loss = model.loss(emb_pred, sim_gt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            breakpoint()
            print(step, '/', len(data_loader), float(loss), float(model.scale))

    
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
