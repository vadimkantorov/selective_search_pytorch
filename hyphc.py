import argparse

import torch
import torch.nn as nn
import torchvision

import selectivesearchsegmentation
from opencv_custom import selectivesearchsegmentation_opencv_custom


class ForestDataset(torch.utils.data.Dataset):
    def __init__(self, regions):
        self.similarities = self.build_graph(regions)
        self.triples = self.generate_triples()

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        triple = self.triples[idx]
        s12 = self.similarities[triple[0], triple[1]]
        s13 = self.similarities[triple[0], triple[2]]
        s23 = self.similarities[triple[1], triple[2]]
        similarities = torch.tensor([s12, s13, s23])
        return triple, similarities
    
    def generate_triples(self):
        I = torch.arange(self.similarities.shape[0])
        return torch.stack(torch.meshgrid(I, I, I, indexing = 'ij'), dim = -1).flatten(end_dim = -2)

    @staticmethod
    def build_graph(regions, neginf = -50):
        sim = torch.full((len(regions), len(regions)), fill_value = neginf, dtype = torch.float32)
        
        for i, ri in enumerate(regions):
            for j, rj in enumerate(regions):
                if i < j and ri['level'] == rj['level'] == 0:
                    for k, rk in enumerate(regions):
                        if ri['id'] in rk['ids'] and rj['id'] in rk['ids']:
                            sim[i, j] = -rk['level']
                            break

        sim.diagonal().fill_(0.0)
        return sim

def main(args):
    selsearch = selectivesearchsegmentation.SelectiveSearch(preset = 'single')
    
    img = torchvision.io.read_image(args.input_path) / 255.0
    boxes_xywh, regions, reg_lab = selsearch(img.unsqueeze(0))
    regions = regions[0]

    dataset = ForestDataset(regions)
    print(dataset[0])

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', '-i', default = 'astronaut.jpg')
    args = parser.parse_args()
    
    main(parser.parse_args())
