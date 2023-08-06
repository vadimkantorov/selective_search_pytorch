import argparse

import torch
import torch.nn as nn
import torchvision

import selectivesearchsegmentation
from opencv_custom import selectivesearchsegmentation_opencv_custom


class ForestDataset(torch.utils.data.Dataset):
    def __init__(self, regions):
        pass

    def __getitem__(self, idx):
        pass

def main(args):
    selsearch = selectivesearchsegmentation.SelectiveSearch(preset = 'single')
    img = torchvision.io.read_image(args.input_path) / 255.0
    boxes_xywh, regions, reg_lab = selsearch(img.unsqueeze(0))

    dataset = ForestDataset(regions)

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', '-i', default = 'astronaut.jpg')
    args = parser.parse_args()
    
    main(parser.parse_args())
