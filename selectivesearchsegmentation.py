import argparse
import random
import dataclasses
import copy

import cv2
import cv2.ximgproc.segmentation
import torch
import torch.nn.functional as F

def image_scharr_gradients(img : 'BCHW') -> 'BC2HW':
    flipped_scharr_x = torch.tensor([
        [-3, 0, 3 ],
        [10, 0, 10],
        [-3, 0, 3 ]
    ])
    kernel = torch.stack([flipped_scharr_x, flipped_scharr_x.t()]).unsqueeze(1)
    components = F.conv2d(img.flatten(end_dim = -3).unsqueeze(1), kernel).unflatten(0, img.shape[:-2])
    
    dx, dy = components.unbind(dim = -3)
    magnitude = lambda dx, dy: (dy ** 2 + dx ** 2) ** 0.5
    angle = lambda dx, dy: torch.atan2(dy, dx)
    
    if mode == 'magnitude': return magnitude(dx, dy)
    if mode == 'angle': return angle(dx, dy)
    if mode == 'magnitude_angle': return torch.stack([magnitude(dx, dy), angle(dx, dy)], dim = -3)

    return components

@dataclasses.dataclass(order = True)
class RegionNode:
    rank : float
    id : int
    level : int
    merged_to : int
    bbox : tuple

@dataclasses.dataclass(order = True)
class Edge:
    similarity : float
    fro : int
    to : int
    removed : bool

def RegionSimilarity(features, strategies):
    return lambda r1, r2: sum((1.0 / len(strategies)) * s(features, r1, r2) for i, s in enumerate(strategies)) 

def bbox_wh(xywh):
    return (xywh[-2], xywh[-1])

def bbox_size(xywh):
    return int(xywh[-2]) * int(xywh[-1])

def bbox_merge(xywh1, xywh2):
    boxPoints = lambda x, y, w, h: [(x, y), (x + w - 1, y), (x, y + h - 1), (x + w - 1, y + h - 1)]
    points = boxPoints(*xywh1) + boxPoints(*xywh2)
    x1 = min(x1 for x1, y1, x2, y2 in points)
    y1 = min(y1 for x1, y1, x2, y2 in points)
    x2 = max(x2 for x1, y1, x2, y2 in points)
    y2 = max(y2 for x1, y1, x2, y2 in points)
    return (x1, y1, x2 - x1, y2 - y1)

def cvtColor(img_rgb, mode):
    return torch.as_tensor(cv2.cvtColor(img_rgb.movedim(-3, -1).numpy(), getattr(cv2, mode))).movedim(-3, -1) 

def image_gaussian_derivatives(img):
    img_height, img_width = img.shape[-2:]
    center1 = (img_width / 2.0, img_height / 2.0)
    rot1 = cv2.getRotationMatrix2D(center1, 45.0, 1.0)
    xywh1 = cv2.boundingRect(cv2.boxPoints((center1, (img_width, img_height), 45.0)))
    rot1[0, 2] += xywh1[-2] / 2.0 - center1[0]
    rot1[1, 2] += xywh1[-1] / 2.0 - center1[1]
    startx1, starty1 = int(max(0, (xywh1[-2] - img_width) / 2)), int(max(0, (xywh1[-1] - img_height) / 2))
    
    img_rotated = cv2.warpAffine(img, rot1, bbox_wh(xywh1))

    center2 = (int(img_plane_rotated.shape[-1] / 2.0), int(img_plane_rotated.shape[-2] / 2.0))
    rot2 = cv2.getRotationMatrix2D(center2, -45.0, 1.0)
    xywh2 = cv2.boundingRect(cv2.boxPoints((center2, (img_rotated.shape[-1], img_rotated.shape[-2]), -45.0)))

    img_gradients = image_scharr_gradients(img.unsqueeze(0)).squeeze(0)

    img_rotated_gradients = image_scharr_gradients(img_rotated.unsqueeze(0)).squeeze(0)
    img_rotated_gradients = cv2.warpAffine(img_rotated_gradients, rot2, bbox_wh(xywh2))[starty1 : starty1 + img_height, startx1 : startx1 + img_width]
    
    img_gaussians = torch.stack([thresholded for img_plane, img_plane_rotated in zip(img.unbind(-3), img_rotated.unbind(-3)) for tmp_gradient in img_gradients + img_rotated_gradients for thresholded in [img.clamp(min = 0), img.clamp(max = 0)]], dim = -3)

    return img_gaussians

def build_segment_graph(reg_lab):
    nb_segs = 1 + int(reg_lab.max())
    graph_adj = torch.zeros((nb_segs, nb_segs), dtype = torch.bool)
    for i in range(reg_lab.shape[0]):
        p, pr = reg_lab[i], (reg_lab[i - 1] if i > 0 else None)
        for j in range(reg_lab.shape[1]): 
            if i > 0 and j > 0:
                graph_adj[p [j - 1], p[j]] = graph_adj[p[j], p [j - 1]] = True
                graph_adj[pr[j    ], p[j]] = graph_adj[p[j], pr[j    ]] = True
                graph_adj[pr[j - 1], p[j]] = graph_adj[p[j], pr[j - 1]] = True
    return graph_adj

def selective_search(img, base_k = 150, inc_k = 150, sigma = 0.8, min_size = 100, fast = True):
    hsv, lab, gray = [cvtColor(img, mode) for mode in ['COLOR_RGB2HSV', 'COLOR_RGB2Lab', 'COLOR_RGB2GRAY']]
    
    if fast:
        images = [hsv, lab]
        segmentations = [cv2.ximgproc.segmentation.createGraphSegmentation(sigma, float(k), min_size) for k in range(base_k, 1 + base_k + inc_k * 2, inc_k)]
        strategies = lambda features: [RegionSimilarity(features, [HandcraftedRegionFeatures.Fill, HandcraftedRegionFeatures.Texture, HandcraftedRegionFeatures.Size, HandcraftedRegionFeatures.Color]), RegionSimilarity(copy.deepcopy(features), [HandcraftedRegionFeatures.Fill, HandcraftedRegionFeatures.Texture, HandcraftedRegionFeatures.Size])]
    else:
        images = [hsv, lab, gray[..., None], hsv[..., :1], torch.stack([img_bgr[..., 2], img_bgr[..., 1], gray], axis = -1)]
        segmentations = [cv2.ximgproc.segmentation.createGraphSegmentation(sigma, float(k), min_size) for k in range(base_k, 1 + base_k + inc_k * 4, inc_k)]
        strategies = lambda features: [RegionSimilarity(features, [HandcraftedRegionFeatures.Fill, HandcraftedRegionFeatures.Texture, HandcraftedRegionFeatures.Size, HandcraftedRegionFeatures.Color]), RegionSimilarity(copy.deepcopy(features), [HandcraftedRegionFeatures.Fill, HandcraftedRegionFeatures.Texture, HandcraftedRegionFeatures.Size]), RegionSimilarity(copy.deepcopy(features), [HandcraftedRegionFeatures.Fill]), RegionSimilarity(copy.deepcopy(features), [HandcraftedRegionFeatures.Size])]
            
    all_regs = []
    for img in images:
        for gs in segmentations:
            reg_lab = torch.as_tensor(gs.processImage(img.numpy()), dtype = torch.int64)
            graph_adj = build_segment_graph(reg_lab)
            features = HandcraftedRegionFeatures(img, reg_lab, len(graph_adj))
            all_regs.extend(reg for strategy in strategies(features) for reg in hierarchical_grouping(strategy, graph_adj, features.xywh))
    
    return list({region.bbox : True for reg in sorted(all_regs, key = lambda r: r.rank)}.keys())

def hierarchical_grouping(strategy, graph_adj, bbox, compute_rank = lambda region: region.level * random.random()):
    regs, PQ = [], []

    for i in range(len(bbox)):
        regs.append(RegionNode(id = i, level = 1, merged_to = -1, bbox = bbox[i], rank = 0))
        for j in range(i + 1, len(bbox)):
            if graph_adj[i, j]:
                PQ.append(Edge(fro = i, to = j, similarity = strategy(i, j), removed = False))
    
    while PQ:
        PQ.sort(key = lambda sim: sim.similarity)
        e = PQ.pop()
        u, v = e.fro, e.to
        
        reg_fro, reg_to = regs[u], regs[v]
        regs.append(RegionNode(id = min(reg_fro.id, reg_to.id), level = 1 + max(reg_fro.level, reg_to.level), merged_to = -1, bbox = bbox_merge(reg_fro.bbox, reg_to.bbox), rank = 0))
        regs[u].merged_to = regs[v].merged_to = len(regs) - 1
        
        strategy.merge(reg_fro.id, reg_to.id)
        
        local_neighbours = set()
        for e in PQ:
            if (u == e.fro or u == e.to) or (v == e.fro or v == e.to):
                local_neighbours.add(e.to if e.fro == u or e.fro == v else e.fro)
                e.removed = True
        PQ = [sim for sim in PQ if not sim.removed]
        for local_neighbour in local_neighbours:
            PQ.append(Edge(fro = len(regs) - 1, to = local_neighbour, similarity = strategy(regs[len(regs) - 1].id, regs[local_neighbour].id), removed = False))

    for region in regs:
        region.rank = compute_rank(region)

    return regs

class HandcraftedRegionFeatures:
    def __init__(self, img, reg_lab, nb_segs, color_histogram_bins_size = 25, texture_histogram_bins_size = 10):
        img_channels, img_height, img_width = img.shape[-3:]
        self.img_size = img_height * img_width
        
        self.xywh = torch.tensor([[img_wdith, img_height, 0, 0]], dtype = torch.int64).repeat(nb_segs, 1)
        for y in range(reg_lab.shape[-2]):
            for x in range(reg_lab.shape[-1]):
                xywh = self.xywh[reg_lab[y, x]]
                xywh[0].clamp_(max = x)
                xywh[1].clamp_(max = y)
                xywh[2].clamp_(min = x)
                xywh[3].clamp_(max = y)
        self.xywh[..., 2] -= self.xywh[..., 0]
        self.xywh[..., 3] -= self.xywh[..., 1]
        
        Z = reg_lab
        self.region_sizes = torch.zeros((nb_segs, ), dtype = torch.float32).scatter_(-1, Z.flatten(start_dim = -2), 1, reduce = 'add')
        
        Z = reg_lab * color_histogram_bins_size + (img / (256.0 / color_histogram_bins_size).to(torch.int64)
        self.color_histograms = torch.zeros((img_channels, nb_segs * color_histogram_bins_size), dtype = torch.float32).scatter_(-1, Z.flatten(start_dim = -2), 1, reduce = 'add').view(img_channels, nb_segs, -1).movedim(-2, -3)
        self.color_histograms /= self.color_histograms.sum(dim = (-2, -1), keepdim = True)

        img_gaussians = image_gaussian_derivatives(img)
        hmin, hmax = img_gaussians.amin(dim = (-2, -1), keepdim = True), img_gaussians.amax(dim = (-2, -1), keepdim = True)
        Z = reg_lab * texture_histogram_bins_size + (((img_gaussians - hmin) / (hmax - hmin) * 255) / (256.0 / texture_histogram_bins_size)).to(torch.int64)
        self.texture_histograms = torch.zeros((img_channels, 8, nb_segs * texture_histogram_bins_size), dtype = torch.float32).scatter_(-1, Z.flatten(start_dim = -2), 1, reduce = 'add').view(img_channels, 8, nb_segs, -1).movedim(-2, -5)
        self.texture_histograms /= self.texture_histograms.sum(dim = (-3, -2, -1), keepdim = True)

    def merge(self, r1, r2):
        self.xywh[r1] = self.xywh[r2] = bbox_merge(self.xywh[r1], self.xywh[r2])
        self.region_sizes[r1] = self.region_sizes[r2] = self.region_sizes[r1] + self.region_sizes[r2]
        self.color_histograms[r1] = self.color_histograms[r2] = (self.color_histograms[r1] * self.region_sizes[r1] + self.color_histograms[r2] * self.region_sizes[r2]) / (self.region_sizes[r1] + self.region_sizes[r2]) 
        self.texture_histograms[r1] = self.texture_histograms[r2] = (self.texture_histograms[r1] * self.region_sizes[r1] + self.texture_histograms[r2] * self.region_sizes[r2]) / (self.region_sizes[r1] + self.region_sizes[r2]) 
    
    def Fill(self, r1, r2):
        return max(0.0, min(1.0, 1.0 - float(bbox_size(bbox_merge(self.xywh[r1], self.xywh[r2])) - self.region_sizes[r1] - self.region_sizes[r2]) / float(self.img_size)))

    def Size(self, r1, r2):
        return max(0.0, min(1.0, 1.0 - float(self.region_sizes[r1] + self.region_sizes[r2]) / float(self.img_size)))
    
    def Color(self, r1, r2):
        return torch.min(self.color_histograms[r1], self.color_histograms[r2]).sum(dim = (-2, -1))

    def Texture(self, r1, r2):
        return torch.min(self.texture_histograms[r1], self.texture_histograms[r2]).sum(dim = (-2, -1))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', '-i')
    parser.add_argument('--output-path', '-o')
    parser.add_argument('--topk', type = int, default = 100)
    parser.add_argument('--fast', action = 'store_true')
    parser.add_argument('--opencv', action = 'store_true')
    args = parser.parse_args()
    args.fast = True

    img = cv2.imread(args.input_path)

    if not args.opencv:
        boxes_xywh = selective_search(img, fast = args.fast)
    else:
        algo = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        algo.setBaseImage(img)
        algo.switchToSelectiveSearchFast() if args.fast else algo.switchToSelectiveSearchQuality()
        boxes_xywh = algo.process()

    print('# boxes', len(boxes_xywh))
    print('height:', img.shape[0], 'width:', img.shape[1])
    print('ymax', max(y + h - 1 for x, y, w, h in boxes_xywh), 'xmax', max(x + w - 1 for x, y, w, h in boxes_xywh))

    for x, y, w, h in boxes_xywh[:args.topk]:
        cv2.rectangle(img, (x, y), (x + w - 1, y + h - 1), (0, 0, 255), 1)
    cv2.imwrite(args.output_path, img)
