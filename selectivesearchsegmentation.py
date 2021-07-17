import argparse
import random
import dataclasses
import copy

import numpy as np
import cv2
import cv2.ximgproc.segmentation
import torch

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
    return xywh[-2] * xywh[-1]

def bbox_merge(xywh1, xywh2):
    boxPoints = lambda x, y, w, h: [(x, y), (x + w - 1, y), (x, y + h - 1), (x + w - 1, y + h - 1)]
    return cv2.boundingRect(np.array(boxPoints(*xywh1) + boxPoints(*xywh2)))

def selective_search(img_bgr, base_k = 150, inc_k = 150, sigma = 0.8, min_size = 100, fast = True):
    hsv, lab, gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV), cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab), cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    if fast:
        images = [hsv, lab]
        segmentations = [ cv2.ximgproc.segmentation.createGraphSegmentation(sigma, float(k), min_size) for k in range(base_k, 1 + base_k + inc_k * 2, inc_k) ]
        strategies = lambda features: [RegionSimilarity(features, [HandcraftedRegionFeatures.Fill, HandcraftedRegionFeatures.Texture, HandcraftedRegionFeatures.Size, HandcraftedRegionFeatures.Color]), RegionSimilarity(copy.deepcopy(features), [HandcraftedRegionFeatures.Fill, HandcraftedRegionFeatures.Texture, HandcraftedRegionFeatures.Size])]
    else:
        images = [hsv, lab, gray[..., None], hsv[..., :1], np.stack([img_bgr[..., 2], img_bgr[..., 1], gray], axis = -1)]
        segmentations = [ cv2.ximgproc.segmentation.createGraphSegmentation(sigma, float(k), min_size) for k in range(base_k, 1 + base_k + inc_k * 4, inc_k) ]
        strategies = lambda features: [RegionSimilarity(features, [HandcraftedRegionFeatures.Fill, HandcraftedRegionFeatures.Texture, HandcraftedRegionFeatures.Size, HandcraftedRegionFeatures.Color]), RegionSimilarity(copy.deepcopy(features), [HandcraftedRegionFeatures.Fill, HandcraftedRegionFeatures.Texture, HandcraftedRegionFeatures.Size]), RegionSimilarity(copy.deepcopy(features), [HandcraftedRegionFeatures.Fill]), RegionSimilarity(copy.deepcopy(features), [HandcraftedRegionFeatures.Size])]
            
    all_regs = []
    for img in images:
        for gs in segmentations:
            reg_lab = gs.processImage(img)
            nb_segs = 1 + int(reg_lab.max())

            graph_adj = np.zeros((nb_segs, nb_segs), dtype = bool)
            for i in range(reg_lab.shape[0]):
                p, pr = reg_lab[i], (reg_lab[i - 1] if i > 0 else None)
                for j in range(reg_lab.shape[1]): 
                    if i > 0 and j > 0:
                        graph_adj[p [j - 1], p[j]] = graph_adj[p[j], p [j - 1]] = True
                        graph_adj[pr[j    ], p[j]] = graph_adj[p[j], pr[j    ]] = True
                        graph_adj[pr[j - 1], p[j]] = graph_adj[p[j], pr[j - 1]] = True

            features = HandcraftedRegionFeatures(img, reg_lab, nb_segs)
            all_regs.extend(reg for strategy in strategies(features) for reg in hierarchical_grouping(strategy, graph_adj, features.xyxy))
    
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
        img = torch.as_tensor(img).movedim(-1, -3)
        reg_lab = torch.as_tensor(reg_lab, dtype = torch.int64)

        img_channels, img_height, img_width = img.shape[-3:]
        self.img_size = img_height * img_width
        self.color_histograms   = torch.zeros((nb_segs, img_channels, 1 * color_histogram_bins_size  ), dtype = torch.float32)
        self.texture_histograms = torch.zeros((nb_segs, img_channels, 8 * texture_histogram_bins_size), dtype = torch.float32)
        self.region_sizes = torch.zeros((nb_segs, ), dtype = torch.int64)
        self.xyxy = torch.zeros((nb_segs, 4), dtype = torch.int64)
        
        
        #points = [[] for k in range(nb_segs)]
        #for i in range(reg_lab.shape[0]):
        #    for j in range(reg_lab.shape[1]): 
        #        points[reg_lab[i, j]].append( (j, i) )
        #        self.region_sizes[reg_lab[i, j]] += 1
        #self.bounding_boxes = [cv2.boundingRect(np.array(pts)) for pts in points]

        for y in range(reg_lab.shape[-2]):
            for x in range(reg_lab.shape[-1]):
                xyxy = self.xyxy[reg_lab[y, x]]
                xyxy[0].clamp_(max = x)
                xyxy[1].clamp_(max = y)
                xyxy[2].clamp_(min = x)
                xyxy[3].clamp_(max = y)
        Z = reg_lab.flatten(start_dim = -2)
        self.region_sizes = torch.zeros((nb_segs, ), dtype = torch.int64).scatter_add_(-1, Z, Z.new_ones(1).expand_as(Z))
        
        for r in range(len(region_sizes)):
            mask = (reg_lab == r).astype(np.uint8) * 255
            for p in range(img_channels):
                self.color_histograms[r, p] = np.squeeze(cv2.calcHist([img[..., p]], [0], mask, [color_histogram_bins_size], [0, 256]))
        self.color_histograms /= np.sum(self.color_histograms, axis = (1, 2), keepdims = True)

        img_gaussians = []
        for p in range(img_channels):
            img_plane = img[..., p]
            
            center = (img_width / 2.0, img_height / 2.0)
            rot = cv2.getRotationMatrix2D(center, 45.0, 1.0)
            xywh = cv2.boundingRect(cv2.boxPoints((center, (img_width, img_height), 45.0)))
            rot[0, 2] += xywh[-2] / 2.0 - center[0]
            rot[1, 2] += xywh[-1] / 2.0 - center[1]
            img_plane_rotated = cv2.warpAffine(img_plane, rot, bbox_wh(xywh))
            img_plane_rotated_size = img_plane_rotated.shape[0] * img_plane_rotated.shape[1]
            
            center = (int(img_plane_rotated.shape[1] / 2.0), int(img_plane_rotated.shape[0] / 2.0))
            rot = cv2.getRotationMatrix2D(center, -45.0, 1.0)
            xywh2 = cv2.boundingRect(cv2.boxPoints((center, (img_plane_rotated.shape[1], img_plane_rotated.shape[0]), -45.0)))
            start_x, start_y = int(max(0, (xywh[-2] - img_width) / 2)), int(max(0, (xywh[-1] - img_height) / 2))

            for gr in [(1, 0), (0, 1)]:
                tmp_gradient = cv2.Scharr(img_plane, cv2.CV_32F, *gr)
                img_gaussians.extend(cv2.threshold(tmp_gradient, 0, 0, type)[-1] for type in [cv2.THRESH_TOZERO, cv2.THRESH_TOZERO_INV])

            for gr in [(1, 0), (0, 1)]:
                tmp_gradient = cv2.Scharr(img_plane_rotated, cv2.CV_32F, *gr)
                tmp_rot = cv2.warpAffine(tmp_gradient, rot, bbox_wh(xywh2))
                tmp_rot = tmp_rot[start_y : start_y + img_height, start_x : start_x + img_width]
                img_gaussians.extend(cv2.threshold(tmp_rot, 0, 0, type)[-1] for type in [cv2.THRESH_TOZERO, cv2.THRESH_TOZERO_INV])

        for i, img_plane in enumerate(img_gaussians):
            hmin, hmax = img_plane.min(), img_plane.max()
            img_gaussians[i] = (255 * (img_plane - hmin) / (hmax - hmin)).astype(np.uint8)

        for i in range(reg_lab.shape[0]):
            for j in range(reg_lab.shape[1]):
                for p in range(img_channels):
                    for k in range(8):
                        val = int(img_gaussians[p * 8 + k][i, j])
                        bin = int(float(val) / (256.0 / texture_histogram_bins_size))
                        self.texture_histograms[reg_lab[i, j], p, k * texture_histogram_bins_size + bin] += 1
        self.texture_histograms /= np.sum(self.texture_histograms, axis = (1, 2), keepdims = True)

    def merge(self, r1, r2):
        self.bounding_boxes[r1] = self.bounding_boxes[r2] = bbox_merge(self.bounding_boxes[r1], self.bounding_boxes[r2])
        self.region_sizes[r1] = self.region_sizes[r2] = self.region_sizes[r1] + self.region_sizes[r2]
        self.color_histograms[r1] = self.color_histograms[r2] = (self.color_histograms[r1] * self.region_sizes[r1] + self.color_histograms[r2] * self.region_sizes[r2]) / (self.region_sizes[r1] + self.region_sizes[r2]) 
        self.texture_histograms[r1] = self.texture_histograms[r2] = (self.texture_histograms[r1] * self.region_sizes[r1] + self.texture_histograms[r2] * self.region_sizes[r2]) / (self.region_sizes[r1] + self.region_sizes[r2]) 

    def Size(self, r1, r2):
        return max(0.0, min(1.0, 1.0 - float(self.region_sizes[r1] + self.region_sizes[r2]) / float(self.img_size)))
    
    def Fill(self, r1, r2):
        return max(0.0, min(1.0, 1.0 - float(bbox_size(bbox_merge(self.bounding_boxes[r1], self.bounding_boxes[r2])) - self.region_sizes[r1] - self.region_sizes[r2]) / float(self.img_size)))
    
    def Color(self, r1, r2):
        return np.sum(np.minimum(self.color_histograms[r1], self.color_histograms[r2]))

    def Texture(self, r1, r2):
        return np.sum(np.minimum(self.texture_histograms[r1], self.texture_histograms[r2]))

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
