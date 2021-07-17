import argparse
import random
import dataclasses
import copy

import numpy as np
import cv2
import cv2.ximgproc.segmentation

@dataclasses.dataclass(order = True)
class Region:
    rank : float
    id : int
    level : int
    merged_to : int
    bbox : tuple

    def compute_rank(self):
        self.rank = self.level * random.random()

@dataclasses.dataclass(order = True)
class Edge:
    similarity : float
    fro : int
    to : int
    removed : bool

class RegionSimilarity:
    def __init__(self, features, strategies):
        self.merge = features.merge
        self.features = features
        self.strategies = strategies
        self.weights = [1.0 / len(strategies)] * len(strategies)

    def __call__(self, r1, r2):
        return sum(self.weights[i] * s(self.features, r1, r2) for i, s in enumerate(self.strategies)) / sum(self.weights)

def bbox_merge(xywh1, xywh2):
    boxPoints = lambda x, y, w, h: [(x, y), (x + w - 1, y), (x, y + h - 1), (x + w - 1, y + h - 1)]
    return cv2.boundingRect(np.array(boxPoints(*xywh1) + boxPoints(*xywh2)))

def bbox_wh(xywh):
    return (xywh[-2], xywh[-1])

def bbox_area(xywh):
    return xywh[-2] * xywh[-1]

def selectiveSearch(img_bgr, base_k = 150, inc_k = 150, sigma = 0.8, min_size = 100, fast = True):
    hsv, lab, gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV), cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab), cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    if fast:
        segmentations = [ cv2.ximgproc.segmentation.createGraphSegmentation(sigma, float(k), min_size) for k in range(base_k, 1 + base_k + inc_k * 2, inc_k) ]
        images = [hsv, lab]
        strategies = lambda features: [RegionSimilarity(features, [HandcraftedRegionFeatures.Fill, HandcraftedRegionFeatures.Texture, HandcraftedRegionFeatures.Size, HandcraftedRegionFeatures.Color]), RegionSimilarity(copy.deepcopy(features), [HandcraftedRegionFeatures.Fill, HandcraftedRegionFeatures.Texture, HandcraftedRegionFeatures.Size])]
    else:
        segmentations = [ cv2.ximgproc.segmentation.createGraphSegmentation(sigma, float(k), min_size) for k in range(base_k, 1 + base_k + inc_k * 4, inc_k) ]
        images = [hsv, lab, gray, hsv[..., 0], np.stack([img_bgr[..., 2], img_bgr[..., 1], gray], axis = -1)]
        strategies = lambda features: [RegionSimilarity(features, [HandcraftedRegionFeatures.Fill, HandcraftedRegionFeatures.Texture, HandcraftedRegionFeatures.Size, HandcraftedRegionFeatures.Color]), RegionSimilarity(copy.deepcopy(features), [HandcraftedRegionFeatures.Fill, HandcraftedRegionFeatures.Texture, HandcraftedRegionFeatures.Size]), RegionSimilarity(copy.deepcopy(features), [HandcraftedRegionFeatures.Fill]), RegionSimilarity(copy.deepcopy(features), [HandcraftedRegionFeatures.Size])]
            
    all_regions = []
    for image in images:
        for gs in segmentations:
            img_regions = gs.processImage(image)
            nb_segs = 1 + int(img_regions.max())

            graph_adj, region_areas = np.zeros((nb_segs, nb_segs), dtype = bool), np.zeros((nb_segs,), dtype = np.int64)
            
            points = [[] for i in range(nb_segs)]
            
            for i in range(img_regions.shape[0]):
                for j in range(img_regions.shape[1]): 
                    p, pr = img_regions[i], (img_regions[i - 1] if i > 0 else None)
                    points[p[j]].append((j, i))
                    region_areas[p[j]] += 1

                    if i > 0 and j > 0:
                        graph_adj[p [j - 1], p[j]] = graph_adj[p[j], p [j - 1]] = True
                        graph_adj[pr[j    ], p[j]] = graph_adj[p[j], pr[j    ]] = True
                        graph_adj[pr[j - 1], p[j]] = graph_adj[p[j], pr[j - 1]] = True

            bounding_rects = [cv2.boundingRect(np.array(pts)) for pts in points]
            features = HandcraftedRegionFeatures(image, img_regions, region_areas)
            
            all_regions.extend(region for strategy in strategies(features) for region in hierarchicalGrouping(strategy, graph_adj, region_areas, bounding_rects))
    
    return list({region.bbox : True for region in sorted(all_regions, key = lambda r: r.rank)}.keys())

def hierarchicalGrouping(s, graph_adj, region_areas, bounding_rects):
    region_areas = region_areas.copy()
    regions, PQ = [], []

    for i in range(len(bounding_rects)):
        regions.append(Region(id = i, level = 1, merged_to = -1, bbox = bounding_rects[i], rank = 0))
        for j in range(i + 1, len(bounding_rects)):
            if graph_adj[i, j]:
                PQ.append(Edge(fro = i, to = j, similarity = s(i, j), removed = False))
    
    while PQ:
        PQ.sort(key = lambda sim: sim.similarity)
        e = PQ.pop()
        u, v = e.fro, e.to
        
        reg_fro, reg_to = regions[u], regions[v]
        regions.append(Region(id = min(reg_fro.id, reg_to.id), level = 1 + max(reg_fro.level, reg_to.level), merged_to = -1, bbox = bbox_merge(reg_fro.bbox, reg_to.bbox), rank = 0))
        regions[u].merged_to = regions[v].merged_to = len(regions) - 1
        s.merge(reg_fro.id, reg_to.id)
        region_areas[reg_to.id] = region_areas[reg_fro.id] = region_areas[reg_fro.id] + region_areas[reg_to.id]
        
        local_neighbours = set()
        for e in PQ:
            if (u == e.fro or u == e.to) or (v == e.fro or v == e.to):
                local_neighbours.add(e.to if e.fro == u or e.fro == v else e.fro)
                e.removed = True
        PQ = [sim for sim in PQ if not sim.removed]
        for local_neighbour in local_neighbours:
            PQ.append(Edge(fro = len(regions) - 1, to = local_neighbour, similarity = s(regions[len(regions) - 1].id, regions[local_neighbour].id), removed = False))

    for region in regions:
        region.compute_rank()

    return regions

class HandcraftedRegionFeatures:
    def __init__(self, img, regions, region_areas, color_histogram_bins_size = 25, texture_histogram_bins_size = 10):
        img_height, img_width, img_channels = img.shape
        self.region_areas = region_areas
        self.img_area = img_height * img_width
        
        points = [[] for k in range(len(region_areas))]
        for i in range(regions.shape[0]):
            for j in range(regions.shape[1]): 
                points[regions[i, j]].append( (j, i) )
        self.bounding_rects = [cv2.boundingRect(np.array(pts)) for pts in points]
        
        self.color_histograms = np.zeros((len(region_areas), img_channels, color_histogram_bins_size), dtype = np.float32)
        for r in range(len(region_areas)):
            mask = (regions == r).astype(np.uint8) * 255
            for p in range(img_channels):
                self.color_histograms[r, p] = np.squeeze(cv2.calcHist([img[..., p]], [0], mask, [color_histogram_bins_size], [0, 256]))
        self.color_histograms /= np.sum(self.color_histograms, axis = (1, 2), keepdims = True)

        self.texture_histograms = np.zeros((len(region_areas), img_channels, 8 * texture_histogram_bins_size), dtype = np.float32)
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

        for i in range(regions.shape[0]):
            for j in range(regions.shape[1]):
                for p in range(img_channels):
                    for k in range(8):
                        val = int(img_gaussians[p * 8 + k][i, j])
                        bin = int(float(val) / (256.0 / texture_histogram_bins_size))
                        self.texture_histograms[regions[i, j], p, k * texture_histogram_bins_size + bin] += 1
        self.texture_histograms /= np.sum(self.texture_histograms, axis = (1, 2), keepdims = True)

    def merge(self, r1, r2):
        self.color_histograms[r1] = self.color_histograms[r2] = (self.color_histograms[r1] * self.region_areas[r1] + self.color_histograms[r2] * self.region_areas[r2]) / (self.region_areas[r1] + self.region_areas[r2]) 
        self.texture_histograms[r1] = self.texture_histograms[r2] = (self.texture_histograms[r1] * self.region_areas[r1] + self.texture_histograms[r2] * self.region_areas[r2]) / (self.region_areas[r1] + self.region_areas[r2]) 
        self.bounding_rects[r1] = self.bounding_rects[r2] = bbox_merge(self.bounding_rects[r1], self.bounding_rects[r2])

    def Size(self, r1, r2):
        return max(0.0, min(1.0, 1.0 - float(self.region_areas[r1] + self.region_areas[r2]) / float(self.img_area)))
    
    def Fill(self, r1, r2):
        return max(0.0, min(1.0, 1.0 - float(bbox_area(bbox_merge(self.bounding_rects[r1], self.bounding_rects[r2])) - self.region_areas[r1] - self.region_areas[r2]) / float(self.img_area)))
    
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
        boxes_xywh = selectiveSearch(img, fast = args.fast)
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
