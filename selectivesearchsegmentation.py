import math
import copy
import heapq
import random
import dataclasses

import cv2.ximgproc.segmentation

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

def rgb_to_hsv(image, eps: float = 1e-6):
    # https://github.com/kornia/kornia/blob/master/kornia/color/hsv.py
    maxc, _ = image.max(-3)
    maxc_mask = image == maxc.unsqueeze(-3)
    _, max_indices = ((maxc_mask.cumsum(-3) == 1) & maxc_mask).max(-3)
    minc = image.min(-3)[0]

    v = maxc  # brightness

    deltac = maxc - minc
    s = deltac / (v + eps)

    # avoid division by zero
    deltac = torch.where(deltac == 0, torch.ones_like(deltac, device=deltac.device, dtype=deltac.dtype), deltac)

    maxc_tmp = maxc.unsqueeze(-3) - image
    rc, gc, bc = maxc_tmp.unbind(dim = -3)

    h = torch.stack([bc - gc, 2.0 * deltac + rc - bc, 4.0 * deltac + gc - rc], dim=-3)

    h = torch.gather(h, dim=-3, index=max_indices[..., None, :, :])
    h = h.squeeze(-3)
    h = h / deltac

    h = (h / 6.0) % 1.0

    h = 2 * math.pi * h

    return torch.stack([h, s, v], dim=-3)


def rgb_to_lab(image):
    # https://github.com/kornia/kornia/blob/master/kornia/color/lab.py
    
    def rgb_to_xyz(image):
        # https://github.com/kornia/kornia/blob/master/kornia/color/xyz.py
        r, g, b = image.unbind(dim = -3)
        x = 0.412453 * r + 0.357580 * g + 0.180423 * b
        y = 0.212671 * r + 0.715160 * g + 0.072169 * b
        z = 0.019334 * r + 0.119193 * g + 0.950227 * b
        return torch.stack([x, y, z], -3)

    def rgb_to_linear_rgb(image):
        # https://github.com/kornia/kornia/blob/master/kornia/color/rgb.py
        return torch.where(image > 0.04045, torch.pow(((image + 0.055) / 1.055), 2.4), image / 12.92)

    # Convert from sRGB to Linear RGB
    lin_rgb = rgb_to_linear_rgb(image)

    xyz_im = rgb_to_xyz(lin_rgb)

    # normalize for D65 white point
    xyz_ref_white = torch.tensor([0.95047, 1.0, 1.08883], device=xyz_im.device, dtype=xyz_im.dtype)[..., :, None, None]
    xyz_normalized = torch.div(xyz_im, xyz_ref_white)

    threshold = 0.008856
    power = torch.pow(xyz_normalized.clamp(min=threshold), 1 / 3.0)
    scale = 7.787 * xyz_normalized + 4.0 / 29.0
    xyz_int = torch.where(xyz_normalized > threshold, power, scale)

    x, y, z = xyz_int.unbind(dim = -3)

    L = (116.0 * y) - 16.0
    a = 500.0 * (x - y)
    b = 200.0 * (y - z)

    return torch.stack([L, a, b], dim=-3)

def rgb_to_grayscale(image, rgb_weights = [0.299, 0.587, 0.114]):
    # https://github.com/kornia/kornia/blob/master/kornia/color/gray.py
    r, g, b = image.unsqueeze(-4).unbind(dim = -3)
    return rgb_weights[0] * r + rgb_weights[1] * g + rgb_weights[2] * b

def image_scharr_gradients(img : 'BCHW') -> 'BC2HW':
    flipped_scharr_x = torch.tensor([
        [-3, 0, 3 ],
        [10, 0, 10],
        [-3, 0, 3 ]
    ], dtype = img.dtype, device = img.device)

    kernel = torch.stack([flipped_scharr_x, flipped_scharr_x.t()]).unsqueeze(1)
    return F.conv2d(img.flatten(end_dim = -3).unsqueeze(1), kernel, padding = 1).unflatten(0, img.shape[:-2])

def bounding_rect(points):
    x1, y1 = min(x for x, y in points), min(y for x, y in points)
    x2, y2 = max(x for x, y in points), max(y for x, y in points)
    return [x1, y1, x2 - x1, y2 - y1]

def bbox_merge(xywh1, xywh2):
    boxPoints = lambda x, y, w, h: [(x, y), (x + w - 1, y), (x, y + h - 1), (x + w - 1, y + h - 1)]
    return bounding_rect(boxPoints(*xywh1) + boxPoints(*xywh2))

def bbox_size(xywh):
    return int(xywh[-2]) * int(xywh[-1])

def normalize_min_max(x, dim, eps = 0):
    hmin, hmax = x.amin(dim = dim, keepdim = True), x.amax(dim = dim, keepdim = True)
    return (x - hmin) / (eps + hmax - hmin) 

def rotated_xywh(img_height, img_width, angle = 45.0, scale = 1.0):
    # https://docs.opencv.org/4.5.3/da/d54/group__imgproc__transform.html#gafbbc470ce83812914a70abfb604f4326
    
    center = (img_width / 2.0, img_height / 2.0)
    alpha, beta = scale * math.cos(math.radians(angle)), scale * math.sin(math.radians(angle))
    rot = torch.tensor([
            [alpha, beta, (1 - alpha) * center[0] - beta * center[1]],
            [-beta, alpha, beta * center[0] + (1 - alpha) * center[1]]
        ])
    rotate = lambda point: (rot @ torch.tensor((point[0], point[1], 1.0), dtype = torch.float32)).tolist()[:2]

    points = [(0, 0), (img_width - 1, 0), (0, img_height - 1), (img_width - 1, img_height - 1)]
    
    return bounding_rect(list(map(rotate, points)))

def image_gaussian_derivatives(img):
    grads = image_scharr_gradients(img)
    
    img_height, img_width = img.shape[-2:]
    xywh = rotated_xywh(img_height, img_width, 45.0)
    startx, starty = int(max(0, (xywh[-2] - img_width) / 2)), int(max(0, (xywh[-1] - img_height) / 2))
    img_rotated = TF.rotate(img, 45.0, expand = True)
    grads_rotated = image_scharr_gradients(img_rotated)
    grads_rotated = TF.rotate(grads_rotated.flatten(end_dim = -3), -45.0, expand = True).unflatten(0, grads_rotated.shape[:-2])
    grads_rotated = grads_rotated[..., starty : starty + img_height, startx : startx + img_width]

    return torch.cat([grads.clamp(min = 0), grads.clamp(max = 0), grads_rotated.clamp(min = 0), grads_rotated.clamp(max = 0)], dim = -3)

class SelectiveSearch(nn.Module):
    @dataclasses.dataclass(order = True)
    class RegionNode:
        rank : float
        id : int
        level : int
        merged_to : int
        bbox : list

    def __init__(self, base_k = 150, inc_k = 150, sigma = 0.8, min_size = 100, fast = True):
        super().__init__()
        self.base_k = base_k
        self.inc_k = inc_k
        self.sigma = sigma
        self.min_size = min_size
        self.fast = fast

        if self.fast:
            self.segmentations = [cv2.ximgproc.segmentation.createGraphSegmentation(self.sigma, float(k), self.min_size) for k in range(self.base_k, 1 + self.base_k + self.inc_k * 2, self.inc_k)]
            self.strategies = [[HandcraftedRegionFeatures.Fill, HandcraftedRegionFeatures.Texture, HandcraftedRegionFeatures.Size, HandcraftedRegionFeatures.Color], [HandcraftedRegionFeatures.Fill, HandcraftedRegionFeatures.Texture, HandcraftedRegionFeatures.Size]]
            self.images = lambda rgb, hsv, lab, gray: torch.stack([hsv], dim = -4)
        
        else:
            self.segmentations = [cv2.ximgproc.segmentation.createGraphSegmentation(self.sigma, float(k), self.min_size) for k in range(self.base_k, 1 + self.base_k + self.inc_k * 4, self.inc_k)]
            self.strategies = [[HandcraftedRegionFeatures.Fill, HandcraftedRegionFeatures.Texture, HandcraftedRegionFeatures.Size, HandcraftedRegionFeatures.Color], [HandcraftedRegionFeatures.Fill, HandcraftedRegionFeatures.Texture, HandcraftedRegionFeatures.Size], [HandcraftedRegionFeatures.Fill], [HandcraftedRegionFeatures.Size]]
            self.images = lambda rgb, hsv, lab, gray: torch.stack([hsv, lab, gray.expand_as(hsv), hsv[..., :1, :, :].expand_as(hsv), torch.cat([rgb[..., :2, :, :],  gray], dim = -3)], dim = -4)

    def forward(self, img):
        hsv, lab, gray = rgb_to_hsv(img), rgb_to_lab(img), rgb_to_grayscale(img)
        hsv[..., 0, :, :] /= 2 * math.pi
        lab[..., 0, :, :] /= 100
        lab[..., 1:, :, :] /= 256
        lab[..., 1:, :, :] += 0.5
                
        imgs = self.images(img, hsv, lab, gray)
        
        reg_labs = torch.stack([torch.as_tensor(gs.processImage(img.movedim(-3, -1).numpy())) for img in imgs.flatten(end_dim = -4) for gs in self.segmentations]).unflatten(0, imgs.shape[:-3] + (len(self.segmentations),))
        
        features = HandcraftedRegionFeatures(img, reg_labs)
        
        img_idx = torch.arange(len(reg_labs))[:, None, None].expand(reg_labs.shape[:-2]).flatten().tolist()
        
        all_regs = [[] for i in range(len(img))]
        for b, reg_lab in zip(img_idx, reg_labs.flatten(end_dim = -3)):
            graph_adj = self.build_segment_graph(reg_lab)
            all_regs[b].extend(reg for strategy in self.strategies for reg in self.hierarchical_grouping(copy.deepcopy(features), strategy, graph_adj))
        
        return [list({reg.bbox : True for reg in sorted(regs, key = lambda r: r.rank)}.keys()) for regs in all_regs]
    
    @staticmethod
    def build_segment_graph(reg_lab):
        num_segments = 1 + int(reg_lab.max())
        graph_adj = torch.zeros((num_segments, num_segments), dtype = torch.bool)
        for i in range(reg_lab.shape[0]):
            p, pr = reg_lab[i], (reg_lab[i - 1] if i > 0 else None)
            for j in range(reg_lab.shape[1]): 
                if i > 0 and j > 0:
                    graph_adj[p [j - 1], p[j]] = graph_adj[p[j], p [j - 1]] = True
                    graph_adj[pr[j    ], p[j]] = graph_adj[p[j], pr[j    ]] = True
                    graph_adj[pr[j - 1], p[j]] = graph_adj[p[j], pr[j - 1]] = True
        return graph_adj

    @staticmethod
    def hierarchical_grouping(features, strategy, graph_adj, compute_rank = lambda region: region.level * random.random()):
        # https://github.com/opencv/opencv_contrib/blob/master/modules/ximgproc/src/selectivesearchsegmentation.cpp
        regs, PQ = [], []

        for i in range(len(bbox)):
            regs.append(SelectiveSearch.RegionNode(id = i, level = 1, merged_to = -1, bbox = features.xywh[i].tolist(), rank = 0))
            for j in range(i + 1, len(bbox)):
                if graph_adj[i, j]:
                    PQ.append((-float(features.compute_region_similarity(i, j, strategy)), i, j))
        
        visited = set()
        PQ = heapq.heapify(PQ)
        while PQ:
            negsimilarity, u, v = heapq.heappop(PQ)
            if u in visited or v in visited:
                continue
            
            reg_fro, reg_to = regs[u], regs[v]
            assert reg_fro.id == u and reg_to.id == v
            w = len(regs)
            regs[u].merged_to = regs[v].merged_to = w
            regs.append(SelectiveSearch.RegionNode(id = min(reg_fro.id, reg_to.id), level = 1 + max(reg_fro.level, reg_to.level), merged_to = -1, bbox = bbox_merge(reg_fro.bbox, reg_to.bbox), rank = 0))
            features.merge_regions(reg_fro.id, reg_to.id)
            
            visited.update([u, v])
            
            for e in PQ:
                if (u == e.fro or u == e.to) or (v == e.fro or v == e.to):
                    vv = e.to if e.fro == u or e.fro == v else e.fro
                    if vv not in visited:
                        heapq.heappush(PQ, (-float(features.compute_region_similarity(regs[w].id, regs[vv].id, strategy)), w, vv))

        for region in regs:
            region.rank = compute_rank(region)

        return regs

class HandcraftedRegionFeatures:
    def __init__(self, img : 'B3HW', reg_lab : 'BIGHW', color_histogram_bins = 25, texture_histogram_bins = 10):
        ones_like_expand = lambda tensor: torch.ones(1, device = tensor.device, dtype = torch.float32).expand_as(tensor)
        
        img_channels, img_height, img_width = img.shape[-3:]
        self.img_size = img_height * img_width

        img_normalized_gaussian_derivatives = normalize_min_max(image_gaussian_derivatives(img), dim = (-2, -1))
        num_segments = 1 + int(reg_lab.max())
        
        Z = reg_lab.flatten(start_dim = -2).to(torch.int64)
        self.region_sizes = torch.zeros((reg_lab.shape[0], reg_lab.shape[1], reg_lab.shape[2], num_segments), dtype = torch.float32).scatter_add_(-1, Z, ones_like_expand(Z))
        
        Z = (reg_lab[..., None, :, :] * color_histogram_bins + img[:, None, None].mul(color_histogram_bins - 1)).flatten(start_dim = -2).to(torch.int64)
        self.color_histograms = torch.zeros((reg_lab.shape[0], reg_lab.shape[1], reg_lab.shape[2], img_channels, num_segments * color_histogram_bins), dtype = torch.float32).scatter_add_(-1, Z, ones_like_expand(Z)).unflatten(-1, (num_segments, color_histogram_bins)).movedim(-2, -3)
        self.color_histograms /= self.color_histograms.sum(dim = (-2, -1), keepdim = True)

        Z = (reg_lab[..., None, None, :, :] * texture_histogram_bins + img_normalized_gaussian_derivatives[:, None, None].mul(texture_histogram_bins - 1)).flatten(start_dim = -2).to(torch.int64)
        self.texture_histograms = torch.zeros((reg_lab.shape[0], reg_lab.shape[1], reg_lab.shape[2], img_channels, 8, num_segments * texture_histogram_bins), dtype = torch.float32).scatter_add_(-1, Z, ones_like_expand(Z)).unflatten(-1, (num_segments, texture_histogram_bins)).movedim(-2, -4)
        self.texture_histograms /= self.texture_histograms.sum(dim = (-3, -2, -1), keepdim = True)
        
        self.xywh = torch.tensor([[img_width, img_height, 0, 0]], dtype = torch.int64).repeat(reg_lab.shape[0], reg_lab.shape[1], reg_lab.shape[2], num_segments, 1)
        for b in range(reg_lab.shape[0]):
            for i in range(reg_lab.shape[1]):
                for g in range(reg_lab.shape[2]):
                    for y in range(reg_lab.shape[3]):
                        for x in range(reg_lab.shape[4]):
                            xywh = self.xywh[b, i, g, reg_lab[y, x]]
                            xywh[0].clamp_(max = x)
                            xywh[1].clamp_(max = y)
                            xywh[2].clamp_(min = x)
                            xywh[3].clamp_(min = y)
        self.xywh[..., 2] -= self.xywh[..., 0]
        self.xywh[..., 3] -= self.xywh[..., 1]

    def Fill(self, r1, r2):
        return max(0.0, min(1.0, 1.0 - float(bbox_size(bbox_merge(self.xywh[r1], self.xywh[r2])) - self.region_sizes[r1] - self.region_sizes[r2]) / float(self.img_size)))

    def Size(self, r1, r2):
        return max(0.0, min(1.0, 1.0 - float(self.region_sizes[r1] + self.region_sizes[r2]) / float(self.img_size)))
    
    def Color(self, r1, r2):
        return torch.min(self.color_histograms[r1], self.color_histograms[r2]).sum(dim = (-2, -1))

    def Texture(self, r1, r2):
        return torch.min(self.texture_histograms[r1], self.texture_histograms[r2]).sum(dim = (-3, -2, -1))

    def compute_region_similarity(self, r1, r2, strategies):
        return sum(s(self, r1, r2) for s in strategies) / len(strategies)
    
    def merge_regions(self, r1, r2):
        self.xywh[r1] = self.xywh[r2] = torch.tensor(bbox_merge(self.xywh[r1].tolist(), self.xywh[r2].tolist()))
        self.region_sizes[r1] = self.region_sizes[r2] = self.region_sizes[r1] + self.region_sizes[r2]
        self.color_histograms[r1] = self.color_histograms[r2] = (self.color_histograms[r1] * self.region_sizes[r1] + self.color_histograms[r2] * self.region_sizes[r2]) / (self.region_sizes[r1] + self.region_sizes[r2]) 
        self.texture_histograms[r1] = self.texture_histograms[r2] = (self.texture_histograms[r1] * self.region_sizes[r1] + self.texture_histograms[r2] * self.region_sizes[r2]) / (self.region_sizes[r1] + self.region_sizes[r2]) 

if __name__ == '__main__':
    import argparse
    import cv2

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
        algo = SelectiveSearch(fast = args.fast)
        boxes_xywh = algo(torch.as_tensor(img)[..., [2, 1, 0]].movedim(-1, -3).unsqueeze(0) / 255.0)[0]
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
