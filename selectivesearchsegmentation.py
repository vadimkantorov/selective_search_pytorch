import cProfile

def profileit(func):
    def wrapper(*args, **kwargs):
        datafn = func.__name__ + ".profile" # Name the data file sensibly
        prof = cProfile.Profile()
        retval = prof.runcall(func, *args, **kwargs)
        prof.dump_stats(datafn)
        return retval

    return wrapper


import time
import math
import copy
import heapq
import random
import itertools
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

def bbox(points):
    x1, y1 = min(x for x, y in points), min(y for x, y in points)
    x2, y2 = max(x for x, y in points), max(y for x, y in points)
    return (x1, y1, x2 - x1 + 1, y2 - y1 + 1)

def bbox_merge(xywh1, xywh2):
    boxPoints = lambda x, y, w, h: [(x, y), (x + w - 1, y), (x, y + h - 1), (x + w - 1, y + h - 1)]
    return bbox(boxPoints(*xywh1) + boxPoints(*xywh2))

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
    
    return bbox(list(map(rotate, points)))

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

def normalize_min_max(x, dim, eps = 0):
    hmin, hmax = x.amin(dim = dim, keepdim = True), x.amax(dim = dim, keepdim = True)
    return (x - hmin) / (eps + hmax - hmin) 

class SelectiveSearch(nn.Module):
    # https://github.com/opencv/opencv_contrib/blob/master/modules/ximgproc/src/selectivesearchsegmentation.cpp
        
    def __init__(self, base_k = 150, inc_k = 150, sigma = 0.8, min_size = 100, preset = 'fast'):
        super().__init__()
        self.base_k = base_k
        self.inc_k = inc_k
        self.sigma = sigma
        self.min_size = min_size
        self.preset = preset
        
        if self.preset == 'single':
            #base_k = 200
            self.images = lambda rgb, hsv, lab, gray: torch.stack([hsv], dim = -4)
            self.segmentations = [cv2.ximgproc.segmentation.createGraphSegmentation(self.sigma, float(base_k), self.min_size)]
            self.strategies = torch.tensor([
                [0.25, 0.25, 0.25, 0.25],
            ])
            
        elif self.preset == 'fast':
            self.images = lambda rgb, hsv, lab, gray: torch.stack([hsv], dim = -4)
            self.segmentations = [cv2.ximgproc.segmentation.createGraphSegmentation(self.sigma, float(k), self.min_size) for k in range(self.base_k, 1 + self.base_k + self.inc_k * 2, self.inc_k)]
            self.strategies = torch.tensor([
                [0.25, 0.25, 0.25, 0.25],
                [0.33, 0.33, 0.33, 0.00]
            ])
        
        elif self.preset == 'quality':
            self.images = lambda rgb, hsv, lab, gray: torch.stack([hsv, lab, gray.expand_as(hsv), hsv[..., :1, :, :].expand_as(hsv), torch.cat([rgb[..., :2, :, :],  gray], dim = -3)], dim = -4)
            self.segmentations = [cv2.ximgproc.segmentation.createGraphSegmentation(self.sigma, float(k), self.min_size) for k in range(self.base_k, 1 + self.base_k + self.inc_k * 4, self.inc_k)]
            self.strategies = torch.tensor([
                [0.25, 0.25, 0.25, 0.25],
                [0.33, 0.33, 0.33, 0.00],
                [1.00, 0.00, 0.00, 0.00],
                [0.00, 0.00, 1.00, 0.00]
            ])

    @staticmethod
    def get_region_mask(reg_lab, region):
        return (reg_lab[region.plane_id][..., None] == torch.tensor(list(region.ids), device = reg_lab.device, dtype = reg_lab.dtype)).any(dim = -1)

    def forward(self, img):
        tic = time.time()
        hsv, lab, gray = rgb_to_hsv(img), rgb_to_lab(img), rgb_to_grayscale(img)
        hsv[..., 0, :, :] /= 2.0 * math.pi 
        hsv *= 255.0
        #lab[..., 0, :, :] *= 255.0/100.0
        #gray *= 255.0
        #img *= 255.0
        
        imgs = self.images(img, hsv, lab, gray)
        
        print('images', time.time() - tic); tic = time.time()

        reg_lab = torch.stack([torch.as_tensor(gs.processImage(img.movedim(-3, -1).numpy())) for img in imgs.flatten(end_dim = -4) for gs in self.segmentations]).unflatten(0, imgs.shape[:-3] + (len(self.segmentations),))
        
        print('graph segmentation', time.time() - tic); tic = time.time()
        
        features = HandcraftedRegionFeatures(imgs, reg_lab)
        
        affinity0 = features.compute_region_affinity()
        
        print('features', time.time() - tic); tic = time.time()
        
        graphadj = self.build_graph(reg_lab, max_num_segments = features.max_num_segments)

        graphadj0 = (graphadj[..., None, None] * affinity0.unsqueeze(-2) * self.strategies).sum(dim = -1)

        print('build graph', time.time() - tic); tic = time.time()
        
        all_regs = [reg for strategy, graphadj in zip(self.strategies, graphadj0.unbind(-1)) for reg in self.hierarchical_grouping(copy.deepcopy(features), graphadj, strategy.tolist()) if reg.level >= 1]
        
        print('grouping', time.time() - tic); tic = time.time()
        
        key_img_id, key_rank = (lambda reg: reg.img_id), (lambda reg: reg.rank)
        by_image = {k: sorted(list(g), key = key_rank) for k, g in itertools.groupby(sorted(all_regs, key = key_img_id), key = key_img_id)}
        
        without_duplicates = [{reg.bbox : i for i, reg in enumerate(by_image.get(b, []))} for b in range(len(img))]
        return [list(without_duplicates[b].keys()) for b in range(len(img))], [[by_image[b][i] for i in without_duplicates[b].values()] for b in range(len(img))], reg_lab
    
    @staticmethod
    def build_graph(reg_lab : 'BIGHW', max_num_segments : int):
        I = torch.stack(torch.meshgrid(*[torch.arange(s) for s in reg_lab.shape[:3]]), dim = -1)[..., None, None]
        DX = torch.stack([reg_lab[..., :, :-1], reg_lab[..., :, 1:]], dim = -1).movedim(-1, -3)
        DY = torch.stack([reg_lab[..., :-1, :], reg_lab[..., 1:, :]], dim = -1).movedim(-1, -3)
        dx = torch.cat([I.expand(-1, -1, -1, -1, *DX.shape[-2:]), DX], dim = -3).flatten(start_dim = -2)
        dy = torch.cat([I.expand(-1, -1, -1, -1, *DY.shape[-2:]), DY], dim = -3).flatten(start_dim = -2)
        i = torch.cat([dx, dy], dim = -1).movedim(-2, 0).flatten(start_dim = 1)
        v = torch.ones(i.shape[1], dtype = torch.bool)
        A = torch.sparse_coo_tensor(i, v, reg_lab.shape[:-2] + (max_num_segments, max_num_segments), dtype = torch.int64)
        A += A.transpose(-1, -2)
        return A.coalesce().to(torch.bool).to_dense().triu(diagonal = 1)

    #@profileit
    def hierarchical_grouping(self, features, graphadj, strategy, compute_rank = lambda region: region.level * random.random()):
        @dataclasses.dataclass(order = True)
        class RegionNode: rank : float; id : int; level : int; parent_id : int; bbox : tuple; plane_id: tuple; img_id: int; ids : set;
        
        tic = time.time()
        
        ga = graphadj.to_sparse()
        b, i, g, r1, r2 = ga.indices()
        v = ga.values()
        r0 = b * i * g * features.max_num_segments
        good = ~v.isnan()
        PQ = list(zip(v.neg()[good].tolist(), (r0 + r1)[good].tolist(), (r0 + r2)[good].tolist()))
        regs = [RegionNode(id = r1, img_id = b, plane_id = (b, i, g), level = 1 if r1 < int(features.num_segments[b, i, g]) else 0, bbox = tuple(features.xywh[b, i, g, r1].tolist()), ids = {r1}, parent_id = -1, rank = 0) for b in range(graphadj.shape[0]) for i in range(graphadj.shape[1]) for g in range(graphadj.shape[2]) for r1 in range(graphadj.shape[3])]
        
        print('init', time.time() - tic); tic = time.time()
        visited = set()
        heapq.heapify(PQ)
        while PQ:
            negsim, u, v = heapq.heappop(PQ)
            if u in visited or v in visited:
                continue
            
            ww = len(regs)
            reg_fro, reg_to = regs[u], regs[v]
            b, i, g = reg_fro.plane_id
            regs[u].parent_id = regs[v].parent_id = ww
            regs.append(RegionNode(id = min(reg_fro.id, reg_to.id), img_id = reg_fro.img_id, plane_id = reg_fro.plane_id, level = 1 + max(reg_fro.level, reg_to.level), bbox = bbox_merge(reg_fro.bbox, reg_to.bbox), ids = reg_fro.ids | reg_to.ids, parent_id = -1, rank = 0))
            features.merge_regions((b, i, g, reg_fro.id), (b, i, g, reg_to.id))
            
            visited.update([u, v])
            
            for _, fro, to in PQ:
                if (u == fro or u == to) or (v == fro or v == to):
                    vv = to if fro == u or fro == v else fro
                    if vv not in visited:
                        heapq.heappush(PQ, (-features.compute_region_affinity((b, i, g, regs[ww].id), (b, i, g, regs[vv].id), *strategy), ww, vv))
        
        print('loop', time.time() - tic); tic = time.time()

        for region in regs:
            region.rank = compute_rank(region)

        return regs

class HandcraftedRegionFeatures:
    def __init__(self, img : 'BI3HW', reg_lab : 'BIGHW', color_histogram_bins = 25, texture_histogram_bins = 10, neginf = -int(1e9), posinf = int(1e9)):
        ones_like_expand = lambda tensor: torch.ones(1, device = tensor.device, dtype = torch.float32).expand_as(tensor)
        
        img_channels, img_height, img_width = img.shape[-3:]
        self.img_size = img_height * img_width
        self.num_segments = 1 + reg_lab.amax(dim = (-2, -1))
        self.max_num_segments = int(self.num_segments.amax())

        img_normalized_gaussian_derivatives = normalize_min_max(image_gaussian_derivatives(img.flatten(end_dim = -4)).unflatten(0, img.shape[:-3]), dim = (-2, -1))
        
        Z = reg_lab.flatten(start_dim = -2).to(torch.int64)
        self.region_sizes = torch.zeros(reg_lab.shape[:-2] + (self.max_num_segments, ), dtype = torch.float32).scatter_add_(-1, Z, ones_like_expand(Z))
        
        Z = (reg_lab[..., None, :, :] * color_histogram_bins + img[:, :, None].mul((color_histogram_bins - 1) / 255.0)).flatten(start_dim = -2).to(torch.int64)
        self.color_histograms = torch.zeros(reg_lab.shape[:-2] + (img_channels, self.max_num_segments * color_histogram_bins), dtype = torch.float32).scatter_add_(-1, Z, ones_like_expand(Z)).unflatten(-1, (self.max_num_segments, color_histogram_bins)).movedim(-2, -3).contiguous()
        self.color_histograms /= self.color_histograms.sum(dim = (-2, -1), keepdim = True)

        Z = (reg_lab[..., None, None, :, :] * texture_histogram_bins + img_normalized_gaussian_derivatives[:, None].mul(texture_histogram_bins - 1)).flatten(start_dim = -2).to(torch.int64)
        self.texture_histograms = torch.zeros(reg_lab.shape[:-2] + (img_channels, img_normalized_gaussian_derivatives.shape[-3], self.max_num_segments * texture_histogram_bins), dtype = torch.float32).scatter_add_(-1, Z, ones_like_expand(Z)).unflatten(-1, (self.max_num_segments, texture_histogram_bins)).movedim(-2, -4).contiguous()
        self.texture_histograms /= self.texture_histograms.sum(dim = (-3, -2, -1), keepdim = True)
        
        yx = torch.stack(torch.meshgrid(torch.arange(reg_lab.shape[-2]), torch.arange(reg_lab.shape[-1])))
        mask = (reg_lab.unsqueeze(-1) == torch.arange(self.max_num_segments)).movedim(-1, -3).unsqueeze(-3)
        masked_min, masked_max = torch.where(mask, yx, posinf), torch.where(mask, yx, neginf)
        (ymin, xmin), (ymax, xmax) = masked_min.amin(dim = (-2, -1)).unbind(-1), masked_max.amax(dim = (-2, -1)).unbind(-1)
        self.xywh = torch.stack([xmin, ymin, xmax - xmin + 1, ymax - ymin + 1], dim = -1)

    def compute_region_affinity(self, r1 = None, r2 = None, fill = 0, texture = 0, size = 0, color = 0):
        bbox_size_ = lambda xywh: xywh[..., 2] * xywh[..., 3]
        bbox_size = lambda xywh: int(xywh[-2]) * int(xywh[-1])

        def bbox_merge_(xywh1, xywh2):
            xmin, ymin = torch.min(xywh1[..., 0], xywh2[..., 0]), torch.min(xywh1[..., 1], xywh2[..., 1])
            xmax, ymax = torch.max(xywh1[..., 0] + xywh1[..., 2] - 1, xywh2[..., 0] + xywh2[..., 2] - 1), torch.max(xywh1[..., 1] + xywh1[..., 3] - 1, xywh2[..., 1] + xywh2[..., 3] - 1)
            return torch.stack([xmin, ymin, xmax - xmin + 1, ymax - ymin + 1], dim = -1)
        
        if r1 is None and r2 is None:
            color_affinity = torch.min(self.color_histograms.unsqueeze(4), self.color_histograms.unsqueeze(3)).sum(dim = (-2, -1))
            texture_affinity = torch.min(self.texture_histograms.unsqueeze(4), self.texture_histograms.unsqueeze(3)).sum(dim = (-3, -2, -1))
            size_affinity = (1 - (self.region_sizes.unsqueeze(4) + self.region_sizes.unsqueeze(3)).div(self.img_size)).clamp(min = 0, max = 1)
            fill_affinity = (1 - (bbox_size_(bbox_merge_(self.xywh.unsqueeze(4), self.xywh.unsqueeze(3))) - self.region_sizes.unsqueeze(4) - self.region_sizes.unsqueeze(3)) / self.img_size).clamp(min = 0, max = 1)
            
            return torch.stack([fill_affinity, texture_affinity, size_affinity, color_affinity], dim = -1)
        else:
            res = 0.0
            if fill > 0:
                res += max(0.0, min(1.0, 1.0 - float(bbox_size(bbox_merge(self.xywh[r1].tolist(), self.xywh[r2].tolist())) - self.region_sizes[r1] - self.region_sizes[r2]) / float(self.img_size)))

            if size > 0:
                res += max(0.0, min(1.0, 1.0 - float(self.region_sizes[r1] + self.region_sizes[r2]) / float(self.img_size)))

            if color > 0:
                res += float(torch.min(self.color_histograms[r1], self.color_histograms[r2]).sum(dim = (-2, -1)))

            if texture > 0:
                res += float(torch.min(self.texture_histograms[r1], self.texture_histograms[r2]).sum(dim = (-3, -2, -1)))

            return res
    
    def merge_regions(self, r1, r2):
        self.xywh[r1] = self.xywh[r2] = torch.tensor(bbox_merge(self.xywh[r1].tolist(), self.xywh[r2].tolist()))
        self.region_sizes[r1] = self.region_sizes[r2] = self.region_sizes[r1] + self.region_sizes[r2]
        self.color_histograms[r1] = self.color_histograms[r2] = (self.color_histograms[r1] * self.region_sizes[r1] + self.color_histograms[r2] * self.region_sizes[r2]) / (self.region_sizes[r1] + self.region_sizes[r2]) 
        self.texture_histograms[r1] = self.texture_histograms[r2] = (self.texture_histograms[r1] * self.region_sizes[r1] + self.texture_histograms[r2] * self.region_sizes[r2]) / (self.region_sizes[r1] + self.region_sizes[r2]) 

if __name__ == '__main__':
    import argparse
    import colorsys
    import cv2
    import matplotlib, matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', '-i')
    parser.add_argument('--output-path', '-o')
    parser.add_argument('--topk', type = int, default = 30)
    parser.add_argument('--preset', choices = ['fast', 'quality', 'single'], default = 'fast')
    parser.add_argument('--opencv', action = 'store_true')
    parser.add_argument('--seed', type = int, default = 42)
    parser.add_argument('--mask', type = int, default = 0)
    args = parser.parse_args()
    args.fast = True

    random.seed(args.seed)

    img = plt.imread(args.input_path).copy()

    if not args.opencv:
        algo = SelectiveSearch(preset = args.preset)
        boxes_xywh, regions, reg_lab = algo(torch.as_tensor(img).movedim(-1, -3).unsqueeze(0) / 255.0)
        mask = algo.get_region_mask(reg_lab, regions[0][args.mask])
        boxes_xywh, regions, reg_lab = boxes_xywh[0], regions[0], reg_lab[0]
    else:
        algo = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        algo.setBaseImage(img[..., [2, 1, 0]])
        dict(fast = algo.switchToSelectiveSearchFast, quality = algo.switchToSelectiveSearchQuality, single = algo.switchToSingleStrategy)[args.preset]()
        boxes_xywh = algo.process()
        x, y, w, h = boxes_xywh[args.mask]
        mask = torch.zeros(img.shape[:-1])
        mask[y : y + h, x : x + w] = 1

    print('boxes', len(boxes_xywh))
    print('height:', img.shape[0], 'width:', img.shape[1])
    print('ymax', max(y + h - 1 for x, y, w, h in boxes_xywh), 'xmax', max(x + w - 1 for x, y, w, h in boxes_xywh))

    #N = 1 + int(reg_lab.max())
    #colors = torch.tensor(list(map(lambda c: colorsys.hsv_to_rgb(*c), [(i / N, 1, 1) for i in range(N)])))
    #plt.imshow(colors[reg_lab[0, 0, 0].to(torch.int64)])
    
    plt.figure()
    plt.subplot(121)
    plt.imshow(img * mask[..., None].numpy() + img * (1 - mask[..., None].numpy()) // 5)
    plt.axis('off')
    for x, y, w, h in boxes_xywh[:args.topk]:
        plt.gca().add_patch(matplotlib.patches.Rectangle((x, y), w, h, linewidth = 1, edgecolor = 'r', facecolor = 'none'))
    plt.subplot(122)
    plt.imshow(mask.to(torch.float32))
    plt.savefig(args.output_path)
    plt.close()
