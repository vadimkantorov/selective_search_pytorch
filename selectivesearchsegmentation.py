import cv2.ximgproc.segmentation

import time
import math
import heapq
import itertools

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

def _rgb_to_hsv(image: torch.Tensor) -> torch.Tensor:
# https://github.com/pytorch/vision/blob/11e49de410ec84ec669293a91dfaa13a53c9bc47/torchvision/transforms/v2/functional/_color.py#L272
    r, g, _ = image.unbind(dim=-3)

    # Implementation is based on
    # https://github.com/python-pillow/Pillow/blob/4174d4267616897df3746d315d5a2d0f82c656ee/src/libImaging/Convert.c#L330
    minc, maxc = torch.aminmax(image, dim=-3)

    # The algorithm erases S and H channel where `maxc = minc`. This avoids NaN
    # from happening in the results, because
    #   + S channel has division by `maxc`, which is zero only if `maxc = minc`
    #   + H channel has division by `(maxc - minc)`.
    #
    # Instead of overwriting NaN afterwards, we just prevent it from occurring so
    # we don't need to deal with it in case we save the NaN in a buffer in
    # backprop, if it is ever supported, but it doesn't hurt to do so.
    eqc = maxc == minc

    channels_range = maxc - minc
    # Since `eqc => channels_range = 0`, replacing denominator with 1 when `eqc` is fine.
    ones = torch.ones_like(maxc)
    s = channels_range / torch.where(eqc, ones, maxc)
    # Note that `eqc => maxc = minc = r = g = b`. So the following calculation
    # of `h` would reduce to `bc - gc + 2 + rc - bc + 4 + rc - bc = 6` so it
    # would not matter what values `rc`, `gc`, and `bc` have here, and thus
    # replacing denominator with 1 when `eqc` is fine.
    channels_range_divisor = torch.where(eqc, ones, channels_range).unsqueeze_(dim=-3)
    rc, gc, bc = ((maxc.unsqueeze(dim=-3) - image) / channels_range_divisor).unbind(dim=-3)

    mask_maxc_neq_r = maxc != r
    mask_maxc_eq_g = maxc == g

    hg = rc.add(2.0).sub_(bc).mul_(mask_maxc_eq_g & mask_maxc_neq_r)
    hr = bc.sub_(gc).mul_(~mask_maxc_neq_r)
    hb = gc.add_(4.0).sub_(rc).mul_(mask_maxc_neq_r.logical_and_(mask_maxc_eq_g.logical_not_()))

    h = hr.add_(hg).add_(hb)
    h = h.mul_(1.0 / 6.0).add_(1.0).fmod_(1.0)
    return torch.stack((h, s, maxc), dim=-3)

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

    # Convert begin sRGB to Linear RGB
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
        [-3,  0, 3 ],
        [-10, 0, 10],
        [-3,  0, 3 ]
    ], dtype = img.dtype, device = img.device)
    kernel = torch.stack([flipped_scharr_x, flipped_scharr_x.t()]).unsqueeze(1)
    return F.conv2d(img.flatten(end_dim = -3).unsqueeze(1), kernel, padding = 1).unflatten(0, img.shape[:-2])

def image_gaussian_grads(img : 'BCHW'):
    grads = image_scharr_gradients(img)
    
    img_height, img_width = img.shape[-2:]
    xywh = rotated_xywh(img_height, img_width, 45.0)
    startx, starty = int(max(0, (xywh[-2] - img_width) / 2)), int(max(0, (xywh[-1] - img_height) / 2))
    img_rotated = TF.rotate(img, 45.0, expand = True)
    grads_rotated = image_scharr_gradients(img_rotated)
    grads_rotated = TF.rotate(grads_rotated.flatten(end_dim = -3), -45.0, expand = True).unflatten(0, grads_rotated.shape[:-2])
    grads_rotated = grads_rotated[..., starty : starty + img_height, startx : startx + img_width]

    return torch.cat([grads.clamp(min = 0), grads.clamp(max = 0), grads_rotated.clamp(min = 0), grads_rotated.clamp(max = 0)], dim = -3)

def normalize_min_max_(x, dim, eps = 1e-12):
    # https://github.com/pytorch/pytorch/issues/61582
    amin, amax = x.amin(dim = dim, keepdim = True), x.amax(dim = dim, keepdim = True)
    return x.sub_(amin).div_(amax.sub_(amin).add_(eps)) 
        
def expand_ones_like(tensor: torch.Tensor, dtype = torch.float32) -> torch.Tensor:
    # needed to work around https://github.com/pytorch/pytorch/issues/5405
    return torch.ones(1, device = tensor.device, dtype = dtype).expand_as(tensor)

def rotated_xywh(img_height : int, img_width : int, angle = 45.0, scale = 1.0):
    # https://docs.opencv.org/4.5.3/da/d54/group__imgproc__transform.html#gafbbc470ce83812914a70abfb604f4326
    center = (img_width / 2.0, img_height / 2.0)
    alpha, beta = scale * math.cos(math.radians(angle)), scale * math.sin(math.radians(angle))
    rot = torch.tensor([
            [alpha, beta, (1 - alpha) * center[0] - beta * center[1]],
            [-beta, alpha, beta * center[0] + (1 - alpha) * center[1]]
        ])
    rotate = lambda point: (rot @ torch.tensor((point[0], point[1], 1.0), dtype = torch.float32)).tolist()[:2]

    points = list(map(rotate, [(0, 0), (img_width - 1, 0), (0, img_height - 1), (img_width - 1, img_height - 1)]))

    x1, y1 = min(x for x, y in points), min(y for x, y in points)
    x2, y2 = max(x for x, y in points), max(y for x, y in points)
    xywh = (x1, y1, x2 - x1 + 1, y2 - y1 + 1)

    return xywh

def bbox_merge_tensor(xywh1 : torch.Tensor, xywh2 : torch.Tensor, dtype = None, return_area = False) -> torch.Tensor:
    assert return_area

    xy1, wh1 = xywh1.movedim(-1, 0).split(2)
    xy2, wh2 = xywh2.movedim(-1, 0).split(2)

    xymax1 = xy1.add(wh1).sub_(1) 
    xymax2 = xy2.add(wh2).sub_(1)
    
    xymin = torch.min(xy1, xy2)
    xymax = torch.max(xymax1, xymax2)
    # work around for https://github.com/pytorch/pytorch/issues/54216
    w, h = xymax.sub_(xymin).add_(1)
    return torch.mul(w.to(dtype), h.to(dtype))

def bbox_merge_tuple(xywh1 : tuple, xywh2 : tuple, return_area = False):
    xmin, ymin = min(xywh1[0], xywh2[0]), min(xywh1[1], xywh2[1])
    xmax, ymax = max(xywh1[0] + xywh1[2] - 1, xywh2[0] + xywh2[2] - 1), max(xywh1[1] + xywh1[3] - 1, xywh2[1] + xywh2[3] - 1)
    x, y, w, h = (xmin, ymin, xmax - xmin + 1, ymax - ymin + 1)
    return (w * h) if return_area else (x, y, w, h)

def bbox_per_segment(reg_lab_int64 : 'BHW', max_num_segments : int, dtype = torch.int16):
    img_height, img_width = reg_lab_int64.shape[-2:]
    # HxW, HxW
    y, x = torch.meshgrid(torch.arange(img_height, dtype = dtype, device = reg_lab_int64.device), torch.arange(img_width, dtype = dtype, device = reg_lab_int64.device), indexing = 'ij')
    y, x = y.expand_as(reg_lab_int64).flatten(start_dim = -2), x.expand_as(reg_lab_int64).flatten(start_dim = -2)
    
    reg_lab_int64 = reg_lab_int64.flatten(start_dim = -2)
    # BxS, BxS, BxS, BxS
    xywh = torch.zeros(4, *reg_lab_int64.shape[:-1], max_num_segments, dtype = dtype, device = reg_lab_int64.device)
    xywh[0].fill_(img_width).scatter_reduce_(-1, reg_lab_int64, x, reduce = 'amin')
    xywh[1].fill_(img_height).scatter_reduce_(-1, reg_lab_int64, y, reduce = 'amin')
    xywh[2].scatter_reduce_(-1, reg_lab_int64, x, reduce = 'amax')
    xywh[3].scatter_reduce_(-1, reg_lab_int64, y, reduce = 'amax')
    xywh[2:] -= xywh[:2]
    
    # BxSx4
    return xywh.movedim(0, -1)

def area_per_segment(reg_lab_int64 : 'BHW', max_num_segments : int, dtype = torch.int32):
    # Bx(HW)
    I = reg_lab_int64.flatten(start_dim = -2)
    # BxS
    return torch.zeros(*reg_lab_int64.shape[:-2], max_num_segments, dtype = dtype).scatter_add_(-1, I, expand_ones_like(I, dtype = dtype))

class SelectiveSearch(torch.nn.Module):
    def __init__(self, base_k = 150, inc_k = 150, sigma = 0.8, min_size = 100, preset = 'fast', postprocess_labels = None, color_hist_bins = 8 * 4, texture_hist_bins = 8, remove_duplicate_boxes = False, randomize_rank = True, return_region_features = False):
        super().__init__()
        self.base_k = base_k
        self.inc_k = inc_k
        self.sigma = sigma
        self.min_size = min_size
        self.preset = preset
        self.remove_duplicate_boxes = remove_duplicate_boxes
        self.color_hist_bins = color_hist_bins
        self.texture_hist_bins = texture_hist_bins
        self.randomize_rank = randomize_rank
        self.return_region_features = return_region_features
        self.postprocess_labels = postprocess_labels if postprocess_labels is not None else (lambda reg_lab: reg_lab)

        if self.preset == 'single': # base_k = 200
            self.stack_color_spaces = lambda rgb, hsv, lab, gray: torch.stack([hsv], dim = -4)
            self.segmentations = [cv2.ximgproc.segmentation.createGraphSegmentation(self.sigma, float(base_k), self.min_size)]
            self.strategies = [
                [0.25, 0.25, 0.25, 0.25],
            ]
            
        elif self.preset == 'fast':
            self.stack_color_spaces = lambda rgb, hsv, lab, gray: torch.stack([hsv, lab], dim = -4) 
            self.segmentations = [cv2.ximgproc.segmentation.createGraphSegmentation(self.sigma, float(k), self.min_size) for k in range(self.base_k, 1 + self.base_k + self.inc_k * 2, self.inc_k)]
            self.strategies = [
                [0.25, 0.25, 0.25, 0.25],
                [0.3333, 0.3333, 0.3333, 0.00],
            ]
        
        elif self.preset == 'quality':
            self.stack_color_spaces = lambda rgb, hsv, lab, gray: torch.stack([hsv, lab, gray.expand_as(hsv), hsv[..., :1, :, :].expand_as(hsv), torch.cat([rgb[..., :2, :, :],  gray], dim = -3)], dim = -4)
            self.segmentations = [cv2.ximgproc.segmentation.createGraphSegmentation(self.sigma, float(k), self.min_size) for k in range(self.base_k, 1 + self.base_k + self.inc_k * 4, self.inc_k)]
            self.strategies = [
                [0.25, 0.25, 0.25, 0.25],
                [0.3333, 0.3333, 0.3333, 0.00],
                [1.00, 0.00, 0.00, 0.00],
                [0.00, 0.00, 1.00, 0.00],
            ]

    @staticmethod
    def get_region_mask(reg_lab, regs):
        return torch.stack([(reg_lab[reg['plane_id'][:-1]].unsqueeze(-1) == torch.tensor(list(reg['ids']), device = reg_lab.device, dtype = reg_lab.dtype)).any(dim = -1) for reg in regs])

    def forward(self, *, img_rgbb3hw_1 : 'B3HW', generator = None, print = print):
        # https://github.com/opencv/opencv_contrib/blob/master/modules/ximgproc/src/selectivesearchsegmentation.cpp
        assert img_rgbb3hw_1.is_contiguous() and img_rgbb3hw_1.is_floating_point() and img_rgbb3hw_1.ndim == 4 and img_rgbb3hw_1.shape[-3] == 3

        tic = time.time()

        # all image planes will be in [0.0; 255.0], input in [0.0, 1.0]
        hsv, lab, gray = rgb_to_hsv(img_rgbb3hw_1), rgb_to_lab(img_rgbb3hw_1), rgb_to_grayscale(img_rgbb3hw_1)
        #_hsv = _rgb_to_hsv(img)
        #hsv_ = torch.as_tensor(cv2.cvtColor(img[0].movedim(0, -1).numpy(), cv2.COLOR_RGB2HSV)) # 360, 1, 1
        
        hsv[..., 0, :, :] /= 2.0 * math.pi
        
        lab[..., 0, :, :] /= 100.0
        lab[..., 1:,:, :] /= 255.0
        lab[..., 1:,:, :] += 128/255.0
        
        # BxCx3xHxW in [0.0, 1.0]
        imgs = self.stack_color_spaces(img_rgbb3hw_1, hsv, lab, gray)
        
        # BxCx3xRxHxW (R = #rotations)
        grads = normalize_min_max_(image_gaussian_grads(imgs.flatten(end_dim = -4)).unflatten(0, imgs.shape[:-3]), dim = (-2, -1))
        print('image feat', time.time() - tic); tic = time.time()

        # BxCxGxHxW (C = #colorspaces, G = #graphsegs)
        # https://github.com/opencv/opencv_contrib/issues/3544 seems that float32 [0.0, 1.0] inputs are not correctly supported
        reg_lab = self.postprocess_labels(torch.stack([torch.as_tensor(gs.processImage(img)) for img in imgs.flatten(end_dim = -4).movedim(-3, -1).mul(255.0).contiguous().numpy() for gs in self.segmentations]).unflatten(0, imgs.shape[:-3] + (len(self.segmentations),)))
        # BxCxG
        num_segments = reg_lab.amax(dim = (-2, -1)).add_(1)
        max_num_segments = int(num_segments.amax())
        print('image seg', time.time() - tic); tic = time.time()

        region_features = HandcraftedRegionFeatures(
            imgs_bins = imgs.mul(self.color_hist_bins - 1).to(torch.int32).unsqueeze(2),
            grads_bins = grads.mul(self.texture_hist_bins - 1).to(torch.int32).unsqueeze(2),
            reg_lab = reg_lab, 
            
            max_num_segments = max_num_segments,
            color_hist_bins = self.color_hist_bins,
            texture_hist_bins = self.texture_hist_bins
        )
        # (BCG)xSxSx4 (S = max #segments)
        affinity = region_features.compute_region_affinity_pdist(
            img_size = region_features.img_size, 
            region_size = region_features.region_size, 
            bbox_xywh = region_features.bbox_xywh, 
            texture_hist = region_features.texture_hist, 
            color_hist = region_features.color_hist
        ).flatten(end_dim = -4)
        region_features.repeat_for_strategies_and_flatten(len(self.strategies))
        print('region feat', time.time() - tic)
        
        # (BCG)xSxS
        graphadj = self.build_graph(reg_lab, max_num_segments = max_num_segments).flatten(end_dim = -3)
        # (BCG)xSxSxT (T = # sTrategies)
        graphadj = (graphadj[..., None, None] * affinity.unsqueeze(-2) * torch.tensor(self.strategies, dtype = affinity.dtype, device = affinity.device)).sum(dim = -1)

        # (BCGT)xSxS
        ga = graphadj.movedim(-1, -3).flatten(end_dim = -3)
        # (BCGT)xSxS
        ga_sparse = ga.to_sparse()
        (plane_idx, r1, r2), sim = ga_sparse.indices(), ga_sparse.values()
        PQ = list(zip(sim.neg().tolist(), (plane_idx * max_num_segments + r1).tolist(), (plane_idx * max_num_segments + r2).tolist()))
        # (BCGT)xSxS
        (plane_idx, r1, r2) = (ga + ga.transpose(-2, -1)).to_sparse().indices()
        
        graph = {k : set(t[1] for t in g) for k, g in itertools.groupby(zip((plane_idx * max_num_segments + r1).tolist(), (plane_idx * max_num_segments + r2).tolist()), key = lambda t: t[0])}
        
        regs = [dict(strategy = strategy, plane_id = (b, c, g, t), plane_idx = plane_idx, id = r1, ids = {r1}, idx = r1, parent_idx = -1, level = 1 if r1 < int(num_segments[b, c, g]) else -1, bbox_xywh = tuple(region_features.bbox_xywh_[plane_idx * max_num_segments + r1]), region_size = region_features.region_size_[plane_idx * max_num_segments + r1], color_hist = region_features.color_hist_[plane_idx * max_num_segments + r1].clone() if self.return_region_features else None, texture_hist = region_features.texture_hist_[plane_idx * max_num_segments + r1].clone() if self.return_region_features else None) for plane_idx, (b, c, g, t, strategy) in enumerate((b, c, g, t, strategy) for b in range(num_segments.shape[0]) for c in range(num_segments.shape[1]) for g in range(num_segments.shape[2]) for t, strategy in enumerate(self.strategies)) for r1 in range(graphadj.shape[-2])]
        
        region_features.merge_regions_counter, region_features.merge_regions_time = 0, 0.0
        region_features.compute_region_affinity_counter, region_features.compute_region_affinity_time = 0, 0.00
        heapq.heapify(PQ)
        while PQ:
            negsim, u, v = heapq.heappop(PQ)
            if u not in graph or v not in graph:
                continue
            
            reg_fro, reg_to = regs[u], regs[v]
            regs.append( dict(strategy = reg_fro['strategy'], plane_id = reg_fro['plane_id'], plane_idx = reg_fro['plane_idx'], id = min(reg_fro['id'], reg_to['id']), ids = reg_fro['ids'] | reg_to['ids'], idx = len(regs), parent_idx = -1, level = 1 + max(reg_fro['level'], reg_to['level'])) )
            reg_fro['parent_idx'] = reg_to['parent_idx'] = len(regs) - 1
            tic = time.time()
            merged : dict = region_features.merge_regions_(reg_fro['id'], reg_to['id'], reg_fro['plane_idx'])
            regs[-1]['bbox_xywh'] = merged['bbox_xywh']
            regs[-1]['region_size'] = merged['region_size']
            regs[-1]['texture_hist'] = merged['texture_hist'].clone() if self.return_region_features else None
            regs[-1]['color_hist'] = merged['color_hist'].clone() if self.return_region_features else None
            region_features.merge_regions_time += time.time() - tic; region_features.merge_regions_counter += 1

            for new_edge in self.contract_graph_edge(region_features, u, v, reg_fro['parent_idx'], regs, graph):
                heapq.heappush(PQ, new_edge)
        
        print('merge reg', region_features.merge_regions_counter, region_features.merge_regions_time)
        print('sim reg', region_features.compute_region_affinity_counter, region_features.compute_region_affinity_time)

        for reg, level_multiplier in zip(regs, torch.rand(len(regs), generator = generator).tolist() if self.randomize_rank else torch.ones(len(regs)).tolist()):
            reg['rank'] = reg['level'] * level_multiplier

        key_img_id, key_rank = (lambda reg: reg['plane_id'][0]), (lambda reg: reg['rank'])
        regions_by_image = list(map({k: sorted(list(g), key = key_rank) for k, g in itertools.groupby(sorted([reg for reg in regs if reg['level'] >= 0], key = key_img_id), key = key_img_id)}.get, range(len(img_rgbb3hw_1))))
        

        if self.remove_duplicate_boxes:
            boxes_xywh_without_duplicates = [{reg['bbox_xywh'] : i for i, reg in enumerate(by_image[b]) } for b in range(len(img_rgbb3hw_1))]
            boxes_xywh = [torch.tensor(list(boxes_xywh_without_duplicates[b].keys()), dtype = torch.int16) for b in range(len(img_rgbb3hw_1))]
            regions = [[by_image[b][i] for i in boxes_xywh_without_duplicates[b].values()] for b in range(len(img_rgbb3hw_1))]
        else:
            boxes_xywh = [torch.tensor([reg['bbox_xywh'] for reg in regs], dtype = torch.int16) for regs in regions_by_image]
            regions = regions_by_image

        return boxes_xywh, regions, reg_lab
    
    @staticmethod
    def build_graph(reg_lab : 'BIGHW', max_num_segments : int):
        I = torch.stack(torch.meshgrid(*[torch.arange(s) for s in reg_lab.shape[:3]], indexing = 'ij'), dim = -1)[..., None, None]
        DX = torch.stack([reg_lab[..., :, :-1], reg_lab[..., :, 1:]], dim = -1).movedim(-1, -3)
        DY = torch.stack([reg_lab[..., :-1, :], reg_lab[..., 1:, :]], dim = -1).movedim(-1, -3)
        dx = torch.cat([I.expand(-1, -1, -1, -1, *DX.shape[-2:]), DX], dim = -3).flatten(start_dim = -2)
        dy = torch.cat([I.expand(-1, -1, -1, -1, *DY.shape[-2:]), DY], dim = -3).flatten(start_dim = -2)
        i = torch.cat([dx, dy], dim = -1).movedim(-2, 0).flatten(start_dim = 1)
        v = torch.ones(i.shape[1], dtype = torch.bool)
        A = torch.sparse_coo_tensor(i, v, reg_lab.shape[:-2] + (max_num_segments, max_num_segments), dtype = torch.int64)
        A += A.transpose(-1, -2)
        return A.coalesce().to(torch.bool).to_dense().triu(diagonal = 1)

    @staticmethod
    def contract_graph_edge(self, u, v, ww, regs, graph):
        graph[ww] = set()
        new_edges = []
        for uu in [u, v]:
            for vv in graph.pop(uu, []):
                if vv == u or vv == v:
                    continue

                tic = time.time()
                #aff = region_features.compute_region_affinity(regs[ww]['id'], regs[vv]['id'], regs[ww]['plane_idx'], None, *regs[ww]['strategy'])
                (plane_idx, r1, r2) = regs[ww]['plane_idx'], regs[ww]['id'], regs[vv]['id'] 
                aff = self.compute_region_affinity_pairwise_distance(
                    self.img_size, 
                    self.region_size_[plane_idx * self.max_num_segments + r1], 
                    self.bbox_xywh_[plane_idx * self.max_num_segments + r1], 
                    self.color_hist_[plane_idx * self.max_num_segments + r1], 
                    self.texture_hist_[plane_idx * self.max_num_segments + r1], 
                    self.region_size_[plane_idx * self.max_num_segments + r2], 
                    self.bbox_xywh_[plane_idx * self.max_num_segments + r2], 
                    self.color_hist_[plane_idx * self.max_num_segments + r2], 
                    self.texture_hist_[plane_idx * self.max_num_segments + r2], 
                    *regs[ww]['strategy'],
                    buf = self.buf
                )

                new_edges.append((-aff, ww, vv))
                self.compute_region_affinity_time += time.time() - tic; self.compute_region_affinity_counter += 1

                graph[vv].remove(uu)
                graph[vv].add(ww)
                graph[ww].add(vv)
        return new_edges

class HandcraftedRegionFeatures:
    def __init__(self, imgs_bins: 'BCG3HW', grads_bins: 'BCG3RHW', reg_lab : 'BCGHW', max_num_segments: int, color_hist_bins : int, texture_hist_bins : int, eps : float = 1e-12):
        img_channels, img_height, img_width = imgs_bins.shape[-3:]
        self.max_num_segments : int = max_num_segments
        self.img_size : int = img_height * img_width
       
        # needed to work around https://github.com/pytorch/pytorch/issues/61819
        reg_lab_int64 = reg_lab.to(torch.int64)
        # BxCxGxS 
        self.region_size = area_per_segment(reg_lab_int64, max_num_segments = self.max_num_segments, dtype = torch.int32)
        # BxCxGxSx4
        self.bbox_xywh = bbox_per_segment(reg_lab_int64, max_num_segments = self.max_num_segments, dtype = torch.int16)

        # BxCxGx3x(HW), img_bins is broadcasted
        I = (reg_lab[..., None, :, :] * color_hist_bins + imgs_bins).flatten(start_dim = -2)
        # BxCxGxSxD1 (f32) where S = #segments and D1 = 96
        self.color_hist = torch.zeros(*reg_lab.shape[:-2], img_channels, self.max_num_segments * color_hist_bins, dtype = torch.float32).scatter_add_(-1, I.to(torch.int64), expand_ones_like(I, dtype = torch.float32)).unflatten(-1, (self.max_num_segments, color_hist_bins)).movedim(-2, -3).flatten(start_dim = -2)
        # for histogram normalization, each pixel contributes to 3 = #colorchannels bins, eps to avoid divbyzero and nans in affinity
        self.color_hist /= self.region_size.mul(float(imgs_bins.shape[-3])).add_(eps).unsqueeze(-1)

        # BxCxGx3xRx(HW), grads_bins is broadcasted
        I = (reg_lab[..., None, None, :, :] * texture_hist_bins + grads_bins).flatten(start_dim = -2)
        # BxCxGxSxD2 (f32) where S = #segments and D1 = 192
        self.texture_hist = torch.zeros(*reg_lab.shape[:-2], img_channels, grads_bins.shape[-3], self.max_num_segments * texture_hist_bins, dtype = torch.float32).scatter_add_(-1, I.to(torch.int64), expand_ones_like(I, dtype = torch.float32)).unflatten(-1, (self.max_num_segments, texture_hist_bins)).movedim(-2, -4).flatten(start_dim = -3)
        # for histogram normalization, each pixel contributes to (3 = #colorchannels) * (8 = #rotations) bins, maybe except exactly 0textures which contribute to several bins, eps to avoid divbyzero and nans in affinity
        self.texture_hist /= self.region_size.mul(float(grads_bins.shape[-4] * grads_bins.shape[-3])).add_(eps).unsqueeze(-1)
        #
        self.buf = torch.empty(max(self.texture_hist.shape[-1], self.color_hist.shape[-1]), dtype = torch.float32)
        
    def repeat_for_strategies_and_flatten(self, num_strategies):
        self.region_size_ = self.region_size.flatten(end_dim = -2).unsqueeze(-2).repeat(1, num_strategies, 1).flatten().tolist()
        self.bbox_xywh_ = self.bbox_xywh.flatten(end_dim = -3).unsqueeze(-3).repeat(1, num_strategies, 1, 1).flatten(end_dim = -2).tolist()
        self.color_hist_ = self.color_hist.flatten(end_dim = -3).unsqueeze(-3).repeat(1, num_strategies, 1, 1).flatten(end_dim = -2)
        self.texture_hist_ = self.texture_hist.flatten(end_dim = -3).unsqueeze(-3).repeat(1, num_strategies, 1, 1).flatten(end_dim = -2)
        
    def merge_regions_(self, r1, r2, plane_idx) -> dict:
        s1, s2 = self.region_size_[plane_idx * self.max_num_segments + r1], self.region_size_[plane_idx * self.max_num_segments + r2]
        s1s2 = s1 + s2

        self.region_size_[plane_idx * self.max_num_segments + r2] = self.region_size_[plane_idx * self.max_num_segments + r1] = s1s2
        self.bbox_xywh_[plane_idx * self.max_num_segments + r2] = self.bbox_xywh_[plane_idx * self.max_num_segments + r1] = bbox_merge_tuple(self.bbox_xywh_[plane_idx * self.max_num_segments + r1], self.bbox_xywh_[plane_idx * self.max_num_segments + r2], return_area = False)
        self.color_hist_[plane_idx * self.max_num_segments + r2].copy_(self.color_hist_[plane_idx * self.max_num_segments + r1].mul_(s1).add_(self.color_hist_[plane_idx * self.max_num_segments + r2].mul_(s2)).div_(s1s2))
        self.texture_hist_[plane_idx * self.max_num_segments + r2].copy_(self.texture_hist_[plane_idx * self.max_num_segments + r1].mul_(s1).add_(self.texture_hist_[plane_idx * self.max_num_segments + r2].mul_(s2)).div_(s1s2))
        
        return dict(
            region_size = self.region_size_[plane_idx * self.max_num_segments + r1], 
            bbox_xywh = self.bbox_xywh_[plane_idx * self.max_num_segments + r1], 
            color_hist = self.color_hist_[plane_idx * self.max_num_segments + r1], 
            texture_hist = self.texture_hist_[plane_idx * self.max_num_segments + r1]
        )

    @staticmethod
    def compute_region_affinity_pdist(img_size, region_size, bbox_xywh, color_hist, texture_hist):
        size_affinity = region_size.unsqueeze(-1).add(region_size.unsqueeze(-2)).div(-float(img_size)).add_(1).clamp_(min = 0, max = 1)
        fill_affinity = bbox_merge_tensor(bbox_xywh.unsqueeze(-2), bbox_xywh.unsqueeze(-3), dtype = region_size.dtype, return_area = True).sub_(region_size.unsqueeze(-1)).sub_(region_size.unsqueeze(-2)).div(-float(img_size)).add_(1).clamp_(min = 0, max = 1)
        color_affinity = torch.min(color_hist.unsqueeze(-2), color_hist.unsqueeze(-3)).sum(dim = -1)
        texture_affinity = torch.min(texture_hist.unsqueeze(-2), texture_hist.unsqueeze(-3)).sum(dim = -1)
        return torch.stack([fill_affinity, texture_affinity, size_affinity, color_affinity], dim = -1)
    
    @staticmethod
    def compute_region_affinity_pairwise_distance(img_size, region_size_1, bbox_xywh_1, color_hist_1, texture_hist_1, region_size_2, bbox_xywh_2, color_hist_2, texture_hist_2, fill = 0.0, texture = 0.0, size = 0.0, color = 0.0, buf = None):
        size = 0.0 if size  == 0 else size * max(0, min(1, 1 - (region_size_1 + region_size_2) / img_size))
        fill = 0.0 if fill  == 0 else fill * max(0, min(1, 1 - (bbox_merge_tuple(bbox_xywh_1, bbox_xywh_2, return_area = True) - region_size_1 - region_size_2) / img_size))
        color= 0.0 if color == 0 else color * float(torch.min(color_hist_1, color_hist_2, out = None if buf is None else buf.resize_(0)).sum(dim = -1))
        texture = 0.0 if texture == 0 else texture * float(torch.min(texture_hist_1, texture_hist_2, out = None if buf is None else buf.resize_(0)).sum(dim = -1))
        return fill + texture + size + color
