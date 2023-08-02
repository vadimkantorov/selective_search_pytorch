import sys
import time
import random
import itertools
import argparse
import matplotlib.animation, matplotlib.pyplot as plt

import torch

try:
    import cv2
except Exception as e:
    print(e)
    cv2 = None

try:
    import selectivesearchsegmentation
except Exception as e:
    print(e)
    selectivesearchsegmentation = None

try:
    from opencv_custom import selectivesearchsegmentation_opencv_custom
except Exception as e:
    print(e)
    selectivesearchsegmentation_opencv_custom = None

parser = argparse.ArgumentParser()
parser.add_argument('--input-path', '-i')
parser.add_argument('--output-path', '-o')
parser.add_argument('--debugexit', action = 'store_true') 
parser.add_argument('--topk', type = int, default = 64)
parser.add_argument('--preset', choices = ['fast', 'quality', 'single'], default = 'fast')
parser.add_argument('--algo', choices = ['pytorch', 'opencv', 'opencv_custom'], default = 'pytorch')
parser.add_argument('--selectivesearchsegmentation_opencv_custom_so', default = 'opencv_custom/selectivesearchsegmentation_opencv_custom_.so')
parser.add_argument('--seed', type = int, default = 42)
parser.add_argument('--begin', type = int, default = 0)
parser.add_argument('--grid', type = int, default = 4)
parser.add_argument('--plane-id', type = int, nargs = 4, default = [0, 0, 0, 0])
args = parser.parse_args()

print(args)
random.seed(args.seed)

img_rgbhwc_255 = torch.as_tensor(plt.imread(args.input_path).copy())
img_bgrhwc_255 = img_rgbhwc_255.flip(-1).numpy()
img_rgb1chw_1 = img_rgbhwc_255.movedim(-1, -3).unsqueeze(0) / 255.0

print(args.input_path, img_rgbhwc_255.shape)

tic = toc = 0.0

if args.algo in ['pytorch', 'opencv_custom']:
    algo = selectivesearchsegmentation.SelectiveSearch(preset = args.preset) if args.algo == 'pytorch' else selectivesearchsegmentation_opencv_custom.SelectiveSearchOpenCVCustom(preset = args.preset, lib_path = args.selectivesearchsegmentation_opencv_custom_so)
    tic = time.time()
    boxes_xywh, regions, reg_lab = algo(img_rgb1chw_1)
    toc = time.time()
    regs = regions[0]
    mask = lambda k: algo.get_region_mask( reg_lab, [ regs[k] ] )[0]
    boxes_xywh = boxes_xywh[0].tolist()
    key_level = lambda reg: reg['level']
    level2regions = lambda plane_id, regs = regs: {k : list(g) for k, g in itertools.groupby(sorted([reg for reg in regs if reg['plane_id'][:-1] == tuple(plane_id)[:-1]], key = key_level), key = key_level)}

else:
    algo = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    algo.setBaseImage(img_bgrhwc_255)
    dict(fast = algo.switchToSelectiveSearchFast, quality = algo.switchToSelectiveSearchQuality, single = algo.switchToSingleStrategy)[args.preset]()
    tic = time.time()
    boxes_xywh = algo.process()
    toc = time.time()
    regions = None
    reg_lab = None
    
    level2regions = lambda plane_id: []
    def mask(k):
        x, y, w, h = boxes_xywh[k]
        res = torch.zeros(img_rgbhwc_255.shape[:-1])
        res[y : y + h, x : x + w] = 1
        return res

print('boxes:', len(boxes_xywh))
print('height:', img_rgbhwc_255.shape[0], 'width:', img_rgbhwc_255.shape[1])
print('ymax:', max(y + h - 1 for x, y, w, h in boxes_xywh), 'xmax:', max(x + w - 1 for x, y, w, h in boxes_xywh))
print('processing time', toc - tic)

max_num_segments = 1 + (max(reg['id'] for reg in regs) if regions is not None else 0)
reg_lab_ = reg_lab[tuple(args.plane_id[:-1])].clone() if reg_lab is not None else []
l2r = level2regions(plane_id = args.plane_id)

if args.debugexit:
    print('debugexit = True, quitting')
    sys.exit(0)

fig = plt.figure(figsize = (args.grid, args.grid))
plt.subplot(args.grid, args.grid, 1)
plt.imshow(img_rgbhwc_255, aspect = 'auto')
plt.axis('off')
for x, y, w, h in boxes_xywh[args.begin : args.begin + args.topk]:
    plt.gca().add_patch(matplotlib.patches.Rectangle((x, y), w, h, linewidth = 1, edgecolor = 'r', facecolor = 'none'))

for k in range(args.grid * args.grid - 1):
    plt.subplot(args.grid, args.grid, 2 + k)
    m = mask(args.begin + k)
    x, y, w, h = boxes_xywh[k]
    
    plt.imshow((img_rgbhwc_255 * m[..., None] + img_rgbhwc_255 * m[..., None].logical_not() // 10).to(torch.uint8), aspect = 'auto')
    
    plt.gca().add_patch(matplotlib.patches.Rectangle((x, y), w, h, linewidth = 1, edgecolor = 'r', facecolor = 'none'))
    plt.axis('off')

plt.subplots_adjust(0, 0, 1, 1, wspace = 0, hspace = 0)
plt.savefig(args.output_path)
plt.close(fig)
print(args.output_path)

if not l2r:
    print('l2r (index of regions for a given merging level) is empty, cannot produce animation, quitting')
    sys.exit(0)

fig = plt.figure(figsize = (args.grid, args.grid))
fig.set_tight_layout(True)
#fig.subplots_adjust(0, 0, 1, 1, wspace = 0, hspace = 0)

def update(level, im = []):
    for reg in l2r[level]:
        min_id = min(reg['ids'])
        for reg_id in reg['ids']:
            reg_lab_[reg_lab_ == reg_id] = min_id
    y = reg_lab_ / max_num_segments

    if not im:
        im.append(plt.imshow(y, animated = True, cmap = 'hsv', aspect = 'auto'))
        plt.axis('off')

    im[0].set_array(y)
    im[0].set_clim(0, 1)
    plt.suptitle(f'level: [{level}]')
    return im

matplotlib.animation.FuncAnimation(fig, update, frames = sorted(l2r), interval = 1000).save(args.output_path + '.gif', dpi = 80)
plt.close(fig)
print(args.output_path + '.gif')
