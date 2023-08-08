import os
import sys
import time
import random
import itertools
import argparse
import subprocess

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
parser.add_argument('--input-path', '-i', default = 'astronaut.jpg')
parser.add_argument('--output-path', '-o', default = './out')
parser.add_argument('--profile', action = 'store_true') 
parser.add_argument('--preset', choices = ['fast', 'quality', 'single'], default = 'fast')
parser.add_argument('--algo', choices = ['pytorch', 'opencv', 'opencv_custom'], default = 'pytorch')
parser.add_argument('--selectivesearchsegmentation_opencv_custom_so', default = 'opencv_custom/selectivesearchsegmentation_opencv_custom_.so')
parser.add_argument('--remove-duplicate-boxes', action = 'store_true') 
parser.add_argument('--seed', type = int, default = 42)
parser.add_argument('--grid', type = int, default = 4)

parser.add_argument('--dot', action = 'store_true')
parser.add_argument('--png', action = 'store_true')
parser.add_argument('--gif', action = 'store_true')

parser.add_argument('--beginboxind', type = int, default = 0)
parser.add_argument('--endboxind', type = int, default = 64)
args = parser.parse_args()
print(args)

img_rgbhwc_255 = torch.as_tensor(plt.imread(args.input_path).copy())
img_bgrhwc_255 = img_rgbhwc_255.flip(-1).numpy()
img_rgb1chw_1 = img_rgbhwc_255.movedim(-1, -3).unsqueeze(0) / 255.0
print('height x width x 3:', img_rgbhwc_255.shape)

tic = toc = 0.0

if args.algo in ['pytorch', 'opencv_custom']:
    algo = selectivesearchsegmentation.SelectiveSearch(preset = args.preset, remove_duplicate_boxes = args.remove_duplicate_boxes) if args.algo == 'pytorch' else selectivesearchsegmentation_opencv_custom.SelectiveSearchOpenCVCustom(preset = args.preset, lib_path = args.selectivesearchsegmentation_opencv_custom_so)
    generator = torch.Generator().manual_seed(args.seed)
    tic = time.time()
    if args.profile:
        with torch.profiler.profile(activities = [torch.profiler.ProfilerActivity.CPU], record_shapes = True) as prof:
            boxes_xywh, regions, reg_lab = algo(img_rgb1chw_1, generator = generator)
        print(prof.key_averages().table(sort_by = 'cpu_time_total', row_limit = 10))
    else:
        boxes_xywh, regions, reg_lab = algo(img_rgb1chw_1, generator = generator)
    toc = time.time()

    plane_ids = set(reg['plane_id'] for reg in sum(regions, []))

else:
    algo = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    algo.setBaseImage(img_bgrhwc_255)
    dict(fast = algo.switchToSelectiveSearchFast, quality = algo.switchToSelectiveSearchQuality, single = algo.switchToSingleStrategy)[args.preset]()
    tic = time.time()
    boxes_xywh = [algo.process()]
    toc = time.time()
    regions = None
    reg_lab = None
    
    level2regions = lambda plane_id: []
    #type(algo).get_region_mask = lambda k: 0wi
    def mask(k):
        x, y, w, h = boxes_xywh[k]
        res = torch.zeros(img_rgbhwc_255.shape[:-1])
        res[y : y + h, x : x + w] = 1
        return res

print('boxes:', sum(map(len, boxes_xywh)))
print('height x width: {0}x{1}'.format(*img_rgbhwc_255.shape[:2]))
print('ymax:', max(y + h - 1 for xywh in boxes_xywh for x, y, w, h in xywh.tolist()), 'xmax:', max(x + w - 1 for xywh in boxes_xywh for x, y, w, h in xywh.tolist()))
print('processing time', toc - tic)

os.makedirs(args.output_path, exist_ok = True)

for plane_id in (plane_ids if args.dot else []):
    regs = [reg for reg in sum(regions, []) if reg['plane_id'] == plane_id]
    suffix = ''.join(map(str, plane_id))
    output_path = os.path.join(args.output_path, 'dot_' + os.path.basename(args.input_path) + f'.{suffix}.dot')
    
    with open(output_path, 'w') as dot:
        dot.write('digraph {\n\n')
        for reg in regs:
            dot.write('{idx} -> {parent_idx} ;\n'.format(**reg))
            if reg['level'] == 0:
                dot.write('{idx} [style=filled, fillcolor=lightblue] ;\n'.format(**reg))
            if reg['parent_idx'] == -1:
                dot.write('{parent_idx} [style=filled, fillcolor=red] ;\n'.format(**reg))
            dot.write('{idx} [label=<{idx}<br/>[{level}]>] ;\n'.format(**reg))
        dot.write('\n}\n')
    print(output_path)
    dot_cmd = ['dot', '-Tpng', output_path, '-o', output_path + '.png']
    try:
        print(' '.join(dot_cmd), '#', subprocess.check_call(dot_cmd))
    except Exception as e:
        print(' '.join(dot_cmd), '# dot cmd failed', e)
    print(output_path + '.png', end = '\n\n')

for plane_id in (plane_ids if args.png else []):
    regs = [reg for reg in sum(regions, []) if reg['plane_id'] == plane_id]
    suffix = ''.join(map(str, plane_id))
    output_path = os.path.join(args.output_path, 'png_' + os.path.basename(args.input_path) + f'.{suffix}.png')

    fig = plt.figure(figsize = (args.grid, args.grid))
    plt.subplot(args.grid, args.grid, 1)
    plt.imshow(img_rgbhwc_255, aspect = 'auto')
    plt.axis('off')
    for reg in regs[args.beginboxind : args.beginboxind + args.endboxind]:
        plt.gca().add_patch(matplotlib.patches.Rectangle(reg['bbox_xywh'][:2], *reg['bbox_xywh'][2:], linewidth = 1, edgecolor = 'r', facecolor = 'none'))

    for k, reg in enumerate(regs[:args.grid * args.grid - 1]):
        plt.subplot(args.grid, args.grid, 2 + k)
        mask = algo.get_region_mask(reg_lab, [reg] )[0]
        
        plt.imshow((img_rgbhwc_255 * mask[..., None] + img_rgbhwc_255 * mask[..., None].logical_not() // 10).to(torch.uint8), aspect = 'auto')
        plt.gca().add_patch(matplotlib.patches.Rectangle(reg['bbox_xywh'][:2], *reg['bbox_xywh'][2:], linewidth = 1, edgecolor = 'r', facecolor = 'none'))
        plt.axis('off')

    plt.subplots_adjust(0, 0, 1, 1, wspace = 0, hspace = 0)
    plt.savefig(output_path)
    plt.close(fig)
    print(output_path)

for plane_id in (plane_ids if args.gif else []):
    regs = [reg for reg in sum(regions, []) if reg['plane_id'] == plane_id]
    suffix = ''.join(map(str, plane_id))
    output_path = os.path.join(args.output_path, 'gif_' + os.path.basename(args.input_path) + f'.{suffix}.gif')
    
    key_level = lambda reg: reg['level']
    level2regions = {k : list(g) for k, g in itertools.groupby(sorted(regs, key = key_level), key = key_level)}

    fig = plt.figure(figsize = (args.grid, args.grid))
    fig.set_tight_layout(True)
    def update(level, im = []):
        segm = reg_lab[plane_id[:-1]].clone()
        max_num_segments = 1 + max(reg['id'] for reg in regs)
        
        for reg in level2regions[level]:
            min_id = min(reg['ids'])
            for reg_id in reg['ids']:
                segm[segm == reg_id] = min_id
        
        img = segm / max_num_segments

        if not im:
            im.append(plt.imshow(img, animated = True, cmap = 'hsv', aspect = 'auto'))
            plt.axis('off')

        im[0].set_array(img)
        im[0].set_clim(0, 1)
        plt.suptitle(f'level: [{level}]')
        return im

    matplotlib.animation.FuncAnimation(fig, update, frames = sorted(level2regions), interval = 1000).save(output_path, dpi = 80)
    plt.close(fig)
    print(output_path)
