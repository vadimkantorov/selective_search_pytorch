import os
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

def main(img_rgbhwc_255, gradio, input_path, output_dir, preset, algo, remove_duplicate_boxes, profile, seed, grid, dot, png, gif, beginboxind, endboxind, selectivesearchsegmentation_opencv_custom_so):

    class print(metaclass = type('', (type,), dict(__repr__ = lambda self: print.log))):
        log = ''
        def __init__(self, *args, **kwargs):
            print.log += str(args) + '\n'
            __builtins__.print(*args, **kwargs)

    print('height x width x 3:', img_rgbhwc_255.shape, 'dtype = ', img_rgbhwc_255.dtype)
    img_rgb1chw_1 = torch.as_tensor(img_rgbhwc_255).movedim(-1, -3).unsqueeze(0) / 255.0

    tic = toc = 0.0

    if algo in ['pytorch', 'opencv_custom']:
        algo = selectivesearchsegmentation.SelectiveSearch(preset = preset, remove_duplicate_boxes = remove_duplicate_boxes) if algo == 'pytorch' else selectivesearchsegmentation_opencv_custom.SelectiveSearchOpenCVCustom(preset = preset, lib_path = selectivesearchsegmentation_opencv_custom_so)
        generator = torch.Generator().manual_seed(seed)
        tic = time.time()
        if profile:
            with torch.profiler.profile(activities = [torch.profiler.ProfilerActivity.CPU], record_shapes = True) as prof:
                boxes_xywh, regions, reg_lab = algo(img_rgb1chw_1, generator = generator, print = print)
            print(prof.key_averages().table(sort_by = 'cpu_time_total', row_limit = 10))
        else:
            boxes_xywh, regions, reg_lab = algo(img_rgb1chw_1, generator = generator, print = print)
        toc = time.time()

        plane_ids = set(reg['plane_id'] for reg in sum(regions, []))

    else:
        algo = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        algo.setBaseImage(img_rgbhwc_255[::-1].copy())
        dict(fast = algo.switchToSelectiveSearchFast, quality = algo.switchToSelectiveSearchQuality, single = algo.switchToSingleStrategy)[preset]()
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

    os.makedirs(output_dir, exist_ok = True)
    if png:
        plot_instance_grid(output_dir, plane_ids, algo, boxes_xywh, regions, reg_lab)
    if dot:
        plot_merging_trees(output_dir, plane_ids, algo, boxes_xywh, regions, reg_lab)
    if gif:
        plot_merging_segments(output_dir, plane_ids, algo, boxes_xywh, regions, reg_lab)
    return img_rgbhwc_255, repr(print)

def plot_instance_grid(output_dir, plane_ids, algo, boxes_xywh, regions, reg_lab):
    for plane_id in plane_ids:
        regs = [reg for reg in sum(regions, []) if reg['plane_id'] == plane_id]
        suffix = ''.join(map(str, plane_id))
        output_path = os.path.join(output_dir, 'png_' + os.path.basename(input_path) + f'.{suffix}.png')

        fig = plt.figure(figsize = (grid, grid))
        plt.subplot(grid, grid, 1)
        plt.imshow(img_rgbhwc_255, aspect = 'auto')
        plt.axis('off')
        for reg in regs[beginboxind : beginboxind + endboxind]:
            plt.gca().add_patch(matplotlib.patches.Rectangle(reg['bbox_xywh'][:2], *reg['bbox_xywh'][2:], linewidth = 1, edgecolor = 'r', facecolor = 'none'))

        for k, reg in enumerate(regs[:grid * grid - 1]):
            plt.subplot(grid, grid, 2 + k)
            mask = algo.get_region_mask(reg_lab, [reg] )[0]
            
            plt.imshow((img_rgbhwc_255 * mask[..., None] + img_rgbhwc_255 * mask[..., None].logical_not() // 10).to(torch.uint8), aspect = 'auto')
            plt.gca().add_patch(matplotlib.patches.Rectangle(reg['bbox_xywh'][:2], *reg['bbox_xywh'][2:], linewidth = 1, edgecolor = 'r', facecolor = 'none'))
            plt.axis('off')

        plt.subplots_adjust(0, 0, 1, 1, wspace = 0, hspace = 0)
        plt.savefig(output_path)
        plt.close(fig)
        print(output_path)


def plot_merging_trees(output_dir, plane_ids, algo, boxes_xywh, regions, reg_lab):
    for plane_id in plane_ids:
        regs = [reg for reg in sum(regions, []) if reg['plane_id'] == plane_id]
        suffix = ''.join(map(str, plane_id))
        output_path = os.path.join(output_dir, 'dot_' + os.path.basename(input_path) + f'.{suffix}.dot')
        
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

def plot_merging_segments(output_dir, plane_ids, algo, boxes_xywh, regions, reg_lab):
    for plane_id in plane_ids:
        regs = [reg for reg in sum(regions, []) if reg['plane_id'] == plane_id]
        suffix = ''.join(map(str, plane_id))
        output_path = os.path.join(output_dir, 'gif_' + os.path.basename(input_path) + f'.{suffix}.gif')
        
        key_level = lambda reg: reg['level']
        level2regions = {k : list(g) for k, g in itertools.groupby(sorted(regs, key = key_level), key = key_level)}

        fig = plt.figure(figsize = (grid, grid))
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--selectivesearchsegmentation_opencv_custom_so', default = 'opencv_custom/selectivesearchsegmentation_opencv_custom_.so')
    parser.add_argument('--gradio', action = 'store_true') 
    parser.add_argument('--input-path', '-i', default = './examples/astronaut.jpg')
    parser.add_argument('--output-dir', '-o', default = './out')
    parser.add_argument('--dot', action = 'store_true')
    parser.add_argument('--png', action = 'store_true')
    parser.add_argument('--gif', action = 'store_true')
    
    parser.add_argument('--preset', choices = ['fast', 'quality', 'single'], default = 'fast')
    parser.add_argument('--algo', choices = ['pytorch', 'opencv', 'opencv_custom'], default = 'pytorch')
    parser.add_argument('--beginboxind', type = int, default = 0)
    parser.add_argument('--endboxind', type = int, default = 64)
    parser.add_argument('--remove-duplicate-boxes', action = 'store_true') 
    parser.add_argument('--profile', action = 'store_true') 
    parser.add_argument('--seed', type = int, default = 42)
    parser.add_argument('--grid', type = int, default = 4)
    
    args = parser.parse_args()

    print(args, end = '\n\n')

    if args.gradio:
        # importing gradio on top, makes the program hang after finish
        import gradio
        gradio_inputs = [
            gradio.Image(args.input_path),
            gradio.Radio(label = 'preset', choices = ['fast', 'quality', 'single'], value = 'fast'),
            gradio.Radio(label = 'algo', choices = ['pytorch', 'opencv', 'opencv_custom'], value = 'pytorch'),
            gradio.Checkbox(label = 'remove duplicate boxes'),
            gradio.Checkbox(label = 'profile'),
            gradio.Number(label = 'beginboxind', minimum = 0, precision = 0, value = 0),
            gradio.Number(label = 'endboxind', minimum = 0, precision = 0, value = 64),
            gradio.Number(label = 'grid', minimum = 0, precision = 0, value = 4),
            gradio.Number(label = 'seed', minimum = 0, precision = 0, value = 42),
        ]
        gradio_outputs = [
            gradio.Image(),
            gradio.Textbox(label = 'log')
        ]
        gradio_submit = lambda img_rgbhwc_255, preset, algo, remove_duplicate_boxes, profile, beginboxind, endboxind, grid, seed: main(img_rgbhwc_255, **dict(vars(args), preset = preset, algo = algo, remove_duplicate_boxes = remove_duplicate_boxes, profile = profile, beginboxind = beginboxind, endboxind = endboxind, seed = seed, grid = grid))
        gradio_demo = gradio.Interface(gradio_submit, inputs = gradio_inputs, outputs = gradio_outputs)
        gradio_demo.launch()
    else:
        img_rgbhwc_255 = plt.imread(args.input_path).copy()
        main(img_rgbhwc_255 = img_rgbhwc_255, **vars(args))
