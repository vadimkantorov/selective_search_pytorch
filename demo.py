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

def main(img_rgbhwc_255, gradio, input_path, output_dir, preset, algo, remove_duplicate_boxes, profile, seed, grid, beginboxind, endboxind, selectivesearchsegmentation_opencv_custom_so, vis_instance_grids, vis_merging_segments, vis_merging_trees):

    class print(metaclass = type('', (type, ), dict(__repr__ = lambda self: print.log))):
        log = ''
        def __init__(self, *args, **kwargs):
            print.log += str(args) + '\n'
            __builtins__.print(*args, **kwargs)

    tic = toc = 0.0

    print('height x width x 3:', img_rgbhwc_255.shape, 'dtype = ', img_rgbhwc_255.dtype)

    if algo in ['pytorch', 'opencv_custom']:
        algo = selectivesearchsegmentation.SelectiveSearch(preset = preset, remove_duplicate_boxes = remove_duplicate_boxes) if algo == 'pytorch' else selectivesearchsegmentation_opencv_custom.SelectiveSearchOpenCVCustom(preset = preset, lib_path = selectivesearchsegmentation_opencv_custom_so)
        generator = torch.Generator().manual_seed(seed)
        img_rgb1chw_1 = torch.as_tensor(img_rgbhwc_255).movedim(-1, -3).div(255).unsqueeze(0)
        
        tic = time.time()
        prof = torch.profiler.profile(activities = [torch.profiler.ProfilerActivity.CPU], record_shapes = True)
        if profile:
            prof.__enter__()
        
        boxes_xywh, regions, reg_lab = algo(img_rgb1chw_1, generator = generator, print = print)
        plane_ids = sorted(set(reg['plane_id'] for reg in sum(regions, [])))
        
        if profile:
            prof.__exit__()
            print(prof.key_averages().table(sort_by = 'cpu_time_total', row_limit = 10))
        toc = time.time()

    else:
        algo = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        algo.setBaseImage(img_rgbhwc_255[::-1].copy())
        dict(fast = algo.switchToSelectiveSearchFast, quality = algo.switchToSelectiveSearchQuality, single = algo.switchToSingleStrategy)[preset]()
        tic = time.time()
        boxes_xywh = [algo.process()]
        regions = [ [dict(plane_id = (0,), idx = idx, bbox_xywh = bbox_xywh) for idx, bbox_xywh in enumerate(boxes_xywh[0].tolist())] ]
        reg_lab = None
        plane_ids = [(0,)]
        toc = time.time()
        def get_region_mask(self, reg_lab, regs):
            mask = torch.zeros(len(regs), *img_rgbhwc_255.shape[:-1], dtype = torch.bool)
            for k, reg in enumerate(regs):
                x, y, w, h = regions[reg['plane_id'][0]][reg['idx']]['bbox_xywh']
                mask[k, y : y + h, x : x + w] = 1
            return mask
        type(algo).get_region_mask = lambda self, reg_lab, regs: get_region_mask(self, reg_lab, regs)

    print('boxes:', sum(map(len, boxes_xywh)))
    print('height x width: {0}x{1}'.format(*img_rgbhwc_255.shape[:2]))
    print('ymax:', max(y + h - 1 for xywh in boxes_xywh for x, y, w, h in xywh.tolist()), 'xmax:', max(x + w - 1 for xywh in boxes_xywh for x, y, w, h in xywh.tolist()))
    print('processing time', toc - tic); tic = time.time()

    os.makedirs(output_dir, exist_ok = True)
    
    vis_instance_grids = plot_instance_grids(os.path.basename(input_path), output_dir, grid, img_rgbhwc_255, beginboxind, endboxind, plane_ids if vis_instance_grids else [], algo, boxes_xywh, regions, reg_lab) if vis_instance_grids in [True, None] else []

    vis_merging_segments = plot_merging_segments(os.path.basename(input_path), output_dir, grid, plane_ids, algo, boxes_xywh, regions, reg_lab) if vis_merging_segments else []
    
    vis_merging_trees = plot_merging_trees(os.path.basename(input_path), output_dir, plane_ids, algo, boxes_xywh, regions, reg_lab) if vis_merging_trees else []

    print('vis time', time.time() - tic, end = '\n\n')

    return repr(print), (vis_instance_grids[0][0] if vis_instance_grids else None), vis_instance_grids[1:], vis_merging_segments, vis_merging_trees

def plot_instance_grids(basename, output_dir, grid, img_rgbhwc_255, beginboxind, endboxind, plane_ids, algo, boxes_xywh, regions, reg_lab):
    res = []
    for plane_id in ['all'] + plane_ids:
        regs = [reg for reg in sum(regions, []) if reg['plane_id'] == plane_id or plane_id == 'all']
        suffix = ''.join(map(str, plane_id))
        output_path = os.path.join(output_dir, 'png_' + basename + f'.{suffix}.png')

        fig = plt.figure(figsize = (grid, grid))
        plt.subplot(grid, grid, 1)
        plt.imshow(img_rgbhwc_255, aspect = 'auto')
        plt.axis('off')
        for reg in regs[beginboxind : beginboxind + endboxind]:
            plt.gca().add_patch(matplotlib.patches.Rectangle(reg['bbox_xywh'][:2], *reg['bbox_xywh'][2:], linewidth = 1, edgecolor = 'r', facecolor = 'none'))

        for k, reg in enumerate(regs[:grid * grid - 1]):
            plt.subplot(grid, grid, 2 + k)
            mask = algo.get_region_mask(reg_lab, [reg] )[0]
            
            plt.imshow((torch.as_tensor(img_rgbhwc_255) * mask[..., None] + torch.as_tensor(img_rgbhwc_255) * mask[..., None].logical_not() // 10).to(torch.uint8), aspect = 'auto')
            plt.gca().add_patch(matplotlib.patches.Rectangle(reg['bbox_xywh'][:2], *reg['bbox_xywh'][2:], linewidth = 1, edgecolor = 'r', facecolor = 'none'))
            plt.axis('off')

        plt.subplots_adjust(0, 0, 1, 1, wspace = 0, hspace = 0)
        plt.savefig(output_path)
        plt.close(fig)
        print(output_path)

        res.append((output_path, suffix)) 
    return res


def plot_merging_trees(basename, output_dir, plane_ids, algo, boxes_xywh, regions, reg_lab):
    res = []
    for plane_id in plane_ids:
        regs = [reg for reg in sum(regions, []) if reg['plane_id'] == plane_id]
        suffix = ''.join(map(str, plane_id))
        output_path = os.path.join(output_dir, 'dot_' + basename + f'.{suffix}.dot')
        
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
        dot_cmd = ['dot', '-Tpng', output_path, '-o']
        output_path += '.png'
        dot_cmd.append(output_path)
        try:
            print(' '.join(dot_cmd), '#', subprocess.check_call(dot_cmd))
            res.append((output_path, suffix))
            print(output_path)
        except Exception as e:
            print(' '.join(dot_cmd), '# dot cmd failed', e)

    return res

def plot_merging_segments(basename, output_dir, grid, plane_ids, algo, boxes_xywh, regions, reg_lab):
    res = []
    for plane_id in plane_ids:
        regs = [reg for reg in sum(regions, []) if reg['plane_id'] == plane_id]
        suffix = ''.join(map(str, plane_id))
        output_path = os.path.join(output_dir, 'gif_' + basename + f'.{suffix}.gif')
        
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

        res.append((output_path, suffix))
    return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--selectivesearchsegmentation_opencv_custom_so', default = 'opencv_custom/selectivesearchsegmentation_opencv_custom_.so', help = 'NumPy will print warning "UserWarning: The value of the smallest subnormal for <class numpy.float64 type is zero" becasue the .so is built with -Ofast which setz FTZ flush-to-zero CPU flag, details at https://moyix.blogspot.com/2022/09/someones-been-messing-with-my-subnormals.html')
    parser.add_argument('--gradio', action = 'store_true')
    parser.add_argument('--vis-instance-grids', action = 'store_true')
    parser.add_argument('--vis-merging-segments', action = 'store_true')
    parser.add_argument('--vis-merging-trees', action = 'store_true')
    parser.add_argument('--input-path', '-i', default = './examples/astronaut.jpg')
    parser.add_argument('--output-dir', '-o', default = './out')
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
            gradio.Image(label = 'input image', value = args.input_path),
            gradio.Radio(label = 'preset', choices = ['fast', 'quality', 'single'], value = 'fast'),
            gradio.Radio(label = 'algo', choices = ['pytorch', 'opencv', 'opencv_custom'], value = 'pytorch'),
            gradio.CheckboxGroup(label = 'visualizations', choices = ['instance grids', 'merging segments', 'merging trees']),
            gradio.Checkbox(label = 'remove duplicate boxes'),
            gradio.Checkbox(label = 'profile'),
            gradio.Number(label = 'beginboxind', minimum = 0, precision = 0, value = 0),
            gradio.Number(label = 'endboxind', minimum = 0, precision = 0, value = 64),
            gradio.Number(label = 'grid', minimum = 0, precision = 0, value = 4),
            gradio.Number(label = 'seed', minimum = 0, precision = 0, value = 42),
        ]
        gradio_outputs = [
            gradio.Textbox(label = 'log'),
            gradio.Image(label = 'top boxes'),
            gradio.Gallery(label = 'instance grids'),
            gradio.Gallery(label = 'merging segments'),
            gradio.Gallery(label = 'merging trees')
        ]
        gradio_submit = lambda img_rgbhwc_255, preset, algo, visualizations, remove_duplicate_boxes, profile, beginboxind, endboxind, grid, seed: main(img_rgbhwc_255, **dict(vars(args), preset = preset, algo = algo, remove_duplicate_boxes = remove_duplicate_boxes, profile = profile, beginboxind = beginboxind, endboxind = endboxind, seed = seed, grid = grid, vis_instance_grids = True if 'instance grids' in visualizations else None, vis_merging_segments = algo != 'opencv' and 'merging segments' in visualizations, vis_merging_trees = algo != 'opencv' and 'merging trees' in visualizations))
        gradio_demo = gradio.Interface(gradio_submit, inputs = gradio_inputs, outputs = gradio_outputs, flagging_dir = args.output_dir, allow_flagging = 'never')
        gradio_demo.launch()
    else:
        img_rgbhwc_255 = plt.imread(args.input_path).copy()
        if args.algo == 'opencv':
            print('disabling [vis_merging_segments] and [vis_merging_trees] for [opencv]')
            args.vis_merging_segments = args.vis_merging_trees = False
        main(img_rgbhwc_255 = img_rgbhwc_255, **vars(args))
