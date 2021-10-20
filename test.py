import selectivesearchsegmentation
import selectivesearchsegmentation_opencv_custom

import argparse
import matplotlib, matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--input-path', '-i')
parser.add_argument('--output-path', '-o')
parser.add_argument('--topk', type = int, default = 64)
parser.add_argument('--preset', choices = ['fast', 'quality', 'single'], default = 'fast')
parser.add_argument('--algo', choices = ['pytorch', 'opencv', 'opencv_custom'], default = 'pytorch')
parser.add_argument('--selectivesearchsegmentation_opencv_custom_so', 'opencv/selectivesearchsegmentation_opencv_custom.so')
parser.add_argument('--seed', type = int, default = 42)
parser.add_argument('--begin', type = int, default = 0)
parser.add_argument('--grid', type = int, default = 4)
parser.add_argument('--plane-id', type = int, nargs = 4, default = [0, 0, 0, 0])
args = parser.parse_args()

random.seed(args.seed)

img = plt.imread(args.input_path).copy()

if args.algo in ['pytorch', 'opencv_custom']:
    algo = selectivesearchsegmentation.SelectiveSearch(preset = args.preset) if args.algo == 'pytorch' else selectivesearchsegmentation_opencv_custom.SelectiveSearchOpenCVCustom(preset = args.preset, lib_path = args.selectivesearchsegmentation_opencv_custom_so)
    boxes_xywh, regions, reg_lab = algo(torch.as_tensor(img).movedim(-1, -3).unsqueeze(0) / 255.0)
    mask = lambda k: algo.get_region_mask(reg_lab, [regions[0][k]])[0]
    boxes_xywh = boxes_xywh[0]
    key_level = lambda reg: reg['level']
    level2regions = lambda plane_id: {k : list(g) for k, g in itertools.groupby(sorted([reg for reg in regions[0] if reg['plane_id'] == tuple(plane_id)], key = key_level), key = key_level)}

else:
    algo = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    algo.setBaseImage(img[..., [2, 1, 0]])
    dict(fast = algo.switchToSelectiveSearchFast, quality = algo.switchToSelectiveSearchQuality, single = algo.switchToSingleStrategy)[args.preset]()
    boxes_xywh = algo.process()
    
    level2regions = lambda plane_id: []
    def mask(k):
        x, y, w, h = boxes_xywh[k]
        res = torch.zeros(img.shape[:-1])
        res[y : y + h, x : x + w] = 1
        return res

max_num_segments = 1 + (max(reg['id'] for reg in regions[0]) if regions else 0)
l2r = level2regions(plane_id = args.plane_id)
reg_lab_ = reg_lab[tuple(args.plane_id)[:-1]].clone()

print('boxes', len(boxes_xywh))
print('height:', img.shape[0], 'width:', img.shape[1])
print('ymax', max(y + h - 1 for x, y, w, h in boxes_xywh), 'xmax', max(x + w - 1 for x, y, w, h in boxes_xywh))

fig = plt.figure(figsize = (args.grid, args.grid))
plt.subplot(args.grid, args.grid, 1)
plt.imshow(img, aspect = 'auto')
plt.axis('off')
for x, y, w, h in boxes_xywh[args.begin : args.begin + args.topk]:
    plt.gca().add_patch(matplotlib.patches.Rectangle((x, y), w, h, linewidth = 1, edgecolor = 'r', facecolor = 'none'))

for k in range(args.grid * args.grid - 1):
    plt.subplot(args.grid, args.grid, 2 + k)
    m = mask(args.begin + k).to(torch.float32)
    x, y, w, h = boxes_xywh[k]
    plt.imshow((img * m[..., None].numpy() + img * (1 - m[..., None].numpy()) // 10).astype('uint8'), aspect = 'auto')
    plt.gca().add_patch(matplotlib.patches.Rectangle((x, y), w, h, linewidth = 1, edgecolor = 'r', facecolor = 'none'))
    plt.axis('off')

plt.subplots_adjust(0, 0, 1, 1, wspace = 0, hspace = 0)
plt.savefig(args.output_path)
plt.close(fig)

if not l2r:
    pass

fig = plt.figure(figsize = (args.grid, args.grid))
fig.set_tight_layout(True)
#fig.subplots_adjust(0, 0, 1, 1, wspace = 0, hspace = 0)

def update(level, im = []):
    for reg in l2r[level]:
        min_id = min(reg['ids'])
        for id in reg['ids']:
            reg_lab_[reg_lab_ == id] = min_id
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
