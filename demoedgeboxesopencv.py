import os
import math
import argparse
import colorsys

import numpy as np
import matplotlib, matplotlib.pyplot as plt
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--sed-model-path', default = 'examples/model.yml.gz')
parser.add_argument('--input-path', '-i', default = 'examples/astronaut.jpg')
parser.add_argument('--output-dir', '-o', default = './out')
parser.add_argument('--topk', type = int, default = 32)
parser.add_argument('--colorwheel-grid', type = int, default = 64)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok = True)
output_path = os.path.join(args.output_dir, os.path.basename(args.input_path))

img_rgbhw3_1 = plt.imread(args.input_path).astype('float32') / 255.0

sed = cv2.ximgproc.createStructuredEdgeDetection(args.sed_model_path)
edges  = sed.detectEdges(img_rgbhw3_1)
orimap = sed.computeOrientation(edges)
edges_nms  = sed.edgesNms(edges, orimap)

edge_boxes = cv2.ximgproc.createEdgeBoxes()
edge_boxes.setMaxBoxes(args.topk)
boxes_xywh, boxes_objn = edge_boxes.getBoundingBoxes(edges_nms, orimap)

print('img:', img_rgbhw3_1.shape, img_rgbhw3_1.dtype)
print('edges:', edges.shape, edges.dtype, 'edges_nms:', edges_nms.shape, edges_nms.dtype)
print('orimap:', orimap.shape, orimap.dtype)
print('boxes_xywh:', boxes_xywh.shape, boxes_xywh.dtype, 'boxes_objn:', boxes_objn.shape, boxes_objn.dtype)

plt.figure(figsize = (6, 6))
plt.subplot(221)
plt.title('#boxes = {topk}'.format(topk = args.topk))
plt.imshow(img_rgbhw3_1)
colormap = matplotlib.colormaps['jet']
for (x, y, w, h), o in zip(boxes_xywh, boxes_objn):
    plt.gca().add_patch(matplotlib.patches.Rectangle((x, y), w, h, linewidth = 1, edgecolor = colormap( float(o - boxes_objn.min()) / boxes_objn.ptp() ), facecolor = 'none'))
plt.axis('off')

plt.subplot(222)
plt.title('edges')
plt.imshow(edges)
plt.colorbar()
plt.clim([0, 1])
plt.axis('off')

plt.subplot(223)
plt.title('orimap_hsv')
xx, yy = np.meshgrid(np.linspace(-1.0, 1.0, args.colorwheel_grid), np.linspace(-1.0, 1.0, args.colorwheel_grid))
H = (np.arctan2(xx, yy) + math.pi) / (math.pi * 2)
S = np.sqrt(xx ** 2 + yy ** 2)
colorwheel = matplotlib.colors.hsv_to_rgb(np.dstack([H, S, np.where((S < 1) & (S > 0.8), 1, 0) ]))
orimap_hsv = matplotlib.colors.hsv_to_rgb( np.dstack([orimap / math.pi, np.full_like(orimap, 1.0), np.full_like(orimap, 1.0)]) ) * edges[..., None] + 1 * (1 - edges[..., None])
orimap_hsv[:colorwheel.shape[0], :colorwheel.shape[1]] = colorwheel
plt.imshow(orimap_hsv)
plt.axis('off')

plt.subplot(224)
plt.title('edges_nms')
plt.imshow(edges_nms)
plt.colorbar()
plt.clim([0, 1])
plt.axis('off')

plt.savefig(output_path, dpi = 300)

print(output_path)

