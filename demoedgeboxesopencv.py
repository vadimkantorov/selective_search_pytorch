import os
import math
import argparse
import colorsys

import numpy as np
import matplotlib, matplotlib.pyplot as plt
import cv2

def computeOrientation(src, gradientNormalizationRadius = 4, eps = 1e-5):
    assert src.dtype == np.float32 and src.ndim == 2
    
    nrml = (gradientNormalizationRadius + 1.0) ** 2
    kernelXY = np.array([(i + 1) / nrml for i in range(1 + gradientNormalizationRadius)] + [(i + 1) / nrml for i in range(0 + gradientNormalizationRadius)][::-1])
    E_conv = cv2.sepFilter2D(src, -1, kernelXY, kernelXY)
    
    Oxx = cv2.Sobel(E_conv, -1, 2, 0)
    Oxy = cv2.Sobel(E_conv, -1, 1, 1)
    Oyy = cv2.Sobel(E_conv, -1, 0, 2)
    
    xysign = -np.sign(Oxy)
    arctan = np.arctan(Oyy * xysign / (Oxx + eps))
    dst = np.fmod(np.where(arctan > 0, arctan, arctan + math.pi), math.pi)

    return dst
 
def edgesNms(E, O, r : int = 2):
    assert E.dtype == np.float32 and E.ndim == 2
    assert O.dtype == np.float32 and O.ndim == 2
    
    Ocos = np.cos(O)
    Osin = np.sin(O)
    
    Ocos_t = Ocos.T
    Osin_t = Osin.T
    O_t = O.T
    E_t = E.T
    dst_t = np.zeros_like(E_t)

    for x in range(E.shape[1]):
        for y in range(E.shape[0]):
            e = dst_t[x, y] = E_t[x, y]
            (coso, sino) = (Ocos_t[x, y], Osin_t[x, y])
            
            if e == 0:
                continue

            for d in sorted(set(range(-r, r + 1)) - set([0])):
                xdcos = max(0, min(x+d*coso, E.shape[1] - 1.001))
                ydsin = max(0, min(y+d*sino, E.shape[0] - 1.001))
                
                (x0, y0) = (int(xdcos), int(ydsin))
                (x1, y1) = (x0 + 1, y0 + 1)
                (dx0, dy0) = (xdcos - x0, ydsin - y0)
                (dx1, dy1) = (1 - dx0, 1 - dy0)
                # bilinear interpolation
                e0 =  E_t[x0, y0] * dx1 * dy1 + E_t[x1, y0] * dx0 * dy1 + E_t[x0, y1] * dx1 * dy0 + E_t[x1, y1] * dx0 * dy0

                if e < e0:
                    dst_t[x, y] = 0
                    break

  
    return dst_t.T

#######################################3
#######################################3
#######################################3
#######################################3

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
orimap_ = computeOrientation(edges)
edges_nms  = sed.edgesNms(edges, orimap)
edges_nms_  = edgesNms(edges, orimap)
breakpoint()


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

#img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)
#sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
#sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
#sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection
#edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200) # Canny Edge Detection
  
#GaussianBlur( src, src, Size(3,3), 0, 0, BORDER_DEFAULT );
#cvtColor( src, src_gray, COLOR_BGR2GRAY );
#Mat grad_x, grad_y;
#Mat abs_grad_x, abs_grad_y;
#//Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
#Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
#//Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
#Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
#convertScaleAbs( grad_x, abs_grad_x );
#convertScaleAbs( grad_y, abs_grad_y );
#addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );
