import os
import math
import argparse
import colorsys

import numpy as np
import matplotlib, matplotlib.pyplot as plt
import cv2

def edgesSpatialGrads(img, mode = 'canny'):
    assert img.dtype == np.uint8 and img.ndim == 3 and img.shape[-1] == 3

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray_blur = cv2.GaussianBlur(img_gray, (3,3), SigmaX = 0, SigmaY = 0)
    
    if mode == 'canny':
        return cv2.Canny(image = img_gray_blur, threshold1 = 100, threshold2 = 200)
    
    if mode == 'sobel':
        #https://docs.opencv.org/3.4/d2/d2c/tutorial_sobel_derivatives.html
        #https://learnopencv.com/edge-detection-using-opencv/
        grad_x = cv2.Sobel(img_gray_blur, ddepth = -1, dx = 1, dy = 0, ksize = 3, scale = 1, delta = 0, borderType = cv2.BORDER_DEFAULT)
        grad_y = cv2.Sobel(img_gray_blur, ddepth = -1, dx = 0, dy = 1, ksize = 3, scale = 1, delta = 0, borderType = cv2.BORDER_DEFAULT)
        return np.abs(grad_x) + np.abs(grady_y)

def computeOrientation(E, gradientNormalizationRadius = 4, eps = 1e-5):
    assert E.dtype == np.float32 and E.ndim == 2
    
    nrml = (gradientNormalizationRadius + 1.0) ** 2
    kernelXY = np.array([(i + 1) / nrml for i in range(1 + gradientNormalizationRadius)] + [(i + 1) / nrml for i in range(0 + gradientNormalizationRadius)][::-1])
    E_conv = cv2.sepFilter2D(E, ddepth = -1, kernelX = kernelXY, kernelY = kernelXY)
    
    Oxx = cv2.Sobel(E_conv, ddepth = -1, dx = 2, dy = 0)
    Oxy = cv2.Sobel(E_conv, ddepth = -1, dx = 1, dy = 1)
    Oyy = cv2.Sobel(E_conv, ddepth = -1, dx = 0, dy = 2)
    
    xysign = -np.sign(Oxy)
    arctan = np.arctan(Oyy * xysign / (Oxx + eps))
    res = np.fmod(np.where(arctan > 0, arctan, arctan + math.pi), math.pi)

    return res
 
def edgesNms(E, O, r = 2):
    assert E.dtype == np.float32 and E.ndim == 2
    assert O.dtype == np.float32 and O.ndim == 2
    
    res = E.copy()
    Ocos, Osin = np.cos(O), np.sin(O)

    ds = sorted(set(range(-r, r + 1)) - set([0]))

    for y in range(E.shape[0]):
        for x in range(E.shape[1]):
            e = E[y, x]
            if e == 0:
                continue

            (coso, sino) = (Ocos[y, x], Osin[y, x])
            for d in ds:
                ydsin = max(0, min(y + d * sino, E.shape[0] - 1.001))
                xdcos = max(0, min(x + d * coso, E.shape[1] - 1.001))
                
                # bilinear interpolation
                (x0, y0) = (int(xdcos), int(ydsin))
                (x1, y1) = (x0 + 1, y0 + 1)
                (dx0, dy0) = (xdcos - x0, ydsin - y0)
                (dx1, dy1) = (1 - dx0, 1 - dy0)
                ed =  E[y0, x0] * dx1 * dy1 + E[y0, x1] * dx0 * dy1 + E[y1, x0] * dx1 * dy0 + E[y1, x1] * dx0 * dy0

                if e < ed:
                    res[y, x] = 0
                    break

    return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sed-model-path', default = 'examples/model.yml.gz')
    parser.add_argument('--input-path', '-i', default = 'examples/astronaut.jpg')
    parser.add_argument('--output-dir', '-o', default = './out')
    parser.add_argument('--topk', type = int, default = 32)
    parser.add_argument('--colorwheel-grid', type = int, default = 64)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok = True)
    output_path = os.path.join(args.output_dir, os.path.basename(args.input_path))

    img_rgbhw3_255 = plt.imread(args.input_path)
    img_rgbhw3_1 = np.divide(img_rgbhw3_255, 255, dtype = 'float32')

    edges_sed = cv2.ximgproc.createStructuredEdgeDetection(args.sed_model_path).detectEdges(img_rgbhw3_1) # orimap = sed.computeOrientation(edges) # edges_nms  = sed.edgesNms(edges, orimap)
    orimap_sed = computeOrientation(edges_sed)
    edgesnms_sed  = edgesNms(edges_sed, orimap_sed)

    edges_canny = np.divide(edgesSpatialGrads(img_rgbhw3_255), 255, dtype = 'float32')
    orimap_canny = computeOrientation(edges_canny)
    edgesnms_canny  = edgesNms(edges_canny, orimap_canny)
    
    plt.figure(figsize = (16, 6))

    for r, (title, img, edges, edgesnms, orimap) in enumerate([('sed', img_rgbhw3_1, edges_sed, edgesnms_sed, orimap_sed), ('canny', img_rgbhw3_1, edges_canny, edgesnms_canny, orimap_canny)]):

        edgeboxes = cv2.ximgproc.createEdgeBoxes()
        edgeboxes.setMaxBoxes(args.topk)
        boxes_xywh, boxes_objn = edgeboxes.getBoundingBoxes(edgesnms, orimap)

        print('img_rgbhw3_255:', img.shape, img.dtype, '[', img.min((0, 1)), ',', img.max((0, 1)), ']')
        print('edges:', edges.shape, edges.dtype, '[', edges.min(), ',', edges.max(), ']', 'edgesnms:', edgesnms.shape, edgesnms.dtype)
        print('orimap:', orimap.shape, orimap.dtype, '[', orimap.min(), ',', orimap.max(), ']')
        print('boxes_xywh:', boxes_xywh.shape, boxes_xywh.dtype, '[', boxes_xywh.min(0), boxes_xywh.max(0), ']', 'boxes_objn:', boxes_objn.shape, boxes_objn.dtype, '[', boxes_objn.min(), ',', boxes_objn.max(), ']')

        
        plt.subplot(2, 4, 4 * r + 1)
        plt.title('#boxes = {topk}'.format(topk = args.topk))
        plt.imshow(img)
        colormap = matplotlib.colormaps['jet']
        for (x, y, w, h), o in zip(boxes_xywh, boxes_objn):
            plt.gca().add_patch(matplotlib.patches.Rectangle((x, y), w, h, linewidth = 1, edgecolor = colormap( float(o - boxes_objn.min()) / boxes_objn.ptp() ), facecolor = 'none'))
        plt.axis('off')

        plt.subplot(2, 4, 4 * r + 2)
        plt.title('edges ' + title)
        plt.imshow(edges)
        plt.colorbar()
        plt.clim([0, 1])
        plt.axis('off')

        plt.subplot(2, 4, 4 * r + 3)
        plt.title('edgesnms')
        plt.imshow(edgesnms)
        plt.colorbar()
        plt.clim([0, 1])
        plt.axis('off')

        plt.subplot(2, 4, 4 * r + 4)
        plt.title('orimap_hsv')
        xx, yy = np.meshgrid(np.linspace(-1.0, 1.0, args.colorwheel_grid), np.linspace(-1.0, 1.0, args.colorwheel_grid))
        #TODO: real orimap is from 0 to pi + colorwheel is yy < 0
        H = (np.arctan2(xx, yy) + math.pi) / (math.pi * 2)
        S = np.sqrt(xx ** 2 + yy ** 2)
        colorwheel = matplotlib.colors.hsv_to_rgb(np.dstack([H, S, np.where((S < 1) & (S > 0.8) & (yy < 0), 1, 0) ]))
        orimap_hsv = matplotlib.colors.hsv_to_rgb(np.dstack([orimap / math.pi, np.full_like(orimap, 1.0), np.full_like(orimap, 1.0)])) * edges[..., None] + 1 * (1 - edges[..., None])
        orimap_hsv[:colorwheel.shape[0], :colorwheel.shape[1]] = colorwheel
        plt.imshow(orimap_hsv)
        plt.axis('off')

    plt.savefig(output_path, dpi = 300)

    print(output_path)
