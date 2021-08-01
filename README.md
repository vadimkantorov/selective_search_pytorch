Selective Search **[Uijlings13]** reimplementation in Python / PyTorch that allows to extract not only the bounding boxes, but also the region masks.

Feature computation and region merging loop is done in Python / PyTorch. The underlying graph segmentation **[Felzenszwalb2004]** is still OpenCV's [cv2.ximgproc.segmentation.createGraphSegmentation](https://docs.opencv.org/master/d5/df0/group__ximgproc__segmentation.html#ga5e3e721c5f16e34d3ad52b9eeb6d2860).

This reimplementation follows the OpenCV's [cv2.ximgproc.segmentation.createSelectiveSearchSegmentation](https://github.com/opencv/opencv_contrib/blob/master/modules/ximgproc/src/selectivesearchsegmentation.cpp). This reimplementation is slower than the original C++ (mainly due to region merging loop in Python, and PyTorch's slower computation of histogram distances), but allows for simpler experimentation.

Most of implementation should be running fine on GPU, but it is not tested yet (and the graph segmentation currently runs only on CPU anyway).

### Usage
```shell
pip install opencv-python opencv-contrib-python

# download astronaut test image https://www.flickr.com/photos/nasacommons/16504233985/ small size
curl https://live.staticflickr.com/8674/16504233985_9f1060624e_w_d.jpg > astronaut.jpg

python3 selectivesearchsegmentation.py -i astronaut.jpg -o test.png

# open test.png and test.png.gif
```
### test.png
![test.png](https://user-images.githubusercontent.com/1041752/127772794-9be6ec05-55cc-4787-99e7-ee31926e41c0.png)

### test.png.gif
![test.png.gif](https://user-images.githubusercontent.com/1041752/127770399-e0133e08-0f48-44ea-8648-19ac7594556c.gif)

### TODO
- implement more straightforward HoG descriptor instead of gaussian derivatives with explicit rotation as in:
    - https://github.com/saisrivatsan/selective-search/blob/dc6f327fd58757099c69c9f76d247b4a16be962a/hist.py#L13

- implement color space treatment as in:
    - https://github.com/smanenfr/rp/blob/4bef556593781ef3866f646531d4e854fea0dcd8/src/rp.h#L40

- simplify graph construction as in:
    - https://github.com/belltailjp/selective_search_py/blob/dbf916129b8cfe6431288079f2246b61e3c7d820/selective_search.py#L26

- implement local binary pattern as in:
    - https://github.com/AlpacaDB/selectivesearch/blob/develop/selectivesearch/selectivesearch.py#L115

- switch to [skimage.segmentation.felzenszwalb](https://github.com/scikit-image/scikit-image/blob/main/skimage/segmentation/_felzenszwalb.py#L7-L75) or [FelzenSegment](https://github.com/smanenfr/rp/tree/master/src/FelzenSegment)

- compute color / texture histograms in uint8
- replace with uint8 histogram computation: https://github.com/pytorch/pytorch/issues/61819 https://github.com/pytorch/pytorch/issues/32306 https://github.com/pytorch/pytorch/issues/43867 
- replace histogram distance to be in uint8 when uint8 aggregation is available: https://github.com/pytorch/pytorch/issues/55366

### References
- https://github.com/AlpacaDB/selectivesearch/, https://github.com/vsakkas/selective-search
- https://github.com/belltailjp/selective_search_py
- https://github.com/saisrivatsan/selective-search
- https://github.com/ChenjieXu/selective_search
- https://github.com/smanenfr/rp/

### Credits
I copied some Kornia's color conversion utils: https://kornia.readthedocs.io/en/latest/color.html

```
@article{Uijlings13,
  author = {J.R.R. Uijlings and K.E.A. van de Sande and T. Gevers and A.W.M. Smeulders},
  title = {Selective Search for Object Recognition},
  journal = {International Journal of Computer Vision},
  year = {2013},
  url = {https://ivi.fnwi.uva.nl/isis/publications/2013/UijlingsIJCV2013/UijlingsIJCV2013.pdf}
}

@article{Felzenszwalb2004,
  author = {Felzenszwalb, Pedro F. and Huttenlocher, Daniel P.},
  title = {Efficient Graph-Based Image Segmentation},
  journal = {International Journal of Computer Vision},
  year = {2004},
  url = {http://people.cs.uchicago.edu/pff/papers/seg-ijcv.pdf}
}

@inproceedings{Manen2013,
  author = {Manen, Santiago and Guillaumin, Matthieu and Van Gool, Luc},
  title = {Prime Object Proposals with Randomized Prim's Algorithm},
  booktitle = {Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
  year = {2013},
  url = {https://openaccess.thecvf.com/content_iccv_2013/papers/Manen_Prime_Object_Proposals_2013_ICCV_paper.pdf}
}
```
