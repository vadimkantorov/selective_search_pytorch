The original OpenCV ximgproc code that doesn't require installing opencv-python:

```shell
conda install -c conda-forge opencv

wget https://raw.githubusercontent.com/opencv/opencv_contrib/master/modules/ximgproc/src/precomp.hpp
wget https://raw.githubusercontent.com/opencv/opencv_contrib/master/modules/ximgproc/include/opencv2/ximgproc/segmentation.hpp
wget https://raw.githubusercontent.com/opencv/opencv_contrib/master/modules/ximgproc/src/selectivesearchsegmentation.cpp
wget https://raw.githubusercontent.com/opencv/opencv_contrib/master/modules/ximgproc/src/graphsegmentation.cpp

diff selectivesearchsegmentation.cpp selectivesearchsegmentation_.cpp

make selectivesearchsegmentation.so
```
