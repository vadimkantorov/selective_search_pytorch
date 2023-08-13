Modified OpenCV `cv::ximgproc::segmentation` code that doesn't require installing opencv-python and binds a small shared library via `ctypes`:

```shell
# working sequence to install opencv and torchvision is at https://github.com/pytorch/vision/issues/4665#issuecomment-947545121

# conda install -c conda-forge opencv
# sudo apt-get install libopencv-dev

# can try
make selectivesearchsegmentation_opencv_custom_.so 

# or specify the paths to discover the include/opencv4 dir and the dir that has libopencv_core.so
# g++ -m64 -xc++ --verbose -E - < /dev/null 2>&1 | grep '^ .*include.*' | xargs find | grep 'opencv2/core.hpp'
# g++ -print-file-name=libopencv_core.so
# g++ -m64 -Xlinker --verbose -lopencv_core -lopencv_imgproc 2>&1 | grep '.*opencv.* succeeded'
# make selectivesearchsegmentation_opencv_custom_.so OPENCVINCLUDEDIR=/path/to/lib/dir/that/contains/libopencv_core.so OPENCVLIBDIR=/path/to/include/dir/that/is/named/opencv4
``` 

```shell
# files below were downloaded and not modified (except one commented line in `precomp.hpp`):
wget https://raw.githubusercontent.com/opencv/opencv_contrib/71f9dbd144c0e32b87746843d4497aea0562a1fe/modules/ximgproc/src/precomp.hpp
wget https://raw.githubusercontent.com/opencv/opencv_contrib/71f9dbd144c0e32b87746843d4497aea0562a1fe/modules/ximgproc/include/opencv2/ximgproc/segmentation.hpp
wget https://raw.githubusercontent.com/opencv/opencv_contrib/71f9dbd144c0e32b87746843d4497aea0562a1fe/modules/ximgproc/src/selectivesearchsegmentation.cpp
wget https://raw.githubusercontent.com/opencv/opencv_contrib/71f9dbd144c0e32b87746843d4497aea0562a1fe/modules/ximgproc/src/graphsegmentation.cpp
wget https://raw.githubusercontent.com/opencv/opencv_contrib/71f9dbd144c0e32b87746843d4497aea0562a1fe/modules/ximgproc/src/edgeboxes.cpp
wget https://raw.githubusercontent.com/opencv/opencv_contrib/71f9dbd144c0e32b87746843d4497aea0562a1fe/modules/ximgproc/include/opencv2/ximgproc/edgeboxes.hpp
```

```diff
diff selectivesearchsegmentation.cpp selectivesearchsegmentation_.cpp
45a46,47
> #define MAX_NUM_BIT_BYTES 64
> 
67a70,75
> 
>                     int image_id;
>                     bool used;
>                     
>                     int idx;
>                     Mat bit;
734a743,746
>                     
>                 public:
>                     std::vector<Region> all_regions;
>                     std::vector<Mat> all_img_regions;
888,889c900,901
< 
<                 std::vector<Region> all_regions;
---
>                 all_regions.clear();
>                 all_img_regions.clear();
901a914
>                         all_img_regions.push_back(img_regions);
951a965
>                                 region->image_id = image_id;
961c975
< 
---
>                 
967a982
>                     region->used = false;
968a984
>                         region->used = true;
973d988
< 
994a1010,1016
>                     
>                     r.idx = i;
>                     r.bit = Mat::zeros(1, MAX_NUM_BIT_BYTES, CV_8UC1);
>                     int byte_idx = i / 8;
>                     uint8_t bit_set = uint8_t(1) << (7 - (i % 8));
>                     assert(byte_idx < MAX_NUM_BIT_BYTES);
>                     r.bit.data[byte_idx] |= bit_set;
1028a1051,1053
>                     
>                     new_r.idx = (int)regions.size();
>                     cv::bitwise_or(region_from.bit, region_to.bit, new_r.bit);
```
