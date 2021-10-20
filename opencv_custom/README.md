The original OpenCV ximgproc code that doesn't require installing opencv-python:

```shell
conda install -c conda-forge opencv

wget https://raw.githubusercontent.com/opencv/opencv_contrib/master/modules/ximgproc/src/precomp.hpp
wget https://raw.githubusercontent.com/opencv/opencv_contrib/master/modules/ximgproc/include/opencv2/ximgproc/segmentation.hpp
wget https://raw.githubusercontent.com/opencv/opencv_contrib/master/modules/ximgproc/src/selectivesearchsegmentation.cpp
wget https://raw.githubusercontent.com/opencv/opencv_contrib/master/modules/ximgproc/src/graphsegmentation.cpp

export OPENCVINCLUDEDIR=/path/to/lib
export OPENCVLIBDIR=/path/to/include/opencv4
make selectivesearchsegmentation_opencv_custom.so
```

```diff
diff selectivesearchsegmentation.cpp selectivesearchsegmentation_.cpp
45a46,47
> #define MAX_NUM_BIT_BYTES 64
>
67a70,73
>
>                     int image_id;
>                     bool used;
>                     Mat bit;
734a741,744
>
>                 public:
>                     std::vector<Region> all_regions;
>                     std::vector<Mat> all_img_regions;
888,889c898,899
<
<                 std::vector<Region> all_regions;
---
>                 all_regions.clear();
>                 all_img_regions.clear();
901a912
>                         all_img_regions.push_back(img_regions);
951a963
>                                 region->image_id = image_id;
961c973
<
---
>
967a980
>                     region->used = false;
968a982
>                         region->used = true;
973d986
<
994a1008,1013
>                     r.bit = Mat::zeros(1, MAX_NUM_BIT_BYTES, CV_8UC1);
>
>                     int bit_idx = i / 8;
>                     uint32_t bit_set = uint8_t(1) << (7 - (i % 8));
>                     assert(bit_idx < MAX_NUM_BIT_BYTES);
>                     r.bit.data[bit_idx] = bit_set;
1028a1048
>                     cv::bitwise_or(region_from.bit, region_to.bit, new_r.bit);
```
