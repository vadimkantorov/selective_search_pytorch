#include <cstdint>
#include <cstring>
#include <cassert>
#include <algorithm>

#include <opencv2/core.hpp>

#include "segmentation.hpp"

#include "graphsegmentation.cpp"

#include "edgeboxes.cpp"

#include "selectivesearchsegmentation_.cpp"

extern "C" int process(
    uint8_t* img_ptr, int32_t img_rows, int32_t img_cols,
    int32_t* rect_ptr, int32_t* rect_rows, 
    int32_t* seg_ptr, int32_t* seg_channels, 
    int32_t* reg_ptr, 
    uint8_t* bit_ptr, int32_t bit_cols,
    const char* strategy, int32_t base_k, int32_t inc_k, float sigma
)
{
    cv::ximgproc::segmentation::SelectiveSearchSegmentationImpl algo;
    algo.setBaseImage(cv::Mat(img_rows, img_cols, CV_8UC3, img_ptr));
    
    if(0 == strcmp("single", strategy))
        (base_k > 0 && sigma > 0) ? algo.switchToSingleStrategy(base_k, sigma) : algo.switchToSingleStrategy();
    if(0 == strcmp("fast", strategy))
        (base_k > 0 && inc_k > 0 && sigma > 0) ? algo.switchToSelectiveSearchFast(base_k, inc_k, sigma) : algo.switchToSelectiveSearchFast();
    if(0 == strcmp("quality", strategy))
        (base_k > 0 && inc_k > 0 && sigma > 0) ? algo.switchToSelectiveSearchQuality(base_k, inc_k, sigma) : algo.switchToSelectiveSearchQuality();

    std::vector<cv::Rect> rects;
    algo.process(rects);
    
    assert(*rect_rows >= rects.size());
    for(size_t c = 0; c < rects.size(); c++)
    {
        cv::Rect& rect = rects[c];
        rect_ptr[4 * c + 0] = rect.x;
        rect_ptr[4 * c + 1] = rect.y;
        rect_ptr[4 * c + 2] = rect.width;
        rect_ptr[4 * c + 3] = rect.height;
    }
    *rect_rows = rects.size();
    
    assert(*seg_channels >= algo.all_img_regions.size());
    for(size_t c = 0; c < algo.all_img_regions.size(); c++)
    {
        cv::Mat& img = algo.all_img_regions[c];
        memcpy(seg_ptr + c * img.total(), &img.data[0], img.total() * img.elemSize());
    }
    *seg_channels = algo.all_img_regions.size();

    assert(bit_cols >= MAX_NUM_BIT_BYTES);
    for(size_t c = 0, k = 0; c < algo.all_regions.size(); c++)
    {
        cv::ximgproc::segmentation::Region& region = algo.all_regions[c];
        if(region.used)
        {
            reg_ptr[3 * k + 0] = region.id;
            reg_ptr[3 * k + 1] = region.level;
            reg_ptr[3 * k + 2] = region.image_id;
            memcpy(bit_ptr + k * bit_cols, &region.bit.data[0], region.bit.cols);
            k++;
        }
    }
    
    return 0;
}
