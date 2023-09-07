#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <algorithm>

#include <opencv2/core.hpp>

#include "segmentation.hpp"

#include "graphsegmentation.cpp"

//#include "selectivesearchsegmentation.cpp"
#include "selectivesearchsegmentation_.cpp"

#include "edgeboxes.hpp"
#include "edgeboxes.cpp"


extern "C" int process(
    uint8_t* img_ptr, int32_t img_rows, int32_t img_cols,
    int32_t* rect_ptr, int32_t* rect_rows, 
    int32_t* seg_ptr, int32_t* seg_channels, 
    int32_t* reg_ptr, 
    uint8_t* bit_ptr, int32_t bit_cols,
    const char* strategy, int32_t base_k, int32_t inc_k, float sigma,
    bool remove_duplicate_boxes, int64_t seed
)
{
    if(seed >= 0)
        srand((unsigned int32_t) seed);

    cv::ximgproc::segmentation::SelectiveSearchSegmentationImpl algo;
    algo.setBaseImage(cv::Mat(img_rows, img_cols, CV_8UC3, img_ptr));
    
    if(0 == strcmp("single", strategy))
    {
        if(base_k > 0 && sigma > 0)
            algo.switchToSingleStrategy(base_k, sigma);
        else
            algo.switchToSingleStrategy();
    }
    if(0 == strcmp("fast", strategy))
    {
        if(base_k > 0 && inc_k > 0 && sigma > 0)
            algo.switchToSelectiveSearchFast(base_k, inc_k, sigma);
        else
            algo.switchToSelectiveSearchFast();
    }
    if(0 == strcmp("quality", strategy))
    {
        if(base_k > 0 && inc_k > 0 && sigma > 0)
            algo.switchToSelectiveSearchQuality(base_k, inc_k, sigma);
        else
            algo.switchToSelectiveSearchQuality();
    }

    std::vector<cv::Rect> rects;
    algo.process(rects);
    
    assert(algo.all_img_regions.size() <= *seg_channels);
    for(size_t c = 0; c < algo.all_img_regions.size(); c++)
    {
        cv::Mat& img = algo.all_img_regions[c];
        memcpy(seg_ptr + c * img.total(), &img.data[0], img.total() * img.elemSize());
    }
    *seg_channels = algo.all_img_regions.size();

    assert(bit_cols >= MAX_NUM_BIT_BYTES);
    assert((remove_duplicate_boxes ? rects.size() : algo.all_regions.size()) <= *rect_rows);
    int k = 0;
    for(size_t c = 0; c < algo.all_regions.size(); c++)
    {
        cv::ximgproc::segmentation::Region& region = algo.all_regions[c];
        cv::Rect& rect = region.bounding_box;
        if( region.used || (remove_duplicate_boxes == false) )
        {
            reg_ptr[5 * k + 0] = region.id;
            reg_ptr[5 * k + 1] = region.level;
            reg_ptr[5 * k + 2] = region.image_id;
            reg_ptr[5 * k + 3] = region.idx;
            reg_ptr[5 * k + 4] = region.merged_to;
            rect_ptr[4 * k + 0] = rect.x;
            rect_ptr[4 * k + 1] = rect.y;
            rect_ptr[4 * k + 2] = rect.width;
            rect_ptr[4 * k + 3] = rect.height;
            memcpy(bit_ptr + k * bit_cols, &region.bit.data[0], region.bit.cols);
            k++;
        }
    }
    *rect_rows = k;
    
    return 0;
}
