#pragma once

#ifndef H_AWCV_REGION
#define H_AWCV_REGION

#include "core.hpp"

namespace awcv
{

#pragma region Region
class Region
{
  public:
    Region(cv::Mat InMat, cv::Point2f Centroid);
    ~Region();

    cv::Mat getRegion();                  // 获取区域
    double getRegionArea();               // 获取区域面积
    cv::Point2f getCentroid();            // 获取区域质心
    cv::Rect getBoundingRect();           // 获取区域的外接矩形
    cv::RotatedRect getMinBoundingRect(); // 获取区域的最小外接矩形
    double getMinBoundingRectArea();      // 获取最小外接矩形面积
  private:
    std::vector<std::vector<cv::Point>> contours; // 区域轮廓

    cv::Point2f _Centroid;            // 区域质心

    cv::Mat _Region;                  // 掩膜区域CV_8UC1
    cv::Rect _BoundingRect;           // 外接矩形
    cv::RotatedRect _MinBoundingRect; // 最小外接矩形

    double RegionArea;          // 区域面积
    double MinBoundingRectArea; // 最小外接矩形面积
};
#pragma endregion

//分割连通域
std::map<int, Region> connection(cv::Mat ThresMat);
} // namespace awcv
#endif