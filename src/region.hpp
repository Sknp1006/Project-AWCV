#pragma once

#ifndef H_AWCV_REGION
#define H_AWCV_REGION

#include "core.hpp"
#include <map>
#include <opencv2/core/types.hpp>

namespace awcv
{

#pragma region Region
class Region
{
  public:
    Region() {};
    explicit Region(const cv::Mat& InMat, const cv::Point2f& Centroid);
    explicit Region(const cv::Mat& InMat): Region(InMat, cv::Point2f(0.0f, 0.0f)) {}
    ~Region();

    cv::Size getMatSize() const;          // 获取原始图像尺寸
    cv::Mat getRegion();                  // 获取区域
    double getRegionArea();               // 获取区域面积
    cv::Point2f getCentroid() const;      // 获取区域质心
    cv::Rect getBoundingRect();           // 获取区域的外接矩形
    cv::RotatedRect getMinBoundingRect(); // 获取区域的最小外接矩形
    double getMinBoundingRectArea();      // 获取最小外接矩形面积
  private:
    int width{0};
    int height{0};

    std::vector<std::vector<cv::Point>> contours{}; // 区域轮廓
    cv::Point2f _Centroid{};                        // 区域质心
    cv::Mat _Region{};                              // 掩膜区域CV_8UC1
    cv::Rect _BoundingRect{};                       // 外接矩形
    cv::RotatedRect _MinBoundingRect{};             // 最小外接矩形

    double RegionArea{};                            // 区域面积
    double MinBoundingRectArea{};                   // 最小外接矩形面积
};
#pragma endregion

// 分割连通域
std::map<int, Region> connection(const cv::Mat &ThresMat);
// 获取最大的连通域
Region getMaxAreaRegion(std::map<int, Region>& Regions);
std::map<int, Region> filterRegionByArea(std::map<int, awcv::Region>& Regions, float MinArea, float MaxArea = 99999.0f);
// 计算区域质心
std::vector<cv::Point2f> calcCentroid(const std::vector<std::vector<cv::Point>>& Contours);
} // namespace awcv
#endif