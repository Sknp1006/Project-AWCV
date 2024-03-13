#include "region.hpp"
#include <memory>
#include <opencv2/core/types.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#pragma region Region
/// @brief Region类构造函数
/// @param InMat 输入区域(CV_8UC1)
/// @param Centroid 输入区域质心
awcv::Region::Region(const cv::Mat& InMat, const cv::Point2f& Centroid)
{
    if (InMat.type() != CV_8UC1)
    {
        // TODO:只能包含0和255值
        CV_Error(cv::Error::StsBadArg, "输入的Region或Centroid不合法。");
    }
    this->width = InMat.cols;
    this->height = InMat.rows;
    // this->_Region = InMat.clone();                                                                                // ROI区域大小
    std::vector<cv::Vec4i> hierarchy;                                                                                // 轮廓层级
    // cv::findContours(this->_Region, this->contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point()); // 计算区域轮廓
    cv::findContours(InMat, this->contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_NONE, cv::Point()); // 计算区域轮廓

    if ((Centroid.x == 0.0f) && (Centroid.y == 0.0f))
    {
        auto Centroids = awcv::calcCentroid(this->contours);    // 计算质心
        this->_Centroid = Centroids[0];
    }
    else
    {
        this->_Centroid = Centroid;                 // 初始化：质心（从外接输入）
    }
        this->_BoundingRect = cv::Rect();           // 初始化：外接矩形
        this->_MinBoundingRect = cv::RotatedRect(); // 初始化：最小外接矩形

        this->RegionArea = 0.0;          // 初始化：区域面积
        this->MinBoundingRectArea = 0.0; // 初始化：最小外接矩形面积
}
/// @brief Region类析构函数
awcv::Region::~Region() {}
/// @brief 获取区域大小
/// @return 区域大小
cv::Size awcv::Region::getMatSize() const { return cv::Size(this->width, this->height); }
/// @brief 获取区域图像
/// @return 区域图像
cv::Mat awcv::Region::getRegion()
{
    if (this->_Region.empty())
    {
        this->_Region = cv::Mat::zeros(this->height, this->width, CV_8UC1);
        cv::drawContours(this->_Region, this->contours, -1, cv::Scalar::all(255), -1);
    }
    return this->_Region;
}
/// @brief 获取区域面积
/// @return 区域面积
double awcv::Region::getRegionArea()
{
    if (this->RegionArea != 0)
    {
        return this->RegionArea;
    }
    else
    {
        this->RegionArea = cv::contourArea(this->contours[0]);
        return this->RegionArea;
    }
}
/// @brief 获取区域质心
/// @return 区域质心
cv::Point2f awcv::Region::getCentroid() const
{
    return this->_Centroid;
}
/// @brief 获取区域的外接矩形
/// @return 区域外接矩形
cv::Rect awcv::Region::getBoundingRect()
{
    if (!this->_BoundingRect.empty())
    {
        return this->_BoundingRect;
    }
    else
    {
        // 计算外接矩形
        this->_BoundingRect = cv::boundingRect(this->contours[0]);
        return this->_BoundingRect;
    }
}
/// @brief 获取区域的最小外接矩形
/// @return 区域最小外接矩形
cv::RotatedRect awcv::Region::getMinBoundingRect()
{
    if (_MinBoundingRect.size.width != 0 && _MinBoundingRect.size.height != 0)
    {
        return _MinBoundingRect;
    }
    else
    {
        this->_MinBoundingRect = cv::minAreaRect(contours[0]); // 连通域的最小外接矩形
        return this->_MinBoundingRect;
    }
}
/// @brief 获取区域的最小外接矩形面积
/// @return 区域最小外接矩形的面积
double awcv::Region::getMinBoundingRectArea()
{
    if (this->MinBoundingRectArea != 0)
    {
        return this->MinBoundingRectArea;
    }
    else
    {
        cv::Mat boxPts;
        cv::boxPoints(this->getMinBoundingRect(), boxPts);   // 获得外接矩形的四个顶点
        this->MinBoundingRectArea = cv::contourArea(boxPts); // 最小外接矩形的面积
        return this->MinBoundingRectArea;
    }
}
#pragma endregion
/// @brief 连通域分割
/// @param ThresMat 输入二值化图像
/// @return 输出连通域字典
std::map<int, awcv::Region> awcv::connection(const cv::Mat &ThresMat)
{
    std::map<int, Region> Regions;
    if (ThresMat.type() != CV_8UC1)
    {
        CV_Error(cv::Error::StsBadArg, "输入的ThresMat不是二值化图像。");
    }
    // Labels:		输出标签矩阵(32S,int)
    // Stats:		输出统计矩阵(32S,int)
    // Centroids:	输出质心矩阵(64F,double)
    cv::TickMeter tm;
    cv::Mat Labels, Stats, Centroids;
    int RegionNum = cv::connectedComponentsWithStats(ThresMat, Labels, Stats, Centroids);       // 2160x3840 耗时: 10ms
    if (RegionNum > 1)
    {
        // Eigen::MatrixXi LabelsMat(Labels.rows, Labels.cols);
        std::shared_ptr<Eigen::MatrixXi> LabelsMat = std::make_shared<Eigen::MatrixXi>(Labels.rows, Labels.cols);
        cv::cv2eigen(Labels, (*LabelsMat));
        for (int i = 1; i < RegionNum; i++)
        {
            // 单次连通域分割：45ms
            {
            // nc::NdArray<int> LabelsArray = Mat2NdArray<int>(Labels);     // labels转NdArray
            // nc::NdArray<int> mask = nc::zeros<int>(LabelsArray.shape()); // 空矩阵
            // mask.putMask(LabelsArray == i, 255);                         // 将对应连通域标记为255
            // cv::Mat connectedRegion = NdArray2Mat(mask);
            
            // Eigen::MatrixXi mask = Eigen::MatrixXi::Zero(Labels.rows, Labels.cols);
            // mask = (LabelsMat.array() == i).cast<int>() * 255;      // 将对应连通域标记为255
            // // std::cout << "mask: " << mask << std::endl;
            // cv::Mat connectedRegion;
            // cv::eigen2cv(mask, connectedRegion);                    // 将Eigen矩阵转换为OpenCV矩阵
            // connectedRegion.convertTo(connectedRegion, CV_8U);      // 第i个连通域Mat
            }

            std::shared_ptr<Eigen::MatrixXi> mask = std::make_shared<Eigen::MatrixXi>(Labels.rows, Labels.cols);
            (*mask).setZero();
            (*mask) = ((*LabelsMat).array() == i).cast<int>() * 255;                 // 将对应连通域标记为255
            cv::Mat connectedRegion;
            cv::eigen2cv((*mask), connectedRegion);                         // 将Eigen矩阵转换为OpenCV矩阵
            connectedRegion.convertTo(connectedRegion, CV_8U);              // 第i个连通域Mat

            cv::Point2f centroid = cv::Point2f(static_cast<float>(Centroids.at<double>(i, 0)), static_cast<float>(Centroids.at<double>(i, 1)));
            Region Aregion = Region(connectedRegion, centroid);           // 区域类：3ms
            Regions.insert(std::pair<int, Region>(i, Aregion));         // 连通域从1开始计算
        }
        return Regions;
    }
    return Regions;
}
/// @brief 获取最大的连通域
/// @param Regions 连通域字典
/// @return 面积最大的连通域
awcv::Region awcv::getMaxAreaRegion(std::map<int, awcv::Region>& Regions)
{
    int MaxIndex = 0;
    double MaxArea = 0;
    for (std::map<int, awcv::Region>::iterator R = Regions.begin(); R != Regions.end(); R++)
    {
        int Index = R->first;
        awcv::Region region = R->second;

        double area = region.getRegionArea();
        if (area > MaxArea)
        {
            MaxArea = area;
            MaxIndex = Index;
        }
    }
    return Regions.at(MaxIndex);
}
/// @brief 按照面积筛选区域
/// @param Regions 连通域字典
/// @param MinArea 最小面积
/// @param MaxArea 最大面积
/// @return 满足条件的连通域
std::map<int, awcv::Region> awcv::filterRegionByArea(std::map<int, awcv::Region>& Regions, float MinArea, float MaxArea)
{
    std::map<int, awcv::Region> _Regions;
    for (std::map<int, awcv::Region>::iterator R = Regions.begin(); R != Regions.end(); R++)
    {
        int Index = R->first;
        awcv::Region region = R->second;

        double area = region.getRegionArea();
        if ((area >= MinArea) && (area <= MaxArea))
        {
            _Regions.insert(std::pair<int, awcv::Region>(Index, region));
        }
    }
    Regions.clear();
    return _Regions;
}
/// @brief 计算区域质心
/// @param Contours 区域轮廓
/// @return 质心集合
std::vector<cv::Point2f> awcv::calcCentroid(const std::vector<std::vector<cv::Point>>& Contours)
{
    // 5、计算每个轮廓所有矩
    std::vector<cv::Moments> mu(Contours.size());       // 创建一个vector,元素个数为contours.size()
    for( int i = 0; i < Contours.size(); i++ )
    {
        mu[i] = moments(Contours[i], false);            // 获得轮廓的所有最高达三阶所有矩
    }
    // 6、计算轮廓的质心
    std::vector<cv::Point2f> mc(Contours.size());
    for( int i = 0; i < Contours.size(); i++ )
    {
        mc[i] = cv::Point2f(static_cast<float>(mu[i].m10/mu[i].m00), static_cast<float>(mu[i].m01/mu[i].m00));   // 质心的 X,Y 坐标：(m10/m00, m01/m00)
    }
    return mc;
}
