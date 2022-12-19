#include "region.hpp"
#pragma region Region
//--------------------------------------------------------------------------------------------------------------------------------------
// 功能:Region类构造函数
// 参数:
//            InMat:            输入区域(CV_8UC1)
//            Centroid:        输入区域质心
//--------------------------------------------------------------------------------------------------------------------------------------
awcv::Region::Region(cv::Mat InMat, cv::Point2f Centroid)
{
    if (InMat.type() != CV_8UC1)
    {
        // TODO:只能包含0和255值
        CV_Error(cv::Error::StsBadArg, "输入的Region或Centroid不合法。");
    }

    this->_Region = InMat.clone();                                                                                   // ROI区域大小
    std::vector<cv::Vec4i> hierarchy;                                                                                // 轮廓层级
    cv::findContours(this->_Region, this->contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point()); // 计算区域轮廓

    this->_Centroid = Centroid;                 // 初始化：质心
    this->_BoundingRect = cv::Rect();           // 初始化：外接矩形
    this->_MinBoundingRect = cv::RotatedRect(); // 初始化：最小外接矩形

    this->RegionArea = 0.0;          // 初始化：区域面积
    this->MinBoundingRectArea = 0.0; // 初始化：最小外接矩形面积
}
//--------------------------------------------------------------------------------------------------------------------------------------
// 功能:Region类析构函数
//--------------------------------------------------------------------------------------------------------------------------------------
awcv::Region::~Region()
{
}
//--------------------------------------------------------------------------------------------------------------------------------------
// 功能:获取区域
// 返回值:    区域图像
//--------------------------------------------------------------------------------------------------------------------------------------
cv::Mat awcv::Region::getRegion()
{
    return this->_Region;
}
//--------------------------------------------------------------------------------------------------------------------------------------
// 功能:获取区域面积
// 返回值:    区域面积
//--------------------------------------------------------------------------------------------------------------------------------------
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
//--------------------------------------------------------------------------------------------------------------------------------------
// 功能:获取区域质心
// 返回值:        区域质心
//--------------------------------------------------------------------------------------------------------------------------------------
cv::Point2f awcv::Region::getCentroid()
{
    return this->_Centroid;
}
//--------------------------------------------------------------------------------------------------------------------------------------
// 功能:获取区域的外接矩形
// 返回值:        区域外接矩形
//--------------------------------------------------------------------------------------------------------------------------------------
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
//--------------------------------------------------------------------------------------------------------------------------------------
// 功能:获取区域的最小外接矩形
// 返回值:        区域最小外接矩形
//--------------------------------------------------------------------------------------------------------------------------------------
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
//--------------------------------------------------------------------------------------------------------------------------------------
// 功能:获取区域的最小外接矩形面积
// 返回值:        区域最小外接矩形的面积
//--------------------------------------------------------------------------------------------------------------------------------------
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
//--------------------------------------------------------------------------------------------------------------------------------------
// 功能:连通域分割
// 参数:
//			ThresMat:			输入二值化图像
// 返回值:
//			std::map<int, Region>:	输出连通域字典
//--------------------------------------------------------------------------------------------------------------------------------------
std::map<int, awcv::Region> awcv::connection(cv::Mat ThresMat)
{
    std::map<int, Region> Regions;
    if (ThresMat.type() != CV_8UC1)
    {
        CV_Error(cv::Error::StsBadArg, "输入的ThresMat不是二值化图像。");
    }
    // Labels:		输出标签矩阵(32S,int)
    // Stats:		输出统计矩阵(32S,int)
    // Centroids:	输出质心矩阵(64F,double)
    cv::Mat Labels, Stats, Centroids;
    int RegionNum = cv::connectedComponentsWithStats(ThresMat, Labels, Stats, Centroids);
    if (RegionNum > 1)
    {
        for (int i = 1; i < RegionNum; i++)
        {
            nc::NdArray<int> LabelsArray = Mat2NdArray<int>(Labels); // labels转NdArray
            nc::NdArray<int> mask = nc::zeros<int>(LabelsArray.shape());   // 空矩阵
            mask.putMask(LabelsArray == i, 255);                           // 将对应连通域标记为255
            cv::Mat connectedRegion = NdArray2Mat(mask);
            connectedRegion.convertTo(connectedRegion, CV_8U); // 第i个连通域Mat

            cv::Point2f centroid = cv::Point2f(static_cast<float>(Centroids.at<double>(i, 0)), static_cast<float>(Centroids.at<double>(i, 1)));
            Region Aregion = Region(connectedRegion, centroid); // 区域类
            Regions.insert(std::pair<int, Region>(i, Aregion)); // 连通域从1开始计算
        }
        return Regions;
    }
    return Regions;
}
