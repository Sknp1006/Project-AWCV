#pragma once
#ifndef H_AWCV_OPTICALFLOW
#define H_AWCV_OPTICALFLOW

#include <NumCpp.hpp>
#include <opencv2/core.hpp>
#include <opencv2/optflow.hpp>

namespace awcv
{
namespace opt
{
enum OFType // 稠密光流法类型
{
    DENSE_DIS,
    DENSE_PCA,
    DENSE_RLOF,
    DENSE_DEFAULT
};
struct OFAttr // 光流法属性
{
    cv::Mat temp_prev_mat, lk_prev_mat;      // 稠密、稀疏光流场:前一张图像（基准图像）
    cv::Mat temp_now_mat, lk_now_mat;        // 稀疏、稠密光流场:当前图像
    std::vector<cv::Point2f> lk_prev_points; // 稀疏光流场:前一张图像的人脸关键点
    std::vector<cv::Point2f> lk_now_points;  // 稀疏光流场:当前图像的人脸关键点
};
//--------------------------------------------------------------------------------------------------------------------------------------
//													DenseOpticalFlow类
//--------------------------------------------------------------------------------------------------------------------------------------
class DenseOpticalFlow // 稠密光流法集合
{
  public:
    DenseOpticalFlow(OFType OfType);
    ~DenseOpticalFlow();

    void calcOpticalFlow(cv::Mat Frame, cv::Mat &Flow); // 【第一步】计算稠密光流场
    void calcOpticalFlowMat(cv::Mat Flow,
                            cv::Mat &FLOWIMG,
                            cv::Mat &Mangitude,
                            cv::Mat &Angle); // 【第二步】计算可视化光流图像
    void clear();                            // 手动清除光流矩阵
  private:
    OFAttr __ofAttr; // 光流数据结构体

    void initDISOF(); // 初始化光流法
    void initFarnebackOF();
    void initPCAOF();
    void initRLOFOF();

    cv::Ptr<cv::DISOpticalFlow> DISOF;                 // DIS稠密光流（精度一般，参数可调）
    cv::Ptr<cv::FarnebackOpticalFlow> FarnebackOF;     // Farneback稠密光流（慢）
    cv::optflow::OpticalFlowPCAFlow PCAOF;             // PCA稠密光流（效果不错，但是慢）
    cv::Ptr<cv::optflow::DenseRLOFOpticalFlow> RLOFOF; // RLOF稠密光流（慢，不会调参）
};
//--------------------------------------------------------------------------------------------------------------------------------------
//													SparseOpticalFlow类
//--------------------------------------------------------------------------------------------------------------------------------------
class SparseOpticalFlow
{
  public:
    SparseOpticalFlow();
    ~SparseOpticalFlow();

    void calcOpticalFlowLK(cv::Mat Frame,
                           std::vector<cv::Point2f> Pts,
                           std::vector<cv::Point2f> &EstimatePts,
                           cv::Mat &Status,
                           cv::Mat &Error); // 计算面部稀疏光流场
    void clear();                           // 手动清除光流矩阵和点
  private:
    OFAttr __ofAttr; // 光流数据结构体
};
} // namespace opt
} // namespace awcv

#endif // !H_CORE_OPTICALFLOW
