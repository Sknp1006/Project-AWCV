#pragma once

#ifndef H_AWCV_DFT
#define H_AWCV_DFT

#include "core.hpp"

namespace awcv
{
enum FilterTypes // 滤波器分类
{
    FILTER_ILPF, // 理想低通滤波器
    FILTER_BLPF, // 巴特沃斯低通滤波器
    FILTER_GLPF, // 高斯低通滤波器

    FILTER_IHPF, // 理想高通滤波器
    FILTER_BHPF, // 巴特沃斯高通滤波器
    FILTER_GHPF, // 高斯高通滤波器

    FILTER_IBPF, // 理想带通滤波器
    FILTER_BBPF, // 巴特沃斯带通滤波器
    FILTER_GBPF, // 高斯带通滤波器

    FILTER_IBRF, // 理想带通滤波器
    FILTER_BBRF, // 巴特沃斯带通滤波器
    FILTER_GBRF, // 高斯带通滤波器
};
// struct DFTMAT // 用于计算的DFTMAT结构体
// {
//     DFTMAT()
//     {
//     }
//     DFTMAT(cv::Mat Img, cv::Mat ComplexM)
//     {
//         this->img = Img.clone();
//         this->complexM = ComplexM.clone();
//     }
//     cv::Mat img;      // 可显示的频谱图
//     cv::Mat complexM; // 用于计算的复数矩阵
// };

class DFTMAT
{
public:
    DFTMAT() {}
    DFTMAT(cv::Mat InMat, cv::Mat ComplexMat)
    {
        this->mat = InMat.clone();
        this->complexMat = ComplexMat.clone();
    }
    void setMat(const cv::Mat &InMat) 
    { 
        this->mat = InMat.clone(); }
    cv::Mat getMat() const { return this->mat; }

    // 检查复数矩阵是否有两个通道，以便正确设置
    static bool isComplexMatValid(const cv::Mat& InMat)
    {
        return InMat.channels() == 2 && (InMat.type() == CV_8UC2 || InMat.type() == CV_32FC2);
    }

    // 可能会抛出无效参数异常
    void setComplexMat(const cv::Mat& InMat) 
    {
        if (!isComplexMatValid(InMat))
            throw std::invalid_argument("Complex matrix must have two channels and either 8-bit unsigned or 32-bit floating point type");

        this->complexMat = InMat.clone();
    }

    cv::Mat getComplexMat() const { return this->complexMat; }
private:
    cv::Mat mat;        // 可显示的频谱图
    cv::Mat complexMat; // 用于计算的复数矩阵
};

// 离散傅里叶变换
void DFT(cv::Mat InMat, DFTMAT &Out);
// 离散傅里叶逆变换
void IDFT(DFTMAT InMat, DFTMAT &Out);
// 频域卷积(对频域使用滤波器)
void convolDFT(DFTMAT InDft, cv::Mat Filter, DFTMAT &OutDft);
// 优化傅里叶图像尺寸
void optimalDFTSize(cv::Mat &InOutMat, int &Width, int &Height);
// 生成带通滤波器
void genBandpassFilter();
// 生成高通滤波器
void genHighpassFilter(cv::Mat &HFilter,
                       cv::Size Size,
                       cv::Point Center,
                       float Radius,
                       int Type,
                       int n = -1);
// 生成低通滤波器
void genLowpassFilter(cv::Mat &LFilter,
                      cv::Size Size,
                      cv::Point Center,
                      float Radius,
                      int Type,
                      int n = -1);
// 图像中心化
void dftShift(cv::Mat &InOutMat);
// 高斯背景估计法
void estimateBackgroundIllumination(cv::Mat InMat, cv::Mat &OutMat);
} // namespace awcv

#endif
