//--------------------------------------------------------------------------------------------------------------------------------------
#pragma once // 防止重复编译
#include <vector>
#ifndef H_AWCV_CORE
#define H_AWCV_CORE

#include <NumCpp.hpp>
#include <opencv2/opencv.hpp>

#include "dft.hpp"

typedef awcv::DFTMAT DFTMAT;

// 通用图像处理方法
namespace awcv
{

enum DecomTypes
{
    DECOM_RGB, // 按RGB拆分通道
    DECOM_HSV, // 按HSV拆分通道
};

enum RegionGrowingFlags
{
    FILLMODE_SIMPLE = 0,
    FILLMODE_FixedRange = 1,
    FILLMODE_Gradient = 2,
};
// 连通域模式
enum ConnectivityMode
{
    Connectivity_4 = 4,
    Connectivity_8 = 8,
};

void bgr2gray(cv::Mat InMat, cv::Mat &OutMat);                                                     // 转灰度图
void resize(cv::Mat &InOutMat, double Ratio);                                                      // 按比例改变图像大小
void resize(cv::Mat &InOutMat, cv::Size Size);                                                     // 等比例改变图像大小
void threshold(const cv::Mat &InMat, cv::Mat &OutMat, double MinGray, double MaxGray);             // 根据区间二值化
void scaleImage(cv::Mat InMat, cv::Mat &OutMat, double MinGray, double MaxGray);                   // 缩放图像灰度值
void zoomGray(cv::Mat &InOutMat, int MaxGrayLevel);                                                // 灰度映射
void gammaImage(cv::Mat InMat, cv::Mat &OutMat, float Gamma);                                      // 伽马矫正
void autoGammaImage(cv::Mat InMat, cv::Mat &OutMat, float C);                                      // 自动伽马矫正
void linearGrayLevelTrans(cv::Mat InMat, cv::Mat &OutMat, int Th1, int Th2, int Goal1, int Goal2); // 灰度线性拉升
void logImage(cv::Mat InMat, cv::Mat &OutMat, float Const);                                        // 灰度对数变换
void equalizeColor(const cv::Mat &InMat, cv::Mat &OutMat);                                         // 色彩均衡
void bilateralFilter(const cv::Mat& InMat, cv::Mat& OutMat, int Iter, int D, int SColor, int SSpace);//多重双边滤波
/*	void watershedsThreshold(	cv::Mat InMat,
                                cv::Mat& OutMat,
                                int Threshold);
 */
// 分水岭分割法
// 区域生长法
void regionGrowingWithSeed( cv::Mat InMat, 
                            cv::Point Seed,  
                            cv::Mat &Mask, 
                            int LoDiff = 30,
                            int UpDiff = 40,
                            int FfillMode = 1, 
                            int Connectivity = 4);      //使用一个种子点的区域生长法
void regionGrowingWithROI(	cv::Mat InMat, 
                            cv::Rect ROI, 
                            cv::Mat& Mask, 
                            int LoDiff = 30,
                            int UpDiff = 40,
                            int FfillMode = 1, 
                            int Connectivity = 4);		//使用roi区域的区域生长法
void regionGrowing(         cv::Mat InMat, 
                            std::vector<cv::Point> SeedPoints, 
                            cv::Mat& Mask, 
                            int LoDiff = 30,
                            int UpDiff = 40,
                            int FfillMode = 1, 
                            int Connectivity = 4);
// void pyrMeanShiftFiltering();				//均值漂移

// 拆分图像通道
void decompose3(const cv::Mat &InMat, std::vector<cv::Mat> &OutArray, DecomTypes type);
// 高斯差分算法（DOG）
void diffOfGaussian(const cv::Mat &InMat, cv::Mat &OutMat, cv::Size KSize = cv::Size(3, 3), double Sigma = 0.3, double SigFactor = 1.5);
// 提取LBP纹理
void LBP(cv::Mat InMat, cv::Mat &OutMat);
// 高斯背景估计法
void estimateBackgroundIllumination(cv::Mat InMat, cv::Mat &OutMat);
// 图片颜色翻转
void invertImage(cv::Mat InMat, cv::Mat &OutMat);
// gabor滤波器
void gaborFilter(cv::Mat InMat, cv::Mat &OutMat, int KernelSize, double Sigma, double Theta, double Lambd, double Gamma, double Psi);
// 加权最小二乘滤波（未实现）
void wlsFilter(cv::Mat InMat, cv::Mat &OutMat, float Sigma, float Lambda, int SolverIteration);
// 图像增强（基于平均值）
void enhanceImageByMean(const cv::Mat &InMat, cv::Mat &OutMat, double M1, double M2);
// 图像增强（基于OTSU）
void enhanceImageByOTSU(const cv::Mat &InMat, cv::Mat &OutMat, int &Thres);

//生成直方图
cv::Mat getHistImage(const cv::Mat &InMat);
//--------------------------------------------------------------------------------------------------------------------------------------
// 下面是一些通用类的声明
//--------------------------------------------------------------------------------------------------------------------------------------
#pragma region GLCM
class GLCM
{
  public:
    class GLCMDATA
    {
      public:
        GLCMDATA()
            : MaxProbability(0.0),
              AngularSecondMoment(0.0),
              Contrast(0.0),
              Correlation(0.0),
              Entropy(0.0),
              Homogeneity(0.0)
        {
        }
        ~GLCMDATA()
        {
        }
        double MaxProbability;      // 最大概率
        double AngularSecondMoment; // 能量/二阶距/均匀性
        double Contrast;            // 对比度/反差
        double Correlation;         // 相关性/自相关性
        double Entropy;             // 熵
        double Homogeneity;         // 逆差距/同质性
    };

    // Gray Level CO-Occurrence Matrix
    GLCM(){};
    ~GLCM()
    {
        this->glcmdata.clear();
        MaxGrayLevel = 0;
    };
    void setMaxGrayLevel(int GrayLevel); // 【第一步】设置最大灰度级
    void calcGLCM(cv::Mat InMat);        // 【第二步】计算灰度共生矩阵与特征描述子
    cv::Mat getGLCMFeatures();           // 【第三步】获得统计结果
  private:
    std::vector<GLCMDATA> glcmdata;      // 分别是0, 45, 90, 135角度的特征描述子
    int MaxGrayLevel = 64;
    GLCMDATA CalcGLCMDATA(cv::Mat GLCMMAT);

    cv::Mat glcm_0;
    cv::Mat glcm_45;
    cv::Mat glcm_90;
    cv::Mat glcm_135;
};
#pragma endregion

void calcHessianMatrix(const cv::Mat &InMat, cv::Mat &Hxx, cv::Mat &Hyy, cv::Mat &Hxy);     // 计算图像的二阶偏导数
}; // namespace awcv

//--------------------------------------------------------------------------------------------------------------------------------------
// cv::Mat转nc::NdArray
//--------------------------------------------------------------------------------------------------------------------------------------
template <typename T = double>
nc::NdArray<T> Mat2NdArray(cv::Mat InMat)
{
    int rows = InMat.rows;
    int cols = InMat.cols;
    nc::NdArray<T> array = nc::zeros<T>(nc::Shape(rows, cols * InMat.channels())); // 单通道

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            switch (InMat.channels())
            {
            case 1:
                if (InMat.type() == CV_8U)
                    array.at(i, j) = InMat.at<uchar>(i, j);
                else if (InMat.type() == CV_32F || InMat.type() == CV_64F)
                    array.at(i, j) = InMat.at<double>(i, j);
                else if (InMat.type() == CV_32S)
                    array.at(i, j) = InMat.at<int>(i, j);
                break;
            case 3:
                if (InMat.type() == CV_8UC3)
                {
                    array.at(i, j * 3 + 0) = InMat.at<cv::Vec3b>(i, j)[0];
                    array.at(i, j * 3 + 1) = InMat.at<cv::Vec3b>(i, j)[1];
                    array.at(i, j * 3 + 2) = InMat.at<cv::Vec3b>(i, j)[2];
                }
                else if (InMat.type() == CV_32FC3)
                {
                    array.at(i, j * 3 + 0) = InMat.at<cv::Vec3f>(i, j)[0];
                    array.at(i, j * 3 + 1) = InMat.at<cv::Vec3f>(i, j)[1];
                    array.at(i, j * 3 + 2) = InMat.at<cv::Vec3f>(i, j)[2];
                }
                else if (InMat.type() == CV_64FC3)
                {
                    array.at(i, j * 3 + 0) = InMat.at<cv::Vec3d>(i, j)[0];
                    array.at(i, j * 3 + 1) = InMat.at<cv::Vec3d>(i, j)[1];
                    array.at(i, j * 3 + 2) = InMat.at<cv::Vec3d>(i, j)[2];
                }
                break;
            default:
                break;
            }
        }
    }
    return array;
}
//--------------------------------------------------------------------------------------------------------------------------------------
// nc::NdArray转cv::Mat
//--------------------------------------------------------------------------------------------------------------------------------------
template <typename T = double>
cv::Mat NdArray2Mat(nc::NdArray<T> InArray, int Channel = 1)
{
    cv::Mat mat;
    int rows = 0;
    int cols = 0;
    if (Channel == 1)
    {
        rows = InArray.shape().rows;
        cols = InArray.shape().cols;
        mat = cv::Mat::zeros(cv::Size(cols, rows), CV_64F);
    }
    else if (Channel == 3)
    {
        rows = InArray.shape().rows;
        cols = InArray.shape().cols / 3;
        mat = cv::Mat::zeros(cv::Size(cols, rows), CV_64FC3);
    }

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            switch (Channel)
            {
            case 1:
                mat.at<double>(i, j) = InArray.at(i, j);
                break;
            case 3:
                mat.at<cv::Vec3d>(i, j)[0] = InArray.at(i, j * 3 + 0);
                mat.at<cv::Vec3d>(i, j)[1] = InArray.at(i, j * 3 + 1);
                mat.at<cv::Vec3d>(i, j)[2] = InArray.at(i, j * 3 + 2);
                break;
            default:
                break;
            }
        }
    }
    return mat;
}

#endif