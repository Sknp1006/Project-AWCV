#include "dft.hpp"
//--------------------------------------------------------------------------------------------------------------------------------------
// 功能:优化图像大小
// 参数:
//          InMat:          输入输出图像
//          Width:          输出优化的宽度
//          Height:         输出优化的高度
//--------------------------------------------------------------------------------------------------------------------------------------
void awcv::optimalDFTSize(cv::Mat &InOutMat, int &Width, int &Height)
{
    int w = InOutMat.cols;
    int h = InOutMat.rows;
    Width = cv::getOptimalDFTSize(InOutMat.cols);
    Height = cv::getOptimalDFTSize(InOutMat.rows);
    cv::copyMakeBorder(InOutMat, InOutMat, 0, Height - h, 0, Width - w, cv::BORDER_CONSTANT, cv::Scalar::all(0));
}
//--------------------------------------------------------------------------------------------------------------------------------------
// 功能:离散傅里叶变换
// 参数:
//          InMat:          输入图像（8UC1、32FC1）
//          OutDft;         输出DFT结构体
//--------------------------------------------------------------------------------------------------------------------------------------
void awcv::DFT(cv::Mat InMat, DFTMAT &OutDft)
{
    awcv::dftShift(InMat); // 先原图Shift到中心
    cv::Mat mag[2], complexI;
    cv::Mat planes[] = {cv::Mat_<float>(InMat), cv::Mat::zeros(InMat.size(), CV_32F)};
    cv::merge(planes, 2, complexI);                      // 合并成复数矩阵
    cv::dft(complexI, complexI, cv::DFT_COMPLEX_OUTPUT); // 傅里叶变换（输出复数矩阵）
    cv::split(complexI, mag);
    cv::magnitude(mag[0], mag[1], mag[0]);
    cv::divide(mag[0], mag[0].cols * mag[0].rows, mag[0]);
    OutDft = DFTMAT(mag[0], complexI);
}
//--------------------------------------------------------------------------------------------------------------------------------------
// 功能:离散傅里叶逆变换
// 参数:
//          InMat:          输入图像（8UC2、32FC2）
//          OutMat;         输出DFT结构体
//--------------------------------------------------------------------------------------------------------------------------------------
void awcv::IDFT(DFTMAT InDft, DFTMAT &OutMat)
{
    cv::Mat mag[2];
    cv::idft(InDft.complexM, OutMat.complexM);
    cv::split(OutMat.complexM, mag);
    cv::magnitude(mag[0], mag[1], mag[0]);
    cv::normalize(mag[0], OutMat.img, 1, 0, cv::NORM_MINMAX);
    OutMat.img.convertTo(OutMat.img, CV_8U, 255, 0);
}
//--------------------------------------------------------------------------------------------------------------------------------------
// 功能:对频域使用滤波器
// 参数:
//          InDft:          输入DFT结构体
//          Filter:         输入滤波器
//          OutDft:         输出DFT结构体
//--------------------------------------------------------------------------------------------------------------------------------------
void awcv::convolDFT(DFTMAT InDft, cv::Mat Filter, DFTMAT &OutDft)
{
    // 《数字图像处理》步骤，实际不需要扩充
    // int InMat_Width = InMat.cols;
    // int InMat_Height = InMat.rows;
    // cv::copyMakeBorder(InMat, InMat, 0, InMat_Height, 0, InMat_Width, cv::BORDER_CONSTANT, cv::Scalar::all(0));//【第一步】将图像扩充为原来的2倍
    // if (InMat.size() != Filter.size()) return;                                  //计算失败
    // dftShift(InMat);                                                            //【第二步】将零频点移到频谱的中间
    // cv::Mat dft; awcv::DFT(InMat, dft);                                         //【第三步】计算图像的DFT
    // dft = dft.mul(Filter);                                                      //【第四步】矩阵与滤波器对应相乘
    // cv::imshow("inmat", dft);
    // cv::Mat idft; awcv::IDFT(dft, idft);                                        //【第五步】傅里叶逆变换
    // OutMat = idft(cv::Rect(0, 0, InMat_Width, InMat_Height)).clone();           //【第六步】取左上象限的区域
    cv::Mat planes[2];
    cv::split(InDft.complexM, planes);
    planes[0] = planes[0].mul(Filter);
    planes[1] = planes[1].mul(Filter);
    cv::merge(planes, 2, OutDft.complexM);
}
//--------------------------------------------------------------------------------------------------------------------------------------
// 功能:生成带通滤波器
// 参数:
//          InMat:          输入图像（8UC2、32FC2）
//          OutMat;         输出图像(
//--------------------------------------------------------------------------------------------------------------------------------------
void awcv::genBandpassFilter()
{
}
//--------------------------------------------------------------------------------------------------------------------------------------
// 功能:生成高通滤波器
// 参数:
//          Filter:         输出滤波器
//          Size:           滤波器尺寸
//          Center:         频谱中心
//          Radius:         截止频率
//          Type:           滤波器类型
//          n:              巴特沃斯系数
//--------------------------------------------------------------------------------------------------------------------------------------
void awcv::genHighpassFilter(cv::Mat &HFilter, cv::Size Size, cv::Point Center, float Radius, int Type, int n)
{
    HFilter = cv::Mat::zeros(Size, CV_32FC1);
    int rows = Size.height;
    int cols = Size.width;
    if (Radius <= 0)
        return;
    for (int r = 0; r < rows; r++)
    {
        for (int c = 0; c < cols; c++)
        {
            double Dist = sqrt(pow(float(r - Center.y), 2.0) + pow(float(c - Center.x), 2.0));
            if (Type == FILTER_IHPF) // 理想高通滤波器
            {
                if (Dist <= Radius)
                    HFilter.at<float>(r, c) = 0;
                else
                    HFilter.at<float>(r, c) = 1;
            }
            else if (Type == FILTER_BLPF) // 巴特沃斯高通滤波器
            {
                if (n < 0)
                    return;
                HFilter.at<float>(r, c) = float(1.0 / (1.0 + pow(Radius / Dist, 2.0 * n)));
            }
            else if (Type == FILTER_GHPF) // 高斯高通滤波器
            {
                HFilter.at<float>(r, c) = float(1 - exp(-(pow(Dist, 2) / (2 * pow(Radius, 2.0)))));
            }
        }
    }
}
//--------------------------------------------------------------------------------------------------------------------------------------
// 功能:生成低通滤波器
// 参数:
//          Filter:         输出滤波器
//          Size:           滤波器尺寸
//          Center:         频谱中心
//          Radius:         截止频率
//          Type:           滤波器类型
//          n:              巴特沃斯系数
//--------------------------------------------------------------------------------------------------------------------------------------
void awcv::genLowpassFilter(cv::Mat &LFilter, cv::Size Size, cv::Point Center, float Radius, int Type, int n)
{
    LFilter = cv::Mat::zeros(Size, CV_32FC1);
    int rows = Size.height;
    int cols = Size.width;
    if (Radius <= 0)
        return;
    for (int r = 0; r < rows; r++)
    {
        for (int c = 0; c < cols; c++)
        {
            double Dist = sqrt(pow(float(r - Center.y), 2.0) + pow(float(c - Center.x), 2.0));
            if (Type == FILTER_ILPF) // 理想低通滤波器
            {
                if (Dist <= Radius)
                    LFilter.at<float>(r, c) = 1;
                else
                    LFilter.at<float>(r, c) = 0;
            }
            else if (Type == FILTER_BLPF) // 巴特沃斯低通滤波器
            {
                if (n < 0)
                    return;
                LFilter.at<float>(r, c) = float(1.0 / (1.0 + pow(Dist / Radius, 2.0 * n)));
            }
            else if (Type == FILTER_GLPF) // 高斯低通滤波器
            {
                LFilter.at<float>(r, c) = float(exp(-(pow(Dist, 2) / (2 * pow(Radius, 2.0)))));
            }
        }
    }
}
//--------------------------------------------------------------------------------------------------------------------------------------
// 功能:将零频点移到频谱的中间
// 参数:
//          InOutMat:       输入输出图像(填充后的图像)
//--------------------------------------------------------------------------------------------------------------------------------------
void awcv::dftShift(cv::Mat &InOutMat)
{
    if (InOutMat.type() != CV_32FC1)
        InOutMat.convertTo(InOutMat, CV_32FC1);
    for (int i = 0; i < InOutMat.rows; i++)
    {
        float *p = InOutMat.ptr<float>(i);
        for (int j = 0; j < InOutMat.cols; j++)
        {
            p[j] = p[j] * pow(-1, i + j);
        }
    }
}
