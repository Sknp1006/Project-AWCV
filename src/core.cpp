#include "core.hpp"
//--------------------------------------------------------------------------------------------------------------------------------------
// 功能:BGR转灰度图
// 参数:
//          InMat:          输入输出图像
//          OutMat;         输出图像
//--------------------------------------------------------------------------------------------------------------------------------------
void awcv::bgr2gray(cv::Mat InMat, cv::Mat &OutMat)
{
    if (InMat.channels() == 1)
        OutMat = InMat.clone();
    cv::cvtColor(InMat, OutMat, cv::COLOR_BGR2GRAY);
}
//--------------------------------------------------------------------------------------------------------------------------------------
// 功能:按比例缩放图片
// 参数:
//          InMat:          输入输出图像
//          Ratio;          缩放比例
//--------------------------------------------------------------------------------------------------------------------------------------
void awcv::resize(cv::Mat &InOutMat, double Ratio)
{
    if (Ratio <= 0)
        return;
    cv::resize(InOutMat, InOutMat, cv::Size(int(InOutMat.cols * Ratio), int(InOutMat.rows * Ratio)));
}
//--------------------------------------------------------------------------------------------------------------------------------------
// 功能:等比例缩放到指定大小
// 参数:
//          InMat:          输入输出图像
//          Size;           指定图像大小
//--------------------------------------------------------------------------------------------------------------------------------------
void awcv::resize(cv::Mat &InOutMat, cv::Size Size)
{
    int height = InOutMat.rows;
    int width = InOutMat.cols;
    if (width / height >= Size.width / Size.height)
    {
        resize(InOutMat, InOutMat, cv::Size(Size.width, int(height * Size.width / width)));
    }
    else
    {
        resize(InOutMat, InOutMat, cv::Size(int(width * Size.height / height), Size.height));
    }
    // copyMakeBorder(InOutMat, InOutMat, 0, Size.height - InOutMat.rows, 0,
    // Size.width - InOutMat.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
}
//--------------------------------------------------------------------------------------------------------------------------------------
// 功能:将灰度范围映射到【1-L】
// 参数:
//          InMat:          输入输出图像
//          MaxGrayLevel:   指定最大的灰度值
//--------------------------------------------------------------------------------------------------------------------------------------
void awcv::zoomGray(cv::Mat &InOutMat, int MaxGrayLevel)
{
    if (MaxGrayLevel < 1 || MaxGrayLevel > 256)
        return;
    // if (CV_8U == CV_8UC1) std::cout << "恭喜 是真的！" << std::endl;
    if (InOutMat.type() == CV_8U)
    {
        InOutMat.convertTo(InOutMat, CV_32F);
    }
    InOutMat = ((MaxGrayLevel - 1.0f) / 255.0f * InOutMat) + 1.0f;
    if (InOutMat.type() == CV_32F)
    {
        InOutMat.convertTo(InOutMat, CV_8U);
    }
}
//--------------------------------------------------------------------------------------------------------------------------------------
// 功能:伽马矫正
// 参数:
//          InMat:          输入图像
//          OutMat:         输出图像
//          Gamma:          gamma参数
//--------------------------------------------------------------------------------------------------------------------------------------
void awcv::gammaImage(cv::Mat InMat, cv::Mat &OutMat, float Gamma)
{
    // cv::Mat InMat_ =
    // InMat.clone();//值传递的Mat，形参和实参指向同一个地址，修改形参也会修改实参
    unsigned char lut[256]; // int channels = InMat_.channels();
    for (int i = 0; i < 256; i++)
    {
        lut[i] = cv::saturate_cast<uchar>(pow((float)i / 255.0f, 1.0f / Gamma) * 255.0f);
    }
    // switch (channels)
    //{
    // case 1:
    //{
    //     cv::MatIterator_<uchar> it = InMat_.begin<uchar>();
    //     cv::MatIterator_<uchar> end = InMat_.end<uchar>();
    //     while (it != end) { *it = lut[(*it)]; it++; }
    //     break;
    // }
    // case 3:
    //{
    //     cv::MatIterator_<cv::Vec3b> it = InMat_.begin<cv::Vec3b>();
    //     cv::MatIterator_<cv::Vec3b> end = InMat_.end<cv::Vec3b>();
    //     while (it != end) { (*it)[0] = lut[(*it)[0]]; (*it)[1] = lut[(*it)[1]];
    //     (*it)[2] = lut[(*it)[2]]; it++; } break;
    // }
    // default:
    //     break;
    // }
    // OutMat = InMat_.clone();
    cv::Mat Lut(1, 256, CV_8UC1, lut); // 使用lut映射
    cv::LUT(InMat, Lut, OutMat);
}
//--------------------------------------------------------------------------------------------------------------------------------------
// 功能:自动伽马矫正
// 参数:
//          InMat:          输入图像
//          OutMat:         输出图像
//          C:              目标平均灰度值[0-1]
//--------------------------------------------------------------------------------------------------------------------------------------
void awcv::autoGammaImage(cv::Mat InMat, cv::Mat &OutMat, float C)
{
    auto meanGray = cv::mean(InMat)[0];
    float gamma_val = static_cast<float>(log10(1 - C) / log10(1 - meanGray / 255.0f)); // 自动gamma参数
    awcv::gammaImage(InMat, OutMat, gamma_val);
}
//---------------------------------------------------------------------------------------------------------------------------------------
// 功能:    线性灰度变换
// 参数:
//          InMat:          输入图像
//          OutMat:         输出图像
//          Th1:
//          Th2:
//          Goal1:
//          Goal2:
//---------------------------------------------------------------------------------------------------------------------------------------
void awcv::linearGrayLevelTrans(cv::Mat InMat, cv::Mat &OutMat, int Th1, int Th2, int Goal1, int Goal2)
{
    cv::Mat temp = InMat.clone();
    if (temp.channels() == 3)
        cv::cvtColor(temp, temp, cv::COLOR_BGR2GRAY);
    if (temp.type() != CV_32F)
        temp.convertTo(temp, CV_32F);
    for (int i = 0; i < temp.rows; i++)
    {
        for (int j = 0; j < temp.cols; j++)
        {
            if (temp.at<float>(i, j) <= Th1)
                temp.at<float>(i, j) = temp.at<float>(i, j) * Goal1 / Th1;
            else if (temp.at<float>(i, j) > Th1 && temp.at<float>(i, j) <= Th2)
                temp.at<float>(i, j) = (temp.at<float>(i, j) - Th1) * (Goal2 - Goal1) / (Th2 - Th1) + Goal1;
            else
                temp.at<float>(i, j) = (temp.at<float>(i, j) - Th2) * (255 - Goal2) / (255 - Th2) + Goal2;
        }
    }
    temp.convertTo(OutMat, CV_8U);
}
//--------------------------------------------------------------------------------------------------------------------------------------
// 功能:对数变换
// 参数:
//          InMat:          输入图像
//          OutMat:         输出图像
//          Const:          对数变换常数
//--------------------------------------------------------------------------------------------------------------------------------------
void awcv::logImage(cv::Mat InMat, cv::Mat &OutMat, float Const)
{
    // cv::Mat InMat_ = InMat.clone();
    unsigned char lut[256]; // int channels = InMat_.channels();
    for (int i = 0; i < 256; i++)
    {
        lut[i] = cv::saturate_cast<uchar>(Const * log(1 + i));
    }
    // switch (channels)
    //{
    // case 1:
    //{
    //     cv::MatIterator_<uchar> it = InMat_.begin<uchar>();
    //     cv::MatIterator_<uchar> end = InMat_.end<uchar>();
    //     while (it != end) { *it = lut[(*it)]; it++; }
    //     break;
    // }
    // case 3:
    //{
    //     cv::MatIterator_<cv::Vec3b> it = InMat_.begin<cv::Vec3b>();
    //     cv::MatIterator_<cv::Vec3b> end = InMat_.end<cv::Vec3b>();
    //     while (it != end) { (*it)[0] = lut[(*it)[0]]; (*it)[1] = lut[(*it)[1]];
    //     (*it)[2] = lut[(*it)[2]]; it++; } break;
    // }
    // default:
    //     break;
    // }
    cv::Mat Lut(1, 256, CV_8UC1, lut);
    cv::LUT(InMat, Lut, OutMat);
    // cv::normalize(OutMat, OutMat, 0, 255, cv::NORM_MINMAX);
    // cv::convertScaleAbs(OutMat, OutMat);
}
//--------------------------------------------------------------------------------------------------------------------------------------
// 功能:使用分水岭法分割图像
// 参数:
//          InMat:          输入图像(3-Channels)
//          OutMat:         输出图像
//          Threshold:      分割阈值
//--------------------------------------------------------------------------------------------------------------------------------------
// void awcv::watershedsThreshold(cv::Mat InMat, cv::Mat& OutMat, int Threshold)
//{
//    if (InMat.channels() != 3)
//    {
//        cv::cvtColor(InMat, InMat, cv::COLOR_GRAY2BGR);
//    }
//
//    if (Thres1 > Thres2 || Thres1 < 0) return;
//    cv::Mat gray; cv::cvtColor(InMat, gray, cv::COLOR_BGR2GRAY);
//    cv::GaussianBlur(gray, gray, cv::Size(5, 5), 0.0f, 0.0f);
//    cv::adaptiveThreshold(gray, gray, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
//    cv::THRESH_BINARY, 5, 0); cv::Canny(gray, gray, Thres1, Thres2);
//
//    std::vector<std::vector<cv::Point>> contours;
//    std::vector<cv::Vec4i> hierarchy;
//    //cv::findContours(gray, contours, hierarchy, cv::RETR_TREE,
//    cv::CHAIN_APPROX_SIMPLE, cv::Point()); cv::findContours(gray, contours,
//    hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, cv::Point());
//    //cv::Mat imageContours = cv::Mat::zeros(InMat.size(), CV_8UC1); //轮廓
//    cv::Mat markers(InMat.size(), CV_32S, cv::Scalar::all(0));
//    //初始化分水岭法markers
//
//
//    int index = 0;
//    int compCount = 0;
//    for (; index >= 0; index = hierarchy[index][0], compCount++)
//    {
//        //对marks进行标记，对不同区域的轮廓进行编号，相当于设置注水点，有多少轮廓，就有多少注水点
//
//        drawcvContours(markers, contours, index, cv::Scalar::all(compCount +
//        1), 1, 8, hierarchy);
//    }
//    cv::watershed(InMat, markers);              //输入彩色图像
//    cv::convertScaleAbs(markers, OutMat);
//}
//--------------------------------------------------------------------------------------------------------------------------------------
// 功能:拆分图片的通道
// 参数:
//          InMat:          输入图像
//          OutArray:       输出颜色通道数组
//          type:           要拆分的通道类型（RGB、HSV）
//--------------------------------------------------------------------------------------------------------------------------------------
void awcv::decompose3(cv::Mat InMat, std::vector<cv::Mat> &OutArray, DecomTypes Type)
{
    if (InMat.channels() != 3)
        return; // 不是三通道
    cv::Mat temp = InMat.clone();
    switch (Type)
    {
    case DECOM_RGB:
        cv::cvtColor(temp, temp, cv::COLOR_BGR2RGB);
        cv::split(temp, OutArray); // RGB
        break;
    case DECOM_HSV:
        cv::cvtColor(temp, temp, cv::COLOR_BGR2HSV);
        cv::split(temp, OutArray); // HSV
        break;
    default:
        return;
    }
}
//--------------------------------------------------------------------------------------------------------------------------------------
// 功能:高斯差分算法
// 参数:
//          InMat:          输入图像
//          OutMat:         输出图像
//          KSize           输入KernelSize
//          Sigma:          输入Sigma参数
//          SigFactor:      输入SigFactor参数
//--------------------------------------------------------------------------------------------------------------------------------------
void awcv::diffOfGaussian(cv::Mat InMat, cv::Mat &OutMat, cv::Size KSize, double Sigma, double SigFactor)
{
    // assert(SigFactor > 0.0f && Sigma > 0.0f); //数据约束
    if (!(SigFactor > 0.0f) || !(Sigma > 0.0f))
        return;
    cv::Mat temp_Out1, temp_Out2;
    double sigma1, sigma2;
    sigma1 = Sigma / sqrt(-2 * (log(1 / SigFactor)) / (pow(SigFactor, 2) - 1));
    sigma2 = Sigma / SigFactor;
    cv::GaussianBlur(InMat, temp_Out1, KSize, sigma1, 0.0f);
    cv::GaussianBlur(InMat, temp_Out2, KSize, sigma2, 0.0f);
    cv::subtract(temp_Out1, temp_Out2, OutMat);
}
//--------------------------------------------------------------------------------------------------------------------------------------
// 功能:计算图像的LBP纹理图
// 参数:
//          InMat:          输入图像
//          OutMat;         输出图像
//--------------------------------------------------------------------------------------------------------------------------------------
void awcv::LBP(cv::Mat InMat, cv::Mat &OutMat)
{
    if (InMat.channels() == 3)
        cv::cvtColor(InMat, InMat, cv::COLOR_BGR2GRAY);
    cv::Mat temp = cv::Mat::zeros(InMat.size(), InMat.type());
    cv::copyMakeBorder(InMat, InMat, 0, 2, 0, 2, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    // OutMat.create(InMat.rows, InMat.cols, InMat.type()); OutMat.setTo(0);
    for (int i = 1; i < InMat.rows - 1; i++)
    {
        for (int j = 1; j < InMat.cols - 1; j++)
        {
            uchar center = InMat.at<uchar>(i, j);
            uchar code = 0;
            code |= (InMat.at<uchar>(i - 1, j - 1) >= center) << 7;
            code |= (InMat.at<uchar>(i - 1, j) >= center) << 6;
            code |= (InMat.at<uchar>(i - 1, j + 1) >= center) << 5;
            code |= (InMat.at<uchar>(i, j + 1) >= center) << 4;
            code |= (InMat.at<uchar>(i + 1, j + 1) >= center) << 3;
            code |= (InMat.at<uchar>(i + 1, j) >= center) << 2;
            code |= (InMat.at<uchar>(i + 1, j - 1) >= center) << 1;
            code |= (InMat.at<uchar>(i, j - 1) >= center) << 0;
            temp.at<uchar>(i - 1, j - 1) = code;
        }
    }
    OutMat = temp.clone();
}
//--------------------------------------------------------------------------------------------------------------------------------------
// 功能:高斯背景估计算法
// 参数:
//          InMat:          输入图像
//          OutMat;         输出图像
//--------------------------------------------------------------------------------------------------------------------------------------
void awcv::estimateBackgroundIllumination(cv::Mat InMat, cv::Mat &OutMat)
{
    int width = InMat.cols;
    int height = InMat.rows; // 获取图像大小
    cv::Mat filter;
    awcv::genLowpassFilter(filter, cv::Size(width, height), cv::Point(width / 2, height / 2), 50,
                           FILTER_GLPF); // 生成高斯频域滤波器
    if (InMat.channels() == 3)
        cv::cvtColor(InMat, InMat, cv::COLOR_BGR2GRAY);
    DFTMAT dft, idft, afterCon;
    DFT(InMat, dft);                  // 傅里叶变换
    convolDFT(dft, filter, afterCon); // 频域卷积
    IDFT(afterCon, idft);             // 傅里叶逆变换
    OutMat = idft.img.clone();
}
//--------------------------------------------------------------------------------------------------------------------------------------
// 功能:图片颜色翻转
// 参数:
//          InMat:          输入图像
//          OutMat;         输出图像
//--------------------------------------------------------------------------------------------------------------------------------------
void awcv::invertImage(cv::Mat InMat, cv::Mat &OutMat)
{
    if (InMat.channels() == 3)
    {
        OutMat = cv::Scalar(255, 255, 255) - InMat;
    }
    else if (InMat.channels() == 1)
    {
        OutMat = cv::Scalar(255) - InMat;
    }
}
//--------------------------------------------------------------------------------------------------------------------------------------
// 功能:gabor滤波器
// 参数:
//          InMat:          输入图像
//          OutMat:         输出图像
//          KernelSize:     输入卷积核大小
//          Sigma:          高斯滤波器的方差，通常取2π
//          Theta:          滤波器方向(0~pi)
//          Lambd:          滤波尺度，通常大于等于2
//          Gamma:          空间纵横比，取1时为圆形，通常取0.5
//          Psi:            调谐函数的相位偏移，取值-180到180
//--------------------------------------------------------------------------------------------------------------------------------------
void awcv::gaborFilter(cv::Mat InMat, cv::Mat &OutMat, int KernelSize, double Sigma, double Theta, double Lambd, double Gamma, double Psi)
{
    if (InMat.channels() == 3)
        cv::cvtColor(InMat, InMat, cv::COLOR_BGR2GRAY);
    InMat.convertTo(InMat, CV_32F);
    cv::Mat kernel = cv::getGaborKernel(cv::Size(KernelSize, KernelSize), Sigma, Theta, Lambd, Gamma, Psi, CV_32F);
    filter2D(InMat, OutMat, CV_32F, kernel);                // 在频域滤波，有负数
    cv::normalize(OutMat, OutMat, 1.0, 0, cv::NORM_MINMAX); // 缩放一下
    cv::convertScaleAbs(OutMat, OutMat, 255);               // 映射到【0~255】8U
}

void awcv::wlsFilter(cv::Mat InMat, cv::Mat &OutMat, float Sigma, float Lambda, int SolverIteration)
{
}

//--------------------------------------------------------------------------------------------------------------------------------------
// 下面是一些通用类的实现
//--------------------------------------------------------------------------------------------------------------------------------------
#pragma region GLCM
//--------------------------------------------------------------------------------------------------------------------------------------
// 功能:设置最大灰度值
// 参数:
//          GrayLevel:      灰度值
//--------------------------------------------------------------------------------------------------------------------------------------
void awcv::GLCM::setMaxGrayLevel(int GrayLevel)
{
    if (GrayLevel < 16)
    {
        this->MaxGrayLevel = 16;
    }
    else
    {
        this->MaxGrayLevel = GrayLevel;
    }
    // 初始化
    this->glcm_0 = cv::Mat::zeros(cv::Size(this->MaxGrayLevel, this->MaxGrayLevel), CV_8U);
    this->glcm_45 = cv::Mat::zeros(cv::Size(this->MaxGrayLevel, this->MaxGrayLevel), CV_8U);
    this->glcm_90 = cv::Mat::zeros(cv::Size(this->MaxGrayLevel, this->MaxGrayLevel), CV_8U);
    this->glcm_135 = cv::Mat::zeros(cv::Size(this->MaxGrayLevel, this->MaxGrayLevel), CV_8U);
}
//--------------------------------------------------------------------------------------------------------------------------------------
// 功能:计算灰度共生矩阵
// 参数:
//          InMat:          输入图像
//--------------------------------------------------------------------------------------------------------------------------------------
void awcv::GLCM::calcGLCM(cv::Mat InMat)
{
    int height = InMat.rows;
    int width = InMat.cols;
    if (!this->glcmdata.empty())
        glcmdata.clear();
    zoomGray(InMat, this->MaxGrayLevel); // 灰度映射(小心多次调用)
    this->glcm_0.setTo(0);
    this->glcm_45.setTo(0);
    this->glcm_90.setTo(0);
    this->glcm_135.setTo(0);
    for (int row = height - 1; row >= 0; row--) // 从左下角开始
    {
        for (int col = 0; col < width; col++)
        {
            int i = InMat.at<uchar>(row, col); // 起始点：左下角点
            if (row != 0)                      // 上边界
            {
                int j_90 = InMat.at<uchar>(row - 1, col); // 上移
                this->glcm_90.at<uchar>(i - 1, j_90 - 1)++;
            }
            if (col != width - 1) // 右边界
            {
                int j_0 = InMat.at<uchar>(row, col + 1); // col右移
                this->glcm_0.at<uchar>(i - 1, j_0 - 1)++;
            }
            if (row != 0 && col != width - 1) // 上右边界
            {
                int j_45 = InMat.at<uchar>(row - 1, col + 1); // 右上移
                this->glcm_45.at<uchar>(i - 1, j_45 - 1)++;

                int i_135 = InMat.at<uchar>(row, col + 1); // 起始点：左下角点右移一位
                int j_135 = InMat.at<uchar>(row - 1, col); // 左上移
                this->glcm_135.at<uchar>(i_135 - 1, j_135 - 1)++;
            }
        }
    }
    // cv::imshow("glcm_0", glcm_0);cv::imshow("glcm_45", glcm_45);cv::imshow("glcm_90", glcm_90);cv::imshow("glcm_135", glcm_135);
    this->glcmdata.push_back(this->CalcGLCMDATA(this->glcm_0));
    this->glcmdata.push_back(this->CalcGLCMDATA(this->glcm_45));
    this->glcmdata.push_back(this->CalcGLCMDATA(this->glcm_90));
    this->glcmdata.push_back(this->CalcGLCMDATA(this->glcm_135));
}
//--------------------------------------------------------------------------------------------------------------------------------------
// 功能:获取灰度共生矩阵描述子
// 参数:
//          Index:          输入序号
//--------------------------------------------------------------------------------------------------------------------------------------
cv::Mat awcv::GLCM::getGLCMFeatures()
{
    if (this->glcmdata.empty())
        return cv::Mat::zeros(cv::Size(1, 24), CV_32F);
    else
    {
        cv::Mat resultMat;
        std::vector<cv::Mat> v_temp;
        for (int i = 0; i < 4; i++)
        {
            double MaxProbability = this->glcmdata[i].MaxProbability;
            double AngularSecondMoment = this->glcmdata[i].AngularSecondMoment;
            double Contrast = this->glcmdata[i].Contrast;
            double Correlation = this->glcmdata[i].Correlation;
            double Entropy = this->glcmdata[i].Entropy;
            double Homogeneity = this->glcmdata[i].Homogeneity;
            cv::Mat Result = (cv::Mat_<float>(1, 6) << MaxProbability, AngularSecondMoment, Contrast, Correlation, Entropy, Homogeneity);
            v_temp.push_back(Result);
        }
        cv::hconcat(v_temp, resultMat);
        return resultMat;
    }
}
//--------------------------------------------------------------------------------------------------------------------------------------
// 功能:计算灰度共生矩阵描述子
// 参数:
//          GLCMMAT:        输入图像
//--------------------------------------------------------------------------------------------------------------------------------------
awcv::GLCM::GLCMDATA awcv::GLCM::CalcGLCMDATA(cv::Mat GLCMMAT)
{
    awcv::GLCM::GLCMDATA temp = awcv::GLCM::GLCMDATA(); // 灰度共生矩阵描述子
    if (GLCMMAT.empty())
        return temp;
    // std::cout << GLCMMAT.size() << std::endl;
    GLCMMAT.convertTo(GLCMMAT, CV_32F);
    nc::NdArray<float> array = Mat2NdArray<float>(GLCMMAT); // Mat转NdArray
    nc::NdArray<float> total = nc::sum(array);
    array /= total.at(0); // 归一化

    nc::NdArray<float> max = nc::max(array);
    temp.MaxProbability = max.at(0);                       // 最大概率 MaxProbability √
    nc::NdArray<float> Pi = nc::sum(array, nc::Axis::COL); // 按行计算和
    nc::NdArray<float> Pj = nc::sum(array, nc::Axis::ROW); // 按列计算和
    // 【注意】由于灰度范围在[1-L]，这里的ij是从1开始的共L个元素的数列；
    nc::NdArray<float> ij = nc::arange<float>(1, this->MaxGrayLevel + 1);                                    // 【1-Level】区间
    nc::NdArray<float> mr = nc::sum(nc::multiply(Pi, ij));                                                   // 按行计算均值
    nc::NdArray<float> mc = nc::sum(nc::multiply(Pj, ij));                                                   // 按列计算均值
    nc::NdArray<float> sigma_r = nc::sqrt(nc::sum(nc::multiply(nc::power(ij - mr.at(0), 2), Pi))).flatten(); // 按行计算方差
    nc::NdArray<float> sigma_c = nc::sqrt(nc::sum(nc::multiply(nc::power(ij - mc.at(0), 2), Pj))).flatten(); // 按列计算方差
    nc::NdArray<float> ij_mr = nc::subtract(ij, mr.at(0));                                                   // 【行】灰度值-灰度均值
    nc::NdArray<float> ij_mc = nc::subtract(ij, mc.at(0));                                                   // 【列】灰度值-灰度均值

    for (int i = 0; i < this->MaxGrayLevel; i++)
    {
        for (int j = 0; j < this->MaxGrayLevel; j++) // 双重Sigma
        {
            float P = array.at(i, j);
            temp.AngularSecondMoment += pow(P, 2); // 均匀度 AngularSecondMoment √
            if (P > 0)
                temp.Entropy -= P * log2(P);                        // 熵 Entropy √
            temp.Contrast += pow(ij.at(i) - ij.at(j), 2) * P;       // 对比度 Contrast √
            temp.Homogeneity += P / (1 + abs(ij.at(i) - ij.at(j))); // 同质性 Homogeneity √
            if (sigma_r.at(0) != 0 && sigma_c.at(0) != 0)
                temp.Correlation += ij_mr.at(i) * ij_mc.at(j) / (sigma_r.at(0) * sigma_c.at(0)) * P; // 相关性 Correlation √
        }
    }
    return temp;
}

#pragma endregion
