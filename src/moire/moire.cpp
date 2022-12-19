#include "moire/moire.hpp"
//--------------------------------------------------------------------------------------------------------------------------------------
//                                          MoirePatternDetector类构造函数
//--------------------------------------------------------------------------------------------------------------------------------------
MoirePatternDetector::MoirePatternDetector()
{
    gl.setMaxGrayLevel(64); // 初始化灰度共生矩阵
}
MoirePatternDetector::MoirePatternDetector(std::string ModelFile)
{
    gl.setMaxGrayLevel(64);   // 初始化灰度共生矩阵
    loadModelFile(ModelFile); // 初始化SVM句柄
}
//--------------------------------------------------------------------------------------------------------------------------------------
//                                          MoirePatternDetector类析构函数
//--------------------------------------------------------------------------------------------------------------------------------------
MoirePatternDetector::~MoirePatternDetector()
{
    svm.release(); // 释放SVM句柄
}
//--------------------------------------------------------------------------------------------------------------------------------------
// 功能:加载SVM模型
// 参数:
//          ModelFile:          模型文件路径
//--------------------------------------------------------------------------------------------------------------------------------------
void MoirePatternDetector::loadModelFile(std::string ModelFile)
{
    if (!boost::filesystem::exists(ModelFile))
    {
        printf("SVM模型文件路径错误: %s", ModelFile.c_str());
        return;
    }
    svm = cv::Algorithm::load<cv::ml::SVM>(ModelFile); // SVM分类器
}
//--------------------------------------------------------------------------------------------------------------------------------------
// 功能:摩尔纹检测
// 参数:
//          InMat:      输入图像
//          Mask:       人脸框
// 返回值:
//          bool:
//                      有摩尔纹true
//                      无摩尔纹false
//--------------------------------------------------------------------------------------------------------------------------------------
bool MoirePatternDetector::detect(cv::Mat InMat, cv::Rect Mask)
{
    if (InMat.channels() == 3)
        awcv::bgr2gray(InMat, InMat); // 转灰度图
    gl.calcGLCM(InMat(Mask));
    cv::Mat feature = gl.getGLCMFeatures();
    // cout << features << endl;
    float res = svm->predict(feature);
    if (res == 1.0f)
    {
        return false;
    } // 无摩尔纹
    else if (res == 0.0f)
    {
        return true;
    } // 有摩尔纹
    else
    {
        return false;
    }
}
//--------------------------------------------------------------------------------------------------------------------------------------
// 功能:摩尔纹检测
// 参数:
//          InMat:      输入图像
//          Mask:       人脸框
// 返回值:
//          bool:
//                      有摩尔纹true
//                      无摩尔纹false
//--------------------------------------------------------------------------------------------------------------------------------------
bool MoirePatternDetector::detect(cv::Mat InMat)
{
    if (InMat.channels() == 3)
        awcv::bgr2gray(InMat, InMat); // 转灰度图
    gl.calcGLCM(InMat);
    cv::Mat feature = gl.getGLCMFeatures();
    // cout << features << endl;
    float res = svm->predict(feature);
    if (res == 1.0f)
    {
        return false;
    } // 无摩尔纹
    else if (res == 0.0f)
    {
        return true;
    } // 有摩尔纹
    else
    {
        return false;
    }
}
