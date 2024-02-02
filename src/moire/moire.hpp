//--------------------------------------------------------------------------------------------------------------------------------------
#pragma once // 防止重复编译
#ifndef H_AWCV_MoireDetector
#define H_AWCV_MoireDetector

#define Aw_DEBUG 1 // DEBUG开关
//--------------------------------------------------------------------------------------------------------------------------------------
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/ml.hpp>
#include <random>
#include <stdio.h>
#include "core.hpp"


using namespace std;
using namespace cv::ml;
//--------------------------------------------------------------------------------------------------------------------------------------
//														MoirePatternDetector类（改进后）
//--------------------------------------------------------------------------------------------------------------------------------------
class MoirePatternDetector
{
  public:
    MoirePatternDetector();
    MoirePatternDetector(std::string ModelFile);
    ~MoirePatternDetector();

    void loadModelFile(std::string ModelFile);
    bool detect(cv::Mat InMat, cv::Rect Mask);
    bool detect(cv::Mat InMat);

  private:
    cv::Ptr<cv::ml::SVM> svm; // svm句柄
    awcv::GLCM gl;            // 灰度共生矩阵类
    //--------------------------------------------------------------------------------------------------------------------------------------
}; // endClass
//--------------------------------------------------------------------------------------------------------------------------------------

#endif
