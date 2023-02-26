#pragma once
#ifndef H_AWCV_TEST
#define H_AWCV_TEST

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "core.hpp"
#include "face.hpp"
#include "file.hpp"
#include "moire/moire.hpp"
#include "opticalFlow.hpp"
#include "sfm.hpp"
#include <iostream>

using namespace awcv;
using namespace awcv::face;
//--------------------------------------------------------------------------------------------------------------------------------------
// 功能:运行时间统计
//--------------------------------------------------------------------------------------------------------------------------------------
inline void spend(void (*Func)(cv::Mat), cv::Mat file)
{
    double begin = (double)cv::getCPUTickCount();
    Func(file);
    double end = (double)cv::getCPUTickCount();
    printf("read time: %0.2f ms\n", (end - begin) / cv::getTickFrequency() * 1000);
}
inline void spend(void (*Func)(bool), bool bb)
{
    double begin = (double)cv::getCPUTickCount();
    Func(bb);
    double end = (double)cv::getCPUTickCount();
    printf("read time: %0.2f ms\n", (end - begin) / cv::getTickFrequency() * 1000);
}
inline void dododo(cv::Mat img)
{
    cv::Mat gray;
    awcv::resize(img, cv::Size(512, 512));
    // awcv::resize(gray, 0.5);
    awcv::bgr2gray(img, gray);
    cv::imshow("灰度图", gray);
    int w = 0, h = 0;
    awcv::optimalDFTSize(gray, w, h); // 优化图像大小
    awcv::zoomGray(gray, 128);
    cv::imshow("img", gray);

#pragma region 高斯差分
    cv::Mat diff;
    awcv::diffOfGaussian(gray, diff, cv::Size(7, 7), 0.5, 1.8); // 高斯差分
    cv::imshow("diff", diff);
#pragma endregion

#pragma region 高斯背景估计法
    cv::Mat background;
    awcv::estimateBackgroundIllumination(gray, background);
    cv::imshow("background", background);
#pragma endregion

#pragma region 分水岭法
    // cv::Mat waterthres;
    // awcv::watershedsThreshold(img, waterthres, 10, 120);
    // cv::imshow("water", waterthres);
#pragma endregion

#pragma region 均值飘移
    // 非常耗时，效果像油画
    // cv::Mat imgShift;
    // cv::pyrMeanShiftFiltering(img, imgShift, 21, 30);
    // cv::imshow("imgShift", imgShift);
#pragma endregion

#pragma region gabor滤波器
    // cv::Mat gabor;
    // awcv::gaborFilter(gray, gabor);
    // cv::imshow("gabor", gabor);
    // cv::Mat thres;
    // cv::threshold(gabor, thres, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
    // cv::imshow("thres", thres);
#pragma endregion

    
    cv::Mat filter;
    awcv::genLowpassFilter(filter, cv::Size(w, h), cv::Point(w / 2, h / 2), 20, awcv::FILTER_BLPF, 8);
    awcv::genHighpassFilter(filter, cv::Size(w, h), cv::Point(w / 2, h / 2), 6, awcv::FILTER_GHPF);
    cv::imshow("filter", filter);
    awcv::DFTMAT lbp_dft, lbp_idft, afterCon;
    awcv::DFT(gray, lbp_dft);                                  //傅里叶变换
    awcv::convolDFT(lbp_dft, filter, afterCon);                //频域滤波
    awcv::IDFT(afterCon, lbp_idft);                            //傅里叶逆变换

    ////cv::Mat result;
    ////cv::adaptiveThreshold(lbp_idft.img, result, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 5, 0);
    ////awcv::LBP(lbp_idft.img, lbp);                            //LBP提取纹理

    // cv::imshow("dft", lbp_dft.img);
    cv::imshow("idft", lbp_idft.getMat() * 10);
    ////cv::imshow("res", result);
    ////cv::imshow("result", lbp);

#pragma region 纹理提取（背景差分）
    cv::Mat equal;
    cv::equalizeHist(gray, equal);
    cv::imshow("equal", equal);
    // cv::Mat sub;
    ////cv::absdiff(gray, background, sub);
    // cv::subtract(gray, background, sub);
    // sub *= 2;
    // cv::imshow("sub", sub);
    // awcv::LBP(sub, sub);
    ////cv::imshow("sub", sub);
#pragma endregion
}

#pragma region 初始化滑条控件
inline void __TrackbarCallback(int pos, void *userdata)
{
    return;
}
inline void initGammaTrackbar()
{
    cv::namedWindow("gammaImage", cv::WINDOW_NORMAL);
    int value = 10;
    cv::createTrackbar("Gamma", "gammaImage", &value, 1000, __TrackbarCallback);
}
inline void initGrayMappingTrackbar()
{
    cv::namedWindow("zoomGray", cv::WINDOW_NORMAL);
    int value = 1;
    cv::createTrackbar("MaxGrayLevel", "zoomGray", &value, 256, __TrackbarCallback);
}
inline void initDiffOfGaussianTrackbar()
{
    cv::namedWindow("diffOfGaussian", cv::WINDOW_NORMAL);
    cv::createTrackbar("KernelSize", "diffOfGaussian", 0, 10, __TrackbarCallback);
    cv::createTrackbar("Sigma", "diffOfGaussian", 0, 20, __TrackbarCallback);
    cv::createTrackbar("SigFactor", "diffOfGaussian", 0, 20, __TrackbarCallback);
}
inline void initGaborFilterTrackbar()
{
    cv::namedWindow("gaborFilter", cv::WINDOW_NORMAL);
    cv::createTrackbar("KernelSize", "gaborFilter", 0, 50, __TrackbarCallback);
    cv::createTrackbar("Sigma", "gaborFilter", 0, 360, __TrackbarCallback);
    cv::createTrackbar("Theta", "gaborFilter", 0, 180, __TrackbarCallback);
    cv::createTrackbar("Lambd", "gaborFilter", 0, 100, __TrackbarCallback);
    cv::createTrackbar("Gamma", "gaborFilter", 0, 100, __TrackbarCallback);
    cv::createTrackbar("Psi", "gaborFilter", 0, 360, __TrackbarCallback);
}
inline void initLinearGrayLevelTrans()
{
    cv::namedWindow("linearGrayLevelTrans", cv::WINDOW_NORMAL);
    cv::createTrackbar("Th1", "linearGrayLevelTrans", 0, 255, __TrackbarCallback);
    cv::createTrackbar("Th2", "linearGrayLevelTrans", 0, 255, __TrackbarCallback);
    cv::createTrackbar("Goal1", "linearGrayLevelTrans", 0, 255, __TrackbarCallback);
    cv::createTrackbar("Goal2", "linearGrayLevelTrans", 0, 255, __TrackbarCallback);
}
inline void initLogImageTrackbar()
{
    cv::namedWindow("logImage", cv::WINDOW_NORMAL);
    int value = 1;
    cv::createTrackbar("Const", "logImage", &value, 1000, __TrackbarCallback);
}
inline void initEnhanceImageByMean()
{
    cv::namedWindow("enhanceImageByMean", cv::WINDOW_NORMAL);
    int min = 0;
    int max = 0;
    cv::createTrackbar("MinBlack", "enhanceImageByMean", &min, 255, __TrackbarCallback);
    cv::createTrackbar("MaxWhite", "enhanceImageByMean", &max, 255, __TrackbarCallback);
}
inline void initEnhanceImageByOTSU()
{
    cv::namedWindow("enhanceImageByOTSU", cv::WINDOW_NORMAL);
    int thres = 0;
    cv::createTrackbar("thres", "enhanceImageByOTSU", &thres, 255, __TrackbarCallback);
}
inline void initBilateralFilter()
{
    cv::namedWindow("bilateralFilter", cv::WINDOW_NORMAL);
    cv::createTrackbar("d", "bilateralFilter", 0, 50, __TrackbarCallback);
    cv::createTrackbar("sigmaColor", "bilateralFilter", 0, 255, __TrackbarCallback);
    cv::createTrackbar("sigmaSpace", "bilateralFilter", 0, 255, __TrackbarCallback);
}
inline void initRegionGrowing()
{
    int loDiff = 20, upDiff = 20;
    cv::namedWindow("regionGrowing", cv::WINDOW_NORMAL);
    cv::createTrackbar("lo_diff", "regionGrowing", &loDiff, 255, 0);
    cv::createTrackbar("up_diff", "regionGrowing", &upDiff, 255, 0);
}
#pragma endregion

#pragma region 测试算法封装
inline void t_gammaImage(cv::Mat InMat, cv::Mat &OutMat)
{
    float gamma = cv::getTrackbarPos("Gamma", "gammaImage") / static_cast<float>(10); // 伽马变换
    // std::cout << gamma << std::endl;
    awcv::gammaImage(InMat, OutMat, gamma);
    cv::imshow("gammaImage", OutMat);
}
inline void t_autoGammaImage(cv::Mat InMat, cv::Mat &OutMat)
{
    ////计算平均灰度值
    // if (InMat.channels() == 3) cv::cvtColor(InMat, InMat, cv::COLOR_BGR2GRAY);
    // auto meanGray = cv::mean(InMat)[0];
    // std::cout << "原图平均灰度值:" << meanGray << std::endl;
    // float gamma_val = log10(1 - 0.5f) / log10(1 - meanGray / 255.0f);               //自动gamma参数
    // std::cout << "自动gamma参数:" << gamma_val << std::endl;
    // awcv::gammaImage(InMat, OutMat, gamma_val);
    // auto meanGray_ = cv::mean(OutMat);
    // std::cout << "变换后的平均灰度:" << meanGray_[0] << std::endl;
    // cv::imshow("自动伽马变换", OutMat);
    awcv::autoGammaImage(InMat, OutMat, 0.3f);
    cv::imshow("autoGammaImage", OutMat);
}
inline void t_zoomGray(cv::Mat InMat, cv::Mat &OutMat)
{
    int grayLevel = cv::getTrackbarPos("MaxGrayLevel", "zoomGray");
    awcv::zoomGray(InMat, grayLevel);
    OutMat = InMat.clone();
    cv::imshow("zoomGray", OutMat);
}
inline void t_diffOfGaussian(cv::Mat InMat, cv::Mat &OutMat)
{
    int KernelSize = cv::getTrackbarPos("KernelSize", "diffOfGaussian");                            // 高斯差分KernelSize
    double Sigma = cv::getTrackbarPos("Sigma", "diffOfGaussian") / static_cast<double>(10);         // 高斯差分Sigma
    double SigFactor = cv::getTrackbarPos("SigFactor", "diffOfGaussian") / static_cast<double>(10); // 高斯差分SigFactor
    cv::cvtColor(InMat, InMat, cv::COLOR_BGR2GRAY);
    if (Sigma != 0 && SigFactor != 0 && KernelSize != 0)
    {
        awcv::diffOfGaussian(InMat, OutMat, cv::Size(KernelSize, KernelSize), Sigma, SigFactor);
        cv::imshow("diffOfGaussian", OutMat * 10);
    }
}
inline void t_LBP(cv::Mat InMat, cv::Mat &OutMat)
{
    awcv::LBP(InMat, OutMat);
    cv::imshow("LBP", OutMat);
}
inline void t_gaborFilter(cv::Mat InMat, cv::Mat &OutMat)
{
    int KernelSize = cv::getTrackbarPos("KernelSize", "gaborFilter");
    // double Sigma = cv::getTrackbarPos("Sigma", "Gabor滤波器") / static_cast<double>(180) * CV_PI;
    float Sigma = static_cast<float>(cv::getTrackbarPos("Sigma", "gaborFilter"));
    float Theta = static_cast<float>(cv::getTrackbarPos("Theta", "gaborFilter") / static_cast<float>(180) * CV_PI);
    // double Lambd = cv::getTrackbarPos("Lambd", "Gabor滤波器") / static_cast<double>(180) * CV_PI;
    float Lambd = cv::getTrackbarPos("Lambd", "gaborFilter") / static_cast<float>(100);
    // float Lambd = cv::getTrackbarPos("Lambd", "Gabor滤波器");
    float Gamma = cv::getTrackbarPos("Gamma", "gaborFilter") / static_cast<float>(100);
    float Psi = static_cast<float>((cv::getTrackbarPos("Psi", "gaborFilter") - 180) / static_cast<float>(180) * CV_PI);

    awcv::gaborFilter(InMat, OutMat, KernelSize, Sigma, Theta, Lambd, Gamma, Psi);
    cv::imshow("gaborFilter", OutMat);                                                                                                               // 显示之前会将【0-1】映射到【0-255】，但是事先convertScaleAbs的结果略有差异
    fprintf(stdout, "KernelSize: %d\tSigma: %.4f\tTheta: %.4f\tLambd: %.4f\tGamma: %.4f\tPsi: %.4f\n", KernelSize, Sigma, Theta, Lambd, Gamma, Psi); // 显示参数

    cv::Mat thres;
    cv::threshold(OutMat, thres, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
    cv::imshow("gaborFilterThres", thres);
}
inline void t_GLCM(cv::Mat InMat, awcv::GLCM gl, cv::Mat &OutMat)
{
    gl.calcGLCM(InMat);
    cv::Mat res = gl.getGLCMFeatures();
    std::cout << res << std::endl;
}
inline void t_linearGrayLevelTrans(cv::Mat InMat, cv::Mat &OutMat)
{
    int th1 = cv::getTrackbarPos("Th1", "linearGrayLevelTrans");
    int th2 = cv::getTrackbarPos("Th2", "linearGrayLevelTrans");
    int goal1 = cv::getTrackbarPos("Goal1", "linearGrayLevelTrans");
    int goal2 = cv::getTrackbarPos("Goal2", "linearGrayLevelTrans");
    awcv::linearGrayLevelTrans(InMat, OutMat, th1, th2, goal1, goal2);
    cv::imshow("linearGrayLevelTrans", OutMat);
}
inline void t_logImage(cv::Mat InMat, cv::Mat &OutMat)
{
    float Const = cv::getTrackbarPos("Const", "logImage") / static_cast<float>(10);
    awcv::logImage(InMat, OutMat, Const);
    // std::cout << OutMat << std::endl;
    cv::imshow("logImage", OutMat);
}
inline void t_enhanceImageByMean(cv::Mat InMat, cv::Mat &OutMat)
{
    double min = cv::getTrackbarPos("MinBlack", "enhanceImageByMean");
    double max = cv::getTrackbarPos("MaxWhite", "enhanceImageByMean");
    awcv::enhanceImageByMean(InMat, OutMat, min, max);
    cv::imshow("enhanceImageByMean", OutMat);
}
inline void t_enhanceImageByOTSU(cv::Mat InMat, cv::Mat &OutMat)
{
    int thres = cv::getTrackbarPos("thres", "enhanceImageByOTSU");
    awcv::enhanceImageByOTSU(InMat, OutMat, thres);
    cv::imshow("enhanceImageByOTSU", OutMat);
}
inline void t_bilateralFilter(cv::Mat InMat, cv::Mat &OutMat)
{
    int d = cv::getTrackbarPos("d", "bilateralFilter");
    int sigmaColor = cv::getTrackbarPos("sigmaColor", "bilateralFilter");
    int sigmaSpace = cv::getTrackbarPos("sigmaSpace", "bilateralFilter");
    cv::bilateralFilter(InMat, OutMat, d, sigmaColor, sigmaSpace, cv::BORDER_DEFAULT);
    cv::imshow("bilateralFilter", OutMat);
}
//inline void t_regionGrowing(cv::Mat InMat, cv::Point Seed, cv::Mat &OutMat)
//{
//    int loDiff = cv::getTrackbarPos("loDiff", "regionGrowing");
//    int upDiff = cv::getTrackbarPos("upDiff", "regionGrowing");
//    awcv::regionGrowing(InMat, OutMat, Seed, loDiff, upDiff);
//    cv::imshow("regionGrowing", OutMat);
//}
#pragma endregion

#pragma region 调用测试算法
inline void testGabor(std::string file, bool useCamera = false)
{
    initGaborFilterTrackbar();
    if (!useCamera)
    {
#pragma region 读取图片
        // boost::filesystem::path root(R"(C:\Users\74001\Desktop\单目RGB图片\save\zhenren)");
        // boost::filesystem::directory_iterator endIter;

        cv::Mat img, gray;
        awcv::GLCM gl;
        gl.setMaxGrayLevel(64);
        // for (boost::filesystem::directory_iterator Iter(root); Iter != endIter; Iter++)
        //{
        //     //spend(dododo, cv::imread(Iter->path().string()));           //计时
        img = cv::imread(file);   //
        awcv::bgr2gray(img, img); // 单通道

        cv::Mat temp;
        while (1)
        {
            t_gaborFilter(img, temp);
            t_GLCM(temp, gl, temp);

            int k = cv::waitKey(1);
            if (k == 32)
                break;
        }

        //    int k = cv::waitKey(1);
        //    if (k == 27) break;
        //}

#pragma endregion
    }
    else
    {
#pragma region 读取视频
        MoirePatternDetector moireHandle = MoirePatternDetector(R"(..\data\svm24_crop.xml)");
        awcv::GLCM gl;
        gl.setMaxGrayLevel(64); // 设置最大灰度级
        cv::VideoCapture camera(0);
        cv::Mat frame;
        while (camera.read(frame))
        {
            awcv::bgr2gray(frame, frame);
            // t_gaborFilter(frame, frame);
            bool hasMoire = moireHandle.detect(frame, cv::Rect(0, 0, frame.cols, frame.rows));
            if (hasMoire)
            {
                cv::putText(frame, cv::format("hasMoire"), cv::Point2i(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
            }
            else
            {
                cv::putText(frame, cv::format("noMoire"), cv::Point2i(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
            }
            cv::imshow("秀一个", frame);
            // t_GLCM(frame, gl, frame);

            int k = cv::waitKey(1);
            if (k == 27)
                break;
        }
#pragma endregion
    }
}
inline void testLinearGrayLevelTrans(std::string Filepath, bool Recursion = false)
{
    // initGammaTrackbar();
    // initLogImageTrackbar();
    // initLinearGrayLevelTrans();

    std::string model = R"(C:\Users\74001\source\repos\antiSpoofingDetector\x64\Release\data\face_detection_yunet_2022mar.onnx)";
    FaceDetectorDNN::param Param = FaceDetectorDNN::param(0, 0, 0.98f, 0.4f, 5000, false, true);
    FaceDetectorDNN faceHandle = FaceDetectorDNN(model, Param);

    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(6, 6)); // 直方图均衡化算子
    FileHandle fileHandle(Filepath, ".jpg", Recursion);              // 初始化人脸数据集路径
    FilesVector files = fileHandle.getFilesVector();
    for (FilesVector::iterator iter = files.begin(); iter != files.end(); iter++)
    {
        for (auto iiter = iter->begin(); iiter != iter->end(); iiter++)
        {
            cv::Mat img, temp, out, gamma, log;
            img = cv::imread(*iiter);              // 【第一步】读取图片
            awcv::bgr2gray(img, img);              // 【第二步】转灰度图
            awcv::resize(img, cv::Size(900, 900)); // 【第三步】缩放图片
            faceHandle.setInputSize(img.size());
            cv::imshow("原图", img);
            // cv::GaussianBlur(img, img, cv::Size(3, 3), 0.0f, 0.0f);
            clahe->apply(img, img); // 直方图均衡
            // cv::imshow("直方图均衡化", temp);
            // cv::GaussianBlur(img, img, cv::Size(5, 5), 0.0f, 0.0f);
            // cv::GaussianBlur(img, img, cv::Size(3, 3), 0.0f, 0.0f);
            while (true)
            {

                FaceDetectorDNN::face Aface = faceHandle.detect(img);
                cv::Mat res = faceHandle.visualize(img, Aface, true);
                cv::imshow("人脸检测结果", res);
                if (cv::waitKey(1) == 32)
                    break; // 空格到下一张
                if (cv::waitKey(1) == 27)
                    return; // Esc退出
            }
        }
    }
}
inline void testLogImage(std::string Filepath, bool Recursion = false)
{
    std::string model = R"(C:\Users\74001\source\repos\antiSpoofingDetector\x64\Release\data\face_detection_yunet_2022mar.onnx)";
    FaceDetectorDNN::param Param;
    FaceDetectorDNN faceHandle = FaceDetectorDNN(model, Param);

    // initLogImageTrackbar();
    initGaborFilterTrackbar();
    FileHandle fileHandle(Filepath, ".jpg", Recursion); // 初始化人脸数据集路径
    FilesVector files = fileHandle.getFilesVector();
    for (FilesVector::iterator iter = files.begin(); iter != files.end(); iter++)
    {
        for (auto iiter = iter->begin(); iiter != iter->end(); iiter++)
        {
            cv::Mat img, lbp, temp, out;
            awcv::DFTMAT lbp_dft;
            img = cv::imread(*iiter); // 【第一步】读取图片
            awcv::bgr2gray(img, img);
            awcv::resize(img, 0.4);                            // 图太大了，要缩放
            img(cv::Rect(230, 0, 334, img.rows)).copyTo(temp); // 脸部ROI区域
            cv::imshow("原图", temp);
            t_LBP(temp, lbp);
            awcv::DFT(lbp, lbp_dft);
            cv::imshow("频谱", lbp_dft.getMat());
            // t_logImage(temp, log);
            if (cv::waitKey(0) == 32)
                break; // 空格到下一张
            if (cv::waitKey(1) == 27)
                return; // Esc退出
        }
    }
}
inline void testFaceDetectorDNN(std::string Filepath, bool Recursion = false)
{
    cv::Mat faces;
    std::string model = R"(C:\Users\74001\source\repos\antiSpoofingDetector\x64\Release\data\face_detection_yunet_2022mar.onnx)";
    FaceDetectorDNN::param Param;
    FaceDetectorDNN faceHandle = FaceDetectorDNN(model, Param);

    FileHandle fileHandle(Filepath, ".jpg", Recursion); // 初始化人脸数据集路径
    FilesVector files = fileHandle.getFilesVector();
    for (FilesVector::iterator iter = files.begin(); iter != files.end(); iter++)
    {
        for (auto iiter = iter->begin(); iiter != iter->end(); iiter++)
        {
            cv::Mat img = cv::imread(*iiter); // 【第一步】读取图片
            awcv::bgr2gray(img, img);
            awcv::resize(img, 0.4);
            cv::imshow("原图", img);
            img(cv::Rect(256, 0, 256, img.rows)).copyTo(img); // 脸部ROI区域
            faceHandle.setInputSize(img.size());              // 设置检测器大小
            FaceDetectorDNN::face Aface = faceHandle.detect(img);
            cv::Mat vis_image = faceHandle.visualize(img, Aface, true);
            cv::imshow("人脸", vis_image);
            if (cv::waitKey(1) == 32)
                break; // 空格到下一张
        }
    }
}
inline void testFaceDetectorDNN()
{
    std::string model = R"(..\data\face_detection_yunet_2022mar.onnx)";
    FaceDetectorDNN::param Param;
    FaceDetectorDNN faceHandle = FaceDetectorDNN(model, Param);

    int deviceId = 0;
    cv::VideoCapture cap;
    cap.open(deviceId, cv::CAP_ANY);
    int frameWidth = int(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frameHeight = int(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    faceHandle.setInputSize(cv::Size(frameWidth, frameHeight)); // 设置宽高

    cv::Mat frame;
    cv::TickMeter tm;
    while (cv::waitKey(1) < 0) // Press any key to exit
    {
        if (!cap.read(frame))
        {
            std::cerr << "No frames grabbed!\n";
            break;
        }

        tm.start();
        // awcv::bgr2gray(frame, frame);
        FaceDetectorDNN::face AFace = faceHandle.detect(frame);
        tm.stop();

        cv::Mat vis_frame = faceHandle.visualize(frame, AFace, false, tm.getFPS());

        imshow("libfacedetection demo", vis_frame);

        tm.reset();
    }
}
inline void testSFM()
{
    awcv::sfm::SFM my_sfm = awcv::sfm::SFM(R"(C:\Users\74001\source\repos\CameraCalibration-OpenCV\CameraCalibration-OpenCV\save_cc\CameraMatrix.bin)"); // 相机内参

    std::string model = R"(..\data\face_detection_yunet_2022mar.onnx)";
    FaceDetectorDNN::param Param;
    FaceDetectorDNN faceHandle = FaceDetectorDNN(model, Param);

    awcv::opt::SparseOpticalFlow sof;

    // 创建SIFT特征检测器
    cv::Ptr<cv::SIFT> siftHandle = cv::SIFT::create();

    cv::VideoCapture Camera(0);
    cv::Mat frame, T_frame;

    awcv::sfm::_Tracks tracks; // 若干点若干帧
    awcv::sfm::_Frames frames; // 一个点的若干帧

    int frameWidth = int(Camera.get(cv::CAP_PROP_FRAME_WIDTH));
    int frameHeight = int(Camera.get(cv::CAP_PROP_FRAME_HEIGHT));
    faceHandle.setInputSize(cv::Size(frameWidth, frameHeight)); // 设置宽高

    std::vector<cv::Mat> rs_est, ts_est, points3d_estimated, point2d;

    int maxCorners = 10;
    // int maxTrackbar = 25;
    std::vector<cv::Point2f> corners;
    double qualityLevel = 0.01;
    double minDistance = 10;
    int blockSize = 3;
    bool useHarrisDetector = true;
    double k = 0.04;

    std::vector<cv::Point2f> estimatePts;

    while (cv::waitKey(1) != 27)
    {
        if (!Camera.read(frame))
        {
            std::cerr << "No frames grabbed!\n"; // 标准错误输出
            break;
        }
        // 进行预处理

        T_frame = frame.clone();
        awcv::bgr2gray(frame, frame);
        if (corners.empty())
            cv::goodFeaturesToTrack(frame, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, useHarrisDetector, k);

        cv::Mat status, err;

        sof.calcOpticalFlowLK(frame, corners, estimatePts, status, err);

        // 生成每帧的点矩阵
        cv::Mat_<double> framePts(2, maxCorners); // 2x角点个数
        for (int i{0}; i < estimatePts.size(); i++)
        {
            std::cout << estimatePts[i] << std::endl;
            cv::circle(T_frame, estimatePts[i], 2, cv::Scalar(0, 0, 255), -1);
            if (status.at<uchar>(i, 0) == 1)
            {
                framePts(0, i) = estimatePts[i].x;
                framePts(1, i) = estimatePts[i].y;
            }
            else
            {
                framePts(0, i) = -1;
                framePts(1, i) = -1;
            }
        }
        cv::imshow("原图", T_frame);
        point2d.push_back(framePts);
    }

    // my_sfm.loadTracks(R"(C:\Users\74001\source\repos\CameraCalibration-OpenCV\CameraCalibration-OpenCV\save_cc\RotationVector.bin)", 3, 10);
    // my_sfm.getTracks(tracks);
    my_sfm.setTracks(point2d);
    my_sfm.reconstruct(rs_est, ts_est, points3d_estimated);
    my_sfm.show(rs_est, ts_est, points3d_estimated);

    for (std::vector<cv::Mat>::iterator PE = points3d_estimated.begin(); PE != points3d_estimated.end(); PE++)
    {
        std::cout << *PE << std::endl;
    }
}
#pragma endregion

#pragma region 临时算法测试
inline void FaceDetectDNNEval(std::string Path, std::string Save)
{
    std::string model = R"(..\data\face_detection_yunet_2022mar.onnx)";
    FaceDetectorDNN::param Param;
    FaceDetectorDNN faceHandle = FaceDetectorDNN(model, Param);

    boost::filesystem::path save(Save);
    FileHandle fileHandle = FileHandle();
    fileHandle.getFiles(Path, ".jpg", false);
    FilesVector parent = fileHandle.getFilesVector();
    for (FilesVector::iterator iter = parent.begin(); iter != parent.end(); iter++)
    {
        for (std::vector<std::string>::iterator iiter = iter->begin(); iiter != iter->end(); iiter++)
        {
            boost::filesystem::path filename(*iiter);
            cv::Mat img = cv::imread(*iiter);
            awcv::resize(img, 0.4);                           // 缩小（加速运算）
            img(cv::Rect(230, 0, 334, img.rows)).copyTo(img); // 脸部ROI区域
            awcv::bgr2gray(img, img);                         // 转灰度图
            faceHandle.setInputSize(img.size());
            FaceDetectorDNN::face Aface = faceHandle.detect(img);
            if (Aface.getHasFace())
            {
                cv::rectangle(img, Aface.getFaceRegion(), cv::Scalar(255));
                cv::imwrite((save / "找到脸" / filename.filename().string()).string().c_str(), img);
            }
            else
            {
                cv::imwrite((save / "找不到脸" / filename.filename().string()).string().c_str(), img);
            }
        }
    }
}
inline void testTextureExtrace()
{
    int deviceID = 0;
    cv::VideoCapture Camera;
    Camera.open(deviceID, cv::CAP_ANY);

    // cv::Ptr<cv::SIFT> sift = cv::SIFT::create();                                    //特征提取：普通
    cv::Ptr<cv::ORB> orb = cv::ORB::create(50); // 特征提取：很快
    // cv::Ptr<cv::BRISK> brisk = cv::BRISK::create();                                 //特征提取：还不行吧
    // cv::Ptr<cv::KAZE> kaze = cv::KAZE::create();                                    //特征提取：特耗内存
    // cv::Ptr<cv::AKAZE> akaze = cv::AKAZE::create();                                 //特征提取：比kaze稍好
    // cv::Ptr<cv::SimpleBlobDetector> sbd = cv::SimpleBlobDetector::create();         //特征提取：怪东西用不了
    // cv::Ptr<cv::GFTTDetector> gftt = cv::GFTTDetector::create();                    //特征提取：怪东西用不了
    // cv::Ptr<cv::AgastFeatureDetector> agast = cv::AgastFeatureDetector::create();   //特征提取：怪东西用不了
    // cv::Ptr<cv::MSER> mser = cv::MSER::create();                                    //特征提取：怪东西用不了
    // cv::Ptr<cv::FastFeatureDetector> ffd = cv::FastFeatureDetector::create();       //特征提取：怪东西用不了

    cv::Ptr<cv::BFMatcher> bfMatcher = cv::BFMatcher::create(cv::NORM_HAMMING); // 特征匹配（NORM_HAMMING should be used with ORB）
    cv::Ptr<cv::FlannBasedMatcher> flann = cv::FlannBasedMatcher::create();     // 特征匹配

    // 角点检测器
    int maxCorners = 50;
    // int maxTrackbar = 25;
    std::vector<cv::Point2f> corners;
    double qualityLevel = 0.01;
    double minDistance = 10;
    int blockSize = 3;
    bool useHarrisDetector = true;
    double k = 0.04;

    // 制作模板
    cv::Mat templete = cv::imread(R"(C:\Users\74001\Desktop\QQ图片20221118153220.jpg)");
    awcv::bgr2gray(templete, templete);
    std::vector<cv::KeyPoint> keyTemplete; // 模板的关键点
    cv::Mat descTemplete;
    cv::goodFeaturesToTrack(templete, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, useHarrisDetector, k);
    cv::KeyPoint::convert(corners, keyTemplete);
    orb->detectAndCompute(templete, cv::Mat(), keyTemplete, descTemplete, true);
    cv::drawKeypoints(templete, keyTemplete, templete, cv::Scalar(0, 0, 255));
    cv::imshow("templete", templete);

    cv::Mat frame, Matches;
    std::vector<cv::KeyPoint> keyPts;

    std::vector<std::vector<cv::KeyPoint>> keyPtss;
    cv::TickMeter tm;
    while (cv::waitKey(1) != 27)
    {
        if (!Camera.read(frame))
        {
            std::cerr << "No frames grabbed!\n";
            break;
        }
        tm.start();
        cv::Mat TFRAME = frame.clone();
        awcv::bgr2gray(TFRAME, TFRAME);
        // cv::GaussianBlur(TFRAME, TFRAME, cv::Size(3, 3), 0.0f, 0.0f);

        cv::Mat description;
        // corners.clear();
        // cv::goodFeaturesToTrack(TFRAME, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, useHarrisDetector, k);
        // cv::KeyPoint::convert(corners, keyPts);
        orb->detectAndCompute(TFRAME, cv::Mat(), keyPts, description); // ORB

        try
        {
            std::vector<cv::DMatch> matches;
            bfMatcher->match(descTemplete, description, matches); // matches和drawMatch顺序要一致

            // flann->knnMatch(description, matchss, 10, cv::Mat());

            //// find good matched points
            // double minDist = 1, maxDist = 10;
            // for (size_t i = 0; i < matches.size(); i++)
            //{
            //     double dist = matches[i].distance;
            //     //std::cout << "距离：" << dist << std::endl;
            //     if (dist > maxDist)
            //         maxDist = dist;
            //     if (dist < minDist)
            //         minDist = dist;
            // }

            // std::vector<cv::DMatch> goodMatches;
            // for (size_t i = 0; i < matches.size(); i++)
            //{
            //     double dist = matches[i].distance;
            //     if (dist < max(3 * minDist, 0.02))
            //     {
            //         goodMatches.push_back(matches[i]);
            //     }
            // }
            ////std::cout << goodMatches.size() << std::endl;
            cv::drawMatches(templete, keyTemplete, frame, keyPts, matches, Matches);
            cv::imshow("match", Matches);
        }
        catch (cv::Exception e)
        {
            std::cout << e.msg << std::endl;
        }

        cv::drawKeypoints(frame, keyPts, frame, cv::Scalar(0, 0, 255));
        tm.stop();
        // cv::waitKey(0);
        cv::putText(frame, cv::format("FPS: %.2f", tm.getFPS()), cv::Point2i(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0));
        cv::imshow("draw", frame);
        tm.reset();
    }
}
#pragma endregion
#endif // !SEDT_TEST
