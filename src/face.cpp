#include "core.hpp"
#include "face.hpp"

using namespace awcv::face;
//--------------------------------------------------------------------------------------------------------------------------------------
//												FaceDetector_DNN构造函数
// 参数:
//          ModelPath:          Path to the model.
//          BackendId:          Backend to run on. 0: default, 1: Halide, 2: Intel's Inference Engine, 3: OpenCV, 4: VKCOM, 5: CUDA
//          TargetId:           Target to run on. 0: CPU, 1: OpenCL, 2: OpenCL FP16, 3: Myriad, 4: Vulkan, 5: FPGA, 6: CUDA, 7: CUDA FP16, 8: HDDL
//          ScoreThreshold:     Filter out faces of score < score_threshold.
//          NmsThreshold:       Suppress bounding boxes of iou >= nms_threshold.
//          TopK:               Keep top_k bounding boxes before NMS.
//          Save:               Set true to save results. This flag is invalid when using camera.
//          Vis:                Set true to open a window for result visualization. This flag is invalid when using camera.
//--------------------------------------------------------------------------------------------------------------------------------------
FaceDetectorDNN::FaceDetectorDNN(std::string ModelPath, FaceDetectorDNN::param Param)
{
    int BackendId = Param.getBackendId();
    int TargetId = Param.getTragetId();
    float ScoreThreshold = Param.getScoreThreshold();
    float NmsThreshold = Param.getNmsThreshold();
    int TopK = Param.getTopK();
    // bool Save = Param.getSave();
    // bool Vis = Param.getVis();
    this->detector = cv::FaceDetectorYN::create(ModelPath, "", cv::Size(320, 320), ScoreThreshold, NmsThreshold, TopK, BackendId, TargetId);
    this->clahe = cv::createCLAHE(3.0, cv::Size(8, 8)); // 直方图均衡化算子
}
//--------------------------------------------------------------------------------------------------------------------------------------
//												FaceDetector_DNN析构函数
//--------------------------------------------------------------------------------------------------------------------------------------
FaceDetectorDNN::~FaceDetectorDNN()
{
    this->detector.release();
}
//--------------------------------------------------------------------------------------------------------------------------------------
// 功能:设置检测器的大小
// 参数:
//          Size:          输入大小
//--------------------------------------------------------------------------------------------------------------------------------------
void FaceDetectorDNN::setInputSize(cv::Size Size)
{
    this->detector->setInputSize(Size);
}
//--------------------------------------------------------------------------------------------------------------------------------------
// 功能:人脸检测
// 参数:
//          InMat:          输入图像
//--------------------------------------------------------------------------------------------------------------------------------------
FaceDetectorDNN::face FaceDetectorDNN::detect(cv::Mat InMat)
{
    cv::Mat InMat_ = InMat.clone(); // clahe会修改形参对应的实参，所以要clone
    if (InMat_.channels() == 3)     // 如果是三通道
    {
        cv::cvtColor(InMat_, InMat_, cv::COLOR_BGR2GRAY); // 拆分成单通道
    }
    awcv::autoGammaImage(InMat_, InMat_, 0.6f); // 自适应gamma变换
    this->clahe->apply(InMat_, InMat_);        // 限制对比度自适应直方图均衡化

    std::vector<cv::Mat> merge = {InMat_, InMat_, InMat_};
    cv::merge(merge, InMat_); // 合并三通道

    cv::Mat faces;                         // 【15 x n】
    this->detector->detect(InMat_, faces); // 人脸检测
    // std::cout << faces << std::endl;
    return FaceDetectorDNN::face(faces, InMat_.size()); // 返回人脸数据结构
}
//--------------------------------------------------------------------------------------------------------------------------------------
// 功能:人脸显示函数
// 参数:
//          input:          输入图像
//          faces:          检测到的人脸
//--------------------------------------------------------------------------------------------------------------------------------------
cv::Mat FaceDetectorDNN::visualize(cv::Mat input, FaceDetectorDNN::face Aface, bool PrintFlag, double FPS, int Thickness)
{
    cv::Mat output = input.clone();
    if (FPS > 0)
    {
        cv::putText(output, cv::format("FPS: %.2f", FPS), cv::Point2i(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0));
    }
    if (PrintFlag)
    {
        std::cout << "area: " << Aface.getFaceRegion().area()
                  << ", top-left coordinates: (" << Aface.getFaceRegion().x << ", " << Aface.getFaceRegion().y << "), "
                  << "box width: " << Aface.getFaceRegion().width << ", box height: " << Aface.getFaceRegion().height << ", "
                  << "score: " << Aface.getFaceScore() << "\n";
    }
    cv::rectangle(output, Aface.getFaceRegion(), cv::Scalar(0, 255, 0), Thickness); // 人脸框（绿色）
    // cv::circle(output, Aface.getLeftEye(), 2, cv::Scalar(255, 0, 0), Thickness);        //图像左眼，实际右眼（蓝色）
    // cv::circle(output, Aface.getRightEye(), 2, cv::Scalar(0, 0, 255), Thickness);       //图像右眼，实际左眼（红色）
    // cv::circle(output, Aface.getNose(), 2, cv::Scalar(0, 255, 0), Thickness);           //图像鼻子（绿色）
    // cv::circle(output, Aface.getLeftMouth(), 2, cv::Scalar(255, 0, 255), Thickness);    //图像左嘴角，实际右嘴角（粉色）
    // cv::circle(output, Aface.getRightMouth(), 2, cv::Scalar(0, 255, 255), Thickness);   //图像右嘴角，实际左嘴角（黄色）
    cv::Rect faceRegion = Aface.getFaceRegion();
    cv::circle(output, Aface.getLeftEye(), static_cast<int>(faceRegion.width * 0.15), cv::Scalar(255, 0, 0), 1);  // 左眼
    cv::circle(output, Aface.getRightEye(), static_cast<int>(faceRegion.width * 0.15), cv::Scalar(0, 0, 255), 1); // 右眼
    cv::circle(output, Aface.getNose(), static_cast<int>(faceRegion.width * 0.15), cv::Scalar(0, 255, 0), 1);     // 鼻子

    cv::rectangle(output, cv::Rect(static_cast<int>(Aface.getLeftMouth().x), 
                                            static_cast<int>(Aface.getLeftMouth().y),
                                            static_cast<int>(Aface.getRightMouth().x - Aface.getLeftMouth().x), 
                                            static_cast<int>(Aface.getFaceRegion().height * 0.14)),
                  cv::Scalar::all(255),
                  1); // 嘴巴
    // cv::circle(output, Aface.getLeftMouth(), faceRegion.width * 0.2, cv::Scalar(255, 0, 255), 1);   //嘴巴
    // cv::circle(output, Aface.getRightMouth(), faceRegion.width * 0.2, cv::Scalar(0, 255, 255), 1);  //嘴巴

    cv::putText(output, cv::format("%.4f", Aface.getFaceScore()), cv::Point2i(Aface.getFaceRegion().x, Aface.getFaceRegion().y + 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0));
    return output;
}