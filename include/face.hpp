//--------------------------------------------------------------------------------------------------------------------------------------
#pragma once // 防止重复编译
#ifndef H_AWCV_FACE
#define H_AWCV_FACE
//--------------------------------------------------------------------------------------------------------------------------------------
#include <opencv2/opencv.hpp>

namespace awcv
{
namespace face
{
//--------------------------------------------------------------------------------------------------------------------------------------
//												FaceDetector_DNN类
//--------------------------------------------------------------------------------------------------------------------------------------
class FaceDetectorDNN
{
  public:
    class param
    {
      public:
        param()
            : backendId(0), targetId(0), scoreThreshold(0.9f), nmsThreshold(0.3f), topK(5000), save(false), vis(true)
        {
        }
        param(int BackendId, int TargetId, float ScoreThreshold, float NmsThreshold, int TopK, bool Save, bool Vis)
        {
            this->backendId = BackendId;
            this->targetId = TargetId;
            this->scoreThreshold = ScoreThreshold;
            this->nmsThreshold = NmsThreshold;
            this->topK = TopK;
            this->save = Save;
            this->vis = Vis;
        }
        ~param()
        {
        }
        inline int getBackendId()
        {
            return this->backendId;
        }
        inline int getTragetId()
        {
            return this->targetId;
        }
        inline float getScoreThreshold()
        {
            return this->scoreThreshold;
        }
        inline float getNmsThreshold()
        {
            return this->nmsThreshold;
        }
        inline int getTopK()
        {
            return this->topK;
        }
        inline bool getSave()
        {
            return this->save;
        }
        inline bool getVis()
        {
            return this->vis;
        }

      private:
        int backendId;
        int targetId;
        float scoreThreshold;
        float nmsThreshold;
        int topK;
        bool save;
        bool vis;
    };
    class face
    {
      public:
        face(cv::Mat faces, cv::Size size)
        {
            if (faces.size() == cv::Size(0, 0))
            {
                // 说明没有人脸
                this->hasFace = false;
            }
            else
            {
                this->hasFace = true;
                int maxArg = 0;
                int maxArea = 0;
                cv::Rect maxRect;
                for (int i = 0; i < faces.rows; i++)
                {
                    float x = (faces.at<float>(i, 0));
                    float y = (faces.at<float>(i, 1));
                    int width = static_cast<int>(faces.at<float>(i, 2));
                    int height = static_cast<int>(faces.at<float>(i, 3));
                    // 防止矩形框超出范围
                    if (x < 0)
                    {
                        width += static_cast<int>(x);
                        x = 0;
                    }
                    if (y < 0)
                    {
                        height += static_cast<int>(y);
                        y = 0;
                    }
                    if (x + width > size.width - 1)
                    {
                        width = size.width - 1 - static_cast<int>(x);
                    }
                    if (y + height > size.height - 1)
                    {
                        height = size.height - 1 - static_cast<int>(y);
                    }
                    cv::Rect cur = cv::Rect2i(static_cast<int>(x), static_cast<int>(y), width, height); // 脸部矩形框
                    if (cur.area() > maxArea)
                    {
                        maxArea = cur.area();
                        maxArg = i;
                        maxRect = cv::Rect2i(static_cast<int>(x), static_cast<int>(y), width, height); // 保存最大的矩形框
                    }
                }
                this->faceRegion = maxRect;
                this->leftEye = cv::Point2f((faces.at<float>(maxArg, 4)), (faces.at<float>(maxArg, 5)));
                this->rightEye = cv::Point2f((faces.at<float>(maxArg, 6)), (faces.at<float>(maxArg, 7)));
                this->nose = cv::Point2f((faces.at<float>(maxArg, 8)), (faces.at<float>(maxArg, 9)));
                this->leftMouth = cv::Point2f((faces.at<float>(maxArg, 10)), (faces.at<float>(maxArg, 11)));
                this->rightMouth = cv::Point2f((faces.at<float>(maxArg, 12)), (faces.at<float>(maxArg, 13)));
                this->faceScore = faces.at<float>(maxArg, 14);
            }
        }
        ~face()
        {
        }
        inline bool getHasFace()
        {
            return this->hasFace;
        }
        inline cv::Rect getFaceRegion()
        {
            return this->faceRegion;
        }
        inline cv::Point2f getLeftEye()
        {
            return this->leftEye;
        }
        inline cv::Point2f getRightEye()
        {
            return this->rightEye;
        }
        inline cv::Point2f getNose()
        {
            return this->nose;
        }
        inline cv::Point2f getLeftMouth()
        {
            return this->leftMouth;
        }
        inline cv::Point2f getRightMouth()
        {
            return this->rightMouth;
        }
        inline float getFaceScore()
        {
            return this->faceScore;
        }

      private:
        bool hasFace = false;
        cv::Rect faceRegion;    // 人脸矩形框
        cv::Point2f leftEye;    // 图像左眼
        cv::Point2f rightEye;   // 图像右眼
        cv::Point2f nose;       // 鼻子
        cv::Point2f leftMouth;  // 图像左嘴角
        cv::Point2f rightMouth; // 图像右嘴角
        float faceScore = 0.0;  // 人脸分数
    };

    FaceDetectorDNN(std::string ModelPath, FaceDetectorDNN::param Param);
    ~FaceDetectorDNN();

    void setInputSize(cv::Size Size);
    FaceDetectorDNN::face detect(cv::Mat InMat);
    cv::Mat visualize(cv::Mat InMat,
                      FaceDetectorDNN::face Aface,
                      bool PrintFlag = false,
                      double FPS = -1,
                      int Thickness = 2);

  private:
    cv::Ptr<cv::FaceDetectorYN> detector;
    cv::Ptr<cv::CLAHE> clahe; // 直方图均衡，也会改变原图
};
} // namespace face
} // namespace awcv

#endif
//--------------------------------------------------------------------------------------------------------------------------------------