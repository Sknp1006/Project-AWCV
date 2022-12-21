#pragma once
#ifndef H_AWCV_SFM
#define H_AWCV_SFM

#include <opencv2/core.hpp>
#define CERES_FOUND 1
#include <NumCpp.hpp>
#include <boost/filesystem.hpp>
#include <opencv2/sfm.hpp>
#include <opencv2/viz.hpp>

namespace awcv
{
namespace sfm
{
typedef std::vector<cv::Vec2d> _Frames; // 一个跟踪点的若干帧数据
typedef std::vector<_Frames> _Tracks;   // 用于三维重建的点
//--------------------------------------------------------------------------------------------------------------------------------------
//												SFM类（三维重建）
//--------------------------------------------------------------------------------------------------------------------------------------
class SFM
{
  public:
    SFM();
    SFM(std::string FilePath);
    ~SFM();

    void loadCameraParam(std::string FilePath); // 【第一步】加载相机内参矩阵
    void loadTracks(std::string FilePath,
                    int N_Frames,
                    int N_Tracks);                // 【第二步】从文件加载跟踪点
    void setTracks(std::vector<cv::Mat> InArray); // 【第二步】直接输入跟踪点（2xN）容器
    void reconstruct(std::vector<cv::Mat> &Rs_est,
                     std::vector<cv::Mat> &Ts_est,
                     std::vector<cv::Mat> &points3d_estimated);
    static void help()
    {
        std::cout
            << "\n------------------------------------------------------------------\n"
            << " This program shows the camera trajectory reconstruction capabilities\n"
            << " in the OpenCV Structure From Motion (SFM) module.\n"
            << " \n"
            << " Usage:\n"
            << "        example_sfm_trajectory_reconstruction <path_to_tracks_file> <f> <cx> <cy>\n"
            << " where: is the tracks file absolute path into your system. \n"
            << " \n"
            << "        The file must have the following format: \n"
            << "        row1 : x1 y1 x2 y2 ... x36 y36 for track 1\n"
            << "        row2 : x1 y1 x2 y2 ... x36 y36 for track 2\n"
            << "        etc\n"
            << " \n"
            << "        i.e. a row gives the 2D measured position of a point as it is tracked\n"
            << "        through frames 1 to 36.  If there is no match found in a view then x\n"
            << "        and y are -1.\n"
            << " \n"
            << "        Each row corresponds to a different point.\n"
            << " \n"
            << "        f  is the focal length in pixels. \n"
            << "        cx is the image principal point x coordinates in pixels. \n"
            << "        cy is the image principal point y coordinates in pixels. \n"
            << "------------------------------------------------------------------\n\n"
            << std::endl;
    }
    void show(std::vector<cv::Mat> Rs_est,
              std::vector<cv::Mat> Ts_est,
              std::vector<cv::Mat> points3d_estimated); // 显示三维重建运动轨迹
  private:
    int N_frames = 0;              // 帧数量
    int N_tracks = 0;              // 跟踪点数量
    std::vector<cv::Mat> Points2d; // 三维重建输入矩阵

    cv::Matx33d CameraParamMat;                        // 3x3内参矩阵
    cv::Mat _NdArray2Mat(nc::NdArray<double> InArray); // double类型的3x3数组转Mat
    void getTracks(_Tracks Tracks);                    // 从_Tracks解析跟踪点
};

} // namespace sfm
} // namespace awcv

#endif // !H_CORE_SFM
