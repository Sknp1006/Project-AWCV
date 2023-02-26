#include "sfm.hpp"

bool _camera_pov = false;
static void keyboard_callback(const cv::viz::KeyboardEvent &event, void *cookie)
{
    if (event.action == 0 && !event.symbol.compare("s"))
        _camera_pov = !_camera_pov;
};
//--------------------------------------------------------------------------------------------------------------------------------------
//												SFM构造函数
//--------------------------------------------------------------------------------------------------------------------------------------
awcv::sfm::SFM::SFM()
{
}
awcv::sfm::SFM::SFM(std::string FilePath)
{
    this->loadCameraParam(FilePath);
}
//--------------------------------------------------------------------------------------------------------------------------------------
//												SFM析构函数
//--------------------------------------------------------------------------------------------------------------------------------------
awcv::sfm::SFM::~SFM()
{
}
//--------------------------------------------------------------------------------------------------------------------------------------
// 功能:加载相机内参矩阵
// 参数:
//			FilePath:		输入相机内参文件
//--------------------------------------------------------------------------------------------------------------------------------------
void awcv::sfm::SFM::loadCameraParam(std::string FilePath)
{
    // 判断文件是否存在
    if (!boost::filesystem::exists(boost::filesystem::path(FilePath)))
    {
        fprintf(stderr, "相机内参文件不存在：%s", FilePath.c_str());
        return;
    }

    this->CameraParamMat = this->_NdArray2Mat(nc::load<double>(FilePath.c_str()).reshape(3, 3)).clone(); // 加载相机内参矩阵
    this->CameraParamMat(1, 1) = this->CameraParamMat(0, 0);
}
//--------------------------------------------------------------------------------------------------------------------------------------
// 功能:从文件加载跟踪点
// 参数:
//			FilePath:		输入跟踪点文件
//          N_Frames:       输入帧数量
//          N_Tracks:       输入跟踪点数量
//--------------------------------------------------------------------------------------------------------------------------------------
void awcv::sfm::SFM::loadTracks(std::string FilePath, int N_Frames, int N_Tracks)
{
    // 判断文件是否存在
    if (!boost::filesystem::exists(boost::filesystem::path(FilePath)))
    {
        fprintf(stderr, "跟踪点文件不存在：%s", FilePath.c_str());
        return;
    }
    nc::NdArray<double> temp = nc::load<double>(FilePath).reshape(N_Tracks, N_Frames);
    // std::cout << temp << std::endl;
    if (temp.shape().cols % 2 != 0)
    {
        fprintf(stderr, "跟踪点文件不合法：%s", "列数是奇数");
        return;
    }
    int n_frames = temp.shape().cols / 2;
    int n_tracks = temp.shape().rows;
    awcv::sfm::_Tracks tracks;
    for (int i_track = 0; i_track < n_tracks; i_track++)
    {
        awcv::sfm::_Frames track;
        for (int i_frame = 0; i_frame < n_frames; i_frame++)
        {
            float x = temp.at(i_track, i_frame * 2 + 0);
            float y = temp.at(i_track, i_frame * 2 + 1);
            if (x > 0 && y > 0)
                track.push_back(cv::Vec2d(x, y));
            else
                track.push_back(cv::Vec2d(-1));
        }
        tracks.push_back(track);
    }
    this->getTracks(tracks);
}
//--------------------------------------------------------------------------------------------------------------------------------------
// 功能:设置跟踪点
// 参数:
//			InArray:		输入（2xn的跟踪点）容器
//--------------------------------------------------------------------------------------------------------------------------------------
void awcv::sfm::SFM::setTracks(std::vector<cv::Mat> InArray)
{
    this->Points2d.clear(); // 更新前清空
    if (InArray.size() == 0)
    {
        fprintf(stderr, "无跟踪点！");
        return;
    }
    this->N_frames = InArray.size();
    this->N_tracks = InArray[0].cols;

    fprintf(stdout, "共有 %d 帧。\n", this->N_frames);
    fprintf(stdout, "共有 %d 个跟踪点。\n", this->N_tracks);

    this->Points2d = InArray;
}
//--------------------------------------------------------------------------------------------------------------------------------------
// 功能:从_Tracks类型解析跟踪点
// 参数:
//			Tracks:			输入跟踪点数据
//--------------------------------------------------------------------------------------------------------------------------------------
void awcv::sfm::SFM::getTracks(_Tracks Tracks)
{
    this->N_tracks = Tracks.size();    // 跟踪点数量
    this->N_frames = Tracks[0].size(); // 帧数量

    this->Points2d.clear(); // 每次获取都要清空上一次数据
    for (int i_frame = 0; i_frame < this->N_frames; i_frame++)
    {
        cv::Mat_<double> frame(2, this->N_tracks);
        for (int i_track = 0; i_track < this->N_tracks; i_track++)
        {
            frame(0, i_track) = Tracks[i_track][i_frame][0];
            frame(1, i_track) = Tracks[i_track][i_frame][1];
        }
        // std::cout << "frame" << frame << std::endl;
        this->Points2d.push_back(cv::Mat(frame));
    }
}
//--------------------------------------------------------------------------------------------------------------------------------------
// 功能:三维重建
//--------------------------------------------------------------------------------------------------------------------------------------
void awcv::sfm::SFM::reconstruct(std::vector<cv::Mat> &Rs_est, std::vector<cv::Mat> &Ts_est, std::vector<cv::Mat> &points3d_estimated)
{
    bool is_projective = true; // false没有实现
    cv::sfm::reconstruct(this->Points2d, Rs_est, Ts_est, this->CameraParamMat, points3d_estimated, is_projective);
    // Print output
    std::cout << "\n----------------------------\n"
              << std::endl;
    std::cout << "Reconstruction: " << std::endl;
    std::cout << "============================" << std::endl;
    std::cout << "Estimated 3D points: " << points3d_estimated.size() << std::endl;
    std::cout << "Estimated cameras: " << Rs_est.size() << std::endl;
    std::cout << "Refined intrinsics: " << std::endl
              << this->CameraParamMat << std::endl
              << std::endl;

    std::cout << "3D Visualization: " << std::endl;
    std::cout << "============================" << std::endl;
}
//--------------------------------------------------------------------------------------------------------------------------------------
// 功能:显示重建结果
//--------------------------------------------------------------------------------------------------------------------------------------
void awcv::sfm::SFM::show(std::vector<cv::Mat> Rs_est, std::vector<cv::Mat> Ts_est, std::vector<cv::Mat> points3d_estimated)
{
    cv::viz::Viz3d window_est("Estimation Coordinate Frame");
    window_est.setBackgroundColor();                         // black by default
    window_est.registerKeyboardCallback(&keyboard_callback); // 创建3D显示窗口

    // Create the pointcloud
    std::cout << "Recovering points  ... ";

    // recover estimated points3d
    std::vector<cv::Vec3f> point_cloud_est;
    for (int i = 0; i < points3d_estimated.size(); ++i)
        point_cloud_est.push_back(cv::Vec3f(points3d_estimated[i]));

    std::cout << "[DONE]" << std::endl;

    /// Recovering cameras
    std::cout << "Recovering cameras ... ";

    std::vector<cv::Affine3d> path_est;
    for (size_t i = 0; i < Rs_est.size(); ++i)
        path_est.push_back(cv::Affine3d(Rs_est[i], Ts_est[i]));

    std::cout << "[DONE]" << std::endl;

    /// Add cameras
    std::cout << "Rendering Trajectory  ... ";

    /// Wait for key 'q' to close the window
    std::cout << std::endl
              << "Press:                       " << std::endl;
    std::cout << " 's' to switch the camera pov" << std::endl;
    std::cout << " 'q' to close the windows    " << std::endl;

    if (path_est.size() > 0)
    {
        // animated trajectory
        int idx = 0, forw = -1, n = static_cast<int>(path_est.size());

        while (!window_est.wasStopped())
        {
            /// Render points as 3D cubes
            for (size_t i = 0; i < point_cloud_est.size(); ++i)
            {
                cv::Vec3d point = point_cloud_est[i];
                cv::Affine3d point_pose(cv::Mat::eye(3, 3, CV_64F), point);

                char buffer[50];
                #if __APPLE__
                    snprintf(buffer, sizeof(buffer), "%d", static_cast<int>(i));
                # else
                    sprintf_s(buffer, "%d", static_cast<int>(i));
                #endif
                

                cv::viz::WCube cube_widget(cv::Point3f(0.1, 0.1, 0.0), cv::Point3f(0.0, 0.0, -0.1), true, cv::viz::Color::blue());
                cube_widget.setRenderingProperty(cv::viz::LINE_WIDTH, 2.0);
                window_est.showWidget("Cube" + cv::String(buffer), cube_widget, point_pose);
            }

            cv::Affine3d cam_pose = path_est[idx];

            cv::viz::WCameraPosition cpw(0.25);                                                        // Coordinate axes
            cv::viz::WCameraPosition cpw_frustum(this->CameraParamMat, 0.3, cv::viz::Color::yellow()); // Camera frustum

            if (_camera_pov)
                window_est.setViewerPose(cam_pose);
            else
            {
                // render complete trajectory
                window_est.showWidget("cameras_frames_and_lines_est", cv::viz::WTrajectory(path_est, cv::viz::WTrajectory::PATH, 1.0, cv::viz::Color::green()));
                window_est.showWidget("CPW", cpw, cam_pose);
                window_est.showWidget("CPW_FRUSTUM", cpw_frustum, cam_pose);
            }

            // update trajectory index (spring effect)
            forw *= (idx == n || idx == 0) ? -1 : 1;
            idx += forw;

            // frame rate 1s
            window_est.spinOnce(1, true);
            window_est.removeAllWidgets();
        }
    }
}
//--------------------------------------------------------------------------------------------------------------------------------------
// 功能:将NdArray转Mat
// 参数:
//			InArray:			输入内参数组
// 返回值:
//			Mat：				返回3x3的内参Mat
//--------------------------------------------------------------------------------------------------------------------------------------
cv::Mat awcv::sfm::SFM::_NdArray2Mat(nc::NdArray<double> InArray)
{
    int rows = InArray.shape().rows;
    int cols = InArray.shape().cols;
    cv::Mat mat = cv::Mat::zeros(cv::Size(cols, rows), CV_64F);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            mat.at<double>(i, j) = InArray.at(i, j);
        }
    }
    return mat;
}
