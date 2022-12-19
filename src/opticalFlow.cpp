#include "opticalFlow.hpp"
//--------------------------------------------------------------------------------------------------------------------------------------
//												OpticalFlowDIS构造函数
//--------------------------------------------------------------------------------------------------------------------------------------
awcv::opt::DenseOpticalFlow::DenseOpticalFlow(OFType OfType)
{
    switch (OfType)
    {
    case DENSE_DIS:
        this->initDISOF();
        break;
    case DENSE_PCA:
        this->initPCAOF();
        break;
    case DENSE_RLOF:
        this->initRLOFOF();
        break;
    case DENSE_DEFAULT:
        this->initFarnebackOF();
        break;
    default:
        this->initDISOF();
        break;
    }
}
//--------------------------------------------------------------------------------------------------------------------------------------
//												OpticalFlowDIS析构函数
//--------------------------------------------------------------------------------------------------------------------------------------
awcv::opt::DenseOpticalFlow::~DenseOpticalFlow()
{
}
//--------------------------------------------------------------------------------------------------------------------------------------
// 功能:计算稠密光流矩阵
//          Frame:          输入图像（单通道）
//          Flow:           光流变化矩阵（CV_32FC2）
//--------------------------------------------------------------------------------------------------------------------------------------
void awcv::opt::DenseOpticalFlow::calcOpticalFlow(cv::Mat Frame, cv::Mat &Flow)
{
    if (this->__ofAttr.temp_prev_mat.empty())
    {
        this->__ofAttr.temp_prev_mat = Frame.clone(); // 第一次加载prev_mat
    }
    this->__ofAttr.temp_now_mat = Frame.clone(); // 当前图像
    Flow = cv::Mat::zeros(this->__ofAttr.temp_prev_mat.size(), CV_32FC2);
    this->DISOF->calc(this->__ofAttr.temp_prev_mat, this->__ofAttr.temp_now_mat, Flow);
    cv::swap(this->__ofAttr.temp_prev_mat, this->__ofAttr.temp_now_mat); // now_mat成为下一次的prev_mat
}
//--------------------------------------------------------------------------------------------------------------------------------------
// 功能:计算可视化RGB光流图像、幅度矩阵、角度矩阵
// 参数:
//          Flow:           输入光流矩阵<CV_32FC2>
//          Result:         输出图像<CV_8UC3>
//          Mangitude:      输入输出幅度矩阵
//          Angle:          输入输出角度矩阵
//--------------------------------------------------------------------------------------------------------------------------------------
void awcv::opt::DenseOpticalFlow::calcOpticalFlowMat(cv::Mat Flow, cv::Mat &FLOWIMG, cv::Mat &Mangitude, cv::Mat &Angle)
{
    cv::Mat temp[] = {cv::Mat::zeros(Flow.size(), CV_32FC1),  // x方向
                      cv::Mat::zeros(Flow.size(), CV_32FC1)}; // y方向
    cv::split(Flow, temp);                                    // 拆分CV_32FC2矩阵
    cv::cartToPolar(temp[0], temp[1], Mangitude, Angle);      // 将移动矢量从笛卡尔坐标转换极坐标，输出像素点移动的幅度和相位（弧度）
    cv::Mat H, S, V;                                          // 以相位值来确定色调H，像素点的不同移动方向形成不同颜色
    H = Angle * 180 / CV_PI / 2;                              // 将弧度转为角度，在OpenCV中H取值范围是0~180°
    cv::convertScaleAbs(H, H);                                // 转成CV_8U
    S = cv::Mat::ones(temp[0].size(), CV_8UC1) * 255;         // 饱和度S设置最大255
    cv::normalize(Mangitude, V, 0, 255, cv::NORM_MINMAX);     // 范围归一化
    cv::convertScaleAbs(V, V);                                // 转成CV_8U
    std::vector<cv::Mat> HSV(3);
    HSV[0] = H;
    HSV[1] = S;
    HSV[2] = V;
    cv::merge(HSV, FLOWIMG);                           // 合成三通道
    cv::cvtColor(FLOWIMG, FLOWIMG, cv::COLOR_HSV2BGR); // 转换到RGB色彩空间进行显示
}
//--------------------------------------------------------------------------------------------------------------------------------------
// 功能:手动清空光流矩阵
//--------------------------------------------------------------------------------------------------------------------------------------
void awcv::opt::DenseOpticalFlow::clear()
{
    if (!this->__ofAttr.temp_prev_mat.empty())
        this->__ofAttr.temp_prev_mat.release(); // 清空上一帧数据
}
//--------------------------------------------------------------------------------------------------------------------------------------
// 功能:初始化DIS稠密光流法
//--------------------------------------------------------------------------------------------------------------------------------------
void awcv::opt::DenseOpticalFlow::initDISOF()
{
    this->DISOF = cv::DISOpticalFlow::create(cv::DISOpticalFlow::PRESET_MEDIUM);
    this->DISOF->setFinestScale(1);
}
//--------------------------------------------------------------------------------------------------------------------------------------
// 功能:初始化Farneback稠密光流法
//--------------------------------------------------------------------------------------------------------------------------------------
void awcv::opt::DenseOpticalFlow::initFarnebackOF()
{
    // TODO: 待实现
    return;
}
//--------------------------------------------------------------------------------------------------------------------------------------
// 功能:初始化PCA稠密光流法
//--------------------------------------------------------------------------------------------------------------------------------------
void awcv::opt::DenseOpticalFlow::initPCAOF()
{
    // TODO: 待实现
    return;
}
//--------------------------------------------------------------------------------------------------------------------------------------
// 功能:初始化RLOF稠密光流法
//--------------------------------------------------------------------------------------------------------------------------------------
void awcv::opt::DenseOpticalFlow::initRLOFOF()
{
    // TODO: 待实现
    return;
}
//--------------------------------------------------------------------------------------------------------------------------------------
//												SparseOpticalFlow构造函数
//--------------------------------------------------------------------------------------------------------------------------------------
awcv::opt::SparseOpticalFlow::SparseOpticalFlow()
{
}
//--------------------------------------------------------------------------------------------------------------------------------------
//												SparseOpticalFlow析构函数
//--------------------------------------------------------------------------------------------------------------------------------------
awcv::opt::SparseOpticalFlow::~SparseOpticalFlow()
{
}
//--------------------------------------------------------------------------------------------------------------------------------------
// 功能:计算稀疏光流点
// 参数:
//			Frame:          输入图像
//			Pts:			输入关键点
//			EstimatePts:    输出光流计算点
//			Status:			输出光流点状态矩阵
//			Error:			输出光流点误差矩阵
//--------------------------------------------------------------------------------------------------------------------------------------
void awcv::opt::SparseOpticalFlow::calcOpticalFlowLK(cv::Mat Frame, std::vector<cv::Point2f> Pts, std::vector<cv::Point2f> &EstimatePts, cv::Mat &Status, cv::Mat &Error)
{
    if (this->__ofAttr.lk_prev_mat.empty())
    {
        this->__ofAttr.lk_prev_mat = Frame.clone(); // 第一次加载prev_mat
    }
    this->__ofAttr.lk_now_mat = Frame.clone(); // 当前图像
    if (this->__ofAttr.lk_prev_points.empty())
    {
        this->__ofAttr.lk_prev_points = Pts; // 第一次加载prev_points
    }
    cv::calcOpticalFlowPyrLK(this->__ofAttr.lk_prev_mat, this->__ofAttr.lk_now_mat, this->__ofAttr.lk_prev_points, EstimatePts, Status, Error);
    swap(this->__ofAttr.lk_prev_mat, this->__ofAttr.lk_now_mat);
    swap(this->__ofAttr.lk_prev_points, EstimatePts);
}
//--------------------------------------------------------------------------------------------------------------------------------------
// 功能:手动清除光流矩阵和点
//--------------------------------------------------------------------------------------------------------------------------------------
void awcv::opt::SparseOpticalFlow::clear()
{
    if (!this->__ofAttr.lk_prev_mat.empty())
        this->__ofAttr.lk_prev_mat.release();
    if (!this->__ofAttr.lk_prev_points.empty())
        this->__ofAttr.lk_prev_points.clear();
}
