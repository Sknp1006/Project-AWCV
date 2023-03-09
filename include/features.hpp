#pragma once
#ifndef H_AWCV_FEATURES
#define H_AWCV_FEATURES

#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/xfeatures2d.hpp>

namespace awcv
{
struct SURFPARAM
{
    /**
    @param hessianThreshold Threshold for hessian keypoint detector used in SURF.
    @param nOctaves Number of pyramid octaves the keypoint detector will use.
    @param nOctaveLayers Number of octave layers within each octave.
    @param extended Extended descriptor flag (true - use extended 128-element descriptors; false - use
    64-element descriptors).
    @param upright Up-right or rotated features flag (true - do not compute orientation of features;
    false - compute orientation).
        */
    float scaleFactor = 1.0f;
    double hessianThreshold = 800;
    int nOctaves = 4;
    int nOctaveLayers = 3;
    bool extended = false;
    bool upright = false;
};
struct SURFDATA
{
	std::vector<cv::KeyPoint> KeyPoints;	// 特征
	cv::Mat Description;					// 描述符
};

class SurfMatcher;
typedef std::shared_ptr<SurfMatcher> Ptr_Matcher;

class SurfMatcher
{
public:
	SurfMatcher(SURFPARAM Param = SURFPARAM());
	~SurfMatcher();

	SURFDATA calcSurfData(const cv::Mat& InMat, const cv::Mat& Mask = cv::Mat());	// 计算SURF特征点与特征描述符
	void trainMatcher(const SURFDATA &TemplateData);								// 学习模板特征
	void match(	const SURFDATA& ToMatch, 
				const SURFDATA& Template, 
				std::vector<cv::DMatch> &Matches, 
				cv::Mat& PerspectiveMat, 
				float Threshold=0.5);												// 配准并输出变换矩阵
	bool wrapPerspective(	const cv::Mat& ToMatchImage, 
							cv::Mat& AfterPerspective, 
							const cv::Mat& PerspectiveMat,
							cv::Size MatchSize);									// 透视变换
	void showMatchResult(	const SURFDATA& Template, 
							const cv::Mat & TemplateMat, 
							const SURFDATA& ToMatch, 
							const cv::Mat& ToMatchMat, 
							std::vector<cv::DMatch>& GoodMatches);
private:
	cv::Ptr<cv::xfeatures2d::SURF> SurfHandle;
	cv::Ptr<cv::FlannBasedMatcher> Matcher;
};
}

#endif