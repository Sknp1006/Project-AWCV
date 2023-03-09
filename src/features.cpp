#include "features.hpp"
awcv::SurfMatcher::SurfMatcher(awcv::SURFPARAM Param)
{
	this->SurfHandle = cv::xfeatures2d::SURF::create(	Param.hessianThreshold, 
														Param.nOctaves, 
														Param.nOctaveLayers, 
														Param.extended, 
														Param.upright);
	this->Matcher = cv::FlannBasedMatcher::create();
}
awcv::SurfMatcher::~SurfMatcher()
{
}
//--------------------------------------------------------------------------------------------------------------------------------------
// 功能: 计算图像的SURF特征点与特征描述符
// 参数:
//			InMat:			输入图像（灰度）
//			Mask:			输入掩膜
// 返回值:
//			SURFDATA:		SURF特征数据
//--------------------------------------------------------------------------------------------------------------------------------------
awcv::SURFDATA awcv::SurfMatcher::calcSurfData(const cv::Mat& InMat, const cv::Mat &Mask)
{
	SURFDATA data;
	if (Mask.empty() == true)
	{
		this->SurfHandle->detect(InMat, data.KeyPoints, Mask);
		this->SurfHandle->compute(InMat, data.KeyPoints, data.Description);
	}
	else
	{
		this->SurfHandle->detect(InMat, data.KeyPoints);
		this->SurfHandle->compute(InMat, data.KeyPoints, data.Description);
	}
	return data;
}
//--------------------------------------------------------------------------------------------------------------------------------------
// 功能: 根据模板描述符训练匹配器
// 参数:
//			Data:			输入SURF特征数据
//--------------------------------------------------------------------------------------------------------------------------------------
void awcv::SurfMatcher::trainMatcher(const SURFDATA& TemplateData)
{
	std::vector<cv::Mat> desc_collection(1, TemplateData.Description);
	this->Matcher->add(desc_collection);
	this->Matcher->train();
}
//--------------------------------------------------------------------------------------------------------------------------------------
// 功能: 计算匹配的特征点
// 参数:
//			ToMatch:		输入SURF特征数据
//			Template:		输入模板SURF特征数据
//			GoodMatches:	输出优秀的匹配点
//			PerspectiveMat:	输出透视变换矩阵
//			Threshold:		距离阈值
// 返回值:
//			std::vector<cv::DMatch>&:	满足条件的匹配点
//--------------------------------------------------------------------------------------------------------------------------------------
void awcv::SurfMatcher::match(const SURFDATA& ToMatch, const SURFDATA& Template, std::vector<cv::DMatch> &GoodMatches, cv::Mat& PerspectiveMat, float Threshold)
{
	std::vector<std::vector<cv::DMatch>> Matches;
	this->Matcher->knnMatch(ToMatch.Description, Matches, 2);
	GoodMatches.clear();
	for (unsigned int i = 0; i < Matches.size(); ++i)
	{
		if (Matches[i][0].distance < Threshold * Matches[i][1].distance)
			GoodMatches.push_back(Matches[i][0]);
	}
	// std::cout << "GoodMatches的个数:" << GoodMatches.size() << std::endl;
	std::vector<cv::Point2f> ToMatchPoints, TemplatePoints;
	// int MinGoodMatches = 100;		// 最小优秀匹配数
	int GoodMatchesSize = 100;
	if ((GoodMatches.size() > 4) && (GoodMatches.size() < GoodMatchesSize))
	{
		GoodMatchesSize = GoodMatches.size();
	}
	else if (GoodMatches.size() <= 4)
	{
		PerspectiveMat = cv::Mat::ones(3, 3, CV_64F);
		return;
	}
	for (unsigned int i = 0; i < GoodMatchesSize; i++)
	{
		ToMatchPoints.push_back(ToMatch.KeyPoints[GoodMatches[i].queryIdx].pt);
		TemplatePoints.push_back(Template.KeyPoints[GoodMatches[i].trainIdx].pt);
	}
	PerspectiveMat = cv::findHomography(ToMatchPoints, TemplatePoints, cv::RANSAC);
}
//--------------------------------------------------------------------------------------------------------------------------------------
// 功能: 计算透视变换后的图像
// 参数:
//			ToMatchImage:		输入需配准图像
//			AfterPerspective:	输出配准后图像
//			PerspectiveMat:		输入透视变换矩阵
//			MatchSize:			输入配准后的大小
// 返回值:
//			bool:				返回透视变换是否成功
//--------------------------------------------------------------------------------------------------------------------------------------
bool awcv::SurfMatcher::wrapPerspective(const cv::Mat& ToMatchImage, cv::Mat & AfterPerspective, const cv::Mat & PerspectiveMat, cv::Size MatchSize)
{
	try
	{
		cv::warpPerspective(ToMatchImage, AfterPerspective, PerspectiveMat, MatchSize);
	}
	catch (cv::Exception& e)
	{
		// std::cout << "配准异常" << std::endl;
		return false;
	}
	return true;
}
//--------------------------------------------------------------------------------------------------------------------------------------
// 功能: 显示匹配结果
// 参数:
//			Template:		输入Template的SURF特征数据
//			TemplateMat:	输入模板图像
//			ToMatch:		输入ToMatch的SURF特征数据
//			ToMatchMat:		输入Match图像
//			GoodMatches:	输入匹配点
// 返回值:
//			std::vector<cv::DMatch>&:	满足条件的匹配点
//--------------------------------------------------------------------------------------------------------------------------------------
void awcv::SurfMatcher::showMatchResult(	const SURFDATA& Template,
									const cv::Mat& TemplateMat,
									const SURFDATA& ToMatch,
									const cv::Mat& ToMatchMat,
									std::vector<cv::DMatch>& GoodMatches)
{
	cv::Mat show = cv::Mat::zeros(TemplateMat.size(), CV_8U);
	cv::drawMatches(ToMatchMat, ToMatch.KeyPoints, TemplateMat, Template.KeyPoints, GoodMatches, show);
	std::string WinName = "image[match]";
	try
	{
		int ret = cv::getWindowProperty(WinName.c_str(), 0);
		if (ret == -1)
		{
			cv::namedWindow(WinName.c_str(), cv::WINDOW_NORMAL);
		}
		else
		{
			cv::imshow(WinName.c_str(), show);
		}
	}
	catch (cv::Exception& e)
	{
		cv::namedWindow(WinName.c_str(), cv::WINDOW_NORMAL);
	}
}
