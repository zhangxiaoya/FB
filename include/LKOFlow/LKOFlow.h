#ifndef __LKOFLOW_H__
#define __LKOFLOW_H__

#include <opencv2/core.hpp>
#include <vector>

class LKOFlow
{
public:
	static std::vector<double> PyramidalLKOpticalFlow(cv::Mat& img1, cv::Mat& img2, cv::Rect& ROI);

	static void Meshgrid(const float lefTopX, const float rightBottomX, const float lefTopY, const float rightBottomY, cv::Mat& X, cv::Mat& Y);

private:
	static void GaussianDownSample(std::vector<cv::Mat>::const_reference srcMat, std::vector<cv::Mat>::reference destMat);

	static void GaussianPyramid(cv::Mat& img, std::vector<cv::Mat>& pyramid, int levels);

	static void IterativeLKOpticalFlow(cv::Mat& Pyramid1, cv::Mat& Pyramid2, cv::Point topLeft, cv::Point bottomRight, std::vector<double>& disc);

	static void ComputeLKFlowParms(cv::Mat& img, cv::Mat& Ht, cv::Mat& G);

	static cv::Mat MergeTwoRows(cv::Mat& up, cv::Mat& down);

	static cv::Mat MergeTwoCols(cv::Mat left, cv::Mat right);

	static cv::Mat ResampleImg(cv::Mat& img, cv::Rect& rect, std::vector<double> disc);

};

#endif // __LKOFLOW_H__