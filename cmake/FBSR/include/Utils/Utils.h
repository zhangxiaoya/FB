#pragma once
#include <vector>
#include <algorithm>
#include <opencv2/core.hpp>

class Utils
{
public:
	static int CalculateCount(const std::vector<bool> value_list, const bool value = true);

	static cv::Mat GetGaussianKernal(const int kernel_size, const double sigma);

	static void CalculatedMedian(const cv::Mat& source_mat, cv::Mat& median_mat);

	static void Sign(const cv::Mat& src_mat, cv::Mat& dest_mat);

	static cv::Mat ReshapedMatColumnFirst(const cv::Mat& srcMat);

	static std::vector<cv::Mat> WarpFrames(const std::vector<cv::Mat>& interp_previous_frames, int borderSize);

	static double Mod(double value, double sr_factor);

private:
	static float GetVectorMedian(std::vector<float>& value_list);

};
