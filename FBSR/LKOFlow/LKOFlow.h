#pragma once
#include <opencv2/core/core.hpp>
#include <vector>

using namespace std;
using namespace cv;

class LKOFlow
{
public:
	static vector<double> PyramidalLKOpticalFlow(Mat& img1, Mat& img2, Rect& ROI);

	static void LKOFlow::Meshgrid(const float lefTopX, const float rightBottomX, const float lefTopY, const float rightBottomY, Mat& X, Mat& Y);

private:
	static void GaussianDownSample(vector<Mat>::const_reference srcMat, vector<Mat>::reference destMat);

	static void GaussianPyramid(Mat& img, vector<Mat>& pyramid, int levels);

	static void IterativeLKOpticalFlow(Mat& Pyramid1, Mat& Pyramid2, Point topLeft, Point bottomRight, vector<double>& disc);

	static void ComputeLKFlowParms(Mat& img, Mat& Ht, Mat& G);

	static Mat MergeTwoRows(Mat& up, Mat& down);

	static Mat MergeTwoCols(Mat left, Mat right);

	static Mat ResampleImg(Mat& img, Rect& rect, vector<double> disc);

};
