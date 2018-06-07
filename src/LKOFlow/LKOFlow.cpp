#include "LKOFlow/LKOFlow.h"
#include "Utils/Utils.h"

#include <opencv2/imgproc.hpp>
#include <cmath>

std::vector<double> LKOFlow::PyramidalLKOpticalFlow(cv::Mat& img1, cv::Mat& img2, cv::Rect& ROI)
{
	auto ROISize = ROI.size();

	auto levels = std::min(6, static_cast<int>(floor(log2(std::min(ROISize.height, ROISize.width)) - 2)));

	std::vector<cv::Mat> image1Pyramid;
	std::vector<cv::Mat> image2Pyramid;
	image1Pyramid.resize(levels);
	image2Pyramid.resize(levels);

	GaussianPyramid(img1, image1Pyramid, levels);
	GaussianPyramid(img2, image2Pyramid, levels);

	std::vector<double> distance = {0.0,0.0};

	for (auto currentLevel = levels - 1; currentLevel >= 0; --currentLevel)
	{
		distance[0] *= 2;
		distance[1] *= 2;

		auto scale = pow(2, currentLevel);

		cv::Point topLeft;
		topLeft.x = std::max(static_cast<int>(ceil(ROI.x / scale)), 1);
		topLeft.y = std::max(static_cast<int>(ceil(ROI.y / scale)), 1);

		cv::Size currentSize;
		currentSize.width = floor(ROISize.width / scale);
		currentSize.height = floor(ROISize.height / scale);

		cv::Point bottomRight;
		bottomRight.x = std::min(topLeft.x + currentSize.width - 1, image1Pyramid[currentLevel].size().width - 1);
		bottomRight.y = std::min(topLeft.y + currentSize.height - 1, image1Pyramid[currentLevel].size().height - 1);

		IterativeLKOpticalFlow(image1Pyramid[currentLevel], image2Pyramid[currentLevel], topLeft, bottomRight, distance);
	}

	return distance;
}

void LKOFlow::GaussianPyramid(cv::Mat& img, std::vector<cv::Mat>& pyramid, int levels)
{
	img.copyTo(pyramid[0]);

	for (auto i = 1; i < levels; ++i)
		GaussianDownSample(pyramid[i - 1], pyramid[i]);
}

void LKOFlow::IterativeLKOpticalFlow(cv::Mat& img1, cv::Mat& img2, cv::Point topLeft, cv::Point bottomRight, std::vector<double>& distance)
{
	auto oldDistance = distance;

	auto maxIterativeCount = 10;
	auto stopThrashold = 0.01;
	cv::Rect ROIRect(topLeft, bottomRight);
	auto img1Rect = img1(ROIRect);

	cv::Mat Ht, G;
	ComputeLKFlowParms(img1, Ht, G);

	auto currentIterativeIndex = 1;
	double normDistrance = 1;
	while (currentIterativeIndex < maxIterativeCount && normDistrance > stopThrashold)
	{
		auto resample_img = ResampleImg(img2, ROIRect, distance);
		cv::Mat It = img1Rect - resample_img;

		auto newIt = Utils::ReshapedMatColumnFirst(It);

		cv::Mat b = Ht * newIt;

		cv::Mat dc = G.inv() * b;
		normDistrance = norm(dc);

		distance[0] += dc.at<float>(0, 0);
		distance[1] += dc.at<float>(1, 0);

		currentIterativeIndex++;
	}
}

void LKOFlow::ComputeLKFlowParms(cv::Mat& img, cv::Mat& Ht, cv::Mat& G)
{
	cv::Mat SobelX, SobelY;
	Sobel(img, SobelX, CV_32F, 1, 0);
	Sobel(img, SobelY, CV_32F, 0, 1);

	cv::Mat kernelX = (cv::Mat_<char>(3, 3) << 1 , 0 , -1 , 2 , 0 , -2 , 1 , 0 , -1);
	cv::Mat kernelY = kernelX.t();

	cv::Mat SSobelX, SSobelY;
	//	filter2D(img, SSobelX, CV_32F, kernelX, Point(-1, -1), 0, cv::BORDER_REFLECT101);
	filter2D(img, SSobelX, CV_32F, kernelX, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);
	//	filter2D(img, SSobelY, CV_32F, kernelY, Point(-1, -1), 0, cv::BORDER_REFLECT101);
	filter2D(img, SSobelY, CV_32F, kernelY, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);

	auto rectSobelX = SSobelX(cv::Rect(1, 1, SobelX.cols - 2, SobelX.rows - 2));
	auto rectSobelY = SSobelY(cv::Rect(1, 1, SobelY.cols - 2, SobelY.rows - 2));

	cv::Mat deepCopyedX, deepCopyedY;
	rectSobelX.copyTo(deepCopyedX);
	rectSobelY.copyTo(deepCopyedY);

	auto reshapedX = Utils::ReshapedMatColumnFirst(deepCopyedX);
	auto reshapedY = Utils::ReshapedMatColumnFirst(deepCopyedY);

	auto H = MergeTwoCols(reshapedX, reshapedY);
	Ht = H.t();

	G = Ht * H;
}

cv::Mat LKOFlow::MergeTwoRows(cv::Mat& up, cv::Mat& down)
{
	auto totalRows = up.rows + down.rows;

	cv::Mat mergedMat(totalRows, up.cols, up.type());

	auto submat = mergedMat.rowRange(0, up.rows);
	up.copyTo(submat);
	submat = mergedMat.rowRange(up.rows, totalRows);
	down.copyTo(submat);

	return mergedMat;
}

cv::Mat LKOFlow::MergeTwoCols(cv::Mat left, cv::Mat right)
{
	auto totalCols = left.cols + right.cols;

	cv::Mat mergedDescriptors(left.rows, totalCols, left.type());

	auto submat = mergedDescriptors.colRange(0, left.cols);
	left.copyTo(submat);
	submat = mergedDescriptors.colRange(left.cols, totalCols);
	right.copyTo(submat);

	return mergedDescriptors;
}

cv::Mat LKOFlow::ResampleImg(cv::Mat& img, cv::Rect& rect, std::vector<double> disc)
{
	cv::Mat X, Y;
	auto leftTop = rect.tl();
	auto bottomeRight = rect.br();

	Meshgrid(leftTop.x - disc[0], bottomeRight.x - 1 - disc[0], leftTop.y - disc[1], bottomeRight.y - 1 - disc[1], X, Y);

	cv::Mat result;
	remap(img, result, X, Y, cv::INTER_LINEAR);

	return result;
}

void LKOFlow::Meshgrid(const float lefTopX, const float rightBottomX, const float lefTopY, const float rightBottomY, cv::Mat& X, cv::Mat& Y)
{
	std::vector<float> t_x, t_y;

	for (auto i = lefTopX; (i - rightBottomX) < 0.001; i++)
		t_x.push_back(i);
	for (auto j = lefTopY; (j - rightBottomY) < 0.001; j++)
		t_y.push_back(j);

	cv::repeat(cv::Mat(t_x).t(), t_y.size(), 1, X);
	cv::repeat(cv::Mat(t_y), 1, t_x.size(), Y);
}

void LKOFlow::GaussianDownSample(std::vector<cv::Mat>::const_reference srcMat, std::vector<cv::Mat>::reference destMat)
{
	cv::Mat kernel = (cv::Mat_<float>(1, 5) << 0.0625 , 0.2500 , 0.3750 , 0.2500 , 0.0625);
	cv::Mat kernelT = kernel.t();

	cv::Mat img, imgT;
	filter2D(srcMat, img, CV_32F, kernel, cv::Point(-1, -1), 0, cv::BORDER_REFLECT);
	filter2D(img, imgT, CV_32F, kernelT, cv::Point(-1, -1), 0, cv::BORDER_REFLECT);

	cv::Size size(ceil(srcMat.cols / 2.0), ceil(srcMat.rows / 2.0));
	cv::Mat tempImg(size, CV_32FC1);

	for (auto r = 0; r < imgT.rows; r += 2)
	{
		auto rowSrcMat = imgT.ptr<float>(r);
		auto rowDstmat = tempImg.ptr<float>(ceil(r / 2.0));

		for (auto c = 0; c < imgT.cols; c += 2)
		{
			int idx = ceil(c / 2.0);
			rowDstmat[idx] = rowSrcMat[c];
		}
	}

	tempImg.copyTo(destMat);
}
