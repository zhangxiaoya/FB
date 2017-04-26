#include <highgui/highgui.hpp>
#include <contrib/contrib.hpp>
#include <iostream>

#include "SuperResolutionBase.h"
#include "ReadEmilyImageList.hpp"
#include "../LKOFlow/LKOFlow.h"
#include "../Utils/Utils.hpp"

SuperResolutionBase::SuperResolutionBase(int buffer_size) : isFirstRun(false), bufferSize(buffer_size), srFactor(4), psfSize(3), psfSigma(1.0)
{
}

bool SuperResolutionBase::SetFrameSource(const cv::Ptr<FrameSource>& frameSource)
{
	this->frameSource = frameSource;
	this->isFirstRun = true;
	return true;
}

void SuperResolutionBase::SetProps(double alpha, double beta, double lambda, double P, int maxIterationCount)
{
	props.alpha = alpha;
	props.beta = beta;
	props.lambda = lambda;
	props.P = P;
	props.maxIterationCount = maxIterationCount;
}

void SuperResolutionBase::SetSRFactor(int sr_factor)
{
	srFactor = sr_factor;
}

bool SuperResolutionBase::Reset()
{
	this->frameSource->reset();
	this->isFirstRun = true;
	return true;
}

int SuperResolutionBase::NextFrame(OutputArray outputFrame)
{
	if (isFirstRun)
	{
		Init(this->frameSource);
		isFirstRun = false;
	}
	return Process(outputFrame);
}

void SuperResolutionBase::SetBufferSize(int buffer_size)
{
	bufferSize = buffer_size;
}

void SuperResolutionBase::Init(Ptr<FrameSource>& frameSource)
{
	if(this->frameBuffer)
	{
		delete this->frameBuffer;
		this->frameBuffer = nullptr;
	}
	this->frameBuffer = new FrameBuffer(bufferSize);

	Mat currentFrame;
	for (auto i = 0; i < bufferSize; ++i)
	{
		frameSource->nextFrame(currentFrame);
		frameBuffer->PushGray(currentFrame);
	}

	frameSize = Size(currentFrame.cols, currentFrame.rows);
	currentFrame.release();
}

void SuperResolutionBase::UpdateZAndA(Mat& Z, Mat& A, int x, int y, const vector<bool>& index, const vector<Mat>& frames, const int len) const
{
	vector<Mat> selectedFrames;
	for (auto i = 0; i < index.size(); ++i)
	{
		if (true == index[i])
			selectedFrames.push_back(frames[i]);
	}

	Mat mergedFrame;
	merge(selectedFrames, mergedFrame);

	Mat medianFrame(frames[0].rows, frames[0].cols, CV_32FC1);
	Utils::CalculatedMedian(mergedFrame, medianFrame);

	for (auto r1 = y - 1, r2 = 0; r1 < Z.rows; r1 += srFactor)
	{
		auto rowOfMatZ = Z.ptr<float>(r1);
		auto rowOfMatA = A.ptr<float>(r1);
		auto rowOfMedianFrame = medianFrame.ptr<float>(r2);

		for (auto c1 = x - 1, c2 = 0; c1 < Z.cols; c1 += srFactor)
		{
			rowOfMatZ[c1] = rowOfMedianFrame[c2];
			rowOfMatA[c1] = static_cast<float>(len);
			++c2;
		}
		++r2;
	}
}

void SuperResolutionBase::MedianAndShift(const vector<Mat>& interp_previous_frames, const vector<vector<double>>& current_distances, const Size& new_size, Mat& Z, Mat& A) const
{
	Z = Mat::zeros(new_size, CV_32FC1);
	A = Mat::ones(new_size, CV_32FC1);

	Mat markMat = Mat::zeros(Size(srFactor, srFactor), CV_8UC1);

	for (auto x = srFactor; x < 2 * srFactor; ++x)
	{
		for (auto y = srFactor; y < 2 * srFactor; ++y)
		{
			vector<bool> index;
			for (auto k = 0; k < current_distances.size(); ++k)
			{
				if (current_distances[k][0] == x && current_distances[k][1] == y)
					index.push_back(true);
				else
					index.push_back(false);
			}

			auto len = Utils::CalculateCount(index, true);
			if (len > 0)
			{
				markMat.at<uchar>(x - srFactor, y - srFactor) = 1;
				UpdateZAndA(Z, A, x, y, index, interp_previous_frames, len);
			}
		}
	}

	Mat noneZeroMapOfMarkMat = markMat == 0;

	vector<int> X, Y;
	for (auto r = 0; r < noneZeroMapOfMarkMat.rows; ++r)
	{
		auto perRow = noneZeroMapOfMarkMat.ptr<uchar>(r);
		for (auto c = 0; c < noneZeroMapOfMarkMat.cols; ++c)
		{
			if (static_cast<int>(perRow[c]) == 0)
			{
				X.push_back(c);
				Y.push_back(r);
			}
		}
	}

	if (X.size() != 0)
	{
		Mat meidianBlurMatOfMatZ;
		medianBlur(Z, meidianBlurMatOfMatZ, 3);

		auto rowCount = Z.rows;
		auto colCount = Z.cols;

		for (auto i = 0; i < X.size(); ++i)
		{
			for (auto r = Y[i] + srFactor - 1; r < rowCount; r += srFactor)
			{
				auto rowOfMatZ = Z.ptr<float>(r);
				auto rowOfMedianBlurMatOfMatZ = meidianBlurMatOfMatZ.ptr<float>(r);

				for (auto c = X[i] + srFactor - 0; c < colCount; c += srFactor)
				{
					auto x = rowOfMedianBlurMatOfMatZ[c];
					rowOfMatZ[c] = x;
				}
			}
		}
	}

	Mat rootedMatA;
	sqrt(A, rootedMatA);
	rootedMatA.copyTo(A);
	rootedMatA.release();
}

Mat SuperResolutionBase::FastGradientBackProject(const Mat& Xn, const Mat& Z, const Mat& A, const Mat& hpsf)
{
	Mat matZAfterGaussianFilter;
	filter2D(Xn, matZAfterGaussianFilter, CV_32FC1, hpsf, Point(-1, -1), 0, BORDER_REFLECT);

	Mat diffOfZandMedianFiltedZ;
	subtract(matZAfterGaussianFilter, Z, diffOfZandMedianFiltedZ);

	Mat multiplyOfdiffZAndA = A.mul(diffOfZandMedianFiltedZ);

	Mat Gsign(multiplyOfdiffZAndA.rows, multiplyOfdiffZAndA.cols, CV_32FC1);
	Utils::Sign(multiplyOfdiffZAndA, Gsign);

	Mat inversedHpsf;
	flip(hpsf, inversedHpsf, -1);
	Mat multiplyOfGsingAndMatA = A.mul(Gsign);

	Mat filterResult;
	filter2D(multiplyOfGsingAndMatA, filterResult, CV_32FC1, inversedHpsf, Point(-1, -1), 0, BORDER_REFLECT);

	return filterResult;
}

Mat SuperResolutionBase::GradientRegulization(const Mat& Xn, const double P, const double alpha) const
{
	Mat G = Mat::zeros(Xn.rows, Xn.cols, CV_32FC1);

	Mat paddedXn;
	copyMakeBorder(Xn, paddedXn, P, P, P, P, BORDER_REFLECT);

	for (int i = -1 * P; i <= P; ++i)
	{
		for (int j = -1 * P; j <= P; ++j)
		{
			Rect shiftedXnRect(Point(0 + P - i, 0 + P - j), Point(paddedXn.cols - P - i, paddedXn.rows - P - j));
			auto shiftedXn = paddedXn(shiftedXnRect);

			Mat diffOfXnAndShiftedXn = Xn - shiftedXn;
			Mat signOfDiff(diffOfXnAndShiftedXn.rows, diffOfXnAndShiftedXn.cols, CV_32FC1);
			Utils::Sign(diffOfXnAndShiftedXn, signOfDiff);

			Mat paddedSignOfDiff;
			copyMakeBorder(signOfDiff, paddedSignOfDiff, P, P, P, P, BORDER_CONSTANT, 0);

			Rect shiftedSignedOfDiffRect(Point(0 + P + i, 0 + P + j), Point(paddedSignOfDiff.cols - P + i, paddedSignOfDiff.rows - P + j));
			auto shiftedSignOfDiff = paddedSignOfDiff(shiftedSignedOfDiffRect);

			Mat diffOfSignAndShiftedSign = signOfDiff - shiftedSignOfDiff;

			auto tempScale = pow(alpha, (abs(i) + abs(j)));
			diffOfSignAndShiftedSign *= tempScale;

			G += diffOfSignAndShiftedSign;
		}
	}
	return G;
}

Mat SuperResolutionBase::FastRobustSR(const vector<Mat>& interp_previous_frames, const vector<vector<double>>& current_distances, Mat hpsf)
{
	Mat Z, A;
	auto originalWidth = interp_previous_frames[0].cols;
	auto originalHeight = interp_previous_frames[0].rows;
	Size newSize((originalWidth + 1) * srFactor - 1, (originalHeight + 1) * srFactor - 1);

	MedianAndShift(interp_previous_frames, current_distances, newSize, Z, A);

	Mat HR;
	Z.copyTo(HR);

	auto iter = 1;

	while (iter < props.maxIterationCount)
	{
		auto Gback = FastGradientBackProject(HR, Z, A, hpsf);
		auto Greg = GradientRegulization(HR, props.P, props.alpha);

		Greg *= props.lambda;
		Mat tempResultOfIteration = (Gback + Greg) * props.beta;
		HR -= tempResultOfIteration;

		++iter;
	}
	return HR;
}

int SuperResolutionBase::UpdateFrameBuffer()
{
	Mat currentFrame;
	frameSource->nextFrame(currentFrame);
	if (currentFrame.empty())
		return -1;

	frameBuffer->PushGray(currentFrame);
	return 0;
}

vector<Mat> SuperResolutionBase::NearestInterp2(const vector<Mat>& previousFrames, const vector<vector<double>>& distances) const
{
	Mat X, Y;
	LKOFlow::Meshgrid(0, frameSize.width - 1, 0, frameSize.height - 1, X, Y);

	vector<Mat> result;
	result.resize(previousFrames.size());

	for (auto i = 0; i < distances.size(); ++i)
	{
		auto currentFrame = previousFrames[i];
		remap(currentFrame, result[i], X, Y, INTER_NEAREST);

		if (distances[i][0] < 0.0)
		{
			auto srcSubFrameSharedMemory = result[i](Rect(static_cast<int>(-distances[i][0]), 0, static_cast<int>(result[i].cols + distances[i][0]), result[i].rows));
			Mat tempSubFrame;
			srcSubFrameSharedMemory.copyTo(tempSubFrame);
			auto destSubFrameSharedMemory = result[i](Rect(0, 0, static_cast<int>(result[i].cols + distances[i][0]), result[i].rows));
			tempSubFrame.copyTo(destSubFrameSharedMemory);

			auto totalCols = result[i].cols;
			for (auto r = 0; r < result[i].rows; r++)
			{
				auto rowOfResult = result[i].ptr<float>(r);
				for (auto j = distances[i][0]; j < 0.0; ++j)
					rowOfResult[static_cast<int>((totalCols + j))] = -1;
			}
		}

		if (distances[i][1] < 0.0)
		{
			auto srcSubFrameSharedMemory = result[i](Rect(0, static_cast<int>(-distances[i][1]), result[i].cols, static_cast<int>(result[i].rows) + distances[i][1]));
			Mat tempSubFrame;
			srcSubFrameSharedMemory.copyTo(tempSubFrame);
			auto destSubFrameSharedMemory = result[i](Rect(0, 0, result[i].cols, static_cast<int>(result[i].rows + distances[i][1])));
			tempSubFrame.copyTo(destSubFrameSharedMemory);

			auto totalRows = result[i].rows;
			for (auto j = distances[i][1]; j < 0.0; ++j)
			{
				auto rowOfResult = result[i].ptr<float>(static_cast<int>(totalRows + j));
				for (auto c = 0; c < result[i].cols; ++c)
					rowOfResult[c] = -1;
			}
		}

		if (distances[i][0] > 0.0)
		{
			auto srcSubFrameSharedMemory = result[i](Rect(0, 0, static_cast<int>(result[i].cols - distances[i][0]), result[i].rows));
			Mat tempSubFrame;
			srcSubFrameSharedMemory.copyTo(tempSubFrame);
			auto destSubFrameSharedmemory = result[i](Rect(static_cast<int>(distances[i][0]), 0, static_cast<int>(result[i].cols - distances[i][0]), result[i].rows));
			tempSubFrame.copyTo(destSubFrameSharedmemory);

			auto totalRows = result[i].rows;
			for (auto r = 0; r < totalRows; ++r)
			{
				auto rowOfResult = result[i].ptr<float>(r);
				for (auto c = 0; c < static_cast<int>(distances[i][0]); ++c)
					rowOfResult[c] = -1;
			}
		}

		if (distances[i][1] > 0.0)
		{
			auto srcSubFrameSharedMemory = result[i](Rect(0, 0, result[i].cols, static_cast<int>(result[i].rows - distances[i][1])));
			Mat tempSubFrame;
			srcSubFrameSharedMemory.copyTo(tempSubFrame);
			auto destSubFrameSharedmemory = result[i](Rect(0, static_cast<int>(distances[i][1]), result[i].cols, static_cast<int>(result[i].rows) - distances[i][1]));
			tempSubFrame.copyTo(destSubFrameSharedmemory);

			auto totalCols = result[i].cols;
			for (auto r = 0; r < static_cast<int>(distances[i][1]); ++r)
			{
				auto rowOfResult = result[i].ptr<float>(r);
				for (auto c = 0; c < totalCols; ++c)
					rowOfResult[c] = -1;
			}
		}
	}
	return result;
}

int SuperResolutionBase::Process(OutputArray outputFrame)
{
	auto frameList = this->frameBuffer->GetAll();
	reverse(frameList.begin(), frameList.end());
	auto registeredDistances = RegisterImages(frameList);

	vector<vector<double>> roundedDistances(registeredDistances.size(), vector<double>(2, 0.0));
	vector<vector<double>> restedDistances(registeredDistances.size(), vector<double>(2, 0.0));
	ReCalculateDistances(registeredDistances, roundedDistances, restedDistances);

	auto interpPreviousFrames = NearestInterp2(frameList, restedDistances);

	auto warpedFrames = Utils::WarpFrames(interpPreviousFrames, 2);

	auto Hpsf = Utils::GetGaussianKernal(psfSize, psfSigma);

	auto highResolutionResult = FastRobustSR(warpedFrames, roundedDistances, Hpsf);

	highResolutionResult.convertTo(outputFrame, CV_8UC1);

	return UpdateFrameBuffer();
}

vector<vector<double>> SuperResolutionBase::RegisterImages(vector<Mat>& frames)
{
	vector<vector<double>> result;
	Rect rectROI(0, 0, frames[0].cols, frames[0].rows);

	result.push_back(vector<double>(2, 0.0));

	for (auto i = 1; i < frames.size(); ++i)
	{
		auto currentDistance = LKOFlow::PyramidalLKOpticalFlow(frames[0], frames[i], rectROI);
		result.push_back(currentDistance);
	}

	return result;
}

void SuperResolutionBase::GetRestDistance(const vector<vector<double>>& roundedDistances, vector<vector<double>>& restedDistances, int srFactor) const
{
	for (auto i = 0; i < roundedDistances.size(); ++i)
		for (auto j = 0; j < roundedDistances[0].size(); ++j)
			restedDistances[i][j] = floor(roundedDistances[i][j] / srFactor);
}

void SuperResolutionBase::RoundAndScale(const vector<vector<double>>& registeredDistances, vector<vector<double>>& roundedDistances, int srFactor) const
{
	for (auto i = 0; i < registeredDistances.size(); ++i)
		for (auto j = 0; j < registeredDistances[0].size(); ++j)
			roundedDistances[i][j] = round(registeredDistances[i][j] * double(srFactor));
}

void SuperResolutionBase::ModAndAddFactor(vector<vector<double>>& roundedDistances, int srFactor) const
{
	for (auto i = 0; i < roundedDistances.size(); ++i)
		for (auto j = 0; j < roundedDistances[0].size(); ++j)
			roundedDistances[i][j] = Utils::Mod(roundedDistances[i][j], static_cast<double>(srFactor)) + srFactor;
}

void SuperResolutionBase::ReCalculateDistances(const vector<vector<double>>& registeredDistances, vector<vector<double>>& roundedDistances, vector<vector<double>>& restedDistances) const
{
	// NOTE: Cannot change order

	RoundAndScale(registeredDistances, roundedDistances, srFactor);

	GetRestDistance(roundedDistances, restedDistances, srFactor);

	ModAndAddFactor(roundedDistances, srFactor);
}
