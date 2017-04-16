#include "SuperResolutionBase.h"
#include <highgui/highgui.hpp>
#include <contrib/contrib.hpp>
#include "../LKOFlow/LKOFlow.h"
#include <algorithm>

SuperResolutionBase::SuperResolutionBase(int bufferSize) : isFirstRun(false), bufferSize(bufferSize), srFactor(4), psfSize(3), psfSigma(3.0)
{
	this->frameBuffer = new FrameBuffer(bufferSize);
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

bool SuperResolutionBase::Reset()
{
	this->frameSource->reset();
	this->isFirstRun = true;
	return true;
}

void SuperResolutionBase::NextFrame(OutputArray outputFrame)
{
	if (isFirstRun)
	{
		Init(this->frameSource);
		isFirstRun = false;
	}
	Process(this->frameSource, outputFrame);
}

void SuperResolutionBase::Init(Ptr<FrameSource>& frameSource)
{
	Mat currentFrame;
	SetProps(0.7, 1, 0.04, 2, 10);
	for (auto i = 0; i < bufferSize; ++i)
	{
		frameSource->nextFrame(currentFrame);
		frameBuffer->PushGray(currentFrame);
	}

	frameSize = Size(currentFrame.rows, currentFrame.cols);
	currentFrame.release();
}

int SuperResolutionBase::GetTrueCount(const vector<bool>& index)
{
	auto count = 0;
	for (auto curElem : index)
		curElem ? count++ : count;
	return count;
}

float SuperResolutionBase::median(vector<unsigned char>& vector) const
{
	sort(vector.begin(), vector.end());

	auto len = vector.size();
	if (len % 2 != 0)
		return vector[len / 2];
	return float(vector[len / 2] + vector[(len - 1) / 2]) / 2.0;
}

void SuperResolutionBase::MedianThirdDim(const Mat& merged_frame, Mat& median_frame)
{
	for (auto i = 0; i < median_frame.rows; ++i)
	{
		auto rowData = median_frame.ptr<float>(i);
		auto srcRowData = merged_frame.ptr<uchar>(i);
		for (auto j = 0; j < median_frame.cols; ++j)
		{
			vector<uchar> element;
			for (auto k = 0; k < merged_frame.channels(); ++k)
				element.push_back(*(srcRowData + j + k));

			rowData[j] = median(element);
		}
	}
}

void SuperResolutionBase::UpdateZAndA(Mat& Z, Mat& A, int x, int y, const vector<bool>& index, const vector<Mat>& frames, const int len)
{
	vector<Mat> selectedFrames;
	for (auto i = 0; i < index.size(); ++i)
	{
		if (index[i])
			selectedFrames.push_back(frames[i]);
	}
	Mat mergedFrame;
	merge(selectedFrames, mergedFrame);

	Mat medianFrame(frames[0].rows, frames[0].cols, CV_32FC1);
	MedianThirdDim(mergedFrame, medianFrame);

	for (auto r = x - 1; r < Z.rows; r += srFactor)
	{
		for (auto c = y - 1; c < Z.cols; c += srFactor)
		{
			Z.at<float>(r, c) = medianFrame.at<float>(r % srFactor, c % srFactor);
			A.at<float>(r, c) = len;
		}
	}
}

void SuperResolutionBase::MedianAndShift(const vector<Mat>& interp_previous_frames, const vector<vector<double>>& current_distances, const Size& new_size, Mat& Z, Mat& A)
{
	Z = Mat::zeros(new_size, CV_32FC1);
	A = Mat::ones(new_size, CV_32FC1);

	Mat S = Mat::zeros(Size(srFactor, srFactor), CV_8UC1);

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

			auto len = GetTrueCount(index);
			if (len > 0)
			{
				S.at<uchar>(x - srFactor, y - srFactor) = 1;

				UpdateZAndA(Z, A, x, y, index, interp_previous_frames, len);
			}
		}
	}

	Mat noneZeroMap = S == 0;
	vector<int> X, Y;
	for (auto r = 0; r < noneZeroMap.rows; ++r)
	{
		auto perRow = noneZeroMap.ptr<uchar>(r);
		for (auto c = 0; c < noneZeroMap.cols; ++c)
		{
			if (static_cast<int>(perRow[c]) != 0)
			{
				X.push_back(c);
				Y.push_back(r);
			}
		}
	}

	if (X.size() != 0)
	{
		Mat Zmedian;
		medianBlur(Z, Zmedian, srFactor+1);
		auto row = Z.rows;
		auto col = Z.cols;

		for (auto i = 0; i < X.size(); ++i)
		{
			for (auto r = Y[i] + srFactor - 2; r < row; r += srFactor)
			{
				auto perLineOfZ = Z.ptr<float>(r);
				auto perLineOfZmedian = Zmedian.ptr<float>(r);

				for (auto c = X[i] + srFactor - 2; c < col; c += srFactor)
					perLineOfZ[c] = perLineOfZmedian[c];
			}
		}
	}

	Mat copiedA;
	cv::sqrt(A, copiedA);
	copiedA.copyTo(A);
}

void SuperResolutionBase::MySign(const Mat& srcMat, Mat& destMat) const
{
	for (auto r = 0; r < srcMat.rows; ++r)
	{
		auto perLineSrc = srcMat.ptr<uchar>(r);
		auto perLineDest = destMat.ptr<uchar>(r);

		for (auto c = 0; c < srcMat.cols; ++c)
		{
			if (static_cast<int>(perLineSrc[c]) > 0)
				perLineDest[c] = static_cast<uchar>(1);
			else if (static_cast<int>(perLineSrc[c]) < 0)
				perLineDest[c] = static_cast<uchar>(-1);
			else
				perLineDest[c] = static_cast<uchar>(0);
		}
	}
}

Mat SuperResolutionBase::FastGradientBackProject(const Mat& hr, const Mat& Z, const Mat& A, const Mat& hpsf) const
{
	Mat newZ;
	filter2D(Z, newZ, CV_32FC1, hpsf, Point(-1, -1), 0, BORDER_REFLECT);
	Mat dis = newZ - Z;
	Mat resMul = A.mul(dis);

	Mat Gsign(resMul.rows, resMul.cols, CV_32FC1);
	MySign(resMul, Gsign);

	Mat newhpsf;
	flip(hpsf, newhpsf, -1);
	Mat newA = A.mul(Gsign);

	Mat res;
	filter2D(newA, res, CV_32FC1, newhpsf, Point(-1, -1), 0, BORDER_REFLECT);

	return res;
}

Mat SuperResolutionBase::GradientRegulization(const Mat& hr, double p, double alpha)
{
	Mat G = Mat::zeros(hr.rows, hr.cols, CV_32FC1);

	Mat paddedHr;
	copyMakeBorder(hr, paddedHr, p, p, p, p, BORDER_REFLECT);
	for (int i = -1 * p; i <= p; ++i)
	{
		for (int j = -1 * p; j <= p; ++j)
		{
			Rect rectOne(Point(0 + p - i, 0 + p - j), Point(paddedHr.cols - p - i, paddedHr.rows - p - j));
			auto selectMat = paddedHr(rectOne);

			Mat dis = hr - selectMat;

			Mat Xsign(dis.rows, dis.cols, CV_32FC1);
			MySign(dis, Xsign);

			Mat paddedXsign;
			copyMakeBorder(Xsign, paddedXsign, p, p, p, p, BORDER_CONSTANT, 0);

			Rect receTwo(Point(0 + p + i, 0 + p + j), Point(paddedXsign.cols - p + i, paddedXsign.rows - p + j));
			auto selectedXsign = paddedXsign(receTwo);

			Mat diss = Xsign - selectedXsign;

			selectedXsign *= (abs(i) + abs(j));
			Mat tempRes;
			pow(selectedXsign, props.alpha, tempRes);

			G += tempRes;
		}
	}
	return G;
}

Mat SuperResolutionBase::FastRobustSR(const vector<Mat>& interp_previous_frames, const vector<vector<double>>& current_distances, Mat hpsf)
{
	Mat Z, A;
	Size newSize((frameSize.width + 1) * srFactor - 1, (frameSize.height + 1) * srFactor - 1);
	MedianAndShift(interp_previous_frames, current_distances, newSize, Z, A);

	Mat HR;
	Z.copyTo(HR);

	auto iter = 0;

	while (iter < props.maxIterationCount)
	{
		auto Gback = FastGradientBackProject(HR, Z, A, hpsf);
		auto Greg = GradientRegulization(HR, props.P, props.alpha);

		Mat temRes = (Greg + Greg.mul(props.lambda));
		HR -= temRes.mul(props.beta);

		iter = iter + 1;
	}
	return HR;
}

Mat SuperResolutionBase::GetGaussianKernal() const
{
	auto halfSize = (static_cast<int>(psfSigma) - 1) / 2;
	Mat K(static_cast<int>(psfSigma), static_cast<int>(psfSigma), CV_32FC1);

	auto s2 = 2.0 * psfSigma * psfSigma;
	for (auto i = (-halfSize); i <= halfSize; i++)
	{
		auto m = i + halfSize;
		for (auto j = (-halfSize); j <= halfSize; j++)
		{
			auto n = j + halfSize;
			float v = exp(-(1.0 * i * i + 1.0 * j * j) / s2);
			K.ptr<float>(m)[n] = v;
		}
	}
	auto all = sum(K);
	Mat gaussK;
	K.convertTo(gaussK, CV_32FC1, (1 / all[0]));

	return gaussK;
}

vector<Mat> SuperResolutionBase::NearestInterp2(const vector<Mat>& previousFrames, const vector<vector<double>>& distances) const
{
	Mat X, Y;
	LKOFlow::Meshgrid(Range(0, frameSize.width - 1), Range(0, frameSize.height - 1), X, Y);

	vector<Mat> result;
	result.resize(previousFrames.size());

	for (auto i = 0; i < distances.size(); ++i)
	{
		Mat shiftX = X + distances[i][0];
		Mat shiftY = Y + distances[i][0];

		Mat formatX, formatY;
		shiftX.convertTo(formatX, CV_32FC1);
		shiftY.convertTo(formatY, CV_32FC1);

		auto currentFrame = previousFrames[i];
		remap(currentFrame, result[i], formatX, formatY, INTER_NEAREST);
	}
	return result;
}

void SuperResolutionBase::Process(Ptr<FrameSource>& frameSource, OutputArray outputFrame)
{
//	namedWindow("Current Frame");

	Mat currentFrame;

	while (frameBuffer->CurrentFrame().data)
	{
//		imshow("Current Frame", frameBuffer->CurrentFrame());

		auto previous_frames = frameBuffer->GetAll();
		auto currentDistances = RegisterImages(previous_frames);
		auto restDistances = CollectParms(currentDistances);
		auto interpPreviousFrames = NearestInterp2(previous_frames, restDistances);

		auto Hpsf = GetGaussianKernal();

		auto Hr = FastRobustSR(interpPreviousFrames, currentDistances, Hpsf);

		Mat sss;
		Hr.convertTo(sss, CV_8UC1);
		imshow("test", sss);

		/*
		 for (auto i = 0; i < bufferSize; ++i)
		{
			imshow("Previous Frames", PreviousFrames[i]);
			waitKey(100);
		}
		 */

		frameSource->nextFrame(currentFrame);
		frameBuffer->PushGray(currentFrame);
	}

	currentFrame.release();
	destroyAllWindows();
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

vector<vector<double>> SuperResolutionBase::GetRestDistance(const vector<vector<double>>& distances, int srFactor) const
{
	vector<vector<double>> result;
	for (auto i = 0; i < distances.size(); ++i)
	{
		vector<double> distance;
		for (auto j = 0; j < distances[0].size(); ++j)
			distance.push_back(floor(distances[i][j] / srFactor));
		result.push_back(distance);
	}
	return result;
}

void SuperResolutionBase::RoundAndScale(vector<vector<double>>& distances, int srFactor) const
{
	for (auto i = 0; i < distances.size(); ++i)
		for (auto j = 0; j < distances[0].size(); ++j)
			distances[i][j] = round(distances[i][j] * double(srFactor));
}

void SuperResolutionBase::ModAndAddFactor(vector<vector<double>>& distances, int srFactor) const
{
	for (auto i = 0; i < distances.size(); ++i)
		for (auto j = 0; j < distances[0].size(); ++j)
			distances[i][j] = fmod(distances[i][j], static_cast<double>(srFactor)) + srFactor;
}

vector<vector<double>> SuperResolutionBase::CollectParms(vector<vector<double>>& distances) const
{
	RoundAndScale(distances, srFactor);
	auto restDistance = GetRestDistance(distances, srFactor);
	ModAndAddFactor(distances, srFactor);

	return restDistance;
}
