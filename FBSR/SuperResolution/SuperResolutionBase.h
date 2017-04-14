#pragma once

#include <opencv2/core/core.hpp>

#include "FrameBuffer/FrameBuffer.h"
#include "../FrameSource/FrameSource.h"
#include "../PropsStruct.h"
#include "../LKOFlow/LKOFlow.h"
#include <algorithm>

using namespace std;
using namespace cv;

class SuperResolutionBase
{
public:

	explicit SuperResolutionBase(int bufferSize = 8);
	bool SetFrameSource(const cv::Ptr<FrameSource>& frameSource);
	void SetProps(double alpha, double beta, double lambda, double P, int maxIterationCount);
	bool Reset();

	void NextFrame(OutputArray outputFrame);

protected:
	void Init(Ptr<FrameSource>& frameSource);
	int GetTrueCount(const vector<bool>& index);
	float median(vector<unsigned char>& vector) const;
	void MedianThirdDim(const Mat& merged_frame, Mat& median_frame);
	void UpdateZAndA(Mat& mat, Mat& A, int x, int y, const vector<bool>& index, const vector<Mat>& mats, const int len);
	void MedianAndShift(const vector<Mat>& interp_previous_frames, const vector<vector<double>>& current_distances, const Size& new_size, Mat& mat, Mat& mat1);
	void MySign(const Mat& srcMat, Mat& destMat) const;
	Mat FastGradientBackProject(const Mat& hr, const Mat& mat, const Mat& mat1, const Mat& hpsf) const;
	void FastRobustSR(const vector<Mat>& interp_previous_frames, const vector<vector<double>>& current_distances, Mat hpsf);
	Mat GetGaussianKernal() const;
	void Process(Ptr<FrameSource>& frameSource, OutputArray output);
	vector<Mat> NearestInterp2(const vector<Mat>& previousFrames, const vector<vector<double>>& currentDistances) const;

private:
	static vector<vector<double> > RegisterImages(vector<Mat>& frames);
	vector<vector<double> > GetRestDistance(const vector<vector<double>>& distances, int srFactor) const;
	void RoundAndScale(vector<vector<double>>& distances, int srFactor) const;

	void ModAndAddFactor(vector<vector<double>>& distances, int srFactor) const;
	vector<vector<double>> CollectParms(vector<vector<double>>& distances) const;

private:
	Ptr<FrameSource> frameSource;
	Ptr<FrameBuffer> frameBuffer;
	bool isFirstRun;
	int bufferSize;
	Size frameSize;

	int srFactor;
	int psfSize;
	double psfSigma;
	PropsStruct props;
};
