#pragma once

#include <opencv2/core/core.hpp>

#include "FrameBuffer/FrameBuffer.h"
#include "../FrameSource/FrameSource.h"
#include "../PropsStruct.h"

using namespace std;
using namespace cv;

class SuperResolutionBase
{
public:

	explicit SuperResolutionBase(int bufferSize = 8);

	bool SetFrameSource(const cv::Ptr<FrameSource>& frameSource);

	void SetProps(double alpha, double beta, double lambda, double P, int maxIterationCount);

	void SetSRFactor(int sr_factor);

	bool Reset();

	int NextFrame(OutputArray outputFrame);

	void SetBufferSize(int buffer_size);

protected:
	void Init(Ptr<FrameSource>& frameSource);

	void UpdateZAndA(Mat& mat, Mat& A, int x, int y, const vector<bool>& index, const vector<Mat>& mats, const int len) const;

	void MedianAndShift(const vector<Mat>& interp_previous_frames, const vector<vector<double>>& current_distances, const Size& new_size, Mat& mat, Mat& mat1) const;

	Mat FastGradientBackProject(const Mat& Xn, const Mat& Z, const Mat& A, const Mat& hpsf);

	Mat GradientRegulization(const Mat& Xn, const double p, const double alpha) const;

	Mat FastRobustSR(const vector<Mat>& interp_previous_frames, const vector<vector<double>>& current_distances, Mat hpsf);

	int UpdateFrameBuffer();

	int Process(OutputArray output);

	vector<Mat> NearestInterp2(const vector<Mat>& previousFrames, const vector<vector<double>>& currentDistances) const;

private:
	static vector<vector<double> > RegisterImages(vector<Mat>& frames);

	void GetRestDistance(const vector<vector<double>>& roundedDistances, vector<vector<double>>& restedDistances, int srFactor) const;

	void RoundAndScale(const vector<vector<double>>& registeredDistances, vector<vector<double>>& roundedDistances, int srFactor) const;

	void ModAndAddFactor(vector<vector<double>>& roundedDistances, int srFactor) const;

	void ReCalculateDistances(const vector<vector<double>>& registeredDistances, vector<vector<double>>& roundedDistances, vector<vector<double>>& restedDistances) const;

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
