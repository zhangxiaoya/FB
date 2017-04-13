#pragma once

#include <opencv2/core/core.hpp>

#include "FrameBuffer/FrameBuffer.h"
#include "../FrameSource/FrameSource.h"
#include "../PropsStruct.h"
#include "../LKOFlow/LKOFlow.h"

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
