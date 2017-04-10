#pragma once

#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>

#include "FrameSource\FrameSource.h"

using namespace std;
using namespace cv;

class SuperResolutionBase
{
public:
	bool SetFrameSource(const cv::Ptr<FrameSource>& frameSource);
	bool Reset();

	void NextFrame(OutputArray outputFrame);

protected:
	virtual void Init(Ptr<FrameSource>& frameSource) = 0;
	virtual void Process(Ptr<FrameSource>& frameSource, OutputArray output) = 0;

private:
	Ptr<FrameSource> frameSource;
	bool isFirstRun;
};
