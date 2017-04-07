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
	bool reset();

	void nextFrame(OutputArray outputFrame);

private:
	Ptr<FrameSource> frameSource;

	bool isFirstRun;
};

bool SuperResolutionBase::SetFrameSource(const cv::Ptr<FrameSource>& frameSource)
{
	this->frameSource = frameSource;

	return true;
}

bool SuperResolutionBase::reset()
{
	this->frameSource->reset();

	return true;
}