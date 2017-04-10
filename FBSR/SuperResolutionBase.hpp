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

bool SuperResolutionBase::SetFrameSource(const cv::Ptr<FrameSource>& frameSource)
{
	this->frameSource = frameSource;
	this->isFirstRun = true;
	return true;
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