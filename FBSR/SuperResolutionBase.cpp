#include "SuperResolutionBase.h"

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

void SuperResolutionBase::Init(Ptr<FrameSource>& frameSource)
{
	for (int i = 0; i < bufferSize; ++i)
	{
		frameSource->nextFrame(currentFrame);
		sourceFrames.push_back(currentFrame);
		currentFrame.copyTo(previousFrame);
	}
}

void SuperResolutionBase::Process(Ptr<FrameSource>& frameSource, OutputArray outputFrame)
{
	namedWindow("Current Frame");
	while (currentFrame.data)
	{
		imshow("Current Frame", currentFrame);
		waitKey(100);

		frameSource->nextFrame(currentFrame);
	}
	destroyAllWindows();
}