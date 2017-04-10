#include "SuperResolutionBase.h"

SuperResolutionBase::SuperResolutionBase(int bufferSize) : bufferSize(bufferSize)
{
	this->frameBuffer = new FrameBuffer(bufferSize);
}

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
	Mat currentFrame;

	for (int i = 0; i < bufferSize; ++i)
	{
		frameSource->nextFrame(currentFrame);
		frameBuffer->Push(currentFrame);
	}

	currentFrame.release();
}

void SuperResolutionBase::Process(Ptr<FrameSource>& frameSource, OutputArray outputFrame)
{
	/*
	*
	* current process is just show image, will finish in the future
	*
	*/
	namedWindow("Current Frame");
	
	Mat currentFrame;

	while (frameBuffer->CurrentFrame().data)
	{
		imshow("Current Frame", frameBuffer->CurrentFrame());
		waitKey(100);
		imshow("Current Frame", frameBuffer->PreviousFrame());
		waitKey(100);
		frameSource->nextFrame(currentFrame);
		frameBuffer->Push(currentFrame);
	}

	currentFrame.release();
	destroyAllWindows();
}