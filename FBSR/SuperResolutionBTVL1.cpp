#include "SuperResolutionBTVL1.h"

void SuperResolutionBTVL1::Init(Ptr<FrameSource>& frameSource)
{
	for (int i = 0; i < bufferSize; ++i)
	{
		frameSource->nextFrame(currentFrame);
		Push(currentFrame);
		currentFrame.copyTo(previousFrame);
	}
}

void SuperResolutionBTVL1::Process(Ptr<FrameSource>& frameSource, OutputArray outputFrame)
{
	namedWindow("Current Frame");
	while (currentFrame.data)
	{
		imshow("Current Frame", currentFrame);
		waitKey(100);

		frameSource->nextFrame(currentFrame);
		Push(currentFrame);
		currentFrame.copyTo(previousFrame);
	}
	destroyAllWindows();
}