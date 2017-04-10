
#include "FrameBuffer.h"

FrameBuffer::FrameBuffer(int bufferSize) : bufferSize(bufferSize), head(0)
{
	sourceFrames.resize(this->bufferSize);
	returnFrames.resize(this->bufferSize);
}

FrameBuffer::~FrameBuffer()
{
	currentFrame.release();
	previousFrame.release();
	sourceFrames.clear();
	returnFrames.clear();
}

void FrameBuffer::Push(Mat& frame)
{
	frame.copyTo(sourceFrames[head]);
	head += 1;
	if (head >= bufferSize)
		head %= bufferSize;
}

vector<Mat> FrameBuffer::GetAll()
{
	for (int i = head, j = 0; j < bufferSize;j++)
	{
		returnFrames[j] = sourceFrames[i];
		i += 1;
		i %= bufferSize;
	}
	return returnFrames;
}

Mat& FrameBuffer::CurrentFrame()
{
	int currentIndex = (head + bufferSize - 1) % bufferSize;
	return sourceFrames[currentIndex];
}

Mat& FrameBuffer::PreviousFrame()
{
	int previousIndex = (head + bufferSize - 2) % bufferSize;
	return sourceFrames[previousIndex];
}