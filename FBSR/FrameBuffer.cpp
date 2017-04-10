
#include "FrameBuffer.h"

FrameBuffer::FrameBuffer(int bufferSize) : bufferSize(bufferSize), head(0)
{
	sourceFrames.resize(this->bufferSize);
}

FrameBuffer::~FrameBuffer()
{
	currentFrame.release();
	previousFrame.release();
	sourceFrames.clear();
}

void FrameBuffer::Push(Mat& frame)
{
	frame.copyTo(sourceFrames[head]);
	head += 1;
	if (head >= bufferSize)
		head %= bufferSize;
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