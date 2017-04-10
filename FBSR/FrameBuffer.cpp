
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
	head += 1;
	if (head >= bufferSize)
		head %= bufferSize;

	frame.copyTo(sourceFrames[head]);
}