
#include "FrameBuffer.h"

FrameBuffer::FrameBuffer(int bufferSize) : bufferSize(bufferSize)
{
	sourceFrames.resize(this->bufferSize);
}

FrameBuffer::~FrameBuffer()
{
	currentFrame.release();
	previousFrame.release();
	sourceFrames.clear();
}