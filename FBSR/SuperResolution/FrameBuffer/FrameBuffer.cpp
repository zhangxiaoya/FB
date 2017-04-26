
#include "FrameBuffer.h"
#include <opencv2/imgproc/imgproc.hpp>

FrameBuffer::FrameBuffer(int bufferSize) : head(0), bufferSize(bufferSize)
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

void FrameBuffer::PushGray(Mat& frame)
{
	Mat grayFrame;
	if (frame.channels() == 3)
		cvtColor(frame, grayFrame, CV_BGR2GRAY);
	else
		grayFrame = frame;

	Mat floatGrayFrame;
	grayFrame.convertTo(floatGrayFrame, CV_32FC1);

	floatGrayFrame.copyTo(sourceFrames[head]);
	head += 1;
	if (head >= bufferSize)
		head %= bufferSize;
}

vector<Mat> FrameBuffer::GetAll()
{
	for (auto i = head, j = 0; j < bufferSize;j++)
	{
		returnFrames[j] = sourceFrames[i];
		i += 1;
		i %= bufferSize;
	}
	return returnFrames;
}

Mat& FrameBuffer::CurrentFrame()
{
	auto currentIndex = (head + bufferSize - 1) % bufferSize;
	return sourceFrames[currentIndex];
}

Mat& FrameBuffer::PreviousFrame()
{
	auto previousIndex = (head + bufferSize - 2) % bufferSize;
	return sourceFrames[previousIndex];
}