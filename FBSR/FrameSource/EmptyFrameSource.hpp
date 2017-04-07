#pragma once
#include "FrameSource.h"

class EmptyFrameSource : public FrameSource
{
public:
	void nextFrame(cv::OutputArray frame);
	void reset();
};

void EmptyFrameSource::nextFrame(cv::OutputArray frame)
{
	frame.release();
}

void EmptyFrameSource::reset()
{

}