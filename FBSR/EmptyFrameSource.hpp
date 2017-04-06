#pragma once
#include "FrameSource.h"

class EmptyFrame : public FrameSource
{
public:
	void nextFrame(cv::OutputArray frame);
	void reset();
};

void EmptyFrame::nextFrame(cv::OutputArray frame)
{
	frame.release();
}

void EmptyFrame::reset()
{

}