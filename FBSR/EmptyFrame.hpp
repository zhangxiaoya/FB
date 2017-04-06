#pragma once
#include "Frame.h"

class EmptyFrame : public Frame
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