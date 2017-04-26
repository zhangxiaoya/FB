#pragma once

#include <opencv2/core/core.hpp>

class FrameSource
{
public:
	virtual ~FrameSource() {}

	virtual void nextFrame(cv::OutputArray frame) = 0;
	virtual void reset() = 0;
};

