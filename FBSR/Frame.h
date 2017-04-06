#pragma once

#include <opencv2\core\core.hpp>

class Frame
{
public:
	virtual ~Frame() {}

	virtual void nextFrame(cv::OutputArray frame) = 0;
	virtual void reset() = 0;
};

