#pragma once

#include "EmptyFrame.hpp"

class FrameFactory
{
public:
	CV_EXPORTS static cv::Ptr<Frame> createEmptyFrame()
	{
		return new EmptyFrame();
	}
};

