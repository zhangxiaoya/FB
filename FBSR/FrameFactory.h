#pragma once

#include "EmptyFrameSource.hpp"

class FrameFactory
{
public:
	CV_EXPORTS static cv::Ptr<FrameSource> createEmptyFrame()
	{
		return new EmptyFrame();
	}
};

