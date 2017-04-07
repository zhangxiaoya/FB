#pragma once

#include "EmptyFrameSource.hpp"
#include "VideoFrameSource.hpp"

class FrameSourceFactory
{
public:
	static cv::Ptr<FrameSource> createEmptyFrameSource()
	{
		return new EmptyFrameSource();
	}

	static cv::Ptr<FrameSource> createFrameSourceFromVideo(const string& videoFileName)
	{
		return new VideoFrameSource(videoFileName);
	}
};

