#pragma once

#include "EmptyFrameSource.hpp"
#include "VideoFrameSource.hpp"
#include "ImageListFrameSource.hpp"

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

	static Ptr<FrameSource> createFrameSourceFromImageList(const int& image_count, string file_name_format)
	{
		return  new ImageListFrameSource(image_count, file_name_format);
	}
};

