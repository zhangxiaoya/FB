#pragma once

#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <vector>

#include "FrameSource.h"

using namespace std;
using namespace cv;

class VideoFrame
{
public:
	VideoFrame(const int initCacheSize = 10);
	bool SetFrameSource(const cv::Ptr<FrameSource>& frameSource);

protected:
	std::vector<Mat> frameCache;

private:
	int cacheSize;
	Ptr<FrameSource> frameSource;
};


VideoFrame::VideoFrame(const int initCacheSize) : cacheSize(initCacheSize)
{
}

bool VideoFrame::SetFrameSource(const cv::Ptr<FrameSource>& frameSource)
{
	this->frameSource = frameSource;
	return true;
}