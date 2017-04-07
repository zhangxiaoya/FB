#pragma once
#include <string>

#include "CaptureFrameSource.hpp"

class VideoFrameSource : public CaptureFrameSource
{
public:
	VideoFrameSource(const string& videoFileName);
	void reset();

private:
	string videoFileName;
};

VideoFrameSource::VideoFrameSource(const string& videoFileName) : videoFileName(videoFileName)
{
	reset();
}

void VideoFrameSource::reset()
{
	videoCapture.release();
	videoCapture.open(videoFileName);
	CV_Assert(videoCapture.isOpened());
}