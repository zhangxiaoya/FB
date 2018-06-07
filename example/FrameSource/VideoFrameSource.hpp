#pragma once
#include <string>

#include "CaptureFrameSource.hpp"

class VideoFrameSource : public CaptureFrameSource
{
public:
	explicit VideoFrameSource(const string& videoFileName);
	void reset() override;

private:
	string videoFileName;
};

inline VideoFrameSource::VideoFrameSource(const string& videoFileName) : videoFileName(videoFileName)
{
	reset();
}

inline void VideoFrameSource::reset()
{
	videoCapture.release();
	videoCapture.open(videoFileName);
	CV_Assert(videoCapture.isOpened());
}