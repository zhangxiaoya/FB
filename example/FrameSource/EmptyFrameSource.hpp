#pragma once
#include "FrameSource/FrameSource.h"

class EmptyFrameSource : public FrameSource
{
public:
	void nextFrame(cv::OutputArray frame) override;
	void reset() override;
};

inline void EmptyFrameSource::nextFrame(cv::OutputArray frame)
{
	frame.release();
}

inline void EmptyFrameSource::reset()
{

}