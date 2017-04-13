#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


#include "FrameSource.h"

using namespace cv;

class CaptureFrameSource : public FrameSource
{
public:
	void nextFrame(OutputArray frame) override;

protected:
	VideoCapture videoCapture;

};

inline void CaptureFrameSource::nextFrame(OutputArray outputFrame)
{
	if (outputFrame.kind() == _InputArray::MAT)
	{
		videoCapture >> outputFrame.getMatRef();
	}
	else
	{
		//some thing wrong
	}
}