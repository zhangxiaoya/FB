#pragma once

#include <opencv2/core/core.hpp>

using namespace cv;

class MultiFrameProcesser
{
protected:
	void SuperResolutionProcess(OutputArray outputFrame);

};

inline void MultiFrameProcesser::SuperResolutionProcess(OutputArray ouputFrame)
{
	// do nothing now, will finish soon
}