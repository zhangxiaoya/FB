#pragma once

#include <opencv2\core\core.hpp>

#include "FrameSource\FrameSource.h"

using namespace cv;

class MultiFrameProcesser
{
protected:
	void SuperResolutionProcess(OutputArray outputFrame);

};

void MultiFrameProcesser::SuperResolutionProcess(OutputArray ouputFrame)
{
	// do nothing now, will finish soon
}