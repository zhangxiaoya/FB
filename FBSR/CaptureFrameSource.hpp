#pragma once

#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>


#include "FrameSource.h"

using namespace cv;

class CaptureFrameSource : public FrameSource
{
public:
	void nextFrame(OutputArray frame);

protected:
	VideoCapture vc_;

private:
	Mat frame_;

};

void CaptureFrameSource::nextFrame(OutputArray _frame)
{
	if (_frame.kind() == _InputArray::MAT)
	{
		vc_ >> _frame.getMatRef();
	}
	else
	{
		//some thing wrong
	}
}