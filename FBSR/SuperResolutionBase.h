#pragma once

#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>

#include "FrameBuffer.h"
#include "FrameSource\FrameSource.h"

using namespace std;
using namespace cv;

class SuperResolutionBase
{
public:

	SuperResolutionBase(int bufferSize = 8);
	bool SetFrameSource(const cv::Ptr<FrameSource>& frameSource);
	bool Reset();

	void NextFrame(OutputArray outputFrame);

protected:
	void Init(Ptr<FrameSource>& frameSource);
	void Process(Ptr<FrameSource>& frameSource, OutputArray output);

private:
	Ptr<FrameSource> frameSource;
	Ptr<FrameBuffer> frameBuffer;
	bool isFirstRun;
	int bufferSize;
};
