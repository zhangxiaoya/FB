#pragma once

#include <opencv2\core\core.hpp>
#include <vector>

using namespace std;
using namespace cv;

class FrameBuffer
{
public:
	FrameBuffer(int bufferSize = 8);
	~FrameBuffer();

	void Push(Mat& frame);

	Mat& CurrentFrame();
	Mat& PreviousFrame();

protected:
	int head;
	int bufferSize;
	Mat currentFrame;
	Mat previousFrame;
	vector<Mat> sourceFrames;
};