#include <iostream>
#include <string>
#include <opencv.hpp>

#include "FrameSource\FrameSourceFactory.h"
#include "FrameSource\FrameSource.h"

using namespace std;
using namespace cv;

int main()
{
	const string videoFileName = "768x576.avi";

	Ptr<FrameSource> videoFrameSource = FrameSourceFactory::createFrameSourceFromVideo(videoFileName);

	Mat currentFrame;

	videoFrameSource->nextFrame(currentFrame);

	namedWindow("Current Frame");
	do {
		if (currentFrame.data)
		{
			imshow("Current Frame", currentFrame);
			waitKey(100);
		}

		videoFrameSource->nextFrame(currentFrame);
	} while (currentFrame.data);

	destroyAllWindows();

	return 0;
}