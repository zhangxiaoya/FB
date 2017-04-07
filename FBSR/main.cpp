#include <iostream>
#include <opencv.hpp>

#include "FrameSourceFactory.h"
#include "FrameSource.h"

using namespace std;
using namespace cv;

int main()
{
	Mat image = imread("timg.jpg");

	Ptr<FrameSource> emptyFrame = FrameSourceFactory::createEmptyFrameSource();

	if (image.data)
	{
		imshow("Test Image", image);
		waitKey(0);

		emptyFrame->nextFrame(image);
		emptyFrame->reset();

		destroyAllWindows();
	}
	return 0;
}