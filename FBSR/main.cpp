#include <iostream>
#include <opencv.hpp>

#include "FrameFactory.h"
#include "FrameSource.h"

using namespace std;
using namespace cv;

int main()
{
	Mat image = imread("timg.jpg");

	Ptr<FrameSource> emptyFrame = FrameFactory::createEmptyFrame();

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