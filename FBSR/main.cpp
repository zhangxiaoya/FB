#include <iostream>
#include <opencv.hpp>

#include "FrameFactory.h"
#include "Frame.h"

using namespace std;
using namespace cv;

int main()
{
	Mat image = imread("timg.jpg");

	Ptr<Frame> emptyFrame = FrameFactory::createEmptyFrame();

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