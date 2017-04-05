#include <iostream>
#include <opencv.hpp>

using namespace std;
using namespace cv;

int main()
{
	Mat image = imread("timg.jpg");
	if (image.data)
	{
		imshow("Test Image",image);
		waitKey(0);
	}
	return 0;
}