#pragma once
#include <core/core.hpp>
#include <vector>
#include <highgui/highgui.hpp>

using namespace std;
using namespace cv;

class ReadBUAAImageList
{
public:
	static void ReadImageList(vector<cv::Mat>& imageList, int imageCount);
};

inline void ReadBUAAImageList::ReadImageList(vector<cv::Mat>& imageList, int imageCount)
{
	if (imageCount != imageList.size())
		return;

	for (auto i = 0; i < imageCount; ++i)
	{
		char name[30];
//		snprintf(name, sizeof(name), "Data/DownScale/down%d.png", i);
		snprintf(name, sizeof(name), "Data/fog_low_gray/%d.png", i);
		
		string fullName(name);
		auto curImg = imread(fullName, CV_LOAD_IMAGE_GRAYSCALE);

		Mat floatGrayImg;
		curImg.convertTo(floatGrayImg, CV_32FC1);

		floatGrayImg.copyTo(imageList[i]);
	}
}
