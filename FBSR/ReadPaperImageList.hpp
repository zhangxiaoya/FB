#pragma once
#include <core/core.hpp>
#include <vector>
#include <highgui/highgui.hpp>

using namespace std;
using namespace cv;

class ReadPaperImageList
{
public:
	static void ReadImageList(vector<cv::Mat>& imageList, int imageCount);
};

inline void ReadPaperImageList::ReadImageList(vector<cv::Mat>& imageList, int imageCount)
{
	auto startIndex = 0;
	if (imageCount != imageList.size())
		return;

	for (auto i = startIndex; i < (imageCount + startIndex); ++i)
	{
		char name[30];
		snprintf(name, sizeof(name), "Data/paper3_low_gray/%d.png", i);

		string fullName(name);
		auto curImg = imread(fullName, CV_LOAD_IMAGE_GRAYSCALE);

		Mat floatGrayImg;
		curImg.convertTo(floatGrayImg, CV_32FC1);

		floatGrayImg.copyTo(imageList[i- startIndex]);
	}
}
