#pragma once
#include <core/core.hpp>
#include <vector>
#include <highgui/highgui.hpp>

using namespace std;
using namespace cv;

class ReadEmilyImageList
{
public:
	static void ReadImageList(vector<cv::Mat>& imageList, int imageCount);
};

inline void ReadEmilyImageList::ReadImageList(vector<cv::Mat>& imageList, int imageCount)
{
	if (imageCount != imageList.size())
		return;

	for (auto i = 1; i <= imageCount; ++i)
	{
		char name[100];
		snprintf(name, sizeof(name), "Emily_samll/%d.jpg", i);
		string fullName(name);
		auto curImg = imread(fullName);

		curImg.copyTo(imageList[i-1]);
	}
}
