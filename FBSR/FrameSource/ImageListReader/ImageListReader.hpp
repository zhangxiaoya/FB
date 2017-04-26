#pragma once
#include <core/core.hpp>
#include <vector>
#include <highgui/highgui.hpp>
#include "ImageListReaderBase.hpp"

class ImageListReader : public ImageListReaderBase
{
public:
	static void ReadImageList(std::vector<cv::Mat>& imageList, int imageCount, std::string file_name_format = "Data/paper3_low_gray/%d.png", int start_index = 0);
};

inline void ImageListReader::ReadImageList(std::vector<cv::Mat>& imageList, int imageCount, std::string file_name_format, int start_index)
{
	auto startIndex = start_index;

	if (imageCount != imageList.size())
		return;

	auto fileNameFormat = file_name_format.c_str();

	for (auto i = startIndex; i < (imageCount + startIndex); ++i)
	{
		char name[50];
		snprintf(name, sizeof(name), fileNameFormat, i);

		std::string fullName(name);
		auto curImg = cv::imread(fullName, CV_LOAD_IMAGE_GRAYSCALE);

		cv::Mat floatGrayImg;
		curImg.convertTo(floatGrayImg, CV_32FC1);

		floatGrayImg.copyTo(imageList[i - startIndex]);
	}
}
