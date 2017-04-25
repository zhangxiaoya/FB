#pragma once
#include "FrameSource/FrameSource.h"
#include "ReadPaperImageList.hpp"

class ImageListFrameSource:public FrameSource
{
public:
	explicit ImageListFrameSource(int image_count, string prefix_file_name);

	void nextFrame(OutputArray frame) override;

	void reset() override;

private:
	vector<cv::Mat> imageList;
	int imageCount;
	int currentIndex;
	string prefixFileName;
};

inline ImageListFrameSource::ImageListFrameSource(int image_count, string prefix_file_name): imageCount(image_count), prefixFileName(prefix_file_name)
{
	ImageListFrameSource::reset();
}

inline void ImageListFrameSource::nextFrame(OutputArray frame)
{
	if (currentIndex < imageCount)
	{
		imageList[currentIndex].copyTo(frame);
		++currentIndex;
	}
	else
	{
		Mat emptyMat;
		emptyMat.copyTo(frame);
	}
}

inline void ImageListFrameSource::reset()
{
	imageList.resize(imageCount);
	currentIndex = 0;
	ReadPaperImageList::ReadImageList(imageList, imageCount);
}
