#pragma once
#include "FrameSource/FrameSource.h"
#include "ImageListReader/ImageListReader.hpp"

class ImageListFrameSource:public FrameSource
{
public:
	explicit ImageListFrameSource(int image_count, string file_name_format, int start_index = 0);

	~ImageListFrameSource();

	void nextFrame(OutputArray frame) override;

	void reset() override;

private:
	vector<cv::Mat> imageList;
	int imageCount;
	int currentIndex;
	int startIndex;
	string fileNameFormat;
	ImageListReader* imageListReader;
};

inline ImageListFrameSource::ImageListFrameSource(int image_count, string file_name_format,int start_index): imageCount(image_count), startIndex(start_index), fileNameFormat(file_name_format)
{
	imageListReader = new ImageListReader();

	ImageListFrameSource::reset();
}

inline ImageListFrameSource::~ImageListFrameSource()
{
	delete imageListReader;
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

	imageListReader->SetFileNameFormat(fileNameFormat);
	imageListReader->SetStartIndex(startIndex);
	imageListReader->ReadImageList(imageList, imageCount);
}
