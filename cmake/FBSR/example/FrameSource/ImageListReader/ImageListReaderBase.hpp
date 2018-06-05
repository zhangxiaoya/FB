#pragma once
#include <opencv2/core.hpp>
#include <vector>

class ImageListReaderBase
{
public:
	explicit ImageListReaderBase(const std::string& file_name_format = "", int start_index = 0)
		: fileNameFormat(file_name_format),
		  startIndex(start_index)
	{
	}

	virtual ~ImageListReaderBase() = default;

	virtual void ReadImageList(std::vector<cv::Mat>& image_list, int image_count) = 0;

	void SetFileNameFormat(std::string file_name_format);

	void SetStartIndex(int start_index);

protected:
	std::string fileNameFormat;
	int startIndex;
};

inline void ImageListReaderBase::SetFileNameFormat(std::string file_name_format)
{
	fileNameFormat = file_name_format;
}

inline void ImageListReaderBase::SetStartIndex(int start_index)
{
	startIndex = start_index;
}

