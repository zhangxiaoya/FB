#pragma once
#include <core/core.hpp>
#include <vector>

using namespace std;
using namespace cv;

class ImageListReaderBase
{
public:
	static void ReadImageList(vector<cv::Mat>& imageList, int imageCount){};
};

