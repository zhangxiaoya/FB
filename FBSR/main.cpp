#include "FrameSource/FrameSourceFactory.h"
#include "SuperResolution/SuperResolutionBase.h"
#include "SuperResolution/SuperResolutionFactory.h"

#include "SuperResolution/ReadEmilyImageList.hpp"
#include <iostream>

using namespace std;
using namespace cv;

int main()
{
	/*******************************************************************************
	 *
	 * Create a Super Resolution object and set basic props
	 *
	 *******************************************************************************/
	auto superResolution = SuperResolutionFactory::CreateSuperResolutionBase();

	auto alpha = 0.7;
	auto beta = 1.0;
	auto lambda = 0.04;
	auto p = 2;
	auto maxIterationCount = 20;

	auto srFactor = 2;
	auto bufferSize = 4;

	superResolution->SetProps(alpha, beta, lambda, p, maxIterationCount);
	superResolution->SetBufferSize(bufferSize);
	superResolution->SetSRFactor(srFactor);

	/*******************************************************************************
	 *
	 * set Data Source
	 *
	 *******************************************************************************/

	/***********************         From Video             ***********************/
//	const string videoFileName = "Data/fog_low_gray.avi";

//	auto videoFrameSource = FrameSourceFactory::createFrameSourceFromVideo(videoFileName);

//	superResolution->SetFrameSource(videoFrameSource);

	/***********************         From Image List         ***********************/
//	auto fileNameFormat = "Data/paper3_low_gray/%d.png";
//	auto fileNameFormat = "Data/fog_low_gray/%d.png";

//	auto startIndex = 60;
//	auto ImageCount = 38;
//	auto fileNameFormat = "Data/dataSets/Books/Book%03d.jpg";
//	auto resultNameFormat = "Result/dataSets/Books/im%03d.png";
//	superResolution->SetBufferSize(8);

//	auto startIndex = 17;
//	auto ImageCount = 11;
//	auto fileNameFormat = "Data/dataSets/Building_Downs2/1%d.jpg";
//	auto resultNameFormat = "Result/dataSets/Building_Downs2/im%03d.png";
//	superResolution->SetBufferSize(8);

//	auto startIndex = 11;
//	auto ImageCount = 4;
//	auto fileNameFormat = "Data/dataSets/Aerial4/data%d.tif";
//	auto resultNameFormat = "Result/dataSets/Aerial4/im%03d.png";

	auto startIndex = 73;
	auto ImageCount = 13;
	auto fileNameFormat = "Data/dataSets/LSU_MAP3/im0%d.jpg";
	auto resultNameFormat = "Result/dataSets/LSU_MAP3/im%03d.png";
	superResolution->SetBufferSize(6);

//	auto startIndex = 1;
//	auto ImageCount = 21;
//	auto fileNameFormat = "Data/dataSets/Office1/im%03d.jpg";
//	auto resultNameFormat = "Result/dataSets/Office1/im%03d.png";
	
	auto imageListFrameSource = FrameSourceFactory::createFrameSourceFromImageList(ImageCount, fileNameFormat, startIndex);

	superResolution->SetFrameSource(imageListFrameSource);

	/*******************************************************************************
	 *
	 * Processing Super Resolution
	 *
	 *******************************************************************************/
	auto index = 0;
	Mat currentFrame;
	while (true)
	{
		cout << index << "..";
		auto currentStatus = superResolution->NextFrame(currentFrame);

		imshow("High Resolution Frame", currentFrame);

		waitKey(100);
		
		char name[50];
		sprintf_s(name, resultNameFormat, index);
		imwrite(name, currentFrame);

		if (currentStatus == -1)
			break;

		++index;
	}
	destroyAllWindows();
	
	cout << endl;
	cout << "All Done!" << endl;
	system("pause");

	return 0;
}
