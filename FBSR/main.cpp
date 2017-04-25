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
	auto paperImageCount = 21;
//	auto fileNameFormat = "Data/paper3_low_gray/%d.png";
//	auto fileNameFormat = "Data/fog_low_gray/%d.png";
//	auto fileNameFormat = "Data/dataSets/Books/Book0%d.jpg";
//	auto fileNameFormat = "Data/dataSets/Building_Downs2/1%d.jpg";
//	auto fileNameFormat = "Data/dataSets/Aerial4/data%d.tif";
//	auto fileNameFormat = "Data/dataSets/LSU_MAP3/im0%d.jpg";
	auto fileNameFormat = "Data/dataSets/Office1/im%03d.jpg";
	auto startIndex = 1;
	
	auto imageListFrameSource = FrameSourceFactory::createFrameSourceFromImageList(paperImageCount, fileNameFormat, startIndex);

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
		
		char name[30];
		sprintf_s(name, "%d.png", index);
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
