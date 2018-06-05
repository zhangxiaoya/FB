#include "FrameSource/FrameSourceFactory.h"
#include "SuperResolution/SuperResolutionBase.h"
#include "SuperResolution/SuperResolutionFactory.h"

#include <iostream>

using namespace std;
using namespace cv;

int main()
{
	/*******************************************************************************
	 *
	 * Create a Super Resolution object and set default props
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

	/*******************************
	*
	*  This is test case for Building_Downs2,
	*  uncomment it if run want get
	*  Building_Downs2 test case result
	*
	******************************/
//	auto startIndex = 60;
//	auto totalImageCount = 38;
//	auto fileNameFormat = "Data/dataSets/Books/Book%03d.jpg";
//	auto resultNameFormat = "Result/dataSets/Books/im%03d.png";
//	superResolution->SetBufferSize(8);

	/*******************************
	*
	*  This is test case for Building_Downs2,
	*  uncomment it if run want get
	*  Building_Downs2 test case result
	*
	******************************/
	//	auto startIndex = 17;
	//	auto totalImageCount = 11;
	//	auto fileNameFormat = "Data/dataSets/Building_Downs2/1%d.jpg";
	//	auto resultNameFormat = "Result/dataSets/Building_Downs2/im%03d.png";
	//	superResolution->SetBufferSize(8);

	/*******************************
	*
	*  This is test case for Aerial4,
	*  uncomment it if run want get
	*  Aerial4 test case result
	*
	******************************/
	//	auto startIndex = 11;
	//	auto totalImageCount = 4;
	//	auto fileNameFormat = "Data/dataSets/Aerial4/data%d.tif";
	//	auto resultNameFormat = "Result/dataSets/Aerial4/im%03d.png";
	//	superResolution->SetBufferSize(4);
	//	superResolution->SetSRFactor(2);

	/*******************************
	*
	*  This is test case for LSU_MAP3,
	*  uncomment it if run want get
	*  LSU_MAP3 test case result
	*
	******************************/
	//	auto startIndex = 73;
	//	auto totalImageCount = 13;
	//	auto fileNameFormat = "Data/dataSets/LSU_MAP3/im0%d.jpg";
	//	auto resultNameFormat = "Result/dataSets/LSU_MAP3/im%03d.png";
	//	superResolution->SetBufferSize(6);
	//	superResolution->SetSRFactor(2);


	/*******************************
	*
	*  This is test case for Office1,
	*  uncomment it if run want get
	*  Office1 test case result
	*
	******************************/
	//	auto startIndex = 1;
	//	auto totalImageCount = 21;
	//	auto fileNameFormat = "Data/dataSets/Office1/im%03d.jpg";
	//	auto resultNameFormat = "Result/dataSets/Office1/im%03d.png";
	//	superResolution->SetBufferSize(6);

	/*******************************
	*
	*  This is test case for Alpaca,
	*  uncomment it if run want get
	*  Alpaca test case result
	*  
	******************************/
	auto startIndex = 1;
	auto totalImageCount = 55;
	auto fileNameFormat = "Data/Alpaca/%d.png";
	auto resultNameFormat = "Result/Alpaca/res%03d.png";
	superResolution->SetBufferSize(55);
	superResolution->SetSRFactor(4);

	/*******************************
	 *
	 *  This is test case for Emily,
	 *  uncomment it if run want get 
	 *  Emliy test case result
	 *  
	 ******************************/
	//	auto startIndex = 1;
	//	auto totalImageCount = 53;
	//	auto fileNameFormat = "Data/Emily_small/%d.png";
	//	auto resultNameFormat = "Result/Emily/res%03d.png";
	//	superResolution->SetBufferSize(53);
	//	superResolution->SetSRFactor(4);


	auto imageListFrameSource = FrameSourceFactory::createFrameSourceFromImageList(totalImageCount, fileNameFormat, startIndex);

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
		sprintf(name, resultNameFormat, index);
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
